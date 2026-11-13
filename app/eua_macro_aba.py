"""
eua_macro_aba.py — abas "Juros (Treasuries)" e "Macro EUA (inflação, PIB, dólar)" do app Price Discovery Scanning
no modo EUA. Lê saidas/eua_macro/ (gerado pelo workflow macro-eua, script macro_eua.py; fonte: FRED).
Mesma lógica de tela das abas brasileiras (Juros ETTJ e macro_aba.py).
"""
import json
from pathlib import Path

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

PASTA_PADRAO = Path(__file__).resolve().parent.parent / "saidas" / "eua_macro"
ARQS = ["treasury_diario.csv.gz", "cpi_mensal.csv", "pib_trimestral.csv", "dolar_diario.csv", "metadados.json"]

CPI_MODOS = ["Trimestral", "12 meses (janela móvel mensal)", "Anual", "Mensal"]
PIB_MODOS = {"Trimestre vs. trimestre anterior, anualizado (padrão BEA)": "tt_anualizado",
             "Trimestre vs. trimestre anterior (sem anualizar, como o IBGE)": "tt",
             "Trimestre vs. mesmo trimestre do ano anterior": "aa",
             "Acumulado em 4 trimestres": "acum4",
             "Anual": "anual"}
DOLAR_SERIES = {"Dólar amplo (índice nominal do Fed, jan/2006 = 100)": ("dolar_amplo", "índice", 2),
                "R$ por US$ (Fed, H.10)": ("brl_usd", "R$ por US$", 4)}
FREQ_MODOS = ["Diário", "Fim do mês", "Média do mês", "Fim do trimestre", "Média do trimestre",
              "Fim do ano", "Média do ano"]


# ----------------------------------------------------------------------------- dados
def _versao(pasta: Path) -> tuple:
    return tuple((pasta / a).stat().st_mtime if (pasta / a).exists() else 0 for a in ARQS)


@st.cache_data(max_entries=1, show_spinner=False)
def carregar(pasta: str, versao: tuple = ()):
    p = Path(pasta)
    ler = lambda n, **k: pd.read_csv(p / n, **k) if (p / n).exists() else pd.DataFrame()  # noqa: E731
    meta = json.loads((p / "metadados.json").read_text()) if (p / "metadados.json").exists() else {}
    return (ler("treasury_diario.csv.gz", parse_dates=["data"]), ler("cpi_mensal.csv"),
            ler("pib_trimestral.csv"), ler("dolar_diario.csv", parse_dates=["data"]), meta)


def _tri(per: pd.Period) -> str:
    return f"{per.quarter}T{str(per.year)[2:]}"


def anos_vertice(col: str) -> float:
    """'nom_3m' -> 0.25; 'real_10a' -> 10."""
    v = col.split("_", 1)[1]
    return int(v[:-1]) / 12 if v.endswith("m") else float(v[:-1])


def rot_vertice(anos: float) -> str:
    return f"{round(anos * 12)} meses" if anos < 1 else f"{anos:g} ano(s)"


def curva(tr: pd.DataFrame, d, tipo: str) -> pd.Series:
    """Vértices (anos -> % a.a.) na data d. 'nominal', 'real' ou 'implicita' (= nominal − real, convenção do
    mercado americano e do FRED para o breakeven; só nos prazos com TIPS)."""
    linha = tr[tr["data"] == pd.Timestamp(d)]
    if linha.empty:
        return pd.Series(dtype=float)
    linha = linha.iloc[0]
    pega = lambda pref: pd.Series({anos_vertice(c): linha[c] for c in tr.columns  # noqa: E731
                                   if c.startswith(pref)}, dtype=float).sort_index()
    if tipo == "nominal":
        return pega("nom_").dropna()
    real = pega("real_")
    if tipo == "real":
        return real.dropna()
    return (pega("nom_").reindex(real.index) - real).dropna()


# ----------------------------------------------------------------------------- Juros (Treasuries)
def render_juros(pasta: Path = PASTA_PADRAO):
    tr, _cpi, _pib, _dol, meta = carregar(str(pasta), _versao(pasta))
    if tr.empty:
        st.info("Curvas ainda não coletadas (saidas/eua_macro/). Rode o workflow macro-eua.")
        return
    tr = tr.sort_values("data").reset_index(drop=True)
    datas = list(tr["data"])
    st.caption(f"Fonte: U.S. Treasury (yields de vencimento constante, nominais e TIPS), via FRED/St. Louis Fed; "
               f"{len(datas)} pregões de {datas[0].date()} a {datas[-1].date()}. Taxas % a.a., base semestral "
               "(bond-equivalent), não comparáveis em convenção à base 252 da ANBIMA.")
    tipos = {"Nominal (Treasuries)": "nominal", "Real (TIPS)": "real", "Inflação implícita (breakeven)": "implicita"}
    j1, j2, j3 = st.columns([1.3, 1.2, 2])
    tc_rot = j1.radio("Curva", list(tipos), key="eua_jur_curva")
    tc = tipos[tc_rot]
    base = j2.selectbox("Data", datas[::-1], format_func=lambda d: d.strftime("%d/%m/%Y"), key="eua_jur_data")
    prazos_cmp = {"1 semana antes": 7, "1 mês antes": 30, "3 meses antes": 91, "6 meses antes": 182,
                  "1 ano antes": 365}
    sel_cmp = j3.multiselect("Comparar com", list(prazos_cmp), default=["1 mês antes", "1 ano antes"],
                             key="eua_jur_cmp")
    extra = j3.multiselect("Outras datas", [d for d in datas[::-1] if d != base],
                           format_func=lambda d: d.strftime("%d/%m/%Y"), key="eua_jur_extra")
    datas_plot, faltam = [(pd.Timestamp(base), "Data escolhida")], []
    for nome in sel_cmp:
        alvo = pd.Timestamp(base) - pd.Timedelta(days=prazos_cmp[nome])
        cand = [d for d in datas if d <= alvo]
        (datas_plot.append((cand[-1], nome)) if cand else faltam.append(nome))
    datas_plot += [(pd.Timestamp(d), "") for d in extra]
    nome_d = lambda d, rot: f"{d.strftime('%d/%m/%Y')} {rot}".strip()  # noqa: E731

    fig = go.Figure()
    for d, rot in datas_plot:
        y = curva(tr, d, tc)
        fig.add_trace(go.Scatter(x=list(y.index), y=y.values / 100, mode="lines+markers", name=nome_d(d, rot),
                                 line=dict(width=4 if rot == "Data escolhida" else 2)))
    fig.update_layout(title=f"Curva {tc_rot.lower()} (% a.a.)", height=460, yaxis_tickformat=".2%",
                      xaxis_title="Prazo (anos)", legend=dict(orientation="h", y=-0.2),
                      margin=dict(t=40, b=20, l=10, r=10))
    st.plotly_chart(fig, width="stretch")
    if faltam:
        st.caption("Sem histórico suficiente para: " + ", ".join(faltam) + ".")

    # tabela de vértices e variação em pontos-base
    tab = pd.DataFrame({nome_d(d, rot): curva(tr, d, tc) for d, rot in datas_plot})
    tab.index = [rot_vertice(a) for a in tab.index]
    base_col = tab.columns[0]
    for c in tab.columns[1:]:
        tab[f"Δ vs {c} (bps)"] = (tab[base_col] - tab[c]) * 100
    st.dataframe(tab.round(2), width="stretch")

    # histórico de vértices
    st.subheader("Histórico de vértices")
    pref = {"nominal": "nom_", "real": "real_", "implicita": "real_"}[tc]
    vert = [anos_vertice(c) for c in tr.columns if c.startswith(pref)]
    padrao = [v for v in (2, 5, 10, 30) if v in vert] or vert[:3]
    h1, h2 = st.columns([3, 1])
    v_hist = h1.multiselect("Vértices", vert, default=padrao, format_func=rot_vertice, key=f"eua_jur_vert_{tc}")
    extras = h2.multiselect("Acrescentar", ["Fed funds (efetiva)", "Inclinação 10a − 2a", "Inclinação 10a − 3m"],
                            default=[], key="eua_jur_extras", disabled=(tc != "nominal"),
                            help="Só na curva nominal.")
    hist = pd.DataFrame({"data": tr["data"]})
    for v in v_hist:
        c = [k for k in tr.columns if k.startswith(pref) and anos_vertice(k) == v][0]
        hist[rot_vertice(v)] = tr[c.replace("real_", "nom_")] - tr[c] if tc == "implicita" else tr[c]
    if tc == "nominal":
        if "Fed funds (efetiva)" in extras:
            hist["Fed funds (efetiva)"] = tr["fed_funds"]
        if "Inclinação 10a − 2a" in extras:
            hist["Inclinação 10a − 2a"] = tr["nom_10a"] - tr["nom_2a"]
        if "Inclinação 10a − 3m" in extras:
            hist["Inclinação 10a − 3m"] = tr["nom_10a"] - tr["nom_3m"]
    series = [c for c in hist.columns if c != "data"]
    if series:
        fig = go.Figure()
        for c in series:
            fig.add_trace(go.Scatter(x=hist["data"], y=hist[c] / 100, mode="lines", name=c))
        fig.update_layout(title=f"{tc_rot}: vértices ao longo do tempo (% a.a.)", height=420,
                          yaxis_tickformat=".2%", legend=dict(orientation="h", y=-0.2),
                          margin=dict(t=40, b=20, l=10, r=10))
        st.plotly_chart(fig, width="stretch")
        exp_h = hist.assign(data=hist["data"].dt.date)
        st.download_button("⬇️ Histórico (CSV)", exp_h.to_csv(index=False, sep=";", decimal=",").encode("utf-8-sig"),
                           file_name=f"treasuries_{tc}_historico.csv", mime="text/csv", key="eua_jur_csv")
    st.caption("Vértices publicados pelo Tesouro americano (sem modelo paramétrico: a linha só liga os vértices). "
               "Inflação implícita = nominal − real (TIPS), como no breakeven do FRED (ex.: T10YIE), só nos prazos "
               "com TIPS (5, 7, 10, 20 e 30 anos). Inclinações em pontos percentuais.")


# ----------------------------------------------------------------------------- Macro EUA
def calc_cpi(c: pd.DataFrame, col: str, modo: str) -> pd.DataFrame:
    """Variações a partir do número-índice (equivale a compor as variações mensais). Períodos incompletos
    ficam marcados como parciais; períodos que incluem mês interpolado também são marcados."""
    c = c.assign(per=pd.PeriodIndex(c["mes"], freq="M")).sort_values("per").reset_index(drop=True)
    idx = pd.Series(c[col].to_numpy(), index=c["per"])
    interp = pd.Series(c["interpolado"].astype(bool).to_numpy(), index=c["per"])
    fim_mes = lambda p: p.to_timestamp(how="end").normalize()  # noqa: E731
    if modo == "Mensal" or modo.startswith("12 meses"):
        n = 1 if modo == "Mensal" else 12
        v = (idx / idx.shift(n) - 1) * 100
        it = interp.rolling(n + 1, min_periods=1).max().astype(bool)
        out = pd.DataFrame({"periodo": [str(p) + (" *" if i else "") for p, i in zip(idx.index, it)],
                            "data": [fim_mes(p) for p in idx.index], "cpi_pct": v.round(2).to_numpy(),
                            "parcial": False})
        return out.dropna(subset=["cpi_pct"]).reset_index(drop=True)
    freq, n_meses = ("Q", 3) if modo == "Trimestral" else ("Y", 12)
    grupo = idx.index.asfreq(freq)
    g = pd.DataFrame({"i": idx.to_numpy(), "it": interp.to_numpy(), "g": grupo})
    ult = g.groupby("g")["i"].last()
    tam = g.groupby("g")["i"].size()
    it_g = g.groupby("g")["it"].max()
    # base = último índice do período anterior (o primeiro período da série fica sem base e sai)
    base_ = ult.shift(1)
    v = (ult / base_ - 1) * 100
    out = pd.DataFrame({"g": ult.index, "cpi_pct": v.round(2).to_numpy(), "parcial": (tam < n_meses).to_numpy(),
                        "interp": it_g.to_numpy()})
    out["periodo"] = [(_tri(p) if freq == "Q" else str(p.year)) + (" (parcial)" if parc else "") + (" *" if i else "")
                      for p, parc, i in zip(out["g"], out["parcial"], out["interp"])]
    out["data"] = [p.to_timestamp(how="end").normalize() for p in out["g"]]
    return out.dropna(subset=["cpi_pct"])[["periodo", "data", "cpi_pct", "parcial"]].reset_index(drop=True)


def calc_pib(p: pd.DataFrame, col: str, modo: str) -> pd.DataFrame:
    """Taxas de crescimento do nível (real ou nominal, US$ bi SAAR). A t/t anualizada do PIB real usa a taxa oficial
    do BEA (A191RL1Q225SBEA) quando disponível."""
    p = p.sort_values(["ano", "tri"]).reset_index(drop=True)
    x = p[col]
    tipo = PIB_MODOS[modo]
    if tipo == "tt_anualizado":
        v = ((x / x.shift(1)) ** 4 - 1) * 100
        if col == "pib_real" and "pib_real_saar_bea" in p:
            v = p["pib_real_saar_bea"].where(p["pib_real_saar_bea"].notna(), v)
    elif tipo == "tt":
        v = (x / x.shift(1) - 1) * 100
    elif tipo == "aa":
        v = (x / x.shift(4) - 1) * 100
    else:  # acumulado em 4 trimestres (e anual = esse valor no 4º trimestre)
        s4 = x.rolling(4).sum()
        v = (s4 / s4.shift(4) - 1) * 100
    d = pd.DataFrame({"periodo": p["trimestre"], "ano": p["ano"], "tri": p["tri"],
                      "data": [pd.Period(year=a, quarter=t, freq="Q").to_timestamp(how="end").normalize()
                               for a, t in zip(p["ano"], p["tri"])], "pib_pct": v.round(2)})
    if tipo == "anual":
        d = d[d["tri"] == 4].assign(periodo=lambda z: z["ano"].astype(str))
    return d.drop(columns=["ano", "tri"]).dropna().reset_index(drop=True)


def calc_freq(df: pd.DataFrame, col: str, modo: str, casas: int) -> pd.DataFrame:
    c = df[["data", col]].dropna().sort_values("data")
    if modo == "Diário":
        return pd.DataFrame({"periodo": c["data"].dt.strftime("%d/%m/%Y"), "data": c["data"], col: c[col]})
    freq = {"mês": "M", "trimestre": "Q", "ano": "Y"}[modo.split(" do ")[-1]]
    g = c.assign(grupo=c["data"].dt.to_period(freq)).groupby("grupo")
    v = g[col].last() if modo.startswith("Fim") else g[col].mean()
    out = v.reset_index()
    out["periodo"] = [str(p) if freq == "M" else (_tri(p) if freq == "Q" else str(p.year)) for p in out["grupo"]]
    out["data"] = g["data"].max().values
    out[col] = out[col].round(casas)
    return out[["periodo", "data", col]]


def _grafico(df, y, titulo, tipo="barra", rot_y="%"):
    if tipo == "barra":
        fig = px.bar(df, x="periodo", y=y, title=titulo, labels={"periodo": "", y: rot_y},
                     color=np.where(df[y] >= 0, "pos", "neg"),
                     color_discrete_map={"pos": "#2E6FBA", "neg": "#C0504D"})
        fig.update_layout(showlegend=False, xaxis_type="category")
        fig.update_xaxes(categoryorder="array", categoryarray=list(df["periodo"]))
    else:
        fig = px.line(df, x="data", y=y, title=titulo, labels={"data": "", y: rot_y})
    fig.update_layout(height=380, margin=dict(l=10, r=10, t=50, b=10))
    return fig


def _br(v: float, casas: int = 2) -> str:
    return f"{v:,.{casas}f}".replace(",", "X").replace(".", ",").replace("X", ".")


def render_macro(excel_bytes, pasta: Path = PASTA_PADRAO):
    _tr, cpi, pib, dol, meta = carregar(str(pasta), _versao(pasta))
    if cpi.empty and pib.empty and dol.empty:
        st.info("Séries macro dos EUA ainda não coletadas (saidas/eua_macro/). Rode o workflow macro-eua.")
        return
    ini_exib = pd.Timestamp(meta.get("janela", {}).get("inicio_exibicao")
                            or f"{pd.Timestamp.today().year - 11}-01-01")
    anos = list(range(ini_exib.year, pd.Timestamp.today().year + 1))
    a0, a1 = st.select_slider("Período exibido (anos)", options=anos, value=(anos[0], anos[-1]), key="eua_mac_anos")
    janela = lambda df: df[(df["data"] >= pd.Timestamp(a0, 1, 1)) &  # noqa: E731
                           (df["data"] <= pd.Timestamp(a1, 12, 31))].reset_index(drop=True)
    exportar = {}

    # ---- Inflação
    st.subheader("Inflação — CPI (BLS)")
    if cpi.empty:
        st.warning("CPI indisponível nesta coleta.")
    else:
        i1, i2 = st.columns([1, 3])
        medida = i1.radio("Índice", ["CPI cheio", "Núcleo (ex-alimentos e energia)"], key="eua_mac_cpi_ind")
        col = "cpi_indice" if medida == "CPI cheio" else "core_indice"
        modo = i2.radio("Periodicidade", CPI_MODOS, horizontal=True, key="eua_mac_cpi_modo")
        t = janela(calc_cpi(cpi, col, modo))
        c1, c2 = st.columns([3, 1])
        c1.plotly_chart(_grafico(t, "cpi_pct", f"{medida} — {modo.lower()} (%)",
                                 "linha" if modo.startswith("12") else "barra"), width="stretch")
        per = pd.PeriodIndex(cpi["mes"], freq="M")
        idx = pd.Series(cpi[col].to_numpy(), index=per)
        dentro = idx[(per >= pd.Period(f"{a0}-01", freq="M")) & (per <= pd.Period(f"{a1}-12", freq="M"))]
        antes = idx[idx.index < pd.Period(f"{a0}-01", freq="M")]
        if len(dentro):
            b = antes.iloc[-1] if len(antes) else dentro.iloc[0]
            c2.metric("CPI acumulado no período", f"{_br((dentro.iloc[-1] / b - 1) * 100)}%")
        mm = (idx.iloc[-1] / idx.iloc[-2] - 1) * 100
        c2.metric("Último mês disponível", str(per[-1]), f"{mm:.2f}% no mês", delta_color="off")
        c2.dataframe(t[["periodo", "cpi_pct"]].iloc[::-1], hide_index=True, height=220,
                     column_config={"periodo": "Período", "cpi_pct": st.column_config.NumberColumn("CPI %", format="%.2f")})
        exportar[f"{'CPI' if col == 'cpi_indice' else 'Core'} {modo[:10]}"] = t.assign(data=t["data"].dt.date)
        meses_int = list(cpi.loc[cpi["interpolado"].astype(bool), "mes"])
        st.caption("Fonte: BLS, CPI-U (todos os itens e núcleo), sem ajuste sazonal, via FRED (CPIAUCNS, CPILFENS). "
                   "Variações calculadas aqui pelo número-índice; períodos incompletos aparecem como parciais."
                   + (f" * Inclui mês sem divulgação do BLS ({', '.join(meses_int)}, shutdown do governo): índice "
                      "interpolado geometricamente entre os meses vizinhos." if meses_int else ""))

    # ---- PIB real e nominal
    for col, titulo, sub in (("pib_real", "Crescimento do PIB real", "real"),
                             ("pib_nominal", "PIB crescimento nominal", "nominal")):
        st.subheader(titulo)
        if pib.empty:
            st.warning("PIB indisponível nesta coleta.")
            continue
        modo = st.radio("Medida", list(PIB_MODOS), horizontal=True, key=f"eua_mac_pib_{sub}")
        t = janela(calc_pib(pib, col, modo))
        c1, c2 = st.columns([3, 1])
        c1.plotly_chart(_grafico(t, "pib_pct", f"PIB {sub} — {modo.lower()} (%)"), width="stretch")
        c2.metric("Último trimestre divulgado", pib["trimestre"].iloc[-1],
                  f"US$ {_br(pib[col].iloc[-1] / 1000, 2)} tri (anualizado)", delta_color="off")
        c2.dataframe(t[["periodo", "pib_pct"]].iloc[::-1], hide_index=True, height=260,
                     column_config={"periodo": "Período", "pib_pct": st.column_config.NumberColumn("PIB %", format="%.1f")})
        exportar[f"PIB {sub} {PIB_MODOS[modo]}"] = t.assign(data=t["data"].dt.date)
    if not pib.empty:
        st.caption("Fonte: BEA, Contas Nacionais (NIPA), via FRED: PIB real (GDPC1, US$ encadeados de 2017) e PIB "
                   "nominal (GDP), ambos com ajuste sazonal e anualizados. O padrão do BEA é a taxa trimestral "
                   "anualizada (t/t elevada à 4ª potência); 'sem anualizar' é a medida comparável à do IBGE. "
                   "No PIB nominal a variação vem do próprio PIB nominal do BEA (não de composição com o CPI). "
                   "'Anual' = acumulado em 4 trimestres no 4º trimestre. O BEA revisa trimestres anteriores; a série "
                   "é regravada inteira a cada coleta.")

    # ---- Dólar
    st.subheader("Dólar")
    if dol.empty:
        st.warning("Câmbio indisponível nesta coleta.")
    else:
        d1, d2 = st.columns([1.3, 3])
        s_rot = d1.radio("Série", list(DOLAR_SERIES), key="eua_mac_dol_serie")
        col, rot_y, casas = DOLAR_SERIES[s_rot]
        modo = d2.radio("Frequência", FREQ_MODOS, horizontal=True, index=1, key="eua_mac_dol_modo")
        t = janela(calc_freq(dol, col, modo, casas))
        c1, c2 = st.columns([3, 1])
        c1.plotly_chart(_grafico(t, col, f"{s_rot.split(' (')[0]} — {modo.lower()}", "linha", rot_y), width="stretch")
        if len(t) > 1:
            var = (t[col].iloc[-1] / t[col].iloc[0] - 1) * 100
            c2.metric(f"Variação {t['periodo'].iloc[0]} → {t['periodo'].iloc[-1]}", f"{var:+.1f}%",
                      help="Dólar amplo subindo = dólar se valorizando contra a cesta de moedas; "
                           "R$/US$ subindo = real se desvalorizando.")
        ult = dol.dropna(subset=[col]).sort_values("data").iloc[-1]
        c2.metric("Último dado", _br(ult[col], casas), ult["data"].strftime("%d/%m/%Y"), delta_color="off")
        c2.dataframe(t[["periodo", col]].iloc[::-1], hide_index=True, height=220,
                     column_config={"periodo": "Período", col: st.column_config.NumberColumn(rot_y, format=f"%.{casas}f")})
        exportar[f"Dolar {col} {modo}"[:31]] = t.assign(data=t["data"].dt.date)
        st.caption("Fonte: Federal Reserve (H.10), via FRED: índice amplo nominal do dólar contra as moedas dos "
                   "principais parceiros comerciais (DTWEXBGS) e taxa de compra ao meio-dia em Nova York, R$ por US$ "
                   "(DEXBZUS; difere da PTAX do BCB). 'Fim' = último pregão do período; 'Média' = média simples. "
                   "O H.10 sai com cerca de uma semana de defasagem.")

    if exportar:
        st.download_button("⬇️ Séries exibidas (XLSX)", excel_bytes({k[:31]: v for k, v in exportar.items()}),
                           file_name=f"macro_eua_{a0}_{a1}.xlsx", key="eua_mac_xlsx",
                           mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet")
