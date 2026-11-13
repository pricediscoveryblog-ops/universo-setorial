"""
macro_aba.py — aba "Macro (inflação, PIB, câmbio)" do app Price Discovery Scanning.
Lê saidas/macro/ (gerado pelo workflow macro, script macro.py) e mostra três blocos: IPCA, PIB e PTAX.
"""
import json
from pathlib import Path

import numpy as np
import pandas as pd
import plotly.express as px
import streamlit as st

ARQS = ["ipca_mensal.csv", "pib_trimestral.csv", "pib_volume_trimestral.csv", "cambio_diario.csv", "metadados.json"]
PASTA_PADRAO = Path(__file__).resolve().parent.parent / "saidas" / "macro"

IPCA_MODOS = ["Trimestral", "12 meses (janela móvel mensal)", "Anual", "Mensal"]
PIB_MODOS = {"Trimestre vs. trimestre anterior (dessazonalizado)": "pib_tri_vs_tri_anterior_dessaz",
             "Trimestre vs. mesmo trimestre do ano anterior": "pib_tri_vs_mesmo_tri_ano_anterior",
             "Acumulado em 4 trimestres": "pib_acum_4_tri",
             "Anual": "anual"}
CAMBIO_MODOS = ["Diário", "Fim do mês", "Média do mês", "Fim do trimestre", "Média do trimestre",
                "Fim do ano", "Média do ano"]


# ----------------------------------------------------------------------------- dados
def _versao(pasta: Path) -> tuple:
    return tuple((pasta / a).stat().st_mtime if (pasta / a).exists() else 0 for a in ARQS)


@st.cache_data(max_entries=1, show_spinner=False)
def carregar(pasta: str, versao: tuple = ()):
    p = Path(pasta)
    ler = lambda n, **k: pd.read_csv(p / n, **k) if (p / n).exists() else pd.DataFrame()  # noqa: E731
    meta = json.loads((p / "metadados.json").read_text()) if (p / "metadados.json").exists() else {}
    return (ler("ipca_mensal.csv"), ler("pib_trimestral.csv"),
            ler("cambio_diario.csv", parse_dates=["data"]), meta, ler("pib_volume_trimestral.csv"))


def _tri(per: pd.Period) -> str:
    return f"{per.quarter}T{str(per.year)[2:]}"


def calc_ipca(m: pd.DataFrame, modo: str) -> pd.DataFrame:
    """Variações compostas a partir da variação mensal. Períodos incompletos ficam marcados como parciais."""
    m = m.assign(per=pd.PeriodIndex(m["mes"], freq="M")).sort_values("per")
    fator = 1 + m["ipca_mensal"] / 100
    if modo == "Mensal":
        return pd.DataFrame({"periodo": m["per"].astype(str), "data": m["per"].dt.to_timestamp(how="end").dt.normalize(),
                             "ipca_pct": m["ipca_mensal"].round(2), "parcial": False})
    if modo.startswith("12 meses"):
        v = fator.rolling(12).apply(np.prod, raw=True) - 1
        out = pd.DataFrame({"periodo": m["per"].astype(str), "data": m["per"].dt.to_timestamp(how="end").dt.normalize(),
                            "ipca_pct": (v * 100).round(2), "parcial": False})
        return out.dropna(subset=["ipca_pct"])
    freq = "Q" if modo == "Trimestral" else "Y"
    g = m.assign(f=fator, grupo=m["per"].dt.asfreq(freq)).groupby("grupo")
    out = pd.DataFrame({"var": g["f"].prod() - 1, "n": g["f"].size()}).reset_index()
    out["parcial"] = out["n"] < (3 if freq == "Q" else 12)
    out["periodo"] = [(_tri(p) if freq == "Q" else str(p.year)) + (" (parcial)" if parc else "")
                      for p, parc in zip(out["grupo"], out["parcial"])]
    out["data"] = out["grupo"].dt.to_timestamp(how="end").dt.normalize()
    out["ipca_pct"] = (out["var"] * 100).round(2)
    return out[["periodo", "data", "ipca_pct", "parcial"]]


def calc_pib(p: pd.DataFrame, modo: str) -> pd.DataFrame:
    col = PIB_MODOS[modo]
    d = p.assign(data=[pd.Period(year=a, quarter=t, freq="Q").to_timestamp(how="end").normalize()
                       for a, t in zip(p["ano"], p["tri"])])
    if col == "anual":
        d = d[d["tri"] == 4]
        return pd.DataFrame({"periodo": d["ano"].astype(str), "data": d["data"],
                             "pib_pct": d["pib_acum_4_tri"]}).dropna()
    return pd.DataFrame({"periodo": d["trimestre"], "data": d["data"], "pib_pct": d[col]}).dropna()


def calc_cambio(c: pd.DataFrame, modo: str) -> pd.DataFrame:
    c = c.sort_values("data")
    if modo == "Diário":
        return pd.DataFrame({"periodo": c["data"].dt.strftime("%d/%m/%Y"), "data": c["data"],
                             "ptax_venda": c["ptax_venda"]})
    freq = {"mês": "M", "trimestre": "Q", "ano": "Y"}[modo.split(" do ")[-1]]
    g = c.assign(grupo=c["data"].dt.to_period(freq)).groupby("grupo")
    v = g["ptax_venda"].last() if modo.startswith("Fim") else g["ptax_venda"].mean()
    out = v.reset_index()
    out["periodo"] = [str(p) if freq == "M" else (_tri(p) if freq == "Q" else str(p.year)) for p in out["grupo"]]
    # data de referência: último pregão do período (no período corrente, o último dado disponível)
    out["data"] = g["data"].max().values
    out["ptax_venda"] = out["ptax_venda"].round(4)
    return out[["periodo", "data", "ptax_venda"]]


# ----------------------------------------------------------------------------- PIB nominal e comparação
def ipca_indice(ipca: pd.DataFrame) -> pd.Series:
    """Número-índice mensal do IPCA (base 100 no mês anterior ao primeiro da série), indexado pelo mês."""
    per = pd.PeriodIndex(ipca["mes"], freq="M")
    return pd.Series((1 + ipca["ipca_mensal"].to_numpy() / 100).cumprod() * 100, index=per)


def media_ipca(idx: pd.Series, fim, meses: int) -> float:
    """Nível médio de preços nos N (= meses) meses terminados no mês da data de fim (NaN se faltar algum mês)."""
    f = pd.Period(pd.Timestamp(fim), freq="M")
    janela = pd.period_range(f - meses + 1, f, freq="M")
    return float(idx.reindex(janela).mean()) if idx.reindex(janela).notna().all() else np.nan


def ipca_trimestral_medio(ipca: pd.DataFrame) -> pd.DataFrame:
    """Nível médio de preços por trimestre e suas variações (base de deflação dos fluxos do PIB)."""
    idx = ipca_indice(ipca)
    q = idx.groupby(idx.index.asfreq("Q")).agg(["mean", "size"])
    q = q[q["size"] == 3]["mean"]
    out = pd.DataFrame({"p": q})
    out["ano"], out["tri"] = out.index.year, out.index.quarter
    out["ipca_tri_ant"] = out["p"] / out["p"].shift(1) - 1
    out["ipca_mesmo_tri"] = out["p"] / out["p"].shift(4) - 1
    s4 = out["p"].rolling(4).sum()
    out["ipca_4_tri"] = s4 / s4.shift(4) - 1
    return out.reset_index(drop=True)


NOMINAL = {"pib_tri_vs_tri_anterior_dessaz": "ipca_tri_ant",
           "pib_tri_vs_mesmo_tri_ano_anterior": "ipca_mesmo_tri",
           "pib_acum_4_tri": "ipca_4_tri"}


def calc_pib_nominal(pib: pd.DataFrame, ipca: pd.DataFrame, modo: str) -> pd.DataFrame:
    """PIB nominal = (1 + PIB real) × (1 + IPCA no mesmo recorte) − 1, com o IPCA medido pelo nível médio de
    preços de cada período (o PIB é fluxo). 'Anual' = acumulado em 4 trimestres no 4º trimestre."""
    col = PIB_MODOS[modo]
    real_col = "pib_acum_4_tri" if col == "anual" else col
    j = pib.merge(ipca_trimestral_medio(ipca), on=["ano", "tri"], how="left")
    infl = j[NOMINAL[real_col]] * 100
    nom = ((1 + j[real_col] / 100) * (1 + infl / 100) - 1) * 100
    out = pd.DataFrame({"periodo": j["trimestre"], "ano": j["ano"], "tri": j["tri"],
                        "data": [pd.Period(year=a, quarter=t, freq="Q").to_timestamp(how="end").normalize()
                                 for a, t in zip(j["ano"], j["tri"])],
                        "pib_real_pct": j[real_col], "ipca_pct": infl.round(2), "pib_nominal_pct": nom.round(2)})
    if col == "anual":
        out = out[out["tri"] == 4].assign(periodo=lambda x: x["ano"].astype(str))
    return out.drop(columns=["ano", "tri"]).dropna(subset=["pib_nominal_pct"]).reset_index(drop=True)


def niveis_macro(fins: pd.Series, meses: int, estoque: bool = False, pasta: Path | None = None) -> pd.DataFrame:
    """Nível de cada série macro por período das empresas (índice = rótulo do período, valor = data de fim).
    Fluxos (receita, EBITDA, lucro...): média do período (IPCA, R$/US$) e soma dos trimestres (PIB nominal =
    índice de volume × nível médio de preços). Estoques (ativo, PL): posição no fim do período."""
    p = Path(pasta or PASTA_PADRAO)
    ipca, _pib, cambio, _meta, vol = carregar(str(p), _versao(p))
    fins = pd.to_datetime(fins)
    out = pd.DataFrame(index=fins.index, columns=["PIB nominal", "IPCA", "R$/US$"], dtype=float)
    if not ipca.empty:
        idx = ipca_indice(ipca)
        for k, f in fins.items():
            fm = pd.Period(f, freq="M")
            out.loc[k, "IPCA"] = (idx.get(fm, np.nan) if estoque else media_ipca(idx, f, meses))
        if not vol.empty:
            pq = ipca_trimestral_medio(ipca)[["ano", "tri", "p"]]
            v = vol.merge(pq, on=["ano", "tri"], how="inner")
            v.index = pd.PeriodIndex([pd.Period(year=a, quarter=t, freq="Q") for a, t in zip(v["ano"], v["tri"])])
            nominal = v["indice_volume"] * v["p"]
            nq = max(1, meses // 3)
            for k, f in fins.items():
                fq = pd.Period(f, freq="Q")
                jan = nominal.reindex(pd.period_range(fq - nq + 1, fq, freq="Q"))
                out.loc[k, "PIB nominal"] = jan.sum() if jan.notna().all() else np.nan
    if not cambio.empty:
        c = cambio.set_index("data")["ptax_venda"].sort_index()
        for k, f in fins.items():
            if estoque:
                ate = c[:f]
                out.loc[k, "R$/US$"] = ate.iloc[-1] if len(ate) and (f - ate.index[-1]).days <= 10 else np.nan
            else:
                ini = (pd.Period(f, freq="M") - meses + 1).to_timestamp()
                jan = c[ini:f]
                out.loc[k, "R$/US$"] = jan.mean() if len(jan) and (f - jan.index[-1]).days <= 10 else np.nan
    return out


CORES_MACRO = {"PIB nominal": "#2CA02C", "IPCA": "#D62728", "R$/US$": "#9467BD"}
OPCOES_MACRO = list(CORES_MACRO)


def _seletor(key: str) -> list:
    return st.multiselect("Comparar com (macro Brasil)", OPCOES_MACRO, default=[], key=key,
                          help="Linhas pontilhadas: PIB nominal, IPCA e R$/US$ no mesmo recorte de tempo das empresas.")


def comparar_base100(fig, fins: pd.Series, meses: int, estoque: bool, key: str) -> list:
    """Acrescenta ao gráfico de crescimento acumulado (base 100 no primeiro período) as séries macro escolhidas."""
    sel = _seletor(key)
    if not sel or fins.empty:
        return sel
    niv = niveis_macro(fins, meses, estoque)
    for nome in sel:
        s = niv[nome]
        base = s.first_valid_index()
        if base is None:
            continue
        y = 100 * s / s[base]
        fig.add_trace(go_scatter(list(y.index), y.values, nome))
    st.caption("Macro no mesmo recorte das empresas, base 100 no primeiro período com dado. "
               + ("Estoque: posição no fim do período (IPCA = índice do mês; R$/US$ = PTAX do último pregão)."
                  if estoque else
                  "Fluxo: IPCA e R$/US$ pela média do período; PIB nominal = volume (IBGE, SIDRA 1620) × nível "
                  "médio de preços (IPCA), somado nos trimestres do período.")
               + " R$/US$ subindo = real se desvalorizando.")
    return sel


def comparar_aa(fig, g: pd.DataFrame, meses: int, key: str) -> list:
    """Acrescenta ao gráfico de crescimento contra o ano anterior (em fração) a variação das séries macro escolhidas
    no mesmo recorte (período de N meses terminado em cada data de fim contra o mesmo período um ano antes)."""
    sel = _seletor(key)
    if not sel or g.empty:
        return sel
    fins = pd.Series(pd.to_datetime(g["dt_fim"]).to_numpy(), index=g["periodo"].to_numpy())
    atual = niveis_macro(fins, meses)
    ant = niveis_macro(fins - pd.DateOffset(years=1), meses)
    for nome in sel:
        y = atual[nome] / ant[nome] - 1
        fig.add_trace(go_scatter(list(y.index), y.values, nome))
    st.caption("Macro: variação do período contra o mesmo período do ano anterior. IPCA e R$/US$ pela média do "
               "período; PIB nominal = volume (IBGE) × nível médio de preços (IPCA). R$/US$ positivo = real se "
               "desvalorizou.")
    return sel


def go_scatter(x, y, nome):
    import plotly.graph_objects as go
    return go.Scatter(x=x, y=y, name=nome, mode="lines+markers",
                      line=dict(color=CORES_MACRO[nome], width=2.5, dash="dot"), marker=dict(size=6))


# ----------------------------------------------------------------------------- tela
def _grafico(df, y, titulo, tipo="barra", rot_y="%"):
    if tipo == "barra":
        fig = px.bar(df, x="periodo", y=y, title=titulo, labels={"periodo": "", y: rot_y},
                     color=np.where(df[y] >= 0, "pos", "neg"),
                     color_discrete_map={"pos": "#2E6FBA", "neg": "#C0504D"})
        fig.update_layout(showlegend=False, xaxis_type="category")
        fig.update_xaxes(categoryorder="array", categoryarray=list(df["periodo"]))   # ordem cronológica
    else:
        fig = px.line(df, x="data", y=y, title=titulo, labels={"data": "", y: rot_y})
    fig.update_layout(height=380, margin=dict(l=10, r=10, t=50, b=10))
    return fig


def render(pasta: Path, excel_bytes):
    ipca, pib, cambio, meta, _vol = carregar(str(pasta), _versao(pasta))
    if ipca.empty and pib.empty and cambio.empty:
        st.info("Séries macro ainda não coletadas (saidas/macro/). Rode o workflow macro.")
        return

    ini_exib = pd.Timestamp(meta.get("janela", {}).get("inicio_exibicao")
                            or f"{pd.Timestamp.today().year - 11}-01-01")
    anos = list(range(ini_exib.year, pd.Timestamp.today().year + 1))
    a0, a1 = st.select_slider("Período exibido (anos)", options=anos, value=(anos[0], anos[-1]), key="mac_anos")
    janela = lambda df: df[(df["data"] >= pd.Timestamp(a0, 1, 1)) &  # noqa: E731
                           (df["data"] <= pd.Timestamp(a1, 12, 31))].reset_index(drop=True)
    exportar = {}

    # ---- Inflação
    st.subheader("Inflação — IPCA")
    if ipca.empty:
        st.warning("IPCA indisponível nesta coleta.")
    else:
        modo = st.radio("Periodicidade", IPCA_MODOS, horizontal=True, key="mac_ipca_modo")
        t = janela(calc_ipca(ipca, modo))
        c1, c2 = st.columns([3, 1])
        c1.plotly_chart(_grafico(t, "ipca_pct", f"IPCA — {modo.lower()} (%)",
                                 "linha" if modo.startswith("12") else "barra"), width="stretch")
        mm = ipca.assign(d=pd.PeriodIndex(ipca["mes"], freq="M").to_timestamp(how="end"))
        mm = mm[(mm["d"] >= pd.Timestamp(a0, 1, 1)) & (mm["d"] <= pd.Timestamp(a1, 12, 31, 23, 59))]
        acum = (np.prod(1 + mm["ipca_mensal"] / 100) - 1) * 100
        c2.metric("IPCA acumulado no período", f"{acum:,.2f}%".replace(",", "X").replace(".", ",").replace("X", "."))
        c2.metric("Último mês disponível", ipca["mes"].iloc[-1], f"{ipca['ipca_mensal'].iloc[-1]:.2f}% no mês",
                  delta_color="off")
        c2.dataframe(t[["periodo", "ipca_pct"]].iloc[::-1], hide_index=True, height=220,
                     column_config={"periodo": "Período", "ipca_pct": st.column_config.NumberColumn("IPCA %", format="%.2f")})
        exportar[f"IPCA {modo[:10]}"] = t.assign(data=t["data"].dt.date)
        st.caption("Fonte: IBGE, via Banco Central (SGS, série 433 — variação mensal). Trimestral, 12 meses e anual "
                   "são calculados aqui por composição das variações mensais; períodos incompletos aparecem como parciais.")

    # ---- PIB
    st.subheader("Crescimento do PIB")
    if pib.empty:
        st.warning("PIB indisponível nesta coleta.")
    else:
        modo = st.radio("Medida", list(PIB_MODOS), horizontal=True, key="mac_pib_modo")
        t = janela(calc_pib(pib, modo))
        c1, c2 = st.columns([3, 1])
        c1.plotly_chart(_grafico(t, "pib_pct", f"PIB — {modo.lower()} (%)"), width="stretch")
        c2.metric("Último trimestre divulgado", pib["trimestre"].iloc[-1])
        c2.dataframe(t[["periodo", "pib_pct"]].iloc[::-1], hide_index=True, height=260,
                     column_config={"periodo": "Período", "pib_pct": st.column_config.NumberColumn("PIB %", format="%.1f")})
        exportar[f"PIB {modo[:20]}"] = t.assign(data=t["data"].dt.date)
        st.caption("Fonte: IBGE, Contas Nacionais Trimestrais (SIDRA, tabela 5932), PIB a preços de mercado, "
                   "variação em volume. 'Anual' = acumulado em 4 trimestres no 4º trimestre. O IBGE revisa "
                   "trimestres anteriores a cada divulgação; a série é regravada inteira a cada coleta.")

    # ---- PIB nominal
    st.subheader("PIB crescimento nominal")
    if pib.empty or ipca.empty:
        st.warning("PIB nominal indisponível: depende do PIB e do IPCA.")
    else:
        modo = st.radio("Medida", list(PIB_MODOS), horizontal=True, key="mac_pibn_modo")
        t = janela(calc_pib_nominal(pib, ipca, modo))
        c1, c2 = st.columns([3, 1])
        c1.plotly_chart(_grafico(t, "pib_nominal_pct", f"PIB nominal — {modo.lower()} (%)"), width="stretch")
        if len(t):
            c2.metric(f"Último: {t['periodo'].iloc[-1]}", f"{t['pib_nominal_pct'].iloc[-1]:.1f}%",
                      f"real {t['pib_real_pct'].iloc[-1]:.1f}% · IPCA {t['ipca_pct'].iloc[-1]:.1f}%", delta_color="off")
        c2.dataframe(t[["periodo", "pib_real_pct", "ipca_pct", "pib_nominal_pct"]].iloc[::-1], hide_index=True,
                     height=260, column_config={"periodo": "Período",
                                                "pib_real_pct": st.column_config.NumberColumn("Real %", format="%.1f"),
                                                "ipca_pct": st.column_config.NumberColumn("IPCA %", format="%.2f"),
                                                "pib_nominal_pct": st.column_config.NumberColumn("Nominal %", format="%.1f")})
        exportar[f"PIB nominal {modo[:12]}"] = t.assign(data=t["data"].dt.date)
        st.caption("Composição: (1 + PIB real) × (1 + IPCA) − 1. O IPCA usa o nível médio de preços de cada período, "
                   "no mesmo recorte da taxa do PIB (o PIB é fluxo). Não é o deflator implícito do IBGE, que mede "
                   "preços de toda a produção e costuma divergir do IPCA. Na medida dessazonalizada, o IPCA não tem "
                   "ajuste sazonal.")

    # ---- Câmbio
    st.subheader("Câmbio — R$ por US$ (PTAX venda)")
    if cambio.empty:
        st.warning("Câmbio indisponível nesta coleta.")
    else:
        modo = st.radio("Frequência", CAMBIO_MODOS, horizontal=True, index=1, key="mac_cambio_modo")
        t = janela(calc_cambio(cambio, modo))
        c1, c2 = st.columns([3, 1])
        c1.plotly_chart(_grafico(t, "ptax_venda", f"PTAX venda — {modo.lower()}",
                                 "linha", "R$ por US$"), width="stretch")
        if len(t) > 1:
            var = (t["ptax_venda"].iloc[-1] / t["ptax_venda"].iloc[0] - 1) * 100
            c2.metric(f"Variação {t['periodo'].iloc[0]} → {t['periodo'].iloc[-1]}", f"{var:+.1f}%",
                      help="Positivo = real se desvalorizou frente ao dólar.")
        ult = cambio.sort_values("data").iloc[-1]
        c2.metric("Última PTAX", f"R$ {ult['ptax_venda']:.4f}".replace(".", ","), ult["data"].strftime("%d/%m/%Y"),
                  delta_color="off")
        c2.dataframe(t[["periodo", "ptax_venda"]].iloc[::-1], hide_index=True, height=220,
                     column_config={"periodo": "Período",
                                    "ptax_venda": st.column_config.NumberColumn("R$/US$", format="%.4f")})
        exportar[f"PTAX {modo}"] = t.assign(data=t["data"].dt.date)
        st.caption("Fonte: Banco Central do Brasil (SGS, série 1 — PTAX venda). 'Fim' = último pregão do período; "
                   "'Média' = média simples dos pregões do período. No período corrente, valores até o último dado.")

    if exportar:
        st.download_button("⬇️ Séries exibidas (XLSX)", excel_bytes(exportar), file_name=f"macro_{a0}_{a1}.xlsx",
                           key="mac_xlsx",
                           mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet")
