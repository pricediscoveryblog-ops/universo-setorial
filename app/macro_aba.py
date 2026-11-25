"""
macro_aba.py — aba "Macro (inflação, PIB, câmbio)" do app Price Discovery Scanning.
Lê saidas/macro/ (gerado pelo workflow macro, script macro.py) e mostra IPCA, PIB, PTAX e o consolidado do Focus.
Usado também pelo app BR Toll Roads Fin Data (cópia idêntica do arquivo).
"""
import json
from pathlib import Path

import numpy as np
import pandas as pd
import plotly.express as px
import streamlit as st

ARQS = ["ipca_mensal.csv", "pib_trimestral.csv", "pib_volume_trimestral.csv", "cambio_diario.csv", "metadados.json",
        "focus_anual.csv"]
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


@st.cache_data(max_entries=1, show_spinner=False)
def carregar_focus(pasta: str, versao: tuple = ()) -> pd.DataFrame:
    p = Path(pasta) / "focus_anual.csv"
    return pd.read_csv(p, parse_dates=["data"]) if p.exists() else pd.DataFrame()


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



# ----------------------------------------------------------------------------- recorte por datas e Focus
def _br(x: float, casas: int = 2, sinal: bool = False) -> str:
    t = f"{x:+,.{casas}f}" if sinal else f"{x:,.{casas}f}"
    return t.replace(",", "X").replace(".", ",").replace("X", ".")


def _datas(rotulo: str, minimo, maximo, padrao_ini, key: str):
    """Par de datas (inicial, final) limitado ao intervalo com dado; corrige inversão."""
    minimo, maximo = pd.Timestamp(minimo).date(), pd.Timestamp(maximo).date()
    padrao_ini = min(max(pd.Timestamp(padrao_ini).date(), minimo), maximo)
    c1, c2, _ = st.columns([1, 1, 2])
    d0 = c1.date_input(f"{rotulo} — data inicial", value=padrao_ini, min_value=minimo, max_value=maximo,
                       format="DD/MM/YYYY", key=f"{key}_ini")
    d1 = c2.date_input(f"{rotulo} — data final", value=maximo, min_value=minimo, max_value=maximo,
                       format="DD/MM/YYYY", key=f"{key}_fim")
    if d0 > d1:
        d0, d1 = d1, d0
    return pd.Timestamp(d0), pd.Timestamp(d1)


def ipca_recorte(ipca: pd.DataFrame, ini, fim) -> pd.DataFrame:
    """Meses cheios do recorte: do mês da data inicial ao mês da data final."""
    per = pd.PeriodIndex(ipca["mes"], freq="M")
    return ipca[(per >= pd.Period(ini, freq="M")) & (per <= pd.Period(fim, freq="M"))].reset_index(drop=True)


def calc_ipca_recorte(ipca: pd.DataFrame, modo: str, ini, fim) -> pd.DataFrame:
    """12 meses usa a série inteira (precisa dos 11 meses anteriores) e mostra só o recorte; as demais
    periodicidades são compostas só com os meses do recorte (períodos de borda incompletos ficam parciais)."""
    if modo.startswith("12 meses"):
        t = calc_ipca(ipca, modo)
        per = pd.PeriodIndex(t["periodo"], freq="M")
        return t[(per >= pd.Period(ini, freq="M")) & (per <= pd.Period(fim, freq="M"))].reset_index(drop=True)
    m = ipca_recorte(ipca, ini, fim)
    return calc_ipca(m, modo) if len(m) else pd.DataFrame(columns=["periodo", "data", "ipca_pct", "parcial"])


def ptax_em(c: pd.DataFrame, d):
    """PTAX do último pregão até a data (ou o primeiro depois, se a data for anterior à série)."""
    c = c.sort_values("data")
    ate = c[c["data"] <= d]
    r = ate.iloc[-1] if len(ate) else c[c["data"] >= d].iloc[0]
    return r["data"], float(r["ptax_venda"])


UNID_FOCUS = {"Câmbio": "R$/US$"}
PADRAO_FOCUS = ["IPCA", "PIB Total", "Câmbio", "Selic"]


def _unid(ind: str) -> str:
    if ind in UNID_FOCUS:
        return UNID_FOCUS[ind]
    return "% do PIB" if ind.startswith(("Resultado", "Dívida")) else "%"


def relatorio_em(datas: pd.Series, d) -> pd.Timestamp:
    """Relatório Focus mais recente até a data escolhida (ou o primeiro disponível)."""
    ok = datas[datas <= pd.Timestamp(d)]
    return ok.max() if len(ok) else datas.min()


def sequencia(s: pd.Series) -> str:
    """Comportamento semanal estilo relatório Focus: direção da última mudança e há quantas semanas seguidas."""
    dif = np.sign(s.diff().round(6).dropna().to_numpy())
    if not len(dif):
        return ""
    ult = dif[-1]
    n = 0
    for x in dif[::-1]:
        if x != ult:
            break
        n += 1
    seta = {1: "▲", -1: "▼", 0: "="}[int(ult)]
    return f"{seta} {n}"


def consolidado_focus(f: pd.DataFrame, base, comp, inds: list, anos: list) -> pd.DataFrame:
    linhas = []
    for ind in inds:
        for ano in anos:
            s = f[(f["indicador"] == ind) & (f["ano_ref"] == ano)].set_index("data").sort_index()
            if s.empty:
                continue
            b = s["mediana"].get(base, np.nan)
            c = s["mediana"].get(comp, np.nan)
            h1 = s.loc[:comp, "mediana"]
            linhas.append({
                "Indicador": ind, "Ano": int(ano), "Unidade": _unid(ind),
                "Mediana base": b, "Mediana comparada": c,
                "Variação": c - b,
                "Variação %": (c / b - 1) * 100 if _unid(ind) == "R$/US$" and b else np.nan,
                "Semana anterior": h1.iloc[-2] if len(h1) > 1 else np.nan,
                "Comportamento semanal": sequencia(h1.tail(60)),
                "Mín–máx (comparada)": (f"{_br(s['minimo'].get(comp, np.nan))} – {_br(s['maximo'].get(comp, np.nan))}"
                                        if comp in s.index else ""),
                "Respondentes": s["respondentes"].get(comp, np.nan),
            })
    return pd.DataFrame(linhas)


def render_focus(f: pd.DataFrame, exportar: dict):
    st.subheader("Expectativas de mercado — relatório Focus (BCB)")
    if f.empty:
        st.info("Focus ainda não coletado (saidas/macro/focus_anual.csv). Rode o workflow macro.")
        return
    datas = pd.Series(sorted(f["data"].unique()))
    ult = datas.max()
    inds_disp = sorted(f["indicador"].unique(), key=lambda x: ((PADRAO_FOCUS + [x]).index(x), x))
    c1, c2, c3, c4 = st.columns([1, 1, 2, 2])
    d_b = c1.date_input("Relatório base", value=(ult - pd.Timedelta(weeks=4)).date(), min_value=datas.min().date(),
                        max_value=ult.date(), format="DD/MM/YYYY", key="foc_base",
                        help="Usa o relatório mais recente até a data escolhida.")
    d_c = c2.date_input("Relatório comparado", value=ult.date(), min_value=datas.min().date(), max_value=ult.date(),
                        format="DD/MM/YYYY", key="foc_comp")
    base, comp = relatorio_em(datas, d_b), relatorio_em(datas, d_c)
    if base > comp:
        base, comp = comp, base
    inds = c3.multiselect("Indicadores", inds_disp, default=[i for i in PADRAO_FOCUS if i in inds_disp],
                          key="foc_inds")
    anos_disp = sorted(f.loc[f["data"] == comp, "ano_ref"].unique())
    padrao_anos = [a for a in anos_disp if comp.year <= a <= comp.year + 1] or anos_disp[:2]
    anos = c4.multiselect("Anos de referência", anos_disp, default=padrao_anos, key="foc_anos")
    st.caption(f"Comparando o relatório de {base:%d/%m/%Y} com o de {comp:%d/%m/%Y} "
               f"({(comp - base).days // 7} semanas). Datas = último dia da semana com dado (sexta), que é o "
               "que o relatório Focus publica na segunda-feira seguinte.")
    if not inds or not anos:
        st.info("Escolha ao menos um indicador e um ano de referência.")
        return

    tab = consolidado_focus(f, base, comp, inds, anos)
    if tab.empty:
        st.warning("Sem projeções para essa combinação.")
        return
    st.dataframe(tab, hide_index=True, width="stretch", column_config={
        "Ano": st.column_config.NumberColumn(format="%d"),
        "Mediana base": st.column_config.NumberColumn(f"Mediana {base:%d/%m/%y}", format="%.2f"),
        "Mediana comparada": st.column_config.NumberColumn(f"Mediana {comp:%d/%m/%y}", format="%.2f"),
        "Variação": st.column_config.NumberColumn("Variação (p.p. / R$)", format="%+.2f"),
        "Variação %": st.column_config.NumberColumn(format="%+.1f%%"),
        "Semana anterior": st.column_config.NumberColumn(format="%.2f"),
        "Respondentes": st.column_config.NumberColumn(format="%d")})
    exportar["Focus consolidado"] = tab

    # evolução das medianas no intervalo escolhido
    g1, g2 = st.columns([1, 3])
    ind_g = g1.selectbox("Evolução da mediana", inds, key="foc_ind_g")
    banda = g1.checkbox("Mostrar mín–máx", value=False, key="foc_banda")
    ev = f[(f["indicador"] == ind_g) & f["ano_ref"].isin(anos) & (f["data"] >= base) & (f["data"] <= comp)]
    ev = ev.sort_values("data").assign(ano=lambda x: x["ano_ref"].astype(str))
    import plotly.graph_objects as go
    fig = px.line(ev, x="data", y="mediana", color="ano", markers=len(ev) < 120,
                  labels={"data": "", "mediana": _unid(ind_g), "ano": "Ano de ref."},
                  title=f"Focus — {ind_g}: mediana das projeções ({base:%d/%m/%Y} a {comp:%d/%m/%Y})")
    if banda:
        for ano, d in ev.groupby("ano"):
            fig.add_trace(go.Scatter(x=list(d["data"]) + list(d["data"][::-1]),
                                     y=list(d["maximo"]) + list(d["minimo"][::-1]), fill="toself",
                                     opacity=0.15, line=dict(width=0), name=f"{ano} mín–máx", hoverinfo="skip"))
    fig.update_layout(height=400, margin=dict(l=10, r=10, t=50, b=10))
    g2.plotly_chart(fig, width="stretch")
    exportar["Focus evolução"] = ev.assign(data=ev["data"].dt.date).drop(columns=["ano"])
    st.caption("Fonte: Banco Central do Brasil, Sistema de Expectativas de Mercado (Focus), API Olinda "
               "ExpectativasMercadoAnuais, base de cálculo 0 (todas as instituições). Variação = mediana comparada − "
               "mediana base, em p.p. (taxas e % do PIB) ou R$ (câmbio). Selic = taxa projetada para o fim do ano; "
               "Câmbio = R$/US$ projetado para o fim do ano. Comportamento semanal: ▲ alta / ▼ queda / = estável e "
               "há quantas semanas seguidas, até o relatório comparado.")

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
    exportar = {}

    # ---- Inflação
    st.subheader("Inflação — IPCA")
    if ipca.empty:
        st.warning("IPCA indisponível nesta coleta.")
    else:
        per = pd.PeriodIndex(ipca["mes"], freq="M")
        i0, i1 = _datas("IPCA", per.min().to_timestamp(), per.max().to_timestamp(how="end"), ini_exib, "mac_ipca_dt")
        modo = st.radio("Periodicidade", IPCA_MODOS, horizontal=True, key="mac_ipca_modo")
        t = calc_ipca_recorte(ipca, modo, i0, i1)
        mm = ipca_recorte(ipca, i0, i1)
        c1, c2 = st.columns([3, 1])
        if len(t):
            c1.plotly_chart(_grafico(t, "ipca_pct", f"IPCA — {modo.lower()} (%)",
                                     "linha" if modo.startswith("12") else "barra"), width="stretch")
        if len(mm):
            acum = (np.prod(1 + mm["ipca_mensal"] / 100) - 1) * 100
            n = len(mm)
            anual = ((1 + acum / 100) ** (12 / n) - 1) * 100
            c2.metric(f"IPCA acumulado {mm['mes'].iloc[0]} → {mm['mes'].iloc[-1]}", f"{_br(acum)}%",
                      f"{_br(anual)}% a.a. ({n} meses)", delta_color="off",
                      help="Composição das variações mensais dos meses cheios do recorte. Anualizado = "
                           "(1 + acumulado)^(12/meses) − 1.")
        c2.metric("Último mês disponível", ipca["mes"].iloc[-1], f"{ipca['ipca_mensal'].iloc[-1]:.2f}% no mês",
                  delta_color="off")
        c2.dataframe(t[["periodo", "ipca_pct"]].iloc[::-1], hide_index=True, height=220,
                     column_config={"periodo": "Período", "ipca_pct": st.column_config.NumberColumn("IPCA %", format="%.2f")})
        exportar[f"IPCA {modo[:10]}"] = t.assign(data=pd.to_datetime(t["data"]).dt.date)
        st.caption("Fonte: IBGE, via Banco Central (SGS, série 433 — variação mensal). Trimestral, 12 meses e anual "
                   "são calculados aqui por composição das variações mensais. O recorte usa meses cheios (do mês da "
                   "data inicial ao mês da data final); períodos incompletos no recorte aparecem como parciais. "
                   "Em '12 meses', cada ponto usa os 12 meses anteriores, mesmo que antes da data inicial.")

    # ---- PIB (recorte em anos, comum aos dois blocos de PIB)
    anos = list(range(ini_exib.year, pd.Timestamp.today().year + 1))
    st.subheader("Crescimento do PIB")
    a0, a1 = st.select_slider("Período exibido (anos) — PIB real e nominal", options=anos,
                              value=(anos[0], anos[-1]), key="mac_anos")
    janela = lambda df: df[(df["data"] >= pd.Timestamp(a0, 1, 1)) &  # noqa: E731
                           (df["data"] <= pd.Timestamp(a1, 12, 31))].reset_index(drop=True)
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
        cs = cambio.sort_values("data")
        k0, k1 = _datas("Câmbio", cs["data"].min(), cs["data"].max(), ini_exib, "mac_cambio_dt")
        modo = st.radio("Frequência", CAMBIO_MODOS, horizontal=True, index=1, key="mac_cambio_modo")
        rec = cs[(cs["data"] >= k0) & (cs["data"] <= k1)]
        c1, c2 = st.columns([3, 1])
        if rec.empty:
            c1.info("Sem pregões no recorte escolhido.")
            t = pd.DataFrame(columns=["periodo", "data", "ptax_venda"])
        else:
            t = calc_cambio(rec, modo)
            c1.plotly_chart(_grafico(t, "ptax_venda", f"PTAX venda — {modo.lower()}",
                                     "linha", "R$ por US$"), width="stretch")
            db, vb = ptax_em(cs, k0)
            df_, vf = ptax_em(cs, k1)
            var = (vf / vb - 1) * 100
            anos_r = max((df_ - db).days, 1) / 365.25
            c2.metric(f"Variação {db:%d/%m/%Y} → {df_:%d/%m/%Y}", f"{_br(var, 1, True)}%",
                      f"R$ {_br(vb, 4)} → R$ {_br(vf, 4)}", delta_color="off",
                      help="PTAX do último pregão até cada data escolhida. Positivo = real se desvalorizou "
                           "frente ao dólar.")
            if anos_r >= 1:
                c2.metric("Variação anualizada", f"{_br(((vf / vb) ** (1 / anos_r) - 1) * 100, 1, True)}% a.a.")
            c2.metric("Média no recorte", f"R$ {_br(rec['ptax_venda'].mean(), 4)}",
                      f"mín {_br(rec['ptax_venda'].min(), 4)} · máx {_br(rec['ptax_venda'].max(), 4)}",
                      delta_color="off")
        ult = cs.iloc[-1]
        c2.metric("Última PTAX", f"R$ {ult['ptax_venda']:.4f}".replace(".", ","), ult["data"].strftime("%d/%m/%Y"),
                  delta_color="off")
        c2.dataframe(t[["periodo", "ptax_venda"]].iloc[::-1], hide_index=True, height=220,
                     column_config={"periodo": "Período",
                                    "ptax_venda": st.column_config.NumberColumn("R$/US$", format="%.4f")})
        exportar[f"PTAX {modo}"] = t.assign(data=pd.to_datetime(t["data"]).dt.date)
        st.caption("Fonte: Banco Central do Brasil (SGS, série 1 — PTAX venda). 'Fim' = último pregão do período; "
                   "'Média' = média simples dos pregões do período, só com os pregões dentro do recorte (períodos "
                   "de borda ficam parciais). Variação ponto a ponto entre a PTAX do último pregão até a data inicial "
                   "e até a data final.")

    # ---- Focus
    render_focus(carregar_focus(str(pasta), _versao(pasta)), exportar)

    if exportar:
        st.download_button("⬇️ Séries exibidas (XLSX)", excel_bytes(exportar), file_name=f"macro_{pd.Timestamp.today():%Y%m%d}.xlsx",
                           key="mac_xlsx",
                           mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet")
