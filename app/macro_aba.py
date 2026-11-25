"""
macro_aba.py — aba "Macro (inflação, PIB, câmbio)" do app Price Discovery Scanning.
Lê saidas/macro/ (gerado pelo workflow macro, script macro.py) e mostra IPCA, PIB, PTAX, contas públicas (NFSP e
dívida), despesas do governo central (RTN), reservas internacionais, balança comercial e o consolidado do Focus.
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
                "Variação %": (f"{_br((c / b - 1) * 100, 1, True)}%" if _unid(ind) == "R$/US$" and b and pd.notna(c)
                               else ""),
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
    d_b = c1.date_input("Relatório base", value=datas.iloc[max(len(datas) - 5, 0)].date(), min_value=datas.min().date(),
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
               f"({int(((datas > base) & (datas <= comp)).sum())} relatórios semanais depois da base). Datas = último dia da semana com dado (sexta), que é o "
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

# ----------------------------------------------------------------------------- contas públicas, despesas e setor externo
ARQS_EXT = ["fiscal_mensal.csv", "rtn_mensal.csv", "reservas_diario.csv", "balanca_mensal.csv"]
ARQS += ARQS_EXT
FLUXO_MODOS = ["Mensal", "Acumulado em 12 meses", "Anual"]
FISCAL_MEDIDAS = ["% do PIB (acumulado em 12 meses)", "R$ bilhões — mensal", "R$ bilhões — acumulado em 12 meses",
                  "R$ bilhões — anual"]
ESFERAS = {"Setor público consolidado": "spc", "Governo central (Gov. Federal + BCB)": "gc"}
RTN_COMPONENTES = {"beneficios_previdenciarios": "Benefícios previdenciários",
                   "pessoal": "Pessoal e encargos sociais",
                   "outras_obrigatorias": "Outras obrigatórias",
                   "obrigatorias_controle_fluxo": "Obrigatórias com controle de fluxo",
                   "discricionarias": "Discricionárias (Executivo)"}
CORES_RTN = ["#1F4E79", "#2E86C1", "#85C1E9", "#A569BD", "#E59866"]
RESERVAS_MODOS = ["Diário", "Fim do mês", "Fim do trimestre", "Fim do ano"]


@st.cache_data(max_entries=1, show_spinner=False)
def carregar_externo(pasta: str, versao: tuple = ()):
    p = Path(pasta)
    ler = lambda n, **k: pd.read_csv(p / n, **k) if (p / n).exists() else pd.DataFrame()  # noqa: E731
    return (ler("fiscal_mensal.csv"), ler("rtn_mensal.csv"), ler("reservas_diario.csv", parse_dates=["data"]),
            ler("balanca_mensal.csv"))


def _mes_fim(per: pd.PeriodIndex) -> pd.Series:
    return pd.Series(per.to_timestamp(how="end").normalize(), index=range(len(per)))


def fluxo(df: pd.DataFrame, cols: list, modo: str, ini=None, fim=None) -> pd.DataFrame:
    """Agrega fluxos mensais (coluna 'mes'). 12 meses: soma móvel sobre a série inteira (meses consecutivos) e depois
    recorta; mensal e anual: só os meses do recorte (anos de borda incompletos ficam marcados como parciais)."""
    m = df.assign(per=pd.PeriodIndex(df["mes"], freq="M")).set_index("per")[cols].sort_index()
    m = m.reindex(pd.period_range(m.index.min(), m.index.max(), freq="M"))
    def no_recorte(idx):
        ok = np.ones(len(idx), dtype=bool)
        if ini is not None:
            ok &= np.asarray(idx >= pd.Period(ini, freq="M"))
        if fim is not None:
            ok &= np.asarray(idx <= pd.Period(fim, freq="M"))
        return ok
    if modo.startswith("Acumulado"):
        v = m.rolling(12, min_periods=12).sum()
        v = v[no_recorte(v.index)].dropna(how="all")
        out = v.reset_index(names="per")
        out["periodo"], out["parcial"] = out["per"].astype(str), False
    else:
        m = m[no_recorte(m.index)].dropna(how="all")
        if modo == "Mensal":
            out = m.reset_index(names="per")
            out["periodo"], out["parcial"] = out["per"].astype(str), False
        else:
            g = m.groupby(m.index.asfreq("Y"))
            out = g.sum(min_count=1)
            out["n"] = g.size()
            out = out.reset_index(names="per")
            out["parcial"] = out["n"] < 12
            out["periodo"] = [f"{p.year}" + (" (parcial)" if parc else "") for p, parc in zip(out["per"], out["parcial"])]
            out = out.drop(columns="n")
    out["data"] = _mes_fim(pd.PeriodIndex(out["per"]))
    return out.drop(columns="per")


def deflator_ipca(ipca: pd.DataFrame, meses: pd.Series, ref: pd.Period) -> pd.Series:
    """Fator que leva R$ de cada mês para R$ do mês de referência (IPCA): índice(ref) / índice(mês)."""
    idx = ipca_indice(ipca)
    base = idx.get(ref, np.nan)
    return pd.Series(base / idx.reindex(pd.PeriodIndex(meses, freq="M")).to_numpy(), index=meses.index)


def _datas_mes(rotulo: str, meses: pd.Series, padrao_ini, key: str):
    per = pd.PeriodIndex(meses, freq="M")
    return _datas(rotulo, per.min().to_timestamp(), per.max().to_timestamp(how="end"), padrao_ini, key)


def _layout(fig, titulo, rot_y, categorias=None, h=400):
    fig.update_layout(title=titulo, height=h, margin=dict(l=10, r=10, t=50, b=10), yaxis_title=rot_y,
                      legend=dict(orientation="h", yanchor="bottom", y=-0.25, x=0), hovermode="x unified")
    if categorias is not None:
        fig.update_xaxes(type="category", categoryorder="array", categoryarray=list(categorias))
    return fig


def render_fiscal(fiscal: pd.DataFrame, ini_exib, exportar: dict):
    import plotly.graph_objects as go
    st.subheader("Contas públicas — resultado primário e nominal")
    if fiscal.empty:
        st.info("Séries fiscais ainda não coletadas (saidas/macro/fiscal_mensal.csv). Rode o workflow macro.")
        return
    c1, c2 = st.columns([1, 2])
    esfera = c1.radio("Esfera", list(ESFERAS), key="mac_fis_esfera")
    medida = c2.radio("Medida", FISCAL_MEDIDAS, horizontal=True, key="mac_fis_medida")
    e = ESFERAS[esfera]
    f0, f1 = _datas_mes("Contas públicas", fiscal["mes"], ini_exib, "mac_fis_dt")
    # resultado = −NFSP (positivo = superávit); juros aparecem negativos (custo)
    nomes = {"primario": "Resultado primário", "juros": "Juros nominais", "nominal": "Resultado nominal"}
    if medida.startswith("%"):
        cols = [f"{e}_{k}_12m_pib" for k in nomes]
        per = pd.PeriodIndex(fiscal["mes"], freq="M")
        t = fiscal[(per >= pd.Period(f0, freq="M")) & (per <= pd.Period(f1, freq="M"))].dropna(subset=cols, how="all")
        t = pd.DataFrame({"periodo": t["mes"], "data": _mes_fim(pd.PeriodIndex(t["mes"], freq="M")).to_numpy(),
                          **{nomes[k]: -t[c].to_numpy() for k, c in zip(nomes, cols)}, "parcial": False})
        unid, casas = "% do PIB", 2
    else:
        cols = [f"{e}_{k}" for k in nomes]
        modo = {"mensal": "Mensal", "acumulado em 12 meses": "Acumulado em 12 meses", "anual": "Anual"}[
            medida.split("— ")[1]]
        a = fluxo(fiscal, cols, modo, f0, f1)
        t = pd.DataFrame({"periodo": a["periodo"], "data": a["data"],
                          **{nomes[k]: (-a[c] / 1000).round(2).to_numpy() for k, c in zip(nomes, cols)},
                          "parcial": a["parcial"]})
        unid, casas = "R$ bilhões", 1
    t = t.reset_index(drop=True)
    g1, g2 = st.columns([3, 1])
    if t.empty:
        g1.info("Sem dados no recorte escolhido.")
        return
    fig = go.Figure()
    if medida.startswith("%"):
        for nome, cor in zip(nomes.values(), ["#2E6FBA", "#C0504D", "#333333"]):
            fig.add_trace(go.Scatter(x=t["data"], y=t[nome], name=nome, mode="lines", line=dict(color=cor, width=2.5)))
        fig.add_hline(y=0, line_width=1, line_color="#999")
        _layout(fig, f"{esfera} — resultado acumulado em 12 meses (% do PIB)", unid)
    else:
        fig.add_trace(go.Bar(x=t["periodo"], y=t["Resultado primário"], name="Resultado primário", marker_color="#2E6FBA"))
        fig.add_trace(go.Bar(x=t["periodo"], y=t["Juros nominais"], name="Juros nominais", marker_color="#C0504D"))
        fig.add_trace(go.Scatter(x=t["periodo"], y=t["Resultado nominal"], name="Resultado nominal", mode="lines+markers",
                                 line=dict(color="#333333", width=2.5), marker=dict(size=5)))
        fig.update_layout(barmode="relative")
        _layout(fig, f"{esfera} — {medida.split('— ')[1]} ({unid})", unid, t["periodo"])
    g1.plotly_chart(fig, width="stretch")
    u = t.iloc[-1]
    fmt = ((lambda x: f"{_br(x, casas, True)}% do PIB") if medida.startswith("%")  # noqa: E731
           else (lambda x: f"R$ {_br(x, casas, True)} bi"))
    g2.metric(f"Primário — {u['periodo']}", fmt(u["Resultado primário"]),
              "superávit" if u["Resultado primário"] >= 0 else "déficit", delta_color="off")
    g2.metric(f"Nominal — {u['periodo']}", fmt(u["Resultado nominal"]),
              f"juros {fmt(u['Juros nominais'])}", delta_color="off")
    g2.dataframe(t[["periodo"] + list(nomes.values())].iloc[::-1], hide_index=True, height=230,
                 column_config={"periodo": "Período", **{n: st.column_config.NumberColumn(n, format=f"%.{casas}f")
                                                         for n in nomes.values()}})
    exportar[f"Fiscal {e} {medida[:12]}"] = t.assign(data=pd.to_datetime(t["data"]).dt.date)
    st.caption("Fonte: Banco Central do Brasil, Estatísticas Fiscais — NFSP sem desvalorização cambial (SGS: setor "
               "público consolidado 4649/4616/4583 e 5793/5760/5727; governo central 4639/4606/4573 e 5783/5750/5717). "
               "O BCB publica as NFSP com sinal positivo para déficit; aqui o sinal foi invertido (positivo = "
               "superávit; juros como custo, negativos). Resultado nominal = primário − juros. Critério 'abaixo da "
               "linha' (variação da dívida), que difere do resultado do Tesouro 'acima da linha'. Anual e 12 meses em "
               "R$ = soma dos fluxos mensais em valores correntes (cálculo próprio).")

    # ---- dívida
    per = pd.PeriodIndex(fiscal["mes"], freq="M")
    d = fiscal[(per >= pd.Period(f0, freq="M")) & (per <= pd.Period(f1, freq="M"))][["mes", "dlsp_pib", "dbgg_pib"]]
    d = d.dropna(how="all", subset=["dlsp_pib", "dbgg_pib"])
    if len(d):
        d = d.assign(data=_mes_fim(pd.PeriodIndex(d["mes"], freq="M")).to_numpy())
        h1, h2 = st.columns([3, 1])
        fig = go.Figure()
        for c, nome, cor in [("dbgg_pib", "Dívida bruta do governo geral (DBGG)", "#C0504D"),
                             ("dlsp_pib", "Dívida líquida do setor público (DLSP)", "#2E6FBA")]:
            fig.add_trace(go.Scatter(x=d["data"], y=d[c], name=nome, mode="lines", line=dict(color=cor, width=2.5)))
        h1.plotly_chart(_layout(fig, "Dívida pública (% do PIB)", "% do PIB", h=360), width="stretch")
        for c, nome in [("dbgg_pib", "DBGG"), ("dlsp_pib", "DLSP")]:
            s = d.dropna(subset=[c])
            if len(s):
                h2.metric(f"{nome} — {s['mes'].iloc[-1]}", f"{_br(s[c].iloc[-1], 1)}% do PIB",
                          f"{_br(s[c].iloc[-1] - s[c].iloc[0], 1, True)} p.p. desde {s['mes'].iloc[0]}",
                          delta_color="inverse")
        exportar["Dívida % PIB"] = d.assign(data=d["data"].dt.date)
        st.caption("Fonte: Banco Central do Brasil (SGS 13762 — DBGG, metodologia a partir de 2008; SGS 4513 — DLSP do "
                   "setor público consolidado). Saldos de fim de mês em % do PIB dos últimos 12 meses.")


def render_despesas(rtn: pd.DataFrame, ipca: pd.DataFrame, ini_exib, exportar: dict):
    import plotly.graph_objects as go
    st.subheader("Despesas e receitas do Governo Central — Tesouro Nacional")
    if rtn.empty:
        st.info("Resultado do Tesouro ainda não coletado (saidas/macro/rtn_mensal.csv). Rode o workflow macro.")
        return
    r0, r1 = _datas_mes("Despesas", rtn["mes"], ini_exib, "mac_rtn_dt")
    c1, c2 = st.columns([2, 1])
    modo = c1.radio("Periodicidade", FLUXO_MODOS, index=1, horizontal=True, key="mac_rtn_modo")
    real = c2.radio("Valores", ["Nominais (correntes)", "Reais (IPCA)"], index=1, horizontal=True,
                    key="mac_rtn_real") == "Reais (IPCA)"
    cols = ["receita_liquida", "despesa_total"] + list(RTN_COMPONENTES) + ["primario_acima_linha"]
    base = rtn[["mes"] + cols].copy()
    ref = pd.Period(base["mes"].iloc[-1], freq="M")
    if real:
        if ipca.empty:
            st.warning("IPCA indisponível: mostrando valores nominais.")
            real = False
        else:
            ult_ipca = pd.Period(ipca["mes"].iloc[-1], freq="M")
            ref = min(ref, ult_ipca)
            fat = deflator_ipca(ipca, base["mes"], ref)
            base[cols] = base[cols].mul(fat, axis=0)
    t = fluxo(base, cols, modo, r0, r1)
    t[cols] = t[cols] / 1000
    if t.empty:
        st.info("Sem dados no recorte escolhido.")
        return
    g1, g2 = st.columns([3, 1])
    fig = go.Figure()
    for (c, nome), cor in zip(RTN_COMPONENTES.items(), CORES_RTN):
        fig.add_trace(go.Bar(x=t["periodo"], y=t[c], name=nome, marker_color=cor))
    fig.add_trace(go.Scatter(x=t["periodo"], y=t["receita_liquida"], name="Receita líquida", mode="lines+markers",
                             line=dict(color="#2CA02C", width=2.5), marker=dict(size=5)))
    fig.update_layout(barmode="stack")
    unid = f"R$ bilhões {'de ' + ref.strftime('%m/%Y') + ' (IPCA)' if real else 'correntes'}"
    g1.plotly_chart(_layout(fig, f"Despesa total por grupo vs. receita líquida — {modo.lower()}", unid, t["periodo"],
                            h=440), width="stretch")
    u = t.iloc[-1]
    g2.metric(f"Despesa total — {u['periodo']}", f"R$ {_br(u['despesa_total'], 1)} bi",
              f"receita líquida R$ {_br(u['receita_liquida'], 1)} bi", delta_color="off")
    g2.metric("Resultado primário (acima da linha)", f"R$ {_br(u['primario_acima_linha'], 1, True)} bi",
              "superávit" if u["primario_acima_linha"] >= 0 else "déficit", delta_color="off")
    comp = pd.DataFrame({"Grupo": list(RTN_COMPONENTES.values()),
                         "R$ bi": [u[c] for c in RTN_COMPONENTES],
                         "% da despesa": [100 * u[c] / u["despesa_total"] for c in RTN_COMPONENTES]})
    g2.dataframe(comp, hide_index=True, height=230, column_config={
        "R$ bi": st.column_config.NumberColumn(format="%.1f"),
        "% da despesa": st.column_config.NumberColumn(format="%.1f%%")})

    # crescimento real em 12 meses (sempre em termos reais, independentemente da escolha acima)
    if not ipca.empty:
        b = rtn[["mes", "despesa_total", "receita_liquida"] + list(RTN_COMPONENTES)].copy()
        ref_r = min(pd.Period(b["mes"].iloc[-1], freq="M"), pd.Period(ipca["mes"].iloc[-1], freq="M"))
        cc = [c for c in b.columns if c != "mes"]
        b[cc] = b[cc].mul(deflator_ipca(ipca, b["mes"], ref_r), axis=0)
        a12 = fluxo(b, cc, "Acumulado em 12 meses")
        a12 = a12.set_index("periodo")[cc]
        var = (a12 / a12.shift(12) - 1) * 100
        per = pd.PeriodIndex(var.index, freq="M")
        var = var[(per >= pd.Period(r0, freq="M")) & (per <= pd.Period(r1, freq="M"))].dropna(how="all")
        if len(var):
            sel = st.multiselect("Crescimento real em 12 meses — séries",
                                 ["Despesa total", "Receita líquida"] + list(RTN_COMPONENTES.values()),
                                 default=["Despesa total", "Receita líquida"], key="mac_rtn_cresc")
            nome_col = {"Despesa total": "despesa_total", "Receita líquida": "receita_liquida",
                        **{v: k for k, v in RTN_COMPONENTES.items()}}
            fig = go.Figure()
            for n in sel:
                fig.add_trace(go.Scatter(x=list(var.index), y=var[nome_col[n]].round(2), name=n, mode="lines"))
            fig.add_hline(y=0, line_width=1, line_color="#999")
            st.plotly_chart(_layout(fig, "Variação real: 12 meses contra os 12 meses anteriores (%)", "%", h=360),
                            width="stretch")
            exportar["RTN cresc real 12m"] = var.reset_index().rename(columns={"index": "periodo"})
    exportar[f"RTN {modo[:12]}{' real' if real else ''}"] = t.assign(data=pd.to_datetime(t["data"]).dt.date)
    st.caption(f"Fonte: Tesouro Nacional, Resultado do Tesouro Nacional (RTN), Série Histórica Mensal, tabela 1.1 "
               f"(Tesouro Transparente); último mês disponível: {rtn['mes'].iloc[-1]} (o RTN sai cerca de um mês depois "
               "do mês de referência). Governo central = Tesouro, Previdência (RGPS) e Banco Central. Receita líquida = "
               "receita total − transferências por repartição a estados e municípios. 'Obrigatórias com controle de "
               "fluxo' e 'discricionárias' compõem as despesas do Executivo sujeitas à programação financeira. Valores "
               "reais: cada mês corrigido pelo IPCA até o mês de referência; 12 meses e anual = soma dos meses "
               "(cálculo próprio). O resultado 'acima da linha' do Tesouro difere do primário do BCB ('abaixo da linha').")


def render_externo(reservas: pd.DataFrame, balanca: pd.DataFrame, ini_exib, exportar: dict):
    import plotly.graph_objects as go
    # ---- reservas
    st.subheader("Reservas internacionais")
    if reservas.empty:
        st.info("Reservas ainda não coletadas (saidas/macro/reservas_diario.csv). Rode o workflow macro.")
    else:
        rs = reservas.sort_values("data")
        k0, k1 = _datas("Reservas", rs["data"].min(), rs["data"].max(), ini_exib, "mac_res_dt")
        modo = st.radio("Frequência", RESERVAS_MODOS, index=1, horizontal=True, key="mac_res_modo")
        rec = rs[(rs["data"] >= k0) & (rs["data"] <= k1)]
        c1, c2 = st.columns([3, 1])
        if rec.empty:
            c1.info("Sem dados no recorte escolhido.")
        else:
            if modo == "Diário":
                t = pd.DataFrame({"periodo": rec["data"].dt.strftime("%d/%m/%Y"), "data": rec["data"],
                                  "reservas": rec["reservas_liquidez"]})
            else:
                freq = {"mês": "M", "trimestre": "Q", "ano": "Y"}[modo.split(" do ")[-1]]
                g = rec.assign(grupo=rec["data"].dt.to_period(freq)).groupby("grupo")
                t = pd.DataFrame({"reservas": g["reservas_liquidez"].last(), "data": g["data"].max()}).reset_index()
                t["periodo"] = [str(p) if freq == "M" else (_tri(p) if freq == "Q" else str(p.year)) for p in t["grupo"]]
            t["reservas"] = (t["reservas"] / 1000).round(2)
            fig = px.line(t, x="data", y="reservas", labels={"data": "", "reservas": "US$ bilhões"})
            c1.plotly_chart(_layout(fig, f"Reservas internacionais — conceito liquidez ({modo.lower()})", "US$ bilhões",
                                    h=380), width="stretch")
            v0, v1 = rec["reservas_liquidez"].iloc[0] / 1000, rec["reservas_liquidez"].iloc[-1] / 1000
            c2.metric(f"Reservas em {rec['data'].iloc[-1]:%d/%m/%Y}", f"US$ {_br(v1, 1)} bi",
                      f"{_br(v1 - v0, 1, True)} bi desde {rec['data'].iloc[0]:%d/%m/%Y}", delta_color="off")
            c2.metric("Mín–máx no recorte", f"US$ {_br(rec['reservas_liquidez'].min() / 1000, 1)}–"
                                           f"{_br(rec['reservas_liquidez'].max() / 1000, 1)} bi")
            c2.dataframe(t[["periodo", "reservas"]].iloc[::-1], hide_index=True, height=200,
                         column_config={"periodo": "Período",
                                        "reservas": st.column_config.NumberColumn("US$ bi", format="%.1f")})
            exportar[f"Reservas {modo}"] = t[["periodo", "data", "reservas"]].assign(data=lambda x: x["data"].dt.date)
        st.caption("Fonte: Banco Central do Brasil (SGS 13982 — reservas internacionais, conceito liquidez, US$ milhões, "
                   "diária). Conceito liquidez inclui as posições em linhas de recompra e empréstimos em moeda "
                   "estrangeira. 'Fim' = último dado do período.")

    # ---- balança comercial
    st.subheader("Balança comercial — exportações e importações de bens")
    if balanca.empty:
        st.info("Balança comercial ainda não coletada (saidas/macro/balanca_mensal.csv). Rode o workflow macro.")
        return
    b0, b1 = _datas_mes("Balança comercial", balanca["mes"], ini_exib, "mac_bal_dt")
    modo = st.radio("Periodicidade", FLUXO_MODOS, index=1, horizontal=True, key="mac_bal_modo")
    cols = ["exportacoes", "importacoes", "saldo"]
    t = fluxo(balanca, cols, modo, b0, b1)
    t[cols] = (t[cols] / 1000).round(2)
    c1, c2 = st.columns([3, 1])
    if t.empty:
        c1.info("Sem dados no recorte escolhido.")
        return
    fig = go.Figure()
    fig.add_trace(go.Bar(x=t["periodo"], y=t["exportacoes"], name="Exportações", marker_color="#2E6FBA"))
    fig.add_trace(go.Bar(x=t["periodo"], y=t["importacoes"], name="Importações", marker_color="#E59866"))
    fig.add_trace(go.Scatter(x=t["periodo"], y=t["saldo"], name="Saldo", mode="lines+markers",
                             line=dict(color="#2CA02C", width=2.5), marker=dict(size=5)))
    fig.update_layout(barmode="group")
    c1.plotly_chart(_layout(fig, f"Balança comercial de bens — {modo.lower()} (US$ bilhões)", "US$ bilhões",
                            t["periodo"]), width="stretch")
    u = t.iloc[-1]
    c2.metric(f"Saldo — {u['periodo']}", f"US$ {_br(u['saldo'], 1, True)} bi",
              f"exp. {_br(u['exportacoes'], 1)} · imp. {_br(u['importacoes'], 1)}", delta_color="off")
    a12 = fluxo(balanca, cols, "Acumulado em 12 meses").set_index("periodo")
    if len(a12) > 12:
        ult = a12.iloc[-1]
        ant = a12.iloc[-13]
        c2.metric(f"12 meses até {a12.index[-1]} vs. 12 meses antes",
                  f"exp. {_br((ult['exportacoes'] / ant['exportacoes'] - 1) * 100, 1, True)}%",
                  f"imp. {_br((ult['importacoes'] / ant['importacoes'] - 1) * 100, 1, True)}%", delta_color="off")
    c2.dataframe(t[["periodo"] + cols].iloc[::-1], hide_index=True, height=200, column_config={
        "periodo": "Período", "exportacoes": st.column_config.NumberColumn("Exp.", format="%.1f"),
        "importacoes": st.column_config.NumberColumn("Imp.", format="%.1f"),
        "saldo": st.column_config.NumberColumn("Saldo", format="%.1f")})
    exportar[f"Balança {modo[:12]}"] = t.assign(data=pd.to_datetime(t["data"]).dt.date)
    st.caption("Fonte: Banco Central do Brasil, Balanço de Pagamentos (BPM6) — SGS 22708 (exportação de bens), 22709 "
               "(importação de bens) e 22707 (saldo), US$ milhões. Critério do balanço de pagamentos (inclui ajustes "
               "de cobertura e mercadorias sob merchanting), que difere da balança da Secex/MDIC. 12 meses e anual = "
               "soma dos meses (cálculo próprio); anos incompletos no recorte aparecem como parciais.")


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

    # ---- Contas públicas, despesas do governo central e setor externo
    fiscal, rtn, reservas, balanca = carregar_externo(str(pasta), _versao(pasta))
    render_fiscal(fiscal, ini_exib, exportar)
    render_despesas(rtn, ipca, ini_exib, exportar)
    render_externo(reservas, balanca, ini_exib, exportar)

    # ---- Focus
    render_focus(carregar_focus(str(pasta), _versao(pasta)), exportar)

    if exportar:
        st.download_button("⬇️ Séries exibidas (XLSX)", excel_bytes(exportar), file_name=f"macro_{pd.Timestamp.today():%Y%m%d}.xlsx",
                           key="mac_xlsx",
                           mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet")
