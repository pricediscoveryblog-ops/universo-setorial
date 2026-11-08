"""
Price Discovery — módulo EUA do app (universo americano, dados da SEC).

Lê a base gerada pelo repositório privado pricediscoveryblog-ops/coletor-eua (branch "dados"),
direto do GitHub, com o token só de leitura guardado nos Secrets do Streamlit:
    GITHUB_TOKEN_EUA = "github_pat_..."
Para testes locais: variável de ambiente EUA_DIR apontando para uma pasta com os arquivos.

Primeira versão: só fundamentos (sem preços/múltiplos). Valores em US$ milhões.
"""
from __future__ import annotations

import io
import json
import os
from pathlib import Path

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import requests
import streamlit as st

REPO = "pricediscoveryblog-ops/coletor-eua"
RAMO = "dados"
LOCAL = os.environ.get("EUA_DIR")
ARQUIVOS = {"LTM": "eua_ltm.parquet", "Anual": "eua_anual.parquet", "Trimestral": "eua_trimestral.parquet"}

# chave: (rótulo, formato, grupo)   formato: mm = US$ milhões | pct = % | x = múltiplo
METRICAS = {
    "receita": ("Receita líquida", "mm", "Resultado"),
    "lucro_bruto": ("Lucro bruto", "mm", "Resultado"),
    "ebit": ("EBIT (operating income)", "mm", "Resultado"),
    "da": ("D&A", "mm", "Resultado"),
    "impairment": ("Impairment", "mm", "Resultado"),
    "ebitda": ("EBITDA (US GAAP)", "mm", "Resultado"),
    "ebitda_pre_ifrs16": ("EBITDA após arrend. financeiros (≈ pré-IFRS 16)", "mm", "Resultado"),
    "ebitda_ifrs16_equiv": ("EBITDA equivalente IFRS 16", "mm", "Resultado"),
    "lucro_liquido": ("Lucro líquido (consolidado)", "mm", "Resultado"),
    "lucro_controladores": ("Lucro líquido (controladores)", "mm", "Resultado"),
    "fco": ("Fluxo de caixa operacional", "mm", "Caixa"),
    "capex_organico": ("Capex orgânico", "mm", "Caixa"),
    "fcl_pos_arrend": ("FCL após arrendamentos", "mm", "Caixa"),
    "aquisicoes_participacoes": ("Aquisições", "mm", "Caixa"),
    "dividendos_pagos": ("Dividendos pagos", "mm", "Caixa"),
    "recompra_acoes": ("Recompra de ações", "mm", "Caixa"),
    "emissao_acoes": ("Emissão de ações", "mm", "Caixa"),
    "retorno_acionista_caixa": ("Retorno ao acionista (div. + recompra − emissão)", "mm", "Caixa"),
    "ativo_total": ("Ativo total", "mm", "Balanço"),
    "caixa": ("Caixa e aplicações de curto prazo", "mm", "Balanço"),
    "emprestimos_financiamentos": ("Dívida bruta (ex-arrendamentos)", "mm", "Balanço"),
    "passivo_arrendamento": ("Passivo de arrendamento", "mm", "Balanço"),
    "div_liq_sem_arrend": ("Dívida líquida sem arrendamentos", "mm", "Balanço"),
    "div_liq_com_arrend": ("Dívida líquida com arrendamentos", "mm", "Balanço"),
    "patrimonio_liquido": ("Patrimônio líquido", "mm", "Balanço"),
    "ncav": ("NCAV (ativo circ. − passivo total)", "mm", "Balanço"),
    "margem_bruta": ("Margem bruta", "pct", "Índices"),
    "margem_ebitda": ("Margem EBITDA", "pct", "Índices"),
    "margem_ebitda_pre_ifrs16": ("Margem EBITDA após arrend. financeiros", "pct", "Índices"),
    "margem_liquida": ("Margem líquida", "pct", "Índices"),
    "roe_pl_final": ("ROE (PL final)", "pct", "Índices"),
    "capex_org_receita": ("Capex orgânico / receita", "pct", "Índices"),
    "capex_org_da": ("Capex orgânico / D&A", "x", "Índices"),
    "div_liq_ebitda_ifrs16": ("Dív. líq. c/ arrend. / EBITDA equiv. IFRS 16", "x", "Índices"),
    "div_liq_ebitda_pre_ifrs16": ("Dív. líq. s/ arrend. / EBITDA após arrend.", "x", "Índices"),
    "cob_juros_caixa": ("Cobertura de juros pagos (EBITDA − IR) / juros", "x", "Índices"),
    "dscr_ebitda": ("DSCR (EBITDA − IR) / (juros + principal)", "x", "Índices"),
}
ROT = {k: v[0] for k, v in METRICAS.items()}
FMT = {k: v[1] for k, v in METRICAS.items()}
IDENT = ["empresa", "ticker", "tickers", "cik", "setor", "subsetor", "segmento", "tipo_especial", "bolsa", "tipo",
         "exercicio", "trimestre", "periodo", "dt_ini", "dt_fim", "dt_arquivamento", "fonte", "notas"]
FLUXOS = [k for k, v in METRICAS.items() if v[1] == "mm" and v[2] in ("Resultado", "Caixa")]
DESTAQUE = ["receita", "ebitda", "lucro_controladores", "fco", "capex_organico", "retorno_acionista_caixa",
            "div_liq_com_arrend", "margem_ebitda", "margem_liquida", "roe_pl_final", "div_liq_ebitda_ifrs16"]


# ------------------------------------------------------------------ formatação (padrão brasileiro)
def _br(v: float, casas: int) -> str:
    s = f"{v:,.{casas}f}"
    return s.replace(",", "§").replace(".", ",").replace("§", ".")


def fmt(v, tipo: str) -> str:
    if v is None or (isinstance(v, float) and not np.isfinite(v)) or pd.isna(v):
        return "–"
    if tipo == "pct":
        return _br(100 * v, 1) + "%"
    if tipo == "x":
        return _br(v, 1) + "x"
    return _br(v, 0)


# ------------------------------------------------------------------ acesso aos dados
def _token() -> str | None:
    try:
        return st.secrets.get("GITHUB_TOKEN_EUA")
    except Exception:                                   # sem secrets.toml / sem a chave
        return None


def _baixar(nome: str) -> bytes:
    if LOCAL:
        return (Path(LOCAL) / nome).read_bytes()
    tok = _token()
    url = f"https://api.github.com/repos/{REPO}/contents/dados/{nome}"
    r = requests.get(url, params={"ref": RAMO}, timeout=180,
                     headers={"Authorization": f"Bearer {tok}", "Accept": "application/vnd.github.raw+json",
                              "X-GitHub-Api-Version": "2022-11-28"})
    if r.status_code in (401, 403, 404):
        raise PermissionError(f"GitHub respondeu {r.status_code} ao ler {nome}: confira o token GITHUB_TOKEN_EUA "
                              f"(acesso de leitura ao repositório {REPO}).")
    r.raise_for_status()
    return r.content


@st.cache_data(ttl=900, show_spinner=False)
def metadados() -> dict:
    """Relido a cada 15 min; a data de geração entra na chave do cache da base (troca a base quando o coletor roda)."""
    return json.loads(_baixar("metadados.json"))


@st.cache_resource(max_entries=2, show_spinner="Carregando a base americana…")
def _base(tipo: str, versao: str) -> pd.DataFrame:
    """cache_resource: o DataFrame é compartilhado sem cópia (economiza memória no Streamlit gratuito).
    Tratar como somente leitura."""
    cols = IDENT + list(METRICAS)
    df = pd.read_parquet(io.BytesIO(_baixar(ARQUIVOS[tipo])), columns=cols)       # só as colunas usadas
    for c in METRICAS:
        if c in df.columns:
            df[c] = pd.to_numeric(df[c], errors="coerce").astype("float32")
    for c in ("setor", "subsetor", "segmento", "tipo_especial", "bolsa", "tipo"):
        if c in df.columns:
            df[c] = df[c].fillna("").astype("category")
    df["dt_fim"] = pd.to_datetime(df["dt_fim"])
    df["rotulo"] = df["ticker"].fillna("?") + " — " + df["empresa"].fillna("")
    return df


@st.cache_data(ttl=3600, show_spinner=False)
def auditoria(versao: str) -> pd.DataFrame:
    return pd.read_csv(io.BytesIO(_baixar("auditoria_eua.csv.gz")), compression="gzip")


def periodo_civil(df: pd.DataFrame, tipo: str) -> pd.Series:
    """Rótulo comum para comparar empresas com anos fiscais diferentes: trimestre/ano CIVIL da data de fim."""
    d = df["dt_fim"]
    if tipo == "Anual":
        return d.dt.year.astype(str)
    return d.dt.quarter.astype(str) + "T" + (d.dt.year % 100).astype(str).str.zfill(2)


def ultima_linha(df: pd.DataFrame) -> pd.DataFrame:
    return df.sort_values("dt_fim").groupby("cik", observed=True).tail(1)


def tabela_formatada(df: pd.DataFrame, metricas: list[str]) -> pd.DataFrame:
    out = df.copy()
    for m in metricas:
        if m in out.columns:
            out[m] = [fmt(v, FMT[m]) for v in out[m]]
    return out.rename(columns=ROT)


# ------------------------------------------------------------------ interface
def _sem_acesso(erro: str | None = None):
    st.title("🇺🇸 Universo EUA")
    if erro:
        st.error(erro)
    st.info(
        "Para ler a base americana, o app precisa de um token **só de leitura** do repositório "
        f"`{REPO}`:\n\n"
        "1. GitHub → Settings → Developer settings → Personal access tokens → Fine-grained tokens → "
        "**Generate new token** (ex.: `leitura-eua-app`), *Only select repositories* = `coletor-eua`, "
        "*Contents* = **Read-only**.\n"
        "2. Streamlit Cloud → o app → **Settings → Secrets**, acrescente a linha:\n\n"
        "```\nGITHUB_TOKEN_EUA = \"cole-aqui-o-token\"\n```\n"
        "3. Salve; o app reinicia sozinho.")


def render():
    if not LOCAL and not _token():
        _sem_acesso()
        return
    try:
        meta = metadados()
    except Exception as e:
        _sem_acesso(str(e))
        return
    versao = meta.get("gerado_em", "")

    st.sidebar.title("🇺🇸 Universo EUA")
    st.sidebar.caption(f"Base SEC gerada em {versao} · {meta.get('empresas_com_dados', '–')} empresas · "
                       "US$ milhões")
    tipo = st.sidebar.radio("Periodicidade", list(ARQUIVOS), key="eua_tipo",
                            help="LTM = últimos 12 meses (soma de 4 trimestres). Trimestral = 3 meses.")
    try:
        base = _base(tipo, versao)
    except Exception as e:
        _sem_acesso(str(e))
        return

    incluir_esp = st.sidebar.toggle("Incluir financeiras, REITs", value=False, key="eua_esp",
                                    help="Bancos, seguradoras e REITs têm outra lógica contábil (receita/EBITDA "
                                         "não se aplicam bem). Ficam fora por padrão.")
    f = base if incluir_esp else base[base["tipo_especial"].astype(str) == ""]
    setores = sorted(f["setor"].astype(str).unique())
    sel_setor = st.sidebar.multiselect("Setor (escritório SEC)", setores, placeholder="Todos", key="eua_setor")
    if sel_setor:
        f = f[f["setor"].isin(sel_setor)]
    subs = sorted(f["subsetor"].astype(str).unique())
    sel_sub = st.sidebar.multiselect("Subsetor (SIC 2 dígitos)", subs, placeholder="Todos", key="eua_sub")
    if sel_sub:
        f = f[f["subsetor"].isin(sel_sub)]
    segs = sorted(f["segmento"].astype(str).unique())
    sel_seg = st.sidebar.multiselect("Segmento (SIC 4 dígitos)", segs, placeholder="Todos", key="eua_seg")
    if sel_seg:
        f = f[f["segmento"].isin(sel_seg)]
    st.sidebar.caption(f"{f['cik'].nunique()} empresas · {len(f)} linhas no filtro")

    st.title("🇺🇸 Universo EUA — fundamentos (SEC)")
    st.caption("Fonte: SEC EDGAR (XBRL companyfacts), US GAAP. Valores em US$ milhões. "
               "Primeira versão: sem preços e múltiplos de mercado.")
    aba_setor, aba_emp, aba_rank, aba_cob, aba_met = st.tabs(
        ["Visão do setor", "Empresa", "Ranking", "Cobertura", "Metodologia"])

    with aba_setor:
        _aba_setor(f, tipo)
    with aba_emp:
        _aba_empresa(base, f, tipo)
    with aba_rank:
        _aba_ranking(f, tipo)
    with aba_cob:
        _aba_cobertura(meta, versao)
    with aba_met:
        _aba_metodologia()


def _aba_setor(f: pd.DataFrame, tipo: str):
    if f.empty:
        st.info("Nenhuma empresa no filtro.")
        return
    opcoes = [m for m in FLUXOS + ["ativo_total", "emprestimos_financiamentos", "div_liq_com_arrend",
                                   "patrimonio_liquido"]]
    met = st.selectbox("Indicador (soma das empresas do filtro)", opcoes, format_func=lambda k: ROT[k],
                       key="eua_setor_met")
    d = f[["cik", "dt_fim", met]].dropna(subset=[met]).copy()
    d["periodo_civil"] = periodo_civil(d, tipo)
    d["ordem"] = d["dt_fim"].dt.to_period("Q" if tipo != "Anual" else "Y")
    g = d.groupby("ordem").agg(valor=(met, "sum"), empresas=("cik", "nunique"),
                               rotulo=("periodo_civil", "first")).reset_index().sort_values("ordem")
    g = g.tail(40 if tipo != "Anual" else 15)
    fig = go.Figure()
    fig.add_bar(x=g["rotulo"], y=g["valor"], name=ROT[met], customdata=g["empresas"],
                hovertemplate="%{x}<br>US$ %{y:,.0f} mi<br>%{customdata} empresas<extra></extra>")
    fig.update_layout(height=380, margin=dict(l=10, r=10, t=30, b=10), yaxis_title="US$ milhões",
                      title=f"{ROT[met]} — soma do filtro por período civil")
    st.plotly_chart(fig, width="stretch")
    st.caption("Períodos alinhados pelo trimestre/ano civil da data de encerramento (anos fiscais diferem entre "
               "empresas). O número de empresas por período varia: compare com cautela.")

    ult = ultima_linha(f)
    ult = ult[ult["dt_fim"] >= ult["dt_fim"].max() - pd.Timedelta(days=550)]
    top = ult.sort_values(met, ascending=False).head(25)
    cols = ["ticker", "empresa", "segmento", "periodo", "dt_fim", met, "margem_ebitda", "margem_liquida"]
    t = top[cols].copy()
    t["dt_fim"] = t["dt_fim"].dt.date
    t["participação"] = [fmt(v, "pct") for v in top[met] / ult[met].sum()] if ult[met].sum() else "–"
    st.markdown(f"**Maiores do filtro — último período de cada empresa ({ROT[met]})**")
    st.dataframe(tabela_formatada(t, [met, "margem_ebitda", "margem_liquida"]), hide_index=True,
                 width="stretch")


def _aba_empresa(base: pd.DataFrame, f: pd.DataFrame, tipo: str):
    lista = f.drop_duplicates("cik").sort_values("ticker")["rotulo"].tolist()
    if not lista:
        st.info("Nenhuma empresa no filtro.")
        return
    padrao = next((i for i, r in enumerate(lista) if r.startswith("AAPL —")), 0)
    esc = st.selectbox("Empresa (digite o ticker ou o nome)", lista, index=padrao, key="eua_emp")
    e = base[base["rotulo"] == esc].sort_values("dt_fim")
    if e.empty:
        return
    u = e.iloc[-1]
    st.subheader(f"{u['empresa']} ({u['ticker']})")
    st.caption(f"{u['setor']} · {u['subsetor']} · {u['segmento']} · {u['bolsa']} · CIK {int(u['cik'])}"
               + (f" · tickers: {u['tickers']}" if isinstance(u['tickers'], str) and '|' in u['tickers'] else "")
               + (f" · **{u['tipo_especial']}**" if str(u['tipo_especial']) else ""))
    st.markdown(f"[Arquivamentos na SEC (EDGAR)](https://www.sec.gov/cgi-bin/browse-edgar?action=getcompany"
                f"&CIK={int(u['cik']):010d}&type=10-&dateb=&owner=include&count=40)")

    n = 12 if tipo != "Anual" else 10
    ult = e.tail(n)
    c1, c2 = st.columns(2)
    with c1:
        d = ult[["periodo", "receita", "ebitda", "lucro_controladores"]].melt("periodo", var_name="m", value_name="v")
        d["m"] = d["m"].map(ROT)
        fig = px.bar(d, x="periodo", y="v", color="m", barmode="group", labels={"v": "US$ milhões", "periodo": "",
                                                                                 "m": ""})
        fig.update_layout(height=340, margin=dict(l=10, r=10, t=10, b=10), legend=dict(orientation="h", y=-0.2))
        st.plotly_chart(fig, width="stretch")
    with c2:
        d = ult[["periodo", "margem_bruta", "margem_ebitda", "margem_liquida"]].melt("periodo", var_name="m",
                                                                                      value_name="v")
        d["v"] = 100 * d["v"]
        d["m"] = d["m"].map(ROT)
        fig = px.line(d, x="periodo", y="v", color="m", markers=True, labels={"v": "%", "periodo": "", "m": ""})
        fig.update_layout(height=340, margin=dict(l=10, r=10, t=10, b=10), legend=dict(orientation="h", y=-0.2))
        st.plotly_chart(fig, width="stretch")

    grupos = sorted({v[2] for v in METRICAS.values()}, key=["Resultado", "Caixa", "Balanço", "Índices"].index)
    for gname in grupos:
        mets = [k for k, v in METRICAS.items() if v[2] == gname]
        t = ult.set_index("periodo")[mets].T
        t.index = [ROT[m] for m in mets]
        tf = t.copy().astype(object)
        for m, r in zip(mets, tf.index):
            tf.loc[r] = [fmt(v, FMT[m]) for v in t.loc[r]]
        with st.expander(gname, expanded=gname in ("Resultado", "Índices")):
            st.dataframe(tf, width="stretch")

    obs = ult[["periodo", "dt_fim", "dt_arquivamento", "fonte", "notas"]].copy()
    obs["dt_fim"] = obs["dt_fim"].dt.date
    with st.expander("Origem dos números (data de divulgação, método do trimestre, observações)"):
        st.dataframe(obs.rename(columns={"dt_fim": "fim do período", "dt_arquivamento": "1ª divulgação"}),
                     hide_index=True, width="stretch")
    st.download_button("Baixar série da empresa (CSV)", e.drop(columns=["rotulo"]).to_csv(index=False).encode(),
                       file_name=f"{u['ticker']}_{tipo.lower()}.csv", mime="text/csv")


def _aba_ranking(f: pd.DataFrame, tipo: str):
    if f.empty:
        st.info("Nenhuma empresa no filtro.")
        return
    c1, c2, c3 = st.columns([2, 1, 1])
    met = c1.selectbox("Indicador", list(METRICAS), index=list(METRICAS).index("margem_ebitda"),
                       format_func=lambda k: ROT[k], key="eua_rank_met")
    rec_min = c2.number_input("Receita mínima (US$ mi)", min_value=0, value=100, step=100, key="eua_rank_rec",
                              help="Evita distorções de empresas muito pequenas ou pré-operacionais.")
    meses = c3.number_input("Dado mais antigo aceito (meses)", min_value=3, value=15, step=3, key="eua_rank_meses")
    u = ultima_linha(f)
    u = u[u["dt_fim"] >= pd.Timestamp.today() - pd.DateOffset(months=int(meses))]
    if rec_min and tipo != "Trimestral":
        u = u[u["receita"].fillna(0) >= rec_min]
    elif rec_min:
        u = u[u["receita"].fillna(0) * 4 >= rec_min]
    u = u.dropna(subset=[met])
    asc = st.toggle("Ordem crescente", value=met.startswith("div_liq"), key="eua_rank_asc")
    u = u.sort_values(met, ascending=asc)
    st.caption(f"{len(u)} empresas com o indicador no filtro (último período de cada uma).")
    top = u.head(30)
    fig = px.bar(top, x=met, y="ticker", orientation="h", hover_data=["empresa", "periodo"],
                 labels={met: ROT[met], "ticker": ""})
    fig.update_layout(height=max(300, 22 * len(top)), margin=dict(l=10, r=10, t=10, b=10),
                      yaxis=dict(autorange="reversed"))
    if FMT[met] == "pct":
        fig.update_xaxes(tickformat=".0%")
    st.plotly_chart(fig, width="stretch")
    cols = ["ticker", "empresa", "segmento", "periodo", "dt_fim", met] + [m for m in DESTAQUE if m != met]
    t = u[cols].copy()
    t["dt_fim"] = t["dt_fim"].dt.date
    st.dataframe(tabela_formatada(t, [met] + [m for m in DESTAQUE if m != met]), hide_index=True,
                 width="stretch", height=500)
    st.download_button("Baixar ranking (CSV)", u.drop(columns=["rotulo"]).to_csv(index=False).encode(),
                       file_name=f"ranking_eua_{met}.csv", mime="text/csv")


def _aba_cobertura(meta: dict, versao: str):
    c = st.columns(4)
    c[0].metric("Empresas no universo", meta.get("empresas_universo", "–"))
    c[1].metric("Empresas com dados", meta.get("empresas_com_dados", "–"))
    linhas = meta.get("linhas", {})
    c[2].metric("Linhas LTM", f"{linhas.get('ltm', 0):,}".replace(",", "."))
    c[3].metric("Linhas trimestrais", f"{linhas.get('trimestral', 0):,}".replace(",", "."))
    st.caption(f"Gerado em {versao} ({meta.get('modo', '')}). Fonte: {meta.get('fonte', '')}.")
    try:
        a = auditoria(versao)
    except Exception as e:
        st.warning(f"Auditoria indisponível: {e}")
        return
    a["situação"] = a["motivo"].fillna("").replace("", "com dados")
    resumo = a.groupby("situação").size().sort_values(ascending=False).rename("empresas").reset_index()
    st.markdown("**Situação de cada empresa do universo (NYSE, Nasdaq, Cboe, TXSE)**")
    st.dataframe(resumo, hide_index=True, width="stretch")
    busca = st.text_input("Procurar empresa na auditoria (ticker ou nome)", key="eua_aud_busca")
    if busca:
        m = a["ticker"].fillna("").str.contains(busca, case=False) | a["empresa"].fillna("").str.contains(
            busca, case=False)
        st.dataframe(a[m][["ticker", "empresa", "setor", "situação", "exercicios", "ultimo_dt_fim", "ausentes",
                           "tags"]], hide_index=True, width="stretch")


def _aba_metodologia():
    st.markdown("""
**Fonte.** SEC EDGAR, base XBRL *companyfacts* (só tags padrão US GAAP; tags próprias das empresas não entram).
Universo: empresas listadas em NYSE, Nasdaq, Cboe e TXSE que arquivam 10-K/10-Q. Emissores estrangeiros (20-F/40-F,
IFRS) e SPACs ficam fora.

**Classificação.** Setor = escritório da SEC responsável pela empresa; subsetor = grupo SIC de 2 dígitos;
segmento = descrição SIC de 4 dígitos.

**Trimestres e LTM.** 3M reportado quando existe; senão YTD diferenciado (6M−3M, 9M−6M); 4T = anual − 9M.
LTM = soma de 4 trimestres contíguos. Exercício = ano civil do encerramento (Walmart: ano fiscal até jan/2026 = 2026).
Para comparar empresas, a aba *Visão do setor* alinha pelo trimestre/ano **civil** da data de encerramento.

**Point-in-time.** O valor é o mais recente (reapresentações valem); a coluna *1ª divulgação* traz a data em que o
período foi publicado pela primeira vez — é ela que servirá para os múltiplos históricos.

**Arrendamentos (ASC 842 ≠ IFRS 16).** Nos EUA o aluguel de arrendamentos operacionais continua no opex.
- *EBITDA (US GAAP)* = EBIT + D&A + impairment.
- *EBITDA após arrend. financeiros* = EBITDA − pagamentos de arrendamento financeiro (principal + juros):
  comparável ao EBITDA pré-IFRS 16 da base brasileira.
- *EBITDA equivalente IFRS 16* = EBITDA + custo de arrendamento operacional (quando divulgado).
- Dívida líquida com/sem arrendamentos é pareada com o EBITDA correspondente.

**Limitações conhecidas.**
- D&A: quando a empresa usa tag própria para o total do fluxo de caixa (ex.: Microsoft), usamos depreciação +
  amortização de intangíveis em tags padrão — o EBITDA fica conservador.
- Juros e IR pagos muitas vezes só aparecem no 10-K: coberturas trimestrais/LTM podem ficar vazias.
- Bancos, seguradoras e REITs: marcados e ocultos por padrão (modelo próprio pendente).
- Ainda sem preços: múltiplos de mercado entram na próxima versão.
""")
