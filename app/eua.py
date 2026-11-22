"""
Price Discovery — base americana para o app (modo EUA).

O app usa as MESMAS abas da base brasileira: este módulo só entrega a base da SEC (repositório privado
pricediscoveryblog-ops/coletor-eua, branch "dados") no formato da base da CVM, ajusta os rótulos que mudam
entre IFRS e US GAAP e traz a metodologia própria.

Acesso: token só de leitura nos Secrets do Streamlit  ->  GITHUB_TOKEN_EUA = "github_pat_..."
Testes locais: variável de ambiente EUA_DIR apontando para uma pasta com os arquivos do coletor.
"""
from __future__ import annotations

import ctypes
import gc
import io
import json
import os
from pathlib import Path

import pandas as pd
import requests
import streamlit as st

REPO = "pricediscoveryblog-ops/coletor-eua"
RAMO = "dados"
LOCAL = os.environ.get("EUA_DIR")
ARQUIVOS = ("eua_anual.parquet", "eua_trimestral.parquet", "eua_ltm.parquet")
INICIO = pd.Timestamp("2014-01-01")      # mesmo horizonte da base brasileira (2015+), com 2014 para o crescimento a/a

# rótulos que mudam no US GAAP (ASC 842: arrendamento operacional continua no opex)
ROTULOS_EUA = {
    "ebitda": "EBITDA (US GAAP)",
    "ebitda_pre_ifrs16": "EBITDA após arrend. financeiros (≈ pré-IFRS 16)",
    "margem_ebitda": "Margem EBITDA (US GAAP)",
    "margem_ebitda_pre_ifrs16": "Margem EBITDA após arrend. financeiros",
    "fco": "FCO (US GAAP, como reportado)",
    "fco_reportado": "FCO reportado",
    "pagamento_arrendamento": "Pagamentos de arrendamento financeiro",
    "fcl_pos_arrend": "FCL pós-arrendamento financeiro",
    "dividendos_pagos": "Dividendos pagos",
    "ir_csll_pagos": "Imposto de renda pago",
    "div_liq_ebitda_ifrs16": "DL c/ arrend. / EBITDA equiv. IFRS 16",
    "div_liq_ebitda_pre_ifrs16": "DL s/ arrend. / EBITDA após arrend. financeiros",
    "conversao_fco_ebitda": "Conversão de caixa (FCO ÷ EBITDA US GAAP)",
    "ev_ebitda_ifrs16": "EV/EBITDA equiv. IFRS 16 (c/ arrend.)",
    "ev_ebitda_pre_ifrs16": "EV/EBITDA após arrend. financeiros (s/ arrend.)",
    "dividend_yield": "Dividend yield (dividendos pagos 12m)",
    "roic_caixa": "ROIC caixa (NOPAT + D&A − capex org. − arrend. financ. ÷ CI)",
    "roic_trad": "ROIC tradicional (NOPAT aj. ÷ CI c/ arrendamentos)",
}
# métricas que só existem na base americana (entram na lista do app no modo EUA)
EXTRAS_EUA = {
    "ebitda_ifrs16_equiv": ("EBITDA equivalente IFRS 16 (+ custo de arrend. operacional)", "mm", "Resultado"),
    "lucro_controladores": ("Lucro líquido (controladores)", "mm", "Resultado"),
    "recompra_acoes": ("Recompra de ações", "mm", "Caixa e investimento"),
    "emissao_acoes": ("Emissão de ações", "mm", "Caixa e investimento"),
    "retorno_acionista_caixa": ("Retorno ao acionista (dividendos + recompra − emissão)", "mm", "Caixa e investimento"),
}
# métricas brasileiras sem equivalente na SEC (FCO comparável e frota): saem da lista no modo EUA
SEM_EQUIVALENTE = ["fco_juros_no_fco", "fco_div_no_fco", "capex_frota_liquido",
                   # ajuste de construção de concessões (ICPC 01): só Brasil
                   "receita_construcao", "custo_construcao", "margem_construcao", "receita_reportada",
                   "lucro_bruto_reportado", "ebit_reportado", "ebitda_reportado", "lucro_liquido_reportado",
                   # EBITDA ajustado do setor elétrico (eletricas.yaml): só Brasil
                   "ebitda_ajustado", "margem_ebitda_ajustado", "div_liq_ebitda_ajustado", "dscr_ebitda_ajustado",
                   "ativo_concessao_saldo", "var_ativo_concessao", "custo_implantacao", "vnr_atualizacao",
                   "var_bonificacao"]


def ajustar_metricas(metricas: dict, agreg: dict) -> None:
    """Altera os dicionários do app (no lugar) para o modo EUA."""
    for k, rot in ROTULOS_EUA.items():
        if k in metricas:
            _, f, g = metricas[k]
            metricas[k] = (rot, f, g)
    for k in SEM_EQUIVALENTE:
        metricas.pop(k, None)
    metricas.update(EXTRAS_EUA)
    # alavancagem do setor no mesmo par usado por empresa: dívida c/ arrend. ÷ EBITDA equivalente IFRS 16
    agreg["div_liq_ebitda_ifrs16"] = ("div_liq_com_arrend", "ebitda_ifrs16_equiv")
    agreg["roe_pl_final"] = ("lucro_controladores", "pl_controladores")


# ------------------------------------------------------------------ acesso
def _token() -> str | None:
    try:
        return st.secrets.get("GITHUB_TOKEN_EUA")
    except Exception:
        return None


def tem_acesso() -> bool:
    return True          # coletor-eua é público desde nov/2026: sem token, lê pelo raw.githubusercontent.com


def _baixar(nome: str) -> bytes:
    if LOCAL:
        return (Path(LOCAL) / nome).read_bytes()
    if not _token():
        r = requests.get(f"https://raw.githubusercontent.com/{REPO}/{RAMO}/dados/{nome}", timeout=180)
        r.raise_for_status()
        return r.content
    r = requests.get(f"https://api.github.com/repos/{REPO}/contents/dados/{nome}", params={"ref": RAMO},
                     timeout=180, headers={"Authorization": f"Bearer {_token()}",
                                           "Accept": "application/vnd.github.raw+json",
                                           "X-GitHub-Api-Version": "2022-11-28"})
    if r.status_code in (401, 403, 404):
        raise PermissionError(f"GitHub respondeu {r.status_code} ao ler {nome}: confira o token GITHUB_TOKEN_EUA "
                              f"(leitura do repositório {REPO}).")
    r.raise_for_status()
    return r.content


@st.cache_data(ttl=900, show_spinner=False)
def metadados() -> dict:
    """Relido a cada 15 min; 'gerado_em' entra na chave do cache da base (troca a base quando o coletor roda)."""
    return json.loads(_baixar("metadados.json"))


def versao() -> str:
    return str(metadados().get("gerado_em", ""))


def trimestre_civil(d: pd.Series) -> pd.Series:
    """Fim de trimestre civil mais próximo da data (27/12 -> 31/12; 31/10 -> 30/09; 03/01 -> 31/12).
    As abas comparam empresas pela data de fechamento; nos EUA cada empresa tem o seu calendário fiscal, então os
    trimestres são alinhados ao trimestre civil mais próximo (calendarização usual em comparações setoriais)."""
    q = d.dt.to_period("Q")
    fim = q.dt.end_time.dt.normalize()
    ant = (q - 1).dt.end_time.dt.normalize()
    return fim.where((fim - d).abs() <= (d - ant).abs(), ant).astype(d.dtype)


def _cnpj(cik) -> str:
    """'CIK' + 11 dígitos = 14 caracteres, o tamanho do CNPJ (o app completa CNPJs com zeros à esquerda até 14)."""
    return "CIK" + str(int(cik)).zfill(11)


def _cik_de(cnpj: pd.Series) -> pd.Series:
    return pd.to_numeric(cnpj.astype(str).str.replace("CIK", "", regex=False), errors="coerce")


# ------------------------------------------------------------------ base no formato da CVM
# colunas da base do coletor que as abas usam (o resto fica no repositório coletor-eua)
COLUNAS = [
    "empresa", "ticker", "setor", "subsetor", "segmento", "cnpj", "tipo", "periodo", "exercicio", "trimestre", "dt_fim",
    "fonte", "regime_contabil", "notas", "tipo_especial",
    "receita", "lucro_bruto", "ebit", "da", "impairment", "ebitda", "ebitda_pre_ifrs16", "ebitda_ifrs16_equiv",
    "lucro_liquido", "lucro_controladores", "resultado_financeiro", "desp_fin_liquida", "fco", "capex_organico",
    "capex_organico_liquido", "vendas_imob_intang", "aquisicoes_participacoes", "capex_total", "pagamento_arrendamento",
    "fcl_pos_arrend", "dividendos_pagos", "recompra_acoes", "emissao_acoes", "retorno_acionista_caixa", "ir_csll_pagos",
    "juros_pagos", "principal_pago", "servico_divida_pago", "caixa", "divida_bruta_ex_arrend", "passivo_arrendamento",
    "div_liq_com_arrend", "div_liq_sem_arrend", "patrimonio_liquido", "pl_controladores", "ativo_total", "ncav",
    "clientes", "estoques", "fornecedores", "cpv", "margem_bruta", "margem_ebitda", "margem_ebitda_pre_ifrs16",
    "margem_liquida", "roe_pl_final", "capex_org_receita", "capex_org_da", "div_liq_ebitda_ifrs16",
    "div_liq_ebitda_pre_ifrs16"]
# ROIC: calculado no coletor (coletor_eua.derivar) e só lido aqui, se o parquet já tiver as colunas (calcular no app
# estourou a memória do Streamlit gratuito)
COLUNAS_ROIC = ["roic_caixa", "roic_trad"]


def compactar(df: pd.DataFrame) -> pd.DataFrame:
    """float64 -> float32 (valores em milhões e índices: precisão de sobra) para caber no Streamlit gratuito."""
    f64 = df.select_dtypes("float64").columns
    if len(f64):
        df[f64] = df[f64].astype("float32")
    return df


def liberar_memoria() -> None:
    """Devolve ao sistema a memória já liberada pelo Python (sem isso o Streamlit gratuito conta o pico como uso)."""
    gc.collect()
    try:
        ctypes.CDLL("libc.so.6").malloc_trim(0)
    except Exception:
        pass


# texto com poucos valores distintos: guardado como categoria (um valor por categoria, não por linha) — só depois do
# enriquecimento, que preenche vazios com ""; economiza memória em cada filtro que as abas fazem na base
CATEGORIAS = ("setor", "subsetor", "segmento", "periodo", "fonte", "regime_contabil")


def categorizar(df: pd.DataFrame) -> pd.DataFrame:
    for c in CATEGORIAS:
        if c in df.columns and df[c].dtype != "category":
            df[c] = df[c].astype("category")
    return df


def base_como_br() -> pd.DataFrame:
    import pyarrow as pa
    import pyarrow.parquet as pq
    # filtros já na leitura (datas ISO comparam como texto) e junção no Arrow: uma única conversão para pandas,
    # sem cópias intermediárias (pico de memória menor)
    filtros = [("dt_fim", ">=", INICIO.strftime("%Y-%m-%d")), ("tipo_especial", "!=", "financeira")]
    tabs = []
    for a in ARQUIVOS:
        buf = _baixar(a)
        nomes = set(pq.read_schema(io.BytesIO(buf)).names)  # parquet sem as colunas do ROIC: segue sem elas
        tabs.append(pq.read_table(io.BytesIO(buf), columns=COLUNAS + [c for c in COLUNAS_ROIC if c in nomes],
                                  filters=filtros))
        del buf
    t = pa.concat_tables(tabs, promote_options="permissive")
    del tabs
    b = t.to_pandas(self_destruct=True)
    del t
    liberar_memoria()
    b["dt_fim"] = pd.to_datetime(b["dt_fim"], errors="coerce")
    b = compactar(b)
    z = lambda s: s.fillna(0)

    # FCO: na SEC não há separação de juros/dividendos dentro do FCO -> o FCO é o reportado
    b["fco_reportado"] = b["fco"]
    # cobertura da dívida com as MESMAS definições dos rótulos brasileiros
    # (arrend. = pagamentos de arrendamento financeiro; o operacional já está dentro do EBITDA US GAAP)
    ir, arr = z(b["ir_csll_pagos"]), z(b["pagamento_arrendamento"])
    b["ebitda_menos_ir"] = b["ebitda"] - ir
    b["caixa_servico_divida"] = b["ebitda"] - ir - arr
    b["caixa_servico_pos_capex"] = b["caixa_servico_divida"] - z(b["capex_organico"])
    b["ebitda_ir_capex"] = b["ebitda_menos_ir"] - z(b["capex_organico"])
    b["servico_divida_c_arrend"] = b["servico_divida_pago"] + arr
    pos = lambda s: s.where(s > 0)
    b["cob_juros_caixa"] = b["caixa_servico_divida"] / pos(b["juros_pagos"])
    b["dscr_ebitda"] = b["caixa_servico_divida"] / pos(b["servico_divida_pago"])
    b["dscr_pos_capex"] = b["caixa_servico_pos_capex"] / pos(b["servico_divida_pago"])
    b["dscr_capex_arrend_den"] = b["ebitda_ir_capex"] / pos(b["servico_divida_c_arrend"])
    b["dscr_ebitda_arrend_den"] = b["ebitda_menos_ir"] / pos(b["servico_divida_c_arrend"])
    b["cob_ebit_res_fin"] = b["ebit"] / pos(b["desp_fin_liquida"])
    # notas: o app marca como "alerta" textos com 'ausente'; na SEC, tag não reportada não é erro de cálculo
    b["notas"] = b["notas"].fillna("").str.replace("ausente", "não reportado", regex=False)
    for c in ("ticker", "empresa", "setor", "subsetor", "segmento"):
        b[c] = b[c].fillna("").astype(str)
    b["cnpj"] = _cik_de(b["cnpj"]).map(_cnpj)             # 'CIK' + 11 dígitos faz o papel do CNPJ
    b = b[b["exercicio"] >= INICIO.year]                     # exercícios 52/53 semanas encerrados em jan/2014 = 2013
    # trimestral e LTM: data de fechamento alinhada ao trimestre civil mais próximo (a data real fica guardada)
    b["dt_fim_original"] = b["dt_fim"]
    tri = b["tipo"].isin(["trimestral", "ltm"])
    b.loc[tri, "dt_fim"] = trimestre_civil(b.loc[tri, "dt_fim"])
    b = b.drop_duplicates(["cnpj", "tipo", "dt_fim"], keep="last")   # troca de exercício fiscal (raro)
    b = compactar(b.drop(columns=["tipo_especial"]).reset_index(drop=True))
    liberar_memoria()
    return b


def calc_roic(b: pd.DataFrame) -> pd.DataFrame:
    """ROIC da base americana já vem calculado do coletor (colunas roic_caixa e roic_trad): nada a fazer aqui."""
    return b


def multiplos() -> pd.DataFrame:
    """Múltiplos de mercado (coletor-eua/multiplos_eua.py) no formato do multiplos.csv.gz brasileiro."""
    try:
        m = pd.read_parquet(io.BytesIO(_baixar("multiplos_eua.parquet")))
    except Exception:                                        # ainda não gerado: o app segue sem múltiplos
        return pd.DataFrame()
    cad = pd.read_parquet(io.BytesIO(_baixar("cadastro_eua.parquet")), columns=["cik", "setor", "subsetor", "segmento"])
    m = m.merge(cad, on="cik", how="left")                  # classificação: usada na mediana de comparáveis
    for c in ("setor", "subsetor", "segmento"):
        m[c] = m[c].fillna("")
    m["cnpj"] = m["cik"].map(_cnpj)
    m["data"] = pd.to_datetime(m["data"], errors="coerce")
    m = m[m["data"] >= INICIO]
    # mesmos trimestres civis da base (o ponto do último pregão, comum a todas as empresas, fica como está)
    alinhar = (m["tipo"] == "trimestral") & (m["data"] != m["data"].max())
    m.loc[alinhar, "data"] = trimestre_civil(m.loc[alinhar, "data"])
    m = m.drop_duplicates(["cik", "tipo", "data"], keep="last")
    for c in ("ticker", "empresa", "notas", "periodo_base"):
        m[c] = m[c].fillna("").astype(str)
    return m.drop(columns=["cik"]).reset_index(drop=True)


@st.cache_data(ttl=3600, max_entries=1, show_spinner=False)
def _precos_recentes(versao: str) -> pd.DataFrame:
    """Fechamento ajustado por desdobramentos dos ~200 últimos dias (cache do coletor-eua)."""
    try:
        p = pd.read_parquet(io.BytesIO(_baixar("precos_recentes.parquet")))
        sp = pd.read_parquet(io.BytesIO(_baixar("splits_eua.parquet")))
    except Exception:  # noqa: BLE001
        return pd.DataFrame()
    p["data"] = pd.to_datetime(p["data"])
    sp["data"] = pd.to_datetime(sp["data"])
    p = p.sort_values(["ticker", "data"])
    p["aj"] = p["bruto"].astype(float)
    for r in sp[sp["data"] >= p["data"].min()].itertuples():     # preço antes do desdobramento ÷ razão
        if pd.notna(r.razao) and r.razao > 0 and r.razao != 1:
            m = (p["ticker"] == r.ticker) & (p["data"] < r.data)
            p.loc[m, "aj"] = p.loc[m, "aj"] / r.razao
    return p[["ticker", "data", "aj"]]


def vm_diario_recente(mult: pd.DataFrame) -> pd.DataFrame:
    """Série diária aproximada de valor de mercado (últimos ~200 dias) para as altas e baixas: valor de mercado do
    último pregão × preço ajustado da data ÷ preço ajustado do último pregão (cálculo próprio)."""
    if mult is None or mult.empty:
        return pd.DataFrame()
    p = _precos_recentes(versao())
    if p.empty:
        return pd.DataFrame()
    u = mult[mult["tipo"] == "trimestral"].sort_values("data").groupby("cnpj").tail(1)
    u = u[u["valor_mercado"] > 0][["cnpj", "ticker", "valor_mercado"]]
    ult = p.sort_values("data").groupby("ticker").tail(1)[["ticker", "aj"]].rename(columns={"aj": "aj_ult"})
    v = p.merge(u, on="ticker").merge(ult, on="ticker")
    v["valor_mercado"] = v["valor_mercado"] * v["aj"] / v["aj_ult"]
    return v[["data", "cnpj", "valor_mercado"]].dropna()


def cobertura() -> pd.DataFrame:
    """Situação de cada empresa do universo, no formato da aba Cobertura."""
    a = pd.read_csv(io.BytesIO(_baixar("auditoria_eua.csv.gz")), compression="gzip")
    cad = pd.read_parquet(io.BytesIO(_baixar("cadastro_eua.parquet")))
    a = a.merge(cad[["cik", "subsetor", "segmento", "bolsa"]], on="cik", how="left")
    a["cnpj"] = a["cik"].map(_cnpj)
    a["situacao"] = a["motivo"].fillna("").replace("", "com dados")
    cols = ["ticker", "empresa", "setor", "subsetor", "segmento", "bolsa", "cnpj", "situacao", "exercicios",
            "ultimo_dt_fim", "ausentes", "tags"]
    return a[[c for c in cols if c in a.columns]].sort_values(["setor", "subsetor", "empresa"])


# ------------------------------------------------------------------ textos
def sem_acesso(erro: str | None = None):
    st.title("🇺🇸 Universo EUA")
    if erro:
        st.error(erro)
    st.info(
        "Para ler a base americana, o app precisa de um token **só de leitura** do repositório "
        f"`{REPO}`:\n\n"
        "1. GitHub → Settings → Developer settings → Personal access tokens → Fine-grained tokens → "
        "**Generate new token**, *Only select repositories* = `coletor-eua`, *Contents* = **Read-only**.\n"
        "2. Streamlit Cloud → o app → **Settings → Secrets**, acrescente a linha:\n\n"
        "```\nGITHUB_TOKEN_EUA = \"cole-aqui-o-token\"\n```\n"
        "3. Salve; o app reinicia sozinho.")


def metodologia():
    st.markdown("""
**Fontes.** SEC EDGAR: base XBRL *companyfacts* (10-K anual e 10-Q trimestral; só tags padrão US GAAP — tags
próprias das empresas não entram), *submissions* (SIC, escritório responsável, ano fiscal) e lista de tickers.
Universo: empresas listadas em NYSE, Nasdaq, Cboe e TXSE. Fora: emissores estrangeiros (20-F/40-F, IFRS), SPACs e,
nesta versão, bancos e seguradoras (SIC 60–64). REITs entram. Valores em US$ milhões. Coleta no repositório
coletor-eua (carga completa aos domingos; incremental de terça a sábado).

**Classificação.** Setor = escritório da SEC responsável pela empresa; subsetor = grupo SIC de 2 dígitos;
segmento = descrição SIC de 4 dígitos.

**Períodos.** Anual = 10-K (vale o arquivamento mais recente, inclusive 10-K/A e comparativos reapresentados).
Trimestral: 3M reportado; sem ele, diferença dos acumulados (6M − 3M, 9M − 6M); 4T = anual − 9M.
LTM = soma de 4 trimestres contíguos. No trimestral e no LTM, a data de fechamento de cada empresa é alinhada ao fim de
trimestre civil mais próximo (ex.: Apple 27/12 → 31/12; Walmart 31/10 → 30/09), para comparar empresas com calendários
fiscais diferentes — o rótulo do período (ex.: 3T27) segue o exercício da empresa. Exercício = ano civil do encerramento (Walmart: ano fiscal até jan/2026 = 2026;
anos de 52/53 semanas encerrados na 1ª semana de janeiro contam no ano anterior). A data da primeira divulgação de
cada período fica na base do coletor (coluna *dt_arquivamento*) e será usada nos múltiplos históricos.

**Arrendamentos (ASC 842 ≠ IFRS 16).** Nos EUA o aluguel de arrendamentos operacionais continua no opex; só o
arrendamento financeiro sai do EBITDA, como no IFRS 16.
- EBITDA (US GAAP) = EBIT (operating income) + D&A + impairment.
- EBITDA após arrend. financeiros = EBITDA − pagamentos de arrendamento financeiro (principal + juros):
  comparável ao EBITDA pré-IFRS 16 da base brasileira.
- EBITDA equivalente IFRS 16 = EBITDA + custo de arrendamento operacional (quando divulgado; costuma vir só no 10-K).
- Pares consistentes: DL c/ arrendamento (financeiro + operacional) ÷ EBITDA equivalente IFRS 16;
  DL s/ arrendamento ÷ EBITDA após arrend. financeiros. No setor, Σ numerador ÷ Σ denominador do mesmo par.

**FCO.** O US GAAP põe juros pagos, IR pago e pagamentos de arrendamento operacional no FCO: não há o "FCO
comparável" da base brasileira (sem juros e dividendos). A conversão de caixa usa o FCO como reportado ÷ EBITDA US GAAP
e tende a ficar abaixo da brasileira por incluir os juros.

**Cobertura da dívida.** Mesmas fórmulas da base brasileira; *arrend.* = pagamentos de arrendamento financeiro; IR =
imposto de renda pago (divulgação suplementar do fluxo de caixa; muitas empresas só divulgam no 10-K, então os
índices trimestrais/LTM podem ficar vazios).

**ROIC (variante Price Discovery, ajustada ao US GAAP).** Calculado no coletor. NOPAT = EBIT × (1 − 25%) quando o
EBIT é positivo (21% federal + média dos impostos estaduais; sem crédito fiscal sobre prejuízo). Capital no fim do
período; só anual e LTM; vazio com capital ≤ 0. Como o arrendamento operacional fica no opex (ASC 842), ele já está
descontado no EBIT:
- ROIC caixa = (NOPAT + D&A − capex orgânico − pagamentos de arrendamento financeiro) ÷ (DL s/ arrendamento + PL).
  Sem pagamento de arrendamento financeiro divulgado e com passivo financeiro, 2,5 × esse passivo entra no capital.
- ROIC tradicional = NOPAT ajustado ÷ (DL s/ arrendamento + passivos de arrendamento financeiro e operacional + PL).
  NOPAT ajustado soma ao EBIT os juros implícitos do arrendamento operacional (passivo operacional × custo da dívida da
  empresa, juros pagos ÷ dívida bruta, entre 3% e 12%; sem dado, 6%; nunca acima do custo de arrendamento operacional
  divulgado), para o EBIT ficar no critério do IFRS 16 e comparável ao ROIC tradicional brasileiro.
- Sem ajuste de goodwill (igual à base brasileira).

**Dívida e caixa.** Dívida bruta = LongTermDebt (inclui a parcela corrente) + empréstimos de curto prazo +
commercial paper (com alternativas por empresa). Caixa = caixa e equivalentes + aplicações de curto prazo.

**Múltiplos de mercado.** Preço de fechamento do Yahoo Finance (via yfinance), sem o ajuste de desdobramentos
(preço efetivamente negociado na data) × ações em circulação (capa do 10-K/10-Q; empresas com várias classes: total do
balanço), corrigidas pelos desdobramentos entre a data do dado e a do preço. Mais de um ticker listado: ações totais ×
preço do ticker principal (aproximação, nas notas). Anual = preço do fim do exercício com os números do exercício.
Trimestral = ponto no tempo: em cada fim de trimestre fiscal, o último 10-Q/10-K já arquivado até a data (lucro, EBITDA
e receita dos 12 meses desse documento); o último ponto é o pregão mais recente, atualizado de terça a sábado.
EV = valor de mercado + dívida líquida + não controladores. EV/EBITDA equiv. IFRS 16 = EV c/ arrendamentos ÷ EBITDA
equivalente IFRS 16 (sem o dado no LTM: EBITDA + custo de arrendamento operacional do último 10-K); EV/EBITDA após
arrend. financeiros = EV s/ arrendamentos ÷ EBITDA após arrend. financeiros. P/L e P/VPA com lucro e PL dos
controladores; dividend yield = dividendos pagos em 12 meses ÷ valor de mercado.

**Limitações conhecidas.**
- D&A: quando a empresa usa tag própria para o total do fluxo de caixa (ex.: Microsoft), usamos depreciação +
  amortização de intangíveis em tags padrão — o EBITDA fica conservador.
- Prazos médios dependem de clientes, estoques, fornecedores e custo das vendas em tags padrão.
- Preços do Yahoo Finance (fonte não oficial; pode mudar ou limitar o acesso).
- Universo só com empresas listadas hoje: medianas históricas têm viés de sobrevivência.
""")
