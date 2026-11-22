"""
Price Discovery Scanning (app Streamlit)
Lê saidas/universo/ (gerado pelo workflow) e permite filtrar, comparar, plotar e baixar.
"""
import io
import os
import re
from pathlib import Path

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

DADOS = Path(os.environ.get("UNIVERSO_DIR", Path(__file__).resolve().parent.parent / "saidas" / "universo"))

# chave: (rótulo, formato, grupo)   formato: mm = R$ milhões | pct = % | x = múltiplo
METRICAS = {
    "receita": ("Receita líquida", "mm", "Resultado"),
    "lucro_bruto": ("Lucro bruto", "mm", "Resultado"),
    "ebit": ("EBIT", "mm", "Resultado"),
    "da": ("D&A", "mm", "Resultado"),
    "impairment": ("Impairment (perda/reversão ao valor recuperável)", "mm", "Resultado"),
    "ebitda": ("EBITDA (IFRS 16)", "mm", "Resultado"),
    "ebitda_pre_ifrs16": ("EBITDA pré-IFRS 16", "mm", "Resultado"),
    "lucro_liquido": ("Lucro líquido", "mm", "Resultado"),
    "resultado_financeiro": ("Resultado financeiro líquido", "mm", "Resultado"),
    # concessões do modelo do intangível (rodovias, saneamento, portos/aeroportos, ferrovias, hidrovias): a receita
    # de construção (ICPC 01) sai da receita e do CPV, e a margem de construção sai dos lucros (ver Metodologia)
    "receita_construcao": ("Receita de construção (concessão)", "mm", "Concessões (construção)"),
    "custo_construcao": ("Custo de construção (concessão)", "mm", "Concessões (construção)"),
    "margem_construcao": ("Margem de construção retirada dos lucros", "mm", "Concessões (construção)"),
    "receita_reportada": ("Receita líquida reportada (c/ construção)", "mm", "Concessões (construção)"),
    "lucro_bruto_reportado": ("Lucro bruto reportado", "mm", "Concessões (construção)"),
    "ebit_reportado": ("EBIT reportado", "mm", "Concessões (construção)"),
    "ebitda_reportado": ("EBITDA reportado (IFRS 16)", "mm", "Concessões (construção)"),
    "lucro_liquido_reportado": ("Lucro líquido reportado", "mm", "Concessões (construção)"),
    # setor elétrico (eletricas.yaml): EBITDA ajustado ≈ EBITDA regulatório/caixa (ver Metodologia). Transmissão:
    # EBITDA + custo de construção − Δ ativo de contrato; distribuição: − atualização do VNR; geração: − Δ bonificação
    "ebitda_ajustado": ("EBITDA ajustado (elétricas ≈ regulatório)", "mm", "Setor elétrico (ajustado)"),
    "margem_ebitda_ajustado": ("Margem EBITDA ajustado (elétricas)", "pct", "Setor elétrico (ajustado)"),
    "div_liq_ebitda_ajustado": ("DL c/ arrend. / EBITDA ajustado (elétricas)", "x", "Setor elétrico (ajustado)"),
    "dscr_ebitda_ajustado": ("(EBITDA ajustado − IR − arrend.) ÷ (juros + principal)", "x", "Setor elétrico (ajustado)"),
    "ativo_concessao_saldo": ("Ativo de contrato/concessão da transmissão (saldo)", "mm", "Setor elétrico (ajustado)"),
    "var_ativo_concessao": ("Δ ativo de contrato/concessão no período", "mm", "Setor elétrico (ajustado)"),
    "custo_implantacao": ("Custo de implantação da infraestrutura (transmissão)", "mm", "Setor elétrico (ajustado)"),
    "vnr_atualizacao": ("Atualização do ativo financeiro indenizável (VNR)", "mm", "Setor elétrico (ajustado)"),
    "var_bonificacao": ("Δ bonificação pela outorga no período", "mm", "Setor elétrico (ajustado)"),
    "fco": ("FCO comparável (sem juros e dividendos)", "mm", "Caixa e investimento"),
    "fco_reportado": ("FCO reportado (DFC)", "mm", "Caixa e investimento"),
    "fco_juros_no_fco": ("Juros pagos/recebidos dentro do FCO (líq.)", "mm", "Caixa e investimento"),
    "fco_div_no_fco": ("Dividendos/JCP dentro do FCO (líq.)", "mm", "Caixa e investimento"),
    "capex_organico": ("Capex orgânico", "mm", "Caixa e investimento"),
    "capex_organico_liquido": ("Capex orgânico líq. de vendas", "mm", "Caixa e investimento"),
    "vendas_imob_intang": ("Vendas de imobilizado/intangível", "mm", "Caixa e investimento"),
    "aquisicoes_participacoes": ("Aquisições de participações", "mm", "Caixa e investimento"),
    "capex_total": ("Capex total c/ aquisições", "mm", "Caixa e investimento"),
    "pagamento_arrendamento": ("Pagamentos de arrendamento", "mm", "Caixa e investimento"),
    "fcl_pos_arrend": ("FCL pós-arrendamento", "mm", "Caixa e investimento"),
    "dividendos_pagos": ("Dividendos + JCP pagos", "mm", "Caixa e investimento"),
    "ir_csll_pagos": ("IR + CSLL pagos", "mm", "Caixa e investimento"),
    "juros_pagos": ("Juros pagos da dívida (ex-arrend.)", "mm", "Caixa e investimento"),
    "principal_pago": ("Principal amortizado (ex-arrend.)", "mm", "Caixa e investimento"),
    "servico_divida_pago": ("Serviço da dívida pago (juros + principal)", "mm", "Caixa e investimento"),
    "capex_frota_liquido": ("Capex de frota no FCO (líq. do valor contábil vendido)", "mm", "Caixa e investimento"),
    "caixa": ("Caixa + aplicações CP", "mm", "Balanço e dívida"),
    "divida_bruta_ex_arrend": ("Dívida bruta ex-arrendamento", "mm", "Balanço e dívida"),
    "passivo_arrendamento": ("Passivo de arrendamento", "mm", "Balanço e dívida"),
    "div_liq_com_arrend": ("Dívida líquida c/ arrendamento", "mm", "Balanço e dívida"),
    "div_liq_sem_arrend": ("Dívida líquida s/ arrendamento", "mm", "Balanço e dívida"),
    "patrimonio_liquido": ("Patrimônio líquido", "mm", "Balanço e dívida"),
    "ativo_total": ("Ativo total", "mm", "Balanço e dívida"),
    "ncav": ("NCAV (Graham)", "mm", "Balanço e dívida"),
    "margem_bruta": ("Margem bruta", "pct", "Margens e retorno"),
    "margem_ebitda": ("Margem EBITDA (IFRS 16)", "pct", "Margens e retorno"),
    "margem_ebitda_pre_ifrs16": ("Margem EBITDA pré-IFRS 16", "pct", "Margens e retorno"),
    "margem_liquida": ("Margem líquida", "pct", "Margens e retorno"),
    "roe_pl_final": ("ROE (LL / PL final)", "pct", "Margens e retorno"),
    "roic_caixa": ("ROIC caixa (NOPAT + D&A − capex org. − arrend. ÷ CI)", "pct", "Margens e retorno"),
    "roic_trad": ("ROIC tradicional (NOPAT ÷ CI c/ arrendamento)", "pct", "Margens e retorno"),
    "capex_org_receita": ("Capex orgânico / receita", "pct", "Margens e retorno"),
    "conversao_fco_ebitda": ("Conversão de caixa (FCO comparável ÷ EBITDA IFRS 16)", "pct", "Margens e retorno"),
    "capex_org_da": ("Capex orgânico / D&A", "x", "Alavancagem e múltiplos"),
    "giro_ativo": ("Giro do ativo (receita ÷ ativo total)", "x", "Margens e retorno"),
    "alavancagem_ativo_pl": ("Alavancagem (ativo total ÷ PL)", "x", "Alavancagem e múltiplos"),
    "div_liq_ebitda_ifrs16": ("DL c/ arrend. / EBITDA IFRS 16", "x", "Alavancagem e múltiplos"),
    "div_liq_ebitda_pre_ifrs16": ("DL s/ arrend. / EBITDA pré-IFRS 16", "x", "Alavancagem e múltiplos"),
    # cobertura da dívida (DFC; IR = IR + CSLL pagos; arrend. = principal + juros de arrendamento pagos)
    "cob_ebit_res_fin": ("EBIT ÷ resultado financeiro líquido", "x", "Cobertura da dívida"),
    "cob_juros_caixa": ("(EBITDA − IR − arrend.) ÷ juros pagos", "x", "Cobertura da dívida"),
    "dscr_ebitda": ("(EBITDA − IR − arrend.) ÷ (juros + principal)", "x", "Cobertura da dívida"),
    "dscr_pos_capex": ("(EBITDA − IR − capex − arrend.) ÷ (juros + principal)", "x", "Cobertura da dívida"),
    "dscr_capex_arrend_den": ("(EBITDA − IR − capex) ÷ (juros + principal + arrend.)", "x", "Cobertura da dívida"),
    "dscr_ebitda_arrend_den": ("(EBITDA − IR) ÷ (juros + principal + arrend.)", "x", "Cobertura da dívida"),
    # crescimento contra o mesmo período do ano anterior (anual: ano anterior; LTM e trimestral: 4 trimestres antes)
    "cresc_receita": ("Crescimento da receita líquida (a/a)", "pct", "Crescimento"),
    "cresc_lucro_bruto": ("Crescimento do lucro bruto (a/a)", "pct", "Crescimento"),
    "cresc_ebitda": ("Crescimento do EBITDA (a/a)", "pct", "Crescimento"),
    "cresc_ebit": ("Crescimento do EBIT (a/a)", "pct", "Crescimento"),
    "cresc_lucro_liquido": ("Crescimento do lucro líquido (a/a)", "pct", "Crescimento"),
    "cresc_ativo_total": ("Crescimento do ativo total (a/a)", "pct", "Crescimento"),
    "cresc_patrimonio_liquido": ("Crescimento do patrimônio líquido (a/a)", "pct", "Crescimento"),
    "cresc_capex_organico": ("Crescimento do capex orgânico (a/a)", "pct", "Crescimento"),
    # graus de alavancagem (a partir das variações a/a): GAO = Δ% EBIT ÷ Δ% receita; GAF = Δ% lucro ÷ Δ% EBIT;
    # GAT = Δ% lucro ÷ Δ% receita (= GAO × GAF)
    "gao": ("Grau de alavancagem operacional (GAO)", "x", "Graus de alavancagem"),
    "gaf": ("Grau de alavancagem financeira (GAF)", "x", "Graus de alavancagem"),
    "gat": ("Grau de alavancagem total (GAT)", "x", "Graus de alavancagem"),
    # prazos médios (dias) e ciclos — capital de giro operacional
    "pmr": ("Prazo médio de recebimento", "d", "Prazos médios"),
    "pme": ("Prazo médio de estocagem/venda", "d", "Prazos médios"),
    "pmp": ("Prazo médio de pagamento", "d", "Prazos médios"),
    "ciclo_operacional": ("Ciclo operacional (PME + PMR)", "d", "Prazos médios"),
    "ciclo_financeiro": ("Ciclo financeiro (ciclo operacional − PMP)", "d", "Prazos médios"),
    # mercado (preço da B3 × ações; ver Metodologia)
    "valor_mercado": ("Valor de mercado", "mm", "Mercado"),
    "ev_com_arrend": ("EV c/ arrendamento", "mm", "Mercado"),
    "ev_sem_arrend": ("EV s/ arrendamento", "mm", "Mercado"),
    "p_vpa": ("P/VPA", "x", "Mercado"),
    "p_l": ("P/L", "x", "Mercado"),
    "ev_ebitda_ifrs16": ("EV/EBITDA IFRS 16 (c/ arrend.)", "x", "Mercado"),
    "ev_ebitda_pre_ifrs16": ("EV/EBITDA pré-IFRS 16 (s/ arrend.)", "x", "Mercado"),
    "ev_receita_com_arrend": ("EV/Receita líquida (EV c/ arrend.)", "x", "Mercado"),
    "ev_receita_sem_arrend": ("EV/Receita líquida (EV s/ arrend.)", "x", "Mercado"),
    "dividend_yield": ("Dividend yield", "pct", "Mercado"),
}
ROT = {k: v[0] for k, v in METRICAS.items()}
# razões agregadas do setor = soma do numerador / soma do denominador (só empresas com os dois)
AGREG = {"margem_bruta": ("lucro_bruto", "receita"), "margem_ebitda": ("ebitda", "receita"),
         "margem_ebitda_pre_ifrs16": ("ebitda_pre_ifrs16", "receita"), "margem_liquida": ("lucro_liquido", "receita"),
         "roe_pl_final": ("lucro_liquido", "patrimonio_liquido"), "capex_org_receita": ("capex_organico", "receita"),
         "capex_org_da": ("capex_organico", "da"), "div_liq_ebitda_ifrs16": ("div_liq_com_arrend", "ebitda"),
         "div_liq_ebitda_pre_ifrs16": ("div_liq_sem_arrend", "ebitda_pre_ifrs16"),
         "giro_ativo": ("receita", "ativo_total"), "alavancagem_ativo_pl": ("ativo_total", "patrimonio_liquido"),
         "conversao_fco_ebitda": ("fco", "ebitda"),
         "margem_ebitda_ajustado": ("ebitda_ajustado", "receita"),
         "div_liq_ebitda_ajustado": ("div_liq_com_arrend", "ebitda_ajustado"),
         "dscr_ebitda_ajustado": ("caixa_servico_ajustado", "servico_divida_pago"),
         # cobertura da dívida do setor = Σ geração de caixa ÷ Σ obrigação (só empresas com denominador > 0)
         "cob_ebit_res_fin": ("ebit", "desp_fin_liquida"), "cob_juros_caixa": ("caixa_servico_divida", "juros_pagos"),
         "dscr_ebitda": ("caixa_servico_divida", "servico_divida_pago"),
         "dscr_pos_capex": ("caixa_servico_pos_capex", "servico_divida_pago"),
         "dscr_capex_arrend_den": ("ebitda_ir_capex", "servico_divida_c_arrend"),
         "dscr_ebitda_arrend_den": ("ebitda_menos_ir", "servico_divida_c_arrend"),
         # múltiplos do setor = Σ valor de mercado (ou EV) ÷ Σ fundamento, na mesma data
         "p_vpa": ("valor_mercado", "m_pl_controladores"), "p_l": ("valor_mercado", "m_lucro_12m"),
         "ev_ebitda_ifrs16": ("ev_com_arrend", "m_ebitda_12m"), "ev_ebitda_pre_ifrs16": ("ev_sem_arrend", "m_ebitda_pre_12m"),
         "ev_receita_com_arrend": ("ev_com_arrend", "m_receita_12m"), "ev_receita_sem_arrend": ("ev_sem_arrend", "m_receita_12m"),
         "dividend_yield": ("m_dividendos_12m", "valor_mercado")}
# bases dos índices de cobertura (não aparecem como métrica; usadas no agregado do setor)
AUX_COB = ["desp_fin_liquida", "ebitda_menos_ir", "caixa_servico_divida", "caixa_servico_pos_capex", "ebitda_ir_capex",
           "servico_divida_c_arrend", "caixa_servico_ajustado"]
AGREG_DEN_POS = {"cob_ebit_res_fin", "cob_juros_caixa", "dscr_ebitda", "dscr_pos_capex", "dscr_capex_arrend_den",
                 "dscr_ebitda_arrend_den", "dscr_ebitda_ajustado", "div_liq_ebitda_ajustado"}
PADRAO = ["receita", "ebitda", "margem_ebitda", "lucro_liquido", "div_liq_ebitda_ifrs16", "capex_organico"]
TIPOS = {"Anual": "anual", "LTM (últimos 12 meses)": "ltm", "Trimestral": "trimestral"}

# Bancos e seguradoras: (rótulo, formato, agregação do setor)
# agregação: "soma" | (numerador, denominador) para razão das somas | None (só mediana)
FIN = {
    "lucro_controladores": ("Lucro líquido (controladores)", "mm", "soma"),
    "lucro_antes_ir": ("Lucro antes de IR/CS", "mm", "soma"),
    "pl_controladores": ("PL (controladores)", "mm", "soma"),
    "ativo_total": ("Ativo total", "mm", "soma"),
    "receita_total": ("Receita total (linha 3.01)", "mm", "soma"),
    "roe": ("ROE", "pct", ("lucro_controladores", "pl_controladores")),
    "roa": ("ROA", "pct", ("lucro_liquido", "ativo_total")),
    "pl_ativo": ("PL / ativo", "pct", ("patrimonio_liquido", "ativo_total")),
    "margem_liquida": ("Margem líquida", "pct", ("lucro_liquido", "receita_total")),
    "resultado_bruto_intermediacao": ("Resultado bruto da intermediação", "mm", "soma"),
    "margem_financeira_antes_pdd": ("Margem financeira antes da PDD", "mm", "soma"),
    "despesa_pdd": ("Despesa de PDD", "mm", "soma"),
    "receita_servicos": ("Receitas de serviços e tarifas", "mm", "soma"),
    "despesas_pessoal_adm": ("Despesas de pessoal e administrativas", "mm", "soma"),
    "carteira_credito": ("Carteira de crédito (aprox.)", "mm", "soma"),
    "carteira_credito_bruta": ("Carteira de crédito bruta", "mm", "soma"),
    "provisao_carteira": ("Provisão para perdas de crédito (saldo)", "mm", "soma"),
    "tvm": ("Títulos e valores mobiliários e derivativos", "mm", "soma"),
    "demais_ativos": ("Demais ativos", "mm", "soma"),
    "depositos": ("Depósitos", "mm", "soma"),
    "custo_credito": ("Custo do crédito (PDD / carteira bruta)", "pct", ("despesa_pdd", "carteira_credito_bruta")),
    "credito_depositos": ("Crédito / depósitos", "x", ("carteira_credito", "depositos")),
    "carteira_bruta_ativo": ("Carteira bruta / ativo", "pct", ("carteira_credito_bruta", "ativo_total")),
    "tvm_ativo": ("TVM e derivativos / ativo", "pct", ("tvm", "ativo_total")),
    "demais_ativos_ativo": ("Demais ativos / ativo", "pct", ("demais_ativos", "ativo_total")),
    "cobertura_carteira": ("Provisão / carteira bruta", "pct", ("provisao_carteira", "carteira_credito_bruta")),
    "indice_eficiencia": ("Índice de eficiência (aprox.)", "pct", None),
    "dividendos_pagos": ("Dividendos + JCP pagos", "mm", "soma"),
    "payout": ("Payout (dividendos pagos ÷ lucro)", "pct", ("dividendos_pagos", "lucro_controladores")),
    "valor_mercado": ("Valor de mercado", "mm", "soma"),
    "p_l": ("P/L", "x", ("valor_mercado", "lucro_controladores")),
    # crescimento contra o exercício anterior; o gráfico mostra o crescimento acumulado (base 100)
    "cresc_pl_controladores": ("Crescimento do PL (controladores) (a/a)", "pct", "cresc"),
    "cresc_carteira_credito": ("Crescimento da carteira de crédito (a/a)", "pct", "cresc"),
    "cresc_ativo_total": ("Crescimento do ativo total (a/a)", "pct", "cresc"),
    "cresc_pdd": ("Crescimento da despesa de PDD (a/a)", "pct", "cresc"),
}
IDENT = ["empresa", "ticker", "setor", "subsetor", "segmento", "cnpj", "tipo", "periodo", "exercicio",
         "trimestre", "dt_fim", "fonte", "regime_contabil"]

st.set_page_config(page_title="Price Discovery Scanning " + ("USA" if globals().get("MERCADO_FIXO") == "EUA" else "BR"),
                   page_icon="📊", layout="wide")

# ------------------------------------------------------------------ mercado: Brasil / EUA
# No modo EUA o app carrega a base americana (coletor-eua, dados da SEC; módulo app/eua.py) no lugar da brasileira
# e roda as MESMAS abas. Abas e seções que dependem de dados só brasileiros (altas e baixas, calendário da CVM,
# bancos e seguradoras, debêntures, aluguel de ações) ficam ocultas. Juros e Macro têm versão própria dos EUA
# (Treasuries, CPI, PIB e dólar via FRED; módulo app/eua_macro_aba.py).
# Dois apps (nov/2026): Price Discovery Scanning BR (entrada app/app.py) e Price Discovery Scanning USA (entrada
# app/app_eua.py, que roda este mesmo arquivo com MERCADO_FIXO = "EUA"). Cada um carrega só a sua base.
MERCADO = globals().get("MERCADO_FIXO") or os.environ.get("PDS_MERCADO", "Brasil")
EUA = MERCADO == "EUA"
URL_APP = {"Brasil": os.environ.get("PDS_URL_BR", "https://pricediscovery-scanning-br.streamlit.app"),
           "EUA": os.environ.get("PDS_URL_EUA", "https://pricediscovery-scanning-usa.streamlit.app")}
st.sidebar.markdown(f"### {'🇺🇸 Price Discovery Scanning USA' if EUA else '🇧🇷 Price Discovery Scanning BR'}")
st.sidebar.link_button("Ir para o app dos EUA 🇺🇸" if not EUA else "Ir para o app do Brasil 🇧🇷",
                       URL_APP["Brasil" if EUA else "EUA"], width="stretch")
MOEDA = "US$" if EUA else "R$"
UN = f"{MOEDA} mm"
if EUA:
    import eua
    eua.ajustar_metricas(METRICAS, AGREG)
    ROT = {k: v[0] for k, v in METRICAS.items()}


# ------------------------------------------------------------------ dados
def macro_mod():
    """Módulo da aba Macro, recarregado quando o arquivo muda (o Streamlit Cloud mantém módulos importados na
    memória entre deploys; sem isso, uma versão nova de macro_aba.py só entraria após reiniciar o app)."""
    import importlib
    import macro_aba
    mt = os.path.getmtime(macro_aba.__file__)
    if getattr(macro_aba, "_mtime_carregado", None) != mt:
        macro_aba = importlib.reload(macro_aba)
        macro_aba._mtime_carregado = mt
    return macro_aba


def eua_macro_mod():
    """Módulo das abas Juros e Macro do modo EUA, recarregado quando o arquivo muda (mesma razão de macro_mod)."""
    import importlib
    import eua_macro_aba
    mt = os.path.getmtime(eua_macro_aba.__file__)
    if getattr(eua_macro_aba, "_mtime_carregado", None) != mt:
        eua_macro_aba = importlib.reload(eua_macro_aba)
        eua_macro_aba._mtime_carregado = mt
    return eua_macro_aba


def mercado_mod():
    """Módulo das seções de cotação, mapa de calor, estrutura societária e modelo de negócio (mesma razão de macro_mod)."""
    import importlib
    import mercado_aba
    mt = os.path.getmtime(mercado_aba.__file__)
    if getattr(mercado_aba, "_mtime_carregado", None) != mt:
        mercado_aba = importlib.reload(mercado_aba)
        mercado_aba._mtime_carregado = mt
    return mercado_aba


def versao_arquivos(*arqs) -> tuple:
    """Data de modificação dos arquivos: entra como argumento das funções em cache, então o app relê os dados
    sozinho quando um workflow grava uma versão nova (sem depender de reiniciar o app)."""
    return tuple(a.stat().st_mtime if a.exists() else 0 for a in arqs)


@st.cache_data(max_entries=1, show_spinner="Carregando a base…")
def carregar(versao: tuple = ()):
    base = pd.read_csv(DADOS / "base.csv.gz", low_memory=False)
    base["dt_fim"] = pd.to_datetime(base["dt_fim"], errors="coerce")
    base["cnpj"] = base["cnpj"].astype(str).str.zfill(14)
    base = calc_roic(enriquecer(base))
    cob = pd.read_csv(DADOS / "cobertura.csv") if (DADOS / "cobertura.csv").exists() else pd.DataFrame()
    meta = pd.read_json(DADOS / "metadados.json", typ="series") if (DADOS / "metadados.json").exists() else None
    return _carregar_resto(base, cob, meta)


# ROIC (variante Price Discovery, nov/2026). Esta função é a da base brasileira (IFRS 16); a americana usa
# eua.calc_roic (US GAAP / ASC 842: arrend. operacional no opex, alíquota de 25%). Sem ajuste de goodwill.
#   NOPAT = EBIT × (1 − 34%) se EBIT > 0 (sem crédito fiscal sobre prejuízo)
#   ROIC caixa = (NOPAT + D&A − capex orgânico − pagamento de arrendamento) ÷ (DL s/ arrend. + PL)
#     sem pagamento identificado na DFC e com passivo de arrendamento: pagamento sai do numerador e
#     2,5 × passivo de arrendamento entra no capital (roic_arrend_capitalizado = True)
#   ROIC tradicional = NOPAT ÷ (DL s/ arrend. + passivo de arrendamento + PL)
#   Capital no fim do período; vazio no trimestral (fluxo de 1 trimestre) e com capital ≤ 0.
ALIQ_IR_BR = 0.34
MULT_ARREND_FALLBACK = 2.5


def calc_roic(base: pd.DataFrame) -> pd.DataFrame:
    """ROIC caixa e tradicional. Coluna ausente ou qualquer erro -> métricas vazias (o app não cai)."""
    try:
        def num(c):
            return pd.to_numeric(base[c], errors="coerce")
        ebit, da, capex = num("ebit"), num("da"), num("capex_organico")
        pag, passivo = num("pagamento_arrendamento"), num("passivo_arrendamento").fillna(0)
        pl, dl = num("patrimonio_liquido"), num("div_liq_sem_arrend")
        nopat = ebit.where(ebit <= 0, ebit * (1 - ALIQ_IR_BR))
        capitaliza = pag.isna() & (passivo > 0)
        num_caixa = nopat + da - capex - pag.fillna(0)
        ci_caixa = dl + pl + (MULT_ARREND_FALLBACK * passivo).where(capitaliza, 0)
        ci_trad = dl + passivo + pl
        fluxo_ok = base["tipo"].astype(str) != "trimestral"
        base["roic_caixa"] = (num_caixa / ci_caixa.where(ci_caixa > 0)).where(fluxo_ok)
        base["roic_trad"] = (nopat / ci_trad.where(ci_trad > 0)).where(fluxo_ok)
        base["roic_arrend_capitalizado"] = capitaliza & fluxo_ok
    except Exception:
        for c in ("roic_caixa", "roic_trad", "roic_arrend_capitalizado"):
            base[c] = np.nan
    return base


def enriquecer(base: pd.DataFrame) -> pd.DataFrame:
    """Colunas derivadas comuns às bases brasileira e americana (DuPont, prazos, crescimento, graus, conversão)."""
    for c in ("ticker", "setor", "subsetor", "segmento", "empresa", "notas"):
        base[c] = base[c].fillna("")
    base["rotulo"] = base["ticker"] + " — " + base["empresa"]
    base["alerta"] = base["notas"].str.contains("implausível|ausente|alerta", regex=True)
    # DuPont: ROE (LL ÷ PL final) = margem líquida × giro do ativo × alavancagem (ativo ÷ PL)
    base["giro_ativo"] = base["receita"] / base["ativo_total"].where(base["ativo_total"] > 0)
    base["alavancagem_ativo_pl"] = base["ativo_total"] / base["patrimonio_liquido"].where(base["patrimonio_liquido"] > 0)
    base = calc_prazos(base)
    base = calc_crescimento(base)
    # graus de alavancagem: vazios quando o denominador varia menos de 1% (o índice explode sem significado)
    def _grau(a, b):
        return a / b.where(b.abs() >= 0.01)
    base["gao"] = _grau(base["cresc_ebit"], base["cresc_receita"])
    base["gaf"] = _grau(base["cresc_lucro_liquido"], base["cresc_ebit"])
    base["gat"] = _grau(base["cresc_lucro_liquido"], base["cresc_receita"])
    # FCO comparável (coletor): garante as colunas novas numa base gerada antes da mudança
    for c in ("fco_reportado", "fco_juros_no_fco", "fco_div_no_fco"):
        if c not in base.columns:
            base[c] = np.nan
    # conversão do EBITDA em caixa: FCO comparável (sem juros e dividendos) ÷ EBITDA IFRS 16. Sob IFRS 16 o
    # arrendamento pago sai do FCO, então FCO e EBITDA IFRS 16 estão no mesmo critério. Vazio com EBITDA ≤ 0.
    base["conversao_fco_ebitda"] = base["fco"] / base["ebitda"].where(base["ebitda"] > 0)
    return base


def garantir_colunas(base: pd.DataFrame) -> pd.DataFrame:
    """Colunas esperadas pelas abas que a base não trouxe (ex.: mercado sem preços) entram vazias."""
    faltam = [k for k in METRICAS if k not in base.columns] + list(
        {"m_pl_controladores", "m_lucro_12m", "m_ebitda_12m", "m_ebitda_pre_12m", "m_dividendos_12m",
         "m_receita_12m"} - set(base.columns)) + [c for c in AUX_COB if c not in base.columns]
    for c in faltam:
        base[c] = np.nan
    return base


def preparar_mult(mult: pd.DataFrame) -> pd.DataFrame:
    """Tipos, colunas ausentes e ROE no critério dos múltiplos (bases brasileira e americana)."""
    mult["cnpj"] = mult["cnpj"].str.zfill(14)
    mult["data"] = pd.to_datetime(mult["data"], errors="coerce")
    mult["notas"] = mult["notas"].fillna("")
    mult["rotulo"] = mult["ticker"].fillna("").astype(str) + " — " + mult["empresa"].fillna("").astype(str)
    # colunas que podem faltar em bases geradas por versões anteriores do coletor
    for c in ("dividend_yield", "dividendos_12m", "ev_ebitda_ifrs16", "ev_ebitda_pre_ifrs16", "ev_com_arrend",
              "ev_sem_arrend", "pl_controladores", "lucro_controladores_12m", "p_vpa", "p_l", "ebitda_12m",
              "ebitda_pre_ifrs16_12m", "valor_mercado", "receita_12m", "ev_receita_com_arrend",
              "ev_receita_sem_arrend"):
        if c not in mult.columns:
            mult[c] = np.nan
    # ROE no mesmo critério dos múltiplos: lucro 12m ÷ PL dos controladores do mesmo documento
    pl_ = mult["pl_controladores"].where(mult["pl_controladores"] > 0)
    mult["roe"] = mult["lucro_controladores_12m"] / pl_
    return mult


def juntar_mult(base: pd.DataFrame, mult: pd.DataFrame, leve: bool = False) -> pd.DataFrame:
    """Leva os múltiplos para a base (visão do setor, ranking, dados) na data de fim de cada período.
    leve=True (base americana, ~10x maior): preenche as colunas na própria base, sem as cópias do concat; exige a
    base ordenada como anual, trimestral, LTM (é a ordem em que ela é montada), o mesmo resultado do modo normal."""
    # leva os múltiplos para a base (visão do setor, ranking, dados): anual na data de fim do exercício;
    # trimestral e LTM na data de fim do trimestre (ponto no tempo: documentos já divulgados até ali)
    ren = {"pl_controladores": "m_pl_controladores", "lucro_controladores_12m": "m_lucro_12m",
           "ebitda_12m": "m_ebitda_12m", "ebitda_pre_ifrs16_12m": "m_ebitda_pre_12m",
           "dividendos_12m": "m_dividendos_12m", "receita_12m": "m_receita_12m"}
    cm = ["valor_mercado", "ev_com_arrend", "ev_sem_arrend", "p_vpa", "p_l", "ev_ebitda_ifrs16",
          "ev_ebitda_pre_ifrs16", "ev_receita_com_arrend", "ev_receita_sem_arrend", "dividend_yield"] + list(ren)
    mm = mult[["cnpj", "tipo", "data"] + cm].rename(columns=ren).drop_duplicates(["cnpj", "tipo", "data"])
    if leve:
        chave = pd.DataFrame({"cnpj": base["cnpj"].to_numpy(), "data": base["dt_fim"].to_numpy(),
                              "tipo": np.where(base["tipo"].to_numpy() == "anual", "anual", "trimestral")})
        j = chave.merge(mm, on=["cnpj", "tipo", "data"], how="left")
        for c in [c for c in mm.columns if c not in ("cnpj", "tipo", "data")]:
            base[c] = j[c].to_numpy()
        return base
    partes = []
    for tipos, tm in ((["anual"], "anual"), (["trimestral", "ltm"], "trimestral")):
        b_ = base[base["tipo"].isin(tipos)]
        m_ = mm[mm["tipo"] == tm].drop(columns="tipo")
        partes.append(b_.merge(m_, left_on=["cnpj", "dt_fim"], right_on=["cnpj", "data"], how="left")
                      .drop(columns="data"))
    base = pd.concat(partes, ignore_index=True)
    return base


def _carregar_resto(base, cob, meta):
    fin = pd.read_csv(DADOS / "financeiras.csv.gz", dtype={"cnpj": str}) if (DADOS / "financeiras.csv.gz").exists() else pd.DataFrame()
    if not fin.empty:
        for k in FIN:  # métricas novas ausentes em bases geradas por versões anteriores do coletor
            if k not in fin.columns:
                fin[k] = np.nan
        for c in ("ticker", "empresa", "segmento"):
            fin[c] = fin[c].fillna("")
        fin["rotulo"] = fin["ticker"] + " — " + fin["empresa"]
        # anual (DFP) segue em fin; trimestral e LTM (ITR) vão para fin_per. Bases antigas não têm tipo: só anual
        if "tipo" not in fin.columns:
            fin["tipo"] = "anual"
        fin["cnpj"] = fin["cnpj"].astype(str).str.zfill(14)
        fin_per = fin[fin["tipo"].isin(["trimestral", "ltm"])].copy()
        fin_per["dt_fim"] = pd.to_datetime(fin_per["dt_fim"], errors="coerce")
        fin_per = calc_cresc_fin_per(fin_per)
        fin = fin[fin["tipo"] == "anual"].copy()
    else:
        fin_per = pd.DataFrame()
    mult = pd.read_csv(DADOS / "multiplos.csv.gz", dtype={"cnpj": str}) if (DADOS / "multiplos.csv.gz").exists() else pd.DataFrame()
    if not mult.empty:
        mult = preparar_mult(mult)
        base = juntar_mult(base, mult)
        if not fin.empty:
            fin["_dt"] = pd.to_datetime(fin["dt_fim"], errors="coerce")
            fin["cnpj"] = fin["cnpj"].astype(str).str.zfill(14)
            mf = mult[mult["tipo"] == "anual"][["cnpj", "data", "valor_mercado", "p_l"]].drop_duplicates(["cnpj", "data"])
            fin = fin.drop(columns=[c for c in ("valor_mercado", "p_l") if c in fin.columns]) \
                     .merge(mf, left_on=["cnpj", "_dt"], right_on=["cnpj", "data"], how="left").drop(columns=["data", "_dt"])
    base = garantir_colunas(base)
    if not fin.empty:
        for k in FIN:
            if k not in fin.columns:
                fin[k] = np.nan
        fin = calc_cresc_fin(fin)
    return base, cob, meta, fin, mult, fin_per


@st.cache_data(max_entries=1, show_spinner=False)
def carregar_aluguel(versao: tuple = ()) -> pd.DataFrame:
    arq = DADOS.parent / "aluguel" / "resumo.csv.gz"
    if not arq.exists():
        return pd.DataFrame()
    a = pd.read_csv(arq)
    a["data"] = pd.to_datetime(a["data"], errors="coerce")
    a["ticker"] = a["ticker"].astype(str)
    return a


def calc_prazos(b: pd.DataFrame) -> pd.DataFrame:
    """Prazos médios com saldos médios (inicial e final). Dias: 365 no anual e LTM; 90 no trimestral.
    PMR = dias ÷ (receita ÷ clientes médio); PME = dias ÷ (CPV ÷ estoque médio);
    PMP = dias ÷ (compras ÷ fornecedores médio), com compras = estoque final − estoque inicial + CPV.
    Saldo inicial: anual = fim do exercício anterior; LTM = 4 trimestres antes; trimestral = trimestre anterior."""
    precisa = {"clientes", "estoques", "fornecedores", "cpv", "receita"}
    if not precisa <= set(b.columns):
        for k in ("pmr", "pme", "pmp", "ciclo_operacional", "ciclo_financeiro"):
            b[k] = np.nan
        return b
    partes = []
    for tp, dias, desloc in (("anual", 365, pd.DateOffset(years=1)), ("ltm", 365, pd.DateOffset(years=1)),
                             ("trimestral", 90, pd.DateOffset(months=3))):
        x = b[b["tipo"] == tp].sort_values("dt_fim")
        if x.empty:
            continue
        ant = x[["cnpj", "dt_fim", "clientes", "estoques", "fornecedores"]].dropna(subset=["dt_fim"]).copy()
        ant["dt_fim"] = ant["dt_fim"] + desloc
        x = pd.merge_asof(x.dropna(subset=["dt_fim"]), ant.sort_values("dt_fim"), on="dt_fim", by="cnpj",
                          suffixes=("", "_ini"), tolerance=pd.Timedelta(days=20), direction="nearest")
        cli = (x["clientes"] + x["clientes_ini"]) / 2
        est = (x["estoques"] + x["estoques_ini"]) / 2
        forn = (x["fornecedores"] + x["fornecedores_ini"]) / 2
        compras = x["estoques"] - x["estoques_ini"] + x["cpv"]
        x["pmr"] = dias * cli / x["receita"].where(x["receita"] > 0)
        x["pme"] = dias * est / x["cpv"].where(x["cpv"] > 0)
        x["pmp"] = dias * forn / compras.where(compras > 0)
        x["ciclo_operacional"] = x["pme"] + x["pmr"]
        x["ciclo_financeiro"] = x["ciclo_operacional"] - x["pmp"]
        partes.append(x.drop(columns=["clientes_ini", "estoques_ini", "fornecedores_ini"]))
    resto = b[~b["tipo"].isin(["anual", "ltm", "trimestral"])]
    return pd.concat(partes + [resto], ignore_index=True)


@st.cache_data(max_entries=1, show_spinner=False)
def carregar_juros(versao: tuple = ()) -> tuple[pd.DataFrame, pd.DataFrame]:
    pasta = DADOS.parent / "juros"
    p = pd.read_csv(pasta / "parametros.csv", parse_dates=["data"]) if (pasta / "parametros.csv").exists() else pd.DataFrame()
    v = pd.read_csv(pasta / "vertices.csv.gz", parse_dates=["data"]) if (pasta / "vertices.csv.gz").exists() else pd.DataFrame()
    return p, v


def svensson(t, b1, b2, b3, b4, l1, l2):
    """Taxa (fração a.a., base 252) da ETTJ ANBIMA no prazo t (anos = dias úteis ÷ 252)."""
    t = np.asarray(t, dtype=float)
    a = (1 - np.exp(-l1 * t)) / (l1 * t)
    c = (1 - np.exp(-l2 * t)) / (l2 * t)
    return b1 + b2 * a + b3 * (a - np.exp(-l1 * t)) + b4 * (c - np.exp(-l2 * t))


def curva_juros(pj: pd.DataFrame, d, tipo_c: str, anos) -> np.ndarray:
    """Curva na data d: 'pre', 'ipca' ou 'implicita' = (1 + pre) / (1 + real) − 1."""
    linha = {c: r for c, r in pj[pj["data"] == d].set_index("curva").iterrows()}
    def taxa(nome):
        if nome not in linha:
            return np.full(len(anos), np.nan)
        r = linha[nome]
        return svensson(anos, r["b1"], r["b2"], r["b3"], r["b4"], r["l1"], r["l2"])
    if tipo_c == "pre":
        return taxa("PREFIXADOS")
    if tipo_c == "ipca":
        return taxa("IPCA")
    return (1 + taxa("PREFIXADOS")) / (1 + taxa("IPCA")) - 1


CRESC_BASE = ["receita", "lucro_bruto", "ebitda", "ebit", "lucro_liquido", "ativo_total", "patrimonio_liquido",
              "capex_organico"]


def calc_crescimento(b: pd.DataFrame) -> pd.DataFrame:
    """Crescimento a/a de cada linha absoluta, guardando também o valor do ano anterior (<col>_ant), usado no
    agregado do setor (Σ atual ÷ Σ anterior − 1, só com empresas que têm os dois valores e base positiva).
    Vazio quando a base anterior é ≤ 0."""
    partes = []
    for tp in ("anual", "ltm", "trimestral"):
        x = b[b["tipo"] == tp].dropna(subset=["dt_fim"]).sort_values("dt_fim")
        if x.empty:
            continue
        cols = [c for c in CRESC_BASE if c in x.columns]
        ant = x[["cnpj", "dt_fim"] + cols].copy()
        ant["dt_fim"] = ant["dt_fim"] + pd.DateOffset(years=1)
        x = pd.merge_asof(x, ant.sort_values("dt_fim"), on="dt_fim", by="cnpj", suffixes=("", "_ant"),
                          tolerance=pd.Timedelta(days=20), direction="nearest")
        for c in cols:
            x["cresc_" + c] = np.where(x[c + "_ant"] > 0, x[c] / x[c + "_ant"].where(x[c + "_ant"] > 0) - 1, np.nan)
        partes.append(x)
    resto = b[~b["tipo"].isin(["anual", "ltm", "trimestral"])]
    return pd.concat(partes + [resto], ignore_index=True)


# crescimento a/a dos bancos e seguradoras: indicador -> nível usado no agregado e no índice acumulado
CRESC_FIN = {"cresc_pl_controladores": "pl_controladores", "cresc_carteira_credito": "carteira_credito",
             "cresc_ativo_total": "ativo_total", "cresc_pdd": "nivel_pdd"}
ROT_NIVEL_FIN = {"pl_controladores": "PL (controladores)", "carteira_credito": "Carteira de crédito (aprox.)",
                 "ativo_total": "Ativo total", "nivel_pdd": "Despesa de PDD"}


def calc_cresc_fin(f: pd.DataFrame) -> pd.DataFrame:
    """Crescimento contra o exercício anterior da mesma instituição (exercício − 1), guardando o valor anterior
    (<col>_ant) para o agregado do grupo (Σ atual ÷ Σ anterior − 1, amostra casada). PDD como despesa em módulo
    (nivel_pdd = −despesa_pdd; ano com reversão líquida fica vazio). Vazio sem o ano anterior ou com base ≤ 0."""
    f = f.copy()
    f["exercicio"] = pd.to_numeric(f["exercicio"], errors="coerce")
    pdd = -pd.to_numeric(f["despesa_pdd"], errors="coerce")
    f["nivel_pdd"] = pdd.where(pdd > 0)
    niveis = list(CRESC_FIN.values())
    ant = f[["cnpj", "exercicio"] + niveis].dropna(subset=["exercicio"]).drop_duplicates(["cnpj", "exercicio"])
    ant = ant.assign(exercicio=ant["exercicio"] + 1).rename(columns={c: c + "_ant" for c in niveis})
    f = f.drop(columns=[c + "_ant" for c in niveis if c + "_ant" in f.columns]).merge(
        ant, on=["cnpj", "exercicio"], how="left")
    for k, c in CRESC_FIN.items():
        a = f[c + "_ant"].where(f[c + "_ant"] > 0)
        f[k] = f[c] / a - 1
    return f


def calc_cresc_fin_per(f: pd.DataFrame) -> pd.DataFrame:
    """Trimestral e LTM dos bancos e seguradoras: crescimento contra o mesmo período do ano anterior (data de fim
    − 12 meses), com a mesma lógica do anual (<col>_ant para o agregado; vazio sem o período anterior ou base ≤ 0)."""
    if f.empty:
        return f
    f = f.copy()
    pdd = -pd.to_numeric(f["despesa_pdd"], errors="coerce")
    f["nivel_pdd"] = pdd.where(pdd > 0)
    niveis = list(CRESC_FIN.values())
    ant = f[["cnpj", "tipo", "dt_fim"] + niveis].dropna(subset=["dt_fim"]).drop_duplicates(["cnpj", "tipo", "dt_fim"])
    ant = ant.assign(dt_fim=ant["dt_fim"] + pd.DateOffset(years=1)).rename(columns={c: c + "_ant" for c in niveis})
    f = f.drop(columns=[c + "_ant" for c in niveis if c + "_ant" in f.columns]).merge(
        ant, on=["cnpj", "tipo", "dt_fim"], how="left")
    for k, c in CRESC_FIN.items():
        a = f[c + "_ant"].where(f[c + "_ant"] > 0)
        f[k] = f[c] / a - 1
    return f


def indice_acumulado(niveis: pd.DataFrame) -> pd.Series:
    """Índice base 100 encadeando a variação entre períodos consecutivos da soma dos níveis (só empresas com valor
    positivo nos dois períodos de cada passo: amostra casada, sem viés de entrada/saída de empresas)."""
    cols = list(niveis.columns)
    idx = [100.0]
    for a, b_ in zip(cols[:-1], cols[1:]):
        ok = (niveis[a] > 0) & (niveis[b_] > 0)
        idx.append(idx[-1] * (niveis.loc[ok, b_].sum() / niveis.loc[ok, a].sum()) if ok.any() else np.nan)
    return pd.Series(idx, index=cols)


@st.cache_data(max_entries=1, show_spinner=False)
def carregar_dividas(versao: tuple = ()) -> pd.DataFrame:
    arq = DADOS.parent / "dividas" / "debentures.csv.gz"
    if not arq.exists():
        return pd.DataFrame()
    d = pd.read_csv(arq, dtype={"cnpj_grupo": str, "cnpj_emissor": str},
                    parse_dates=["data_emissao", "vencimento"])
    d["cnpj_grupo"] = d["cnpj_grupo"].str.zfill(14)
    return d


def rotulo_remuneracao(r) -> str:
    """Remuneração de emissão: '105% DI', 'DI + 1,6%', 'IPCA + 6,5%' ou o índice."""
    ind, pct, tx = str(r.get("indice") or "").strip(), r.get("percentual_indice"), r.get("taxa_emissao")
    if ind.upper().startswith("DI") and pd.notna(pct) and pct not in (0, 100) and (pd.isna(tx) or tx == 0):
        return f"{pct:g}% DI"
    if pd.notna(tx) and tx != 0:
        return f"{ind} + {tx:g}%".replace(".", ",")
    return ind or "-"


def classe_indexador(ind) -> str:
    """Classe do indexador da debênture: CDI (pós-fixada), IPCA, PRÉ ou Outros (IGP-M, TR, câmbio, sem índice)."""
    u = str(ind or "").strip().upper()
    return "CDI" if u.startswith("DI") else "IPCA" if u.startswith("IPCA") else "PRÉ" if u.startswith("PR") else "Outros"


def pct_di_para_spread(pct, cdi):
    """%DI -> spread equivalente sobre o CDI (fração a.a., base 252), dado o CDI esperado no prazo (fração a.a.)."""
    d = (1 + cdi) ** (1 / 252) - 1
    return (1 + d * pct / 100) ** 252 / (1 + cdi) - 1


CLASSES_CUSTO = {"CDI": ("CDI+ (pós-fixada)", "CDI"), "IPCA": ("IPCA+", "NTN-B (ETTJ real ANBIMA)"),
                 "PRÉ": ("Prefixada", "LTN/NTN-F (ETTJ pré ANBIMA)"),
                 "Outros": ("Outros (IGP-M, TR, câmbio, sem índice)", "-")}


def custo_debentures(dd: pd.DataFrame, pj: pd.DataFrame) -> tuple[pd.DataFrame, pd.Timestamp]:
    """Custo de cada debênture (% a.a., base 252) e a referência soberana no mesmo prazo, na curva ANBIMA da data
    do preço (ou a mais recente antes dela).
    - taxa_contratada: remuneração de emissão (CDI: spread sobre o CDI, com %DI convertido em spread equivalente;
      IPCA: taxa real; PRÉ: taxa nominal). É o custo da dívida existente.
    - taxa_mercado: taxa indicativa ANBIMA na mesma convenção (custo de refinanciar hoje).
    - soberano: ETTJ real (IPCA) ou pré no prazo = duration ANBIMA (sem preço: prazo até o vencimento).
    - premio_soberano: taxa de mercado − soberano (p.p.); no CDI, o próprio spread de mercado sobre o CDI.
    - nominal_contratado / nominal_mercado: custo nominal equivalente com a curva de hoje
      (CDI: (1 + pré no prazo) × (1 + spread); IPCA: (1 + real) × (1 + inflação implícita no prazo))."""
    d = dd.copy()
    d["classe"] = d["indice"].map(classe_indexador)
    hoje = pd.Timestamp.today().normalize()
    venc_anos = ((d["vencimento"] - hoje).dt.days / 365.25).clip(lower=1 / 12)
    tem_dur = d["duration_anos"].notna() & (d["duration_anos"] > 0)
    d["prazo_anos"] = d["duration_anos"].where(tem_dur, venc_anos)
    d["prazo_por_vencimento"] = ~tem_dur
    dt_c = pd.NaT
    pre = real = np.full(len(d), np.nan)
    if not pj.empty:
        datas = sorted(pj["data"].unique())
        ref = pd.to_datetime(d["data_anbima"], errors="coerce").max() if "data_anbima" in d else pd.NaT
        cand = [x for x in datas if pd.isna(ref) or pd.Timestamp(x) <= ref]
        dt_c = pd.Timestamp(cand[-1] if cand else datas[-1])
        t = d["prazo_anos"].to_numpy(float)
        pre, real = curva_juros(pj, dt_c, "pre", t), curva_juros(pj, dt_c, "ipca", t)
    d["curva_pre"], d["curva_real"] = pre * 100, real * 100
    impl = (1 + pre) / (1 + real) - 1
    cl, pct, tx, mkt = d["classe"], d["percentual_indice"], d["taxa_emissao"], d["taxa_indicativa"]
    ic = d["indice_correcao"].fillna("").astype(str) if "indice_correcao" in d else pd.Series("", index=d.index)
    cdi = pd.Series(pre, index=d.index)              # CDI esperado no prazo ≈ curva prefixada
    eh_cdi = cl == "CDI"
    eh_pct = eh_cdi & pct.notna() & ~pct.isin([0, 100]) & (tx.isna() | (tx == 0))
    mkt_pct = eh_cdi & ic.str.contains(r"%\s*(?:do\s+)?DI", case=False, regex=True)
    contr = tx.where(~cl.isin(["IPCA", "PRÉ"]) | (tx != 0))       # CDI com taxa 0 = 100% do DI (spread zero)
    contr = contr.where(~eh_pct, pct_di_para_spread(pct, cdi) * 100)
    merc = mkt.where(~mkt_pct, pct_di_para_spread(mkt, cdi) * 100)
    d["taxa_contratada"] = contr.where(cl != "Outros")
    d["taxa_mercado"] = merc.where(cl != "Outros")
    sob = pd.Series(np.nan, index=d.index)
    sob = sob.mask(cl == "IPCA", d["curva_real"]).mask(cl == "PRÉ", d["curva_pre"])
    d["soberano"] = sob
    d["premio_soberano"] = (d["taxa_mercado"] - sob).where(~eh_cdi, d["taxa_mercado"])
    d["contratada_vs_soberano"] = (d["taxa_contratada"] - sob).where(~eh_cdi, d["taxa_contratada"])
    # remuneração que destoa da curva (ex.: prefixada muito abaixo do soberano): provável erro de cadastro no SND
    d["taxa_suspeita"] = cl.isin(["IPCA", "PRÉ"]) & ((d["contratada_vs_soberano"] < -5) | (d["taxa_contratada"] < 0))

    def nominal(taxa):
        s = pd.Series(np.nan, index=d.index)
        s = s.mask(eh_cdi, ((1 + cdi) * (1 + taxa / 100) - 1) * 100)
        s = s.mask(cl == "IPCA", ((1 + taxa / 100) * (1 + pd.Series(impl, index=d.index)) - 1) * 100)
        return s.mask(cl == "PRÉ", taxa)
    d["nominal_contratado"] = nominal(d["taxa_contratada"]).where(~d["taxa_suspeita"])
    d["nominal_mercado"] = nominal(d["taxa_mercado"])
    return d, dt_c


def resumo_custo(d: pd.DataFrame) -> pd.DataFrame:
    """Médias ponderadas pelo saldo, por classe de indexador, mais a linha total (custo nominal equivalente).
    Taxa de mercado, soberano e prêmio usam só as debêntures com preço ANBIMA (mesma amostra nas três)."""
    def mp(v, w):
        ok = v.notna() & w.notna() & (w > 0)
        return float((v[ok] * w[ok]).sum() / w[ok].sum()) if ok.any() else np.nan
    tot = d["saldo_rs_mm"].sum()
    linhas = []
    for c, (rot, ref) in CLASSES_CUSTO.items():
        g = d[d["classe"] == c]
        if g.empty:
            continue
        w, gv = g["saldo_rs_mm"], g[~g["taxa_suspeita"]]
        m = g[g["taxa_mercado"].notna()]
        linhas.append({"Indexador": rot, "Qtde": len(g), "Saldo (R$ mm)": w.sum(),
                       "% das debêntures": w.sum() / tot if tot else np.nan,
                       "Prazo médio (anos)": mp(g["prazo_anos"], w),
                       "Custo contratado (% a.a.)": mp(gv["taxa_contratada"], gv["saldo_rs_mm"]),
                       "Taxa de mercado (% a.a.)": mp(m["taxa_mercado"], m["saldo_rs_mm"]),
                       "Referência": ref,
                       "Soberano no prazo (% a.a.)": mp(m["soberano"], m["saldo_rs_mm"]),
                       "Prêmio de mercado (p.p.)": mp(m["premio_soberano"], m["saldo_rs_mm"]),
                       "Contratado − soberano hoje (p.p.)": mp(gv["contratada_vs_soberano"], gv["saldo_rs_mm"]),
                       "Custo nominal contratado (% a.a.)": mp(gv["nominal_contratado"], gv["saldo_rs_mm"]),
                       "Custo nominal a mercado (% a.a.)": mp(m["nominal_mercado"], m["saldo_rs_mm"]),
                       "% do saldo com preço ANBIMA": m["saldo_rs_mm"].sum() / w.sum() if w.sum() else np.nan})
    if linhas:
        ok, m = d[~d["taxa_suspeita"]], d[d["taxa_mercado"].notna()]
        linhas.append({"Indexador": "Total (custo nominal equivalente)", "Qtde": len(d), "Saldo (R$ mm)": tot,
                       "% das debêntures": 1.0, "Prazo médio (anos)": mp(d["prazo_anos"], d["saldo_rs_mm"]),
                       "Referência": "-",
                       "Custo nominal contratado (% a.a.)": mp(ok["nominal_contratado"], ok["saldo_rs_mm"]),
                       "Custo nominal a mercado (% a.a.)": mp(m["nominal_mercado"], m["saldo_rs_mm"]),
                       "% do saldo com preço ANBIMA": m["saldo_rs_mm"].sum() / tot if tot else np.nan})
    return pd.DataFrame(linhas)


def custo_implicito_caixa(base: pd.DataFrame, cnpj: str) -> tuple[float, object]:
    """Juros pagos da dívida nos últimos 12 meses (DFC, ex-arrendamento) ÷ dívida bruta média ex-arrendamento
    (fim do LTM e um ano antes). Cobre a dívida toda (bancos, BNDES, externa), mas em regime de caixa."""
    if "juros_pagos" not in base.columns:
        return np.nan, None
    lt = base[(base["cnpj"] == cnpj) & (base["tipo"] == "ltm")].sort_values("dt_fim")
    lt = lt[lt["juros_pagos"].notna() & lt["divida_bruta_ex_arrend"].notna()]
    if lt.empty:
        return np.nan, None
    fim = lt.iloc[-1]
    ant = lt[lt["dt_fim"] <= fim["dt_fim"] - pd.Timedelta(days=330)].tail(1)
    divs = [fim["divida_bruta_ex_arrend"]] + ([ant["divida_bruta_ex_arrend"].iloc[0]] if len(ant) else [])
    dm_ = float(np.mean(divs))
    return (float(fim["juros_pagos"]) / dm_ * 100 if dm_ > 0 else np.nan), fim["dt_fim"]


PALETA_OV = ["#1f77b4", "#ff7f0e", "#2ca02c", "#d62728", "#9467bd", "#8c564b", "#e377c2", "#7f7f7f", "#bcbd22",
             "#17becf"]


def alvo(cor: str) -> dict:
    """Marcador do valor no último pregão: alvo (círculo com ponto) na mesma cor da curva."""
    return dict(symbol="circle-open-dot", size=18, color=cor, line=dict(width=3, color=cor))


def fmt_escala(v) -> str:
    """Valor em milhões com a ordem de grandeza ajustada ao tamanho: trilhão (4 casas), bilhão (2 casas) ou milhão
    (sem casas). Usado nos valores de mercado e EV do modo EUA (ex.: US$ 3,8124 tri em vez de US$ 3.812.400 mi)."""
    if pd.isna(v):
        return "-"
    a = abs(v)
    n, un, casas = (v / 1e6, "tri", 4) if a >= 999_995 else (v / 1e3, "bi", 2) if a >= 999.5 else (v, "mi", 0)
    return f"{MOEDA} " + f"{n:,.{casas}f}".replace(",", "X").replace(".", ",").replace("X", ".") + f" {un}"


def painel_mercado(cnpj: str):
    """Valores e múltiplos de mercado na cotação mais recente (último pregão)."""
    if mult.empty:
        return
    h = mult[(mult["cnpj"] == str(cnpj).zfill(14)) & (mult["tipo"] == "trimestral")].sort_values("data").tail(1)
    if h.empty:
        h = mult[mult["cnpj"] == str(cnpj).zfill(14)].sort_values("data").tail(1)
    if h.empty:
        return
    h = h.iloc[0]
    fmt_mm = fmt_escala if EUA else (lambda v: f"{MOEDA} {v:,.0f} mi".replace(",", ".") if pd.notna(v) else "-")
    fmt_x = lambda v: f"{v:.2f}x".replace(".", ",") if pd.notna(v) else "-"
    fmt_p = lambda v: f"{v:.1%}".replace(".", ",") if pd.notna(v) else "-"
    per = str(h.get("periodo_base") or "")
    d_bal = pd.to_datetime(h.get("dt_fim_base"), errors="coerce")
    if pd.isna(d_bal) and per:                 # bases antigas não traziam a data: busca pelo período
        for df_ in (base, fin):
            if not df_.empty and {"cnpj", "periodo", "dt_fim"} <= set(df_.columns):
                m_ = df_[(df_["cnpj"].astype(str).str.zfill(14) == str(cnpj).zfill(14)) & (df_["periodo"] == per)]
                if len(m_):
                    d_bal = pd.to_datetime(m_["dt_fim"].iloc[0], errors="coerce")
                    break
    atraso = (pd.Timestamp(h["data"]) - d_bal).days if pd.notna(d_bal) else np.nan
    aviso = (f" ⚠️ balanço com {atraso // 30} meses: a companhia não "
             + ("arquivou 10-Q/10-K mais recentes na SEC" if EUA else "entregou ITRs mais recentes à CVM")
             if pd.notna(atraso) and atraso > 200 else "")
    st.markdown(f"**Na cotação de {pd.Timestamp(h['data']).strftime('%d/%m/%Y')}** — balanço: "
                f"{h.get('periodo_base', '') or 'último documento divulgado'}"
                + (f" (até {d_bal.strftime('%d/%m/%Y')})" if pd.notna(d_bal) else "") + aviso)
    k1, k2, k3 = st.columns(3)
    k1.metric("Valor de mercado", fmt_mm(h.get("valor_mercado")))
    k2.metric("EV c/ arrendamento", fmt_mm(h.get("ev_com_arrend")))
    k3.metric("EV s/ arrendamento", fmt_mm(h.get("ev_sem_arrend")))
    k4, k5, k6, k7 = st.columns(4)
    k4.metric("P/VPA", fmt_x(h.get("p_vpa")))
    k5.metric("P/L", fmt_x(h.get("p_l")))
    k6.metric("EV/EBITDA equiv. IFRS 16" if EUA else "EV/EBITDA IFRS 16", fmt_x(h.get("ev_ebitda_ifrs16")))
    k7.metric("EV/EBITDA após arrend. fin." if EUA else "EV/EBITDA pré-IFRS 16", fmt_x(h.get("ev_ebitda_pre_ifrs16")))
    k8, k9, k10, k11 = st.columns(4)
    k8.metric("EV/Receita (EV c/ arrend.)", fmt_x(h.get("ev_receita_com_arrend")))
    k9.metric("EV/Receita (EV s/ arrend.)", fmt_x(h.get("ev_receita_sem_arrend")))
    k10.metric("ROE (12m)", fmt_p(h.get("roe")))
    k11.metric("Dividend yield (12m)", fmt_p(h.get("dividend_yield")))
    if str(h.get("notas") or ""):
        st.caption("Notas do cálculo: " + str(h.get("notas")))


@st.cache_data(max_entries=1, show_spinner=False)
def carregar_vm_diario(versao: tuple = ()) -> pd.DataFrame:
    arq = DADOS / "valor_mercado_diario.csv.gz"
    if not arq.exists():
        return pd.DataFrame()
    v = pd.read_csv(arq, dtype={"cnpj": str}, parse_dates=["data"])
    v["cnpj"] = v["cnpj"].str.zfill(14)
    return v


def fmt_tabela(df: pd.DataFrame, cols: list[str]) -> dict:
    cfg = {}
    for c in cols:
        rot, f, _ = METRICAS[c]
        if f == "mm":
            cfg[c] = st.column_config.NumberColumn(f"{rot} ({UN})", format="%,.1f")
        elif f == "pct":
            cfg[c] = st.column_config.NumberColumn(rot, format="%.1f%%")
        elif f == "d":
            cfg[c] = st.column_config.NumberColumn(f"{rot} (dias)", format="%.0f")
        else:
            cfg[c] = st.column_config.NumberColumn(rot, format="%.2fx")
    return cfg


def para_exibir(df: pd.DataFrame, cols: list[str]) -> pd.DataFrame:
    out = df.copy()
    for c in cols:
        if METRICAS[c][1] == "pct":
            out[c] = out[c] * 100
    return out


def excel_bytes(abas: dict[str, pd.DataFrame]) -> bytes:
    buf = io.BytesIO()
    with pd.ExcelWriter(buf, engine="openpyxl") as xw:
        for nome, df in abas.items():
            df.to_excel(xw, sheet_name=re.sub(r"[\\/*?:\[\]]", "-", nome)[:31], index=False)
    return buf.getvalue()


def eixo_x(tipo: str) -> str:
    return "exercicio" if tipo == "anual" else "dt_fim"


@st.cache_resource(max_entries=1, show_spinner="Carregando a base americana…")
def carregar_eua(versao: str):
    """Base da SEC no formato da CVM. cache_resource (sem cópia por sessão): a base americana é ~10x a brasileira
    e o Streamlit gratuito tem pouca memória. As abas só filtram a base, não a alteram."""
    b = eua.base_como_br()
    b = eua.compactar(eua.calc_roic(enriquecer(b)))                  # derivadas e ROIC em float32 já na saída do enriquecimento
    eua.liberar_memoria()
    m = eua.multiplos()
    if not m.empty:                                   # mesmo tratamento dos múltiplos brasileiros
        m = eua.compactar(preparar_mult(m))
        b = juntar_mult(b, m, leve=True)
    b = eua.categorizar(eua.compactar(garantir_colunas(b)))
    eua.liberar_memoria()
    return b, eua.cobertura(), pd.Series(eua.metadados()), pd.DataFrame(), m, pd.DataFrame()


if EUA:
    carregar.clear()                                  # um mercado por vez na memória (Streamlit gratuito)
    if not eua.tem_acesso():
        eua.sem_acesso()
        st.stop()
    try:
        base, cob, meta, fin, mult, fin_per = carregar_eua(eua.versao())
    except Exception as e:
        eua.sem_acesso(str(e))
        st.stop()
else:
    carregar_eua.clear()                              # um mercado por vez na memória (Streamlit gratuito)
    try:
        base, cob, meta, fin, mult, fin_per = carregar(versao_arquivos(*(DADOS / f for f in (
            "base.csv.gz", "cobertura.csv", "metadados.json", "financeiras.csv.gz", "multiplos.csv.gz"))))
    except FileNotFoundError:
        st.error(f"Base não encontrada em {DADOS}. Rode o workflow no modo universo.")
        st.stop()

# ------------------------------------------------------------------ navegação: clique numa empresa -> aba Empresa
_dest = st.session_state.pop("_ir_empresa", None)
if _dest:
    if not fin.empty and _dest in set(fin["rotulo"]):
        st.session_state["grupo_emp"] = "Bancos e seguradoras"
        st.session_state["emp_fin"] = _dest
    else:
        st.session_state["grupo_emp"] = "Não financeiras"
        st.session_state["emp_sel"] = _dest
    st.session_state["aba_principal"] = "Empresa"           # abre a aba Empresa (st.tabs com chave)


def chave_clicavel(chave: str) -> str:
    """Chave versionada do widget clicável. A seleção de linha/ponto fica guardada no navegador, e apagar a
    chave do session_state não a limpa: no rerun seguinte (trocar de empresa, de aba...) ela voltava e
    disparava a navegação de novo, prendendo o usuário na empresa clicada. Trocar a versão cria um widget novo,
    sem seleção."""
    return f"{chave}__v{st.session_state.get('_clk_ver_' + chave, 0)}"


def ir_para_empresa(rotulo, chave: str):
    """Guarda a empresa clicada, descarta a seleção da tabela/gráfico e recarrega já na aba Empresa."""
    if isinstance(rotulo, str) and " — " in rotulo:
        st.session_state["_ir_empresa"] = rotulo
        st.session_state.pop(chave_clicavel(chave), None)
        st.session_state["_clk_ver_" + chave] = st.session_state.get("_clk_ver_" + chave, 0) + 1
        st.rerun()


def tabela_clicavel(dados: pd.DataFrame, rotulos, chave: str, alvo=None, **kw):
    """st.dataframe com seleção de linha: clicar numa empresa abre a aba Empresa com ela."""
    ev = (alvo or st).dataframe(dados, on_select="rerun", selection_mode="single-row", key=chave_clicavel(chave), **kw)
    linhas = list(getattr(getattr(ev, "selection", None), "rows", []) or [])
    if linhas:
        ir_para_empresa(list(rotulos)[linhas[0]], chave)


def multiselect_seguro(alvo, rotulo: str, opcoes: list, padrao: list, key: str, **kw) -> list:
    """Multiselect que tolera a troca da lista de opções (grupo, periodicidade): descarta da seleção guardada o
    que não existe mais e só passa o padrão na primeira vez (evita erro e aviso do Streamlit)."""
    if key in st.session_state:
        st.session_state[key] = [k for k in st.session_state[key] if k in opcoes]
        return alvo.multiselect(rotulo, opcoes, key=key, **kw)
    return alvo.multiselect(rotulo, opcoes, default=[k for k in padrao if k in opcoes], key=key, **kw)


# ------------------------------------------------------------------ barra lateral
st.sidebar.title("🇺🇸 Price Discovery Scanning · EUA" if EUA else "📊 Price Discovery Scanning")
if meta is not None:
    st.sidebar.caption(f"Base gerada em {meta.get('gerado_em', '?')} · {meta.get('empresas_com_dados', '?')} empresas · "
                       + ("Fonte: SEC (10-K/10-Q, XBRL) + classificação SEC/SIC · US$ mm" if EUA else
                          "Fonte: CVM (DFP/ITR) + classificação B3"))

tipo_rot = st.sidebar.radio("Periodicidade", list(TIPOS), horizontal=False)
tipo = TIPOS[tipo_rot]
df_t = base[base["tipo"] == tipo]

setores = sorted(x for x in df_t["setor"].unique() if x)
sel_setor = st.sidebar.multiselect("Setor", setores, placeholder="Todos")
f1 = df_t[df_t["setor"].isin(sel_setor)] if sel_setor else df_t
subs = sorted(x for x in f1["subsetor"].unique() if x)
sel_sub = st.sidebar.multiselect("Subsetor", subs, placeholder="Todos")
f2 = f1[f1["subsetor"].isin(sel_sub)] if sel_sub else f1
segs = sorted(x for x in f2["segmento"].unique() if x)
sel_seg = st.sidebar.multiselect("Segmento", segs, placeholder="Todos")
f3 = f2[f2["segmento"].isin(sel_seg)] if sel_seg else f2
emps = sorted(f3["rotulo"].unique())
sel_emp = st.sidebar.multiselect("Empresas", emps, placeholder="Todas do filtro")

setor_f = f3  # universo do setor filtrado (base da participação), antes do filtro de empresa
JANELA = None  # período escolhido na barra lateral: (coluna, início, fim); usado também nos gráficos da aba Empresa
if tipo == "anual":
    anos = sorted(setor_f["exercicio"].dropna().astype(int).unique())
    if anos:
        a0, a1 = st.sidebar.select_slider("Exercícios", options=anos, value=(anos[0], anos[-1]))
        setor_f = setor_f[setor_f["exercicio"].between(a0, a1)]
        JANELA = ("exercicio", a0, a1)
else:
    datas = sorted(setor_f["dt_fim"].dropna().unique())
    if datas:
        rot_d = [pd.Timestamp(d).strftime("%m/%Y") for d in datas]
        i0, i1 = st.sidebar.select_slider("Período (data de fechamento)", options=list(range(len(datas))),
                                          value=(max(0, len(datas) - 12), len(datas) - 1),
                                          format_func=lambda i: rot_d[i])
        setor_f = setor_f[setor_f["dt_fim"].between(datas[i0], datas[i1])]
        JANELA = ("dt_fim", datas[i0], datas[i1])

if st.sidebar.checkbox("Esconder períodos com alerta de qualidade", value=False,
                       help="Trimestres com diferença implausível, acumulado ausente ou LTM que inclui esses trimestres."):
    setor_f = setor_f[~setor_f["alerta"]]
filtrado = setor_f[setor_f["rotulo"].isin(sel_emp)] if sel_emp else setor_f

grupos = {}
for k, (_, _, g) in METRICAS.items():
    grupos.setdefault(g, []).append(k)
sel_met = st.sidebar.multiselect("Métricas", list(METRICAS), default=PADRAO, format_func=lambda k: ROT[k])
if not sel_met:
    sel_met = PADRAO

st.sidebar.caption(f"{filtrado['cnpj'].nunique()} empresas · {len(filtrado)} linhas no filtro")

def empresa_financeira():
    """Aba Empresa para bancos e seguradoras: indicadores próprios vs. mediana do segmento + múltiplos.
    Segue a periodicidade da barra lateral: anual (DFP), LTM ou trimestral (ITR)."""
    emp_f = st.selectbox("Instituição", sorted(fin["rotulo"].unique()), key="emp_fin")
    anual = tipo == "anual" or fin_per.empty
    pares = fin if anual else fin_per[fin_per["tipo"] == tipo]
    d = pares[pares["rotulo"] == emp_f].sort_values("exercicio" if anual else "dt_fim")
    if d.empty and not anual:
        st.info("Sem ITRs desta instituição na base: mostrando a série anual (DFP).")
        anual, pares = True, fin
        d = fin[fin["rotulo"] == emp_f].sort_values("exercicio")
    x_col = "exercicio" if anual else "dt_fim"
    seg_f = d["segmento"].iloc[0]
    desc = ("série anual (DFP)" if anual else
            "últimos 12 meses (fluxos de 4 trimestres somados; saldos do último ITR/DFP)" if tipo == "ltm" else
            "trimestral (ITR; fluxo do trimestre por diferença de acumulados; ROE, ROA e custo do crédito "
            "anualizados ×4)")
    st.caption(f"Financeiro / {d['subsetor'].iloc[0]} / **{seg_f}** · {desc} · "
               "base separada, com plano de contas de instituição financeira")
    # ---- grupo de comparação (mesma lógica das não financeiras): segmento, subsetor, todas ou cesta escolhida
    cc1, cc2 = st.columns([1, 2])
    niveis_f = {"Segmento": "segmento", "Subsetor": "subsetor", "Todas as financeiras": None}
    nivel_f = cc1.radio("Comparar com (mediana)", list(niveis_f) + ["Cesta de instituições"], horizontal=True,
                        key="cmp_nivel_fin")
    if nivel_f == "Cesta de instituições":
        cesta_f = cc2.multiselect("Instituições da cesta (qualquer banco ou seguradora da base)",
                                  sorted(fin["rotulo"].unique()), default=[emp_f], key="cmp_cesta_fin")
        cn_cmp = set(fin.loc[fin["rotulo"].isin(cesta_f), "cnpj"].astype(str).str.zfill(14))
        nome_cmp = f"da cesta ({len(cesta_f)})"
    elif niveis_f[nivel_f] is None:
        cn_cmp, nome_cmp = None, "das financeiras"
    else:
        col_n = niveis_f[nivel_f]
        cn_cmp = set(fin.loc[fin[col_n] == d[col_n].iloc[0], "cnpj"].astype(str).str.zfill(14))
        nome_cmp = str(d[col_n].iloc[0])

    def cmp_f(df):
        """Recorte do grupo de comparação; a própria instituição entra só se estiver no grupo (ou na cesta)."""
        return df if cn_cmp is None else df[df["cnpj"].astype(str).str.zfill(14).isin(cn_cmp)]

    n_cmp = cmp_f(pares)["cnpj"].nunique()
    cc2.caption(f"Mediana calculada sobre {n_cmp} instituição(ões)" +
                ("" if nivel_f == "Cesta de instituições" else f": {nome_cmp}") +
                ". Vale para os indicadores e para os múltiplos de mercado abaixo.")
    disp_k = [k for k in FIN if k in d.columns and d[k].notna().any() and FIN[k][1] in ("pct", "x")]
    padrao = [k for k in ("roe", "roa", "pl_ativo", "indice_eficiencia", "custo_credito", "payout")
              if k in disp_k]
    sel = st.multiselect("Indicadores", disp_k, default=padrao, format_func=lambda k: FIN[k][0], key="ind_emp_fin")
    med = cmp_f(pares).groupby(x_col)[sel].median() if sel else pd.DataFrame()
    cols = st.columns(2)
    for i, k in enumerate(sel):
        mm = med[med.index.isin(d[x_col])]
        fig = go.Figure()
        fig.add_trace(go.Scatter(x=d[x_col], y=d[k], mode="lines+markers", name=emp_f.split(" — ")[0]))
        fig.add_trace(go.Scatter(x=mm.index, y=mm[k], mode="lines", line=dict(dash="dash"), name=f"Mediana {nome_cmp}"))
        add_medias(fig, d[x_col], d[k], mm[k])
        fig.update_layout(title=FIN[k][0] + {"mm": " (R$ mm)", "pct": "", "x": " (x)"}[FIN[k][1]], height=320,
                          margin=dict(t=40, b=20, l=10, r=10), legend=dict(orientation="h", y=-0.2))
        if FIN[k][1] == "pct":
            fig.update_yaxes(tickformat=".0%")
        cols[i % 2].plotly_chart(fig, width="stretch")

    # ---- composição do ativo: carteira bruta − provisão + TVM e derivativos + demais ativos = ativo total
    comp = [("carteira_credito_bruta", "Carteira de crédito bruta"), ("tvm", "TVM e derivativos"),
            ("demais_ativos", "Demais ativos"), ("provisao_carteira", "(−) Provisão para perdas")]
    if all(k in d.columns for k, _ in comp) and d["carteira_credito_bruta"].notna().any():
        st.subheader("Composição do ativo")
        at_ = d["ativo_total"].where(d["ativo_total"] > 0)
        c_a, c_b = st.columns(2)
        for col_, (tit_, den_) in zip((c_a, c_b), (("% do ativo total", at_), ("R$ mm", 1))):
            fig = go.Figure()
            for k, nome in comp:
                fig.add_trace(go.Bar(x=d[x_col], y=d[k] * (-1 if k == "provisao_carteira" else 1) / den_, name=nome))
            fig.update_layout(barmode="relative", title=tit_, height=360, legend=dict(orientation="h", y=-0.25),
                              margin=dict(t=40, b=20, l=10, r=10))
            if den_ is at_:
                fig.update_yaxes(tickformat=".0%")
            col_.plotly_chart(fig, width="stretch")
        st.caption("Carteira bruta − provisão + TVM e derivativos + demais ativos = ativo total. Carteira bruta: contas "
                   "de operações de crédito do balanço, com a provisão somada de volta quando vem dentro da conta "
                   "(COSIF); sob IFRS a conta já é bruta e a provisão é conta irmã (pode incluir perdas esperadas de "
                   "TVM ao custo amortizado). Demais ativos = resíduo: caixa, aplicações interfinanceiras, "
                   "compulsórios, tributos, investimentos etc. Vazio quando a captura não fecha.")

    st.subheader("Múltiplos de mercado")
    cnpj_f = str(d["cnpj"].iloc[0]).zfill(14)
    tipo_m = "anual" if anual else "trimestral"
    dm = mult[(mult["cnpj"] == cnpj_f) & (mult["tipo"] == tipo_m)].sort_values("data") if not mult.empty else pd.DataFrame()
    if not anual and len(dm):
        dm = dm[dm["data"] >= pd.Timestamp(d["dt_fim"].min())]
    if dm.empty:
        st.info("Múltiplos de bancos e seguradoras ainda não gerados: aparecem depois da próxima execução do coletor.")
        return emp_f, d, seg_f
    painel_mercado(d["cnpj"].iloc[0])
    st.caption("Anual: preço do último pregão do exercício com lucro e PL dos controladores do exercício. "
               "EV/EBITDA não se aplica a instituições financeiras." if anual else
               "Ponto no tempo: em cada data (fim de trimestre e último pregão), PL dos controladores do último "
               "ITR/DFP divulgado até a data e lucro e dividendos dos 12 meses desse documento. "
               "EV/EBITDA não se aplica a instituições financeiras.")
    pares_m = cmp_f(mult[mult["tipo"] == tipo_m])
    med_m = pares_m.groupby("data")[["p_vpa", "p_l", "roe", "dividend_yield"]].median()
    med_m = med_m[med_m.index.isin(dm["data"])]
    graficos = [("P/VPA", "p_vpa", "x"), ("P/L", "p_l", "x"), ("ROE (lucro ÷ PL dos controladores)", "roe", "pct"),
                ("Dividend yield (dividendos + JCP pagos ÷ valor de mercado)", "dividend_yield", "pct")]
    cm = [c for _ in range(2) for c in st.columns(2)]
    for (tit, k, fmt), col in zip(graficos, cm):
        fig = go.Figure()
        fig.add_trace(go.Scatter(x=dm["data"], y=dm[k], mode="lines+markers", name=emp_f.split(" — ")[0]))
        fig.add_trace(go.Scatter(x=med_m.index, y=med_m[k], mode="lines", line=dict(dash="dash"), name=f"Mediana {nome_cmp}"))
        add_medias(fig, dm["data"], dm[k], med_m[k])
        fig.update_layout(title=tit, height=320, margin=dict(t=40, b=20, l=10, r=10), legend=dict(orientation="h", y=-0.25))
        if fmt == "pct":
            fig.update_yaxes(tickformat=".0%")
        else:
            fig.update_yaxes(ticksuffix="x")
        col.plotly_chart(fig, width="stretch")
    st.caption("No setor financeiro, P/VPA × ROE é a leitura central: P/VPA = P/L × ROE em cada data.")
    with st.expander("Valor de mercado, preços, ações e notas"):
        cols_m = ["data", "periodo_base", "valor_mercado", "ticker_on", "preco_on", "acoes_on", "ticker_pn", "preco_pn",
                  "acoes_pn", "p_vpa", "p_l", "roe", "dividendos_12m", "dividend_yield", "notas"]
        tab_m = dm[[c for c in cols_m if c in dm.columns]].assign(data=dm["data"].dt.date)
        st.dataframe(tab_m, hide_index=True, width="stretch")
    return emp_f, d, seg_f


IND_FIN_EXTRA = {"p_vpa": ("P/VPA", "x"), "dividend_yield": ("Dividend yield", "pct")}


def serie_financeira(rot: str) -> pd.DataFrame:
    """Série de um banco/seguradora na periodicidade da barra lateral + múltiplos de mercado da mesma data.
    Coluna _x: exercício (anual) ou data de fim em ano decimal (trimestral/LTM), no eixo do ponto do último pregão."""
    if tipo != "anual" and not fin_per.empty:
        d = fin_per[(fin_per["rotulo"] == rot) & (fin_per["tipo"] == tipo)].sort_values("dt_fim").copy()
        if len(d):
            extra = list(IND_FIN_EXTRA) + ["valor_mercado", "p_l"]
            if not mult.empty:
                cn = str(d["cnpj"].iloc[0]).zfill(14)
                m = mult[(mult["cnpj"] == cn) & (mult["tipo"] == "trimestral")][["data"] + extra] \
                    .drop_duplicates("data")
                d = d.drop(columns=[c for c in extra if c in d.columns]).merge(
                    m, left_on="dt_fim", right_on="data", how="left").drop(columns="data")
            dt_ = pd.to_datetime(d["dt_fim"])
            d["_x"] = dt_.dt.year + (dt_.dt.dayofyear - 1) / 366
            return d
    d = fin[fin["rotulo"] == rot].sort_values("exercicio").copy()
    if not mult.empty and len(d):
        cn = str(d["cnpj"].iloc[0]).zfill(14)
        m = mult[(mult["cnpj"] == cn) & (mult["tipo"] == "anual")].assign(exercicio=lambda x: x["data"].dt.year)
        d = d.drop(columns=[c for c in IND_FIN_EXTRA if c in d.columns]).merge(
            m[["exercicio"] + list(IND_FIN_EXTRA)].drop_duplicates("exercicio"), on="exercicio", how="left")
    d["_x"] = d["exercicio"]
    return d


def ponto_atual_fin(cnpj: str, chave: str):
    """Último ponto de mercado da instituição (pregão mais recente), para marcar no gráfico."""
    if mult.empty:
        return None
    hm = mult[mult["cnpj"] == str(cnpj).zfill(14)].sort_values("data")
    if hm.empty or chave not in hm.columns or pd.isna(hm[chave].iloc[-1]):
        return None
    dh = pd.Timestamp(hm["data"].iloc[-1])
    return dh.year + (dh.dayofyear - 1) / 366, float(hm[chave].iloc[-1]), dh


def extras_financeira(emp_f: str, d: pd.DataFrame, seg_f: str):
    """Aluguel de ações e gráficos sobrepostos para bancos e seguradoras (mesma lógica das demais empresas)."""
    # ---- aluguel de ações
    st.subheader("Aluguel de ações (custo do short)")
    alug = carregar_aluguel(versao_arquivos(DADOS.parent / "aluguel" / "resumo.csv.gz"))
    raiz_e = str(d["ticker"].iloc[0])[:4]
    a_emp = alug[alug["ticker"].str[:4] == raiz_e] if not alug.empty else pd.DataFrame()
    if alug.empty:
        st.info("Dados de aluguel ainda não disponíveis (saidas/aluguel/resumo.csv.gz).")
    elif a_emp.empty:
        st.info(f"Sem registros de aluguel para os papéis {raiz_e}*.")
    else:
        tks = a_emp.groupby("ticker")["saldo_alugado_qtd"].mean().sort_values(ascending=False).index.tolist()
        tk = st.selectbox("Papel", tks, key="alug_tk_fin")
        at = a_emp[a_emp["ticker"] == tk].sort_values("data")
        grupo = fin[fin["segmento"] == seg_f][["cnpj", "ticker"]].drop_duplicates("cnpj")
        raiz_cnpj = dict(zip(grupo["ticker"].astype(str).str[:4], grupo["cnpj"].astype(str).str.zfill(14)))
        ag = alug[alug["ticker"].str[:4].isin(raiz_cnpj)]
        princ = (ag.groupby("ticker")["saldo_alugado_qtd"].mean().reset_index()
                 .assign(r=lambda x: x["ticker"].str[:4]).sort_values("saldo_alugado_qtd").groupby("r").tail(1)["ticker"])
        med_tx = ag[ag["ticker"].isin(princ)].groupby("data")["taxa_tomador_media"].median()
        a1, a2 = st.columns(2)
        fig = go.Figure()
        fig.add_trace(go.Scatter(x=at["data"], y=at["taxa_tomador_media"], mode="lines", name=f"Tomador {tk}"))
        fig.add_trace(go.Scatter(x=at["data"], y=at["taxa_doador_media"], mode="lines", name=f"Doador {tk}"))
        mt = med_tx.reindex(at["data"])
        fig.add_trace(go.Scatter(x=at["data"], y=mt, mode="lines", line=dict(dash="dash"), name=f"Mediana tomador {seg_f}"))
        add_medias(fig, at["data"], at["taxa_tomador_media"], mt, "tomador")
        fig.update_layout(title="Taxa de aluguel (média ponderada do dia, % a.a.)", height=360, yaxis_tickformat=".2%",
                          legend=dict(orientation="h", y=-0.25), margin=dict(t=40, b=20, l=10, r=10))
        a1.plotly_chart(fig, width="stretch")
        # ações em circulação: última composição disponível (bancos têm múltiplos anuais)
        if not mult.empty:
            ult_m = mult.sort_values("data").groupby("cnpj").tail(1)
            acoes = (ult_m["acoes_on"].fillna(0) + ult_m["acoes_pn"].fillna(0)).where(lambda x: x > 0)
            acoes_por_cnpj = dict(zip(ult_m["cnpj"], acoes))
        else:
            acoes_por_cnpj = {}

        def pct_alugado(df_r, cnpj_):
            q = acoes_por_cnpj.get(cnpj_)
            serie = df_r.groupby("data")["saldo_alugado_qtd"].sum(min_count=1)
            return serie / q if q else serie * np.nan
        p_emp = pct_alugado(a_emp, str(d["cnpj"].iloc[0]).zfill(14))
        pares_pct = [pct_alugado(ag[ag["ticker"].str[:4] == r], c).rename(r) for r, c in raiz_cnpj.items()]
        med_pct = pd.concat(pares_pct, axis=1).median(axis=1) if pares_pct else pd.Series(dtype=float)
        fig = go.Figure()
        fig.add_trace(go.Scatter(x=p_emp.index, y=p_emp.values, mode="lines", name=f"{raiz_e} (todas as classes)"))
        mp = med_pct.reindex(p_emp.index)
        fig.add_trace(go.Scatter(x=p_emp.index, y=mp.values, mode="lines", line=dict(dash="dash"), name=f"Mediana {seg_f}"))
        add_medias(fig, p_emp.index, p_emp.values, mp.values)
        fig.update_layout(title="Saldo alugado (% das ações em circulação)", height=360, yaxis_tickformat=".1%",
                          legend=dict(orientation="h", y=-0.25), margin=dict(t=40, b=20, l=10, r=10))
        a2.plotly_chart(fig, width="stretch")
        st.caption(f"Fonte: B3 (empréstimo de ativos), via coletor-aluguel-b3; série desde {alug['data'].min().date()}. "
                   "Taxas ao ano (a B3 informa em fração), média ponderada pela quantidade contratada no dia. Saldo "
                   "alugado ÷ ações em circulação (total menos tesouraria) da última composição do capital. "
                   f"Comparáveis: o papel com maior saldo alugado de cada instituição do segmento {seg_f}.")

    # ---- gráficos sobrepostos
    st.subheader("Gráficos sobrepostos")
    ind_ov = {**{k: (v[0], v[1]) for k, v in FIN.items() if not k.startswith("cresc_")}, **IND_FIN_EXTRA}
    se = serie_financeira(emp_f)
    cand = [k for k in ind_ov if k in se.columns and se[k].notna().any()]
    o1, o2 = st.columns([3, 2])
    sel_ov = o1.multiselect("Linhas da instituição", cand, default=[k for k in ("valor_mercado",) if k in cand],
                            format_func=lambda k: ind_ov[k][0], key="ov_ind_fin")
    outras = o1.multiselect("Outras instituições (opcional)", sorted(r for r in fin["rotulo"].unique() if r != emp_f),
                            key="ov_emp_fin")
    ind_outras = o1.selectbox("Indicador das outras instituições", cand or list(ind_ov),
                              index=(cand.index("valor_mercado") if "valor_mercado" in cand else 0),
                              format_func=lambda k: ind_ov[k][0], key="ov_ind_outras_fin") if outras else None
    escala = o2.radio("Escala", ["Base 100 (primeiro ponto)", "Valores (dois eixos)"], key="ov_esc_fin")
    cn_f = str(d["cnpj"].iloc[0]).zfill(14)
    series_ov = [(f"{ind_ov[k][0]} {emp_f.split(' — ')[0]}", se["_x"], se[k], ind_ov[k][1], k, cn_f)
                 for k in sel_ov]
    for o in outras:
        so = serie_financeira(o)
        if ind_outras in so.columns:
            series_ov.append((f"{ind_ov[ind_outras][0]} {o.split(' — ')[0]}", so["_x"], so[ind_outras],
                              ind_ov[ind_outras][1], ind_outras,
                              str(fin[fin["rotulo"] == o]["cnpj"].iloc[0]).zfill(14)))
    if not series_ov:
        st.info("Escolha ao menos uma linha.")
        return
    fig = go.Figure()
    ignoradas = []
    DE_MERCADO_FIN = {"valor_mercado", "p_l", "p_vpa", "dividend_yield"}
    for i_s, (nome, x, y, f, chave, cn_s) in enumerate(series_ov):
        cor = PALETA_OV[i_s % len(PALETA_OV)]
        y = pd.to_numeric(y, errors="coerce").reset_index(drop=True)
        x = pd.Series(list(x))
        # valor no último pregão, para indicadores que dependem de preço
        ponto = ponto_atual_fin(cn_s, chave) if chave in DE_MERCADO_FIN else None
        if escala.startswith("Base 100"):
            validos = y[y.notna()]
            if validos.empty or validos.iloc[0] <= 0:
                ignoradas.append(nome)
                continue
            fig.add_trace(go.Scatter(x=x, y=y / validos.iloc[0] * 100, mode="lines+markers", name=nome,
                                     line=dict(color=cor), marker=dict(color=cor)))
            if ponto:
                fig.add_trace(go.Scatter(x=[ponto[0]], y=[ponto[1] / validos.iloc[0] * 100], mode="markers",
                                         marker=alvo(cor),
                                         name=f"{nome}: cotação de {ponto[2].strftime('%d/%m/%Y')}"))
        else:
            eixo = "y" if f == "mm" else "y2"
            fig.add_trace(go.Scatter(x=x, y=y * (100 if f == "pct" else 1), mode="lines+markers",
                                     name=nome + (" (%)" if f == "pct" else " (x)" if f == "x" else f" ({UN})"),
                                     yaxis=eixo, line=dict(color=cor), marker=dict(color=cor)))
            if ponto:
                fig.add_trace(go.Scatter(x=[ponto[0]], y=[ponto[1] * (100 if f == "pct" else 1)], mode="markers",
                                         marker=alvo(cor), yaxis=eixo,
                                         name=f"{nome}: cotação de {ponto[2].strftime('%d/%m/%Y')}"))
    if escala.startswith("Base 100"):
        fig.update_layout(yaxis_title="Base 100 = primeiro ponto")
        fig.add_hline(y=100, line_dash="dot", line_color="gray")
    else:
        fig.update_layout(yaxis=dict(title=UN),
                          yaxis2=dict(title="Múltiplos (x) e %", overlaying="y", side="right", showgrid=False))
    fig.update_layout(height=480, legend=dict(orientation="h", y=-0.2), margin=dict(t=30, b=20, l=10, r=10))
    st.plotly_chart(fig, width="stretch")
    if ignoradas:
        st.caption("Fora da base 100 (primeiro valor zero ou negativo): " + ", ".join(ignoradas) +
                   ". Use a escala de valores para vê-las.")
    st.caption("Série na periodicidade da barra lateral (anual: DFP; LTM e trimestral: ITR). Base 100 compara o "
               "comportamento relativo desde o primeiro ponto; em valores, "
               f"{UN} à esquerda e múltiplos/% à direita. ◎ = valor calculado com o preço do último pregão, na cor da curva.")
    # ---- cotação, estrutura societária e modelo de negócio (mesmas seções das não financeiras)
    _mm = mercado_mod()
    _cn = str(d["cnpj"].iloc[0]).zfill(14)
    _tk = str(d["ticker"].iloc[0] or "")
    st.divider()
    _mm.secao_cotacao(_tk[:4], _cn, emp_f, moeda=MOEDA, chave="cot_f")
    st.divider()
    _mm.secao_estrutura(_cn, emp_f.split(" — ")[-1] if " — " in emp_f else emp_f, chave="soc_f")
    st.divider()
    _mm.secao_modelo(_tk[:4], emp_f)


def add_medias(fig, x, y, ym=None, rotulo=""):
    """Linhas horizontais pontilhadas: média da empresa e média das medianas dos comparáveis, no período exibido."""
    xs = [v for v in pd.Series(list(x)).tolist() if pd.notna(v)]
    if not xs:
        return
    xr = [xs[0], xs[-1]]
    suf = f" ({rotulo})" if rotulo else ""
    me = pd.to_numeric(pd.Series(list(y)), errors="coerce").mean()
    if pd.notna(me):
        fig.add_trace(go.Scatter(x=xr, y=[me, me], mode="lines", name="Média da empresa" + suf,
                                 line=dict(dash="dot", width=1.5, color="#1f77b4"), opacity=0.7))
    if ym is not None:
        mm = pd.to_numeric(pd.Series(list(ym)), errors="coerce").mean()
        if pd.notna(mm):
            fig.add_trace(go.Scatter(x=xr, y=[mm, mm], mode="lines", name="Média da mediana" + suf,
                                     line=dict(dash="dot", width=1.5, color="#ff7f0e"), opacity=0.7))


NIVEIS_CMP = {"Segmento": "segmento", "Subsetor": "subsetor", "Setor": "setor"}


def grupo_comparacao(df: pd.DataFrame, linha: pd.Series, nivel: str, cesta: list[str]) -> pd.DataFrame:
    """Recorte usado nas medianas da aba Empresa: segmento, subsetor ou setor da empresa, ou uma cesta
    escolhida pelo usuário. A própria empresa entra na mediana do segmento/subsetor/setor; na cesta,
    só se for incluída."""
    if nivel == "Cesta de empresas":
        return df[df["rotulo"].isin(cesta)]
    col = NIVEIS_CMP[nivel]
    return df[df[col] == linha[col]]


RELATIVAS_PADRAO = ["margem_bruta", "margem_ebitda", "margem_liquida", "roe_pl_final", "giro_ativo",
                    "alavancagem_ativo_pl", "div_liq_ebitda_ifrs16", "capex_org_receita"]


def serie_anual_ou_ltm(rotulo: str) -> tuple[pd.DataFrame, str]:
    """Série para crescimento e DuPont: anual na visão anual; LTM nas visões trimestral/LTM
    (trimestre isolado não serve para crescimento anual nem para giro)."""
    t = "anual" if tipo == "anual" else "ltm"
    d = base[(base["rotulo"] == rotulo) & (base["tipo"] == t)].sort_values("dt_fim").copy()
    return d, t


def crescimento(d: pd.DataFrame, cols: list[str]) -> pd.DataFrame:
    """Variação contra o mesmo período do ano anterior (anual: ano anterior; LTM: 4 trimestres antes).
    Vazio quando a base anterior é ≤ 0 (crescimento sobre base negativa não tem leitura)."""
    d = d.drop(columns=[c + "_ant" for c in cols if c + "_ant" in d.columns])
    ant = d.copy()
    ant["dt_fim"] = ant["dt_fim"] + pd.DateOffset(years=1)
    j = pd.merge_asof(d.sort_values("dt_fim"), ant[["dt_fim"] + cols].sort_values("dt_fim"), on="dt_fim",
                      suffixes=("", "_ant"), tolerance=pd.Timedelta(days=20), direction="nearest")
    out = j[["dt_fim", "periodo"]].copy()
    for c in cols:
        base_ant = j[c + "_ant"]
        out["g_" + c] = np.where(base_ant > 0, j[c] / base_ant - 1, np.nan)
    return out


def graus_alavancagem(g: pd.DataFrame) -> pd.DataFrame:
    """GAO = Δ% EBIT ÷ Δ% receita; GAF = Δ% lucro líquido ÷ Δ% EBIT; GAT = GAO × GAF = Δ% LL ÷ Δ% receita.
    Vazio quando a variação do denominador é menor que 1% em módulo (o índice explode e perde sentido)."""
    def razao(a, b):
        return np.where((np.abs(b) >= 0.01) & pd.notna(a) & pd.notna(b), a / b, np.nan)
    out = g[["dt_fim", "periodo"]].copy()
    out["GAO"] = razao(g["g_ebit"], g["g_receita"])
    out["GAF"] = razao(g["g_lucro_liquido"], g["g_ebit"])
    out["GAT"] = razao(g["g_lucro_liquido"], g["g_receita"])
    return out


def secao_crescimento_dupont(rotulo: str, cmp, nome_cmp: str):
    d, t = serie_anual_ou_ltm(rotulo)
    if len(d) < 2:
        st.info("Série insuficiente para crescimento e DuPont.")
        return
    nota_t = "anuais" if t == "anual" else "dos últimos 12 meses (LTM), comparados com 4 trimestres antes"
    # ---- crescimento
    st.subheader("Crescimento")
    g = crescimento(d, ["receita", "ebitda", "ebit", "lucro_liquido"])
    # recorte pelo período da barra lateral (Exercícios): o a/a do primeiro exercício da janela continua usando o
    # ano anterior como base, e o acumulado começa (base 100) no primeiro exercício da janela
    d_jan = d[d[JANELA[0]].between(JANELA[1], JANELA[2])] if JANELA else d
    g_jan = g[g["dt_fim"].isin(d_jan["dt_fim"])]
    cc_v, cc_s = st.columns([1, 2])
    visao_cres = cc_v.radio("Visão", ["Contra o ano anterior", "Acumulado (base 100)"], horizontal=True,
                            key="emp_cres_visao")
    _todas_cres = {"Receita": "receita", "EBITDA": "ebitda", "Lucro líquido": "lucro_liquido"}
    _sel_cres = cc_s.multiselect("Séries", list(_todas_cres), default=["Receita"], key="emp_cres_series",
                                 help="Receita por padrão; EBITDA e lucro líquido são opcionais.")
    series_cres = tuple((_todas_cres[n], n) for n in _todas_cres if n in _sel_cres)
    fig = go.Figure()
    if not series_cres:
        st.info("Escolha ao menos uma série para o gráfico de crescimento.")
    elif d_jan.empty:
        st.info("Sem dados da empresa no período selecionado (Exercícios, na barra lateral).")
    elif visao_cres == "Contra o ano anterior":
        for c, nome in series_cres:
            fig.add_trace(go.Scatter(x=g_jan["periodo"], y=g_jan["g_" + c], name=nome, mode="lines+markers"))
        if not EUA:          # macro Brasil no mesmo recorte (PIB nominal, IPCA, R$/US$)
            macro_aba = macro_mod()
            macro_aba.comparar_aa(fig, g_jan, 12, key="mac_cmp_emp")
        fig.add_hline(y=0, line_dash="dot", line_color="gray")
        fig.update_layout(height=380, yaxis_tickformat=".0%", legend=dict(orientation="h", y=-0.2),
                          margin=dict(t=30, b=20, l=10, r=10), title=f"Crescimento contra o ano anterior (valores {nota_t})")
        st.plotly_chart(fig, width="stretch")
    else:
        # base 100 no primeiro período com receita positiva; uma série com base ≤ 0 fica de fora, e um valor ≤ 0 no
        # meio da série encerra a leitura dali em diante (índice sobre base negativa não tem leitura)
        dd = d_jan.sort_values("dt_fim").reset_index(drop=True)
        ok_base = dd.index[pd.to_numeric(dd["receita"], errors="coerce") > 0]
        if len(ok_base) < 2:
            st.info("Receita positiva em menos de dois períodos: sem crescimento acumulado para mostrar.")
        else:
            dd = dd.loc[ok_base[0]:].reset_index(drop=True)
            fora, cortadas = [], []
            for c, nome in series_cres:
                v = pd.to_numeric(dd[c], errors="coerce").to_numpy(dtype=float)
                if not v[0] > 0:
                    fora.append(nome)
                    continue
                idx = 100 * v / v[0]
                ruim = np.flatnonzero(idx <= 0)
                if len(ruim):
                    idx[ruim[0]:] = np.nan
                    cortadas.append(f"{nome} (a partir de {dd['periodo'].iloc[ruim[0]]})")
                fig.add_trace(go.Scatter(x=list(dd["periodo"]), y=idx, name=nome, mode="lines+markers"))
            if not EUA:      # macro Brasil no mesmo recorte, mesma base
                macro_aba = macro_mod()
                macro_aba.comparar_base100(fig, pd.Series(pd.to_datetime(dd["dt_fim"]).to_numpy(),
                                                          index=dd["periodo"].to_numpy()), 12, False, key="mac_cmp_emp")
            fig.add_hline(y=100, line_dash="dot", line_color="gray")
            fig.update_layout(height=380, legend=dict(orientation="h", y=-0.2), yaxis_title=f"Base 100 = {dd['periodo'].iloc[0]}",
                              margin=dict(t=30, b=20, l=10, r=10), title=f"Crescimento acumulado (valores {nota_t})")
            st.plotly_chart(fig, width="stretch")
            if fora:
                st.caption(" e ".join(fora) + f" fora do gráfico: valor não positivo no período-base ({dd['periodo'].iloc[0]}).")
            if cortadas:
                st.caption("Linha interrompida por valor não positivo: " + "; ".join(cortadas) + ".")
    # ---- graus de alavancagem
    st.subheader("Graus de alavancagem operacional e financeira")
    ga = graus_alavancagem(g)
    ga_jan = graus_alavancagem(g_jan)          # gráfico no período da barra lateral (a tabela abaixo segue completa)
    fig = go.Figure()
    for c, nome in (("GAO", "GAO = Δ% EBIT ÷ Δ% receita"), ("GAF", "GAF = Δ% lucro ÷ Δ% EBIT"),
                    ("GAT", "GAT = Δ% lucro ÷ Δ% receita")):
        fig.add_trace(go.Scatter(x=ga_jan["periodo"], y=ga_jan[c], mode="lines+markers", name=nome))
    fig.update_layout(height=340, yaxis_ticksuffix="x", legend=dict(orientation="h", y=-0.2),
                      margin=dict(t=30, b=20, l=10, r=10))
    st.plotly_chart(fig, width="stretch")
    st.caption("Leitura: GAO 3x = cada 1% de variação da receita moveu o EBIT em 3%. Graus calculados período a "
               "período a partir das variações observadas (não da estrutura de custos). Ficam vazios quando a receita "
               "ou o EBIT variaram menos de 1% ou quando a base anterior era negativa; valores muito altos indicam "
               "variação pequena no denominador, não necessariamente alavancagem alta.")
    # ---- DuPont
    st.subheader("Análise DuPont")
    dp = d[["periodo", "margem_liquida", "giro_ativo", "alavancagem_ativo_pl", "roe_pl_final"]].copy()
    dp["roe_dupont"] = dp["margem_liquida"] * dp["giro_ativo"] * dp["alavancagem_ativo_pl"]
    cols = st.columns(4)
    pares = cmp(base[base["tipo"] == t])
    med = pares.groupby("periodo")[["margem_liquida", "giro_ativo", "alavancagem_ativo_pl", "roe_pl_final"]].median()
    for (k, tit, fmt), col in zip([("margem_liquida", "Margem líquida", "pct"), ("giro_ativo", "Giro do ativo", "x"),
                                   ("alavancagem_ativo_pl", "Alavancagem (ativo ÷ PL)", "x"),
                                   ("roe_pl_final", "ROE (= produto)", "pct")], cols):
        fig = go.Figure()
        fig.add_trace(go.Scatter(x=dp["periodo"], y=dp[k], mode="lines+markers", name="Empresa"))
        mm = med.reindex(dp["periodo"])
        fig.add_trace(go.Scatter(x=dp["periodo"], y=mm[k], mode="lines", line=dict(dash="dash"), name=f"Mediana {nome_cmp}"))
        add_medias(fig, dp["periodo"], dp[k], mm[k])
        fig.update_layout(title=tit, height=280, showlegend=False, margin=dict(t=40, b=20, l=10, r=10))
        fig.update_yaxes(tickformat=".0%" if fmt == "pct" else None, ticksuffix="" if fmt == "pct" else "x")
        col.plotly_chart(fig, width="stretch")
    # decomposição da variação do ROE entre o primeiro e o último período (substituição sequencial)
    v = dp.dropna(subset=["margem_liquida", "giro_ativo", "alavancagem_ativo_pl"])
    if len(v) >= 2:
        opcoes = list(v["periodo"])
        c1, c2 = st.columns(2)
        p0 = c1.selectbox("De", opcoes, index=0, key="dp_de")
        p1 = c2.selectbox("Até", opcoes, index=len(opcoes) - 1, key="dp_ate")
        a, b = v[v["periodo"] == p0].iloc[0], v[v["periodo"] == p1].iloc[0]
        m0, g0, l0 = a["margem_liquida"], a["giro_ativo"], a["alavancagem_ativo_pl"]
        m1, g1, l1 = b["margem_liquida"], b["giro_ativo"], b["alavancagem_ativo_pl"]
        r0, rm, rg, r1 = m0 * g0 * l0, m1 * g0 * l0, m1 * g1 * l0, m1 * g1 * l1
        fig = go.Figure(go.Waterfall(
            x=[f"ROE {p0}", "Efeito margem", "Efeito giro", "Efeito alavancagem", f"ROE {p1}"],
            measure=["absolute", "relative", "relative", "relative", "total"],
            y=[r0, rm - r0, rg - rm, r1 - rg, r1], text=[f"{x:.1%}" for x in (r0, rm - r0, rg - rm, r1 - rg, r1)]))
        fig.update_layout(height=360, yaxis_tickformat=".0%", margin=dict(t=30, b=20, l=10, r=10),
                          title="De onde veio a variação do ROE")
        st.plotly_chart(fig, width="stretch")
        st.caption("ROE = margem líquida × giro do ativo × alavancagem, com lucro líquido consolidado e PL e ativo "
                   "do fim do período. Efeitos por substituição sequencial (margem, depois giro, depois alavancagem); "
                   "a ordem afeta um pouco a divisão entre os efeitos, não o total.")
    with st.expander("Tabela: crescimento, graus de alavancagem e DuPont"):
        tab = dp.merge(g.drop(columns="dt_fim"), on="periodo").merge(ga.drop(columns="dt_fim"), on="periodo")
        st.dataframe(tab, hide_index=True, width="stretch")


# ------------------------------------------------------------------ abas
if EUA:
    _abas_eua = ["Altas e baixas", "Visão do setor", "Empresa", "Ranking", "Filtro por indicador", "Juros (Treasuries)",
                 "Macro EUA (inflação, PIB, dólar)", "Dados e download", "Cobertura", "Metodologia"]
    if st.session_state.get("aba_principal") not in (None, *_abas_eua):
        st.session_state["aba_principal"] = "Altas e baixas"
    aba_var, aba_setor, aba_emp, aba_rank, aba_scr, aba_jur, aba_mac, aba_dados, aba_cob, aba_met = st.tabs(
        _abas_eua, key="aba_principal", on_change="rerun")
    aba_cal = aba_fin = None          # sem dados nos EUA (blocos abaixo ficam desligados)
else:
    aba_var, aba_setor, aba_emp, aba_rank, aba_scr, aba_cal, aba_fin, aba_jur, aba_mac, aba_dados, aba_cob, aba_met = st.tabs(
        ["Altas e baixas", "Visão do setor", "Empresa", "Ranking", "Filtro por indicador", "Calendário de resultados",
         "Financeiro (bancos e seguradoras)", "Juros (ETTJ)", "Macro (inflação, PIB, câmbio)", "Dados e download", "Cobertura", "Metodologia"],
        key="aba_principal", on_change="rerun")
st.caption("Dica: clique numa empresa nas tabelas ou no gráfico do Ranking para abri-la na aba Empresa.")

# -------- Visão do setor: um indicador, série histórica completa, participação de cada empresa
def col_periodo(df: pd.DataFrame) -> pd.Series:
    """Rótulo comum de período para comparar empresas. Trimestral e LTM: trimestre CIVIL da data de fim (ex.: 2T26 =
    junho/2026). O rótulo do ITR segue o exercício social da empresa, e companhias com exercício encerrado em março
    chamariam junho/2026 de 1T27 — misturando datas diferentes na mesma coluna e bagunçando a ordem."""
    if tipo == "anual":
        return df["exercicio"].astype("Int64").astype(str)
    dt = pd.to_datetime(df["dt_fim"])
    return dt.dt.quarter.astype("Int64").astype(str) + "T" + (dt.dt.year % 100).astype("Int64").astype(str).str.zfill(2)


def ordem_periodos(df: pd.DataFrame) -> list:
    o = df.assign(_c=col_periodo(df)).groupby("_c")["dt_fim"].max().sort_values()
    return list(o.index)


def agregado_setor(df: pd.DataFrame, m: str) -> pd.Series:
    """Total do setor (métricas em R$) ou razão agregada (soma num / soma den) por período."""
    d = df.assign(_c=col_periodo(df))
    if METRICAS[m][1] == "mm":
        return d.groupby("_c")[m].sum(min_count=1)
    if m in ("gao", "gaf", "gat"):
        num, den = {"gao": ("cresc_ebit", "cresc_receita"), "gaf": ("cresc_lucro_liquido", "cresc_ebit"),
                    "gat": ("cresc_lucro_liquido", "cresc_receita")}[m]
        a_, b_ = agregado_setor(df, num), agregado_setor(df, den)
        return (a_ / b_.where(b_.abs() >= 0.01)).astype(float)
    if m.startswith("cresc_") and m[6:] + "_ant" in d.columns:
        c = m[6:]
        ok = d[(d[c + "_ant"] > 0) & d[c].notna()]
        g = ok.groupby("_c")[[c, c + "_ant"]].sum()
        return (g[c] / g[c + "_ant"] - 1).astype(float)
    if m in AGREG:
        if tipo == "trimestral" and m in ("roe_pl_final", "div_liq_ebitda_ifrs16", "div_liq_ebitda_pre_ifrs16",
                                           "div_liq_ebitda_ajustado"):
            return pd.Series(dtype=float)
        n, dn = AGREG[m]
        ok = d.dropna(subset=[n, dn])
        if m in AGREG_DEN_POS:
            ok = ok[ok[dn] > 0]
        g = ok.groupby("_c")[[n, dn]].sum()
        return (g[n] / g[dn].where(g[dn] != 0)).astype(float)
    return pd.Series(dtype=float)


with aba_setor:
    if setor_f.empty:
        st.info("Nenhum dado para o filtro.")
    else:
        c1, c2 = st.columns([2, 1])
        rot_ind = lambda k: ROT[k] + {"mm": f" ({UN})", "pct": " (%)", "x": " (x)", "d": " (dias)"}[METRICAS[k][1]]
        inds = multiselect_seguro(c1, "Indicadores", list(METRICAS), ["receita"], "set_inds", format_func=rot_ind,
                                  help="Escolha um ou mais: cada indicador aparece em sequência, com tabela, "
                                       "gráfico e download próprios.")
        modo = c2.radio("Mostrar", ["Valores", "% do setor"], horizontal=True, key="set_modo",
                        disabled=not any(METRICAS[k][1] == "mm" for k in inds),
                        help="Participação = valor da empresa ÷ total do setor filtrado (setor/subsetor/segmento), "
                             f"no mesmo período. Vale para os indicadores em {MOEDA}; os demais seguem em valores.")
        if not inds:
            st.info("Escolha ao menos um indicador.")
        periodos = ordem_periodos(setor_f)
        d_set = setor_f.assign(_c=col_periodo(setor_f))
        d_emp = filtrado.assign(_c=col_periodo(filtrado))
        exp_todos = {}
        area_g = st.container()          # gráficos no topo da página
        area_t = st.container()          # tabelas e downloads no fim da página
        if inds:
            with area_t:
                st.divider()
                st.subheader("Tabelas")
        for ind in inds:
            fmt = METRICAS[ind][1]
            pct_setor = (modo == "% do setor") and fmt == "mm"
            total = agregado_setor(setor_f, ind).reindex(periodos)
            mediana = d_set.groupby("_c")[ind].median().reindex(periodos)
            n_emp = d_set.dropna(subset=[ind]).groupby("_c")["cnpj"].nunique().reindex(periodos).fillna(0).astype(int)

            tab = d_emp.pivot_table(index="rotulo", columns="_c", values=ind, aggfunc="first").reindex(columns=periodos)
            # ordena pelo último período em que a maioria das empresas já tem dado (evita que um ano com poucas
            # empresas, como exercícios encerrados em março, defina a ordem)
            ult = [p for p in periodos[::-1] if p in tab.columns and n_emp.get(p, 0) >= 0.5 * n_emp.max()] or \
                  [p for p in periodos[::-1] if p in tab.columns and tab[p].notna().any()]
            if ult:
                tab = tab.sort_values(ult[0], ascending=False)
            if pct_setor:
                tab = tab.div(total, axis=1)
            linhas_extra = {}
            if fmt == "mm":
                if sel_emp:
                    demais = total - d_emp.groupby("_c")[ind].sum(min_count=1).reindex(periodos).fillna(0)
                    linhas_extra["Demais empresas do setor"] = demais / total if pct_setor else demais
                linhas_extra["TOTAL DO SETOR (filtro)"] = total / total if pct_setor else total
            else:
                linhas_extra["SETOR AGREGADO"] = total
            if not pct_setor:
                linhas_extra["Mediana do setor"] = mediana
            tab_full = pd.concat([tab, pd.DataFrame(linhas_extra).T.reindex(columns=periodos)])
            tab_full.index.name = "Empresa"

            if pct_setor:
                cfg = {p: st.column_config.NumberColumn(p, format="%.1f%%") for p in periodos}
                vis = tab_full * 100
            elif fmt == "pct":
                cfg = {p: st.column_config.NumberColumn(p, format="%.1f%%") for p in periodos}
                vis = tab_full * 100
            elif fmt == "x":
                cfg = {p: st.column_config.NumberColumn(p, format="%.2fx") for p in periodos}
                vis = tab_full
            elif fmt == "d":
                cfg = {p: st.column_config.NumberColumn(p, format="%.0f dias") for p in periodos}
                vis = tab_full
            else:
                cfg = {p: st.column_config.NumberColumn(p, format="%,.1f") for p in periodos}
                vis = tab_full
            with area_t:
                if len(inds) > 1:
                    st.markdown(f"**{rot_ind(ind)}**")
                tabela_clicavel(vis, vis.index, f"clk_setor_{ind}", column_config=cfg, width="stretch", height=min(650, 38 + 35 * len(vis)))
                st.caption("Empresas com dado por período: " + " · ".join(f"{p}: {n}" for p, n in n_emp.items()) +
                           ". Mudanças no total também refletem entrada de empresas (IPOs) na base.")
                if fmt == "mm" and (setor_f[ind] < 0).any():
                    st.caption("⚠️ Há valores negativos neste indicador; a participação percentual pode não fazer sentido.")
                if fmt != "mm":
                    st.caption("Setor agregado = soma do numerador ÷ soma do denominador das empresas com os dois dados "
                               "(ex.: margem EBITDA = Σ EBITDA ÷ Σ receita).")

                exp_t = tab_full.reset_index()
                exp_todos[f"{len(exp_todos) + 1}. {ROT[ind]}"] = exp_t
                c1, c2 = st.columns(2)
                c1.download_button("⬇️ Tabela (CSV)", exp_t.to_csv(index=False, sep=";", decimal=",").encode("utf-8-sig"),
                                   file_name=f"setor_{ind}_{tipo}.csv", mime="text/csv", key=f"set_csv_{ind}")
                c2.download_button("⬇️ Tabela (XLSX)", excel_bytes({ROT[ind][:31]: exp_t}), file_name=f"setor_{ind}_{tipo}.xlsx",
                                   mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet", key=f"set_xlsx_{ind}")
            with area_g:
                if len(inds) > 1:
                    st.divider()
                    st.subheader(rot_ind(ind))
                # gráfico
                if fmt == "mm":
                    top = tab.index[:10] if not sel_emp else tab.index
                    plot = tab.loc[top].T if len(tab) else pd.DataFrame()
                    if not sel_emp and len(tab) > 10:
                        plot["Demais"] = tab.drop(index=top).sum(min_count=1)
                    elif sel_emp:
                        plot["Demais empresas do setor"] = linhas_extra["Demais empresas do setor"]
                    longo = plot.reset_index(names="Período").melt(id_vars="Período", var_name="Empresa", value_name="v")
                    fig = px.bar(longo, x="Período", y="v", color="Empresa",
                                 labels={"v": "% do setor" if pct_setor else ROT[ind] + f" ({UN})"})
                    fig.update_layout(barmode="relative", height=480, legend_title_text="")
                    if pct_setor:
                        fig.update_yaxes(tickformat=".0%")
                    st.plotly_chart(fig, width="stretch", key=f"set_fig_{ind}")
                    if not sel_emp and len(tab) > 10:
                        st.caption("Gráfico mostra as 10 maiores do último período e agrupa o restante em 'Demais'. "
                                   "Selecione empresas na barra lateral para vê-las isoladamente.")
                elif ind.startswith("cresc_"):
                    # crescimento acumulado, base 100 no primeiro período: por setor (sem recorte) ou por empresa (com recorte)
                    col = ind[6:]
                    niv = d_set.pivot_table(index="rotulo", columns="_c", values=col, aggfunc="first").reindex(columns=periodos)
                    por_setor = not (sel_setor or sel_sub or sel_seg or sel_emp)
                    linhas_idx = {}
                    if por_setor:
                        mapa_setor = d_set.drop_duplicates("rotulo").set_index("rotulo")["setor"]
                        for st_nome, rots in mapa_setor.groupby(mapa_setor).groups.items():
                            if str(st_nome).strip():
                                linhas_idx[st_nome] = indice_acumulado(niv.loc[niv.index.isin(rots)])
                    else:
                        vis_rot = tab.index if sel_emp else niv.index
                        for r in vis_rot:
                            v = niv.loc[r]
                            ini = v.first_valid_index()
                            if ini is None or v[ini] <= 0:
                                continue
                            idx = 100 * v / v[ini]
                            neg = idx[idx <= 0].index
                            if len(neg):           # base negativa no meio da série: o índice deixa de ter leitura
                                idx[idx.index >= neg[0]] = np.nan
                            linhas_idx[r] = idx
                        linhas_idx["Agregado do filtro"] = indice_acumulado(niv)
                    nomes = list(linhas_idx)
                    if por_setor:
                        padrao_l = nomes
                    else:
                        ult_p = ult[0] if ult else periodos[-1]
                        tam = niv[ult_p].sort_values(ascending=False) if ult_p in niv.columns else pd.Series(dtype=float)
                        padrao_l = [r for r in tam.index[:12] if r in linhas_idx] + ["Agregado do filtro"]
                    chave_l = f"cres_linhas_{ind}_{tipo}_" + str(abs(hash((tuple(sel_setor), tuple(sel_sub), tuple(sel_seg),
                                                                          tuple(sel_emp)))))
                    sel_l = st.multiselect("Linhas no gráfico", nomes, default=padrao_l, key=chave_l,
                                           help="Também dá para ocultar uma linha clicando no nome dela na legenda.")
                    fig = go.Figure()
                    for nome in sel_l:
                        y = linhas_idx[nome]
                        fig.add_trace(go.Scatter(x=list(y.index), y=y.values, mode="lines+markers",
                                                 name=nome.split(" — ")[0] if " — " in nome else nome,
                                                 line=dict(width=4, dash="dash") if nome == "Agregado do filtro" else None))
                    if not EUA:      # macro Brasil no mesmo recorte, base 100 no primeiro período
                        macro_aba = macro_mod()
                        macro_aba.comparar_base100(fig, d_set.groupby("_c")["dt_fim"].max().reindex(periodos),
                                                   3 if tipo == "trimestral" else 12,
                                                   col in ("ativo_total", "patrimonio_liquido"),
                                                   key=f"mac_cmp_set_{ind}_{tipo}")
                    fig.add_hline(y=100, line_dash="dot", line_color="gray")
                    fig.update_layout(height=480, legend_title_text="", yaxis_title="Base 100 = primeiro período",
                                      title=f"{ROT[col] if col in ROT else col}: crescimento acumulado (base 100)")
                    st.plotly_chart(fig, width="stretch", key=f"set_fig_{ind}")
                    st.caption(("Uma linha por setor: soma das empresas do setor, encadeando a variação entre períodos "
                                "consecutivos só com empresas presentes nos dois períodos (amostra casada)." if por_setor else
                                "Uma linha por empresa: valor ÷ valor do primeiro período × 100; a linha é interrompida se o "
                                "valor fica zero ou negativo. 'Agregado do filtro' encadeia a soma das empresas em amostra casada.")
                               + " A tabela acima mostra o crescimento de cada período contra o mesmo período do ano anterior.")
                elif ind in ("gao", "gaf", "gat") and not (sel_setor or sel_sub or sel_seg or sel_emp):
                    # sem recorte: uma linha por setor, com o grau calculado sobre os agregados do setor
                    linhas_g = {}
                    for st_nome in sorted(x for x in setor_f["setor"].dropna().unique() if str(x).strip()):
                        linhas_g[st_nome] = agregado_setor(setor_f[setor_f["setor"] == st_nome], ind).reindex(periodos)
                    linhas_g["Todas as empresas"] = total
                    sel_g = st.multiselect("Linhas no gráfico", list(linhas_g), default=list(linhas_g),
                                           key=f"graus_linhas_{ind}_{tipo}",
                                           help="Também dá para ocultar uma linha clicando no nome dela na legenda.")
                    fig = go.Figure()
                    for nome in sel_g:
                        y = linhas_g[nome]
                        fig.add_trace(go.Scatter(x=periodos, y=y.values, mode="lines+markers", name=nome,
                                                 line=dict(width=4, dash="dash") if nome == "Todas as empresas" else None))
                    fig.add_hline(y=1, line_dash="dot", line_color="gray")
                    fig.update_layout(height=480, legend_title_text="", yaxis_ticksuffix="x",
                                      title=f"{ROT[ind]} por setor")
                    st.plotly_chart(fig, width="stretch", key=f"set_fig_{ind}")
                    st.caption("Grau de cada setor calculado sobre os agregados do setor (variação da soma de EBIT, lucro e "
                               "receita, só com empresas presentes nos dois períodos). Vazio quando o denominador variou menos "
                               "de 1%; valores muito altos indicam variação pequena no denominador, não alavancagem alta.")
                else:
                    fig = go.Figure()
                    for r in tab.index[:10]:
                        fig.add_trace(go.Scatter(x=periodos, y=tab.loc[r], mode="lines+markers", name=r.split(" — ")[0]))
                    fig.add_trace(go.Scatter(x=periodos, y=total, mode="lines", name="Setor agregado", line=dict(width=4, dash="dash")))
                    fig.add_trace(go.Scatter(x=periodos, y=mediana, mode="lines", name="Mediana", line=dict(dash="dot")))
                    fig.update_layout(height=480, legend_title_text="")
                    if fmt == "pct":
                        fig.update_yaxes(tickformat=".0%")
                    st.plotly_chart(fig, width="stretch", key=f"set_fig_{ind}")

        with area_t:
            if len(inds) > 1:
                st.divider()
                st.download_button("⬇️ Todos os indicadores selecionados (XLSX, uma aba por indicador)",
                                   excel_bytes(exp_todos), file_name=f"setor_{len(inds)}_indicadores_{tipo}.xlsx",
                                   mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
                                   key="set_xlsx_todos")

        with area_g:
            with st.expander("Dispersão entre dois indicadores (um período)"):
                ref_opts = periodos[::-1]
                ref = st.selectbox("Período", ref_opts, key="disp_per")
                snap = filtrado[col_periodo(filtrado) == ref]
                c1, c2, c3 = st.columns(3)
                x = c1.selectbox("Eixo X", list(METRICAS), index=list(METRICAS).index("div_liq_ebitda_ifrs16"), format_func=lambda k: ROT[k])
                y = c2.selectbox("Eixo Y", list(METRICAS), index=list(METRICAS).index("margem_ebitda"), format_func=lambda k: ROT[k])
                tam = c3.selectbox("Tamanho do ponto", [None] + [k for k in METRICAS if METRICAS[k][1] == "mm"],
                                   format_func=lambda k: "—" if k is None else ROT[k])
                pts = snap.dropna(subset=[x, y])
                if tam:
                    pts = pts[pts[tam] > 0]
                if pts.empty:
                    st.info("Sem dados suficientes para os eixos escolhidos.")
                else:
                    fig = px.scatter(pts, x=x, y=y, color="segmento", size=tam, hover_name="empresa",
                                     hover_data={"ticker": True, x: ":.2f", y: ":.2f"},
                                     labels={x: ROT[x], y: ROT[y]})
                    for eixo, k in (("x", x), ("y", y)):
                        if METRICAS[k][1] == "pct":
                            fig.update_layout(**{f"{eixo}axis_tickformat": ".0%"})
                    fig.update_layout(height=520, legend_title_text="Segmento")
                    st.plotly_chart(fig, width="stretch")

# -------- Empresa: séries históricas vs. mediana do segmento
with aba_emp:
    grupo_e = (st.radio("Grupo de empresas", ["Não financeiras", "Bancos e seguradoras"], horizontal=True,
                        key="grupo_emp") if not fin.empty else "Não financeiras")
    if grupo_e == "Bancos e seguradoras":
        ctx_f = empresa_financeira()
        if ctx_f:
            extras_financeira(*ctx_f)
    elif filtrado.empty:
        st.info("Nenhum dado para o filtro.")
    else:
        opts_emp = sorted(filtrado["rotulo"].unique())
        alvo_e = st.session_state.get("emp_sel")
        if alvo_e and alvo_e not in opts_emp and alvo_e in set(df_t["rotulo"]):
            opts_emp = sorted(set(opts_emp) | {alvo_e})      # empresa clicada fora do filtro da barra lateral
        emp = st.selectbox("Empresa", opts_emp, key="emp_sel")
        fonte_e = filtrado if emp in set(filtrado["rotulo"]) else df_t
        d_emp = fonte_e[fonte_e["rotulo"] == emp].sort_values(eixo_x(tipo))
        seg = d_emp["segmento"].iloc[0]
        st.caption(f"{d_emp['setor'].iloc[0]} / {d_emp['subsetor'].iloc[0]} / **{seg}** · base {d_emp['fonte'].iloc[-1]}")
        linha_e = d_emp.iloc[-1]
        cc1, cc2 = st.columns([1, 2])
        nivel_cmp = cc1.radio("Comparar com (mediana)", list(NIVEIS_CMP) + ["Cesta de empresas"], horizontal=True,
                              key="cmp_nivel")
        cesta = []
        if nivel_cmp == "Cesta de empresas":
            cesta = cc2.multiselect("Empresas da cesta (qualquer empresa da base)", sorted(base["rotulo"].unique()),
                                    default=[emp], key="cmp_cesta")
            nome_cmp = f"da cesta ({len(cesta)})"
        else:
            nome_cmp = str(linha_e[NIVEIS_CMP[nivel_cmp]])
        cmp = lambda df: grupo_comparacao(df, linha_e, nivel_cmp, cesta)
        n_cmp = cmp(base[base["tipo"] == tipo])["cnpj"].nunique()
        if nivel_cmp == "Cesta de empresas":
            cc2.caption(f"Mediana calculada sobre {n_cmp} empresa(s) da cesta.")
        else:
            cc2.caption(f"Mediana calculada sobre {n_cmp} empresa(s): {nome_cmp}.")
        relativas = [k for k in METRICAS if METRICAS[k][1] in ("pct", "x", "d")]
        sel_rel = st.multiselect("Indicadores relativos", relativas, default=RELATIVAS_PADRAO,
                                 format_func=lambda k: ROT[k], key="rel_emp")
        pares = cmp(base[base["tipo"] == tipo])
        med_seg = pares.groupby(eixo_x(tipo))[sel_rel].median(numeric_only=True).reset_index()
        cols = st.columns(2)
        for i, m in enumerate(sel_rel):
            fig = go.Figure()
            fig.add_trace(go.Scatter(x=d_emp[eixo_x(tipo)], y=d_emp[m], mode="lines+markers", name=emp.split(" — ")[0]))
            mm = med_seg[med_seg[eixo_x(tipo)].isin(d_emp[eixo_x(tipo)])]
            fig.add_trace(go.Scatter(x=mm[eixo_x(tipo)], y=mm[m], mode="lines", name=f"Mediana {nome_cmp}",
                                     line=dict(dash="dash")))
            add_medias(fig, d_emp[eixo_x(tipo)], d_emp[m], mm[m])
            alertas = d_emp[d_emp["alerta"]]
            if len(alertas):
                fig.add_trace(go.Scatter(x=alertas[eixo_x(tipo)], y=alertas[m], mode="markers", name="Alerta",
                                         marker=dict(symbol="x", size=11, color="red")))
            unidade = {"mm": f" ({UN})", "pct": "", "x": " (x)", "d": " (dias)"}[METRICAS[m][1]]
            fig.update_layout(title=ROT[m] + unidade, height=320, margin=dict(t=40, b=20, l=10, r=10),
                              legend=dict(orientation="h", y=-0.2))
            if METRICAS[m][1] == "pct":
                fig.update_yaxes(tickformat=".0%")
            cols[i % 2].plotly_chart(fig, width="stretch")
        # ---- conversão do EBITDA em caixa
        st.subheader("Conversão do EBITDA em caixa (FCO ÷ EBITDA)")
        ex_ = eixo_x(tipo)
        if d_emp["conversao_fco_ebitda"].notna().any():
            med_cv = pares.groupby(ex_)["conversao_fco_ebitda"].median().reindex(d_emp[ex_])
            k1, k2 = st.columns(2)
            fig = go.Figure()
            fig.add_trace(go.Bar(x=d_emp[ex_], y=d_emp["ebitda"], name="EBITDA US GAAP" if EUA else "EBITDA IFRS 16",
                                 opacity=0.6))
            fig.add_trace(go.Bar(x=d_emp[ex_], y=d_emp["fco"], name="FCO reportado" if EUA else "FCO comparável",
                                 opacity=0.9))
            if d_emp["fco_reportado"].notna().any() and (d_emp["fco_reportado"] - d_emp["fco"]).abs().max() > 0.05:
                fig.add_trace(go.Bar(x=d_emp[ex_], y=d_emp["fco_reportado"], name="FCO reportado", opacity=0.4))
            fig.update_layout(title=f"EBITDA e FCO ({UN})", barmode="group", height=360,
                              legend=dict(orientation="h", y=-0.25), margin=dict(t=40, b=20, l=10, r=10))
            k1.plotly_chart(fig, width="stretch")
            fig = go.Figure()
            fig.add_trace(go.Scatter(x=d_emp[ex_], y=d_emp["conversao_fco_ebitda"], mode="lines+markers",
                                     name="FCO ÷ EBITDA"))
            fig.add_trace(go.Scatter(x=d_emp[ex_], y=med_cv.values, mode="lines", line=dict(dash="dash"),
                                     name=f"Mediana {nome_cmp}"))
            add_medias(fig, d_emp[ex_], d_emp["conversao_fco_ebitda"], med_cv.values)
            fig.add_hline(y=1, line_color="gray", line_width=1, line_dash="dot")
            fig.update_layout(title="Conversão de caixa (FCO ÷ EBITDA)", height=360, yaxis_tickformat=".0%",
                              legend=dict(orientation="h", y=-0.25), margin=dict(t=40, b=20, l=10, r=10))
            k2.plotly_chart(fig, width="stretch")
            st.caption("FCO como reportado ÷ EBITDA US GAAP do mesmo período (vazio com EBITDA ≤ 0). No US GAAP o FCO "
                       "já desconta juros pagos, imposto pago e aluguel de arrendamento operacional, então a conversão "
                       "tende a ficar abaixo da brasileira. Abaixo de 100%: parte do EBITDA ficou em capital de giro, "
                       "juros ou imposto." if EUA else
                       "FCO comparável ÷ EBITDA IFRS 16 do mesmo período (vazio com EBITDA ≤ 0). O FCO comparável tira "
                       "do operacional os juros e dividendos/JCP pagos ou recebidos que a empresa classifica ali. "
                       "Abaixo de 100%: parte do EBITDA ficou em capital de giro ou IR/CSLL pagos. "
                       "No trimestral, a sazonalidade do capital de giro pesa; LTM e anual são mais estáveis.")
        else:
            st.info("Sem FCO ou EBITDA positivo para calcular a conversão de caixa.")
        # ---- prazos médios e ciclos
        st.subheader("Prazos médios e ciclos")
        if d_emp[["pmr", "pme", "pmp"]].notna().any().any():
            ex_ = eixo_x(tipo)
            med_pz = pares.groupby(ex_)[["pmr", "pme", "pmp", "ciclo_operacional", "ciclo_financeiro"]].median()
            med_pz = med_pz.reindex(d_emp[ex_])
            p1, p2 = st.columns(2)
            fig = go.Figure()
            for k, nome in (("pmr", "PMR (recebimento)"), ("pme", "PME (estocagem/venda)"), ("pmp", "PMP (pagamento)")):
                fig.add_trace(go.Scatter(x=d_emp[ex_], y=d_emp[k], mode="lines+markers", name=nome))
            fig.update_layout(title="Prazos médios (dias)", height=360, legend=dict(orientation="h", y=-0.25),
                              margin=dict(t=40, b=20, l=10, r=10))
            p1.plotly_chart(fig, width="stretch")
            fig = go.Figure()
            fig.add_trace(go.Bar(x=d_emp[ex_], y=d_emp["ciclo_operacional"], name="Ciclo operacional", opacity=0.6))
            fig.add_trace(go.Scatter(x=d_emp[ex_], y=d_emp["ciclo_financeiro"], mode="lines+markers",
                                     name="Ciclo financeiro", line=dict(width=3)))
            fig.add_trace(go.Scatter(x=d_emp[ex_], y=med_pz["ciclo_financeiro"].values, mode="lines",
                                     line=dict(dash="dash"), name=f"Mediana ciclo financeiro {nome_cmp}"))
            add_medias(fig, d_emp[ex_], d_emp["ciclo_financeiro"], med_pz["ciclo_financeiro"].values, "ciclo financeiro")
            fig.add_hline(y=0, line_color="gray", line_width=1)
            fig.update_layout(title="Ciclo operacional e ciclo financeiro (dias)", height=360,
                              legend=dict(orientation="h", y=-0.25), margin=dict(t=40, b=20, l=10, r=10))
            p2.plotly_chart(fig, width="stretch")
            st.caption("Dias: 365 no anual e LTM; 90 no trimestral. Saldos médios (inicial e final) de clientes, estoques "
                       "e fornecedores, circulante + não circulante. Compras = estoque final − estoque inicial + CPV. "
                       "Ciclo operacional = PME + PMR; ciclo financeiro = ciclo operacional − PMP (negativo = fornecedores "
                       "financiam o giro). Os três prazos também estão na lista de indicadores relativos, com mediana.")
        else:
            st.info("Prazos médios ainda não disponíveis para esta empresa (exigem a coleta com clientes, estoques, "
                    "fornecedores e CPV).")
        if tipo == "trimestral":
            st.caption("No trimestral, giro do ativo usa a receita do trimestre (não anualizada); ROE e alavancagem "
                       "sobre EBITDA ficam vazios. Para leitura anualizada, use LTM.")
        notas = d_emp[d_emp["notas"] != ""][["periodo", "fonte", "notas"]]
        with st.expander(f"Notas e alertas ({len(notas)} períodos)"):
            st.dataframe(notas, hide_index=True, width="stretch")

        secao_crescimento_dupont(emp, cmp, nome_cmp)

        # ---- múltiplos de mercado (valor de mercado com preços da B3 sem ajuste)
        st.subheader("Múltiplos de mercado")
        if mult.empty:
            st.info("Múltiplos ainda não gerados (multiplos_eua.parquet no coletor-eua)." if EUA else
                    "Múltiplos ainda não gerados (multiplos.csv.gz). Rode o workflow no modo universo.")
        else:
            tipo_m = "anual" if tipo == "anual" else "trimestral"
            cnpj_e = d_emp["cnpj"].iloc[0]
            ini = d_emp["dt_fim"].min()
            dm = mult[(mult["cnpj"] == cnpj_e) & (mult["tipo"] == tipo_m) & (mult["data"] >= ini)].sort_values("data")
            if dm.empty:
                st.info("Sem múltiplos para esta empresa (sem cotação ou sem composição do capital).")
            else:
                painel_mercado(cnpj_e)
                st.caption("Anual: preço do último pregão do exercício com os números do exercício. "
                           if tipo_m == "anual" else
                           "Trimestral (ponto no tempo): em cada data, PL, dívida e ações do último ITR/DFP divulgado até a "
                           "data; lucro e EBITDA dos 12 meses desse documento. O último ponto é o pregão mais recente. ")
                pares_m = cmp(mult[mult["tipo"] == tipo_m])
                cols_med = ["p_vpa", "p_l", "ev_ebitda_ifrs16", "ev_ebitda_pre_ifrs16", "roe", "dividend_yield",
                            "ev_receita_com_arrend", "ev_receita_sem_arrend"]
                if EUA:
                    # anos fiscais diferentes entre empresas: mediana pelo ano/trimestre CIVIL da data
                    # (o ponto do último pregão, comum a todas, fica à parte)
                    ult_m = mult["data"].max()
                    chave_m = lambda s: pd.Series(np.where(s == ult_m, "ultimo", s.dt.to_period(
                        "Y" if tipo_m == "anual" else "Q").astype(str)), index=s.index)
                    med_k = pares_m.assign(_k=chave_m(pares_m["data"])).groupby("_k")[cols_med].median()
                    med_m = dm[["data"]].assign(_k=chave_m(dm["data"])).join(med_k, on="_k") \
                        .drop_duplicates("data").set_index("data")[cols_med]
                else:
                    med_m = pares_m.groupby("data")[cols_med].median()
                    med_m = med_m[med_m.index.isin(dm["data"])]
                graficos = [("P/VPA (valor de mercado ÷ PL dos controladores)", [("p_vpa", "P/VPA")]),
                            ("P/L (valor de mercado ÷ lucro 12m dos controladores)", [("p_l", "P/L")]),
                            ("EV/EBITDA IFRS 16 (EV c/ arrendamento ÷ EBITDA IFRS 16)",
                             [("ev_ebitda_ifrs16", "EV/EBITDA IFRS 16")]),
                            ("EV/EBITDA pré-IFRS 16 (EV s/ arrendamento ÷ EBITDA pré-IFRS 16)",
                             [("ev_ebitda_pre_ifrs16", "EV/EBITDA pré-IFRS 16")]),
                            ("EV/Receita líquida (EV c/ arrendamento ÷ receita 12m)",
                             [("ev_receita_com_arrend", "EV/Receita c/ arrendamento")]),
                            ("EV/Receita líquida (EV s/ arrendamento ÷ receita 12m)",
                             [("ev_receita_sem_arrend", "EV/Receita s/ arrendamento")]),
                            ("ROE (lucro 12m ÷ PL dos controladores)", [("roe", "ROE")]),
                            ("Dividend yield (dividendos + JCP pagos em 12m ÷ valor de mercado)",
                             [("dividend_yield", "Dividend yield")])]
                cm = [c for _ in range((len(graficos) + 1) // 2) for c in st.columns(2)]
                for (tit, series), col in zip(graficos, cm):
                    fig = go.Figure()
                    for k, nome in series:
                        fig.add_trace(go.Scatter(x=dm["data"], y=dm[k], mode="lines+markers", name=nome))
                        fig.add_trace(go.Scatter(x=med_m.index, y=med_m[k], mode="lines", line=dict(dash="dash"),
                                                 name=f"Mediana {nome_cmp}" + ("" if len(series) == 1 else f" ({nome.split(' ')[0]})")))
                        add_medias(fig, dm["data"], dm[k], med_m[k], "" if len(series) == 1 else nome.split(" ")[0])
                    com_nota = dm[dm["notas"] != ""]
                    if len(com_nota):
                        fig.add_trace(go.Scatter(x=com_nota["data"], y=com_nota[series[0][0]], mode="markers",
                                                 name="Nota no cálculo", marker=dict(symbol="circle-open", size=12, color="orange")))
                    fig.update_layout(title=tit, height=340, margin=dict(t=40, b=20, l=10, r=10),
                                      legend=dict(orientation="h", y=-0.25))
                    if series[0][0] in ("roe", "dividend_yield"):
                        fig.update_yaxes(tickformat=".0%")
                    else:
                        fig.update_yaxes(ticksuffix="x")
                    col.plotly_chart(fig, width="stretch")
                st.caption("Vazio = denominador ≤ 0 (prejuízo, EBITDA ou PL negativos) ou sem cotação. "
                           "EV = valor de mercado + dívida líquida + participação de não controladores. "
                           "ROE usa o mesmo lucro 12m e PL dos múltiplos, então P/VPA = P/L × ROE em cada ponto. "
                           + ("Dividend yield em regime de caixa: dividendos pagos nos 12 meses (fluxo de caixa). Preços "
                              "do Yahoo Finance, sem o ajuste de desdobramentos (preço negociado na data)." if EUA else
                              "Dividend yield em regime de caixa: dividendos e JCP efetivamente pagos nos 12 meses (DFC), "
                              "não pela data-com; exclui pagamentos a não controladores."))
                with st.expander("Valor de mercado, EV, preços, ações e notas"):
                    cols_m = ["data", "periodo_base", "valor_mercado", "ev_com_arrend", "ev_sem_arrend", "ticker_on", "preco_on",
                              "data_preco_on", "acoes_on", "ticker_pn", "preco_pn", "data_preco_pn", "acoes_pn",
                              "p_vpa", "p_l", "ev_ebitda_ifrs16", "ev_ebitda_pre_ifrs16", "ev_receita_com_arrend",
                              "ev_receita_sem_arrend", "roe", "dividendos_12m", "dividend_yield", "notas"]
                    tab_m = dm[cols_m].assign(data=dm["data"].dt.date)
                    st.dataframe(tab_m, hide_index=True, width="stretch", column_config={
                        "valor_mercado": st.column_config.NumberColumn(f"Valor de mercado ({UN})", format="%,.1f"),
                        "ev_com_arrend": st.column_config.NumberColumn(f"EV c/ arrend. ({UN})", format="%,.1f"),
                        "ev_sem_arrend": st.column_config.NumberColumn(f"EV s/ arrend. ({UN})", format="%,.1f"),
                        "acoes_on": st.column_config.NumberColumn("Ações (todas as classes)" if EUA else "Ações ON",
                                                                  format="%,.0f"),
                        "acoes_pn": st.column_config.NumberColumn("Ações PN", format="%,.0f"),
                        **{k: st.column_config.NumberColumn(k, format="%.2fx") for k in
                           ("p_vpa", "p_l", "ev_ebitda_ifrs16", "ev_ebitda_pre_ifrs16", "ev_receita_com_arrend",
                            "ev_receita_sem_arrend")},
                        "roe": st.column_config.NumberColumn("ROE", format="percent"),
                        "dividendos_12m": st.column_config.NumberColumn(("Dividendos 12m" if EUA else "Dividendos + JCP 12m") + f" ({UN})",
                                                                        format="%,.1f"),
                        "dividend_yield": st.column_config.NumberColumn("Dividend yield", format="percent")})
                    st.download_button("⬇️ Múltiplos (CSV)", tab_m.to_csv(index=False, sep=";", decimal=",").encode("utf-8-sig"),
                                       file_name=f"multiplos_{emp.split(' — ')[0]}_{tipo_m}.csv", mime="text/csv", key="mult_csv")

        # ---- cotação diária (ajustável por proventos e eventos, nominal ou real), valor de mercado e EV
        _cn_e = str(linha_e["cnpj"]).zfill(14)
        _tk_e = str(linha_e.get("ticker") or "")
        _mm = mercado_mod()
        if EUA:
            _vme = mult[(mult["cnpj"] == _cn_e) & (mult["tipo"] == "trimestral")][
                ["data", "valor_mercado", "ev_com_arrend", "ev_sem_arrend"]] if not mult.empty else pd.DataFrame()
            _mm.secao_cotacao("", _cn_e, emp, eua=True, ticker_eua=_tk_e, vm_ev_eua=_vme, moeda=MOEDA, chave="cot_e")
        else:
            _mm.secao_cotacao(_tk_e[:4], _cn_e, emp, moeda=MOEDA, chave="cot_e")

        if not EUA:          # debêntures (SND/ANBIMA) e aluguel de ações (B3): só Brasil
            # ---- dívida em debêntures (mercado de capitais) vs. dívida do balanço
            st.subheader("Dívida em debêntures × dívida do balanço")
            dvs = carregar_dividas(versao_arquivos(DADOS.parent / "dividas" / "debentures.csv.gz"))
            cn_e = str(linha_e["cnpj"]).zfill(14)
            dd = dvs[dvs["cnpj_grupo"] == cn_e].copy() if not dvs.empty else pd.DataFrame()
            ult_bal = base[(base["cnpj"] == cn_e) & base["tipo"].isin(["trimestral", "anual"])].sort_values("dt_fim").tail(1)
            if dvs.empty:
                st.info("Base de debêntures ainda não coletada (saidas/dividas/). Rode o workflow dividas-debentures.")
            elif dd.empty:
                st.info("Nenhuma debênture ativa encontrada para a empresa ou suas controladas no SND.")
            else:
                dt_bal = ult_bal["dt_fim"].iloc[0] if len(ult_bal) else pd.NaT
                div_bal = ult_bal["divida_bruta_ex_arrend"].iloc[0] if len(ult_bal) else np.nan
                dd["remuneracao"] = dd.apply(rotulo_remuneracao, axis=1)
                dd["premio_mercado"] = dd["taxa_indicativa"] - dd["taxa_emissao"].where(dd["taxa_emissao"] != 0)
                pj_c, _ = carregar_juros(versao_arquivos(DADOS.parent / "juros" / "parametros.csv",
                                                         DADOS.parent / "juros" / "vertices.csv.gz"))
                dd, dt_curva = custo_debentures(dd, pj_c)
                na_data = dd[dd["data_emissao"] <= dt_bal] if pd.notna(dt_bal) else dd
                tot_deb = na_data["saldo_rs_mm"].sum()
                w = dd["saldo_rs_mm"].where(dd["duration_anos"].notna())
                dur = (dd["duration_anos"] * w).sum() / w.sum() if w.sum() > 0 else np.nan
                m1, m2, m3, m4 = st.columns(4)
                m1.metric(f"Dívida bruta no balanço ({dt_bal.strftime('%d/%m/%Y') if pd.notna(dt_bal) else '-'})",
                          f"R$ {div_bal:,.0f} mi".replace(",", ".") if pd.notna(div_bal) else "-")
                m2.metric("Debêntures (saldo estimado)", f"R$ {tot_deb:,.0f} mi".replace(",", "."),
                          f"{tot_deb / div_bal:.0%} da dívida" if pd.notna(div_bal) and div_bal > 0 else None,
                          delta_color="off")
                dif = div_bal - tot_deb if pd.notna(div_bal) else np.nan
                m3.metric("Dívida comercial/bancária (diferença)", f"R$ {dif:,.0f} mi".replace(",", ".") if pd.notna(dif) else "-",
                          f"{dif / div_bal:.0%} da dívida" if pd.notna(dif) and div_bal > 0 else None, delta_color="off")
                m4.metric("Duration média ponderada", f"{dur:.1f} anos" if pd.notna(dur) else "-")
                if pd.notna(dif) and dif < 0:
                    st.caption("⚠️ Debêntures acima da dívida do balanço: pode haver debênture de controlada não consolidada, "
                               "amortização recente não refletida no SND, ou diferença entre saldo estimado e valor contábil.")
                cols_d = ["codigo", "emissor", "via_controlada", "data_emissao", "vencimento", "remuneracao", "taxa_indicativa",
                          "premio_mercado", "soberano", "premio_soberano", "pct_pu_par", "duration_anos", "saldo_rs_mm",
                          "incentivada"]
                tab_d = dd[[c for c in cols_d if c in dd.columns]].sort_values("vencimento").assign(
                    data_emissao=lambda x: x["data_emissao"].dt.date, vencimento=lambda x: x["vencimento"].dt.date)
                st.dataframe(tab_d, hide_index=True, width="stretch", column_config={
                    "codigo": "Código", "emissor": "Emissor", "via_controlada": st.column_config.CheckboxColumn("Controlada"),
                    "data_emissao": "Emissão", "vencimento": "Vencimento", "remuneracao": "Remuneração (emissão)",
                    "taxa_indicativa": st.column_config.NumberColumn("Taxa indicativa ANBIMA", format="%.2f"),
                    "premio_mercado": st.column_config.NumberColumn("Mercado − emissão (p.p.)", format="%+.2f"),
                    "soberano": st.column_config.NumberColumn("Soberano no prazo (% a.a.)", format="%.2f"),
                    "premio_soberano": st.column_config.NumberColumn("Prêmio s/ soberano ou CDI (p.p.)",
                                                                     format="%+.2f"),
                    "pct_pu_par": st.column_config.NumberColumn("% PU par", format="%.1f"),
                    "duration_anos": st.column_config.NumberColumn("Duration (anos)", format="%.1f"),
                    "saldo_rs_mm": st.column_config.NumberColumn("Saldo (R$ mm)", format="%,.1f"),
                    "incentivada": st.column_config.CheckboxColumn("Incentivada (12.431)")})
                g1, g2 = st.columns(2)
                venc = dd.assign(ano=dd["vencimento"].dt.year).groupby(["ano", "indice"])["saldo_rs_mm"].sum().reset_index()
                fig = px.bar(venc, x="ano", y="saldo_rs_mm", color="indice", labels={"saldo_rs_mm": "R$ mm", "ano": "",
                                                                                    "indice": "Indexador"})
                fig.update_layout(title="Cronograma de vencimentos (saldo atual)", height=340, margin=dict(t=40, b=20, l=10, r=10))
                g1.plotly_chart(fig, width="stretch")
                comp = pd.DataFrame({"Origem": ["Debêntures", "Comercial/bancária"],
                                     "R$ mm": [tot_deb, max(dif, 0) if pd.notna(dif) else np.nan]})
                fig = px.pie(comp, names="Origem", values="R$ mm", hole=0.5)
                fig.update_layout(title="Composição da dívida bruta", height=340, margin=dict(t=40, b=20, l=10, r=10))
                g2.plotly_chart(fig, width="stretch")
                # ---- custo médio das debêntures por indexador e prêmio sobre o soberano
                st.markdown("##### Custo médio das debêntures por indexador")
                res_c = resumo_custo(dd)
                cust_impl, dt_impl = custo_implicito_caixa(base, cn_e)
                tot_c = res_c[res_c["Indexador"].str.startswith("Total")]
                ipca_c = res_c[res_c["Indexador"] == "IPCA+"]
                f2 = lambda v, suf=" % a.a.": f"{v:.2f}".replace(".", ",") + suf if pd.notna(v) else "-"
                k1, k2, k3, k4 = st.columns(4)
                k1.metric("Custo nominal contratado (debêntures)",
                          f2(tot_c["Custo nominal contratado (% a.a.)"].iloc[0]) if len(tot_c) else "-",
                          help="Remuneração de emissão convertida em taxa nominal com a curva de hoje, ponderada pelo "
                               "saldo. É o custo da dívida existente em debêntures.")
                k2.metric("Custo nominal a mercado (debêntures)",
                          f2(tot_c["Custo nominal a mercado (% a.a.)"].iloc[0]) if len(tot_c) else "-",
                          help="Taxa indicativa ANBIMA convertida em taxa nominal: quanto custaria refinanciar hoje.")
                k3.metric("Prêmio IPCA+ sobre a NTN-B (mercado)",
                          f2(ipca_c["Prêmio de mercado (p.p.)"].iloc[0], " p.p.") if len(ipca_c) else "-",
                          help="Taxa indicativa ANBIMA − ETTJ real no mesmo prazo (duration), ponderado pelo saldo.")
                k4.metric("Custo implícito caixa da dívida toda" + (f" (LTM {dt_impl:%m/%Y})" if dt_impl is not None else ""),
                          f2(cust_impl),
                          help="Juros pagos da dívida em 12 meses (DFC, ex-arrendamento) ÷ dívida bruta média. Inclui "
                               "a dívida bancária, mas é regime de caixa (ver nota abaixo).")
                if res_c.empty:
                    st.info("Sem debêntures para calcular o custo.")
                else:
                    st.dataframe(res_c, hide_index=True, width="stretch", column_config={
                        "Saldo (R$ mm)": st.column_config.NumberColumn(format="%,.1f"),
                        "% das debêntures": st.column_config.NumberColumn(format="percent"),
                        "% do saldo com preço ANBIMA": st.column_config.NumberColumn(format="percent"),
                        **{c: st.column_config.NumberColumn(format="%.2f") for c in
                           ("Prazo médio (anos)", "Custo contratado (% a.a.)", "Taxa de mercado (% a.a.)",
                            "Soberano no prazo (% a.a.)", "Custo nominal contratado (% a.a.)",
                            "Custo nominal a mercado (% a.a.)")},
                        **{c: st.column_config.NumberColumn(format="%+.2f") for c in
                           ("Prêmio de mercado (p.p.)", "Contratado − soberano hoje (p.p.)")}})
                    g3, g4 = st.columns(2)
                    sc = dd[dd["classe"].isin(["IPCA", "PRÉ"]) & dd["premio_soberano"].notna()]
                    if len(sc):
                        fig = px.scatter(sc, x="prazo_anos", y="premio_soberano", color="classe", size="saldo_rs_mm",
                                         hover_name="codigo",
                                         hover_data={"taxa_mercado": ":.2f", "soberano": ":.2f", "saldo_rs_mm": ":,.0f",
                                                     "incentivada": True},
                                         labels={"prazo_anos": "Duration (anos)", "classe": "Indexador",
                                                 "premio_soberano": "Prêmio sobre o soberano (p.p.)",
                                                 "taxa_mercado": "Taxa ANBIMA", "soberano": "Soberano",
                                                 "saldo_rs_mm": "Saldo (R$ mm)", "incentivada": "Incentivada"})
                        fig.add_hline(y=0, line_dash="dot", line_color="gray")
                        fig.update_layout(title="Prêmio de mercado sobre o soberano × prazo", height=340,
                                          margin=dict(t=40, b=20, l=10, r=10))
                        g3.plotly_chart(fig, width="stretch")
                    else:
                        g3.info("Sem debêntures IPCA+ ou prefixadas com preço ANBIMA.")
                    comp_i = res_c[~res_c["Indexador"].str.startswith("Total")]
                    fig = px.pie(comp_i, names="Indexador", values="Saldo (R$ mm)", hole=0.5)
                    fig.update_layout(title="Debêntures por indexador (saldo)", height=340,
                                      margin=dict(t=40, b=20, l=10, r=10))
                    g4.plotly_chart(fig, width="stretch")
                susp = dd.loc[dd["taxa_suspeita"], "codigo"].tolist()
                st.caption(
                    "Custo contratado = remuneração de emissão (CDI+: spread sobre o CDI, com %DI convertido em spread "
                    "equivalente usando o CDI esperado no prazo pela curva prefixada; IPCA+: taxa real; prefixada: taxa "
                    "nominal). Taxa de mercado = taxa indicativa ANBIMA na mesma convenção. Soberano = ETTJ ANBIMA "
                    "(Svensson) de " + (f"{dt_curva:%d/%m/%Y}" if pd.notna(dt_curva) else "-") + " no prazo da duration "
                    "(sem preço ANBIMA: prazo até o vencimento), real (NTN-B) para IPCA+ e prefixada (LTN/NTN-F) para "
                    "pré. Prêmio de mercado = taxa de mercado − soberano, em p.p.; no CDI+ o prêmio é o próprio spread "
                    "sobre o CDI. Custo nominal equivalente: CDI+ = (1 + pré no prazo) × (1 + spread) − 1; IPCA+ = "
                    "(1 + real) × (1 + inflação implícita no prazo) − 1. Médias ponderadas pelo saldo estimado; taxa de "
                    "mercado, soberano e prêmio só com as debêntures que têm preço ANBIMA. Debêntures incentivadas "
                    "(Lei 12.431) são isentas de IR para pessoa física e por isso negociam com prêmio menor (às vezes "
                    "negativo) sobre a NTN-B. A separação por indexador cobre só as debêntures: a dívida bancária, "
                    "BNDES, CCB e externa não tem indexador nos dados estruturados da CVM. Custo implícito caixa = juros "
                    "pagos da dívida em 12 meses (DFC, ex-arrendamento) ÷ dívida bruta média (fim do LTM e um ano antes): "
                    "cobre a dívida toda, mas não inclui a correção monetária do principal IPCA ainda não paga nem os "
                    "juros capitalizados, e oscila com o calendário de pagamento de cupons."
                    + (f" ⚠️ Fora das médias de custo contratado por taxa incompatível com a curva (provável erro de "
                       f"cadastro no SND): {', '.join(susp)}." if susp else ""))
                st.caption("Fontes: SND/debentures.com.br (debêntures ativas, emissão, vencimento, quantidade em mercado, VNA, "
                           "remuneração e CNPJ do emissor), ANBIMA (mercado secundário: taxa indicativa, PU, % do PU par e "
                           "duration em dias úteis ÷ 252) e CVM/FRE item 9.1.c (controladas). Saldo estimado = quantidade em "
                           "mercado × PU par (ou VNA, sem preço ANBIMA) na data mais recente, não na data do balanço. "
                           "Na comparação com o balanço entram só debêntures emitidas até a data dele. A diferença inclui "
                           "empréstimos bancários, BNDES, CCB, notas comerciais, dívida externa e a diferença entre o saldo "
                           "estimado e o valor contábil (custo amortizado).")

            # ---- aluguel de ações (custo do short)
            st.subheader("Aluguel de ações (custo do short)")
            alug = carregar_aluguel(versao_arquivos(DADOS.parent / "aluguel" / "resumo.csv.gz"))
            raiz_e = str(linha_e["ticker"])[:4]
            a_emp = alug[alug["ticker"].str[:4] == raiz_e] if not alug.empty else pd.DataFrame()
            if alug.empty:
                st.info("Dados de aluguel ainda não disponíveis (saidas/aluguel/resumo.csv.gz).")
            elif a_emp.empty:
                st.info(f"Sem registros de aluguel para os papéis {raiz_e}*.")
            else:
                tks = a_emp.groupby("ticker")["saldo_alugado_qtd"].mean().sort_values(ascending=False).index.tolist()
                tk = st.selectbox("Papel", tks, key="alug_tk")
                at = a_emp[a_emp["ticker"] == tk].sort_values("data")
                # comparáveis: em cada empresa do grupo, o papel com maior saldo alugado médio
                grupo = cmp(base[base["tipo"] == tipo])[["cnpj", "ticker"]].drop_duplicates("cnpj")
                raiz_cnpj = dict(zip(grupo["ticker"].astype(str).str[:4], grupo["cnpj"]))
                ag = alug[alug["ticker"].str[:4].isin(raiz_cnpj)]
                princ = (ag.groupby("ticker")["saldo_alugado_qtd"].mean().reset_index()
                         .assign(r=lambda x: x["ticker"].str[:4]).sort_values("saldo_alugado_qtd").groupby("r").tail(1)["ticker"])
                med_tx = ag[ag["ticker"].isin(princ)].groupby("data")["taxa_tomador_media"].median()
                a1, a2 = st.columns(2)
                fig = go.Figure()
                fig.add_trace(go.Scatter(x=at["data"], y=at["taxa_tomador_media"], mode="lines", name=f"Tomador {tk}"))
                fig.add_trace(go.Scatter(x=at["data"], y=at["taxa_doador_media"], mode="lines", name=f"Doador {tk}"))
                mt = med_tx.reindex(at["data"])
                fig.add_trace(go.Scatter(x=at["data"], y=mt, mode="lines", line=dict(dash="dash"),
                                         name=f"Mediana tomador {nome_cmp}"))
                add_medias(fig, at["data"], at["taxa_tomador_media"], mt, "tomador")
                fig.update_layout(title="Taxa de aluguel (média ponderada do dia, % a.a.)", height=360,
                                  yaxis_tickformat=".2%", legend=dict(orientation="h", y=-0.25),
                                  margin=dict(t=40, b=20, l=10, r=10))
                a1.plotly_chart(fig, width="stretch")
                # saldo alugado em % das ações em circulação (todas as classes da empresa)
                if not mult.empty:
                    ult_m = mult[mult["tipo"] == "trimestral"].sort_values("data").groupby("cnpj").tail(1)
                    acoes = (ult_m["acoes_on"].fillna(0) + ult_m["acoes_pn"].fillna(0)).where(lambda x: x > 0)
                    acoes_por_cnpj = dict(zip(ult_m["cnpj"], acoes))
                else:
                    acoes_por_cnpj = {}
                def pct_alugado(df_r, cnpj_):
                    q = acoes_por_cnpj.get(cnpj_)
                    serie = df_r.groupby("data")["saldo_alugado_qtd"].sum(min_count=1)
                    return serie / q if q else serie * np.nan
                p_emp = pct_alugado(a_emp, linha_e["cnpj"])
                pares_pct = [pct_alugado(ag[ag["ticker"].str[:4] == r], c).rename(r) for r, c in raiz_cnpj.items()]
                med_pct = pd.concat(pares_pct, axis=1).median(axis=1) if pares_pct else pd.Series(dtype=float)
                fig = go.Figure()
                fig.add_trace(go.Scatter(x=p_emp.index, y=p_emp.values, mode="lines", name=f"{raiz_e} (todas as classes)"))
                mp = med_pct.reindex(p_emp.index)
                fig.add_trace(go.Scatter(x=p_emp.index, y=mp.values, mode="lines", line=dict(dash="dash"),
                                         name=f"Mediana {nome_cmp}"))
                add_medias(fig, p_emp.index, p_emp.values, mp.values)
                fig.update_layout(title="Saldo alugado (% das ações em circulação)", height=360, yaxis_tickformat=".1%",
                                  legend=dict(orientation="h", y=-0.25), margin=dict(t=40, b=20, l=10, r=10))
                a2.plotly_chart(fig, width="stretch")
                st.caption(f"Fonte: B3 (empréstimo de ativos), via coletor-aluguel-b3; série desde "
                           f"{alug['data'].min().date()}. Taxas ao ano (a B3 informa em fração; 0,04 = 4% a.a.), média ponderada pela quantidade contratada no "
                           "dia (eletrônico e balcão). Saldo alugado ÷ ações em circulação da última composição do capital. "
                           "Comparáveis: o papel com maior saldo alugado de cada empresa do grupo escolhido.")

        # ---- gráficos sobrepostos: valor de mercado (padrão) + outras linhas para comparar comportamento
        st.subheader("Gráficos sobrepostos")
        ex_ = eixo_x(tipo)
        cand = [k for k in METRICAS if k in d_emp.columns and d_emp[k].notna().any()]
        o1, o2 = st.columns([3, 2])
        padrao_ov = [k for k in ("valor_mercado",) if k in cand] or [k for k in ("receita", "lucro_liquido") if k in cand]
        sel_ov = o1.multiselect("Linhas da empresa", cand, default=padrao_ov,
                                format_func=lambda k: ROT[k], key="ov_ind")
        outras = o1.multiselect("Valor de mercado de outras empresas (opcional)",
                                sorted(r for r in base["rotulo"].unique() if r != emp), key="ov_emp")
        escala = o2.radio("Escala", ["Base 100 (primeiro ponto do período)", "Valores (dois eixos)"], key="ov_esc")
        series_ov = [(ROT[k], d_emp[ex_], d_emp[k], METRICAS[k][1], k, linha_e["cnpj"]) for k in sel_ov]
        for o in outras:
            dd = base[(base["rotulo"] == o) & (base["tipo"] == tipo) & base[ex_].isin(d_emp[ex_])].sort_values(ex_)
            series_ov.append((f"Valor de mercado {o.split(' — ')[0]}", dd[ex_], dd["valor_mercado"], "mm",
                              "valor_mercado", dd["cnpj"].iloc[0] if len(dd) else None))
        if not series_ov:
            st.info("Escolha ao menos uma linha.")
        else:
            fig = go.Figure()
            ignoradas = []
            DE_MERCADO = {"valor_mercado", "ev_com_arrend", "ev_sem_arrend", "p_vpa", "p_l", "ev_ebitda_ifrs16",
                          "ev_ebitda_pre_ifrs16", "dividend_yield", "ev_receita_com_arrend", "ev_receita_sem_arrend"}
            for i_s, (nome, x, y, f, chave, cn_s) in enumerate(series_ov):
                cor = PALETA_OV[i_s % len(PALETA_OV)]
                y = pd.to_numeric(y, errors="coerce").reset_index(drop=True)
                x = pd.Series(list(x))
                # valor na cotação do dia (último pregão), para indicadores que dependem de preço
                ponto = None
                if chave in DE_MERCADO and not mult.empty and cn_s is not None:
                    hm = mult[(mult["cnpj"] == str(cn_s).zfill(14)) & (mult["tipo"] == "trimestral")].sort_values("data")
                    if len(hm) and pd.notna(hm[chave].iloc[-1]):
                        dh = pd.Timestamp(hm["data"].iloc[-1])
                        xh = dh.year + (dh.dayofyear - 1) / 366 if tipo == "anual" else dh
                        ponto = (xh, float(hm[chave].iloc[-1]), dh)
                if escala.startswith("Base 100"):
                    validos = y[y.notna()]
                    if validos.empty or validos.iloc[0] <= 0:
                        ignoradas.append(nome)
                        continue
                    fig.add_trace(go.Scatter(x=x, y=y / validos.iloc[0] * 100, mode="lines+markers", name=nome,
                                             line=dict(color=cor), marker=dict(color=cor)))
                    if ponto:
                        fig.add_trace(go.Scatter(x=[ponto[0]], y=[ponto[1] / validos.iloc[0] * 100], mode="markers",
                                                 marker=alvo(cor),
                                                 name=f"{nome}: cotação de {ponto[2].strftime('%d/%m/%Y')}"))
                else:
                    eixo = "y" if f == "mm" else "y2"
                    fig.add_trace(go.Scatter(x=x, y=y * (100 if f == "pct" else 1), mode="lines+markers",
                                             name=nome + (" (%)" if f == "pct" else " (x)" if f == "x" else f" ({UN})"),
                                             yaxis=eixo, line=dict(color=cor), marker=dict(color=cor)))
                    if ponto:
                        fig.add_trace(go.Scatter(x=[ponto[0]], y=[ponto[1] * (100 if f == "pct" else 1)], mode="markers",
                                                 marker=alvo(cor), yaxis=eixo,
                                                 name=f"{nome}: cotação de {ponto[2].strftime('%d/%m/%Y')}"))
            if escala.startswith("Base 100"):
                fig.update_layout(yaxis_title="Base 100 = primeiro ponto")
                fig.add_hline(y=100, line_dash="dot", line_color="gray")
            else:
                fig.update_layout(yaxis=dict(title=UN),
                                  yaxis2=dict(title="Múltiplos (x) e % ", overlaying="y", side="right", showgrid=False))
            fig.update_layout(height=480, legend=dict(orientation="h", y=-0.2), margin=dict(t=30, b=20, l=10, r=10))
            st.plotly_chart(fig, width="stretch")
            if ignoradas:
                st.caption("Fora da base 100 (primeiro valor zero ou negativo): " + ", ".join(ignoradas) +
                           ". Use a escala de valores para vê-las.")
            st.caption("Base 100 mostra o comportamento relativo de cada linha desde o início do período escolhido "
                       f"na barra lateral. Em valores, {UN} ficam no eixo da esquerda; múltiplos e % no da direita. "
                       "◎ = indicador de mercado calculado com o preço do último pregão, na cor da curva correspondente.")

        # ---- estrutura societária e modelo de negócio (fim da visão da empresa)
        st.divider()
        _mm.secao_estrutura(_cn_e, emp.split(" — ")[-1] if " — " in emp else emp, eua=EUA, chave="soc_e")
        st.divider()
        _mm.secao_modelo(_tk_e if EUA else _tk_e[:4], emp)

# -------- Filtro por indicador (screener): fundamentos do último LTM + mercado na última cotação
MERCADO_SCR = ["valor_mercado", "ev_com_arrend", "ev_sem_arrend", "p_vpa", "p_l", "ev_ebitda_ifrs16",
               "ev_ebitda_pre_ifrs16", "ev_receita_com_arrend", "ev_receita_sem_arrend", "dividend_yield"]
EXTRA_SCR = {"cresc_receita_12m": ("Crescimento da receita 12m (a/a)", "pct"),
             "cresc_ebitda_12m": ("Crescimento do EBITDA 12m (a/a)", "pct"),
             "cresc_lucro_12m": ("Crescimento do lucro 12m (a/a)", "pct")}


def snapshot_atual(cnpjs) -> pd.DataFrame:
    """Uma linha por empresa: fundamentos do LTM mais recente (ou do último anual, se não houver LTM) e
    múltiplos de mercado na última cotação disponível (com PL, dívida e lucro do último documento divulgado)."""
    b = base[base["cnpj"].isin(cnpjs)]
    ltm = b[b["tipo"] == "ltm"]
    anu = b[(b["tipo"] == "anual") & ~b["cnpj"].isin(ltm["cnpj"])]
    fonte = pd.concat([ltm, anu])
    ult = fonte.sort_values("dt_fim").groupby("cnpj").tail(1).copy()
    # crescimento contra o mesmo período 1 ano antes
    ref = ult[["cnpj", "dt_fim"]].assign(dt_ant=lambda x: x["dt_fim"] - pd.DateOffset(years=1))
    j = ref.merge(fonte[["cnpj", "dt_fim", "receita", "ebitda", "lucro_liquido"]], on="cnpj", suffixes=("", "_a"))
    j = j[(j["dt_fim_a"] - j["dt_ant"]).abs() <= pd.Timedelta(days=20)].drop_duplicates("cnpj")
    j = j.set_index("cnpj")
    for c, k in (("receita", "cresc_receita_12m"), ("ebitda", "cresc_ebitda_12m"), ("lucro_liquido", "cresc_lucro_12m")):
        ant = ult["cnpj"].map(j[c]) if len(j) else np.nan
        ult[k] = np.where(pd.notna(ant) & (ant > 0), ult[c] / ant - 1, np.nan)
    ult = ult.rename(columns={"periodo": "periodo_fundamentos"})
    # mercado: ponto mais recente da série trimestral de múltiplos (= último pregão)
    ult = ult.drop(columns=[c for c in MERCADO_SCR if c in ult.columns])
    if not mult.empty:
        mm = mult[(mult["tipo"] == "trimestral") & mult["cnpj"].isin(cnpjs)].sort_values("data").groupby("cnpj").tail(1)
        dpo = pd.to_datetime(mm.get("data_preco_on"), errors="coerce")
        dpp = pd.to_datetime(mm.get("data_preco_pn"), errors="coerce")
        mm = mm.assign(data_ultimo_negocio=pd.concat([dpo, dpp], axis=1).max(axis=1))
        ult = ult.merge(mm[["cnpj", "data", "data_ultimo_negocio", "notas"] + MERCADO_SCR]
                        .rename(columns={"data": "data_cotacao", "notas": "notas_mercado"}), on="cnpj", how="left")
    else:
        for c in MERCADO_SCR + ["data_cotacao", "data_ultimo_negocio", "notas_mercado"]:
            ult[c] = np.nan
    return ult


with aba_scr:
    st.caption("Uma linha por empresa. Fundamentos: últimos 12 meses (LTM) mais recentes. Indicadores de mercado: "
               "última cotação disponível, com PL, dívida e lucro do último documento divulgado até essa data. "
               "Respeita os filtros de setor, subsetor, segmento e empresas da barra lateral.")
    snap = snapshot_atual(set(f3["cnpj"]) if sel_emp == [] else set(f3[f3["rotulo"].isin(sel_emp)]["cnpj"]))
    rot_scr = {**{k: ROT[k] for k in METRICAS}, **{k: v[0] for k, v in EXTRA_SCR.items()}}
    fmt_scr = {**{k: METRICAS[k][1] for k in METRICAS}, **{k: v[1] for k, v in EXTRA_SCR.items()}}
    opcoes_scr = [k for k in list(METRICAS) + list(EXTRA_SCR) if k in snap.columns and snap[k].notna().any()]
    c1, c2 = st.columns([3, 2])
    sel_scr = c1.multiselect("Indicadores para filtrar", opcoes_scr,
                             default=[k for k in ("p_l", "p_vpa", "div_liq_ebitda_ifrs16", "roe_pl_final") if k in opcoes_scr],
                             format_func=lambda k: rot_scr[k] + (" (%)" if fmt_scr[k] == "pct" else
                                                                 f" ({UN})" if fmt_scr[k] == "mm" else
                                                                 " (dias)" if fmt_scr[k] == "d" else " (x)"),
                             key="scr_ind")
    sem_velha = c2.checkbox("Excluir cotações antigas (último negócio há mais de 30 dias)", value=True, key="scr_velha")
    sem_alerta = c2.checkbox("Excluir empresas com alerta de qualidade no último LTM", value=False, key="scr_alerta")
    res = snap.copy()
    if sem_velha and "data_ultimo_negocio" in res and res["data_ultimo_negocio"].notna().any():
        ref_d = pd.to_datetime(res["data_cotacao"]).max()
        res = res[(ref_d - pd.to_datetime(res["data_ultimo_negocio"])).dt.days <= 30]
    if sem_alerta:
        res = res[~res["alerta"].fillna(False).astype(bool)]
    st.markdown("**Faixas** (deixe em branco para não limitar; percentuais em %, ex.: 15 = 15%)")
    for k in sel_scr:
        f = fmt_scr[k]
        serie = res[k] * (100 if f == "pct" else 1)
        med = serie.median()
        a, b_, c_ = st.columns([2, 1, 1])
        a.markdown(f"{rot_scr[k]}  \n<small>mediana do filtro: {med:,.2f}{'%' if f == 'pct' else 'x' if f == 'x' else ' dias' if f == 'd' else ''}"
                   f" · {serie.notna().sum()} empresas com dado</small>", unsafe_allow_html=True)
        lo = b_.number_input("Mínimo", value=None, key=f"scr_lo_{k}", label_visibility="collapsed", placeholder="mínimo")
        hi = c_.number_input("Máximo", value=None, key=f"scr_hi_{k}", label_visibility="collapsed", placeholder="máximo")
        if lo is not None or hi is not None:
            v = res[k] * (100 if f == "pct" else 1)
            ok = v.notna()
            if lo is not None:
                ok &= v >= lo
            if hi is not None:
                ok &= v <= hi
            res = res[ok]
    st.markdown(f"**{len(res)} empresas** atendem aos critérios (de {len(snap)} no recorte).")
    base_cols = ["p_l", "p_vpa", "ev_ebitda_ifrs16", "dividend_yield", "roe_pl_final", "margem_ebitda",
                 "div_liq_ebitda_ifrs16", "cresc_receita_12m", "valor_mercado"]
    mostrar = list(dict.fromkeys(sel_scr + [c for c in base_cols if c in res.columns]))
    tab_r = res[["rotulo", "segmento", "periodo_fundamentos", "data_cotacao"] + mostrar].copy()
    tab_r["data_cotacao"] = pd.to_datetime(tab_r["data_cotacao"]).dt.date
    cfg = {"rotulo": st.column_config.TextColumn("Empresa"), "segmento": "Segmento",
           "periodo_fundamentos": "Fundamentos", "data_cotacao": "Cotação de"}
    vis_r = tab_r.copy()
    for k in mostrar:
        if fmt_scr[k] == "pct":
            vis_r[k] = vis_r[k] * 100
            cfg[k] = st.column_config.NumberColumn(rot_scr[k], format="%.1f%%")
        elif fmt_scr[k] == "x":
            cfg[k] = st.column_config.NumberColumn(rot_scr[k], format="%.2fx")
        elif fmt_scr[k] == "d":
            cfg[k] = st.column_config.NumberColumn(rot_scr[k] + " (dias)", format="%.0f")
        else:
            cfg[k] = st.column_config.NumberColumn(rot_scr[k] + f" ({UN})", format="%,.0f")
    vis_r = vis_r.sort_values(mostrar[0] if mostrar else "rotulo")
    tabela_clicavel(vis_r, vis_r["rotulo"], "clk_scr", column_config=cfg, hide_index=True,
                    width="stretch", height=min(700, 38 + 35 * len(vis_r)))
    st.caption("Clique no cabeçalho para ordenar. Múltiplos vazios = denominador ≤ 0 ou sem cotação.")
    exp_r = tab_r.rename(columns={k: rot_scr[k] for k in mostrar})
    d1, d2 = st.columns(2)
    d1.download_button("⬇️ Resultado (CSV)", exp_r.to_csv(index=False, sep=";", decimal=",").encode("utf-8-sig"),
                       file_name="filtro_indicadores.csv", mime="text/csv", key="scr_csv")
    d2.download_button("⬇️ Resultado (XLSX)", excel_bytes({"filtro": exp_r}), file_name="filtro_indicadores.xlsx",
                       mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet", key="scr_xlsx")

# -------- Ranking
with aba_rank:
    if filtrado.empty:
        st.info("Nenhum dado para o filtro.")
    else:
        c1, c2, c3 = st.columns(3)
        met = c1.selectbox("Métrica", list(METRICAS), index=list(METRICAS).index("margem_ebitda"), format_func=lambda k: ROT[k])
        if tipo == "anual":
            per = c2.selectbox("Exercício", sorted(filtrado["exercicio"].dropna().astype(int).unique(), reverse=True), key="rk_ano")
            d = filtrado[filtrado["exercicio"] == per]
        else:
            rot_rk = col_periodo(filtrado)
            per = c2.selectbox("Período", sorted(rot_rk.unique(),
                               key=lambda p: filtrado.loc[rot_rk == p, "dt_fim"].max(), reverse=True), key="rk_per")
            d = filtrado[rot_rk == per]
        n = c3.slider("Quantas empresas", 5, 50, 20)
        ordem = st.radio("Ordem", ["Maiores", "Menores"], horizontal=True)
        d = d.dropna(subset=[met]).sort_values(met, ascending=(ordem == "Menores")).head(n)
        fig = px.bar(d, x=met, y="rotulo", orientation="h", color="segmento", labels={met: ROT[met], "rotulo": ""})
        fig.update_layout(height=max(350, 28 * len(d)), yaxis=dict(autorange="reversed"), legend_title_text="Segmento")
        if METRICAS[met][1] == "pct":
            fig.update_xaxes(tickformat=".0%")
        ev_rk = st.plotly_chart(fig, width="stretch", on_select="rerun", selection_mode="points", key=chave_clicavel("clk_rank"))
        pts = list(getattr(getattr(ev_rk, "selection", None), "points", []) or [])
        if pts:
            ir_para_empresa(pts[0].get("y"), "clk_rank")

# -------- Altas e baixas (variação do valor de mercado)
JANELAS = {"Diária": ("D", 1), "Semanal": ("D", 7), "Mensal": ("M", 1), "Trimestral": ("M", 3),
           "Semestral": ("M", 6), "Anual": ("M", 12), "2 anos": ("M", 24), "3 anos": ("M", 36),
           "5 anos": ("M", 60)}

if True:
    with aba_var:
        if EUA:
            vmd = eua.vm_diario_recente(mult)
        else:
            _mm = mercado_mod()
            vmd = _mm.vm_ev_todos(_mm._versao("vm_ev_*.parquet"))
            if vmd.empty:                       # base antiga (até nov/2026): um CSV com a série desde 2020
                vmd = carregar_vm_diario(versao_arquivos(DADOS / "valor_mercado_diario.csv.gz"))
        if vmd.empty:
            st.info("Série diária de valor de mercado ainda não gerada: aparece depois da próxima execução do coletor.")
        else:
            pregoes = sorted(vmd["data"].unique())
            c0, c1, c2 = st.columns([1.4, 1.4, 1.4])
            d_fim = c0.selectbox("Data final (pregão)", pregoes[::-1],
                                 format_func=lambda d: pd.Timestamp(d).strftime("%d/%m/%Y"), key="var_dfim")
            d_fim = pd.Timestamp(d_fim)
            jan = c1.selectbox("Período", list(JANELAS) if not EUA else list(JANELAS)[:6], index=0, key="var_jan")
            n_top = c2.slider("Quantas de cada lado", 5, 50, 30, 5, key="var_n")
            c3 = st.container()
            unidade, passo = JANELAS[jan]
            alvo = d_fim - (pd.Timedelta(days=passo) if unidade == "D" else pd.DateOffset(months=passo))
            tol = 3 if unidade == "D" else 20       # tolerância em dias para achar o pregão de referência
            fim = vmd[vmd["data"] == d_fim][["cnpj", "valor_mercado"]].rename(columns={"valor_mercado": "vm_fim"})
            if jan == "Diária":                    # diária: pregão imediatamente anterior ao escolhido
                anteriores = [d for d in pregoes if d < d_fim]
                alvo = pd.Timestamp(anteriores[-1]) if anteriores else alvo
            ant = vmd[vmd["data"] <= alvo].sort_values("data").groupby("cnpj").tail(1)
            ant = ant[(alvo - ant["data"]).dt.days <= tol][["cnpj", "data", "valor_mercado"]].rename(
                columns={"data": "data_ini", "valor_mercado": "vm_ini"})
            v = fim.merge(ant, on="cnpj").assign(variacao=lambda x: x["vm_fim"] / x["vm_ini"] - 1)
            info = base.drop_duplicates("cnpj")[["cnpj", "rotulo", "setor", "subsetor", "segmento"]]
            if not fin.empty:
                f_i = fin.assign(cnpj=fin["cnpj"].astype(str).str.zfill(14)).drop_duplicates("cnpj")
                for c in ("rotulo", "setor", "subsetor", "segmento"):
                    if c not in f_i.columns:
                        f_i[c] = "Financeiro" if c == "setor" else np.nan
                info = pd.concat([info, f_i[["cnpj", "rotulo", "setor", "subsetor", "segmento"]]])
            v = v.merge(info.drop_duplicates("cnpj"), on="cnpj", how="left")
            v = v[v["rotulo"].notna()]              # fora da base (sem fundamentos): não entra no ranking nem no mapa
            set_f = c3.multiselect("Setores (vazio = todos)", sorted(v["setor"].dropna().unique()), key="var_setor")
            if set_f:
                v = v[v["setor"].isin(set_f)]
            vm_min = st.slider(f"Valor de mercado mínimo ({MOEDA} bi)", 0.0, 20.0, 0.5 if not EUA else 2.0, 0.25,
                               key="var_vmmin", help="Filtra empresas pequenas, onde um negócio isolado move muito o preço.")
            v = v[(v["vm_fim"] >= vm_min * 1000) & v["variacao"].notna()]
            v["vm_bi"] = v["vm_fim"] / 1000        # a série vem em milhões
            mercado_mod().mapa_calor(v, f"{jan.lower()}: {pd.Timestamp(alvo).strftime('%d/%m/%Y')} a "
                                        f"{pd.Timestamp(d_fim).strftime('%d/%m/%Y')}", MOEDA, chave="var_mapa")
            st.caption(f"Variação do valor de mercado entre {pd.Timestamp(d_fim).strftime('%d/%m/%Y')} e o pregão mais "
                       f"próximo de {pd.Timestamp(alvo).strftime('%d/%m/%Y')}, em {len(v)} empresas. " +
                       ("Nos EUA a variação é a do preço do ticker principal (Yahoo Finance) corrigido por "
                        "desdobramentos, aplicada ao valor de mercado do último pregão; histórico de ~200 dias. "
                        if EUA else "O valor de mercado usa as ações vigentes em cada data, então desdobramentos e "
                        "grupamentos não viram variação; emissões e recompras, sim. ") +
                       "Não inclui dividendos (é variação de preço, não retorno total).")
            cfg_v = {"rotulo": "Empresa", "setor": "Setor", "variacao": st.column_config.NumberColumn("Variação", format="percent"),
                     "vm_bi": st.column_config.NumberColumn(f"Valor de mercado ({MOEDA} bi)", format="%,.1f"),
                     "data_ini": "Pregão inicial"}
            cols_v = ["rotulo", "setor", "variacao", "vm_bi", "data_ini"]
            altas = v.nlargest(n_top, "variacao")[cols_v].assign(data_ini=lambda x: x["data_ini"].dt.date)
            baixas = v.nsmallest(n_top, "variacao")[cols_v].assign(data_ini=lambda x: x["data_ini"].dt.date)
            t1, t2 = st.columns(2)
            t1.markdown(f"**{n_top} maiores altas**")
            tabela_clicavel(altas, altas["rotulo"], "clk_altas", alvo=t1, hide_index=True, width="stretch",
                            column_config=cfg_v, height=min(900, 38 + 35 * n_top))
            t2.markdown(f"**{n_top} maiores baixas**")
            tabela_clicavel(baixas, baixas["rotulo"], "clk_baixas", alvo=t2, hide_index=True, width="stretch",
                            column_config=cfg_v, height=min(900, 38 + 35 * n_top))
            st.subheader("Variação por setor")
            por_setor = v.groupby("setor").apply(
                lambda g: pd.Series({"Mediana": g["variacao"].median(),
                                     "Ponderada pelo valor de mercado": g["vm_fim"].sum() / (g["vm_fim"] / (1 + g["variacao"])).sum() - 1,
                                     "Empresas": len(g)}), include_groups=False).sort_values("Ponderada pelo valor de mercado")
            fig = go.Figure()
            for c, cor in (("Mediana", None), ("Ponderada pelo valor de mercado", "#16a34a")):
                fig.add_trace(go.Bar(y=por_setor.index, x=por_setor[c], orientation="h", name=c, marker_color=cor))
            fig.add_vline(x=0, line_color="gray")
            fig.update_layout(height=max(360, 26 * len(por_setor)), barmode="group", xaxis_tickformat=".1%",
                              legend=dict(orientation="h", y=-0.15), margin=dict(t=20, b=20, l=10, r=10))
            st.plotly_chart(fig, width="stretch")
            exp_v = v[cols_v + ["subsetor", "segmento", "vm_fim", "vm_ini"]].sort_values("variacao", ascending=False)
            st.download_button("⬇️ Todas as empresas (CSV)",
                               exp_v.assign(data_ini=exp_v["data_ini"].dt.date).to_csv(index=False, sep=";", decimal=",").encode("utf-8-sig"),
                               file_name=f"variacoes_{jan.lower()}.csv", mime="text/csv", key="var_csv")


# -------- Calendário de resultados (entregas de ITR/DFP à CVM)
@st.cache_data(max_entries=1, show_spinner=False)
def carregar_entregas(versao: tuple = ()) -> pd.DataFrame:
    """Dados Abertos da CVM (histórico, com ~1 semana de defasagem) + consulta pública ENET (quase tempo real)."""
    partes = [pd.read_csv(DADOS / n, dtype={"cnpj": str}) for n in ("entregas.csv.gz", "entregas_rad.csv.gz")
              if (DADOS / n).exists()]
    if not partes:
        return pd.DataFrame()
    e = pd.concat(partes, ignore_index=True)
    e["cnpj"] = e["cnpj"].astype(str).str.zfill(14)
    # as duas fontes gravam formatos diferentes (só data nos Dados Abertos; data e hora na consulta ENET):
    # format="ISO8601" lê os dois. Sem ele, o pandas 2+ deduz o formato pela 1a linha e anula as demais.
    e["dt_refer"] = pd.to_datetime(e["dt_refer"], errors="coerce", format="ISO8601")
    e["dt_receb"] = pd.to_datetime(e["dt_receb"], errors="coerce", format="ISO8601")
    e = e.dropna(subset=["dt_refer", "dt_receb"]).sort_values("dt_receb")
    return e.drop_duplicates(["cnpj", "documento", "dt_refer"])     # primeira entrega de cada documento


if not EUA:          # sem dados equivalentes nos EUA
    with aba_cal:
        ent = carregar_entregas(versao_arquivos(DADOS / "entregas.csv.gz", DADOS / "entregas_rad.csv.gz"))
        if ent.empty:
            st.info("Datas de entrega ainda não geradas: aparecem depois da próxima execução do coletor.")
        else:
            hoje = pd.Timestamp.today().normalize()
            refs = sorted(d for d in ent["dt_refer"].unique() if pd.Timestamp(d).month in (3, 6, 9, 12))
            fins_tri = [pd.Timestamp(d) for d in pd.date_range(end=hoje, periods=2, freq="QE")]
            padrao = fins_tri[-1] if fins_tri[-1] <= hoje else fins_tri[0]
            opcoes = sorted(set(pd.Timestamp(d) for d in refs[-12:]) | {padrao}, reverse=True)
            k1, k2 = st.columns([1.3, 2])
            ref = k1.selectbox("Trimestre de referência", opcoes, index=opcoes.index(padrao),
                               format_func=lambda d: f"{(d.month - 1) // 3 + 1}T{str(d.year)[2:]} (encerrado em "
                                                     f"{d.strftime('%d/%m/%Y')})", key="cal_ref")
            ref = pd.Timestamp(ref)
            ant_ref = ref - pd.DateOffset(years=1)
            # universo: empresas da base (e financeiras) que entregaram algum documento nos 15 meses anteriores
            info = base.drop_duplicates("cnpj")[["cnpj", "rotulo", "setor"]]
            if not fin.empty:
                f_i = fin.assign(cnpj=fin["cnpj"].astype(str).str.zfill(14)).drop_duplicates("cnpj")
                f_i["setor"] = f_i["setor"] if "setor" in f_i.columns else "Financeiro"
                info = pd.concat([info, f_i[["cnpj", "rotulo", "setor"]]]).drop_duplicates("cnpj")
            ativas = ent[(ent["dt_refer"] < ref) & (ent["dt_refer"] >= ref - pd.DateOffset(months=15))]["cnpj"].unique()
            univ_c = info[info["cnpj"].isin(ativas)].copy()
            setores_c = k2.multiselect("Setores (vazio = todos)", sorted(univ_c["setor"].dropna().unique()), key="cal_set")
            if setores_c:
                univ_c = univ_c[univ_c["setor"].isin(setores_c)]
            atual = ent[ent["dt_refer"] == ref].sort_values("dt_receb").drop_duplicates("cnpj")
            passado = ent[ent["dt_refer"] == ant_ref].sort_values("dt_receb").drop_duplicates("cnpj")
            t = univ_c.merge(atual[["cnpj", "documento", "dt_receb", "link"]], on="cnpj", how="left").merge(
                passado[["cnpj", "documento", "dt_receb"]].rename(columns={"documento": "doc_ant", "dt_receb": "receb_ant"}),
                on="cnpj", how="left")
            # prazo legal: ITR 45 dias após o fim do trimestre; DFP 3 meses após o fim do exercício
            doc_esp = t["documento"].fillna(t["doc_ant"]).fillna("DFP" if ref.month == 12 else "ITR")
            t["prazo_legal"] = np.where(doc_esp == "DFP", ref + pd.DateOffset(months=3), ref + pd.Timedelta(days=45))
            t["prazo_legal"] = pd.to_datetime(t["prazo_legal"])
            # data estimada: entrega do mesmo trimestre do ano anterior + 364 dias (mesmo dia da semana)
            t["data_estimada"] = t["receb_ant"].dt.normalize() + pd.Timedelta(days=364)
            divulgou = t["dt_receb"].notna()
            n_tot, n_div = len(t), int(divulgou.sum())
            m1, m2, m3 = st.columns(3)
            m1.metric("Já divulgaram", f"{n_div} de {n_tot}", f"{n_div / n_tot:.0%}" if n_tot else None, delta_color="off")
            pend = t[~divulgou].copy()
            atrasadas = pend[hoje > pend["prazo_legal"]]
            m2.metric("Pendentes", f"{len(pend)}")
            m3.metric("Passaram do prazo legal", f"{len(atrasadas)}")

            st.subheader("Já divulgaram")
            div = t[divulgou].sort_values("dt_receb", ascending=False)
            tabela_clicavel(div[["rotulo", "setor", "documento", "dt_receb"]].assign(
                dt_receb=lambda x: x["dt_receb"].dt.strftime("%d/%m/%Y %H:%M")), div["rotulo"], "clk_cal_div",
                hide_index=True, width="stretch", height=min(600, 38 + 35 * len(div)), column_config={
                    "rotulo": "Empresa", "setor": "Setor", "documento": "Documento", "dt_receb": "Entregue à CVM em"})

            st.subheader("Ainda não divulgaram")
            st.caption("⚠️ **Data estimada** com base na data em que a empresa entregou o mesmo trimestre no ano anterior "
                       "(mesmo dia da semana, 52 semanas depois). Não é a data anunciada pela companhia: confira o "
                       "calendário de eventos corporativos dela para a data oficial.")
            pend["situacao"] = np.select(
                [hoje > pend["prazo_legal"], pend["data_estimada"].isna(), pend["data_estimada"] < hoje],
                ["Passou do prazo legal", "Sem histórico: usar o prazo legal", "Estimativa já passou"], "No prazo")
            pend = pend.sort_values(["data_estimada", "prazo_legal"], na_position="last")
            tabela_clicavel(pend[["rotulo", "setor", "data_estimada", "receb_ant", "prazo_legal", "situacao"]].assign(
                data_estimada=lambda x: x["data_estimada"].dt.date, receb_ant=lambda x: x["receb_ant"].dt.date,
                prazo_legal=lambda x: x["prazo_legal"].dt.date), pend["rotulo"], "clk_cal_pend", hide_index=True, width="stretch",
                height=min(600, 38 + 35 * len(pend)), column_config={
                    "rotulo": "Empresa", "setor": "Setor",
                    "data_estimada": "Data estimada (pela entrega do ano anterior)",
                    "receb_ant": f"Entregou {(ant_ref.month - 1) // 3 + 1}T{str(ant_ref.year)[2:]} em",
                    "prazo_legal": "Prazo legal", "situacao": "Situação"})

            # calendário: entregas realizadas e estimadas por dia
            cal_r = div.assign(dia=div["dt_receb"].dt.normalize()).groupby("dia").size().rename("Divulgadas")
            cal_e = pend.dropna(subset=["data_estimada"]).groupby("data_estimada").size().rename("Estimadas")
            cal = pd.concat([cal_r, cal_e], axis=1, sort=True).fillna(0)
            if len(cal):
                fig = go.Figure()
                fig.add_trace(go.Bar(x=cal.index, y=cal["Divulgadas"], name="Já divulgadas", marker_color="#16a34a"))
                fig.add_trace(go.Bar(x=cal.index, y=cal["Estimadas"], name="Estimadas (pelo ano anterior)",
                                     marker_color="#9ca3af"))
                fig.add_vline(x=hoje, line_dash="dot", line_color="gray")
                fig.update_layout(barmode="stack", height=340, title="Divulgações por dia", yaxis_title="Empresas",
                                  legend=dict(orientation="h", y=-0.2), margin=dict(t=40, b=20, l=10, r=10))
                st.plotly_chart(fig, width="stretch")
            st.caption("Fonte: consulta pública de documentos da CVM (ENET), atualizada a cada 2 horas, e índices de "
                       "Dados Abertos da CVM (histórico). "
                       "Data de divulgação = data de recebimento da primeira versão do documento pela CVM. Prazo legal: "
                       "ITR até 45 dias após o fim do trimestre; DFP até 3 meses após o fim do exercício.")


# -------- Financeiro: bancos e seguradoras (plano de contas próprio, base separada)
if not EUA:          # sem dados equivalentes nos EUA
    with aba_fin:
        if fin.empty:
            st.info("Base de bancos e seguradoras ainda não gerada (financeiras.csv.gz). Rode o workflow no modo universo.")
        else:
            st.caption("Base separada: bancos e seguradoras usam plano de contas próprio e não entram nas medianas das "
                       "demais abas. Fonte: CVM (DFP no anual; ITR no LTM e no trimestral, conforme a periodicidade da "
                       "barra lateral). Métricas são definições próprias; conferir com os releases.")
            c1, c2, c3 = st.columns([1.2, 1.2, 1])
            grupos_f = {"Bancos e financeiras": ~fin["segmento"].str.contains("Segurad|Ressegurad", regex=True),
                        "Seguradoras e resseguradoras": fin["segmento"].str.contains("Segurad|Ressegurad", regex=True)}
            grp = c1.radio("Grupo", list(grupos_f), horizontal=True)
            ff = fin[grupos_f[grp]]
            anual_f = tipo == "anual" or fin_per.empty or not (fin_per["tipo"] == tipo).any()
            if not anual_f:
                # LTM / trimestral (ITR), na periodicidade da barra lateral: fluxos LTM (4 trimestres somados) ou do
                # trimestre e saldos do último trimestre; mercado no ponto no tempo (fim do trimestre)
                cn_g = set(ff["cnpj"].astype(str).str.zfill(14))
                ff = fin_per[(fin_per["tipo"] == tipo)
                             & fin_per["cnpj"].astype(str).str.zfill(14).isin(cn_g)].dropna(subset=["dt_fim"]).copy()
                ff["cnpj"] = ff["cnpj"].astype(str).str.zfill(14)
                ff = ff.sort_values("dt_fim").drop_duplicates(["cnpj", "dt_fim"], keep="last")
                ff = ff.drop(columns=[c for c in ("valor_mercado", "p_l") if c in ff.columns])
                if not mult.empty:
                    mq = mult[mult["tipo"] == "trimestral"][["cnpj", "data", "valor_mercado", "p_l"]].drop_duplicates(
                        ["cnpj", "data"])
                    ff = ff.merge(mq, left_on=["cnpj", "dt_fim"], right_on=["cnpj", "data"], how="left").drop(
                        columns="data")
                else:
                    ff["valor_mercado"], ff["p_l"] = np.nan, np.nan
                # P/L do grupo = Σ valor de mercado ÷ Σ lucro dos 12 meses implícito no P/L de cada instituição
                ff["lucro_12m_pl"] = ff["valor_mercado"] / ff["p_l"].where(ff["p_l"] != 0)
                st.caption("Periodicidade da barra lateral: " + (
                    "LTM = fluxos dos últimos 4 trimestres somados; saldos (PL, ativo, carteira) do último trimestre."
                    if tipo == "ltm" else
                    "trimestral = fluxo do trimestre (diferença dos acumulados do ITR); ROE, ROA e custo do crédito "
                    "anualizados ×4.") + " Valor de mercado e P/L na data de fim do trimestre (P/L com lucro 12m).")
            elif tipo != "anual":
                st.info("Sem ITRs de bancos e seguradoras na base: mostrando a série anual (DFP).")
            opcoes = [k for k in FIN if ff[k].notna().any()] if len(ff) else list(FIN)
            rot_ind_f = lambda k: FIN[k][0] + {"mm": " (R$ mm)", "pct": " (%)", "x": " (x)"}[FIN[k][1]]
            inds_f = multiselect_seguro(c2, "Indicadores", opcoes, opcoes[:1], "fin_inds", format_func=rot_ind_f,
                                        help="Escolha um ou mais: cada indicador aparece em sequência, com tabela, "
                                             "gráfico e download próprios.")
            modo_f = c3.radio("Mostrar", ["Valores", "% do grupo"], horizontal=True, key="fin_modo",
                              disabled=not any(FIN[k][1] == "mm" for k in inds_f))
            if anual_f:
                anos_f = sorted(ff["exercicio"].dropna().astype(int).unique())
                if anos_f:
                    fa0, fa1 = st.select_slider("Exercícios", options=anos_f, value=(anos_f[0], anos_f[-1]), key="fin_anos")
                    ff = ff[ff["exercicio"].between(fa0, fa1)]
            else:
                datas_f = sorted(ff["dt_fim"].unique())
                if datas_f:
                    rot_df = [pd.Timestamp(x).strftime("%m/%Y") for x in datas_f]
                    fi0, fi1 = st.select_slider("Período (data de fechamento)", options=list(range(len(datas_f))),
                                                value=(max(0, len(datas_f) - 12), len(datas_f) - 1),
                                                format_func=lambda i: rot_df[i], key=f"fin_periodo_{tipo}")
                    ff = ff[ff["dt_fim"].between(datas_f[fi0], datas_f[fi1])]
            sel_f = st.multiselect("Empresas (vazio = todas do grupo)", sorted(ff["rotulo"].unique()), key="fin_emp")
            if anual_f:
                periodos_f = [str(a) for a in sorted(ff["exercicio"].dropna().astype(int).unique())]
                ff = ff.assign(_c=ff["exercicio"].astype(int).astype(str))
            else:
                periodos_f = [pd.Timestamp(x).strftime("%m/%Y") for x in sorted(ff["dt_fim"].unique())]
                ff = ff.assign(_c=ff["dt_fim"].dt.strftime("%m/%Y"))
            nome_per = "Exercício" if anual_f else "Período"
            if not inds_f:
                st.info("Escolha ao menos um indicador.")
            tabs_fin = {}
            for ind_f in inds_f:
                if len(inds_f) > 1:
                    st.divider()
                    st.subheader(rot_ind_f(ind_f))
                rot_f, fmt_f, agg_f = FIN[ind_f]
                if not anual_f and ind_f == "p_l":
                    agg_f = ("valor_mercado", "lucro_12m_pl")
                pct_f = modo_f == "% do grupo" and fmt_f == "mm"
                if agg_f == "cresc":
                    # crescimento do grupo: Σ atual ÷ Σ anterior − 1, só com instituições que têm os dois exercícios
                    cn_ = CRESC_FIN[ind_f]
                    ok = ff[(ff[cn_ + "_ant"] > 0) & ff[cn_].notna()]
                    g_ = ok.groupby("_c")[[cn_, cn_ + "_ant"]].sum()
                    tot_f = g_[cn_] / g_[cn_ + "_ant"] - 1
                    n_cresc = ok.groupby("_c")["cnpj"].nunique().reindex(periodos_f).fillna(0).astype(int)
                elif agg_f == "soma":
                    tot_f = ff.groupby("_c")[ind_f].sum(min_count=1)
                elif isinstance(agg_f, tuple):
                    n_, d_ = agg_f
                    ok = ff.dropna(subset=[n_, d_])
                    g_ = ok.groupby("_c")[[n_, d_]].sum()
                    tot_f = g_[n_].abs() / g_[d_].where(g_[d_] != 0) if ind_f == "custo_credito" else g_[n_] / g_[d_].where(g_[d_] != 0)
                    if not anual_f and tipo == "trimestral" and ind_f in ("roe", "roa", "custo_credito"):
                        tot_f = tot_f * 4  # mesmo critério da série por instituição (anualizado ×4)
                else:
                    tot_f = pd.Series(dtype=float)
                tot_f = tot_f.reindex(periodos_f)
                med_f = ff.groupby("_c")[ind_f].median().reindex(periodos_f)
                vis_f = ff[ff["rotulo"].isin(sel_f)] if sel_f else ff
                tab_f = vis_f.pivot_table(index="rotulo", columns="_c", values=ind_f, aggfunc="first").reindex(columns=periodos_f)
                ultf = [p for p in periodos_f[::-1] if p in tab_f.columns and tab_f[p].notna().any()]
                if ultf:
                    tab_f = tab_f.sort_values(ultf[0], ascending=False)
                if pct_f:
                    tab_f = tab_f.div(tot_f, axis=1)
                extra_f = {}
                if fmt_f == "mm":
                    if sel_f:
                        dem = tot_f - vis_f.groupby("_c")[ind_f].sum(min_count=1).reindex(periodos_f).fillna(0)
                        extra_f["Demais do grupo"] = dem / tot_f if pct_f else dem
                    extra_f["TOTAL DO GRUPO"] = tot_f / tot_f if pct_f else tot_f
                elif len(tot_f.dropna()):
                    extra_f["GRUPO AGREGADO"] = tot_f
                if not pct_f:
                    extra_f["Mediana do grupo"] = med_f
                full_f = pd.concat([tab_f, pd.DataFrame(extra_f).T.reindex(columns=periodos_f)])
                full_f.index.name = "Empresa"
                if pct_f or fmt_f == "pct":
                    cfg_f = {p: st.column_config.NumberColumn(p, format="%.1f%%") for p in periodos_f}
                    vis_tab = full_f * 100
                elif fmt_f == "x":
                    cfg_f = {p: st.column_config.NumberColumn(p, format="%.2fx") for p in periodos_f}
                    vis_tab = full_f
                else:
                    cfg_f = {p: st.column_config.NumberColumn(p, format="%,.1f") for p in periodos_f}
                    vis_tab = full_f
                tabela_clicavel(vis_tab, vis_tab.index, f"clk_fin_{ind_f}", column_config=cfg_f, width="stretch",
                                height=min(650, 38 + 35 * len(vis_tab)))
                if agg_f == "cresc":
                    st.caption(("Grupo agregado = Σ valor ÷ Σ valor do exercício anterior − 1, só com as instituições que têm "
                                "os dois exercícios (amostra casada). Instituições por exercício: " if anual_f else
                                "Grupo agregado = Σ valor ÷ Σ valor do mesmo período do ano anterior − 1, só com as "
                                "instituições que têm os dois períodos (amostra casada). Instituições por período: ") +
                               " · ".join(f"{p}: {n}" for p, n in n_cresc.items()) + ".")
                if fmt_f != "mm" and isinstance(agg_f, tuple):
                    st.caption(f"Grupo agregado = Σ {agg_f[0]} ÷ Σ {agg_f[1]} das instituições com os dois dados.")
                notas_fin = {
                    "lucro_antes_ir": "Linha da DRE 'Resultado antes dos tributos sobre o lucro' (IFRS) ou 'Resultado "
                                      "antes da tributação/participações' (COSIF), consolidado (inclui não controladores). "
                                      "No COSIF vem antes da participação nos lucros (PLR); no IFRS a PLR já está nas despesas.",
                    "indice_eficiencia": "Definição própria: |pessoal + administrativas| ÷ (margem financeira antes da PDD + "
                                         "serviços fora da intermediação). Difere do índice divulgado por cada banco; use para "
                                         "comparar tendência, não nível. Vazio quando o banco não abre a PDD na DRE.",
                    "carteira_credito": "Aproximação pelas contas de operações de crédito / empréstimos a clientes do balanço; "
                                        "pode diferir da carteira ampliada divulgada.",
                    "despesa_pdd": "Vazio quando a PDD não aparece separada na DRE (ex.: Santander).",
                    "custo_credito": "PDD do período ÷ carteira de crédito bruta no fim do período (aproximação).",
                    "carteira_credito_bruta": "Operações de crédito do balanço, com a provisão somada de volta quando vem "
                                              "dentro da conta (COSIF). Sob IFRS a conta já é bruta.",
                    "provisao_carteira": "Saldo redutor de PDD / perdas esperadas ligado à carteira. Sob IFRS pode incluir "
                                         "perdas esperadas de TVM ao custo amortizado.",
                    "tvm": "Títulos e valores mobiliários e derivativos, somando todas as categorias (custo amortizado, "
                           "VJORA e VJR).",
                    "demais_ativos": "Resíduo: ativo total − carteira líquida − TVM (caixa, aplicações interfinanceiras, "
                                     "compulsórios, tributos, investimentos etc.).",
                    "cresc_pl_controladores": "PL dos controladores (PL total − não controladores), mesmo critério do ROE.",
                    "cresc_carteira_credito": "Carteira aproximada pelo balanço: quebras de captura (conta que muda de nome "
                                              "ou exercício sem dado) aparecem como saltos ou lacunas; confira os níveis em "
                                              "'Carteira de crédito'.",
                    "cresc_pdd": "Despesa de PDD do exercício contra a do anterior, em módulo. Exercícios com reversão "
                                 "líquida (PDD positiva na DRE) e bancos que não abrem a PDD na DRE ficam vazios.",
                }
                if ind_f in notas_fin:
                    st.caption("ℹ️ " + notas_fin[ind_f])
                if fmt_f == "mm":
                    topf = tab_f.index[:10] if not sel_f else tab_f.index
                    plotf = tab_f.loc[topf].T if len(tab_f) else pd.DataFrame()
                    if not sel_f and len(tab_f) > 10:
                        plotf["Demais"] = tab_f.drop(index=topf).sum(min_count=1)
                    elif sel_f and "Demais do grupo" in extra_f:
                        plotf["Demais do grupo"] = extra_f["Demais do grupo"]
                    lf = plotf.reset_index(names=nome_per).melt(id_vars=nome_per, var_name="Empresa", value_name="v")
                    fig = px.bar(lf, x=nome_per, y="v", color="Empresa", labels={"v": "% do grupo" if pct_f else rot_f + " (R$ mm)"})
                    fig.update_layout(barmode="relative", height=460, legend_title_text="")
                    if pct_f:
                        fig.update_yaxes(tickformat=".0%")
                elif agg_f == "cresc":
                    # crescimento acumulado, base 100 no primeiro exercício (mesma lógica da Visão do setor): uma linha por
                    # instituição (valor ÷ primeiro valor × 100) e o agregado do grupo encadeado em amostra casada
                    cn_ = CRESC_FIN[ind_f]
                    niv_f = ff.pivot_table(index="rotulo", columns="_c", values=cn_, aggfunc="first").reindex(columns=periodos_f)
                    idx_f = {}
                    for r in (tab_f.index if sel_f else niv_f.index):
                        if r not in niv_f.index:
                            continue
                        v = niv_f.loc[r]
                        ini = v.first_valid_index()
                        if ini is None or v[ini] <= 0:
                            continue
                        idx = 100 * v / v[ini]
                        neg = idx[idx <= 0].index
                        if len(neg):  # base negativa no meio da série: o índice deixa de ter leitura
                            idx[idx.index >= neg[0]] = np.nan
                        idx_f[r] = idx
                    if periodos_f:
                        idx_f["Agregado do grupo"] = indice_acumulado(niv_f)
                    nomes_f = list(idx_f)
                    ult_pf = ultf[0] if ultf and ultf[0] in niv_f.columns else (periodos_f[-1] if periodos_f else None)
                    tam_f = niv_f[ult_pf].sort_values(ascending=False) if ult_pf else pd.Series(dtype=float)
                    padrao_f = [r for r in (tab_f.index if sel_f else tam_f.index[:12]) if r in idx_f]
                    padrao_f += [n for n in ("Agregado do grupo",) if n in idx_f]
                    chave_lf = f"fin_cres_linhas_{ind_f}_{grp}_" + str(abs(hash((tuple(sel_f), tuple(periodos_f)))))
                    sel_lf = st.multiselect("Linhas no gráfico", nomes_f, default=padrao_f, key=chave_lf,
                                            help="Também dá para ocultar uma linha clicando no nome dela na legenda.")
                    fig = go.Figure()
                    for nome in sel_lf:
                        y = idx_f[nome]
                        fig.add_trace(go.Scatter(x=list(y.index), y=y.values, mode="lines+markers",
                                                 name=nome.split(" — ")[0] if " — " in nome else nome,
                                                 line=dict(width=4, dash="dash") if nome == "Agregado do grupo" else None))
                    fig.add_hline(y=100, line_dash="dot", line_color="gray")
                    fig.update_layout(height=460, legend_title_text="", yaxis_title="Base 100 = primeiro " + ("exercício" if anual_f else "período"),
                                      title=f"{ROT_NIVEL_FIN[cn_]}: crescimento acumulado (base 100)")
                else:
                    fig = go.Figure()
                    for r in tab_f.index[:10]:
                        fig.add_trace(go.Scatter(x=periodos_f, y=tab_f.loc[r], mode="lines+markers", name=r.split(" — ")[0]))
                    if len(tot_f.dropna()):
                        fig.add_trace(go.Scatter(x=periodos_f, y=tot_f, mode="lines", name="Grupo agregado", line=dict(width=4, dash="dash")))
                    fig.add_trace(go.Scatter(x=periodos_f, y=med_f, mode="lines", name="Mediana", line=dict(dash="dot")))
                    fig.update_layout(height=460, legend_title_text="")
                    if fmt_f == "pct":
                        fig.update_yaxes(tickformat=".0%")
                if not anual_f:
                    fig.update_xaxes(type="category", categoryorder="array", categoryarray=periodos_f)
                st.plotly_chart(fig, width="stretch", key=f"fin_fig_{ind_f}")
                if agg_f == "cresc":
                    st.caption("Uma linha por instituição: valor ÷ valor do primeiro período × 100; a linha é interrompida "
                               "se o valor fica zero ou negativo. 'Agregado do grupo' encadeia a soma das instituições em "
                               "amostra casada. A tabela acima mostra o crescimento de cada período contra o mesmo período "
                               "do ano anterior.")
                ex_f = full_f.reset_index()
                tabs_fin[f"{len(tabs_fin) + 1}. {rot_f}"] = ex_f
                c1, c2 = st.columns(2)
                c1.download_button("⬇️ Tabela (CSV)", ex_f.to_csv(index=False, sep=";", decimal=",").encode("utf-8-sig"),
                                   file_name=f"financeiro_{ind_f}.csv", mime="text/csv", key=f"fin_csv_{ind_f}")
            bruto_f = ff.drop(columns=["_c", "rotulo", "nivel_pdd"] + [c for c in ff.columns if c.endswith("_ant")],
                              errors="ignore")
            st.download_button("⬇️ Base completa do grupo (XLSX: uma aba por indicador + base)",
                               excel_bytes({**tabs_fin, "base": bruto_f}), file_name="financeiro_base.xlsx", key="fin_xlsx",
                               mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet")

# -------- Juros: estrutura a termo (ANBIMA)
if not EUA:          # sem dados equivalentes nos EUA
    with aba_jur:
        pj, vj = carregar_juros(versao_arquivos(DADOS.parent / "juros" / "parametros.csv",
                                                DADOS.parent / "juros" / "vertices.csv.gz"))
        if pj.empty:
            st.info("Curvas ainda não coletadas (saidas/juros/). Rode o workflow juros-ettj.")
        else:
            datas_j = sorted(pj["data"].unique())
            st.caption(f"Fonte: ANBIMA, Estrutura a Termo das Taxas de Juros (modelo Svensson), {len(datas_j)} datas de "
                       f"{pd.Timestamp(datas_j[0]).date()} a {pd.Timestamp(datas_j[-1]).date()}. Taxas % a.a., base 252. "
                       "A ANBIMA libera gratuitamente só os últimos 5 dias úteis: o histórico cresce com a coleta diária.")
            tipos_c = {"Prefixada": "pre", "Real (IPCA)": "ipca", "Inflação implícita": "implicita"}
            j1, j2, j3 = st.columns([1.3, 1.2, 2])
            tc_rot = j1.radio("Curva", list(tipos_c), key="jur_curva")
            tc = tipos_c[tc_rot]
            base_j = j2.selectbox("Data", datas_j[::-1], format_func=lambda d: pd.Timestamp(d).strftime("%d/%m/%Y"),
                                  key="jur_data")
            prazos_cmp = {"1 semana antes": 7, "1 mês antes": 30, "3 meses antes": 91, "6 meses antes": 182,
                          "1 ano antes": 365}
            sel_cmp = j3.multiselect("Comparar com", list(prazos_cmp), default=["1 mês antes", "1 ano antes"],
                                     key="jur_cmp")
            extra = j3.multiselect("Outras datas", [d for d in datas_j[::-1] if d != base_j],
                                   format_func=lambda d: pd.Timestamp(d).strftime("%d/%m/%Y"), key="jur_extra")
            max_anos = st.slider("Prazo máximo (anos)", 2, 30, 10, key="jur_max",
                                 help="A ANBIMA publica vértices prefixados até 10 anos; além disso a curva prefixada e a "
                                      "implícita são extrapolação do modelo Svensson.")
            anos = np.linspace(0.25, max_anos, 120)
            datas_plot, faltam = [(pd.Timestamp(base_j), "Data escolhida")], []
            for nome in sel_cmp:
                alvo = pd.Timestamp(base_j) - pd.Timedelta(days=prazos_cmp[nome])
                cand = [d for d in datas_j if pd.Timestamp(d) <= alvo]
                if cand:
                    datas_plot.append((pd.Timestamp(cand[-1]), nome))
                else:
                    faltam.append(nome)
            datas_plot += [(pd.Timestamp(d), "") for d in extra]
            fig = go.Figure()
            for d, rot in datas_plot:
                y = curva_juros(pj, d, tc, anos)
                fig.add_trace(go.Scatter(x=anos, y=y, mode="lines", name=f"{d.strftime('%d/%m/%Y')} {rot}".strip(),
                                         line=dict(width=4 if rot == "Data escolhida" else 2)))
            fig.update_layout(title=f"Curva {tc_rot.lower()} (% a.a.)", height=460, yaxis_tickformat=".2%",
                              xaxis_title="Prazo (anos)", legend=dict(orientation="h", y=-0.2),
                              margin=dict(t=40, b=20, l=10, r=10))
            st.plotly_chart(fig, width="stretch")
            if faltam:
                st.caption("Sem histórico suficiente para: " + ", ".join(faltam) + ".")
            # tabela de vértices e variação em pontos-base
            vert = [v for v in (0.5, 1, 2, 3, 5, 10, 15, 20, 30) if v <= max_anos]
            tab_j = pd.DataFrame({f"{d.strftime('%d/%m/%Y')} {rot}".strip(): curva_juros(pj, d, tc, vert) * 100
                                  for d, rot in datas_plot}, index=[f"{v:g} ano(s)" for v in vert])
            base_col = tab_j.columns[0]
            for c in tab_j.columns[1:]:
                tab_j[f"Δ vs {c} (bps)"] = (tab_j[base_col] - tab_j[c]) * 100
            st.dataframe(tab_j.round(2), width="stretch")
            # histórico de vértices
            st.subheader("Histórico de vértices")
            v_hist = st.multiselect("Vértices (anos)", [0.5, 1, 2, 3, 5, 10, 15, 20, 30], default=[1, 2, 5, 10],
                                    key="jur_vert")
            if v_hist:
                hist = pd.DataFrame({"data": [pd.Timestamp(d) for d in datas_j]})
                for v in v_hist:
                    hist[f"{v:g}a"] = [curva_juros(pj, d, tc, [v])[0] for d in datas_j]
                fig = go.Figure()
                for v in v_hist:
                    fig.add_trace(go.Scatter(x=hist["data"], y=hist[f"{v:g}a"], mode="lines+markers", name=f"{v:g} ano(s)"))
                fig.update_layout(title=f"{tc_rot}: vértices ao longo do tempo (% a.a.)", height=400,
                                  yaxis_tickformat=".2%", legend=dict(orientation="h", y=-0.2),
                                  margin=dict(t=40, b=20, l=10, r=10))
                st.plotly_chart(fig, width="stretch")
                exp_h = hist.assign(data=hist["data"].dt.date)
                st.download_button("⬇️ Histórico (CSV)", exp_h.to_csv(index=False, sep=";", decimal=",").encode("utf-8-sig"),
                                   file_name=f"ettj_{tc}_historico.csv", mime="text/csv", key="jur_csv")
            st.caption("Curvas reconstruídas pelos parâmetros Svensson divulgados pela ANBIMA (reproduzem os vértices "
                       "publicados). Inflação implícita = (1 + prefixada) ÷ (1 + real) − 1.")

# -------- Juros e macro dos EUA (Treasuries, CPI, PIB, dólar; FRED) — módulo app/eua_macro_aba.py
if EUA:
    with aba_jur:
        eua_macro_mod().render_juros()
    with aba_mac:
        eua_macro_mod().render_macro(excel_bytes)

# -------- Macro: inflação, PIB e câmbio (IBGE/BCB)
if not EUA:          # sem dados equivalentes nos EUA
    with aba_mac:
        macro_aba = macro_mod()
        macro_aba.render(DADOS.parent / "macro", excel_bytes)

# -------- Dados e download
with aba_dados:
    cols_out = [c for c in IDENT if c in filtrado.columns] + sel_met + ["notas"]
    todas = st.checkbox("Incluir todas as métricas no download (não só as selecionadas)")
    if todas:
        cols_out = [c for c in IDENT if c in filtrado.columns] + list(METRICAS) + ["notas"]
    if "fonte_construcao" in filtrado.columns and filtrado["fonte_construcao"].fillna("").ne("").any():
        cols_out.append("fonte_construcao")
    if "fonte_ajuste_eletrico" in filtrado.columns and filtrado["fonte_ajuste_eletrico"].fillna("").ne("").any():
        cols_out.append("fonte_ajuste_eletrico")
    saida = filtrado[cols_out].sort_values(["empresa", "dt_fim"])
    mets_vis = [c for c in cols_out if c in METRICAS]
    # base americana: ~10x a brasileira; tabela na tela limitada e arquivos gerados só sob pedido (cada interação
    # do app roda todas as abas, e montar CSV/XLSX de 100 mil+ linhas a cada clique estoura a memória do Streamlit)
    LIM_EXIB = 5000
    exib = saida.head(LIM_EXIB) if EUA else saida
    st.dataframe(para_exibir(exib, mets_vis), column_config=fmt_tabela(exib, mets_vis), hide_index=True,
                 width="stretch", height=500)
    if EUA and len(saida) > LIM_EXIB:
        st.caption(f"Mostrando as primeiras {LIM_EXIB:,} de {len(saida):,} linhas (use os filtros da barra lateral); "
                   "o download traz todas.".replace(",", "."))
    st.caption(f"Valores em {MOEDA} milhões; margens e retornos em %. No download, percentuais vão como fração (0,15 = 15%).")
    gerar = (not EUA) or st.toggle("Preparar arquivos para download (CSV e XLSX)", key="dl_eua",
                                   help="Na base americana os arquivos são grandes: só são gerados quando você pede.")
    if gerar:
        rot_cols = {c: (ROT[c] + (f" ({UN})" if METRICAS[c][1] == "mm" else "")) for c in mets_vis}
        exp = saida.assign(dt_fim=saida["dt_fim"].dt.date).rename(columns=rot_cols)
        c1, c2 = st.columns(2)
        c1.download_button("⬇️ Baixar CSV", exp.to_csv(index=False, sep=";", decimal=",").encode("utf-8-sig"),
                           file_name=f"universo_{tipo}.csv", mime="text/csv")
        abas = {"dados": exp}
        if not cob.empty:
            abas["cobertura"] = cob[cob["cnpj"].astype(str).str.zfill(14).isin(saida["cnpj"].astype(str).str.zfill(14))]
        abas["fontes"] = pd.DataFrame({"item": ["Fonte", "Gerado em", "Definições"],
                                       "descrição": ["SEC EDGAR (XBRL companyfacts, 10-K/10-Q) + classificação SEC/SIC"
                                                     if EUA else "CVM Dados Abertos (DFP/ITR) + classificação setorial B3",
                                                     str(meta.get("gerado_em", "")) if meta is not None else "",
                                                     "Ver aba Metodologia do app / README do repositório"]})
        c2.download_button("⬇️ Baixar XLSX", excel_bytes(abas), file_name=f"universo_{tipo}.xlsx",
                           mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet")

# -------- Cobertura
with aba_cob:
    if cob.empty:
        st.info("cobertura.csv não encontrado.")
    else:
        c = cob.copy()
        for col, sel in (("setor", sel_setor), ("subsetor", sel_sub), ("segmento", sel_seg)):
            if sel:
                c = c[c[col].isin(sel)]
        st.dataframe(c, hide_index=True, width="stretch", height=520)
        st.caption("Inclui empresas sem dados, emissores estrangeiros, SPACs, bancos e seguradoras, com o motivo."
                   if EUA else "Inclui empresas sem dados e os bancos e seguradoras (fora da v1), com o motivo.")

# -------- Metodologia
with aba_met:
  if EUA:
    eua.metodologia()
  else:
    st.markdown("""
**Fontes.** Demonstrações: CVM Dados Abertos (DFP anual e ITR trimestral). Universo e classificação setorial:
lista de empresas listadas da B3. Bancos e seguradoras fora da v1 (plano de contas de instituição financeira);
shoppings, imóveis, holdings e demais empresas do setor "Financeiro" da B3 estão incluídos.

**Períodos.**
- *Anual*: DFP; usa a versão reapresentada (exercício anterior da DFP seguinte) quando existe.
- *Trimestral*: balanço = posição na data. Fluxos (DRE e DFC) por diferença dos acumulados:
  2T = 6M − 3M, 3T = 9M − 6M, 4T = anual − 9M.
- *LTM*: soma dos últimos 4 trimestres consecutivos + balanço do último trimestre.

**Definições próprias (não são números reportados pelas empresas).**
- EBITDA = EBIT (DRE 3.05) + D&A + impairment da DFC (perda ou reversão ao valor recuperável de ativos não
  financeiros, inclusive ágio; fora recebíveis, estoques, tributos e ativos biológicos). Quando a DFC junta D&A e
  impairment numa linha, o impairment está dentro da D&A (com nota). Não inclui outros ajustes que as empresas divulgam.
- EBITDA pré-IFRS 16 = EBITDA − pagamentos de arrendamento (principal + juros) da DFC; antes de 2019 = EBITDA.
- Concessões do modelo do intangível (segmentos B3: exploração de rodovias, água e saneamento, serviços de apoio e
  armazenagem, transporte ferroviário e hidroviário; energia elétrica fica de fora por ter ativo financeiro/de
  contrato): a receita de construção da infraestrutura (ICPC 01) sai da receita líquida e o custo de construção sai
  do CPV. Se a empresa reconhece margem de construção (receita > custo), a margem sai do lucro bruto, EBIT, EBITDA e
  (líquida de 34% de IR/CS) do lucro líquido; múltiplos, margens e ROIC usam os valores ajustados. Fonte, em ordem:
  subconta da DRE (3.01/3.02) → DVA (7.01.03 "Receitas refs. à construção de ativos próprios", aceita só quando a
  receita líquida da DRE a comporta; o custo vem da DVA ou é receita − margem da DFC) → custo da DRE + margem da DFC
  (sem margem identificada, receita = custo). Sem receita de construção identificada, só a margem de construção da DFC
  (se houver) sai dos lucros. Coluna `fonte_construcao` mostra a origem; os valores reportados seguem disponíveis
  (grupo "Concessões (construção)").
- Setor elétrico — EBITDA ajustado (≈ EBITDA regulatório/caixa). No IFRS (CPC 47 / ICPC 01) o EBITDA de transmissora
  inclui receita de construção, remuneração e correção monetária (IPCA/IGP-M) do ativo de contrato, que não são caixa;
  o de distribuidora inclui a atualização do ativo financeiro indenizável (VNR). A B3 põe todas as elétricas num só
  segmento, então o tipo de cada empresa vem de curadoria (`eletricas.yaml`):
  transmissão: EBITDA + custo de construção − Δ ativo de contrato/concessão do balanço (circulante + não circulante),
  o que tira construção, remuneração e correção e deixa a RAP faturada + O&M − custos operacionais; o custo de
  construção vem da DRE/DVA (ou, sem ela, do investimento em ativo de contrato da DFC). Distribuição: EBITDA − VNR
  (ajuste não-caixa da DFC; supõe o VNR na receita operacional). Geração com bonificação pela outorga: − Δ do saldo
  da bonificação. Integradas com distribuição: só VNR/bonificação, transmissão sem ajuste (o consolidado mistura as
  duas). Em períodos com aquisição de SPEs o Δ do ativo inclui a consolidação (nota em `fonte_ajuste_eletrico`).
  É aproximação, não o EBITDA regulatório divulgado pelas empresas; covenants de debêntures têm definição própria.
  Indicadores: DL c/ arrend. ÷ EBITDA ajustado e (EBITDA ajustado − IR − arrend.) ÷ serviço da dívida.
- Alavancagem em pares consistentes: DL c/ arrendamento ÷ EBITDA IFRS 16; DL s/ arrendamento ÷ EBITDA pré-IFRS 16.
- Capex orgânico = saídas para imobilizado (inclui propriedade para investimento), intangível e ativo de
  contrato/concessão no bloco de investimento.
- FCO comparável = FCO da DFC (6.01) sem juros e dividendos/JCP pagos ou recebidos que a empresa classifica no
  operacional (o CPC 03 permite operacional, investimento ou financiamento). Assim, a diferença para o EBITDA fica
  em capital de giro e IR/CSLL pagos. Juros recebidos que só estão dentro do lucro (sem linha própria) não saem.
  O FCO reportado continua disponível.
- FCL pós-arrendamento = FCO comparável − capex orgânico líquido de vendas − arrendamento pago (principal + juros),
  contado uma vez. Como o FCO comparável não tem juros, o FCL é antes do serviço da dívida.
- NCAV (Graham) = ativo circulante − passivo total.
- Prazos médios (dias; 365 no anual/LTM, 90 no trimestral), com saldos médios inicial e final (CP + LP):
  PMR = dias ÷ (receita ÷ clientes); PME = dias ÷ (CPV ÷ estoques); PMP = dias ÷ (compras ÷ fornecedores),
  compras = estoque final − estoque inicial + CPV. Ciclo operacional = PME + PMR; financeiro = operacional − PMP.
- Mercado na visão do setor/ranking/dados: anual = preço do fim do exercício; trimestral e LTM = múltiplo na data de
  fim do trimestre, com os documentos já divulgados até ali (por isso pode usar o trimestre anterior).
  Setor agregado: Σ valor de mercado (ou EV) ÷ Σ PL, lucro 12m ou EBITDA 12m das empresas com os dois dados.
- Giro do ativo = receita ÷ ativo total; alavancagem = ativo total ÷ PL (fim do período).
- Conversão de caixa = FCO comparável ÷ EBITDA IFRS 16 do mesmo período (vazio com EBITDA ≤ 0).
- Cobertura da dívida (fluxos do mesmo período; IR = IR + CSLL **pagos** na DFC; arrend. = principal + juros de
  arrendamento pagos; juros e principal = dívida ex-arrendamento, pagos na DFC; capex = capex orgânico + compra de
  frota/ativo para locação no FCO, líquida do valor contábil dos ativos vendidos, caso das locadoras):
  EBIT ÷ resultado financeiro líquido (vazio se o resultado financeiro for ≥ 0);
  (EBITDA − IR − arrend.) ÷ juros; (EBITDA − IR − arrend.) ÷ (juros + principal);
  (EBITDA − IR − capex − arrend.) ÷ (juros + principal); (EBITDA − IR − capex) ÷ (juros + principal + arrend.);
  (EBITDA − IR) ÷ (juros + principal + arrend.).
  Vazios com denominador ≤ 0 (empresa sem dívida paga no período). Linha da DFC que junta principal e juros
  entra só no serviço total: o índice só de juros fica vazio. Juros capitalizados no investimento (6.02) ficam
  fora. IR, juros e principal não identificados na DFC = 0 (com nota). Setor = Σ numerador ÷ Σ denominador.
  Dívida e arrendamento na mesma linha da DFC: só os índices com arrendamento no denominador são calculados; os
  índices com arrendamento no numerador ficam vazios. IR pago aceita linhas "pagos e retidos".
- DuPont: ROE (lucro líquido ÷ PL) = margem líquida × giro × alavancagem.
- Graus de alavancagem também na visão do setor (por empresa e, sem recorte, por setor sobre os agregados).
- Crescimento contra o mesmo período do ano anterior (anual ou LTM). GAO = Δ% EBIT ÷ Δ% receita;
  GAF = Δ% lucro líquido ÷ Δ% EBIT; GAT = GAO × GAF. Vazios quando o denominador varia menos de 1%.

**Múltiplos de mercado (aba Empresa).** Preço de fechamento da série histórica da B3 (COTAHIST), sem ajuste
por eventos corporativos, × ações em circulação (composição do capital do ITR/DFP, sem tesouraria), somando
todas as classes. Classe sem negócio na data: último preço negociado; classe que nunca negociou: preço da outra
classe (indicado nas notas). Trimestral = ponto no tempo (último documento divulgado até a data, lucro e
EBITDA dos 12 meses); anual = preço do fim do exercício com os números do exercício.
P/VPA e P/L usam PL e lucro dos controladores; EV inclui a participação de não controladores.
Dividend yield = dividendos + JCP pagos nos 12 meses (DFC, regime de caixa, só acionistas da controladora) ÷ valor de mercado.

**Bancos e seguradoras (aba Financeiro).** Base separada, a partir das demonstrações completas (plano de
contas próprio). Contas identificadas por descrição, porque os códigos mudam entre instituições e anos.
ROE = lucro dos controladores ÷ PL dos controladores (PL total − não controladores), no fim do ano.
Eficiência e carteira de crédito são aproximações próprias. Basileia e inadimplência não constam da DFP.
Nas abas Financeiro e Empresa, a série segue a periodicidade da barra lateral: anual (DFP), trimestral (ITR; fluxo do trimestre
por diferença dos acumulados; ROE, ROA e custo do crédito anualizados × 4) e LTM (4 trimestres somados, saldos do
último trimestre). Múltiplos trimestrais no ponto no tempo: PL do último ITR/DFP divulgado e lucro LTM.
Abertura do ativo: carteira de crédito bruta, (−) provisão, TVM e derivativos e demais ativos (resíduo).
Crescimento (PL dos controladores, carteira de crédito, ativo total, despesa de PDD) = exercício ÷ exercício anterior − 1 (LTM e trimestral: contra o mesmo período do ano anterior);
o gráfico mostra o crescimento acumulado em base 100 (por instituição e agregado do grupo em amostra casada).

**Limitações conhecidas.**
- D&A, capex e arrendamento vêm de busca textual na DFC: vazio = não identificado (nunca zero).
- Arrendamento somado à dívida numa linha só fica vazio (com nota).
- Locadoras de veículos registram a compra de frota no FCO, não no capex.
- Universo só com empresas listadas hoje: medianas históricas têm viés de sobrevivência.
- Trimestres podem divergir do anual quando há reapresentação (o anual usa a versão reapresentada).
""")


# devolve ao sistema a memória liberada nesta execução (base americana é grande; Streamlit gratuito)
if EUA:
    eua.liberar_memoria()
