"""
financeiras.py — Price Discovery
Métricas de bancos e seguradoras a partir da estrutura completa das demonstrações (DFP e ITR),
que usam plano de contas próprio. Os códigos mudam entre empresas e anos, então a
identificação é por descrição da conta, sempre no nível hierárquico esperado.
Todas as métricas são definições próprias — conferir com os releases das instituições.

Períodos (mesma regra das empresas não financeiras, em universo.py):
- anual : DFP do exercício
- trimestral : fluxos (DRE/DFC) por diferença dos acumulados (2T = 6M − 3M, 3T = 9M − 6M,
               4T = anual − 9M); balanço = posição na data. ROE, ROA e custo do crédito anualizados (× 4)
- ltm : soma dos últimos 4 trimestres consecutivos (fluxos) + balanço do último trimestre
"""
import re
import unicodedata
import warnings

import numpy as np
import pandas as pd

warnings.filterwarnings("ignore", message="This pattern is interpreted as a regular expression")

# fluxos: acumulados no exercício no documento; viram trimestre por diferença e LTM por soma
FLUXOS = ["receita_total", "lucro_liquido", "lucro_controladores", "resultado_bruto_intermediacao", "despesa_pdd",
          "receita_servicos", "receita_servicos_fora", "despesas_pessoal_adm", "margem_financeira_antes_pdd",
          "dividendos_pagos"]
# saldos: posição na data do documento
ESTOQUES = ["ativo_total", "patrimonio_liquido", "pl_controladores", "carteira_credito", "carteira_credito_bruta",
            "provisao_carteira", "tvm", "demais_ativos", "depositos"]
RAZOES = ["roe", "roa", "pl_ativo", "margem_liquida", "indice_eficiencia", "custo_credito", "credito_depositos",
          "payout", "carteira_bruta_ativo", "tvm_ativo", "demais_ativos_ativo", "cobertura_carteira"]

RX_CRED = (r"operacoes de credito|emprestimos e adiantamentos a clientes|emprestimos e recebiveis de clientes|"
           r"carteira de credito")
RX_CRED_EXCL = r"provisao|perda|cessao|arrendamento mercantil a receber"
# provisão para perdas com crédito (saldo redutor do ativo)
RX_PROV = (r"provisao para creditos|provisao para perdas|liquidacao duvidosa|perdas? esperadas?|"
           r"perda esperada|risco de credito|perdas? (?:por|de|com) credito|\bpdd\b")
# títulos e valores mobiliários e derivativos (COSIF: uma linha; IFRS: uma linha em cada categoria de ativo financeiro)
RX_TVM = (r"titulos e valores mobiliarios|instrumentos financeiros derivativos|^derivativos|"
          r"aplicacoes em titulos")
RX_TVM_EXCL = r"provisao|perda"


def _n(s) -> str:
    s = unicodedata.normalize("NFKD", str(s)).encode("ascii", "ignore").decode().lower()
    return re.sub(r"\s+", " ", re.sub(r"[()]", "", s)).strip()  # "(Perda) esperada" -> "perda esperada"


def _nivel(c: str) -> int:
    return str(c).count(".") + 1


def _topo(df: pd.DataFrame) -> pd.DataFrame:
    if df.empty:
        return df
    cods = set(df["cd_conta"])
    def anc(c):
        p = c.split(".")
        return any(".".join(p[:i]) in cods for i in range(1, len(p)))
    return df[~df["cd_conta"].map(anc)] if len(df) else df


def _um(df, rx, nivel=None, excl=None, prefixo=None):
    """Primeira conta (menor código) que casa com rx no nível pedido."""
    d = df
    if prefixo:
        d = d[d["cd_conta"].str.startswith(prefixo)]
    if nivel:
        d = d[d["nivel"] == nivel]
    d = d.loc[d["d"].str.contains(rx).astype(bool)]
    if excl:
        d = d.loc[~d["d"].str.contains(excl).astype(bool)]
    return d.sort_values("cd_conta").iloc[0] if len(d) else None


def _soma(df, rx, prefixo=None, excl=None):
    d = df if prefixo is None else df[df["cd_conta"].str.startswith(prefixo)]
    d = d.loc[d["d"].str.contains(rx).astype(bool)]
    if excl:
        d = d.loc[~d["d"].str.contains(excl).astype(bool)]
    d = _topo(d)
    return (d["valor_rs_mm"].sum() if len(d) else np.nan), d


def _pai(c: str) -> str:
    return c.rsplit(".", 1)[0] if "." in c else ""


def abertura_ativo(bpa: pd.DataFrame, m: dict, reg) -> None:
    """Carteira de crédito bruta, provisão (saldo), TVM e derivativos e demais ativos (= ativo total − carteira
    líquida − TVM). A provisão é somada de volta à carteira quando está DENTRO da conta de crédito (COSIF: conta
    de crédito líquida, com a provisão como subconta); quando é conta irmã da carteira (IFRS: provisão ao lado
    das operações de crédito), a carteira já está bruta e a provisão só entra no saldo."""
    for k in ("carteira_credito", "carteira_credito_bruta", "provisao_carteira", "tvm", "demais_ativos"):
        m[k] = np.nan
    if bpa.empty:
        return
    cred, dcred = _soma(bpa, RX_CRED, excl=RX_CRED_EXCL)
    m["carteira_credito"] = cred if (pd.notna(cred) and cred > 0) else np.nan
    reg("carteira_credito", dcred)
    if pd.notna(m["carteira_credito"]):
        prov = _topo(bpa[bpa["d"].str.contains(RX_PROV) & (bpa["valor_rs_mm"] < 0)])
        cods = list(dcred["cd_conta"])
        pais = {_pai(c) for c in cods}
        # máscaras booleanas explícitas: com prov vazio, prov[Series vazia] seria seleção de colunas
        em_cred = prov["cd_conta"].map(lambda c: any(c.startswith(x + ".") for x in cods)).astype(bool)
        dentro = prov.loc[em_cred]
        irma = prov.loc[~em_cred & prov["cd_conta"].map(_pai).isin(pais).astype(bool)]
        m["carteira_credito_bruta"] = cred - dentro["valor_rs_mm"].sum()
        # sem conta de provisão identificada: saldo vazio (não zero)
        m["provisao_carteira"] = -(dentro["valor_rs_mm"].sum() + irma["valor_rs_mm"].sum()) if len(dentro) + len(irma) \
            else np.nan
        reg("provisao_carteira", pd.concat([dentro, irma]))
    tvm, dtvm = _soma(bpa, RX_TVM, excl=RX_TVM_EXCL)
    if pd.notna(tvm) and tvm > 0:
        m["tvm"] = tvm
        reg("tvm", dtvm)
    if pd.notna(m["carteira_credito_bruta"]) and pd.notna(m["tvm"]) and pd.notna(m.get("ativo_total")):
        prov_ = m["provisao_carteira"] if pd.notna(m["provisao_carteira"]) else 0.0
        dem = m["ativo_total"] - (m["carteira_credito_bruta"] - prov_) - m["tvm"]
        m["demais_ativos"] = dem if dem >= 0 else np.nan  # negativo = sobreposição na captura: não informa


def metricas_documento(g: pd.DataFrame) -> tuple[dict, list]:
    """Valores de um documento (DFP ou ITR): fluxos acumulados no exercício até a data e saldos na data."""
    g = g.copy()
    g["d"] = g["ds_conta"].map(_n)
    g["nivel"] = g["cd_conta"].map(_nivel)
    dre, bpa, bpp = g[g["demonstracao"] == "DRE"], g[g["demonstracao"] == "BPA"], g[g["demonstracao"] == "BPP"]
    m, aud = {}, []

    def reg(nome, linha_ou_df):
        if linha_ou_df is None:
            return
        rows = [linha_ou_df] if isinstance(linha_ou_df, pd.Series) else [r for _, r in linha_ou_df.iterrows()]
        for r in rows:
            aud.append({"metrica": nome, "demonstracao": r["demonstracao"], "cd_conta": r["cd_conta"],
                        "ds_conta": r["ds_conta"], "valor_rs_mm": r["valor_rs_mm"]})

    r301 = dre[dre["cd_conta"] == "3.01"]
    d301 = r301["d"].iloc[0] if len(r301) else ""
    layout = ("banco" if "intermediacao financeira" in d301 else
              "seguradora" if ("seguradora" in d301 or "operacoes" in d301) else "geral")
    m["layout_dfp"] = layout
    m["receita_total"] = r301["valor_rs_mm"].iloc[0] if len(r301) else np.nan

    # lucro líquido consolidado (nível 2, sem "antes") e parcela dos controladores
    ll = _um(dre, r"lucro|prejuizo|resultado liquido", nivel=2,
             excl=r"antes|continuadas|descontinuadas|por acao|participac")
    # o lucro do período é a ÚLTIMA conta de nível 2 desse tipo
    cand = dre[(dre["nivel"] == 2) & dre["d"].str.contains(r"lucro|prejuizo") &
               dre["d"].str.contains(r"periodo|exercicio") & ~dre["d"].str.contains(r"antes|por acao")]
    if len(cand):
        ll = cand.sort_values("cd_conta").iloc[-1]
    m["lucro_liquido"] = ll["valor_rs_mm"] if ll is not None else np.nan
    reg("lucro_liquido", ll)
    ctrl = None
    if ll is not None:
        ctrl = _um(dre, r"controlador", prefixo=ll["cd_conta"] + ".", excl=r"nao controlador")
    # algumas empresas deixam a abertura por controlador zerada: nesse caso vale o lucro total
    usa_ctrl = ctrl is not None and pd.notna(ctrl["valor_rs_mm"]) and (ctrl["valor_rs_mm"] != 0 or m["lucro_liquido"] == 0)
    m["lucro_controladores"] = ctrl["valor_rs_mm"] if usa_ctrl else m["lucro_liquido"]
    reg("lucro_controladores", ctrl if usa_ctrl else ll)

    # balanço
    at = bpa[bpa["cd_conta"] == "1"]
    m["ativo_total"] = at["valor_rs_mm"].iloc[0] if len(at) else np.nan
    pl = _um(bpp, r"patrimonio liquido", nivel=2)
    m["patrimonio_liquido"] = pl["valor_rs_mm"] if pl is not None else np.nan
    reg("patrimonio_liquido", pl)
    # PL dos controladores = PL total − participação de não controladores (mais robusto que buscar
    # a linha "controladores", que em alguns anos é o lucro atribuível dentro do PL)
    nc = _um(bpp, r"nao controlador", prefixo=pl["cd_conta"] + ".") if pl is not None else None
    m["pl_controladores"] = (m["patrimonio_liquido"] - nc["valor_rs_mm"]) if nc is not None and pd.notna(nc["valor_rs_mm"]) \
        else m["patrimonio_liquido"]
    reg("pl_nao_controladores", nc)

    for k in ("resultado_bruto_intermediacao", "despesa_pdd", "receita_servicos", "receita_servicos_fora",
              "despesas_pessoal_adm", "carteira_credito", "carteira_credito_bruta", "provisao_carteira", "tvm",
              "demais_ativos", "depositos", "margem_financeira_antes_pdd"):
        m[k] = np.nan
    # bancos: intermediação, crédito, serviços, despesas, abertura do ativo
    if layout == "banco":
        rb = _um(dre, r"resultado bruto", nivel=2)
        m["resultado_bruto_intermediacao"] = rb["valor_rs_mm"] if rb is not None else np.nan
        reg("resultado_bruto_intermediacao", rb)
        # PDD pode estar dentro das despesas de intermediação (3.02) ou em outras despesas (3.04)
        pdd, d = _soma(dre, r"provisao para creditos|liquidacao duvidosa|perdas? (?:de |por )?credito|credito esperad|"
                            r"perdas? esperadas?|risco de credito|pdd", excl=r"reversao de outras|outros ativos|demais ativos")
        pdd = pdd if pd.notna(pdd) and pdd != 0 else np.nan
        m["despesa_pdd"] = pdd
        reg("despesa_pdd", d)
        pdd_em_302 = len(d) > 0 and d["cd_conta"].str.startswith("3.02").any()
        serv, d = _soma(dre, r"prestacao de servicos|tarifas bancarias|receitas de servicos")
        serv = serv if pd.notna(serv) and serv != 0 else np.nan
        m["receita_servicos"] = serv
        reg("receita_servicos", d)
        m["receita_servicos_fora"] = d[d["cd_conta"].str.startswith("3.04")]["valor_rs_mm"].sum() if len(d) else 0.0
        adm, d = _soma(dre, r"despesas? (?:de |com )?pessoal|despesas? (?:de )?administrativas|"
                            r"outras despesas (?:de )?administrativas|despesas gerais e administrativas", prefixo="3.04.")
        adm = adm if pd.notna(adm) and adm != 0 else np.nan
        m["despesas_pessoal_adm"] = adm
        reg("despesas_pessoal_adm", d)
        abertura_ativo(bpa, m, reg)
        dep, d = _soma(bpp, r"^depositos$|^depositos |depositos de clientes", excl=r"judiciais")
        m["depositos"] = dep if (pd.notna(dep) and dep > 0) else np.nan
        reg("depositos", d)
        rbif = m["resultado_bruto_intermediacao"]
        if pd.isna(pdd):
            mf = np.nan  # PDD não aberta: não dá para isolar a margem antes da PDD
        elif pdd_em_302:
            mf = rbif - pdd  # resultado bruto já está líquido da PDD: soma de volta
        else:
            mf = rbif  # PDD fora do resultado bruto (em 3.04)
        m["margem_financeira_antes_pdd"] = mf

    # dividendos e JCP pagos aos acionistas da controladora (DFC; positivo = pago)
    dfc = g[g["demonstracao"].isin(["DFC_MI", "DFC_MD"])]
    if len(dfc):
        dv = dfc[~dfc["cd_conta"].str.match(r"^6\.01\.0[12]|^6\.02") &
                 dfc["d"].str.contains(r"dividend|capital proprio|\bjcp\b") & (dfc["valor_rs_mm"] < 0) &
                 ~dfc["d"].str.contains(r"receb|nao controlador|minoritari|a pagar")]
        dv = _topo(dv)
        m["dividendos_pagos"] = -dv["valor_rs_mm"].sum() if len(dv) else 0.0
        reg("dividendos_pagos", dv)
    else:
        m["dividendos_pagos"] = np.nan
    return m, aud


def razoes(m: dict, fator: float = 1.0) -> dict:
    """Índices do período. fator anualiza as razões fluxo ÷ saldo no trimestral (× 4); razões entre fluxos
    (margem, eficiência, payout) e entre saldos não mudam."""
    def div(a, b):
        return a / b if pd.notna(a) and pd.notna(b) and b != 0 else np.nan
    m = dict(m)
    m["roe"] = div(m["lucro_controladores"], m["pl_controladores"]) * fator
    m["roa"] = div(m["lucro_liquido"], m["ativo_total"]) * fator
    m["pl_ativo"] = div(m["patrimonio_liquido"], m["ativo_total"])
    m["margem_liquida"] = div(m["lucro_liquido"], m["receita_total"])
    mf, serv_fora, adm = m.get("margem_financeira_antes_pdd"), m.get("receita_servicos_fora"), m.get("despesas_pessoal_adm")
    receita_op = mf + (serv_fora if pd.notna(serv_fora) else 0) if pd.notna(mf) else np.nan
    m["indice_eficiencia"] = (abs(adm) / receita_op) if pd.notna(adm) and pd.notna(receita_op) and receita_op > 0 else np.nan
    pdd = m.get("despesa_pdd")
    den = m["carteira_credito_bruta"] if pd.notna(m.get("carteira_credito_bruta")) else m.get("carteira_credito")
    m["custo_credito"] = (abs(pdd) / den * fator) if pd.notna(pdd) and pd.notna(den) and den > 0 else np.nan
    m["credito_depositos"] = div(m["carteira_credito"], m["depositos"])
    lc = m["lucro_controladores"]
    m["payout"] = div(m["dividendos_pagos"], lc) if pd.notna(lc) and lc > 0 else np.nan
    at = m["ativo_total"] if pd.notna(m["ativo_total"]) and m["ativo_total"] > 0 else np.nan
    m["carteira_bruta_ativo"] = div(m["carteira_credito_bruta"], at)
    m["tvm_ativo"] = div(m["tvm"], at)
    m["demais_ativos_ativo"] = div(m["demais_ativos"], at)
    m["cobertura_carteira"] = div(m["provisao_carteira"], m["carteira_credito_bruta"])
    return m


def metricas_empresa_ano(g: pd.DataFrame) -> tuple[dict, list]:
    """Compatibilidade: valores e índices de um documento anual."""
    m, aud = metricas_documento(g)
    return razoes(m), aud


def _trimestre(dt_ini: pd.Timestamp, dt_fim: pd.Timestamp) -> int:
    meses = (dt_fim.year - dt_ini.year) * 12 + dt_fim.month - dt_ini.month + 1
    return int(min(4, max(1, round(meses / 3))))


def calcular_financeiras(estrutura: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Uma linha por instituição e período: anual (DFP), trimestral e LTM (ITR + DFP como 4º trimestre)."""
    est = estrutura.copy()
    est["cd_conta"] = est["cd_conta"].astype(str)
    if "documento" not in est.columns:  # estrutura gerada antes da coleta de ITR: só DFP
        est["documento"] = "DFP"
    est["dt_fim"] = pd.to_datetime(est["dt_fim"])
    if "dt_ini" not in est.columns:
        est["dt_ini"] = est["dt_fim"] - pd.DateOffset(years=1) + pd.Timedelta(days=1)
    est["dt_ini"] = pd.to_datetime(est["dt_ini"])
    if "trimestre" not in est.columns:
        est["trimestre"] = 4
    linhas, auds = [], []
    for cnpj, ge in est.groupby("cnpj"):
        snaps = []
        for (doc, dfim), g in ge.groupby(["documento", "dt_fim"]):
            m, aud = metricas_documento(g)
            ident = {k: g[k].iloc[0] for k in ("empresa", "ticker", "subsetor", "segmento", "base") if k in g}
            dini = g["dt_ini"].min()
            tri = 4 if doc == "DFP" else _trimestre(dini, dfim)
            snaps.append({"doc": doc, "dt_fim": dfim, "dt_ini": dini, "exercicio": int(g["exercicio"].iloc[0]),
                          "trimestre": tri, "ident": ident, "m": m, "aud": aud})
        snaps.sort(key=lambda s: (s["dt_fim"], s["doc"] == "DFP"))

        def linha(s, tipo, m, periodo, fonte, notas, fator=1.0):
            r = razoes(m, fator)
            return {"cnpj": cnpj, "tipo": tipo, "exercicio": s["exercicio"], "trimestre": s["trimestre"],
                    "periodo": periodo, "dt_ini": s["dt_ini"].date(), "dt_fim": s["dt_fim"].date(), "fonte": fonte,
                    **s["ident"], **{k: r.get(k) for k in ["layout_dfp"] + FLUXOS + ESTOQUES + RAZOES},
                    "notas": "; ".join(n for n in notas if n)}

        # ---- anual: DFP
        for s in snaps:
            if s["doc"] == "DFP":
                linhas.append(linha(s, "anual", s["m"], str(s["exercicio"]), "DFP", []))
                auds += [{"cnpj": cnpj, "exercicio": s["exercicio"], "periodo": str(s["exercicio"]),
                          "documento": "DFP", "empresa": s["ident"].get("empresa"), **a} for a in s["aud"]]
        # ---- trimestral: acumulados do ano (ITR 1-3 + DFP como 4)
        ytd = {}
        for s in snaps:
            ytd[(s["exercicio"], s["trimestre"])] = s  # em duplicidade, fica o último (DFP prevalece no 4T)
        trimestres = []
        for (ex, k) in sorted(ytd):
            s = ytd[(ex, k)]
            m, notas = dict(s["m"]), []
            if k > 1:
                ant = ytd.get((ex, k - 1))
                if ant is None:
                    for f in FLUXOS:
                        m[f] = np.nan
                    notas.append(f"acumulado do {k - 1}T ausente: fluxos do trimestre indisponíveis")
                else:
                    for f in FLUXOS:
                        a, b = m.get(f), ant["m"].get(f)
                        m[f] = a - b if pd.notna(a) and pd.notna(b) else np.nan
            periodo = f"{k}T{str(ex)[2:]}"
            fonte = f"{s['doc']}" + ("" if k == 1 else " (dif. de acumulados)")
            row = linha(s, "trimestral", m, periodo, fonte, notas + ["ROE, ROA e custo do crédito anualizados (×4)"],
                        fator=4.0)
            linhas.append(row)
            trimestres.append((s, m, row))
            if s["doc"] == "ITR":
                auds += [{"cnpj": cnpj, "exercicio": ex, "periodo": f"acum. {periodo}", "documento": "ITR",
                          "empresa": s["ident"].get("empresa"), **a} for a in s["aud"]]
        # ---- LTM: 4 trimestres consecutivos
        for i in range(3, len(trimestres)):
            janela = trimestres[i - 3:i + 1]
            datas = [t[0]["dt_fim"] for t in janela]
            if any(not (80 <= (datas[j + 1] - datas[j]).days <= 100) for j in range(3)):
                continue
            s, m_ult, _ = janela[-1]
            m = dict(m_ult)
            for f in FLUXOS:
                vals = [t[1].get(f) for t in janela]
                m[f] = float(np.sum(vals)) if all(pd.notna(v) for v in vals) else np.nan
            problemas = [t[2]["periodo"] for t in janela if "ausente" in t[2]["notas"]]
            notas = [f"LTM inclui trimestres com alerta: {', '.join(problemas)}"] if problemas else []
            linhas.append(linha(s, "ltm", m, f"LTM {s['trimestre']}T{str(s['exercicio'])[2:]}",
                                "soma 4 trimestres", notas))
    return pd.DataFrame(linhas), pd.DataFrame(auds)
