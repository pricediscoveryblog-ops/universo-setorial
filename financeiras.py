"""
financeiras.py — Price Discovery
Métricas de bancos e seguradoras a partir da estrutura completa das demonstrações (DFP),
que usam plano de contas próprio. Os códigos mudam entre empresas e anos, então a
identificação é por descrição da conta, sempre no nível hierárquico esperado.
Todas as métricas são definições próprias — conferir com os releases das instituições.
"""
import re
import unicodedata
import warnings

import numpy as np
import pandas as pd

warnings.filterwarnings("ignore", message="This pattern is interpreted as a regular expression")


def _n(s) -> str:
    s = unicodedata.normalize("NFKD", str(s)).encode("ascii", "ignore").decode().lower()
    return re.sub(r"\s+", " ", re.sub(r"[()]", "", s)).strip()  # "(Perda) esperada" -> "perda esperada"


def _nivel(c: str) -> int:
    return str(c).count(".") + 1


def _topo(df: pd.DataFrame) -> pd.DataFrame:
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
    d = d[d["d"].str.contains(rx)]
    if excl:
        d = d[~d["d"].str.contains(excl)]
    return d.sort_values("cd_conta").iloc[0] if len(d) else None


def _soma(df, rx, prefixo=None, excl=None):
    d = df if prefixo is None else df[df["cd_conta"].str.startswith(prefixo)]
    d = d[d["d"].str.contains(rx)]
    if excl:
        d = d[~d["d"].str.contains(excl)]
    d = _topo(d)
    return (d["valor_rs_mm"].sum() if len(d) else np.nan), d


def metricas_empresa_ano(g: pd.DataFrame) -> tuple[dict, list]:
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

    # bancos: intermediação, crédito, serviços, despesas
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
        serv_fora = d[d["cd_conta"].str.startswith("3.04")]["valor_rs_mm"].sum() if len(d) else 0.0
        adm, d = _soma(dre, r"despesas? (?:de |com )?pessoal|despesas? (?:de )?administrativas|"
                            r"outras despesas (?:de )?administrativas|despesas gerais e administrativas", prefixo="3.04.")
        adm = adm if pd.notna(adm) and adm != 0 else np.nan
        m["despesas_pessoal_adm"] = adm
        reg("despesas_pessoal_adm", d)
        cred, d = _soma(bpa, r"operacoes de credito|emprestimos e adiantamentos a clientes|"
                             r"emprestimos e recebiveis de clientes|carteira de credito",
                        excl=r"provisao|perda|cessao|arrendamento mercantil a receber")
        m["carteira_credito"] = cred if (pd.notna(cred) and cred > 0) else np.nan
        reg("carteira_credito", d)
        dep, d = _soma(bpp, r"^depositos$|^depositos |depositos de clientes", excl=r"judiciais")
        m["depositos"] = dep if (pd.notna(dep) and dep > 0) else np.nan
        reg("depositos", d)
        rbif = m["resultado_bruto_intermediacao"]
        if pd.isna(pdd):
            mf = np.nan                       # PDD não aberta: não dá para isolar a margem antes da PDD
        elif pdd_em_302:
            mf = rbif - pdd                   # resultado bruto já está líquido da PDD: soma de volta
        else:
            mf = rbif                         # PDD fora do resultado bruto (em 3.04)
        m["margem_financeira_antes_pdd"] = mf
        receita_op = mf + (serv_fora or 0) if pd.notna(mf) else np.nan
        m["indice_eficiencia"] = (abs(adm) / receita_op) if pd.notna(adm) and receita_op and receita_op > 0 else np.nan
        m["custo_credito"] = (abs(pdd) / m["carteira_credito"]) if pd.notna(pdd) and pd.notna(m["carteira_credito"]) else np.nan
        m["credito_depositos"] = (m["carteira_credito"] / m["depositos"]) if pd.notna(m["carteira_credito"]) and pd.notna(m["depositos"]) else np.nan

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

    def div(a, b):
        return a / b if pd.notna(a) and pd.notna(b) and b != 0 else np.nan
    m["roe"] = div(m["lucro_controladores"], m["pl_controladores"])
    m["roa"] = div(m["lucro_liquido"], m["ativo_total"])
    m["pl_ativo"] = div(m["patrimonio_liquido"], m["ativo_total"])
    m["margem_liquida"] = div(m["lucro_liquido"], m["receita_total"])
    m["payout"] = div(m["dividendos_pagos"], m["lucro_controladores"]) if m["lucro_controladores"] and m["lucro_controladores"] > 0 else np.nan
    return m, aud


def calcular_financeiras(estrutura: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    linhas, auds = [], []
    est = estrutura.copy()
    est["cd_conta"] = est["cd_conta"].astype(str)
    for (cnpj, ex), g in est.groupby(["cnpj", "exercicio"]):
        m, aud = metricas_empresa_ano(g)
        ident = {k: g[k].iloc[0] for k in ("empresa", "ticker", "subsetor", "segmento", "base", "dt_fim") if k in g}
        linhas.append({"cnpj": cnpj, "exercicio": ex, **ident, **m})
        auds += [{"cnpj": cnpj, "exercicio": ex, "empresa": ident.get("empresa"), **a} for a in aud]
    return pd.DataFrame(linhas), pd.DataFrame(auds)
