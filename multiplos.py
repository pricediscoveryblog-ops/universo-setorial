"""
multiplos.py — Price Discovery
Múltiplos de mercado por empresa: P/VPA, P/L e EV/EBITDA (com e sem arrendamento).

Preços: série histórica oficial da B3 (COTAHIST), preço de fechamento SEM ajuste por eventos
corporativos — combinado com o número de ações da mesma época (composição do capital da CVM).

Regras (definidas pelo Artur):
- trimestral (ponto no tempo): em cada data, PL, dívida líquida e ações vêm do último ITR/DFP
  DIVULGADO até a data (data de recebimento na CVM); lucro e EBITDA são os LTM desse documento.
- anual: preço do último pregão do exercício com os números do próprio exercício.
- valor de mercado = Σ classes (ON e PN) × preço da classe. Classe sem negócio na data: último
  preço negociado antes; classe que nunca negociou: preço da outra classe. Sempre indicado em 'notas'.
- P/VPA = VM ÷ PL dos controladores; P/L = VM ÷ lucro LTM dos controladores (vazio se ≤ 0)
- dividend yield = dividendos + JCP pagos nos 12 meses (DFC, regime de caixa) ÷ VM
- EV = VM + dívida líquida + participação de não controladores (a valor contábil)
  EV/EBITDA IFRS 16 = (VM + DL c/ arrendamento + minoritários) ÷ EBITDA IFRS 16
  EV/EBITDA pré-IFRS 16 = (VM + DL s/ arrendamento + minoritários) ÷ EBITDA pré-IFRS 16
"""
import zipfile
from bisect import bisect_right

import numpy as np
import pandas as pd

import coletor_setorial as cs

URLS = ["https://bvmf.bmfbovespa.com.br/InstDados/SerHist/COTAHIST_A{ano}.ZIP"]
SUFIXOS = {b"3": "ON", b"4": "PN", b"5": "PN", b"6": "PN", b"7": "PN", b"8": "PN"}
DIAS_PRECO_VELHO = 90


def _baixar(ano: int, forcar: bool):
    destino = cs.CACHE / f"COTAHIST_A{ano}.ZIP"
    for u in URLS:
        try:
            return cs.baixar(u.format(ano=ano), destino, forcar=forcar)
        except Exception as e:  # noqa: BLE001
            print(f"  ! COTAHIST {ano}: {e}")
    return None


def ler_cotacoes(anos: list[int], raizes: set[str]) -> pd.DataFrame:
    """Fechamento diário (mercado à vista) dos papéis ON/PN cujas 4 primeiras letras estão em raizes."""
    rz = {r.encode() for r in raizes if r}
    tk, dt, pr, vol = [], [], [], []
    for ano in anos:
        p = _baixar(ano, forcar=(ano == max(anos) and not cs.OFFLINE))
        if p is None:
            continue
        with zipfile.ZipFile(p) as z:
            for nome in z.namelist():
                with z.open(nome) as f:
                    for ln in f:
                        if ln[:2] != b"01" or ln[24:27] != b"010":
                            continue
                        cod = ln[12:24].strip()
                        if len(cod) != 5 or cod[:4] not in rz or cod[4:5] not in SUFIXOS:
                            continue
                        fat = int(ln[210:217] or b"1") or 1
                        tk.append(cod.decode())
                        dt.append(ln[2:10].decode())
                        pr.append(int(ln[108:121]) / 100 / fat)
                        vol.append(int(ln[170:188]) / 100)
        print(f"  COTAHIST {ano}: {len(tk)} registros acumulados")
    df = pd.DataFrame({"ticker": tk, "data": pd.to_datetime(dt, format="%Y%m%d"), "preco": pr, "volume": vol})
    return df.sort_values(["ticker", "data"]).reset_index(drop=True)


class Precos:
    def __init__(self, cot: pd.DataFrame):
        cot = cot.sort_values(["ticker", "data"])
        self.serie = {t: (list(g["data"]), list(g["preco"])) for t, g in cot.groupby("ticker")}
        liq = cot.groupby("ticker")["volume"].sum()
        self.por_raiz = {}
        for t in self.serie:
            self.por_raiz.setdefault(t[:4], []).append(t)
        for r in self.por_raiz:
            self.por_raiz[r].sort(key=lambda t: -liq.get(t, 0))

    def classe(self, raiz: str, cls: str) -> list[str]:
        return [t for t in self.por_raiz.get(raiz, []) if SUFIXOS[t[4:5].encode()] == cls]

    def ultimo(self, tickers: list[str], d: pd.Timestamp):
        """Último preço até a data d, tentando os papéis da classe em ordem de liquidez."""
        for t in tickers:
            datas, precos = self.serie[t]
            i = bisect_right(datas, d)
            if i:
                return t, datas[i - 1], precos[i - 1]
        return None


def _valor_mercado(px: Precos, raiz: str, on: float, pn: float, d: pd.Timestamp):
    notas = []
    p_on = px.ultimo(px.classe(raiz, "ON"), d) if on > 0 else None
    p_pn = px.ultimo(px.classe(raiz, "PN"), d) if pn > 0 else None
    for nome, p, qt in (("ON", p_on, on), ("PN", p_pn, pn)):
        if qt > 0 and p is not None and (d - p[1]).days > DIAS_PRECO_VELHO:
            notas.append(f"{nome}: sem negócio desde {p[1].date()}, usado o último preço ({p[0]})")
    if on > 0 and p_on is None:
        alt = px.ultimo(px.classe(raiz, "PN"), d)
        if alt:
            p_on = alt
            notas.append(f"ON sem negociação: usado o preço da PN ({alt[0]})")
    if pn > 0 and p_pn is None:
        alt = px.ultimo(px.classe(raiz, "ON"), d)
        if alt:
            p_pn = alt
            notas.append(f"PN sem negociação: usado o preço da ON ({alt[0]})")
    if (on > 0 and p_on is None) or (pn > 0 and p_pn is None):
        return np.nan, None, None, notas + ["sem cotação para alguma classe: valor de mercado não calculado"]
    vm = (on * (p_on[2] if p_on else 0) + pn * (p_pn[2] if p_pn else 0)) / 1e6  # R$ milhões
    return vm, p_on, p_pn, notas


def _div(a, b):
    return a / b if pd.notna(a) and pd.notna(b) and b > 0 else np.nan


def _linha(meta, tipo, d, fund, flu, on, pn, px, raiz, notas_base):
    vm, p_on, p_pn, notas = _valor_mercado(px, raiz, on, pn, d)
    nc = fund.get("nao_controladores")
    nc = 0.0 if pd.isna(nc) else nc
    ev_c = vm + fund.get("div_liq_com_arrend") + nc if pd.notna(vm) and pd.notna(fund.get("div_liq_com_arrend")) else np.nan
    ev_s = vm + fund.get("div_liq_sem_arrend") + nc if pd.notna(vm) and pd.notna(fund.get("div_liq_sem_arrend")) else np.nan
    return {**meta, "tipo": tipo, "data": d.date(), "periodo_base": fund.get("periodo"), "dt_fim_base": fund.get("dt_fim"),
            "ticker_on": p_on[0] if p_on else None, "preco_on": p_on[2] if p_on else np.nan,
            "data_preco_on": p_on[1].date() if p_on else None,
            "ticker_pn": p_pn[0] if p_pn else None, "preco_pn": p_pn[2] if p_pn else np.nan,
            "data_preco_pn": p_pn[1].date() if p_pn else None,
            "acoes_on": on, "acoes_pn": pn, "valor_mercado": vm, "ev_com_arrend": ev_c, "ev_sem_arrend": ev_s,
            "pl_controladores": fund.get("pl_controladores"), "lucro_controladores_12m": flu.get("lucro_controladores"),
            "ebitda_12m": flu.get("ebitda"), "ebitda_pre_ifrs16_12m": flu.get("ebitda_pre_ifrs16"),
            "p_vpa": _div(vm, fund.get("pl_controladores")),
            "p_l": _div(vm, flu.get("lucro_controladores")),
            "ev_ebitda_ifrs16": _div(ev_c, flu.get("ebitda")),
            "ev_ebitda_pre_ifrs16": _div(ev_s, flu.get("ebitda_pre_ifrs16")),
            "dividendos_12m": flu.get("dividendos_pagos"),
            "dividend_yield": (flu.get("dividendos_pagos") / vm) if pd.notna(vm) and vm > 0
                              and pd.notna(flu.get("dividendos_pagos")) else np.nan,
            "notas": "; ".join(n for n in notas_base + notas if n)}


def _acoes_em(cnpj, dt_fim, compos):
    k = str(pd.Timestamp(dt_fim).date())
    if (cnpj, k) in compos:
        return compos[(cnpj, k)], ""
    ant = sorted((kk[1], v) for kk, v in compos.items() if kk[0] == cnpj and kk[1] <= k)
    if ant:
        return ant[-1][1], f"ações do documento de {ant[-1][0]} (sem composição em {k})"
    return None, "sem composição do capital"


def _financeiras(fin: pd.DataFrame, raiz_de: dict, compos: dict, px) -> list[dict]:
    """Bancos e seguradoras (base anual da DFP): P/VPA, P/L, dividend yield e ROE no fim do exercício.
    EV/EBITDA não se aplica a instituições financeiras."""
    linhas = []
    for cnpj, g in fin.groupby("cnpj"):
        raiz = raiz_de.get(cnpj, "")
        if not raiz or raiz not in px.por_raiz:
            continue
        meta = {"cnpj": cnpj, "empresa": g["empresa"].iloc[0], "ticker": g["ticker"].iloc[0], "setor": "Financeiro",
                "subsetor": g["subsetor"].iloc[0], "segmento": g["segmento"].iloc[0]}
        for _, r in g.iterrows():
            dfim = pd.Timestamp(r["dt_fim"])
            qt, nota = _acoes_em(cnpj, dfim, compos)
            if qt is None:
                continue
            vm, p_on, p_pn, notas = _valor_mercado(px, raiz, qt[0], qt[1], dfim)
            div_ = r.get("dividendos_pagos")
            linhas.append({**meta, "tipo": "anual", "data": dfim.date(), "periodo_base": str(r["exercicio"]),
                           "ticker_on": p_on[0] if p_on else None, "preco_on": p_on[2] if p_on else np.nan,
                           "data_preco_on": p_on[1].date() if p_on else None,
                           "ticker_pn": p_pn[0] if p_pn else None, "preco_pn": p_pn[2] if p_pn else np.nan,
                           "data_preco_pn": p_pn[1].date() if p_pn else None,
                           "acoes_on": qt[0], "acoes_pn": qt[1], "valor_mercado": vm,
                           "ev_com_arrend": np.nan, "ev_sem_arrend": np.nan,
                           "pl_controladores": r.get("pl_controladores"),
                           "lucro_controladores_12m": r.get("lucro_controladores"),
                           "p_vpa": _div(vm, r.get("pl_controladores")), "p_l": _div(vm, r.get("lucro_controladores")),
                           "ev_ebitda_ifrs16": np.nan, "ev_ebitda_pre_ifrs16": np.nan,
                           "dividendos_12m": div_,
                           "dividend_yield": (div_ / vm) if pd.notna(vm) and vm > 0 and pd.notna(div_) else np.nan,
                           "notas": "; ".join(n for n in [nota] + notas if n)})
    return linhas


def calcular(base: pd.DataFrame, univ: pd.DataFrame, disponib: dict, compos: dict, suc: dict,
             n_anos: int = 10, fin: pd.DataFrame | None = None) -> pd.DataFrame:
    if base.empty:
        return pd.DataFrame()
    b = base.copy()
    b["dt_fim"] = pd.to_datetime(b["dt_fim"])
    hoje = pd.Timestamp.today().normalize()
    anos = list(range(int(b["exercicio"].min()) - 1, hoje.year + 1))
    anos = [a for a in anos if a >= hoje.year - n_anos - 1]
    raiz_de = {r["cnpj_cvm"]: str(r.get("issuingCompany") or "")[:4] for _, r in univ.iterrows()}
    cot = ler_cotacoes(anos, set(raiz_de.values()))
    if cot.empty:
        print("  ! nenhuma cotação lida: múltiplos não calculados")
        return pd.DataFrame()
    px = Precos(cot)
    ultimo_pregao = cot["data"].max()
    datas_tri = [d for d in pd.date_range(f"{hoje.year - n_anos}-03-31", ultimo_pregao, freq="QE")]
    if not datas_tri or datas_tri[-1] < ultimo_pregao:
        datas_tri.append(ultimo_pregao)  # ponto mais recente ("hoje")

    linhas = []
    for cnpj, g in b.groupby("cnpj"):
        raiz = raiz_de.get(cnpj, "")
        if not raiz or raiz not in px.por_raiz:
            continue
        cands = [cnpj] + list(suc.get(cnpj, []))
        meta = {k: g[k].iloc[0] for k in ("cnpj", "empresa", "ticker", "setor", "subsetor", "segmento")}

        def acoes(dt_fim):
            k = str(pd.Timestamp(dt_fim).date())
            for c in cands:
                if (c, k) in compos:
                    return compos[(c, k)], ""
            ant = sorted((kk[1], v) for c in cands for kk, v in compos.items() if kk[0] == c and kk[1] <= k)
            if ant:
                return ant[-1][1], f"ações do documento de {ant[-1][0]} (sem composição em {k})"
            return None, "sem composição do capital"

        def divulgado(dt_fim):
            k = str(pd.Timestamp(dt_fim).date())
            for c in cands:
                if (c, k) in disponib:
                    return pd.Timestamp(disponib[(c, k)]), ""
            return pd.Timestamp(dt_fim) + pd.Timedelta(days=90), "data de divulgação estimada (fim do período + 90 dias)"

        # --- anual: preço do fim do exercício com os números do exercício
        for _, r in g[g["tipo"] == "anual"].iterrows():
            qt, nota = acoes(r["dt_fim"])
            if qt is None:
                continue
            linhas.append(_linha(meta, "anual", r["dt_fim"], r, r, qt[0], qt[1], px, raiz, [nota]))

        # --- trimestral: ponto no tempo
        tri = g[g["tipo"] == "trimestral"].set_index("dt_fim")
        ltm = g[g["tipo"] == "ltm"].set_index("dt_fim")
        anu = g[g["tipo"] == "anual"].set_index("dt_fim")
        docs = sorted(set(tri.index) | set(anu.index))
        disp = [(divulgado(dfim), dfim) for dfim in docs]          # ((data_divulgação, nota), dt_fim)
        for d in datas_tri:
            vis = [x for x in disp if x[0][0] <= d]                    # documentos já divulgados em d
            if not vis:
                continue
            (_, nota_disp), dfim = max(vis, key=lambda x: x[1])      # o mais recente entre eles
            fund = tri.loc[dfim] if dfim in tri.index else anu.loc[dfim]
            if isinstance(fund, pd.DataFrame):
                fund = fund.iloc[-1]
            if dfim in ltm.index:
                flu = ltm.loc[dfim]
                flu = flu.iloc[-1] if isinstance(flu, pd.DataFrame) else flu
                nota_flu = ""
            elif dfim in anu.index:
                flu = anu.loc[dfim]
                flu = flu.iloc[-1] if isinstance(flu, pd.DataFrame) else flu
                nota_flu = ""
            else:
                flu, nota_flu = {}, "sem LTM para o último documento divulgado"
            qt, nota_qt = acoes(dfim)
            if qt is None:
                continue
            row = _linha(meta, "trimestral", d, fund, flu, qt[0], qt[1], px, raiz, [nota_disp, nota_flu, nota_qt])
            row["periodo_base"] = fund.get("periodo")
            linhas.append(row)
    if fin is not None and len(fin):
        linhas += _financeiras(fin, raiz_de, compos, px)
    out = pd.DataFrame(linhas)
    if len(out):
        out = out.sort_values(["cnpj", "tipo", "data"])
    return out
