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


def ler_cotacoes(anos: list[int], raizes: set[str]) -> tuple[pd.DataFrame, dict]:
    """Fechamento diário (mercado à vista) dos papéis ON/PN. Retorna (cotações, ticker -> raiz atual).
    Papéis que mudaram de código (ex.: CCRO3 -> MOTV3) são ligados à empresa pelo ISIN, que não muda."""
    tk, dt, pr, vol, isin = [], [], [], [], []
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
                        if len(cod) != 5 or cod[4:5] not in SUFIXOS:
                            continue
                        fat = int(ln[210:217] or b"1") or 1
                        tk.append(cod.decode())
                        dt.append(ln[2:10].decode())
                        pr.append(int(ln[108:121]) / 100 / fat)
                        vol.append(int(ln[170:188]) / 100)
                        isin.append(ln[230:242].decode().strip())
        print(f"  COTAHIST {ano}: {len(tk)} registros acumulados")
    df = pd.DataFrame({"ticker": tk, "data": pd.to_datetime(dt, format="%Y%m%d"), "preco": pr, "volume": vol,
                       "isin": isin})
    if df.empty:
        return df, {}
    # papel atual de cada empresa (raiz = 4 letras do código de emissor da B3) -> ISINs -> todos os códigos
    # que já usaram esses ISINs
    ult_ano = df["data"].dt.year.max()
    recente = df[df["data"].dt.year >= ult_ano - 1]
    isins_da_raiz = recente[recente["ticker"].str[:4].isin(raizes)].groupby(recente["ticker"].str[:4])["isin"].unique()
    cod_por_isin = df.groupby("isin")["ticker"].unique()
    mapa = {}
    for raiz, isins in isins_da_raiz.items():
        for i in isins:
            if i and i in cod_por_isin.index:
                for c in cod_por_isin[i]:
                    mapa.setdefault(c, raiz)
    for c in df["ticker"].unique():           # papéis da raiz que não aparecem no período recente
        if c[:4] in raizes:
            mapa.setdefault(c, c[:4])
    df = df[df["ticker"].isin(mapa)]
    antigos = {c: r for c, r in mapa.items() if c[:4] != r}
    if antigos:
        print(f"  códigos antigos ligados pelo ISIN: {len(antigos)} (ex.: {list(antigos.items())[:5]})")
    return df.sort_values(["ticker", "data"]).reset_index(drop=True), mapa


class Precos:
    def __init__(self, cot: pd.DataFrame, mapa: dict | None = None):
        cot = cot.sort_values(["ticker", "data"])
        self.serie = {t: (list(g["data"]), list(g["preco"])) for t, g in cot.groupby("ticker")}
        self.vol_acum = {t: [0.0] + list(np.cumsum(g["volume"].to_numpy())) for t, g in cot.groupby("ticker")}
        liq = cot.groupby("ticker")["volume"].sum()
        self.por_raiz = {}
        for t in self.serie:
            self.por_raiz.setdefault((mapa or {}).get(t, t[:4]), []).append(t)
        for r in self.por_raiz:
            self.por_raiz[r].sort(key=lambda t: -liq.get(t, 0))

    def classe(self, raiz: str, cls: str) -> list[str]:
        return [t for t in self.por_raiz.get(raiz, []) if SUFIXOS[t[4:5].encode()] == cls]

    def volume_janela(self, tickers: list[str], d: pd.Timestamp, dias: int = 90) -> float:
        """Volume financeiro negociado nos `dias` anteriores a d (inclusive), somando os papéis informados."""
        ini = d - pd.Timedelta(days=dias)
        tot = 0.0
        for t in tickers:
            datas = self.serie[t][0]
            i, j = bisect_right(datas, d), bisect_right(datas, ini)
            tot += self.vol_acum[t][i] - self.vol_acum[t][j]
        return tot

    def ultimo(self, tickers: list[str], d: pd.Timestamp):
        """Último preço até a data d entre os papéis da classe: o de negócio mais recente
        (em empate, o mais líquido). Assim o código novo de um papel renomeado prevalece sobre o antigo."""
        melhor = None
        for t in tickers:
            datas, precos = self.serie[t]
            i = bisect_right(datas, d)
            if i and (melhor is None or datas[i - 1] > melhor[1]):
                melhor = (t, datas[i - 1], precos[i - 1])
        return melhor


def _valor_mercado(px: Precos, raiz: str, on: float, pn: float, d: pd.Timestamp):
    notas = []
    p_on = px.ultimo(px.classe(raiz, "ON"), d) if on > 0 else None
    p_pn = px.ultimo(px.classe(raiz, "PN"), d) if pn > 0 else None
    if on > 0 and pn > 0 and p_pn is not None:
        v_on = px.volume_janela(px.classe(raiz, "ON"), d)
        v_pn = px.volume_janela(px.classe(raiz, "PN"), d)
        if v_pn > v_on:
            p_on = p_pn
            notas.append(f"ON precificada pela PN ({p_pn[0]}), mais líquida nos 90 dias anteriores")
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


def _escala_acoes(vm, pl, receita=None) -> int:
    """Algumas companhias informam a quantidade de ações em milhares na composição do capital (ex.: Vale,
    Ambev, Itaú). Detecta pelo absurdo econômico: valor de mercado < 2% do PL (ou < R$ 20 mi sem PL positivo)
    que vira um P/VPA plausível quando multiplicado por 1.000."""
    if pd.isna(vm) or vm <= 0:
        return 1
    if pd.notna(pl) and pl > 0:
        return 1000 if vm / pl < 0.02 and 0.05 <= vm * 1000 / pl <= 200 else 1
    return 1000 if vm < 20 else 1


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


DIAG_ACOES: list[dict] = []   # diagnóstico da série de ações (gravado pelo universo)


def montar_serie_acoes(cands, compos: dict, fre: dict) -> list[dict]:
    """Todas as quantidades de ações da empresa (composição do capital do ITR/DFP e, antes dela, o FRE)."""
    tl = []
    for (c, k), (on, pn) in compos.items():
        if c in cands:
            tl.append({"data": pd.Timestamp(k), "on": on, "pn": pn, "fonte": f"ITR/DFP {k}"})
    ini = min((e["data"] for e in tl), default=pd.Timestamp.max)
    for (c, ano), (on, pn) in (fre or {}).items():
        d = pd.Timestamp(f"{ano}-06-30")       # FRE retrata o capital por volta do meio do ano
        if c in cands and d < ini:
            tl.append({"data": d, "on": on, "pn": pn, "fonte": f"FRE {ano}"})
    tl = sorted({e["data"]: e for e in tl}.values(), key=lambda e: e["data"])
    return [e for e in tl if (e["on"] or 0) + (e["pn"] or 0) > 0]


def normalizar_serie_acoes(tl: list[dict], px, raiz: str, pl_em) -> list[dict]:
    """Corrige a unidade da quantidade de ações (algumas companhias informam em milhares, e às vezes alternam
    entre documentos). Regras, nesta ordem:
    1) em cada documento, escolhe a escala (1 ou 1.000) que dá P/VPA plausível (0,05x a 100x) — ou, sem PL
       positivo, valor de mercado entre R$ 20 mi e R$ 3 tri — quando só uma das duas é plausível;
    2) documentos indefinidos herdam a escala que deixa a quantidade mais próxima do vizinho já definido;
    3) de trás para frente, a partir do documento mais recente: quantidade que difere dos documentos seguintes
       por ~10x, 100x ou 1.000x (para cima ou para baixo) sem variação inversa do preço é reescalada; com o preço
       acompanhando (desdobramento/grupamento), fica como está."""
    if not tl:
        return tl
    todos = px.por_raiz.get(raiz, [])
    for e in tl:
        p = px.ultimo(todos, e["data"])
        e["preco"] = p[2] if p else np.nan
        tot, pl = e["on"] + e["pn"], pl_em(e["data"])
        def plaus(sc):
            if not tot or pd.isna(e["preco"]):
                return None
            vm = tot * sc * e["preco"] / 1e6
            if pd.notna(pl) and pl > 0:
                return 0.05 <= vm / pl <= 100
            return 20 <= vm <= 3e6
        a1, a1000 = plaus(1), plaus(1000)
        e["escala"] = 1 if (a1 and not a1000) else 1000 if (a1000 and not a1) else None
        e["nota"] = ""
    definidos = [i for i, e in enumerate(tl) if e["escala"]]
    for i, e in enumerate(tl):
        if e["escala"] is None:
            if not definidos:
                e["escala"] = 1
                continue
            j = min(definidos, key=lambda k: (abs(k - i), k > i))
            ref = (tl[j]["on"] + tl[j]["pn"]) * tl[j]["escala"]
            tot = e["on"] + e["pn"]
            e["escala"] = 1000 if abs(np.log10(tot * 1000 / ref)) < abs(np.log10(tot / ref)) else 1
    # 3) continuidade da série (de trás para frente, ancorada no documento mais recente): quantidade que difere
    #    da do documento seguinte (já tratado) por um fator próximo de uma potência
    #    de 10 (10x, 100x, 1.000x, para cima ou para baixo) sem variação inversa do preço é erro de unidade e é
    #    reescalada; se o preço se moveu na proporção inversa, é desdobramento/grupamento e fica como está.
    tot_aj = lambda e: (e["on"] + e["pn"]) * e["escala"]
    for i in range(len(tl) - 2, -1, -1):
        e = tl[i]
        ref = tot_aj(tl[i + 1])                        # documento seguinte, já tratado (a série é corrigida de trás
        preco_ref = tl[i + 1]["preco"]                 # para frente, então a referência não atravessa eventos)
        t = tot_aj(e)
        if t <= 0 or ref <= 0:
            continue
        lq = np.log10(t / ref)
        k = int(round(lq))
        if k == 0 or abs(lq - k) > 0.15:
            continue                                   # diferença que não é potência de 10: emissão/recompra real
        pr = e["preco"] / preco_ref if pd.notna(e["preco"]) and pd.notna(preco_ref) and preco_ref else np.nan
        if pd.notna(pr) and abs(np.log10((t / ref) * pr)) < 0.3:
            e["nota"] = "variação de 10x ou mais na quantidade acompanhada pelo preço (desdobramento/grupamento)"
            continue
        e["escala"] = e["escala"] / (10 ** k)
        e["nota"] = (f"quantidade {10 ** abs(k):,.0f}x {'maior' if k > 0 else 'menor'} que a dos documentos "
                     f"seguintes sem mudança de preço: ajustada (÷{10 ** k:g})").replace(",", ".")
    for e in tl:
        if e["escala"] == 1000 and not e["nota"]:
            e["nota"] = "quantidade informada em milhares na CVM: multiplicada por 1.000"
    return tl


def acoes_da_serie(tl: list[dict], dt_fim) -> tuple:
    """Ações vigentes na data: documento da própria data; senão o último anterior; senão o primeiro até 1 ano depois."""
    d = pd.Timestamp(dt_fim)
    if not tl:
        return None, "sem composição do capital"
    escolhido, nota = None, ""
    exato = [e for e in tl if e["data"] == d]
    if exato:
        escolhido = exato[0]
    else:
        ant = [e for e in tl if e["data"] <= d]
        if ant:
            escolhido = ant[-1]
            nota = f"ações de {escolhido['fonte']}"
        else:
            depois = [e for e in tl if d < e["data"] <= d + pd.Timedelta(days=366)]
            if depois:
                escolhido = depois[0]
                nota = f"ações de {escolhido['fonte']} (documento posterior)"
    if escolhido is None:
        return None, "sem composição do capital"
    if escolhido["fonte"].startswith("FRE"):
        nota = f"ações do Formulário de Referência ({escolhido['fonte']}; sem desconto de tesouraria)"
    f = escolhido["escala"]
    notas = "; ".join(x for x in (nota, escolhido.get("nota", "")) if x)
    return (escolhido["on"] * f, escolhido["pn"] * f), notas


def _acoes_fre(cnpjs, dt_fim, fre):
    """Fallback: Formulário de Referência do mesmo ano (ou o mais próximo, até 1 ano de distância)."""
    ano = pd.Timestamp(dt_fim).year
    for a in (ano, ano + 1, ano - 1):
        for c in cnpjs:
            if (c, a) in fre:
                return fre[(c, a)], (f"ações do Formulário de Referência {a} (capital social; sem desconto de "
                                     "tesouraria; aproximação para anos sem composição do capital no ITR/DFP)")
    return None, "sem composição do capital"


def _acoes_em(cnpj, dt_fim, compos, fre=None):
    k = str(pd.Timestamp(dt_fim).date())
    if (cnpj, k) in compos:
        return compos[(cnpj, k)], ""
    ant = sorted((kk[1], v) for kk, v in compos.items() if kk[0] == cnpj and kk[1] <= k)
    if ant:
        return ant[-1][1], f"ações do documento de {ant[-1][0]} (sem composição em {k})"
    return _acoes_fre([cnpj], dt_fim, fre or {})


def _financeiras(fin: pd.DataFrame, raiz_de: dict, acoes_fn, px) -> list[dict]:
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
            qt, nota = acoes_fn(cnpj, dfim)
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
             n_anos: int = 10, fin: pd.DataFrame | None = None, fre: dict | None = None) -> pd.DataFrame:
    if base.empty:
        return pd.DataFrame()
    b = base.copy()
    b["dt_fim"] = pd.to_datetime(b["dt_fim"])
    hoje = pd.Timestamp.today().normalize()
    anos = list(range(int(b["exercicio"].min()) - 1, hoje.year + 1))
    anos = [a for a in anos if a >= hoje.year - n_anos - 1]
    raiz_de = {r["cnpj_cvm"]: str(r.get("issuingCompany") or "")[:4] for _, r in univ.iterrows()}
    cot, mapa = ler_cotacoes(anos, set(r for r in raiz_de.values() if r))
    if cot.empty:
        print("  ! nenhuma cotação lida: múltiplos não calculados")
        return pd.DataFrame()
    px = Precos(cot, mapa)
    ultimo_pregao = cot["data"].max()
    datas_tri = [d for d in pd.date_range(f"{hoje.year - n_anos}-03-31", ultimo_pregao, freq="QE")]
    if not datas_tri or datas_tri[-1] < ultimo_pregao:
        datas_tri.append(ultimo_pregao)  # ponto mais recente ("hoje")

    # PL dos controladores por data (para checar a unidade das ações) e série de ações por empresa
    pl_series = {}
    for cnpj_, g_ in b[b["tipo"].isin(["trimestral", "anual"])].groupby("cnpj"):
        pl_series[cnpj_] = g_[["dt_fim", "pl_controladores"]].dropna().sort_values("dt_fim")
    if fin is not None and len(fin):
        for cnpj_, g_ in fin.groupby("cnpj"):
            pl_series[cnpj_] = g_.assign(dt_fim=pd.to_datetime(g_["dt_fim"]))[["dt_fim", "pl_controladores"]] \
                .dropna().sort_values("dt_fim")

    def pl_em(cnpj_):
        s_ = pl_series.get(cnpj_)
        def f(d):
            if s_ is None or s_.empty:
                return np.nan
            k = (s_["dt_fim"] - d).abs().idxmin()
            return s_.loc[k, "pl_controladores"] if abs((s_.loc[k, "dt_fim"] - d).days) <= 200 else np.nan
        return f

    series_acoes = {}
    DIAG_ACOES.clear()

    def serie_de(cnpj_, raiz_):
        if cnpj_ not in series_acoes:
            cands_ = [cnpj_] + list(suc.get(cnpj_, []))
            tl = normalizar_serie_acoes(montar_serie_acoes(cands_, compos, fre or {}), px, raiz_, pl_em(cnpj_))
            series_acoes[cnpj_] = tl
            DIAG_ACOES.extend({"cnpj": cnpj_, "raiz": raiz_, "data": e["data"].date(), "fonte": e["fonte"],
                               "acoes_on_informadas": e["on"], "acoes_pn_informadas": e["pn"], "escala": e["escala"],
                               "preco_ref": e["preco"], "nota": e["nota"]} for e in tl)
        return series_acoes[cnpj_]

    linhas = []
    for cnpj, g in b.groupby("cnpj"):
        raiz = raiz_de.get(cnpj, "")
        if not raiz or raiz not in px.por_raiz:
            continue
        cands = [cnpj] + list(suc.get(cnpj, []))
        tl_emp = serie_de(cnpj, raiz)
        meta = {k: g[k].iloc[0] for k in ("cnpj", "empresa", "ticker", "setor", "subsetor", "segmento")}

        def acoes(dt_fim, tl_emp=tl_emp):
            return acoes_da_serie(tl_emp, dt_fim)

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
            # ações vigentes na data da cotação (a quantidade é um fato da data, divulgado por fato relevante),
            # e não as do documento usado para os fundamentos: evita misturar preço pós-desdobramento com ações antigas
            qt, nota_qt = acoes(d)
            if qt is None:
                continue
            row = _linha(meta, "trimestral", d, fund, flu, qt[0], qt[1], px, raiz, [nota_disp, nota_flu, nota_qt])
            row["periodo_base"] = fund.get("periodo")
            linhas.append(row)
    if fin is not None and len(fin):
        linhas += _financeiras(fin, raiz_de, lambda c, d: acoes_da_serie(serie_de(c, raiz_de.get(c, "")), d), px)
    out = pd.DataFrame(linhas)
    if len(out):
        out = out.sort_values(["cnpj", "tipo", "data"])
    return out
