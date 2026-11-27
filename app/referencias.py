"""
Séries de referência para comparar o desempenho das ações (aba Empresa, gráfico de cotação).

Sem Streamlit: usado pelo coletor (mercado.py grava saidas/mercado/indices.parquet) e, como plano B, pelo app.

Brasil
- Ibovespa: fechamento diário (Yahoo Finance, ^BVSP).
- SMLL (Small Cap): fechamento diário das estatísticas históricas da B3 (sistemaswebb3-listados, IndexStatistics).
- CDI: taxa DI diária (% a.d.) do Banco Central (SGS 12), acumulada em número-índice (cálculo próprio).
- IMA-B: número-índice ANBIMA. Até mai/2023 pela série 12466 do SGS (BCB), que replica o índice da ANBIMA e deixou de
  ser atualizada; depois, pelo quadro-resumo diário do site da ANBIMA (ima-sh-down.asp). Plano B, se a ANBIMA não
  responder: variação do ETF IMAB11 (Yahoo Finance), que replica o IMA-B com taxa de administração (aproximação).

EUA (Yahoo Finance): S&P 500 (^GSPC, preço), S&P 500 retorno total (^SP500TR), Nasdaq Composite (^IXIC),
Russell 2000 (^RUT, preço).

Formato comum: DataFrame (data, ticker, indice, fech, fonte).
"""
from __future__ import annotations

import base64
import json
from concurrent.futures import ThreadPoolExecutor
from datetime import date

import numpy as np
import pandas as pd
import requests

INICIO = "2015-01-01"
UA = {"User-Agent": "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 (KHTML, like Gecko) "
                    "Chrome/126.0 Safari/537.36"}
SGS = "https://api.bcb.gov.br/dados/serie/bcdata.sgs.{cod}/dados"
B3_ESTAT = "https://sistemaswebb3-listados.b3.com.br/indexStatisticsProxy/IndexCall/GetPortfolioDay/{arg}"
ANBIMA_IMA = "https://www.anbima.com.br/informacoes/ima/ima-sh-down.asp"
SGS_CDI, SGS_IMAB = 12, 12466
COLS = ["data", "ticker", "indice", "fech", "fonte"]

# nome exibido -> código gravado na coluna ticker
BR = {"Ibovespa": "^BVSP", "SMLL (Small Cap)": "SMLL", "CDI": "CDI", "IMA-B": "IMA-B"}
EUA = {"S&P 500": "^GSPC", "S&P 500 (retorno total)": "^SP500TR", "Nasdaq Composite": "^IXIC",
       "Russell 2000": "^RUT"}
# o que cada índice mede: entra na legenda do app
NOTAS = {
    "Ibovespa": "retorno total (B3 reinveste os proventos no índice)",
    "SMLL (Small Cap)": "retorno total (B3)",
    "CDI": "taxa DI acumulada (BCB/SGS 12)",
    "IMA-B": "número-índice ANBIMA (NTN-B a mercado)",
    "S&P 500": "só preço (sem dividendos)",
    "S&P 500 (retorno total)": "com dividendos reinvestidos",
    "Nasdaq Composite": "só preço (sem dividendos)",
    "Russell 2000": "só preço (sem dividendos)",
}


def _num_br(t) -> float:
    """'2.156,55' -> 2156.55 (ponto = milhar, vírgula = decimal)."""
    t = str(t or "").strip().replace(".", "").replace(",", ".")
    try:
        return float(t)
    except ValueError:
        return np.nan


def _df(s: pd.Series, ticker: str, indice: str, fonte: str) -> pd.DataFrame:
    s = s.dropna()
    s = s[s > 0]
    return pd.DataFrame({"data": pd.to_datetime(s.index).normalize(), "ticker": ticker, "indice": indice,
                         "fech": s.to_numpy(float), "fonte": fonte})


# ------------------------------------------------------------------ fontes
def sgs(cod: int, ini=INICIO, fim=None) -> pd.Series:
    """Série diária do SGS/BCB (janelas de 5 anos: o SGS limita consultas de séries diárias a 10 anos)."""
    ini, fim = pd.Timestamp(ini), pd.Timestamp(fim or date.today())
    partes = []
    a = ini
    while a <= fim:
        b = min(a + pd.DateOffset(years=5) - pd.Timedelta(days=1), fim)
        r = requests.get(SGS.format(cod=cod), params={"formato": "json", "dataInicial": a.strftime("%d/%m/%Y"),
                                                       "dataFinal": b.strftime("%d/%m/%Y")}, timeout=60, headers=UA)
        if r.status_code == 200 and r.text.strip().startswith("["):
            partes += r.json()
        a = b + pd.Timedelta(days=1)
    if not partes:
        return pd.Series(dtype=float)
    d = pd.DataFrame(partes)
    s = pd.Series(pd.to_numeric(d["valor"], errors="coerce").to_numpy(),
                  index=pd.to_datetime(d["data"], format="%d/%m/%Y"))
    return s[~s.index.duplicated(keep="last")].sort_index().dropna()


def cdi(ini=INICIO) -> pd.DataFrame:
    """Número-índice do CDI (base 1 no primeiro dia). A taxa do dia d rende de d para d+1, então o índice no
    fechamento de d acumula as taxas dos dias anteriores."""
    t = sgs(SGS_CDI, ini)
    if t.empty:
        return pd.DataFrame(columns=COLS)
    idx = (1 + t / 100).cumprod().shift(1).fillna(1.0)
    return _df(idx, "CDI", "CDI", "BCB/SGS 12 (taxa DI), acumulado: cálculo próprio")


def b3_indice(indice: str, anos) -> pd.Series:
    """Fechamentos diários de um índice da B3 (estatísticas históricas: uma consulta por ano)."""
    pts = {}
    for ano in anos:
        payload = {"language": "pt-br", "index": indice, "year": str(ano)}
        arg = base64.b64encode(json.dumps(payload, separators=(",", ":")).encode()).decode()
        r = requests.get(B3_ESTAT.format(arg=arg), timeout=60, headers={**UA, "Accept": "application/json"})
        r.raise_for_status()
        d = r.json()
        d = json.loads(d) if isinstance(d, str) else d
        for it in d.get("results") or []:
            dia = int(it.get("day") or 0)
            for m in range(1, 13):
                v = _num_br(it.get(f"rateValue{m}"))
                if dia and np.isfinite(v) and v > 0:
                    try:
                        pts[pd.Timestamp(int(ano), m, dia)] = v
                    except ValueError:
                        pass
    return pd.Series(pts, dtype=float).sort_index()


def smll(anos) -> pd.DataFrame:
    return _df(b3_indice("SMLL", anos), "SMLL", "SMLL (Small Cap)", "B3 (estatísticas históricas do índice)")


def yahoo(ticker: str, ini=INICIO) -> pd.Series:
    import yfinance as yf
    h = yf.Ticker(ticker).history(start=str(pd.Timestamp(ini).date()), auto_adjust=False)
    if h is None or h.empty:
        return pd.Series(dtype=float)
    return pd.Series(h["Close"].to_numpy(float), index=pd.to_datetime(h.index).tz_localize(None).normalize())


def _ima_b_anbima_dia(d: pd.Timestamp) -> float:
    """Número-índice do IMA-B no quadro-resumo da ANBIMA de uma data (NaN em feriado ou sem resposta)."""
    try:
        r = requests.get(ANBIMA_IMA, timeout=30, headers=UA, params={
            "Tipo": "", "DataRef": "", "Pai": "ima", "escolha": "2", "Idioma": "PT", "saida": "csv",
            "Dt_Ref_Ver": d.strftime("%Y%m%d"), "Dt_Ref": d.strftime("%d/%m/%Y")})
        if r.status_code != 200:
            return np.nan
        for linha in r.content.decode("latin-1", errors="ignore").splitlines():
            p = linha.split(";")
            if len(p) > 2 and p[0].strip() == "IMA-B":
                return _num_br(p[2])
    except requests.RequestException:
        pass
    return np.nan


def ima_b_anbima(ini, fim=None, paralelo: int = 6) -> pd.Series:
    dias = pd.bdate_range(pd.Timestamp(ini), pd.Timestamp(fim or date.today()))
    with ThreadPoolExecutor(paralelo) as ex:
        vals = list(ex.map(_ima_b_anbima_dia, dias))
    return pd.Series(vals, index=dias, dtype=float).dropna()


def serie_confiavel(s: pd.Series, max_parada_du: int = 15, max_salto: float = 0.08) -> bool:
    """Rejeita séries com cotação parada por muitos dias úteis seguidos ou saltos diários grandes (sinal de dado
    ruim, como já visto no IMAB11 do Yahoo). Para um índice de renda fixa, ambos são impossíveis."""
    s = s.dropna().sort_index()
    if len(s) < 5:
        return False
    du = pd.bdate_range(s.index.min(), s.index.max())
    if len(du) > 1.5 * len(s):                        # faltam dias demais
        return False
    r = s.pct_change().dropna()
    if (r.abs() > max_salto).any():
        return False
    parado = (r == 0).astype(int)
    return int(parado.groupby((parado == 0).cumsum()).sum().max() or 0) <= max_parada_du


def _emendar(base: pd.Series, ext: pd.Series) -> pd.Series:
    """Continua 'base' com as variações de 'ext' depois da última data de 'base' (escala pela data comum mais
    recente; sem data comum, junta os níveis como estão)."""
    if base.empty:
        return ext
    ext = ext[ext.index >= base.index.max() - pd.Timedelta(days=10)]
    comum = base.index.intersection(ext.index)
    if len(comum):
        d0 = comum.max()
        ext = ext * base[d0] / ext[d0]
    return pd.concat([base, ext[ext.index > base.index.max()]]).sort_index()


def ima_b(ant: pd.Series | None = None) -> pd.DataFrame:
    """IMA-B: série já gravada (ant) ou SGS 12466 desde 2015, completada pela ANBIMA; sem ANBIMA, pelo IMAB11."""
    base = ant.dropna().sort_index() if ant is not None and len(ant) else sgs(SGS_IMAB, INICIO)
    fonte = "ANBIMA (até mai/2023 via BCB/SGS 12466; depois, site da ANBIMA)"
    if base.empty:
        return pd.DataFrame(columns=COLS)
    novo = ima_b_anbima(base.index.max() - pd.Timedelta(days=7))
    if len(novo):
        s = _emendar(base, novo)
    else:
        etf = yahoo("IMAB11.SA", base.index.max() - pd.Timedelta(days=30))
        if len(etf) and serie_confiavel(etf):
            s = _emendar(base, etf)
            fonte += "; trecho final pela variação do ETF IMAB11 (ANBIMA indisponível)"
        else:
            s = base
            fonte += "; sem atualização recente (ANBIMA e IMAB11 indisponíveis)"
    return _df(s, "IMA-B", "IMA-B", fonte)


def ima_b_rapido() -> pd.DataFrame:
    """Plano B do app (sem esperar centenas de consultas à ANBIMA): SGS 12466 + variação do ETF IMAB11."""
    base = sgs(SGS_IMAB, INICIO)
    if base.empty:
        return pd.DataFrame(columns=COLS)
    etf = yahoo("IMAB11.SA", base.index.max() - pd.Timedelta(days=30))
    if len(etf) and serie_confiavel(etf):
        return _df(_emendar(base, etf), "IMA-B", "IMA-B",
                   "ANBIMA via BCB/SGS 12466 até mai/2023; depois, variação do ETF IMAB11 (aproximação)")
    return _df(base, "IMA-B", "IMA-B", "ANBIMA via BCB/SGS 12466, só até mai/2023 (série completa: próxima "
                                       "execução do coletor mercado.py)")


def eua(nome: str) -> pd.DataFrame:
    tk = EUA[nome]
    return _df(yahoo(tk), tk, nome, f"Yahoo Finance ({tk})")
