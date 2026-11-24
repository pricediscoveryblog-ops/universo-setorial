"""
Price Discovery Scanning — seções de mercado do app (módulo carregado por app.py):

- cotação diária com ajuste opcional por proventos (dividendos, JCP, rendimentos) e por eventos (desdobramentos,
  grupamentos, bonificações), e opção de valores reais (IPCA no Brasil, CPI nos EUA), mais valor de mercado e EV diários;
- mapa de calor das altas e baixas por setor (área = valor de mercado, cor = variação);
- estrutura societária (Formulário de Referência da CVM) com esquema de controle e gráfico de participações;
- modelo de negócio e breve histórico (curadoria em app/curadoria/<RAIZ>.yaml).

Fontes: cotações brutas da série histórica da B3 (COTAHIST), proventos da página de empresas listadas da B3, FRE da
CVM (Dados Abertos); EUA: Yahoo Finance (yfinance). Ajustes, índices reais e agregados são cálculo próprio.
"""
from __future__ import annotations

import os
import textwrap
import unicodedata
from pathlib import Path

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st

RAIZ_APP = Path(__file__).resolve().parent
MERC = Path(os.environ.get("MERCADO_DIR", RAIZ_APP.parent / "saidas" / "mercado"))
MACRO = RAIZ_APP.parent / "saidas"
CURADORIA = RAIZ_APP / "curadoria"

EV_ACOES_APLICADOS = ("DESDOBRAMENTO", "GRUPAMENTO", "BONIFICACAO")



def _n(s) -> str:
    return unicodedata.normalize("NFKD", str(s)).encode("ascii", "ignore").decode().upper().strip()


def _versao(*padroes) -> tuple:
    return tuple(sorted((p.name, p.stat().st_mtime) for pad in padroes for p in MERC.glob(pad)))


# ================================================================== leitura (Brasil)
@st.cache_data(max_entries=40, show_spinner=False)
def precos_br(raiz: str, versao: tuple = ()) -> pd.DataFrame:
    arqs = sorted(MERC.glob("precos_*.parquet"))
    if not arqs:
        return pd.DataFrame()
    partes = [pd.read_parquet(a, filters=[("raiz", "==", raiz)]) for a in arqs]
    p = pd.concat([x for x in partes if len(x)], ignore_index=True) if any(len(x) for x in partes) else pd.DataFrame()
    if len(p):
        p["data"] = pd.to_datetime(p["data"])
        p = p.sort_values(["ticker", "data"]).drop_duplicates(["ticker", "data"], keep="last")
    return p


@st.cache_data(max_entries=40, show_spinner=False)
def vm_ev_br(cnpj: str, versao: tuple = ()) -> pd.DataFrame:
    arqs = sorted(MERC.glob("vm_ev_*.parquet"))
    if not arqs:
        return pd.DataFrame()
    partes = [pd.read_parquet(a, filters=[("cnpj", "==", cnpj)]) for a in arqs]
    v = pd.concat(partes, ignore_index=True)
    v["data"] = pd.to_datetime(v["data"])
    return v.sort_values("data")


@st.cache_data(max_entries=1, show_spinner=False)
def vm_ev_todos(versao: tuple = ()) -> pd.DataFrame:
    """Série diária de valor de mercado e EV de todas as empresas (base das altas e baixas)."""
    arqs = sorted(MERC.glob("vm_ev_*.parquet"))
    if not arqs:
        return pd.DataFrame()
    v = pd.concat([pd.read_parquet(a, columns=["data", "cnpj", "valor_mercado"]) for a in arqs], ignore_index=True)
    v["data"] = pd.to_datetime(v["data"])
    v["cnpj"] = v["cnpj"].astype(str).str.zfill(14)
    return v


@st.cache_data(max_entries=8, show_spinner=False)
def liquidez_br(d_fim, n_pregoes: int = 21, versao: tuple = ()) -> pd.DataFrame:
    """cnpj, liq: volume financeiro médio diário (R$ milhões) nos n pregões até d_fim, somando todas as classes de
    ações da empresa (COTAHIST da B3; cálculo próprio)."""
    d_fim = pd.Timestamp(d_fim)
    arqs = [MERC / f"precos_{a}.parquet" for a in (d_fim.year - 1, d_fim.year)]
    filtro = [("data", ">=", d_fim - pd.Timedelta(days=60)), ("data", "<=", d_fim)]
    partes = []
    for a in arqs:
        if a.exists():
            try:
                partes.append(pd.read_parquet(a, columns=["data", "raiz", "volume"], filters=filtro))
            except Exception:  # noqa: BLE001 — coluna data gravada como texto em arquivos antigos
                x = pd.read_parquet(a, columns=["data", "raiz", "volume"])
                x["data"] = pd.to_datetime(x["data"])
                partes.append(x[(x["data"] >= filtro[0][2]) & (x["data"] <= d_fim)])
    if not partes:
        return pd.DataFrame(columns=["cnpj", "liq"])
    p = pd.concat(partes, ignore_index=True)
    p["data"] = pd.to_datetime(p["data"])
    dias = sorted(p["data"].unique())[-n_pregoes:]
    p = p[p["data"].isin(dias)]
    por_dia = p.groupby(["raiz", "data"])["volume"].sum().reset_index()      # todas as classes no mesmo dia
    liq = (por_dia.groupby("raiz")["volume"].sum() / len(dias) / 1e6).rename("liq").reset_index()
    a_vm = MERC / f"vm_ev_{d_fim.year}.parquet"
    if not a_vm.exists():
        return pd.DataFrame(columns=["cnpj", "liq"])
    mapa = pd.read_parquet(a_vm, columns=["cnpj", "raiz"]).drop_duplicates()
    mapa["cnpj"] = mapa["cnpj"].astype(str).str.zfill(14)
    return mapa.merge(liq, on="raiz").groupby("cnpj", as_index=False)["liq"].sum()


@st.cache_data(ttl=6 * 3600, show_spinner=False)
def ibovespa(versao: tuple = ()) -> pd.DataFrame:
    """Ibovespa diário (data, indice, fech): arquivo do coletor (mercado.py); sem ele, direto do Yahoo Finance."""
    a = MERC / "indices.parquet"
    i = pd.read_parquet(a) if a.exists() else pd.DataFrame()
    if i.empty:
        try:
            import yfinance as yf
            h = yf.Ticker("^BVSP").history(period="2y", auto_adjust=False)
            i = pd.DataFrame({"data": pd.to_datetime(h.index).tz_localize(None).normalize(), "indice": "Ibovespa",
                              "fech": h["Close"].to_numpy(float)})
        except Exception:  # noqa: BLE001
            return pd.DataFrame(columns=["data", "indice", "fech"])
    i["data"] = pd.to_datetime(i["data"])
    return i[["data", "indice", "fech"]]


def retorno_indices(ind: pd.DataFrame, d_ini, d_fim, tol_ini: int = 3) -> dict:
    """{índice: (variação, data inicial, data final)} entre o último fechamento até d_ini (com até tol_ini dias de
    atraso) e o último até d_fim (até 3 dias). Mesma regra de datas das empresas nas altas e baixas."""
    out = {}
    if ind is None or ind.empty:
        return out
    d_ini, d_fim = pd.Timestamp(d_ini), pd.Timestamp(d_fim)
    for nome, g in ind.dropna(subset=["fech"]).sort_values("data").groupby("indice", sort=False):
        f = g[g["data"] <= d_fim].tail(1)
        i = g[g["data"] <= d_ini].tail(1)
        if f.empty or i.empty or (d_fim - f["data"].iloc[0]).days > 3 or (d_ini - i["data"].iloc[0]).days > tol_ini:
            continue
        out[nome] = (float(f["fech"].iloc[0] / i["fech"].iloc[0] - 1), i["data"].iloc[0], f["data"].iloc[0])
    return out


@st.cache_data(max_entries=1, show_spinner=False)
def proventos_br(versao: tuple = ()) -> pd.DataFrame:
    a = MERC / "proventos.parquet"
    if not a.exists():
        return pd.DataFrame()
    p = pd.read_parquet(a)
    for c in ("data_com", "data_aprovacao", "data_pagamento"):
        p[c] = pd.to_datetime(p[c], errors="coerce")
    return p


@st.cache_data(max_entries=1, show_spinner=False)
def acionistas_br(versao: tuple = ()) -> tuple[pd.DataFrame, pd.DataFrame]:
    a = pd.read_parquet(MERC / "acionistas.parquet") if (MERC / "acionistas.parquet").exists() else pd.DataFrame()
    d = pd.read_parquet(MERC / "distribuicao_capital.parquet") if (MERC / "distribuicao_capital.parquet").exists() \
        else pd.DataFrame()
    return a, d


@st.cache_data(max_entries=2, show_spinner=False)
def indice_precos(eua: bool) -> pd.Series:
    """Índice de preços mensal (IPCA no Brasil, CPI-U nos EUA), indexado pelo mês (Period M)."""
    try:
        if eua:
            c = pd.read_csv(MACRO / "eua_macro" / "cpi_mensal.csv")
            s = pd.Series(c["cpi_indice"].to_numpy(float), index=pd.PeriodIndex(c["mes"], freq="M"))
        else:
            c = pd.read_csv(MACRO / "macro" / "ipca_mensal.csv")
            s = pd.Series((1 + c["ipca_mensal"].to_numpy(float) / 100).cumprod(), index=pd.PeriodIndex(c["mes"], freq="M"))
        return s.sort_index()
    except Exception:  # noqa: BLE001
        return pd.Series(dtype=float)


# ================================================================== leitura (EUA, Yahoo Finance)
@st.cache_data(ttl=12 * 3600, max_entries=30, show_spinner="Baixando cotações do Yahoo Finance…")
def precos_eua(ticker: str) -> tuple[pd.DataFrame, pd.DataFrame]:
    """(preços brutos, eventos) de um ticker americano. O Yahoo devolve o fechamento já ajustado por desdobramentos;
    aqui ele volta ao preço negociado na data (fechamento × desdobramentos posteriores)."""
    import yfinance as yf
    h = yf.Ticker(ticker).history(start="2015-01-01", auto_adjust=False, actions=True)
    if h is None or h.empty:
        return pd.DataFrame(), pd.DataFrame()
    h = h.reset_index()
    h["data"] = pd.to_datetime(h["Date"]).dt.tz_localize(None).dt.normalize()
    sp = h["Stock Splits"].replace(0, 1).fillna(1).to_numpy(float)
    posteriores = np.cumprod(sp[::-1])[::-1] / sp            # produto dos desdobramentos depois de cada data
    p = pd.DataFrame({"data": h["data"], "ticker": ticker, "preco": h["Close"].to_numpy(float) * posteriores,
                      "volume": (h["Volume"] * h["Close"]).to_numpy(float)})
    ev = []
    for d, v, s in zip(h["data"], h["Dividends"].fillna(0), h["Stock Splits"].fillna(0)):
        if v:   # dividendo do Yahoo vem ajustado por desdobramentos posteriores: volta à base da data
            ev.append({"data_ex": d, "tipo": "dinheiro", "evento": "DIVIDENDO", "valor": float(v), "fator": np.nan})
        if s and s != 1:
            ev.append({"data_ex": d, "tipo": "acoes", "evento": "DESDOBRAMENTO" if s > 1 else "GRUPAMENTO",
                       "valor": np.nan, "fator": float(s)})
    ev = pd.DataFrame(ev, columns=["data_ex", "tipo", "evento", "valor", "fator"])
    if len(ev):
        pos = dict(zip(p["data"], posteriores))
        m = ev["tipo"] == "dinheiro"
        ev.loc[m, "valor"] = ev.loc[m, "valor"] * ev.loc[m, "data_ex"].map(pos).fillna(1)
    return p, ev


# ================================================================== eventos e ajustes (cálculo próprio)
def fator_acoes_b3(evento: str, fator_b3: float) -> float:
    """Novas ações por ação antiga a partir do fator publicado pela B3.
    A B3 publica desdobramento e bonificação em % de ações novas (100 = 1 nova para cada 1 antiga → 2,0) e grupamento
    como a quantidade resultante por ação antiga (0,1 = 10 para 1)."""
    if pd.isna(fator_b3) or fator_b3 <= 0:
        return np.nan
    e = _n(evento)
    if e.startswith("GRUPAMENTO"):
        return fator_b3 if fator_b3 < 1 else 1 / fator_b3
    if e.startswith("DESDOBRAMENTO") or e.startswith("BONIFICACAO"):
        return 1 + fator_b3 / 100
    return np.nan


def _classe_isin(isin: str) -> str:
    """Espécie a partir do ISIN da B3 (BRPETRACNPR6 -> PN; BRTAEECDAM10 -> UNT)."""
    i = str(isin or "").upper()
    if len(i) < 11:
        return ""
    if i[6:10] == "CDAM":
        return "UNT"
    return {"ACNOR": "ON", "ACNPR": "PN", "ACNPA": "PNA", "ACNPB": "PNB", "ACNPC": "PNC", "ACNPD": "PND"}.get(i[6:11], "")


CLASSE_SUF = {"3": "ON", "4": "PN", "5": "PNA", "6": "PNB", "7": "PNC", "8": "PND", "11": "UNT"}


def eventos_br(prov: pd.DataFrame, precos_tk: pd.DataFrame) -> pd.DataFrame:
    """Eventos do papel, com a data ex = primeiro pregão depois da data-com.
    Fontes da B3: complemento recente (por ISIN) e histórico de proventos em dinheiro (por espécie ON/PN/UNT).
    O histórico da B3 às vezes lista um provento só na ON quando ON e PN recebem o mesmo valor; quando a espécie do
    papel não tem linha naquela data-com, usa-se a da ON (indicado em 'obs'). Units só usam linhas próprias."""
    vazio = pd.DataFrame(columns=["data_ex", "data_com", "tipo", "evento", "valor", "fator", "obs"])
    if prov.empty or precos_tk.empty:
        return vazio
    tk = str(precos_tk["ticker"].iloc[-1])
    raiz = str(precos_tk["raiz"].iloc[-1]) if "raiz" in precos_tk else tk[:4]
    alvo = CLASSE_SUF.get(tk[4:], "")
    isins = set(precos_tk["isin"].dropna().astype(str)) - {""}
    e = prov[prov["raiz"].eq(raiz) | prov["isin"].isin(isins)].copy()
    if e.empty:
        return vazio
    e["classe"] = [(_classe_isin(i) if str(i or "") else _n(c).replace(" ", ""))
                   for i, c in zip(e["isin"].fillna(""), e["classe_b3"].fillna(""))]
    e["classe"] = e["classe"].replace({"UNIT": "UNT", "UNITS": "UNT"})
    proprias = e[e["isin"].isin(isins) | (e["classe"] == alvo)]
    if alvo not in ("ON", "UNT", ""):
        on = e[(e["classe"] == "ON") & (e["tipo"] == "dinheiro")]
        chave = lambda x: x["data_com"].dt.strftime("%Y-%m-%d") + "|" + x["evento"]  # noqa: E731
        falta = on[~chave(on).isin(set(chave(proprias[proprias["tipo"] == "dinheiro"])))]
        if len(falta):
            falta = falta.assign(obs="valor da ON (a B3 listou o provento só na ON nesta data-com)")
            proprias = pd.concat([proprias, falta])
    e = proprias.assign(_v=proprias["valor"].round(6)).drop_duplicates(
        ["data_com", "tipo", "evento", "_v", "fator_b3"]).drop(columns="_v")
    datas = precos_tk["data"].sort_values().to_numpy()
    e = e[e["data_com"] >= pd.Timestamp(datas[0])]          # só eventos dentro da série de preços
    if e.empty:
        return vazio
    k = np.searchsorted(datas, e["data_com"].to_numpy(dtype="datetime64[ns]"), side="right")
    e["data_ex"] = [pd.Timestamp(datas[i]) if i < len(datas) else pd.NaT for i in k]
    e["fator"] = [fator_acoes_b3(ev, f) if t == "acoes" else np.nan for ev, f, t in zip(e["evento"], e["fator_b3"], e["tipo"])]
    e = e.dropna(subset=["data_ex"])
    return e[["data_ex", "data_com", "tipo", "evento", "valor", "fator", "obs"]].sort_values("data_ex")


def ajustar(precos_tk: pd.DataFrame, eventos: pd.DataFrame, por_proventos: bool, por_eventos: bool,
            real: bool, indice: pd.Series) -> tuple[pd.Series, list[str]]:
    """Série de fechamento ajustada (cálculo próprio).
    - eventos: preços antes da data ex ÷ (novas ações por antiga);
    - proventos: preços antes da data ex × (1 − provento ÷ fechamento da data-com) — equivale a reinvestir o provento
      no próprio papel (retorno total); JCP pelo valor bruto;
    - real: preço × índice de preços do último mês disponível ÷ índice do mês da cotação."""
    s = precos_tk.set_index("data")["preco"].astype(float).sort_index()
    notas = []
    fator = pd.Series(1.0, index=s.index)
    if len(eventos):
        if por_eventos:
            for r in eventos[(eventos["tipo"] == "acoes") & eventos["fator"].notna()].itertuples():
                fator[fator.index < r.data_ex] /= r.fator
            ign = eventos[(eventos["tipo"] == "acoes") & eventos["fator"].isna()]
            if len(ign):
                notas.append("eventos em ações não aplicados (sem fator de quantidade): " +
                             ", ".join(sorted(set(ign["evento"]))))
        if por_proventos:
            dn = eventos[(eventos["tipo"] == "dinheiro") & (eventos["valor"] > 0)]
            for dex, g in dn.groupby("data_ex"):
                ant = s[s.index < dex]
                if ant.empty or ant.iloc[-1] <= 0:
                    continue
                f_ = 1 - g["valor"].sum() / ant.iloc[-1]
                if 0 < f_ < 1:
                    fator[fator.index < dex] *= f_
                else:
                    notas.append(f"provento de {pd.Timestamp(dex).date()} ignorado (maior que o preço: confira a fonte)")
    aj = s * fator
    if real and len(indice):
        meses = aj.index.to_period("M")
        idx = indice.reindex(meses).ffill().bfill().to_numpy()
        aj = aj * indice.iloc[-1] / idx
        notas.append(f"valores em moeda de {indice.index[-1].strftime('%m/%Y')} (último índice de preços disponível)")
    return aj, notas


# ================================================================== tela: cotação, valor de mercado e EV
def secao_cotacao(raiz: str, cnpj: str, nome: str, eua: bool = False, ticker_eua: str | None = None,
                  vm_ev_eua: pd.DataFrame | None = None, moeda: str = "R$", chave: str = "cot"):
    st.subheader("Cotação, valor de mercado e EV")
    if eua:
        if not ticker_eua:
            st.info("Sem ticker para esta empresa.")
            return
        try:
            p, ev = precos_eua(ticker_eua)
        except Exception as e:  # noqa: BLE001
            st.warning(f"Não consegui baixar as cotações do Yahoo Finance agora ({e}). Tente de novo em alguns minutos.")
            return
        if p.empty:
            st.info("Yahoo Finance não devolveu cotações para este ticker.")
            return
        tickers = [ticker_eua]
    else:
        p = precos_br(raiz, _versao("precos_*.parquet"))
        if p.empty:
            st.info("Cotações diárias ainda não geradas (saidas/mercado/precos_<ano>.parquet): aparecem depois da "
                    "próxima execução do coletor setorial.")
            return
        liq = p[p["data"] >= p["data"].max() - pd.Timedelta(days=120)].groupby("ticker")["volume"].sum()
        tickers = sorted(p["ticker"].unique(), key=lambda t: -liq.get(t, 0))
        prov = proventos_br(_versao("proventos.parquet"))
    c1, c2, c3, c4 = st.columns([1.3, 1.2, 1.2, 1.2])
    tk = c1.selectbox("Papel", tickers, key=f"{chave}_tk",
                      help="Ordenados pelo volume negociado nos últimos 120 dias (o primeiro é o mais líquido).")
    a_prov = c2.toggle("Ajustar por proventos", value=False, key=f"{chave}_prov",
                       help="Dividendos, JCP (bruto) e rendimentos: preço antes da data ex × (1 − provento ÷ preço da "
                            "data-com). Mostra o retorno total, com reinvestimento no próprio papel.")
    a_ev = c3.toggle("Ajustar por desdobramentos, grupamentos e bonificações", value=True, key=f"{chave}_ev")
    real = c4.toggle("Valores reais (moeda de hoje)", value=False, key=f"{chave}_real",
                     help=("Corrige pelo CPI-U (BLS/FRED)" if eua else "Corrige pelo IPCA (IBGE, via Banco Central)")
                          + " até o último mês divulgado.")
    ptk = p[p["ticker"] == tk].sort_values("data")
    ev_tk = ev if eua else eventos_br(prov, ptk)
    serie, notas = ajustar(ptk, ev_tk, a_prov, a_ev, real, indice_precos(eua))
    ini = st.select_slider("Período", options=["1 ano", "3 anos", "5 anos", "Desde 2015"], value="Desde 2015",
                           key=f"{chave}_jan")
    anos = {"1 ano": 1, "3 anos": 3, "5 anos": 5}.get(ini)
    if anos:
        serie = serie[serie.index >= serie.index.max() - pd.DateOffset(years=anos)]
    fig = go.Figure(go.Scatter(x=serie.index, y=serie.values, mode="lines", name=tk, line=dict(width=1.6)))
    if len(ev_tk):
        e_vis = ev_tk[(ev_tk["data_ex"] >= serie.index.min())]
        for tipo, simb, cor, rot in (("dinheiro", "triangle-down", "#16a34a", "Provento (data ex)"),
                                     ("acoes", "diamond", "#d97706", "Evento em ações (data ex)")):
            e_ = e_vis[e_vis["tipo"] == tipo]
            if len(e_):
                y_ = serie.reindex(e_["data_ex"]).to_numpy()
                txt = [f"{r.evento}: " + (f"{moeda} {r.valor:,.4f}".replace(",", "X").replace(".", ",").replace("X", ".")
                                         if tipo == "dinheiro" else f"fator {r.fator:g} ação(ões) por ação")
                       for r in e_.itertuples()]
                fig.add_trace(go.Scatter(x=e_["data_ex"], y=y_, mode="markers", name=rot, text=txt,
                                         hovertemplate="%{x|%d/%m/%Y}<br>%{text}<extra></extra>",
                                         marker=dict(symbol=simb, size=8, color=cor)))
    desc = (("ajustado por proventos" if a_prov else "") + (" e " if a_prov and a_ev else "") +
            ("ajustado por eventos" if a_ev else "")) or "sem ajuste (preço negociado na data)"
    fig.update_layout(height=420, margin=dict(t=40, b=20, l=10, r=10), legend=dict(orientation="h", y=-0.15),
                      title=f"{tk}: fechamento {desc}" + (" — valores reais" if real else ""),
                      yaxis_title=moeda + (" de hoje" if real else ""))
    st.plotly_chart(fig, width="stretch", key=f"{chave}_fig")
    if len(serie) > 1:
        r_tot = serie.iloc[-1] / serie.iloc[0] - 1
        anos_ = (serie.index[-1] - serie.index[0]).days / 365.25
        st.caption(f"Variação no período: {r_tot:+.1%}".replace(".", ",") +
                   (f" ({(1 + r_tot) ** (1 / anos_) - 1:+.1%} ao ano)".replace(".", ",") if anos_ >= 1 and r_tot > -1 else "")
                   + f" · {desc}" + (" · valores reais" if real else "") + ".")
    st.caption(("Fonte: Yahoo Finance (cotações, dividendos e desdobramentos). " if eua else
                "Fonte: série histórica da B3 (COTAHIST, fechamento sem ajuste) e proventos da página de empresas "
                "listadas da B3 (data-com, valor bruto por ação). ") +
               "Ajustes e correção pela inflação: cálculo próprio. " + " ".join(n.capitalize() + "." for n in notas))
    if len(ev_tk):
        with st.expander(f"Proventos e eventos do papel ({len(ev_tk)})"):
            vis = ev_tk.sort_values("data_ex", ascending=False).copy()
            for c in ("data_ex", "data_com"):
                if c in vis:
                    vis[c] = pd.to_datetime(vis[c]).dt.date
            st.dataframe(vis, hide_index=True, width="stretch", column_config={
                "data_ex": "Data ex", "data_com": "Data-com", "tipo": "Tipo", "evento": "Evento",
                "valor": st.column_config.NumberColumn(f"Valor por ação ({moeda})", format="%.6f"),
                "fator": st.column_config.NumberColumn("Ações por ação antiga", format="%.4f"), "obs": "Observação"})
            if not eua:
                anual = ev_tk[ev_tk["tipo"] == "dinheiro"].assign(ano=lambda x: x["data_ex"].dt.year) \
                    .groupby("ano")["valor"].sum()
                if len(anual):
                    fig2 = go.Figure(go.Bar(x=anual.index, y=anual.values, marker_color="#16a34a"))
                    fig2.update_layout(height=260, margin=dict(t=30, b=20, l=10, r=10),
                                       title=f"Proventos por ação por ano da data ex ({moeda}, sem ajuste por eventos)")
                    st.plotly_chart(fig2, width="stretch", key=f"{chave}_prov_ano")

    # ---- valor de mercado e EV diários
    v = vm_ev_eua if eua else vm_ev_br(cnpj, _versao("vm_ev_*.parquet"))
    if v is None or v.empty:
        return
    v = v.sort_values("data")
    if anos:
        v = v[v["data"] >= v["data"].max() - pd.DateOffset(years=anos)]
    cols = [c for c in ("valor_mercado", "ev_com_arrend", "ev_sem_arrend") if c in v and v[c].notna().any()]
    nomes = {"valor_mercado": "Valor de mercado", "ev_com_arrend": "EV c/ arrendamento", "ev_sem_arrend": "EV s/ arrendamento"}
    sel = st.multiselect("Valor de mercado e EV", cols, default=cols[:2], format_func=nomes.get, key=f"{chave}_vmev")
    fig3 = go.Figure()
    fat = 1.0
    if real:
        idx = indice_precos(eua)
        if len(idx):
            fat = idx.iloc[-1] / idx.reindex(v["data"].dt.to_period("M")).ffill().bfill().to_numpy()
    for c in sel:
        fig3.add_trace(go.Scatter(x=v["data"], y=v[c].to_numpy(float) * fat / 1000, mode="lines", name=nomes[c]))
    fig3.update_layout(height=360, margin=dict(t=40, b=20, l=10, r=10), legend=dict(orientation="h", y=-0.2),
                       title=f"{moeda} bilhões" + (" de hoje" if real else ""), yaxis_title=f"{moeda} bi")
    st.plotly_chart(fig3, width="stretch", key=f"{chave}_vmev_fig")
    st.caption("Valor de mercado diário = Σ classes (ações vigentes em cada data × fechamento da classe). EV = valor de "
               "mercado + dívida líquida + não controladores do último ITR/DFP já divulgado em cada data (ponto no "
               "tempo). Cálculo próprio." if not eua else
               "Pontos de fim de trimestre fiscal e último pregão (os EUA não têm série diária de ações na base).")


# ================================================================== tela: mapa de calor
PESOS = {"vm_fim": "valor de mercado", "liq": "liquidez diária"}


def mapa_calor(v: pd.DataFrame, rotulo_periodo: str, moeda: str = "R$", chave: str = "mapa", peso: str = "vm_fim",
               marcas: dict | None = None):
    """v: colunas rotulo, setor, subsetor, vm_fim (milhões), variacao e, opcional, liq (volume médio diário, milhões).
    peso: 'vm_fim' (área = valor de mercado) ou 'liq' (área = liquidez diária). marcas: {índice: variação} para a
    régua de cores."""
    v = v.dropna(subset=["variacao", peso])
    v = v[v[peso] > 0].copy()
    if v.empty:
        st.info("Sem dados para o mapa de calor" + (" (liquidez diária ainda não disponível para estas empresas)."
                                                    if peso == "liq" else "."))
        return
    c1, c2 = st.columns([1, 1])
    nivel = c1.radio("Agrupar por", ["Setor", "Setor e subsetor"], horizontal=True, key=f"{chave}_niv")
    n_max = c2.slider(f"Empresas no mapa (as de maior {PESOS[peso]})", min(30, len(v)), max(30, len(v)),
                      min(150, len(v)), 10, key=f"{chave}_n")
    v = v.nlargest(n_max, peso)
    if "liq" not in v.columns:
        v["liq"] = np.nan
    for c_, vazio in (("setor", "Sem setor"), ("subsetor", "—"), ("rotulo", "?")):
        v[c_] = v[c_].astype("object").where(v[c_].notna(), vazio).astype(str).replace({"": vazio, "nan": vazio})
    v["tk"] = v["rotulo"].astype(str).str.split(" — ").str[0]
    v["vm_ini"] = v["vm_fim"] / (1 + v["variacao"])

    def pond(g):
        if peso == "vm_fim":                               # Σ valor final ÷ Σ valor inicial − 1
            return g["vm_fim"].sum() / g["vm_ini"].sum() - 1
        return float(np.average(g["variacao"], weights=g["liq"]))     # média ponderada pela liquidez

    def soma(g):
        return g[peso].sum()

    ids, par, lab, val, cor, hov, vms, lqs = [], [], [], [], [], [], [], []
    for s_, gs in v.groupby("setor"):
        ids.append(f"S|{s_}"); par.append(""); lab.append(s_); val.append(soma(gs)); cor.append(pond(gs))
        hov.append(f"{len(gs)} empresas"); vms.append(gs["vm_fim"].sum()); lqs.append(gs["liq"].sum())
        if nivel == "Setor e subsetor":
            for ss, gss in gs.groupby("subsetor"):
                ids.append(f"SS|{s_}|{ss}"); par.append(f"S|{s_}"); lab.append(ss); val.append(soma(gss))
                cor.append(pond(gss)); hov.append(f"{len(gss)} empresas")
                vms.append(gss["vm_fim"].sum()); lqs.append(gss["liq"].sum())
        for r in gs.itertuples():
            ids.append(f"E|{r.rotulo}"); par.append(f"SS|{s_}|{r.subsetor}" if nivel == "Setor e subsetor" else f"S|{s_}")
            lab.append(r.tk); val.append(getattr(r, peso)); cor.append(r.variacao); hov.append(str(r.rotulo))
            vms.append(r.vm_fim); lqs.append(r.liq)
    lim = float(np.nanquantile(np.abs(v["variacao"]), 0.9)) or 0.01
    marcas = {k: x for k, x in (marcas or {}).items() if x is not None and np.isfinite(x)}
    lim = max(lim, *(abs(x) * 1.15 for x in marcas.values())) if marcas else lim   # índice sempre dentro da régua
    liq_txt = np.array([f"{moeda} {x:,.1f} mi/dia" if pd.notna(x) and x > 0 else "n/d" for x in lqs], dtype=object)
    fig = go.Figure(go.Treemap(
        ids=ids, parents=par, labels=lab, values=val, branchvalues="total",
        customdata=np.column_stack([cor, hov, np.array(vms, dtype=float) / 1000, liq_txt]),
        texttemplate="<b>%{label}</b><br>%{customdata[0]:+.2%}",
        hovertemplate="<b>%{label}</b><br>%{customdata[1]}<br>Variação: %{customdata[0]:+.2%}<br>"
                      f"Valor de mercado: {moeda} " + "%{customdata[2]:,.1f} bi<br>"
                      "Liquidez média: %{customdata[3]}<extra></extra>",
        marker=dict(colors=cor, colorscale=ESCALA, cmid=0, cmin=-lim, cmax=lim, showscale=False),
        tiling=dict(pad=1), pathbar=dict(visible=True)))
    fig.update_layout(height=640, margin=dict(t=30, b=10, l=10, r=10),
                      title=f"Mapa de calor — {rotulo_periodo} — área: {PESOS[peso]}")
    st.plotly_chart(fig, width="stretch", key=f"{chave}_fig")
    regua(lim, marcas, chave=f"{chave}_regua")
    if peso == "vm_fim":
        txt = ("Área = valor de mercado no pregão final. Setores e subsetores: variação ponderada pelo valor de mercado "
               "(Σ valor final ÷ Σ valor inicial − 1).")
    else:
        txt = (f"Área = liquidez diária: volume financeiro médio dos 21 pregões até a data final ({moeda} milhões/dia, "
               "todas as classes de ações somadas). Setores e subsetores: média das variações ponderada pela liquidez.")
    st.caption(txt + " Cor = variação no período; a régua abaixo do mapa é a escala de cores (saturada no percentil 90 "
               "das variações absolutas, ampliada se preciso para caber os índices), com a variação do(s) índice(s) "
               "de referência no mesmo período. Clique num setor para ampliar. Cálculo próprio.")


ESCALA = [[0, "#b91c1c"], [0.5, "#3f3f46"], [1, "#15803d"]]
CORES_INDICE = {"Ibovespa": "#f59e0b", "S&P 500": "#38bdf8", "Nasdaq Composite": "#c084fc"}


def regua(lim: float, marcas: dict, chave: str = "regua"):
    """Régua horizontal da escala de cores do mapa (−lim a +lim) com uma marcação por índice de referência."""
    x = np.linspace(-lim, lim, 241)
    fig = go.Figure(go.Heatmap(x=x, y=[0], z=[x], colorscale=ESCALA, zmid=0, zmin=-lim, zmax=lim, showscale=False,
                               hovertemplate="%{x:+.2%}<extra></extra>"))
    fig.add_vline(x=0, line_color="#e5e7eb", line_width=1)
    for k, (nome, r) in enumerate(sorted(marcas.items(), key=lambda t: t[1])):
        cor = CORES_INDICE.get(nome, "#facc15")
        fig.add_shape(type="line", x0=r, x1=r, y0=-0.5, y1=0.5, line=dict(color=cor, width=3))
        fig.add_trace(go.Scatter(x=[r], y=[0.5], mode="markers", marker=dict(symbol="triangle-down", size=13, color=cor,
                                 line=dict(color="white", width=1)), name=nome, showlegend=False,
                                 hovertemplate=f"{nome}: {r:+.2%}<extra></extra>"))
        fig.add_annotation(x=r, y=0.5, yshift=14 + 16 * (k % 2), text=f"<b>{nome} {r:+.2%}</b>", showarrow=False,
                           font=dict(color=cor, size=12))
    fig.update_layout(height=120 if marcas else 80, margin=dict(t=48 if marcas else 6, b=24, l=10, r=10),
                      xaxis=dict(tickformat=".1%", range=[-lim, lim], showgrid=False, zeroline=False),
                      yaxis=dict(visible=False, range=[-0.5, 0.5]))
    st.plotly_chart(fig, width="stretch", key=chave, config={"displayModeBar": False})


# ================================================================== tela: estrutura societária
def _rot(t, largura=26):
    return "\\n".join(textwrap.wrap(str(t).replace('"', "'"), largura)) or "?"


def _col(df, *ps, ex=()):
    nrm = lambda c: unicodedata.normalize("NFKD", c).encode("ascii", "ignore").decode().lower()  # noqa: E731
    return next((c for c in df.columns if all(p in nrm(c) for p in ps) and not any(x in nrm(c) for x in ex)), None)


def secao_estrutura(cnpj: str, nome: str, eua: bool = False, chave: str = "soc"):
    st.subheader("Estrutura societária")
    if eua:
        st.info("Ainda não disponível para empresas americanas: a SEC não publica a posição acionária em formato "
                "estruturado como o Formulário de Referência da CVM (13D/13G e proxy statements são texto livre).")
        return
    a, d = acionistas_br(_versao("acionistas.parquet", "distribuicao_capital.parquet"))
    if a.empty:
        st.info("Estrutura societária ainda não coletada (saidas/mercado/acionistas.parquet; workflow 'mercado').")
        return
    b = a[a["cnpj"] == str(cnpj).zfill(14)].copy()
    if b.empty:
        st.info("A companhia não tem posição acionária no Formulário de Referência dos últimos anos.")
        return
    c_nome = _col(b, "acionista", ex=("cpf", "cnpj", "tipo", "id", "controlador", "residente", "relacion", "acordo",
                                      "nacionalidade", "estado", "pais", "data", "quantidade", "percentual"))
    c_id, c_pai = _col(b, "id", "acionista", ex=("relacion",)), _col(b, "relacion")
    c_pct = _col(b, "percentual", "total")
    c_on, c_pn = _col(b, "percentual", "ordinaria"), _col(b, "percentual", "preferencial")
    c_ctrl, c_ref = _col(b, "controlador"), _col(b, "data", "referencia")
    if not (c_nome and c_pct):
        st.warning(f"Formato inesperado do FRE (colunas: {list(b.columns)[:10]}…).")
        return
    num = lambda s: pd.to_numeric(s.astype(str).str.replace(",", ".", regex=False), errors="coerce")  # noqa: E731
    b["pct"] = num(b[c_pct])
    b["pct_on"] = num(b[c_on]) if c_on else np.nan
    b["pct_pn"] = num(b[c_pn]) if c_pn else np.nan
    b["ctrl"] = b[c_ctrl].astype(str).str.upper().str.startswith("S") if c_ctrl else False
    b["nm"] = b[c_nome].astype(str).str.strip()
    b = b[~b["nm"].map(_n).str.contains(r"TESOURARIA|^OUTROS$|^TOTAL$", regex=True)]
    ref = b[c_ref].iloc[0] if c_ref else ""
    # nível 1 = acionistas diretos da companhia (sem acionista relacionado); níveis seguintes = acionistas dos acionistas
    pai_de = {str(i): str(p).strip() for i, p in zip(b[c_id], b[c_pai])} if c_id and c_pai else {}

    def nivel(i, k=0):
        p_ = pai_de.get(i, "")
        return 1 if p_ in ("", "nan", "None", "0") or p_ not in pai_de or k > 10 else 1 + nivel(p_, k + 1)

    b["idac"] = b[c_id].astype(str) if c_id else b.index.astype(str)
    b["nv"] = [nivel(i) for i in b["idac"]] if c_id else 1
    diretos = b[b["nv"] == 1]
    # free float informado no FRE (ações em circulação)
    ff = np.nan
    if len(d):
        dd = d[d["cnpj"] == str(cnpj).zfill(14)]
        c_ff = _col(dd, "percentual", "total", "circulacao") if len(dd) else None
        if c_ff:
            ff = num(dd[c_ff]).dropna().iloc[-1] if num(dd[c_ff]).notna().any() else np.nan
    k1, k2 = st.columns([1.25, 1])
    with k1:
        niveis = st.slider("Níveis da cadeia de controle", 1, 4, 2, key=f"{chave}_niv")
        pct_min = st.slider("Participação mínima nos níveis acima do 1º (%)", 0.0, 50.0, 5.0, 1.0, key=f"{chave}_min")
        nomes_id = dict(zip(b[c_id].astype(str), b["nm"])) if c_id else {}
        linhas = ['digraph G {', 'rankdir=BT; bgcolor="transparent"; nodesep=0.25; ranksep=0.45;',
                  'node [shape=box, style="rounded,filled", fontname="Helvetica", fontsize=10, color="#888888", '
                  'fillcolor="#F4F4F4", fontcolor="#222222"];',
                  'edge [color="#888888", fontname="Helvetica", fontsize=9, fontcolor="#555555"];',
                  f'"{_rot(nome)}" [fillcolor="#1f77b4", fontcolor="white", penwidth=2, fontsize=12];']
        incl, vistos = set(), set()
        for r in b.sort_values(["nv", "pct"], ascending=[True, False]).itertuples():
            i = str(r.idac)
            pai = pai_de.get(i, "")
            if r.nv > niveis or pd.isna(r.pct) or r.pct <= 0 or (r.nv > 1 and (r.pct < pct_min or pai not in incl)):
                continue
            incl.add(i)
            o, d_ = _rot(r.nm), _rot(nomes_id.get(pai, nome) if r.nv > 1 else nome)
            if (o, d_) in vistos or o == d_:
                continue
            vistos.add((o, d_))
            if r.ctrl:
                linhas.append(f'"{o}" [color="#1f77b4", penwidth=2];')
            lab = f"{r.pct:.1f}%".replace(".", ",")
            linhas.append(f'"{o}" -> "{d_}" [label="{lab}"{", penwidth=2" if r.ctrl else ""}];')
        if pd.notna(ff):
            linhas.append(f'"Free float (FRE)" [fillcolor="#E8F5E9"]; "Free float (FRE)" -> "{_rot(nome)}" '
                          f'[label="{str(round(ff, 1)).replace(".", ",")}%", style=dashed];')
        linhas.append("}")
        if len(vistos):
            st.graphviz_chart("\n".join(linhas), width="stretch")
        else:
            st.info("Sem acionistas com participação informada.")
    with k2:
        dir_ = diretos.groupby("nm", as_index=False).agg(pct=("pct", "max"), pct_on=("pct_on", "max"),
                                                           pct_pn=("pct_pn", "max"), ctrl=("ctrl", "max"))
        dir_ = dir_.sort_values("pct", ascending=False)
        resto = 100 - dir_["pct"].sum()
        fat = dir_[["nm", "pct"]].copy()
        if resto > 0.05:
            fat = pd.concat([fat, pd.DataFrame({"nm": ["Demais acionistas / mercado"], "pct": [resto]})])
        fig = go.Figure(go.Pie(labels=fat["nm"].str.slice(0, 40), values=fat["pct"], hole=0.5, sort=False,
                               textinfo="percent", hovertemplate="%{label}: %{value:.2f}%<extra></extra>"))
        fig.update_layout(height=340, margin=dict(t=30, b=10, l=10, r=10), showlegend=True,
                          legend=dict(orientation="h", y=-0.1, font=dict(size=10)),
                          title="Capital total — acionistas diretos")
        st.plotly_chart(fig, width="stretch", key=f"{chave}_pie")
        if dir_["pct_on"].notna().any() and dir_["pct_pn"].notna().any() and dir_["pct_pn"].fillna(0).sum() > 0:
            top = dir_.head(8)
            fig2 = go.Figure([go.Bar(y=top["nm"].str.slice(0, 32), x=top[c], orientation="h", name=n)
                              for c, n in (("pct_on", "% das ON"), ("pct_pn", "% das PN"), ("pct", "% do total"))])
            fig2.update_layout(height=320, barmode="group", margin=dict(t=30, b=10, l=10, r=10),
                               xaxis_ticksuffix="%", title="Direito de voto (ON) × capital (PN e total)",
                               legend=dict(orientation="h", y=-0.15), yaxis=dict(autorange="reversed"))
            st.plotly_chart(fig2, width="stretch", key=f"{chave}_bar")
    vis = dir_.rename(columns={"nm": "Acionista", "pct": "% total", "pct_on": "% ON", "pct_pn": "% PN",
                               "ctrl": "Controlador"})
    with st.expander("Tabela de acionistas diretos"):
        st.dataframe(vis, hide_index=True, width="stretch",
                     column_config={c: st.column_config.NumberColumn(c, format="%.2f%%") for c in ("% total", "% ON", "% PN")})
    st.caption(f"Fonte: Formulário de Referência (CVM Dados Abertos, posição acionária, item 15.1), referência {ref}. "
               "Setas = participação no capital total; borda grossa = controlador; níveis acima do 1º são os acionistas "
               "dos acionistas, como a companhia informa a cadeia de controle. Free float: ações em circulação "
               "informadas no FRE. Esquema e gráficos montados aqui (cálculo próprio).")


# ================================================================== tela: modelo de negócio e histórico (curadoria)
@st.cache_data(max_entries=200, show_spinner=False)
def _curadoria(raiz: str, mt: float) -> dict | None:
    import yaml
    p = CURADORIA / f"{raiz.upper()}.yaml"
    if not p.exists():
        return None
    return yaml.safe_load(p.read_text(encoding="utf-8")) or None


def secao_modelo(raiz: str, nome: str):
    st.subheader("Modelo de negócio e breve histórico")
    p = CURADORIA / f"{str(raiz).upper()}.yaml"
    c = _curadoria(str(raiz), p.stat().st_mtime if p.exists() else 0)
    if not c:
        st.info(f"Texto de {nome} em curadoria: as empresas entram aos poucos, começando pelas maiores e pelas "
                "acompanhadas no Price Discovery.")
        return
    if c.get("modelo_negocio"):
        st.markdown("**Como a empresa ganha dinheiro**")
        st.markdown(str(c["modelo_negocio"]).strip())
    if c.get("segmentos"):
        st.markdown("**Principais negócios**")
        st.markdown("\n".join(f"- {s}" for s in c["segmentos"]))
    if c.get("historico"):
        st.markdown("**Breve histórico**")
        st.markdown("\n".join(f"- **{h.get('ano', '')}** — {h.get('fato', '')}" if isinstance(h, dict) else f"- {h}"
                              for h in c["historico"]))
    fontes = c.get("fontes") or []
    st.caption("Texto de curadoria do Price Discovery, redigido a partir das fontes: " +
               ("; ".join(f"[{f.get('nome', f.get('url'))}]({f.get('url')})" if isinstance(f, dict) else str(f)
                          for f in fontes) or "—") + (f". Revisado em {c['revisado_em']}." if c.get("revisado_em") else "."))
