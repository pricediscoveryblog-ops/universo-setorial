"""
Aba "Calendário de resultados" do Price Discovery Scanning USA.

Fontes (geradas pelo coletor-eua, passo calendario_eua.py, branch dados):
  calendario_eua.parquet  calendário de resultados da Nasdaq (press release): datas agendadas, horário,
                          consenso de EPS e, nas datas passadas, EPS realizado e surpresa.
  entregas_eua.parquet    1a entrega do 10-Q/10-K de cada trimestre à SEC (XBRL companyfacts).
Nos EUA o resultado sai primeiro no press release (8-K) e o 10-Q/10-K costuma vir no mesmo dia ou semanas depois;
por isso as duas datas aparecem separadas.
"""
from __future__ import annotations

import io

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st

import eua

COR_HORARIO = {"Antes da abertura": "#2563eb", "Após o fechamento": "#7c3aed", "Não informado": "#9ca3af"}


def _tk(s: pd.Series) -> pd.Series:
    """Ticker normalizado para cruzar Nasdaq e SEC (BRK.B / BRK/B / BRK-B -> BRK-B)."""
    return s.fillna("").astype(str).str.upper().str.strip().str.replace(r"[./]", "-", regex=True)


@st.cache_data(ttl=3600, max_entries=1, show_spinner=False)
def carregar(versao_: str = "") -> tuple[pd.DataFrame, pd.DataFrame, dict]:
    def ler(nome):
        try:
            return pd.read_parquet(io.BytesIO(eua._baixar(nome)))
        except Exception:  # noqa: BLE001  (arquivo ainda não gerado pelo coletor)
            return pd.DataFrame()
    cal, ent = ler("calendario_eua.parquet"), ler("entregas_eua.parquet")
    try:
        import json
        meta = json.loads(eua._baixar("calendario_meta.json"))
    except Exception:  # noqa: BLE001
        meta = {}
    if len(cal):
        for c in ("data", "data_ano_anterior"):
            cal[c] = pd.to_datetime(cal[c], errors="coerce")
        cal["tk"] = _tk(cal["ticker"])
    if len(ent):
        ent["dt_fim"] = pd.to_datetime(ent["dt_fim"], errors="coerce")
        ent["dt_arquivamento"] = pd.to_datetime(ent["dt_arquivamento"], errors="coerce")
        ent = ent.dropna(subset=["dt_fim", "dt_arquivamento"])
        ent["dt_civil"] = eua.trimestre_civil(ent["dt_fim"])
        ent["tk"] = _tk(ent["ticker"])
    return cal, ent, meta


def _info_empresas(base: pd.DataFrame, ent: pd.DataFrame, mult: pd.DataFrame) -> pd.DataFrame:
    """tk -> cnpj, empresa, setor, rotulo (só para quem está na base do app: abre a aba Empresa), valor de mercado."""
    partes = []
    if len(base):
        partes.append(base.drop_duplicates("cnpj")[["cnpj", "ticker", "empresa", "setor", "rotulo"]])
    if len(ent):
        e = ent.drop_duplicates("cnpj")[["cnpj", "ticker", "empresa", "setor"]].copy()
        e["rotulo"] = None
        partes.append(e)
    try:
        cad = eua.cadastro(eua.versao())
        if len(cad):
            c = cad[["cnpj", "ticker", "empresa", "setor"]].copy()
            c["rotulo"] = None
            partes.append(c)
    except Exception:  # noqa: BLE001
        pass
    if not partes:
        return pd.DataFrame(columns=["tk", "cnpj", "empresa", "setor", "rotulo", "vm_mm"])
    i = pd.concat(partes, ignore_index=True)
    i["tk"] = _tk(i["ticker"])
    i = i[i["tk"] != ""].drop_duplicates("tk")                   # a base (com rótulo clicável) vem primeiro
    if len(mult) and {"cnpj", "data", "valor_mercado"} <= set(mult.columns):
        vm = mult.dropna(subset=["valor_mercado"]).sort_values("data").drop_duplicates("cnpj", keep="last")
        i = i.merge(vm[["cnpj", "valor_mercado"]].rename(columns={"valor_mercado": "vm_mm"}), on="cnpj", how="left")
    else:
        i["vm_mm"] = np.nan
    return i


def _prazo_sec(vm_mm: pd.Series, anual: pd.Series) -> pd.Series:
    """Prazo da SEC em dias pelo porte. O critério oficial é o free float (public float) do fim do 2o trimestre
    fiscal; aqui o valor de mercado faz o papel do free float (aproximação). Sem valor de mercado: prazo máximo."""
    grande, acel = vm_mm >= 700, (vm_mm >= 75) & (vm_mm < 700)
    d_10k = np.select([grande, acel], [60, 75], 90)
    d_10q = np.select([grande | acel], [40], 45)
    return pd.Series(np.where(anual, d_10k, d_10q), index=vm_mm.index)


def _fmt_bi(v):
    return None if pd.isna(v) else round(v / 1e9, 2)


def render(base: pd.DataFrame, mult: pd.DataFrame, tabela_clicavel) -> None:
    cal, ent, meta = carregar(eua.versao())
    if cal.empty and ent.empty:
        st.info("Calendário ainda não gerado: aparece depois da próxima execução do coletor-eua "
                "(passo 'Calendário de resultados').")
        return
    hoje = pd.Timestamp.today().normalize()
    info = _info_empresas(base, ent, mult)
    sp = eua.constituintes_sp500()
    sp_tk = set(_tk(sp["ticker"])) if len(sp) else set()

    st.caption(
        "**Fontes:** datas de divulgação, horário, consenso e EPS realizado = calendário de resultados da **Nasdaq** "
        "(api.nasdaq.com; consenso de analistas compilado pela Zacks). Entregas de 10-Q/10-K = **SEC** (XBRL, data "
        "da 1ª entrega de cada trimestre). Setor e empresa = cadastro SEC do app. "
        + (f"Última coleta da Nasdaq: {str(meta.get('ultima_coleta_ok', '?'))[:16].replace('T', ' ')} UTC."
           if meta else ""))

    # ---------------- filtros
    f1, f2, f3 = st.columns([1.2, 2, 1])
    univ = f1.radio("Universo", ["Todas do calendário", "Só empresas da base do app", "S&P 500"], key="cal_eua_univ")
    setores = sorted(info["setor"].dropna().replace("", np.nan).dropna().unique())
    sel_set = f2.multiselect("Setores (vazio = todos)", setores, key="cal_eua_set")
    vm_min = f3.number_input("Valor de mercado mínimo (US$ bi)", 0.0, 5000.0, 0.0, 1.0, key="cal_eua_vm")

    def filtrar(d: pd.DataFrame) -> pd.DataFrame:
        d = d.merge(info[["tk", "empresa", "setor", "rotulo", "vm_mm"]], on="tk", how="left")
        if "empresa_nasdaq" in d.columns:
            d["empresa"] = d["empresa"].fillna(d["empresa_nasdaq"])
        if "ticker" not in d.columns:
            d["ticker"] = d["tk"]
        d["setor"] = d["setor"].replace("", np.nan).fillna("Fora do universo SEC do app")
        if univ == "Só empresas da base do app":
            d = d[d["rotulo"].notna()]
        elif univ == "S&P 500":
            d = d[d["tk"].isin(sp_tk)]
        if sel_set:
            d = d[d["setor"].isin(sel_set)]
        if vm_min > 0:
            vm = d["valor_mercado_usd"] / 1e9 if "valor_mercado_usd" in d.columns else d["vm_mm"] / 1e3
            vm = vm.fillna(d["vm_mm"] / 1e3)
            d = d[vm >= vm_min]
        return d

    # ---------------- próximas divulgações
    st.subheader("Próximas divulgações (press release)")
    if cal.empty:
        st.info("Calendário da Nasdaq ainda não coletado.")
    else:
        fut_all = cal[cal["data"] >= hoje]
        lim = fut_all["data"].max() if len(fut_all) else hoje
        c1, c2 = st.columns([1.3, 2])
        per = c1.date_input("Período", (hoje.date(), min(lim, hoje + pd.Timedelta(days=14)).date()),
                                 min_value=hoje.date(), max_value=max(lim, hoje).date(), key="cal_eua_fut",
                                 format="DD/MM/YYYY") if len(fut_all) else (hoje.date(), hoje.date())
        ini, fim = (tuple(per) * 2)[:2] if len(per) == 1 else per
        fut = filtrar(fut_all[(fut_all["data"] >= pd.Timestamp(ini)) & (fut_all["data"] <= pd.Timestamp(fim))])
        fut = fut.sort_values(["data", "valor_mercado_usd"], ascending=[True, False])
        base_eps = fut["eps_ano_anterior"].where(fut["eps_ano_anterior"] > 0)
        fut["cresc_eps"] = (fut["eps_consenso"] / base_eps - 1) * 100
        m1, m2, m3 = st.columns(3)
        m1.metric("Empresas no período", f"{len(fut)}")
        m2.metric("Antes da abertura / após o fechamento",
                  f"{(fut['horario'] == 'Antes da abertura').sum()} / {(fut['horario'] == 'Após o fechamento').sum()}")
        m3.metric("Da base do app", f"{fut['rotulo'].notna().sum()}")
        if len(fut):
            por_dia = fut.groupby([fut["data"].dt.normalize(), "horario"]).size().unstack(fill_value=0)
            fig = go.Figure()
            for h in ["Antes da abertura", "Após o fechamento", "Não informado"]:
                if h in por_dia.columns:
                    fig.add_trace(go.Bar(x=por_dia.index, y=por_dia[h], name=h, marker_color=COR_HORARIO[h]))
            fig.update_layout(barmode="stack", height=300, title="Divulgações por dia", yaxis_title="Empresas",
                              legend=dict(orientation="h", y=-0.25), margin=dict(t=40, b=20, l=10, r=10))
            st.plotly_chart(fig, width="stretch")
            tab = fut.assign(
                data=lambda x: x["data"].dt.date, data_ano_anterior=lambda x: x["data_ano_anterior"].dt.date,
                vm_bi=lambda x: x["valor_mercado_usd"].map(_fmt_bi).fillna((x["vm_mm"] / 1e3).round(2)),
                empresa_tab=lambda x: x["ticker"] + " — " + x["empresa"].fillna(""))
            tabela_clicavel(
                tab[["data", "horario", "empresa_tab", "setor", "trimestre_fiscal", "eps_consenso", "n_estimativas",
                     "eps_ano_anterior", "cresc_eps", "data_ano_anterior", "vm_bi"]],
                tab["rotulo"], "clk_cal_eua_fut", hide_index=True, width="stretch",
                height=min(600, 38 + 35 * len(tab)), column_config={
                    "data": "Data", "horario": "Horário", "empresa_tab": "Empresa", "setor": "Setor",
                    "trimestre_fiscal": "Trimestre fiscal (fim)",
                    "eps_consenso": st.column_config.NumberColumn("EPS consenso (US$)", format="%.2f"),
                    "n_estimativas": st.column_config.NumberColumn("Nº estimativas", format="%d"),
                    "eps_ano_anterior": st.column_config.NumberColumn("EPS mesmo tri. ano ant. (US$)", format="%.2f"),
                    "cresc_eps": st.column_config.NumberColumn(
                        "Cresc. implícito do EPS (a/a)", format="%.1f%%",
                        help="Cálculo do app: consenso ÷ EPS do mesmo trimestre do ano anterior − 1 "
                             "(só quando o EPS anterior é positivo)."),
                    "data_ano_anterior": "Divulgou o tri. do ano ant. em",
                    "vm_bi": st.column_config.NumberColumn("Valor de mercado (US$ bi)", format="%.2f")})
            st.caption("⚠️ A Nasdaq não indica se a data futura foi **confirmada** pela empresa ou é **estimada** pelo "
                       "histórico; datas distantes tendem a ser estimativas. Confira o site de RI para a data oficial. "
                       "O EPS da Nasdaq/Zacks segue a base do consenso (em geral ajustado/non-GAAP), não o lucro "
                       "por ação GAAP do 10-Q. Clique numa linha (empresas da base) para abrir a aba Empresa.")

    # ---------------- divulgadas recentemente
    st.subheader("Divulgadas recentemente: EPS realizado × consenso")
    if not cal.empty:
        pas_all = cal[(cal["data"] < hoje + pd.Timedelta(days=1)) & cal["eps_real"].notna()]
        if pas_all.empty:
            st.info("Ainda sem divulgações passadas com EPS realizado na base.")
        else:
            d0 = max(pas_all["data"].min(), hoje - pd.Timedelta(days=14))
            per_p = st.date_input("Período", (d0.date(), hoje.date()), min_value=pas_all["data"].min().date(),
                                         max_value=hoje.date(), key="cal_eua_pas", format="DD/MM/YYYY")
            ini_p, fim_p = (tuple(per_p) * 2)[:2] if len(per_p) == 1 else per_p
            pas = filtrar(pas_all[(pas_all["data"] >= pd.Timestamp(ini_p)) & (pas_all["data"] <= pd.Timestamp(fim_p))])
            # 10-Q/10-K já entregue? (entrega à SEC com fim de período até ~4 meses antes da divulgação)
            if len(ent) and len(pas):
                e = ent[["tk", "dt_fim", "dt_arquivamento", "documento"]].sort_values("dt_fim")
                x = pas[["tk", "data"]].reset_index()
                x = x.merge(e, on="tk", how="left")
                x = x[(x["dt_fim"] < x["data"]) & (x["dt_fim"] >= x["data"] - pd.Timedelta(days=130))]
                x = x.sort_values("dt_fim").drop_duplicates("index", keep="last").set_index("index")
                pas["doc_sec"] = x["documento"]
                pas["entrega_sec"] = x["dt_arquivamento"]
            else:
                pas["doc_sec"], pas["entrega_sec"] = None, pd.NaT
            n = len(pas)
            sup = pas["surpresa_pct"]
            m1, m2, m3 = st.columns(3)
            m1.metric("Divulgaram no período", f"{n}")
            m2.metric("Acima do consenso", f"{(sup > 0).sum()}", f"{(sup > 0).mean():.0%}" if n else None,
                      delta_color="off")
            m3.metric("Surpresa mediana", f"{sup.median():+.1f}%" if sup.notna().any() else "—")
            pas = pas.sort_values(["data", "valor_mercado_usd"], ascending=[False, False])
            tab = pas.assign(data=lambda x: x["data"].dt.date,
                             entrega_sec=lambda x: pd.to_datetime(x["entrega_sec"]).dt.date,
                             empresa_tab=lambda x: x["ticker"] + " — " + x["empresa"].fillna(""),
                             vm_bi=lambda x: x["valor_mercado_usd"].map(_fmt_bi))
            tabela_clicavel(
                tab[["data", "horario", "empresa_tab", "setor", "trimestre_fiscal", "eps_real", "eps_consenso",
                     "surpresa_pct", "doc_sec", "entrega_sec", "vm_bi"]],
                tab["rotulo"], "clk_cal_eua_pas", hide_index=True, width="stretch",
                height=min(600, 38 + 35 * len(tab)), column_config={
                    "data": "Divulgou em", "horario": "Horário", "empresa_tab": "Empresa", "setor": "Setor",
                    "trimestre_fiscal": "Trimestre fiscal (fim)",
                    "eps_real": st.column_config.NumberColumn("EPS realizado (US$)", format="%.2f"),
                    "eps_consenso": st.column_config.NumberColumn("EPS consenso (US$)", format="%.2f"),
                    "surpresa_pct": st.column_config.NumberColumn("Surpresa", format="%+.1f%%"),
                    "doc_sec": "Documento SEC", "entrega_sec": "Entregue à SEC em",
                    "vm_bi": st.column_config.NumberColumn("Valor de mercado (US$ bi)", format="%.2f")})
            st.caption("Surpresa e EPS: Nasdaq/Zacks (base do consenso, em geral ajustada). 'Entregue à SEC em' vazio = "
                       "o 10-Q/10-K do trimestre ainda não entrou na base da SEC do app (o press release costuma sair "
                       "antes) ou a empresa não está no universo SEC do app.")

    # ---------------- entregas de 10-Q/10-K por trimestre civil (equivalente ao calendário da CVM no app BR)
    st.subheader("Entregas de 10-Q/10-K à SEC por trimestre civil")
    if ent.empty:
        st.info("Datas de entrega à SEC ainda não geradas.")
        return
    refs = sorted(ent["dt_civil"].dropna().unique())
    fins_tri = [pd.Timestamp(d) for d in pd.date_range(end=hoje, periods=2, freq="QE")]
    padrao = fins_tri[-1] if fins_tri[-1] <= hoje else fins_tri[0]
    opcoes = sorted(set(pd.Timestamp(d) for d in refs[-8:]) | {padrao}, reverse=True)
    ref = pd.Timestamp(st.selectbox(
        "Trimestre civil de referência", opcoes, index=opcoes.index(padrao), key="cal_eua_ref",
        format_func=lambda d: f"{(d.month - 1) // 3 + 1}T{str(d.year)[2:]} (fechamentos próximos de "
                              f"{d.strftime('%d/%m/%Y')})"))
    ant_ref = ref - pd.DateOffset(years=1)
    # universo: quem entregou algum trimestre nos 15 meses anteriores
    ativas = ent[(ent["dt_civil"] < ref) & (ent["dt_civil"] >= ref - pd.DateOffset(months=15))]["tk"].unique()
    atual = ent[ent["dt_civil"] == ref].sort_values("dt_arquivamento").drop_duplicates("tk")
    passado = ent[ent["dt_civil"] == ant_ref].sort_values("dt_arquivamento").drop_duplicates("tk")
    t = pd.DataFrame({"tk": sorted(set(ativas) | set(atual["tk"]))})
    t = t.merge(atual[["tk", "documento", "dt_fim", "dt_arquivamento"]], on="tk", how="left").merge(
        passado[["tk", "documento", "dt_fim", "dt_arquivamento"]].rename(
            columns={"documento": "doc_ant", "dt_fim": "fim_ant", "dt_arquivamento": "receb_ant"}), on="tk", how="left")
    t["valor_mercado_usd"] = np.nan
    t = filtrar(t)
    if not cal.empty:   # valor de mercado mais recente da Nasdaq como reserva para o porte
        vm_nq = cal.dropna(subset=["valor_mercado_usd"]).sort_values("data").drop_duplicates("tk", keep="last")
        t["vm_mm"] = t["vm_mm"].fillna(t["tk"].map(vm_nq.set_index("tk")["valor_mercado_usd"] / 1e6))
    divulgou = t["dt_arquivamento"].notna()
    n_tot, n_div = len(t), int(divulgou.sum())
    pend = t[~divulgou].copy()
    fim_est = pend["fim_ant"] + pd.Timedelta(days=364)
    anual = pend["doc_ant"].eq("10-K")
    pend["prazo_sec"] = fim_est + pd.to_timedelta(_prazo_sec(pend["vm_mm"], anual), unit="D")
    pend["data_estimada"] = pend["receb_ant"] + pd.Timedelta(days=364)
    m1, m2, m3 = st.columns(3)
    m1.metric("Já entregaram", f"{n_div} de {n_tot}", f"{n_div / n_tot:.0%}" if n_tot else None, delta_color="off")
    m2.metric("Pendentes", f"{len(pend)}")
    m3.metric("Passaram do prazo SEC estimado", f"{int((pend['prazo_sec'] < hoje).sum())}")
    with st.expander(f"Já entregaram ({n_div})", expanded=False):
        dv = t[divulgou].sort_values("dt_arquivamento", ascending=False)
        tabela_clicavel(dv[["ticker", "empresa", "setor", "documento", "dt_fim", "dt_arquivamento"]].assign(
            dt_fim=lambda x: x["dt_fim"].dt.date, dt_arquivamento=lambda x: x["dt_arquivamento"].dt.date),
            dv["rotulo"], "clk_cal_eua_sec_div", hide_index=True, width="stretch",
            height=min(600, 38 + 35 * len(dv)), column_config={
                "ticker": "Ticker", "empresa": "Empresa", "setor": "Setor", "documento": "Documento",
                "dt_fim": "Fim do período", "dt_arquivamento": "Entregue à SEC em"})
    with st.expander(f"Ainda não entregaram ({len(pend)})", expanded=True):
        pend["situacao"] = np.select(
            [pend["prazo_sec"] < hoje, pend["data_estimada"].isna(), pend["data_estimada"] < hoje],
            ["Passou do prazo SEC estimado", "Sem histórico", "Estimativa já passou"], "No prazo")
        pend = pend.sort_values(["data_estimada", "prazo_sec"], na_position="last")
        tabela_clicavel(pend[["ticker", "empresa", "setor", "data_estimada", "receb_ant", "prazo_sec", "situacao"]].assign(
            data_estimada=lambda x: x["data_estimada"].dt.date, receb_ant=lambda x: x["receb_ant"].dt.date,
            prazo_sec=lambda x: x["prazo_sec"].dt.date),
            pend["rotulo"], "clk_cal_eua_sec_pend", hide_index=True, width="stretch",
            height=min(600, 38 + 35 * len(pend)), column_config={
                "ticker": "Ticker", "empresa": "Empresa", "setor": "Setor",
                "data_estimada": "Data estimada (pela entrega do ano anterior)",
                "receb_ant": f"Entregou {(ant_ref.month - 1) // 3 + 1}T{str(ant_ref.year)[2:]} em",
                "prazo_sec": "Prazo SEC estimado", "situacao": "Situação"})
    st.caption("Trimestre civil: cada empresa tem seu calendário fiscal; o fechamento é alinhado ao fim de trimestre "
               "civil mais próximo (mesmo critério das outras abas). **Data estimada** = entrega do mesmo trimestre do "
               "ano anterior + 364 dias (cálculo do app, não é data anunciada). **Prazo SEC estimado** (cálculo do app): "
               "10-Q em 40 dias (accelerated e large accelerated filers) ou 45 dias; 10-K em 60, 75 ou 90 dias. O porte "
               "oficial é pelo free float; aqui usa-se o valor de mercado como aproximação (≥ US$ 700 mi = large "
               "accelerated; US$ 75–700 mi = accelerated; sem dado = prazo máximo). Pedidos de prorrogação "
               "(Form 12b-25, +5 dias no 10-Q e +15 no 10-K) não são considerados.")
