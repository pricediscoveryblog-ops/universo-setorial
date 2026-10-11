"""
Price Discovery — Universo setorial (app Streamlit)
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
    "ebitda": ("EBITDA (IFRS 16)", "mm", "Resultado"),
    "ebitda_pre_ifrs16": ("EBITDA pré-IFRS 16", "mm", "Resultado"),
    "lucro_liquido": ("Lucro líquido", "mm", "Resultado"),
    "fco": ("Fluxo de caixa operacional", "mm", "Caixa e investimento"),
    "capex_organico": ("Capex orgânico", "mm", "Caixa e investimento"),
    "capex_organico_liquido": ("Capex orgânico líq. de vendas", "mm", "Caixa e investimento"),
    "vendas_imob_intang": ("Vendas de imobilizado/intangível", "mm", "Caixa e investimento"),
    "aquisicoes_participacoes": ("Aquisições de participações", "mm", "Caixa e investimento"),
    "capex_total": ("Capex total c/ aquisições", "mm", "Caixa e investimento"),
    "pagamento_arrendamento": ("Pagamentos de arrendamento", "mm", "Caixa e investimento"),
    "fcl_pos_arrend": ("FCL pós-arrendamento", "mm", "Caixa e investimento"),
    "dividendos_pagos": ("Dividendos + JCP pagos", "mm", "Caixa e investimento"),
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
    "capex_org_receita": ("Capex orgânico / receita", "pct", "Margens e retorno"),
    "capex_org_da": ("Capex orgânico / D&A", "x", "Alavancagem e múltiplos"),
    "div_liq_ebitda_ifrs16": ("DL c/ arrend. / EBITDA IFRS 16", "x", "Alavancagem e múltiplos"),
    "div_liq_ebitda_pre_ifrs16": ("DL s/ arrend. / EBITDA pré-IFRS 16", "x", "Alavancagem e múltiplos"),
}
ROT = {k: v[0] for k, v in METRICAS.items()}
# razões agregadas do setor = soma do numerador / soma do denominador (só empresas com os dois)
AGREG = {"margem_bruta": ("lucro_bruto", "receita"), "margem_ebitda": ("ebitda", "receita"),
         "margem_ebitda_pre_ifrs16": ("ebitda_pre_ifrs16", "receita"), "margem_liquida": ("lucro_liquido", "receita"),
         "roe_pl_final": ("lucro_liquido", "patrimonio_liquido"), "capex_org_receita": ("capex_organico", "receita"),
         "capex_org_da": ("capex_organico", "da"), "div_liq_ebitda_ifrs16": ("div_liq_com_arrend", "ebitda"),
         "div_liq_ebitda_pre_ifrs16": ("div_liq_sem_arrend", "ebitda_pre_ifrs16")}
PADRAO = ["receita", "ebitda", "margem_ebitda", "lucro_liquido", "div_liq_ebitda_ifrs16", "capex_organico"]
TIPOS = {"Anual": "anual", "LTM (últimos 12 meses)": "ltm", "Trimestral": "trimestral"}

# Bancos e seguradoras: (rótulo, formato, agregação do setor)
# agregação: "soma" | (numerador, denominador) para razão das somas | None (só mediana)
FIN = {
    "lucro_controladores": ("Lucro líquido (controladores)", "mm", "soma"),
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
    "depositos": ("Depósitos", "mm", "soma"),
    "custo_credito": ("Custo do crédito (PDD / carteira)", "pct", ("despesa_pdd", "carteira_credito")),
    "credito_depositos": ("Crédito / depósitos", "x", ("carteira_credito", "depositos")),
    "indice_eficiencia": ("Índice de eficiência (aprox.)", "pct", None),
}
IDENT = ["empresa", "ticker", "setor", "subsetor", "segmento", "cnpj", "tipo", "periodo", "exercicio",
         "trimestre", "dt_fim", "fonte", "regime_contabil"]

st.set_page_config(page_title="Price Discovery · Universo setorial", page_icon="📊", layout="wide")


# ------------------------------------------------------------------ dados
@st.cache_data(show_spinner="Carregando a base…")
def carregar():
    base = pd.read_csv(DADOS / "base.csv.gz", low_memory=False)
    base["dt_fim"] = pd.to_datetime(base["dt_fim"], errors="coerce")
    base["cnpj"] = base["cnpj"].astype(str).str.zfill(14)
    for c in ("ticker", "setor", "subsetor", "segmento", "empresa", "notas"):
        base[c] = base[c].fillna("")
    base["rotulo"] = base["ticker"] + " — " + base["empresa"]
    base["alerta"] = base["notas"].str.contains("implausível|ausente|alerta", regex=True)
    cob = pd.read_csv(DADOS / "cobertura.csv") if (DADOS / "cobertura.csv").exists() else pd.DataFrame()
    meta = pd.read_json(DADOS / "metadados.json", typ="series") if (DADOS / "metadados.json").exists() else None
    fin = pd.read_csv(DADOS / "financeiras.csv.gz", dtype={"cnpj": str}) if (DADOS / "financeiras.csv.gz").exists() else pd.DataFrame()
    if not fin.empty:
        for c in ("ticker", "empresa", "segmento"):
            fin[c] = fin[c].fillna("")
        fin["rotulo"] = fin["ticker"] + " — " + fin["empresa"]
    mult = pd.read_csv(DADOS / "multiplos.csv.gz", dtype={"cnpj": str}) if (DADOS / "multiplos.csv.gz").exists() else pd.DataFrame()
    if not mult.empty:
        mult["cnpj"] = mult["cnpj"].str.zfill(14)
        mult["data"] = pd.to_datetime(mult["data"], errors="coerce")
        mult["notas"] = mult["notas"].fillna("")
        # ROE no mesmo critério dos múltiplos: lucro 12m ÷ PL dos controladores do mesmo documento
        pl_ = mult["pl_controladores"].where(mult["pl_controladores"] > 0)
        mult["roe"] = mult["lucro_controladores_12m"] / pl_
    return base, cob, meta, fin, mult


def fmt_tabela(df: pd.DataFrame, cols: list[str]) -> dict:
    cfg = {}
    for c in cols:
        rot, f, _ = METRICAS[c]
        if f == "mm":
            cfg[c] = st.column_config.NumberColumn(f"{rot} (R$ mm)", format="%,.1f")
        elif f == "pct":
            cfg[c] = st.column_config.NumberColumn(rot, format="%.1f%%")
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


try:
    base, cob, meta, fin, mult = carregar()
except FileNotFoundError:
    st.error(f"Base não encontrada em {DADOS}. Rode o workflow no modo universo.")
    st.stop()

# ------------------------------------------------------------------ barra lateral
st.sidebar.title("📊 Universo setorial")
if meta is not None:
    st.sidebar.caption(f"Base gerada em {meta.get('gerado_em', '?')} · {meta.get('empresas_com_dados', '?')} empresas · "
                       "Fonte: CVM (DFP/ITR) + classificação B3")

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
if tipo == "anual":
    anos = sorted(setor_f["exercicio"].dropna().astype(int).unique())
    if anos:
        a0, a1 = st.sidebar.select_slider("Exercícios", options=anos, value=(anos[0], anos[-1]))
        setor_f = setor_f[setor_f["exercicio"].between(a0, a1)]
else:
    datas = sorted(setor_f["dt_fim"].dropna().unique())
    if datas:
        rot_d = [pd.Timestamp(d).strftime("%m/%Y") for d in datas]
        i0, i1 = st.sidebar.select_slider("Período (data de fechamento)", options=list(range(len(datas))),
                                          value=(max(0, len(datas) - 12), len(datas) - 1),
                                          format_func=lambda i: rot_d[i])
        setor_f = setor_f[setor_f["dt_fim"].between(datas[i0], datas[i1])]

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

# ------------------------------------------------------------------ abas
aba_setor, aba_emp, aba_rank, aba_fin, aba_dados, aba_cob, aba_met = st.tabs(
    ["Visão do setor", "Empresa", "Ranking", "Financeiro (bancos e seguradoras)", "Dados e download", "Cobertura",
     "Metodologia"])

# -------- Visão do setor: um indicador, série histórica completa, participação de cada empresa
def col_periodo(df: pd.DataFrame) -> pd.Series:
    return df["exercicio"].astype("Int64").astype(str) if tipo == "anual" else df["periodo"]


def ordem_periodos(df: pd.DataFrame) -> list:
    o = df.assign(_c=col_periodo(df)).groupby("_c")["dt_fim"].max().sort_values()
    return list(o.index)


def agregado_setor(df: pd.DataFrame, m: str) -> pd.Series:
    """Total do setor (métricas em R$) ou razão agregada (soma num / soma den) por período."""
    d = df.assign(_c=col_periodo(df))
    if METRICAS[m][1] == "mm":
        return d.groupby("_c")[m].sum(min_count=1)
    if m in AGREG:
        if tipo == "trimestral" and m in ("roe_pl_final", "div_liq_ebitda_ifrs16", "div_liq_ebitda_pre_ifrs16"):
            return pd.Series(dtype=float)
        n, dn = AGREG[m]
        ok = d.dropna(subset=[n, dn])
        g = ok.groupby("_c")[[n, dn]].sum()
        return (g[n] / g[dn].where(g[dn] != 0)).astype(float)
    return pd.Series(dtype=float)


with aba_setor:
    if setor_f.empty:
        st.info("Nenhum dado para o filtro.")
    else:
        c1, c2 = st.columns([2, 1])
        ind = c1.selectbox("Indicador", list(METRICAS), index=list(METRICAS).index("receita"),
                           format_func=lambda k: ROT[k] + {"mm": " (R$ mm)", "pct": " (%)", "x": " (x)"}[METRICAS[k][1]])
        fmt = METRICAS[ind][1]
        modo = c2.radio("Mostrar", ["Valores", "% do setor"], horizontal=True, disabled=(fmt != "mm"),
                        help="Participação = valor da empresa ÷ total do setor filtrado (setor/subsetor/segmento), "
                             "no mesmo período. Disponível para indicadores em R$.")
        pct_setor = (modo == "% do setor") and fmt == "mm"
        periodos = ordem_periodos(setor_f)
        d_set = setor_f.assign(_c=col_periodo(setor_f))
        total = agregado_setor(setor_f, ind).reindex(periodos)
        mediana = d_set.groupby("_c")[ind].median().reindex(periodos)
        n_emp = d_set.dropna(subset=[ind]).groupby("_c")["cnpj"].nunique().reindex(periodos).fillna(0).astype(int)

        d_emp = filtrado.assign(_c=col_periodo(filtrado))
        tab = d_emp.pivot_table(index="rotulo", columns="_c", values=ind, aggfunc="first").reindex(columns=periodos)
        ult = [p for p in periodos[::-1] if p in tab.columns and tab[p].notna().any()]
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
        else:
            cfg = {p: st.column_config.NumberColumn(p, format="%,.1f") for p in periodos}
            vis = tab_full
        st.dataframe(vis, column_config=cfg, width="stretch", height=min(650, 38 + 35 * len(vis)))
        st.caption("Empresas com dado por período: " + " · ".join(f"{p}: {n}" for p, n in n_emp.items()) +
                   ". Mudanças no total também refletem entrada de empresas (IPOs) na base.")
        if fmt == "mm" and (setor_f[ind] < 0).any():
            st.caption("⚠️ Há valores negativos neste indicador; a participação percentual pode não fazer sentido.")
        if fmt != "mm":
            st.caption("Setor agregado = soma do numerador ÷ soma do denominador das empresas com os dois dados "
                       "(ex.: margem EBITDA = Σ EBITDA ÷ Σ receita).")

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
                         labels={"v": "% do setor" if pct_setor else ROT[ind] + " (R$ mm)"})
            fig.update_layout(barmode="relative", height=480, legend_title_text="")
            if pct_setor:
                fig.update_yaxes(tickformat=".0%")
            st.plotly_chart(fig, width="stretch")
            if not sel_emp and len(tab) > 10:
                st.caption("Gráfico mostra as 10 maiores do último período e agrupa o restante em 'Demais'. "
                           "Selecione empresas na barra lateral para vê-las isoladamente.")
        else:
            fig = go.Figure()
            for r in tab.index[:10]:
                fig.add_trace(go.Scatter(x=periodos, y=tab.loc[r], mode="lines+markers", name=r.split(" — ")[0]))
            fig.add_trace(go.Scatter(x=periodos, y=total, mode="lines", name="Setor agregado", line=dict(width=4, dash="dash")))
            fig.add_trace(go.Scatter(x=periodos, y=mediana, mode="lines", name="Mediana", line=dict(dash="dot")))
            fig.update_layout(height=480, legend_title_text="")
            if fmt == "pct":
                fig.update_yaxes(tickformat=".0%")
            st.plotly_chart(fig, width="stretch")

        exp_t = tab_full.reset_index()
        c1, c2 = st.columns(2)
        c1.download_button("⬇️ Tabela (CSV)", exp_t.to_csv(index=False, sep=";", decimal=",").encode("utf-8-sig"),
                           file_name=f"setor_{ind}_{tipo}.csv", mime="text/csv")
        c2.download_button("⬇️ Tabela (XLSX)", excel_bytes({ROT[ind][:31]: exp_t}), file_name=f"setor_{ind}_{tipo}.xlsx",
                           mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet")

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
    if filtrado.empty:
        st.info("Nenhum dado para o filtro.")
    else:
        emp = st.selectbox("Empresa", sorted(filtrado["rotulo"].unique()))
        d_emp = filtrado[filtrado["rotulo"] == emp].sort_values(eixo_x(tipo))
        seg = d_emp["segmento"].iloc[0]
        st.caption(f"{d_emp['setor'].iloc[0]} / {d_emp['subsetor'].iloc[0]} / **{seg}** · base {d_emp['fonte'].iloc[-1]}")
        pares = base[(base["tipo"] == tipo) & (base["segmento"] == seg)]
        med_seg = pares.groupby(eixo_x(tipo))[sel_met].median(numeric_only=True).reset_index()
        cols = st.columns(2)
        for i, m in enumerate(sel_met):
            fig = go.Figure()
            fig.add_trace(go.Scatter(x=d_emp[eixo_x(tipo)], y=d_emp[m], mode="lines+markers", name=emp.split(" — ")[0]))
            mm = med_seg[med_seg[eixo_x(tipo)].isin(d_emp[eixo_x(tipo)])]
            fig.add_trace(go.Scatter(x=mm[eixo_x(tipo)], y=mm[m], mode="lines", name=f"Mediana {seg}",
                                     line=dict(dash="dash")))
            alertas = d_emp[d_emp["alerta"]]
            if len(alertas):
                fig.add_trace(go.Scatter(x=alertas[eixo_x(tipo)], y=alertas[m], mode="markers", name="Alerta",
                                         marker=dict(symbol="x", size=11, color="red")))
            unidade = {"mm": " (R$ mm)", "pct": "", "x": " (x)"}[METRICAS[m][1]]
            fig.update_layout(title=ROT[m] + unidade, height=320, margin=dict(t=40, b=20, l=10, r=10),
                              legend=dict(orientation="h", y=-0.2))
            if METRICAS[m][1] == "pct":
                fig.update_yaxes(tickformat=".0%")
            cols[i % 2].plotly_chart(fig, width="stretch")
        notas = d_emp[d_emp["notas"] != ""][["periodo", "fonte", "notas"]]
        with st.expander(f"Notas e alertas ({len(notas)} períodos)"):
            st.dataframe(notas, hide_index=True, width="stretch")

        # ---- múltiplos de mercado (valor de mercado com preços da B3 sem ajuste)
        st.subheader("Múltiplos de mercado")
        if mult.empty:
            st.info("Múltiplos ainda não gerados (multiplos.csv.gz). Rode o workflow no modo universo.")
        else:
            tipo_m = "anual" if tipo == "anual" else "trimestral"
            cnpj_e = d_emp["cnpj"].iloc[0]
            ini = d_emp["dt_fim"].min()
            dm = mult[(mult["cnpj"] == cnpj_e) & (mult["tipo"] == tipo_m) & (mult["data"] >= ini)].sort_values("data")
            if dm.empty:
                st.info("Sem múltiplos para esta empresa (sem cotação ou sem composição do capital).")
            else:
                st.caption("Anual: preço do último pregão do exercício com os números do exercício. "
                           if tipo_m == "anual" else
                           "Trimestral (ponto no tempo): em cada data, PL, dívida e ações do último ITR/DFP divulgado até a "
                           "data; lucro e EBITDA dos 12 meses desse documento. O último ponto é o pregão mais recente. ")
                pares_m = mult[(mult["segmento"] == seg) & (mult["tipo"] == tipo_m)]
                med_m = pares_m.groupby("data")[["p_vpa", "p_l", "ev_ebitda_ifrs16", "ev_ebitda_pre_ifrs16", "roe",
                                                    "dividend_yield"]].median()
                med_m = med_m[med_m.index.isin(dm["data"])]
                graficos = [("P/VPA (valor de mercado ÷ PL dos controladores)", [("p_vpa", "P/VPA")]),
                            ("P/L (valor de mercado ÷ lucro 12m dos controladores)", [("p_l", "P/L")]),
                            ("EV/EBITDA", [("ev_ebitda_ifrs16", "c/ arrendamento ÷ EBITDA IFRS 16"),
                                           ("ev_ebitda_pre_ifrs16", "s/ arrendamento ÷ EBITDA pré-IFRS 16")]),
                            ("ROE (lucro 12m ÷ PL dos controladores)", [("roe", "ROE")]),
                            ("Dividend yield (dividendos + JCP pagos em 12m ÷ valor de mercado)",
                             [("dividend_yield", "Dividend yield")])]
                cm = [c for _ in range((len(graficos) + 1) // 2) for c in st.columns(2)]
                for (tit, series), col in zip(graficos, cm):
                    fig = go.Figure()
                    for k, nome in series:
                        fig.add_trace(go.Scatter(x=dm["data"], y=dm[k], mode="lines+markers", name=nome))
                        fig.add_trace(go.Scatter(x=med_m.index, y=med_m[k], mode="lines", line=dict(dash="dash"),
                                                 name=f"Mediana {seg}" + ("" if len(series) == 1 else f" ({nome.split(' ')[0]})")))
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
                           "Dividend yield em regime de caixa: dividendos e JCP efetivamente pagos nos 12 meses (DFC), "
                           "não pela data-com; exclui pagamentos a não controladores.")
                with st.expander("Valor de mercado, EV, preços, ações e notas"):
                    cols_m = ["data", "periodo_base", "valor_mercado", "ev_com_arrend", "ev_sem_arrend", "ticker_on", "preco_on",
                              "data_preco_on", "acoes_on", "ticker_pn", "preco_pn", "data_preco_pn", "acoes_pn",
                              "p_vpa", "p_l", "ev_ebitda_ifrs16", "ev_ebitda_pre_ifrs16", "roe",
                              "dividendos_12m", "dividend_yield", "notas"]
                    tab_m = dm[cols_m].assign(data=dm["data"].dt.date)
                    st.dataframe(tab_m, hide_index=True, width="stretch", column_config={
                        "valor_mercado": st.column_config.NumberColumn("Valor de mercado (R$ mm)", format="%,.1f"),
                        "ev_com_arrend": st.column_config.NumberColumn("EV c/ arrend. (R$ mm)", format="%,.1f"),
                        "ev_sem_arrend": st.column_config.NumberColumn("EV s/ arrend. (R$ mm)", format="%,.1f"),
                        "acoes_on": st.column_config.NumberColumn("Ações ON", format="%,.0f"),
                        "acoes_pn": st.column_config.NumberColumn("Ações PN", format="%,.0f"),
                        **{k: st.column_config.NumberColumn(k, format="%.2fx") for k in
                           ("p_vpa", "p_l", "ev_ebitda_ifrs16", "ev_ebitda_pre_ifrs16")},
                        "roe": st.column_config.NumberColumn("ROE", format="percent"),
                        "dividendos_12m": st.column_config.NumberColumn("Dividendos + JCP 12m (R$ mm)", format="%,.1f"),
                        "dividend_yield": st.column_config.NumberColumn("Dividend yield", format="percent")})
                    st.download_button("⬇️ Múltiplos (CSV)", tab_m.to_csv(index=False, sep=";", decimal=",").encode("utf-8-sig"),
                                       file_name=f"multiplos_{emp.split(' — ')[0]}_{tipo_m}.csv", mime="text/csv", key="mult_csv")

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
            per = c2.selectbox("Período", sorted(filtrado["periodo"].unique(),
                               key=lambda p: filtrado.loc[filtrado["periodo"] == p, "dt_fim"].max(), reverse=True), key="rk_per")
            d = filtrado[filtrado["periodo"] == per]
        n = c3.slider("Quantas empresas", 5, 50, 20)
        ordem = st.radio("Ordem", ["Maiores", "Menores"], horizontal=True)
        d = d.dropna(subset=[met]).sort_values(met, ascending=(ordem == "Menores")).head(n)
        fig = px.bar(d, x=met, y="rotulo", orientation="h", color="segmento", labels={met: ROT[met], "rotulo": ""})
        fig.update_layout(height=max(350, 28 * len(d)), yaxis=dict(autorange="reversed"), legend_title_text="Segmento")
        if METRICAS[met][1] == "pct":
            fig.update_xaxes(tickformat=".0%")
        st.plotly_chart(fig, width="stretch")

# -------- Financeiro: bancos e seguradoras (plano de contas próprio, base separada)
with aba_fin:
    if fin.empty:
        st.info("Base de bancos e seguradoras ainda não gerada (financeiras.csv.gz). Rode o workflow no modo universo.")
    else:
        st.caption("Base separada: bancos e seguradoras usam plano de contas próprio e não entram nas medianas das "
                   "demais abas. Fonte: CVM (DFP anual). Métricas são definições próprias; conferir com os releases.")
        c1, c2, c3 = st.columns([1.2, 1.2, 1])
        grupos_f = {"Bancos e financeiras": ~fin["segmento"].str.contains("Segurad|Ressegurad", regex=True),
                    "Seguradoras e resseguradoras": fin["segmento"].str.contains("Segurad|Ressegurad", regex=True)}
        grp = c1.radio("Grupo", list(grupos_f), horizontal=True)
        ff = fin[grupos_f[grp]]
        opcoes = [k for k in FIN if ff[k].notna().any()] if len(ff) else list(FIN)
        ind_f = c2.selectbox("Indicador", opcoes, format_func=lambda k: FIN[k][0] + {"mm": " (R$ mm)", "pct": " (%)", "x": " (x)"}[FIN[k][1]],
                             key="fin_ind")
        rot_f, fmt_f, agg_f = FIN[ind_f]
        modo_f = c3.radio("Mostrar", ["Valores", "% do grupo"], horizontal=True, disabled=(fmt_f != "mm"), key="fin_modo")
        pct_f = modo_f == "% do grupo" and fmt_f == "mm"
        anos_f = sorted(ff["exercicio"].dropna().astype(int).unique())
        if anos_f:
            fa0, fa1 = st.select_slider("Exercícios", options=anos_f, value=(anos_f[0], anos_f[-1]), key="fin_anos")
            ff = ff[ff["exercicio"].between(fa0, fa1)]
        sel_f = st.multiselect("Empresas (vazio = todas do grupo)", sorted(ff["rotulo"].unique()), key="fin_emp")
        periodos_f = [str(a) for a in sorted(ff["exercicio"].dropna().astype(int).unique())]
        ff = ff.assign(_c=ff["exercicio"].astype(int).astype(str))
        if agg_f == "soma":
            tot_f = ff.groupby("_c")[ind_f].sum(min_count=1)
        elif isinstance(agg_f, tuple):
            n_, d_ = agg_f
            ok = ff.dropna(subset=[n_, d_])
            g_ = ok.groupby("_c")[[n_, d_]].sum()
            tot_f = g_[n_].abs() / g_[d_].where(g_[d_] != 0) if ind_f == "custo_credito" else g_[n_] / g_[d_].where(g_[d_] != 0)
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
        st.dataframe(vis_tab, column_config=cfg_f, width="stretch", height=min(650, 38 + 35 * len(vis_tab)))
        if fmt_f != "mm" and isinstance(agg_f, tuple):
            st.caption(f"Grupo agregado = Σ {agg_f[0]} ÷ Σ {agg_f[1]} das instituições com os dois dados.")
        notas_fin = {
            "indice_eficiencia": "Definição própria: |pessoal + administrativas| ÷ (margem financeira antes da PDD + "
                                 "serviços fora da intermediação). Difere do índice divulgado por cada banco; use para "
                                 "comparar tendência, não nível. Vazio quando o banco não abre a PDD na DRE.",
            "carteira_credito": "Aproximação pelas contas de operações de crédito / empréstimos a clientes do balanço; "
                                "pode diferir da carteira ampliada divulgada.",
            "despesa_pdd": "Vazio quando a PDD não aparece separada na DRE (ex.: Santander).",
            "custo_credito": "PDD do ano ÷ carteira no fim do ano (aproximação).",
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
            lf = plotf.reset_index(names="Exercício").melt(id_vars="Exercício", var_name="Empresa", value_name="v")
            fig = px.bar(lf, x="Exercício", y="v", color="Empresa", labels={"v": "% do grupo" if pct_f else rot_f + " (R$ mm)"})
            fig.update_layout(barmode="relative", height=460, legend_title_text="")
            if pct_f:
                fig.update_yaxes(tickformat=".0%")
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
        st.plotly_chart(fig, width="stretch")
        ex_f = full_f.reset_index()
        c1, c2 = st.columns(2)
        c1.download_button("⬇️ Tabela (CSV)", ex_f.to_csv(index=False, sep=";", decimal=",").encode("utf-8-sig"),
                           file_name=f"financeiro_{ind_f}.csv", mime="text/csv", key="fin_csv")
        bruto_f = ff.drop(columns=["_c", "rotulo"], errors="ignore")
        c2.download_button("⬇️ Base completa do grupo (XLSX)", excel_bytes({"tabela": ex_f, "base": bruto_f}),
                           file_name="financeiro_base.xlsx", key="fin_xlsx",
                           mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet")

# -------- Dados e download
with aba_dados:
    cols_out = [c for c in IDENT if c in filtrado.columns] + sel_met + ["notas"]
    todas = st.checkbox("Incluir todas as métricas no download (não só as selecionadas)")
    if todas:
        cols_out = [c for c in IDENT if c in filtrado.columns] + list(METRICAS) + ["notas"]
    saida = filtrado[cols_out].sort_values(["empresa", "dt_fim"])
    mets_vis = [c for c in cols_out if c in METRICAS]
    st.dataframe(para_exibir(saida, mets_vis), column_config=fmt_tabela(saida, mets_vis), hide_index=True,
                 width="stretch", height=500)
    st.caption("Valores em R$ milhões; margens e retornos em %. No download, percentuais vão como fração (0,15 = 15%).")
    rot_cols = {c: (ROT[c] + (" (R$ mm)" if METRICAS[c][1] == "mm" else "")) for c in mets_vis}
    exp = saida.assign(dt_fim=saida["dt_fim"].dt.date).rename(columns=rot_cols)
    c1, c2 = st.columns(2)
    c1.download_button("⬇️ Baixar CSV", exp.to_csv(index=False, sep=";", decimal=",").encode("utf-8-sig"),
                       file_name=f"universo_{tipo}.csv", mime="text/csv")
    abas = {"dados": exp}
    if not cob.empty:
        abas["cobertura"] = cob[cob["cnpj"].astype(str).str.zfill(14).isin(saida["cnpj"].astype(str).str.zfill(14))]
    abas["fontes"] = pd.DataFrame({"item": ["Fonte", "Gerado em", "Definições"],
                                   "descrição": ["CVM Dados Abertos (DFP/ITR) + classificação setorial B3",
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
        st.caption("Inclui empresas sem dados e os bancos e seguradoras (fora da v1), com o motivo.")

# -------- Metodologia
with aba_met:
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
- EBITDA = EBIT (DRE 3.05) + D&A da DFC. Não inclui os ajustes que as empresas divulgam.
- EBITDA pré-IFRS 16 = EBITDA − pagamentos de arrendamento (principal + juros) da DFC; antes de 2019 = EBITDA.
- Alavancagem em pares consistentes: DL c/ arrendamento ÷ EBITDA IFRS 16; DL s/ arrendamento ÷ EBITDA pré-IFRS 16.
- Capex orgânico = saídas para imobilizado (inclui propriedade para investimento), intangível e ativo de
  contrato/concessão no bloco de investimento.
- FCL pós-arrendamento = FCO − capex orgânico líquido de vendas − arrendamento pago fora do FCO.
- NCAV (Graham) = ativo circulante − passivo total.

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

**Limitações conhecidas.**
- D&A, capex e arrendamento vêm de busca textual na DFC: vazio = não identificado (nunca zero).
- Arrendamento somado à dívida numa linha só fica vazio (com nota).
- Locadoras de veículos registram a compra de frota no FCO, não no capex.
- Universo só com empresas listadas hoje: medianas históricas têm viés de sobrevivência.
- Trimestres podem divergir do anual quando há reapresentação (o anual usa a versão reapresentada).
""")
