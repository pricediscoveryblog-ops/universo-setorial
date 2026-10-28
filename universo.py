"""
universo.py — Price Discovery
Coleta TODAS as empresas com classificação setorial na B3 (exceto setor financeiro) e gera:
  - anual      : DFP; versão reapresentada (DFP seguinte) quando existir
  - trimestral : fluxos (DRE/DFC) por diferença dos acumulados (2T = 6M − 3M, 3T = 9M − 6M,
                 4T = anual − 9M); balanço = posição na data (não acumula)
  - ltm        : soma dos últimos 4 trimestres consecutivos (fluxos) + balanço do último trimestre

Saídas em saidas/universo/: base.csv.gz, cobertura.csv, auditoria.csv.gz, metadados.json,
diagnostico_dfc.csv.gz (linhas da DFC dos anos em que a busca textual não achou D&A, capex
ou pagamento de arrendamento — serve para calibrar os filtros)
"""
import gc
import json
import re
import zipfile
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd
import requests
import yaml

import coletor_setorial as cs

URL_DOC_DIR = {"DFP": f"{cs.BASE}/DOC/DFP/DADOS/", "ITR": f"{cs.BASE}/DOC/ITR/DADOS/"}
# preenchidos por ler_doc: data de divulgação (recebimento na CVM) e número de ações por documento
DISPONIB: dict[tuple, str] = {}   # (cnpj, DT_REFER) -> menor DT_RECEB
COMPOS: dict[tuple, tuple] = {}   # (cnpj, DT_REFER) -> (ações ON, ações PN) em circulação (sem tesouraria)
ENTREGAS: dict[tuple, tuple] = {}  # (cnpj, documento, DT_REFER) -> (primeira DT_RECEB, link do documento)
SAIDA = cs.SAIDAS / "universo"

FLUXOS = ["receita", "lucro_bruto", "ebit", "lucro_liquido", "lucro_controladores", "cpv", "da", "ebitda",
          "pagamento_arrendamento", "pag_arrend_juros", "pag_arrend_principal", "ebitda_pre_ifrs16",
          "fco", "capex_imobilizado", "capex_intangivel", "capex_imob_intang_combinado",
          "capex_ativo_contrato", "capex_organico", "vendas_imob_intang", "capex_organico_liquido",
          "aquisicoes_participacoes", "capex_total", "capex_total_liquido", "ativo_contrato_no_fco",
          "fcl_pos_arrend", "dividendos_pagos"]
ESTOQUES = ["ativo_total", "ativo_circulante", "caixa", "passivo_circulante", "passivo_nao_circulante",
            "patrimonio_liquido", "emprestimos_financiamentos", "passivo_arrendamento",
            "divida_bruta_ex_arrend", "pl_controladores", "nao_controladores", "clientes", "estoques",
            "fornecedores"]
# fluxos que não deveriam ser negativos num trimestre (sinal de quebra na diferença)
CHECAR_NEG = ["receita", "da", "capex_organico", "pagamento_arrendamento"]


# --------------------------------------------------------------- utilidades
def anos_disponiveis(doc: str) -> list[int]:
    padrao = rf"{doc.lower()}_cia_aberta_(\d{{4}})\.zip"
    if not cs.OFFLINE:
        try:
            html = requests.get(URL_DOC_DIR[doc], timeout=60).text
            anos = sorted({int(a) for a in re.findall(padrao, html)})
            if anos:
                return anos
        except requests.RequestException as e:
            print(f"  ! não listei o diretório {doc} ({e}); usando cache")
    return sorted({int(m.group(1)) for p in cs.CACHE.glob(f"{doc.lower()}_cia_aberta_*.zip")
                   if (m := re.search(r"(\d{4})", p.name))})


def ler_doc(doc: str, ano: int, cnpjs: set[str], forcar: bool) -> pd.DataFrame:
    nome_zip = f"{doc.lower()}_cia_aberta_{ano}.zip"
    try:
        p = cs.baixar(URL_DOC_DIR[doc] + nome_zip, cs.CACHE / nome_zip, forcar=forcar)
    except (FileNotFoundError, RuntimeError) as e:
        print(f"  ! {doc} {ano} indisponível: {e}")
        return pd.DataFrame()
    partes, com_con = [], set()
    with zipfile.ZipFile(p) as z:
        nomes = set(z.namelist())
        _ler_disponib_e_capital(z, nomes, doc, ano, cnpjs)
        # consolidado primeiro; individual só para quem não tem consolidado (economiza memória)
        for base in ("con", "ind"):
            alvo = cnpjs if base == "con" else cnpjs - com_con
            if not alvo:
                continue
            for demo in cs.DEMOS:
                nome = f"{doc.lower()}_cia_aberta_{demo}_{base}_{ano}.csv"
                if nome not in nomes:
                    continue
                df = pd.read_csv(z.open(nome), sep=";", encoding="latin-1", dtype=str,
                                 usecols=lambda c: c in cs.COLS)
                df = df[df["CNPJ_CIA"].map(cs.so_digitos).isin(alvo)]
                if df.empty:
                    continue
                if base == "con" and demo == "DRE":
                    com_con |= set(df["CNPJ_CIA"].map(cs.so_digitos))
                df["DEMO"], df["BASE"], df["ARQ_ANO"] = demo, base, ano
                partes.append(df)
    return pd.concat(partes, ignore_index=True) if partes else pd.DataFrame()


def _ler_disponib_e_capital(z, nomes, doc, ano, cnpjs):
    """Data de recebimento de cada documento (arquivo-índice) e composição do capital."""
    idx = f"{doc.lower()}_cia_aberta_{ano}.csv"
    if idx in nomes:
        try:
            ix = pd.read_csv(z.open(idx), sep=";", encoding="latin-1", dtype=str,
                             usecols=lambda c: c in ("CNPJ_CIA", "DT_REFER", "DT_RECEB", "LINK_DOC"))
            ix["c"] = ix["CNPJ_CIA"].map(cs.so_digitos)
            ix = ix[ix["c"].isin(cnpjs)]
            # calendário de resultados: primeira entrega de cada documento (a versão original, não reapresentações)
            for _, lin in ix.sort_values("DT_RECEB").drop_duplicates(["c", "DT_REFER"]).iterrows():
                k = (lin["c"], doc.upper(), str(lin["DT_REFER"])[:10])
                dr = str(lin["DT_RECEB"])[:19]
                if k not in ENTREGAS or dr < ENTREGAS[k][0]:
                    ENTREGAS[k] = (dr, lin.get("LINK_DOC", "") or "")
            for (c, r), dr in ix.groupby(["c", "DT_REFER"])["DT_RECEB"].min().items():
                k = (c, str(r)[:10])
                DISPONIB[k] = min(DISPONIB.get(k, str(dr)[:10]), str(dr)[:10])
        except Exception as e:  # noqa: BLE001
            print(f"  ! índice {idx}: {e}")
    cp = [n for n in nomes if "composicao_capital" in n]
    if not cp:
        return
    try:
        c = pd.read_csv(z.open(cp[0]), sep=";", encoding="latin-1", dtype=str)
        def achar(*partes):
            return next((col for col in c.columns if all(p in col.upper() for p in partes)), None)
        on_i, pn_i = achar("ORDIN", "INTEGR"), achar("PREF", "INTEGR")
        on_t, pn_t = achar("ORDIN", "TESOUR"), achar("PREF", "TESOUR")
        if not on_i:
            print(f"  ! composição do capital sem colunas esperadas: {list(c.columns)[:12]}")
            return
        c["c"] = c["CNPJ_CIA"].map(cs.so_digitos)
        c = c[c["c"].isin(cnpjs)]
        if "VERSAO" in c.columns:
            c["VERSAO"] = pd.to_numeric(c["VERSAO"], errors="coerce").fillna(0)
            c = c.sort_values("VERSAO").drop_duplicates(["c", "DT_REFER"], keep="last")
        num = lambda col: pd.to_numeric(c[col], errors="coerce").fillna(0) if col else pd.Series(0.0, index=c.index)
        on = num(on_i) - num(on_t)
        pn = num(pn_i) - num(pn_t)
        for cc, r, a, b in zip(c["c"], c["DT_REFER"], on, pn):
            COMPOS[(cc, str(r)[:10])] = (float(a), float(b))
    except Exception as e:  # noqa: BLE001
        print(f"  ! composição do capital {cp[0]}: {e}")


def ler_fre_capital(anos: list[int], cnpjs: set[str]) -> dict:
    """Ações ON/PN do Formulário de Referência (item 17.1, capital social), por empresa e ano de referência.
    Usado nos anos sem composição do capital no ITR/DFP (os pacotes antigos não trazem essa seção).
    Preferência: capital integralizado > subscrito > emitido; última versão entregue no ano."""
    out = {}
    for ano in anos:
        nome_zip = f"fre_cia_aberta_{ano}.zip"
        try:
            p = cs.baixar(f"{cs.BASE}/DOC/FRE/DADOS/{nome_zip}", cs.CACHE / nome_zip, forcar=False)
        except Exception as e:  # noqa: BLE001
            print(f"  ! FRE {ano}: {e}")
            continue
        try:
            with zipfile.ZipFile(p) as z:
                arq = [n for n in z.namelist() if "capital_social" in n.lower()
                       and not re.search(r"classe|titulo|aumento|desdobramento|reducao", n.lower())]
                if not arq:
                    print(f"  ! FRE {ano}: arquivo de capital social não encontrado")
                    continue
                c = pd.read_csv(z.open(arq[0]), sep=";", encoding="latin-1", dtype=str)
        except Exception as e:  # noqa: BLE001
            print(f"  ! FRE {ano}: {e}")
            continue
        col = lambda *ps: next((x for x in c.columns if all(p in cs.norm(x) for p in ps)), None)
        c_cnpj, c_ver, c_tipo = col("cnpj"), col("versao"), col("tipo", "capital")
        c_on, c_pn = col("quantidade", "ordinaria"), col("quantidade", "preferencia")
        if not (c_cnpj and c_on):
            print(f"  ! FRE {ano}: colunas inesperadas {list(c.columns)[:12]}")
            continue
        c["c"] = c[c_cnpj].map(cs.so_digitos)
        c = c[c["c"].isin(cnpjs)].copy()
        prio = {"integralizado": 0, "subscrito": 1, "emitido": 2}
        c["_p"] = c[c_tipo].map(lambda t: next((v for k, v in prio.items() if k in cs.norm(t)), 9)) if c_tipo else 0
        c["_v"] = pd.to_numeric(c[c_ver], errors="coerce").fillna(0) if c_ver else 0
        c = c[c["_p"] < 9] if c_tipo else c
        c = c.sort_values(["c", "_v", "_p"], ascending=[True, False, True]).drop_duplicates("c")
        num = lambda x: pd.to_numeric(c[x], errors="coerce").fillna(0) if x else pd.Series(0.0, index=c.index)
        for cc, on, pn in zip(c["c"], num(c_on), num(c_pn)):
            if on + pn > 0:
                out[(cc, ano)] = (float(on), float(pn))
        print(f"  FRE {ano}: capital social de {len(c)} empresas")
    return out


def _fim_exercicio(dt_ini: pd.Timestamp) -> pd.Timestamp:
    return dt_ini + pd.DateOffset(years=1) - pd.Timedelta(days=1)


def _trimestre(dt_ini: pd.Timestamp, dt_fim: pd.Timestamp) -> int:
    meses = (dt_fim.year - dt_ini.year) * 12 + dt_fim.month - dt_ini.month + 1
    return int(min(4, max(1, round(meses / 3))))


# ---------------------------------------------------------- snapshots (YTD)
def snapshots(df: pd.DataFrame, doc: str, aud: list) -> list[dict]:
    """Um registro por (empresa, documento, data): métricas acumuladas no exercício até a data."""
    out = []
    if doc == "ITR":  # no ITR o 'penúltimo' é comparativo; usamos só o período corrente
        df = df[df["ORDEM"] == "ultimo"]
    for (cnpj, _dt_ref, ordem), g in df.groupby(["CNPJ_NUM", "DT_REFER", "ORDEM"], sort=False):
        base = "con" if ((g["BASE"] == "con") & (g["DEMO"] == "DRE")).any() else "ind"
        g = g[g["BASE"] == base]
        fluxo = g["DEMO"].isin(["DRE", "DFC_MI", "DFC_MD"])
        if not fluxo.any() or g.loc[fluxo, "DT_INI"].isna().all():
            continue
        dt_ini = g.loc[fluxo, "DT_INI"].min()           # início do exercício (acumulado no ano)
        g = g[~fluxo | (g["DT_INI"] == dt_ini)]          # descarta a coluna "trimestre isolado" da DRE
        g = g.drop_duplicates(["DEMO", "CD_CONTA", "DT_INI", "DT_FIM"])
        dt_fim = g.loc[g["DEMO"].isin(["DRE", "DFC_MI", "DFC_MD"]), "DT_FIM"].max()
        if pd.isna(dt_fim):
            continue
        tipo = "reapresentado" if ordem == "penultimo" else "original"
        ctx = {"cnpj": cnpj, "doc": doc, "tipo": tipo, "dt_ini": dt_ini, "dt_fim": dt_fim,
               "arq_ano": int(g["ARQ_ANO"].iloc[0])}
        a_tmp = []
        m = cs.calcular(g, a_tmp, ctx)
        exerc = _fim_exercicio(dt_ini)
        diag = []
        faltou = [k for k in ("da", "capex_organico") if pd.isna(m.get(k))]
        if pd.isna(m.get("pagamento_arrendamento")) and (m.get("passivo_arrendamento") or 0) > 0:
            faltou.append("pagamento_arrendamento")
        if "misturada com dívida" in (m.get("notas") or ""):
            faltou.append("arrendamento_misto")
        if doc == "DFP" and ordem == "ultimo" and faltou:
            dfc = g[g["DEMO"].isin(["DFC_MI", "DFC_MD"])]
            diag = [{"cnpj": cnpj, "exercicio": exerc.year, "faltou": ",".join(faltou), "demonstracao": r.DEMO,
                     "cd_conta": r.CD_CONTA, "ds_conta": r.DS_CONTA, "valor_rs_mm": r.VL}
                    for r in dfc.itertuples()]
        out.append({"cnpj": cnpj, "doc": doc, "tipo": tipo, "dt_ini": dt_ini, "dt_fim": dt_fim,
                    "exercicio": exerc.year, "trimestre": 4 if doc == "DFP" else _trimestre(dt_ini, dt_fim),
                    "base": "consolidado" if base == "con" else "individual",
                    "denominacao": g["DENOM_CIA"].iloc[0], "arq_ano": ctx["arq_ano"], "m": m, "aud": a_tmp,
                    "diag": diag})
    return out


# ------------------------------------------------------- montagem dos períodos
def _linha(s: dict, tipo: str, m: dict, periodo: str, fonte: str, notas: list) -> dict:
    extra = [m.get("notas", "")] + notas
    return {"cnpj": s["cnpj"], "tipo": tipo, "exercicio": s["exercicio"], "trimestre": s["trimestre"],
            "periodo": periodo, "dt_ini": s["dt_ini"].date(), "dt_fim": s["dt_fim"].date(),
            "fonte": fonte, "base": s["base"], "regime_contabil": m.get("regime_contabil"),
            **{k: m.get(k) for k in FLUXOS + ESTOQUES + DERIVADAS},
            "notas": "; ".join(x for x in extra if x)}


DERIVADAS = ["div_liq_com_arrend", "div_liq_sem_arrend", "ncav", "margem_bruta", "margem_ebitda",
             "margem_ebitda_pre_ifrs16", "margem_liquida", "capex_org_receita", "capex_org_da",
             "div_liq_ebitda_ifrs16", "div_liq_ebitda_pre_ifrs16", "roe_pl_final"]


def montar_empresa(snaps: list[dict], n_anos: int) -> tuple[list, list, list]:
    """Retorna (linhas, auditoria, info) para uma empresa."""
    linhas, aud = [], []
    dfp = [s for s in snaps if s["doc"] == "DFP"]
    itr = [s for s in snaps if s["doc"] == "ITR"]

    # ---- anual: reapresentado (penúltimo da DFP seguinte) > original
    anual = {}
    for s in dfp:
        atual = anual.get(s["exercicio"])
        if atual is None or (s["tipo"] == "reapresentado" and atual["tipo"] == "original") or \
                (s["tipo"] == atual["tipo"] and s["arq_ano"] > atual["arq_ano"]):
            anual[s["exercicio"]] = s
    anos = sorted(anual)[-n_anos:]
    for ex in anos:
        s = anual[ex]
        m = cs.derivadas(dict(s["m"]), anual=True)
        linhas.append(_linha(s, "anual", m, str(ex), f"DFP {s['arq_ano']} ({s['tipo']})", []))
        aud += [{**a, "periodo": str(ex), "tipo_linha": "anual"} for a in s["aud"]]
    ano_min = anos[0] if anos else 9999

    # ---- trimestral: acumulados do ano (ITR 1-3 + DFP original como 4)
    ytd = {}
    for s in itr + [s for s in dfp if s["tipo"] == "original"]:
        if s["exercicio"] < ano_min:
            continue
        chave = (s["exercicio"], s["trimestre"])
        if chave not in ytd or s["arq_ano"] >= ytd[chave]["arq_ano"]:
            ytd[chave] = s
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
        neg = [f for f in CHECAR_NEG if pd.notna(m.get(f)) and m[f] < -1e-6]
        if neg:
            notas.append("trimestre implausível (negativo): " + ", ".join(neg) +
                         " — provável mudança de conta entre acumulados")
        m = cs.derivadas(m, anual=False)
        periodo = f"{k}T{str(ex)[2:]}"
        fonte = f"{s['doc']} {s['arq_ano']}" + ("" if k == 1 else " (dif. de acumulados)")
        row = _linha(s, "trimestral", m, periodo, fonte, notas)
        linhas.append(row)
        trimestres.append((s, m, row))
        if s["doc"] == "ITR":
            aud += [{**a, "periodo": f"acum. {periodo}", "tipo_linha": "trimestral (acumulado)"} for a in s["aud"]]

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
        m = cs.derivadas(m, anual=True)
        problemas = [t[2]["periodo"] for t in janela if "implausível" in t[2]["notas"] or "ausente" in t[2]["notas"]]
        notas = [f"LTM inclui trimestres com alerta: {', '.join(problemas)}"] if problemas else []
        linhas.append(_linha(s, "ltm", m, f"LTM {s['trimestre']}T{str(s['exercicio'])[2:]}",
                             "soma 4 trimestres", notas))

    info = {"anos_anuais": len(anos), "inicio_anual": anos[0] if anos else None,
            "fim_anual": anos[-1] if anos else None, "trimestres": len(trimestres),
            "ultimo_trimestre": trimestres[-1][2]["periodo"] if trimestres else None,
            "mes_fim_exercicio": sorted({s["dt_fim"].month for s in dfp}) if dfp else []}
    return linhas, aud, info


# --------------------------------------------------------------------- rodar
def rodar(n_anos: int = 10, atualizar_b3: bool = False, limite: int = 0):
    cs.CACHE.mkdir(parents=True, exist_ok=True)
    b3 = cs.carregar_b3(atualizar_b3)
    cad = cs.carregar_cadastro()
    univ = b3[(b3["hasQuotation"] == "S") & (b3.get("na_lista_atual", "S") != "N") & (b3["cnpj"] != "")]
    univ = univ[univ["segmento"].map(cs.norm).str.len() > 0]
    univ = univ[univ["segmento"].map(cs.norm) != cs.B3_NAO_CLASSIF]
    univ = univ.drop_duplicates("cnpj")
    # fora só quem usa plano de contas de instituição financeira (bancos, seguradoras); o setor
    # "Financeiro" da B3 também tem shoppings, imóveis, holdings e B3/Cielo, que usam o plano geral
    financ = (univ["subsetor"].map(cs.norm).str.contains("intermediarios financeiros") |
              univ["segmento"].map(cs.norm).isin(["seguradoras", "resseguradoras"]))
    excluidas = univ[financ]
    univ = univ[~financ]
    if limite:
        univ = univ.head(limite)

    # CNPJ da B3 x CNPJ registrado na CVM: se não bate, tenta pela raiz (8 primeiros dígitos)
    cad_ok = cad.copy()
    cad_ok["_raiz"] = cad_ok["CNPJ_NUM"].str[:8]
    cad_ok["_ativo"] = cad_ok.get("SIT", pd.Series("", index=cad_ok.index)).map(cs.norm).str.contains("ativo")
    no_cad = set(cad_ok["CNPJ_NUM"])
    mapa, ajuste = {}, {}
    for c in pd.concat([univ["cnpj"], excluidas["cnpj"]]):
        if c in no_cad:
            mapa[c] = c
            continue
        cand = cad_ok[cad_ok["_raiz"] == c[:8]].sort_values("_ativo", ascending=False)
        if len(cand["CNPJ_NUM"].unique()) >= 1:
            mapa[c] = cand["CNPJ_NUM"].iloc[0]
            ajuste[c] = f"CNPJ da B3 ({c}) não está no cadastro CVM; usado {mapa[c]} (mesma raiz)"
        else:
            mapa[c] = c
    univ = univ.assign(cnpj_cvm=univ["cnpj"].map(mapa))
    excluidas = excluidas.assign(cnpj_cvm=excluidas["cnpj"].map(mapa))
    fin_cnpjs = set(excluidas["cnpj_cvm"])

    # sucessões societárias: CNPJ atual <- CNPJs anteriores (sucessoes.yaml)
    suc = {}
    if Path("sucessoes.yaml").exists():
        bruto_suc = yaml.safe_load(open("sucessoes.yaml", encoding="utf-8")) or {}
        suc = {cs.so_digitos(k): [cs.so_digitos(x) for x in (v or [])] for k, v in bruto_suc.items()}
    anteriores = {a for lst in suc.values() for a in lst}
    cnpjs = set(univ["cnpj_cvm"]) | anteriores | fin_cnpjs
    estrutura_fin = []  # demonstrações completas de bancos e seguradoras (para mapear o plano de contas)
    print(f"Universo: {len(univ)} empresas ({len(excluidas)} bancos/seguradoras fora da v1)")

    snaps: dict[str, list] = {c: [] for c in cnpjs}
    for doc in ("DFP", "ITR"):
        anos = anos_disponiveis(doc)
        if not anos:
            print(f"  ! nenhum arquivo {doc} disponível")
            continue
        alvo = [a for a in anos if a >= max(anos) - n_anos]
        for ano in alvo:
            print(f"  {doc} {ano}...", flush=True)
            bruto = ler_doc(doc, ano, cnpjs, forcar=(ano >= max(anos) - 1 and not cs.OFFLINE))
            if bruto.empty:
                continue
            prep = cs.preparar(bruto)
            if doc == "DFP" and fin_cnpjs:
                f = prep[prep["CNPJ_NUM"].isin(fin_cnpjs) & (prep["ORDEM"] == "ultimo")]
                for (cn, _ref), g in f.groupby(["CNPJ_NUM", "DT_REFER"]):
                    b = "con" if ((g["BASE"] == "con") & (g["DEMO"] == "DRE")).any() else "ind"
                    g = g[g["BASE"] == b].drop_duplicates(["DEMO", "CD_CONTA", "DT_FIM"])
                    estrutura_fin.append(pd.DataFrame({
                        "cnpj": cn, "exercicio": g["DT_FIM"].dt.year, "dt_fim": g["DT_FIM"].dt.date,
                        "base": "consolidado" if b == "con" else "individual", "demonstracao": g["DEMO"],
                        "cd_conta": g["CD_CONTA"], "ds_conta": g["DS_CONTA"], "valor_rs_mm": g["VL"]}))
                prep = prep[~prep["CNPJ_NUM"].isin(fin_cnpjs - set(univ["cnpj_cvm"]))]
            for s in snapshots(prep, doc, []):
                snaps[s["cnpj"]].append(s)
            del bruto
            gc.collect()

    linhas, aud, cob, diag = [], [], [], []
    ident = univ.set_index("cnpj_cvm")
    for cnpj in sorted(set(univ["cnpj_cvm"])):
        r = ident.loc[cnpj]
        if isinstance(r, pd.DataFrame):
            r = r.iloc[0]
        notas_emp = [ajuste[r["cnpj"]]] if r["cnpj"] in ajuste else []
        # emenda séries de CNPJs anteriores nos exercícios que o atual não cobre
        for ant in suc.get(cnpj, []) + suc.get(r["cnpj"], []):
            cobertos = {(s["doc"], s["exercicio"], s["trimestre"], s["tipo"]) for s in snaps.get(cnpj, [])}
            herd = [s for s in snaps.get(ant, []) if (s["doc"], s["exercicio"], s["trimestre"], s["tipo"]) not in cobertos]
            for s in herd:
                s = dict(s); s["cnpj"] = cnpj
                snaps.setdefault(cnpj, []).append(s)
            if herd:
                notas_emp.append(f"série emendada com CNPJ anterior {ant}")
        meta = {"empresa": r["tradingName"], "ticker": r["ticker_principal"], "setor": r["setor"],
                "subsetor": r["subsetor"], "segmento": r["segmento"]}
        if not snaps.get(cnpj):
            cob.append({"cnpj": cnpj, **meta, "anos_anuais": 0, "trimestres": 0,
                        "motivo": "nenhuma DFP/ITR estruturada encontrada", "observacao": "; ".join(notas_emp)})
            continue
        for s in snaps[cnpj]:
            diag += [{**d, "empresa": meta["empresa"], "ticker": meta["ticker"]} for d in s.get("diag", [])]
        l, a, info = montar_empresa(snaps[cnpj], n_anos)
        linhas += [{**meta, **x} for x in l]
        aud += [{"cnpj": cnpj, "empresa": meta["empresa"], **x} for x in a]
        cad_e = cad[cad["CNPJ_NUM"] == cnpj]
        motivo = "janela completa" if info["anos_anuais"] >= n_anos else "série mais curta"
        if info["anos_anuais"] < n_anos and "DT_REG" in cad_e and len(cad_e):
            dt_reg = pd.to_datetime(cad_e["DT_REG"], errors="coerce").min()
            if pd.notna(dt_reg) and info["inicio_anual"] and dt_reg.year >= info["inicio_anual"] - 1:
                motivo = f"registro na CVM em {dt_reg.date()}"
        obs = "" if info["mes_fim_exercicio"] in ([], [12]) else \
            f"exercício social termina no mês {info['mes_fim_exercicio']}"
        obs = "; ".join(x for x in [obs] + notas_emp if x)
        cob.append({"cnpj": cnpj, **meta, **{k: v for k, v in info.items() if k != "mes_fim_exercicio"},
                    "motivo": motivo, "observacao": obs})
    for _, r in excluidas.iterrows():
        cob.append({"cnpj": r["cnpj"], "empresa": r["tradingName"], "ticker": r["ticker_principal"],
                    "setor": r["setor"], "subsetor": r["subsetor"], "segmento": r["segmento"],
                    "anos_anuais": 0, "trimestres": 0,
                    "motivo": "banco/seguradora: plano de contas de instituição financeira (fora da v1)"})

    SAIDA.mkdir(parents=True, exist_ok=True)
    base = pd.DataFrame(linhas)
    base.to_csv(SAIDA / "base.csv.gz", index=False, compression="gzip")
    pd.DataFrame(cob).to_csv(SAIDA / "cobertura.csv", index=False)
    if aud:
        ad = pd.DataFrame(aud).drop(columns=["dt_ini", "dt_fim"], errors="ignore")
        ad.to_csv(SAIDA / "auditoria.csv.gz", index=False, compression="gzip")
    fb = None
    if estrutura_fin:
        ef = pd.concat(estrutura_fin, ignore_index=True)
        ident_f = excluidas.drop_duplicates("cnpj_cvm").set_index("cnpj_cvm")
        for c, col in (("empresa", "tradingName"), ("ticker", "ticker_principal"), ("subsetor", "subsetor"),
                       ("segmento", "segmento")):
            ef[c] = ef["cnpj"].map(ident_f[col])
        ef.to_csv(SAIDA / "estrutura_financeiras.csv.gz", index=False, compression="gzip")
        print(f"  -> estrutura_financeiras.csv.gz ({len(ef)} linhas, {ef['cnpj'].nunique()} empresas)")
        import financeiras
        fb, fa = financeiras.calcular_financeiras(ef)
        ult = fb.groupby("cnpj")["exercicio"].transform(lambda s: s.nlargest(n_anos).min())
        fb = fb[fb["exercicio"] >= ult]
        fb.to_csv(SAIDA / "financeiras.csv.gz", index=False, compression="gzip")
        fa.to_csv(SAIDA / "auditoria_financeiras.csv.gz", index=False, compression="gzip")
        print(f"  -> financeiras.csv.gz ({len(fb)} linhas), auditoria_financeiras.csv.gz")
    if diag:
        pd.DataFrame(diag).to_csv(SAIDA / "diagnostico_dfc.csv.gz", index=False, compression="gzip")
    # múltiplos de mercado (valor de mercado × PL, lucro e EBITDA), trimestrais e anuais
    try:
        import multiplos
        anos_compos = sorted({int(k[1][:4]) for k in COMPOS})
        ano_ini = int(base["exercicio"].min()) if len(base) else 0
        fre = ler_fre_capital(list(range(ano_ini, (anos_compos[0] if anos_compos else ano_ini) + 1)), cnpjs) \
            if ano_ini else {}
        mult = multiplos.calcular(base, pd.concat([univ, excluidas], ignore_index=True), DISPONIB, COMPOS, suc,
                                  n_anos, fin=fb, fre=fre)
        if len(mult):
            mult.to_csv(SAIDA / "multiplos.csv.gz", index=False, compression="gzip")
            print(f"  -> multiplos.csv.gz ({len(mult)} linhas, {mult['cnpj'].nunique()} empresas)")
        if multiplos.SERIE_VM:
            vm = pd.concat(multiplos.SERIE_VM, ignore_index=True).dropna(subset=["valor_mercado"])
            vm.to_csv(SAIDA / "valor_mercado_diario.csv.gz", index=False, compression="gzip", date_format="%Y-%m-%d")
            print(f"  -> valor_mercado_diario.csv.gz ({len(vm)} linhas, {vm['cnpj'].nunique()} empresas, "
                  f"{vm['data'].min()} a {vm['data'].max()})")
        if ENTREGAS:
            en = pd.DataFrame([(c, d, r, rb, lk) for (c, d, r), (rb, lk) in ENTREGAS.items()],
                              columns=["cnpj", "documento", "dt_refer", "dt_receb", "link"])
            en.to_csv(SAIDA / "entregas.csv.gz", index=False, compression="gzip")
            print(f"  -> entregas.csv.gz ({len(en)} documentos, {en['cnpj'].nunique()} empresas)")
        if multiplos.DIAG_ACOES:
            da = pd.DataFrame(multiplos.DIAG_ACOES)
            da.to_csv(SAIDA / "diagnostico_acoes.csv.gz", index=False, compression="gzip")
            print(f"  -> diagnostico_acoes.csv.gz ({len(da)} documentos; {int((da['escala'] == 1000).sum())} em milhares; "
                  f"{int(da['nota'].str.contains('salto|evento', na=False).sum())} saltos tratados)")
    except Exception as e:  # noqa: BLE001 — múltiplos não podem derrubar a base
        print(f"  ! múltiplos não calculados: {e}")
    meta = {"gerado_em": datetime.now().isoformat(timespec="minutes"), "empresas": len(univ),
            "empresas_com_dados": int((pd.DataFrame(cob)["anos_anuais"] > 0).sum()) if cob else 0,
            "linhas": len(base), "janela_anos": n_anos,
            "fontes": ["CVM Dados Abertos DFP/ITR", "B3 empresas listadas (classificação setorial)"]}
    (SAIDA / "metadados.json").write_text(json.dumps(meta, ensure_ascii=False, indent=2))
    print(f"  -> {SAIDA}/base.csv.gz ({len(base)} linhas), cobertura.csv, auditoria.csv.gz, "
          f"diagnostico_dfc.csv.gz ({len(diag)} linhas)")
