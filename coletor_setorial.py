#!/usr/bin/env python3
"""
coletor_setorial.py — Price Discovery
Coleta métricas setoriais (DRE, BP, DFC) dos Dados Abertos da CVM (DFP anual)
para grupos de empresas definidos em grupos.yaml, nos últimos N exercícios
(padrão 10) ou no maior período disponível.

Uso:
  python coletor_setorial.py setores                       # árvore setorial B3 com contagem
  python coletor_setorial.py filtrar --segmento "refino"   # empresas B3 de um setor/subsetor/segmento
  python coletor_setorial.py descobrir "vibra|raizen"      # busca no cadastro CVM (não listadas)
  python coletor_setorial.py coletar --grupo distribuidoras_combustiveis
  python coletor_setorial.py coletar                       # todos os grupos
  python coletor_setorial.py universo                      # todas as empresas: anual, trimestral, LTM

Variáveis de ambiente:
  CVM_CACHE=cache_cvm   pasta de cache dos ZIPs
  CVM_OFFLINE=1         usa só o cache (sem rede)
  B3_CACHE_DIAS=30      validade do cache da lista B3
"""
import argparse
import base64
import json
import os
import re
import sys
import time
import unicodedata
import zipfile
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd
import requests
import yaml

BASE = "https://dados.cvm.gov.br/dados/CIA_ABERTA"
URL_CAD = f"{BASE}/CAD/DADOS/cad_cia_aberta.csv"
URL_DFP_DIR = f"{BASE}/DOC/DFP/DADOS/"
CACHE = Path(os.environ.get("CVM_CACHE", "cache_cvm"))
OFFLINE = os.environ.get("CVM_OFFLINE") == "1"
SAIDAS = Path("saidas")
B3_PROXY = "https://sistemaswebb3-listados.b3.com.br/listedCompaniesProxy/CompanyCall"
B3_CACHE = Path("cache_b3/empresas_b3.csv")
B3_CACHE_DIAS = int(os.environ.get("B3_CACHE_DIAS", "30"))
B3_HEADERS = {"User-Agent": "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 "
              "(KHTML, like Gecko) Chrome/126.0 Safari/537.36", "Accept": "application/json"}
B3_NAO_CLASSIF = "nao classificados"
DEMOS = ["DRE", "BPA", "BPP", "DFC_MI", "DFC_MD", "DVA"]  # DVA: receita de construção (7.01.03) das concessões
COLS = ["CNPJ_CIA", "DT_REFER", "VERSAO", "DENOM_CIA", "ESCALA_MOEDA",
        "ORDEM_EXERC", "DT_INI_EXERC", "DT_FIM_EXERC", "CD_CONTA", "DS_CONTA", "VL_CONTA"]
LIMIAR_REAPRES = 0.005  # 0,5% de diferença entre original e reapresentado


# ----------------------------------------------------------------- utilidades
def norm(s) -> str:
    s = "" if pd.isna(s) else str(s)
    s = unicodedata.normalize("NFKD", s).encode("ascii", "ignore").decode()
    return s.lower().strip()


def so_digitos(cnpj: str) -> str:
    d = re.sub(r"\D", "", str(cnpj))
    return d.zfill(14) if d else ""


def baixar(url: str, destino: Path, forcar: bool = False) -> Path:
    if destino.exists() and not forcar:
        return destino
    if OFFLINE:
        if destino.exists():
            return destino
        raise FileNotFoundError(f"Modo offline e arquivo ausente no cache: {destino}")
    destino.parent.mkdir(parents=True, exist_ok=True)
    for tentativa in range(3):
        try:
            r = requests.get(url, timeout=180, stream=True,
                             headers={"User-Agent": "PriceDiscovery-coletor/1.0"})
            r.raise_for_status()
            tmp = destino.with_suffix(destino.suffix + ".part")
            with open(tmp, "wb") as f:
                for bloco in r.iter_content(1 << 20):
                    f.write(bloco)
            tmp.replace(destino)
            return destino
        except requests.RequestException as e:
            print(f"  ! falha ao baixar {url} ({e}); tentativa {tentativa + 1}/3")
            time.sleep(5 * (tentativa + 1))
    raise RuntimeError(f"Não foi possível baixar {url}")


def anos_dfp_disponiveis() -> list[int]:
    """Descobre os anos publicados listando o diretório da CVM (sem anos fixos no código)."""
    if not OFFLINE:
        try:
            html = requests.get(URL_DFP_DIR, timeout=60).text
            anos = sorted({int(a) for a in re.findall(r"dfp_cia_aberta_(\d{4})\.zip", html)})
            if anos:
                return anos
        except requests.RequestException as e:
            print(f"  ! não consegui listar o diretório da CVM ({e}); usando cache")
    anos = sorted({int(m.group(1)) for p in CACHE.glob("dfp_cia_aberta_*.zip")
                   if (m := re.search(r"(\d{4})", p.name))})
    if not anos:
        raise RuntimeError("Nenhum ano de DFP disponível (rede e cache vazios).")
    return anos


def carregar_cadastro(forcar: bool = True) -> pd.DataFrame:
    p = baixar(URL_CAD, CACHE / "cad_cia_aberta.csv", forcar=forcar and not OFFLINE)
    cad = pd.read_csv(p, sep=";", encoding="latin-1", dtype=str)
    cad["CNPJ_NUM"] = cad["CNPJ_CIA"].map(so_digitos)
    cad["_busca"] = (cad["DENOM_SOCIAL"].map(norm) + " | "
                     + cad.get("DENOM_COMERC", pd.Series("", index=cad.index)).map(norm))
    return cad


# ------------------------------------------------------- lista de empresas B3
def _b3_get(metodo: str, payload: dict):
    """Chama o proxy público da página de empresas listadas da B3 (parâmetro = JSON em base64)."""
    arg = base64.b64encode(json.dumps(payload, separators=(",", ":")).encode()).decode()
    url = f"{B3_PROXY}/{metodo}/{arg}"
    for tentativa in range(4):
        try:
            r = requests.get(url, timeout=60, headers=B3_HEADERS)
            r.raise_for_status()
            d = r.json()
            if isinstance(d, str):  # algumas respostas vêm como JSON dentro de string
                try:
                    d = json.loads(d)
                except ValueError:
                    d = None
            return d if isinstance(d, (dict, list)) else None
        except (requests.RequestException, ValueError) as e:
            espera = 3 * (tentativa + 1)
            print(f"  ! B3 {metodo} falhou ({e}); nova tentativa em {espera}s")
            time.sleep(espera)
    return None


def _listar_b3() -> list[dict]:
    res, pagina = [], 1
    while True:
        d = _b3_get("GetInitialCompanies", {"language": "pt-br", "pageNumber": pagina, "pageSize": 120})
        if not d or not d.get("results"):
            break
        res.extend(d["results"])
        total = (d.get("page") or {}).get("totalPages", pagina)
        if pagina >= total:
            break
        pagina += 1
        time.sleep(0.3)
    return res


def carregar_b3(atualizar: bool = False) -> pd.DataFrame:
    """Tabela de empresas da B3 com CNPJ, tickers e classificação setorial (setor/subsetor/segmento).
    Cache em cache_b3/empresas_b3.csv; o detalhe de cada empresa só é buscado se ainda não estiver no cache."""
    cache = pd.read_csv(B3_CACHE, dtype=str, sep=";") if B3_CACHE.exists() else pd.DataFrame()
    idade = (time.time() - B3_CACHE.stat().st_mtime) / 86400 if B3_CACHE.exists() else 1e9
    if OFFLINE or (not atualizar and idade < B3_CACHE_DIAS and not cache.empty):
        if cache.empty:
            raise SystemExit("Cache da B3 vazio e modo offline/sem atualização.")
        return _preparar_b3(cache)

    print("Atualizando lista de empresas da B3...")
    lista = _listar_b3()
    if not lista:
        if cache.empty:
            raise SystemExit("Não consegui obter a lista da B3 e não há cache.")
        print("  ! lista B3 indisponível; usando cache existente")
        return _preparar_b3(cache)
    conhecidos = set(cache["codeCVM"]) if not cache.empty else set()
    novos = []
    # Só busca o detalhe (setor/subsetor/tickers) de quem tem segmento setorial na lista;
    # "Não Classificados" são fundos, BDRs e emissores só de dívida.
    classificados = [e for e in lista if norm(e.get("segment")) not in ("", B3_NAO_CLASSIF)]
    pend = [e for e in classificados if str(e.get("codeCVM")) not in conhecidos or atualizar]
    print(f"  {len(lista)} emissores na lista, {len(classificados)} com classificação setorial; "
          f"buscando detalhe de {len(pend)}")
    falhas = 0
    for i, e in enumerate(pend, 1):
        det = _b3_get("GetDetail", {"codeCVM": str(e.get("codeCVM")), "language": "pt-br"})
        if not isinstance(det, dict):
            det, falhas = {}, falhas + 1
        outros = det.get("otherCodes") or []
        novos.append({
            "codeCVM": str(e.get("codeCVM") or det.get("codeCVM") or ""),
            "issuingCompany": det.get("issuingCompany") or e.get("issuingCompany"),
            "companyName": det.get("companyName") or e.get("companyName"),
            "tradingName": det.get("tradingName") or e.get("tradingName"),
            "cnpj": so_digitos(det.get("cnpj") or e.get("cnpj") or ""),
            "industryClassification": det.get("industryClassification") or "",
            "segmento_lista": e.get("segment") or "",
            "tickers": ",".join(sorted({o.get("code", "") for o in outros if o.get("code")})),
            "market": det.get("market") or e.get("market") or "",
            "status": det.get("status") or e.get("status") or "",
            "hasQuotation": det.get("hasQuotation") or ("S" if str(e.get("status")) == "A" else ""),
            "hasEmissions": str(det.get("hasEmissions", "")),
            "atualizado_em": datetime.now().strftime("%Y-%m-%d"),
        })
        if i % 100 == 0:
            print(f"  ... {i}/{len(pend)}")
        time.sleep(0.25)
    if falhas:
        print(f"  ! {falhas} detalhes sem resposta válida: setor/subsetor vazios, segmento vem da lista")
    novos = pd.DataFrame(novos)
    if not cache.empty and not novos.empty:
        cache = cache[~cache["codeCVM"].isin(novos["codeCVM"])]
    tudo = pd.concat([cache, novos], ignore_index=True)
    ativos = {str(e.get("codeCVM")) for e in lista}
    tudo["na_lista_atual"] = tudo["codeCVM"].isin(ativos).map({True: "S", False: "N"})
    B3_CACHE.parent.mkdir(parents=True, exist_ok=True)
    tudo.to_csv(B3_CACHE, index=False, sep=";")
    return _preparar_b3(tudo)


def _preparar_b3(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy().fillna("")
    partes = df["industryClassification"].str.split(r"\s*/\s*", n=2, expand=True, regex=True)
    for i, c in enumerate(["setor", "subsetor", "segmento"]):
        df[c] = partes[i].fillna("").str.strip() if i in partes.columns else ""
    if "segmento_lista" in df.columns:  # sem detalhe: usa o segmento da lista
        df["segmento"] = df["segmento"].where(df["segmento"] != "", df["segmento_lista"])
    df["cnpj"] = df["cnpj"].map(so_digitos)
    df["ticker_principal"] = df["tickers"].map(
        lambda t: next((x for x in t.split(",") if re.fullmatch(r"[A-Z0-9]{4}(3|4|11)", x)), t.split(",")[0] if t else ""))
    df["ticker_principal"] = df["ticker_principal"].where(df["ticker_principal"] != "", df["issuingCompany"])
    return df


def filtrar_b3(b3: pd.DataFrame, setor=None, subsetor=None, segmento=None,
               so_negociadas: bool = True) -> pd.DataFrame:
    m = b3.copy()
    for col, val in (("setor", setor), ("subsetor", subsetor), ("segmento", segmento)):
        if val:
            m = m[m[col].map(norm).str.contains(norm(val), regex=True)]
    if so_negociadas:
        m = m[(m["hasQuotation"] == "S") & (m.get("na_lista_atual", "S") != "N")]
    return m[m["cnpj"] != ""]


def cmd_setores(atualizar: bool):
    b3 = carregar_b3(atualizar)
    b3 = b3[(b3["hasQuotation"] == "S") & (b3["setor"] != "")]
    arv = b3.groupby(["setor", "subsetor", "segmento"]).size().reset_index(name="empresas")
    with pd.option_context("display.max_rows", 500, "display.width", 200):
        print(arv.to_string(index=False))


def cmd_filtrar(a):
    b3 = filtrar_b3(carregar_b3(a.atualizar_b3), a.setor, a.subsetor, a.segmento, not a.todas)
    cols = ["ticker_principal", "tradingName", "cnpj", "setor", "subsetor", "segmento", "market"]
    with pd.option_context("display.max_rows", 500, "display.width", 250, "display.max_colwidth", 50):
        print(b3[cols].sort_values("tradingName").to_string(index=False) if len(b3) else "Nenhuma empresa.")


# ------------------------------------------------------------------ descobrir
def descobrir(padrao: str):
    cad = carregar_cadastro()
    m = cad[cad["_busca"].str.contains(norm(padrao), regex=True, na=False)]
    cols = [c for c in ["CNPJ_CIA", "DENOM_SOCIAL", "SIT", "CATEG_REG", "DT_REG",
                        "DT_CANCEL", "SETOR_ATIV"] if c in m.columns]
    if m.empty:
        print("Nenhuma companhia encontrada.")
        return
    with pd.option_context("display.max_rows", 200, "display.width", 220,
                           "display.max_colwidth", 60):
        print(m[cols].drop_duplicates().sort_values("DENOM_SOCIAL").to_string(index=False))


# ------------------------------------------------------- resolução de empresas
def _bate_ticker(row, lista) -> bool:
    alvo = {t.upper() for t in lista}
    tks = {t.upper() for t in str(row["tickers"]).split(",") if t}
    emissor = str(row["issuingCompany"]).upper()
    # casa pelo ticker completo ou, sem tickers no cache, pelo código do emissor (4 letras)
    return bool(alvo & tks) or any(a[:4] == emissor for a in alvo if emissor)


def resolver_empresas(cfg_grupo: dict, cad: pd.DataFrame, b3: pd.DataFrame | None) -> list[dict]:
    empresas = []
    f = cfg_grupo.get("filtro_b3")
    if f:
        sel = filtrar_b3(b3, f.get("setor"), f.get("subsetor"), f.get("segmento"),
                         not f.get("incluir_nao_negociadas", False))
        if f.get("incluir_tickers"):
            sel = sel[sel.apply(_bate_ticker, axis=1, lista=f["incluir_tickers"])]
        if f.get("excluir_tickers"):
            sel = sel[~sel.apply(_bate_ticker, axis=1, lista=f["excluir_tickers"])]
        print(f"  filtro B3 {dict((k, v) for k, v in f.items() if k in ('setor', 'subsetor', 'segmento'))}: "
              f"{len(sel)} empresas -> {', '.join(sel['ticker_principal'])}")
        for _, r in sel.sort_values("tradingName").iterrows():
            empresas.append({"nome": f"{r['tradingName']} ({r['ticker_principal']})", "cnpjs": [r["cnpj"]],
                             "ticker": r["ticker_principal"],
                             "nota": f"B3: {r['setor']} / {r['subsetor']} / {r['segmento']}"})
    for e in cfg_grupo.get("empresas") or []:
        if not isinstance(e, dict) or not e.get("nome") or not (e.get("cnpjs") or e.get("busca")):
            print(f"  ! entrada manual incompleta ignorada (precisa de nome e de busca ou cnpjs): {e}")
            continue
        cnpjs = [so_digitos(c) for c in (e.get("cnpjs") or []) if so_digitos(c)]
        if not cnpjs:
            m = cad[cad["_busca"].str.contains(norm(e["busca"]), regex=True, na=False)]
            unicos = m["CNPJ_NUM"].unique()
            if len(unicos) == 1:
                cnpjs = [unicos[0]]
                print(f"  {e['nome']}: CNPJ resolvido pela busca -> {m['CNPJ_CIA'].iloc[0]} "
                      f"({m['DENOM_SOCIAL'].iloc[0]}). Recomendo fixar no grupos.yaml.")
            elif len(unicos) == 0:
                msg = f"  {e['nome']}: nenhuma companhia para a busca '{e['busca']}'"
                if e.get("opcional"):
                    print(msg + " (opcional, ignorada)")
                    continue
                raise SystemExit(msg)
            else:
                print(f"  {e['nome']}: busca ambígua, fixe o CNPJ no grupos.yaml:")
                print(m[["CNPJ_CIA", "DENOM_SOCIAL", "SIT"]].drop_duplicates().to_string(index=False))
                if e.get("opcional"):
                    continue
                raise SystemExit(1)
        if any(set(cnpjs) & set(x["cnpjs"]) for x in empresas):
            continue  # já veio pelo filtro B3
        empresas.append({"nome": e["nome"], "cnpjs": cnpjs, "ticker": e.get("ticker", ""),
                         "nota": e.get("nota", "manual (grupos.yaml)")})
    if not empresas:
        raise SystemExit("Nenhuma empresa resolvida para o grupo.")
    return empresas


# --------------------------------------------------------------- leitura DFP
def ler_dfp(ano: int, cnpjs: set[str], forcar: bool) -> pd.DataFrame:
    url = f"{URL_DFP_DIR}dfp_cia_aberta_{ano}.zip"
    try:
        p = baixar(url, CACHE / f"dfp_cia_aberta_{ano}.zip", forcar=forcar)
    except (FileNotFoundError, RuntimeError) as e:
        print(f"  ! DFP {ano} indisponível: {e}")
        return pd.DataFrame()
    partes = []
    with zipfile.ZipFile(p) as z:
        nomes = set(z.namelist())
        for demo in DEMOS:
            for base in ("con", "ind"):
                nome = f"dfp_cia_aberta_{demo}_{base}_{ano}.csv"
                if nome not in nomes:
                    continue
                df = pd.read_csv(z.open(nome), sep=";", encoding="latin-1", dtype=str,
                                 usecols=lambda c: c in COLS)
                df = df[df["CNPJ_CIA"].map(so_digitos).isin(cnpjs)]
                if df.empty:
                    continue
                df["DEMO"], df["BASE"], df["ARQ_ANO"] = demo, base, ano
                partes.append(df)
    return pd.concat(partes, ignore_index=True) if partes else pd.DataFrame()


def preparar(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df["CNPJ_NUM"] = df["CNPJ_CIA"].map(so_digitos)
    df["VERSAO"] = pd.to_numeric(df["VERSAO"], errors="coerce").fillna(0).astype(int)
    # mantém só a última versão entregue de cada documento
    chave = ["CNPJ_NUM", "DT_REFER", "ARQ_ANO"]
    df = df[df["VERSAO"] == df.groupby(chave)["VERSAO"].transform("max")]
    escala = df["ESCALA_MOEDA"].map(norm).map({"mil": 1e3, "unidade": 1.0}).fillna(1e3)
    df["VL"] = pd.to_numeric(df["VL_CONTA"], errors="coerce") * escala / 1e6  # R$ milhões
    df["ORDEM"] = df["ORDEM_EXERC"].map(norm).map(
        lambda s: "penultimo" if s.startswith("pen") else "ultimo")
    df["DT_FIM"] = pd.to_datetime(df["DT_FIM_EXERC"], errors="coerce")
    df["DT_INI"] = (pd.to_datetime(df["DT_INI_EXERC"], errors="coerce")
                    if "DT_INI_EXERC" in df.columns else pd.NaT)  # não existe no balanço
    df["DESC"] = df["DS_CONTA"].map(norm)
    return df


# ------------------------------------------------------------ regras de busca
RX_DA = re.compile(r"deprec|amortiz|exaust")
RX_DA_EXC = re.compile(r"emprest|(?<!nao )(?<!nao-)financ|debent|custo.{0,10}transac|agio|desagio|juro|capta")
# Perda (reversão) por redução ao valor recuperável de ativos não financeiros (impairment, CPC 01)
RX_IMPAIR = re.compile(r"impairment|valor recuperavel|recuperabilidade")
RX_IMPAIR_EXC = re.compile(r"receber|clientes|credito|estoque|tributo|imposto|produtos? agricol|biologic|"
                           r"(?<!nao )(?<!nao-)financeir|atualizacao monetaria|indenizac")
RX_ARR = re.compile(r"arrendament|leasing|direito de uso")
RX_ARR_PAG = re.compile(r"pag|amortiz|liquid|quitac|contraprestac|encargo")
RX_ARR_EXC = re.compile(r"receb|receita|subarrend")
RX_ARR_MISTO = re.compile(r"emprest|financiament(?!os? por arrend)|debent|divida|captac")  # linha que mistura dívida e arrendamento
RX_TOTAL = re.compile(r"principal e juros|juros e principal|total")
# Capex e aquisições (bloco de investimento da DFC, 6.02)
RX_INV_VERBO = re.compile(r"aquisi|adic|adico|invest|compra|aplica|pagamento|aporte|integraliz|aumento de capital|desembols")
RX_INV_EXC = re.compile(r"venda|aliena|baixa|receb|recursos? (?:proveniente|obtido|recebido)|resgat|dividend|juros sobre|reembols|devoluc|reducao de capital")
RX_INV_DISP = re.compile(r"venda|aliena|receb|dividend")  # descarte de verdade (linha líquida "(aquisição) baixa" é aceita)
RX_IMOB = re.compile(r"imobilizad|propriedades? (?:para|de) investimento")  # inclui propriedade p/ investimento (shoppings, imóveis)
RX_INTANG = re.compile(r"intangive")
RX_ATIVO_CONTRATO = re.compile(r"ativos? de contrato|ativos? de concess|infraestrutura de concess|ativos? financeiros? de concess")
RX_AQUIS = re.compile(r"controlad|coligad|participac|investida|joint|empreendimentos? control|combinacao de neg|aquisicao de (?:empresa|negocio|sociedade)|subsidiari")
RX_VENDA = re.compile(r"venda|aliena|recebimento")
RX_VENDA_ATIVO = re.compile(r"imobilizad|intangive|ativos? permanente|ativos? de longo prazo|bens do ativo|propriedades? (?:para|de) investimento")
RX_AQUIS_EXC = re.compile(r"aplicac\w* financ|titulo|fundo|mutuo|emprestimo")
# IR/CSLL pagos e serviço da dívida (DFC)
RX_IR = re.compile(r"imposto de renda|contribuicao social|\bcsll\b|\birpj\b|\bir\b|(?:tributos|impostos) sobre (?:o )?lucro")
RX_IR_EXC = re.compile(r"diferid|receb|restitu|compensa|parcelament|\bpis\b|cofins|icms|\biss\b|inss|previd|faturament")
# variação de saldo ("a pagar", "a recolher") e retenção só ficam fora quando a linha não diz que houve pagamento
RX_A_PAGAR = re.compile(r"a pagar|a recolher|\bretid")
RX_PAGO = re.compile(r"\bpag(?:o|os|a|as|amentos?)\b|amortiz|liquidac|quitac")
RX_JUROS = re.compile(r"juro|encargos? (?:financeir|da divida|de emprestim|sobre (?:emprestim|financ|debent))")
RX_DIVIDA = re.compile(r"emprestim|financiament|debent|divida|notas? comercia|notas? promissori|\bbonds?\b|\bnotes\b|eurobond|\bcri\b|\bcra\b|\bccb\b|capital de giro|finame|bndes|cedulas? de credito|titulos? de divida|pre-?pagamento de export|\bppe\b|\bacc\b|\bace\b|finimp|"
                       r"\bempres\w*\.|\bfinanc\w*\.|\bdeben?\w*\.")  # abreviações ("emprés.", "financ.", "debên.")
# arrendamento dentro de linha de dívida (inclui abreviação "arrends.")
RX_ARR_SD = re.compile(r"arrendament|leasing|direito de uso|\barrends?\.")
RX_SACADO = re.compile(r"risco sac|convenio|forfait")
RX_SD_VERBO = re.compile(r"amortiz|pag|liquid|quitac|resgat|recompra|reembols|devoluc")
RX_SD_EXC = re.compile(r"capital proprio|\bjcp\b|dividend|receb|apropriad|provis|incorrid|derivativ|swap|hedge|capta|ingress|obtenc|acoes|tesouraria|custo")
# Locadoras: compra de frota/ativo para locação no FCO e valor contábil dos ativos vendidos (ajuste não-caixa)
RX_FROTA_VERBO = re.compile(r"compra|aquisi|adic|investimento|renovac")
RX_FROTA = re.compile(r"imobilizad|veiculo|frota|\bcarros?\b|ativos? (?:operaciona|para loca|de loca)|bens (?:para|de) loca|locac")
RX_FROTA_EXC = re.compile(r"titulo|aplicac|valores mobiliarios|estoque|venda|aliena|receb|ativos? de contrato|concess|direito de uso|arrend")
RX_BAIXA_FROTA = re.compile(r"desmobiliz|desativ|seminov|(?:custo|baixa|valor residual|valor contabil).{0,40}(?:veiculo|carro|frota|ativos? (?:operaciona|de loca|para loca))")
RX_SD_MISTO = re.compile(r"principal|(?:emprestim|financiament|debent|divida)\w* e (?:juro|encargo)|juros e (?:amortiz|principal)")
# FCO comparável: juros e dividendos/JCP pagos ou recebidos que a empresa classifica no operacional (o CPC 03
# permite operacional, investimento ou financiamento). Saem do FCO para que a diferença para o EBITDA seja
# basicamente capital de giro e IR/CSLL pagos.
RX_FCO_JUR = re.compile(r"juro|encargos? (?:financeir|da divida|de emprestim|sobre (?:emprestim|financ|debent))|rendimento")
RX_FCO_DIV = re.compile(r"dividend|capital proprio|\bjscp\b|\bjcp\b")
RX_FCO_PAG = re.compile(r"\bpag(?:o|os|a|as|amentos?)\b|\bpagt|liquidad|quitad|desembols")
RX_FCO_REC = re.compile(r"\breceb(?:ido|idos|ida|idas|imentos?)\b|recebiment")
RX_FCO_ACCR = re.compile(r"a pagar|a recolher|\bretid|a receber|provis|apropri|incorrid|capitaliz|acruad|nao pag|nao receb|"
                         r"propost|calculad|variac|atualizac|reversa|declarad|deliberad|descontinuad")
RX_FCO_ACCR_FORTE = re.compile(r"provis|apropri|incorrid|capitaliz|acruad|nao pag|nao receb|variac")
# Concessões (ICPC 01 / modelo do intangível): receita e custo de construção da infraestrutura
RX_CONSTR = re.compile(r"constru")
RX_CONSTR_EXC = re.compile(r"construcao civil|incorpora|imobiliari|provis|perda|ineficien|naval|embarcac|manutenc")
RX_MARGEM_CONSTR = re.compile(r"margem.{0,40}constru")
RX_MARGEM_CONSTR_EXC = re.compile(r"transmiss|remunerac|ativos? de contrato|operac")
# Setor elétrico (EBITDA ajustado; classificação por empresa em eletricas.yaml)
# custo de implantação da transmissão: subcontas do CPV/DVA ou ajuste não-caixa da DFC com construção/implementação
# de infraestrutura; linhas que juntam O&M ficam à parte (RX_IMPLANT_OM)
RX_IMPLANT = re.compile(r"constru|implementac|implantac|infraestrutura")
RX_IMPLANT_EXC = re.compile(r"pessoal|energia (?:comprada|eletrica)|provis|perda|ineficien|civil|usina|geracao|receita|"
                            r"remunerac|margem|baixa|amortiz|deprec")
RX_IMPLANT_OM = re.compile(r"operac|manutenc")
# saldo do ativo de contrato/concessão da transmissão no balanço (circulante + não circulante)
RX_AC_SALDO = re.compile(r"ativos? (?:de|do) contratos?|ativos? contratua|ativos? (?:da|de) concess|"
                         r"ativos? financeiros? (?:da|de) concess|concessao (?:do|de) servico publico \(ativo|\brbse\b|"
                         r"infraestrutura de concess")
RX_AC_SALDO_EXC = re.compile(r"exclusividade|seguro|direito de uso|bonifica|setorial|passivo|\bimpost|tribut|diferid")
RX_AC_FIN = re.compile(r"financeir")   # parte "ativo financeiro" do saldo (na geração = bonificação pela outorga)
# bonificação pela outorga (geração): saldo no balanço
RX_BONIF = re.compile(r"bonifica\w* (?:pel[ao] |da |de )?outorga|outorga.{0,20}bonifica")
# atualização do ativo financeiro indenizável (VNR) da distribuição: ajuste não-caixa da DFC (6.01.01/6.01.03)
RX_VNR = re.compile(r"ativos? financeiros? (?:indeniz|da concess|de concess|\(concess)|ativos? (?:sujeito a )?indeniz|"
                    r"sujeito a indeniza|bens reversiveis e indeniz|concessao (?:do )?servico publico \(ativo financeiro|\bvnr\b|"
                    r"atualizac\w* d[oa]s? ativos?.{0,40}financeiro|valor (?:novo )?de reposicao|"
                    r"expectativa do fluxo de caixa.{0,30}concess")
RX_VNR_VERBO = re.compile(r"atualiza|remunera|valor justo|receita|correc|ajuste|marcacao")
RX_VNR_EXC = re.compile(r"setorial|disponive|venda|receb|baixa|perda|amortiza|itaipu|aplicac|transmiss|aumento|"
                        r"reduc|diminui|variac|investiment|adic|aquisi|parcela a")
# PIS/COFINS diferidos sobre a receita do ativo de contrato (não-caixa; o Δ do ativo sai pelo valor bruto)
RX_PIS_DIF = re.compile(r"(?:pis|cofins|tributos indiretos).{0,30}diferid")
RX_EQUIV = re.compile(r"equivalencia patrimonial")
# Δ operacional do ativo de contrato pela DFC (ajustes não-caixa + variação no capital de giro): exclui a
# consolidação de SPEs compradas, que entra no Δ do balanço
RX_AC_DFC = re.compile(r"ativos? (?:de|do) contrat|ativos? contratua|ativos? (?:da|de) concess|concess.{0,25}\(ativo|"
                       r"implementac.{0,20}infra|receita de infra|receita de constru|construcao de infraestrutura")
RX_AC_DFC_EXC = re.compile(r"custo|financeiro de concess|bonifica|baixa|intangivel|aquisic|alienac|impost|tribut")
IR_CS_CONSTRUCAO = 0.34  # alíquota nominal de IR + CS usada para retirar a margem de construção do lucro líquido
# campos reportados preservados antes do ajuste de construção (o principal passa a ser o ajustado)
REPORTADOS = {"receita": "receita_reportada", "cpv": "cpv_reportado", "lucro_bruto": "lucro_bruto_reportado",
              "ebit": "ebit_reportado", "ebitda": "ebitda_reportado",
              "ebitda_pre_ifrs16": "ebitda_pre_ifrs16_reportado", "lucro_liquido": "lucro_liquido_reportado",
              "lucro_controladores": "lucro_controladores_reportado"}


def classe_jur_div_fco(desc: str, vl: float, ajuste_nao_caixa: bool):
    """'juros' ou 'dividendos' quando a linha do FCO (6.01.xx) é juros ou dividendos/JCP pagos ou recebidos; senão None.
    Nos ajustes não-caixa (6.01.01) só vale linha com verbo de caixa e sinal coerente (pago < 0, recebido > 0)."""
    e_div = bool(RX_FCO_DIV.search(desc))
    e_jur = bool(RX_FCO_JUR.search(desc)) and "capital proprio" not in desc
    if not (e_jur or e_div) or RX_IR.search(desc):
        return None
    pg, rc = bool(RX_FCO_PAG.search(desc)), bool(RX_FCO_REC.search(desc))
    if RX_FCO_ACCR.search(desc) and (not (pg or rc) or RX_FCO_ACCR_FORTE.search(desc)):
        return None
    if ajuste_nao_caixa:
        if not (pg or rc) or (pg and not rc and vl > 0) or (rc and not pg and vl < 0):
            return None
    elif not (pg or rc):  # sem verbo de caixa: juros só como saída, rendimento só como entrada
        if "rendimento" in desc and vl < 0:
            return None
        if e_jur and "rendimento" not in desc and vl > 0:
            return None
    return "dividendos" if e_div else "juros"


def topo(rows: pd.DataFrame) -> pd.DataFrame:
    """Mantém só as contas mais altas da hierarquia (evita somar pai + filhos)."""
    if rows.empty:
        return rows
    codigos = set(rows["CD_CONTA"])
    def tem_ancestral(c):
        partes = c.split(".")
        return any(".".join(partes[:i]) in codigos for i in range(1, len(partes)))
    return rows[~rows["CD_CONTA"].map(tem_ancestral)]


def conta(sub, demo, codigo):
    r = sub[(sub["DEMO"] == demo) & (sub["CD_CONTA"] == codigo)]
    return r["VL"].sum() if len(r) else np.nan


def calcular(sub: pd.DataFrame, aud: list, ctx: dict) -> dict:
    m, notas = {}, []
    dfc_demo = "DFC_MI" if (sub["DEMO"] == "DFC_MI").any() else "DFC_MD"
    dfc = sub[sub["DEMO"] == dfc_demo]

    def registrar(rows, uso):
        for _, r in rows.iterrows():
            aud.append({**ctx, "uso": uso, "demonstracao": r["DEMO"], "cd_conta": r["CD_CONTA"],
                        "ds_conta": r["DS_CONTA"], "valor_rs_mm": r["VL"]})

    # DRE
    m["receita"] = conta(sub, "DRE", "3.01")
    m["lucro_bruto"] = conta(sub, "DRE", "3.03")
    m["ebit"] = conta(sub, "DRE", "3.05")
    # resultado financeiro (DRE 3.06) e aberturas: receitas (3.06.01) e despesas (3.06.02, com sinal negativo)
    m["resultado_financeiro"] = conta(sub, "DRE", "3.06")
    m["receitas_financeiras"] = conta(sub, "DRE", "3.06.01")
    m["despesas_financeiras"] = conta(sub, "DRE", "3.06.02")
    m["lucro_liquido"] = conta(sub, "DRE", "3.11")
    # lucro atribuído aos controladores (abertura da 3.11); sem abertura, vale o lucro total
    ctrl = sub[(sub["DEMO"] == "DRE") & sub["CD_CONTA"].str.startswith("3.11.") &
               sub["DESC"].str.contains("controlador") & ~sub["DESC"].str.contains("nao controlador")]
    lc = ctrl["VL"].sum() if len(ctrl) else np.nan
    m["lucro_controladores"] = lc if pd.notna(lc) and (lc != 0 or m["lucro_liquido"] == 0) else m["lucro_liquido"]
    # Capital de giro operacional (base dos prazos médios): CPV, clientes, estoques e fornecedores (CP + LP)
    cpv = conta(sub, "DRE", "3.02")
    m["cpv"] = -cpv if pd.notna(cpv) else np.nan

    def _soma_txt(demo, prefixo, rx):
        d = sub[(sub["DEMO"] == demo) & sub["CD_CONTA"].str.match(prefixo) & sub["DESC"].str.contains(rx)]
        d = topo(d)
        return d["VL"].sum() if len(d) else np.nan

    def _total(cp, lp):
        return np.nan if pd.isna(cp) and pd.isna(lp) else np.nansum([cp, lp])
    cli_cp = conta(sub, "BPA", "1.01.03")
    if pd.isna(cli_cp):
        cli_cp = _soma_txt("BPA", r"^1\.01\.", r"contas a receber|clientes")
    m["clientes"] = _total(cli_cp, _soma_txt("BPA", r"^1\.02\.01\.", r"contas a receber|clientes"))
    est_cp = conta(sub, "BPA", "1.01.04")
    m["estoques"] = _total(est_cp, _soma_txt("BPA", r"^1\.02\.01\.", r"^estoques?"))
    forn_cp = conta(sub, "BPP", "2.01.02")
    if pd.isna(forn_cp):
        forn_cp = _soma_txt("BPP", r"^2\.01\.", r"fornecedores")
    m["fornecedores"] = _total(forn_cp, _soma_txt("BPP", r"^2\.02\.", r"fornecedores"))
    # Balanço
    m["ativo_total"] = conta(sub, "BPA", "1")
    m["ativo_circulante"] = conta(sub, "BPA", "1.01")
    m["caixa"] = np.nansum([conta(sub, "BPA", "1.01.01"), conta(sub, "BPA", "1.01.02")])
    m["passivo_circulante"] = conta(sub, "BPP", "2.01")
    m["passivo_nao_circulante"] = conta(sub, "BPP", "2.02")
    m["patrimonio_liquido"] = conta(sub, "BPP", "2.03")
    nc = sub[(sub["DEMO"] == "BPP") & sub["CD_CONTA"].str.startswith("2.03.") & sub["DESC"].str.contains("nao controlador")]
    m["nao_controladores"] = topo(nc)["VL"].sum() if len(nc) else 0.0
    m["pl_controladores"] = m["patrimonio_liquido"] - m["nao_controladores"]
    m["emprestimos_financiamentos"] = np.nansum([conta(sub, "BPP", "2.01.04"),
                                                 conta(sub, "BPP", "2.02.01")])

    # Passivo de arrendamento (estoque) — busca textual no passivo
    bpp = sub[(sub["DEMO"] == "BPP") & sub["CD_CONTA"].str.match(r"^2\.0[12]\.")]
    arr = topo(bpp[bpp["DESC"].str.contains(RX_ARR) & ~bpp["DESC"].str.contains(r"receb|ativo")])
    registrar(arr, "passivo_arrendamento")
    # IFRS 16 vale para exercícios iniciados a partir de 01/01/2019 (início do exercício social)
    dt_ini = ctx.get("dt_ini")
    if dt_ini is None or pd.isna(dt_ini):
        di = sub["DT_INI"].dropna() if "DT_INI" in sub.columns else pd.Series(dtype="datetime64[ns]")
        dt_ini = di.min() if len(di) else pd.NaT
    pos_ifrs16 = (dt_ini >= pd.Timestamp("2019-01-01")) if pd.notna(dt_ini) \
        else (ctx["dt_fim"] >= pd.Timestamp("2019-12-31"))
    if len(arr):
        m["passivo_arrendamento"] = arr["VL"].sum()
    elif pos_ifrs16:
        m["passivo_arrendamento"] = np.nan  # não identificado ≠ zero
        notas.append("passivo de arrendamento não identificado")
    else:
        m["passivo_arrendamento"] = 0.0  # IAS 17: arrendamento operacional fora do balanço
    dentro = arr[arr["CD_CONTA"].str.match(r"^2\.01\.04\.|^2\.02\.01\.")]
    m["divida_bruta_ex_arrend"] = m["emprestimos_financiamentos"] - (dentro["VL"].sum() if len(dentro) else 0.0)

    # D&A — ajustes da DFC indireta (6.01.01.*)
    if dfc_demo == "DFC_MI":
        aj = dfc[dfc["CD_CONTA"].str.match(r"^6\.01\.0[13]\.")]  # ajustes e, em algumas empresas, "outros"
        da = aj[aj["DESC"].str.contains(RX_DA) & ~aj["DESC"].str.contains(RX_DA_EXC)
                & ~aj["DESC"].str.contains(r"custo a? ?amortiz")]
        da = topo(da[da["CD_CONTA"].str.startswith("6.01.01.") | (da["VL"] > 0)])
        registrar(da, "d&a")
        m["da"] = da["VL"].sum() if len(da) else np.nan
        if da.empty:
            notas.append("D&A não identificada na DFC")
        elif da["DESC"].str.contains(RX_IMPAIR).any():
            notas.append("D&A e impairment na mesma linha da DFC (impairment incluído na D&A)")
        # Impairment em linha própria (não-caixa): somado ao EBITDA. Fica fora o que já entrou na D&A e as
        # perdas de ativos financeiros, recebíveis, estoques, tributos e ativos biológicos.
        imp = aj[aj["DESC"].str.contains(RX_IMPAIR) & ~aj["DESC"].str.contains(RX_IMPAIR_EXC)
                 & ~aj["DESC"].str.contains(RX_DA)]
        imp = topo(imp)
        registrar(imp, "impairment")
        m["impairment"] = float(imp["VL"].sum()) if len(imp) else 0.0
    else:
        m["da"] = m["impairment"] = np.nan
        notas.append("DFC pelo método direto: D&A indisponível")
    # EBITDA = EBIT + D&A + impairment (perda ou reversão ao valor recuperável de ativos não financeiros)
    m["ebitda"] = m["ebit"] + m["da"] + (m["impairment"] if pd.notna(m["impairment"]) else 0.0)

    # Pagamentos de arrendamento — DFC inteira, exceto ajustes não-caixa (6.01.01)
    pg = dfc[~dfc["CD_CONTA"].str.startswith("6.01.01")]
    # linha só com o substantivo ("Contratos de arrendamento") vale quando é saída de caixa no
    # financiamento (6.03) ou, no operacional, quando cita o passivo de arrendamento
    sem_verbo = (pg["VL"] < 0) & (pg["CD_CONTA"].str.startswith("6.03") |
                                  pg["DESC"].str.contains("passivos? de arrendament"))
    pg = pg[pg["DESC"].str.contains(RX_ARR) & (pg["DESC"].str.contains(RX_ARR_PAG) | sem_verbo)
            & ~pg["DESC"].str.contains(RX_ARR_EXC)]
    misto = pg[pg["DESC"].str.contains(RX_ARR_MISTO)]
    if len(misto):
        registrar(topo(misto), "excluido:arrendamento_misturado_com_divida")
        notas.append("linha de arrendamento misturada com dívida: excluída (pagamento pode estar subestimado)")
    pg = topo(pg[~pg["DESC"].str.contains(RX_ARR_MISTO)]).copy()
    if len(pg):
        pg["classe"] = np.where(pg["DESC"].str.contains(RX_TOTAL), "total",
                                np.where(pg["DESC"].str.contains("juro"), "juros", "principal"))
        if (pg["classe"] == "total").any() and (pg["classe"] != "total").any():
            pg = pg[pg["classe"] == "total"]
            notas.append("linha de total e componentes coexistem: usado só o total")
        registrar(pg, "pagamento_arrendamento")
        m["pagamento_arrendamento"] = -pg["VL"].sum()  # positivo = saída de caixa
        m["pag_arrend_juros"] = -pg.loc[pg["classe"] == "juros", "VL"].sum()
        m["pag_arrend_principal"] = -pg.loc[pg["classe"] == "principal", "VL"].sum()
    else:
        m["pagamento_arrendamento"] = m["pag_arrend_juros"] = m["pag_arrend_principal"] = np.nan

    # Capex e aquisições — bloco de investimento (6.02), só contas de nível mais alto
    inv = dfc[dfc["CD_CONTA"].str.startswith("6.02.")]
    so_subst = (inv["VL"] < 0) & inv["DESC"].str.contains(RX_IMOB.pattern + "|" + RX_INTANG.pattern + "|" +
                                                          RX_ATIVO_CONTRATO.pattern)
    inv = inv[(inv["DESC"].str.contains(RX_INV_VERBO) & ~inv["DESC"].str.contains(RX_INV_EXC)) |
              (so_subst & ~inv["DESC"].str.contains(RX_INV_DISP))]
    e_aquis = inv["DESC"].str.contains(RX_AQUIS) & ~inv["DESC"].str.contains(RX_AQUIS_EXC)
    e_imob = inv["DESC"].str.contains(RX_IMOB)
    e_intang = inv["DESC"].str.contains(RX_INTANG)
    e_contr = inv["DESC"].str.contains(RX_ATIVO_CONTRATO)
    inv = topo(inv[e_aquis | e_imob | e_intang | e_contr]).copy()
    if dfc.empty:
        for k in ("capex_imobilizado", "capex_intangivel", "capex_imob_intang_combinado",
                  "capex_ativo_contrato", "capex_organico", "aquisicoes_participacoes"):
            m[k] = np.nan
        notas.append("DFC ausente: capex indisponível")
    else:
        def classe(d):
            if RX_AQUIS.search(d) and not RX_AQUIS_EXC.search(d) and not (RX_IMOB.search(d) or RX_INTANG.search(d)):
                return "aquisicao"
            if RX_IMOB.search(d) and RX_INTANG.search(d):
                return "imob_intang"
            if RX_IMOB.search(d):
                return "imobilizado"
            if RX_INTANG.search(d):
                return "intangivel"
            return "ativo_contrato"
        inv["classe"] = inv["DESC"].map(classe)
        for _, r in inv.iterrows():
            aud.append({**ctx, "uso": f"capex:{r['classe']}", "demonstracao": r["DEMO"], "cd_conta": r["CD_CONTA"],
                        "ds_conta": r["DS_CONTA"], "valor_rs_mm": r["VL"]})
        soma = lambda c: -inv.loc[inv["classe"] == c, "VL"].sum()  # positivo = saída de caixa
        m["capex_imobilizado"] = soma("imobilizado")
        m["capex_intangivel"] = soma("intangivel")
        m["capex_imob_intang_combinado"] = soma("imob_intang")
        m["capex_ativo_contrato"] = soma("ativo_contrato")
        m["aquisicoes_participacoes"] = soma("aquisicao")
        organico = inv["classe"].isin(["imobilizado", "intangivel", "imob_intang", "ativo_contrato"])
        if organico.any():
            m["capex_organico"] = -inv.loc[organico, "VL"].sum()
        else:
            m["capex_organico"] = np.nan  # não identificado ≠ zero
            notas.append("capex de imobilizado/intangível não identificado")
        if (inv["classe"] == "imob_intang").any():
            notas.append("imobilizado e intangível em linha única: só o total orgânico é confiável")
        if not (inv["classe"] == "aquisicao").any():
            notas.append("nenhuma aquisição de participação na DFC (assumido 0)")
    # Vendas de imobilizado/intangível (entrada de caixa, positivo)
    inv_all = dfc[dfc["CD_CONTA"].str.startswith("6.02.")]
    vd = inv_all[inv_all["DESC"].str.contains(RX_VENDA) & inv_all["DESC"].str.contains(RX_VENDA_ATIVO)
                 & ~inv_all["DESC"].str.contains(RX_AQUIS)]
    vd = topo(vd)
    for _, r in vd.iterrows():
        aud.append({**ctx, "uso": "venda_imob_intang", "demonstracao": r["DEMO"], "cd_conta": r["CD_CONTA"],
                    "ds_conta": r["DS_CONTA"], "valor_rs_mm": r["VL"]})
    m["vendas_imob_intang"] = vd["VL"].sum() if len(vd) else (0.0 if not dfc.empty else np.nan)
    m["capex_organico_liquido"] = m["capex_organico"] - m["vendas_imob_intang"]
    m["capex_total"] = m["capex_organico"] + m["aquisicoes_participacoes"]
    m["capex_total_liquido"] = m["capex_organico_liquido"] + m["aquisicoes_participacoes"]

    # Informativo: ativo de contrato dentro do FCO (algumas concessionárias classificam
    # o investimento em ativo de contrato no operacional). NÃO entra no capex.
    fco_ac = dfc[dfc["CD_CONTA"].str.startswith("6.01.") & ~dfc["CD_CONTA"].str.startswith("6.01.01")]
    fco_ac = topo(fco_ac[fco_ac["DESC"].str.contains(RX_ATIVO_CONTRATO)])
    for _, r in fco_ac.iterrows():
        aud.append({**ctx, "uso": "info:ativo_contrato_no_fco", "demonstracao": r["DEMO"], "cd_conta": r["CD_CONTA"],
                    "ds_conta": r["DS_CONTA"], "valor_rs_mm": r["VL"]})
    m["ativo_contrato_no_fco"] = -fco_ac["VL"].sum() if len(fco_ac) else np.nan
    if len(fco_ac):
        notas.append("ativo de contrato no FCO: parte do investimento em concessão pode estar no operacional (não somado ao capex)")

    # Locadoras: compra de frota/ativo para locação classificada no FCO (fora do bloco 6.02), líquida do valor
    # contábil dos ativos vendidos (ajuste não-caixa, ex.: "custo de venda de ativos desmobilizados"), porque a
    # receita da venda já está no EBITDA. Entra só no capex dos índices de cobertura (o FCO já a desconta).
    fr = dfc[dfc["CD_CONTA"].str.match(r"^6\.01\.0[23]\.") & (dfc["VL"] < 0)]
    fr = topo(fr[fr["DESC"].str.contains(RX_FROTA_VERBO) & fr["DESC"].str.contains(RX_FROTA)
                 & ~fr["DESC"].str.contains(RX_FROTA_EXC)])
    registrar(fr, "capex_frota_fco")
    if dfc.empty:
        m["capex_frota_fco"] = m["baixa_frota_valor_contabil"] = m["capex_frota_liquido"] = np.nan
    elif len(fr):
        bx = dfc[dfc["CD_CONTA"].str.startswith("6.01.01") & (dfc["VL"] > 0)]
        bx = topo(bx[bx["DESC"].str.contains(RX_BAIXA_FROTA)])
        registrar(bx, "baixa_frota_valor_contabil")
        m["capex_frota_fco"] = float(-fr["VL"].sum())
        m["baixa_frota_valor_contabil"] = float(bx["VL"].sum()) if len(bx) else 0.0
        m["capex_frota_liquido"] = m["capex_frota_fco"] - m["baixa_frota_valor_contabil"]
        if not len(bx):
            notas.append("capex de frota/imobilizado no FCO sem valor contábil vendido identificado (usado bruto)")
    else:
        m["capex_frota_fco"] = m["baixa_frota_valor_contabil"] = m["capex_frota_liquido"] = 0.0

    # Dividendos e JCP pagos aos acionistas da controladora (saída de caixa; positivo = pago).
    # Exclui ajustes não-caixa, variação de "dividendos a pagar", investimento (dividendos recebidos)
    # e pagamentos a não controladores.
    dv = dfc[~dfc["CD_CONTA"].str.match(r"^6\.01\.0[12]|^6\.02")]
    dv = dv[dv["DESC"].str.contains(r"dividend|capital proprio|\bjcp\b") & (dv["VL"] < 0)
            & ~dv["DESC"].str.contains(r"receb|nao controlador|minoritari|a pagar")]
    dv = topo(dv)
    for _, r in dv.iterrows():
        aud.append({**ctx, "uso": "dividendos_pagos", "demonstracao": r["DEMO"], "cd_conta": r["CD_CONTA"],
                    "ds_conta": r["DS_CONTA"], "valor_rs_mm": r["VL"]})
    if dfc.empty:
        m["dividendos_pagos"] = np.nan
    else:
        m["dividendos_pagos"] = -dv["VL"].sum() if len(dv) else 0.0
        if not len(dv):
            notas.append("nenhum pagamento de dividendos/JCP identificado na DFC (assumido 0)")

    # IR e CSLL pagos (positivo = pago): linhas de pagamento na DFC, fora dos ajustes não-caixa (6.01.01)
    # e do investimento (6.02). Exclui variação de saldos "a pagar/a recolher", diferidos, retidos e
    # parcelamentos (que costumam misturar outros tributos).
    tx = dfc[~dfc["CD_CONTA"].str.match(r"^6\.01\.01|^6\.02") & (dfc["VL"] < 0)]
    tx = topo(tx[tx["DESC"].str.contains(RX_IR) & tx["DESC"].str.contains(r"pag|recolhid|liquid")
                 & ~tx["DESC"].str.contains(RX_IR_EXC)
                 & ~(tx["DESC"].str.contains(RX_A_PAGAR) & ~tx["DESC"].str.contains(RX_PAGO))])
    registrar(tx, "ir_csll_pagos")
    if dfc.empty:
        m["ir_csll_pagos"] = np.nan
    else:
        m["ir_csll_pagos"] = float(-tx["VL"].sum()) if len(tx) else 0.0
        if not len(tx):
            notas.append("IR/CSLL pagos não identificados na DFC (assumido 0)")

    # Serviço da dívida pago, ex-arrendamento (positivo = pago): juros (operacional ou financiamento)
    # e amortização de principal (financiamento). Linha que junta principal e juros entra só no total.
    # Linha que junta dívida e arrendamento ("amortização de empréstimos, arrendamentos e debêntures") vai para
    # servico_div_arrend_misto: só o índice com arrendamento no denominador é calculável.
    sdv = dfc[dfc["CD_CONTA"].str.match(r"^6\.0[13]\.") & ~dfc["CD_CONTA"].str.startswith("6.01.01")
              & (dfc["VL"] < 0)]
    e_div = sdv["DESC"].str.contains(RX_DIVIDA)
    sdv = sdv[~(sdv["DESC"].str.contains(RX_ARR_SD) & ~e_div) & ~(sdv["DESC"].str.contains(RX_SACADO) & ~e_div)
              & ~sdv["DESC"].str.contains(RX_SD_EXC)
              & ~(sdv["DESC"].str.contains(RX_A_PAGAR) & ~sdv["DESC"].str.contains(RX_PAGO))]
    no_fin = sdv["CD_CONTA"].str.startswith("6.03")
    e_jur = sdv["DESC"].str.contains(RX_JUROS) & (sdv["DESC"].str.contains(RX_SD_VERBO) | no_fin)
    e_pri = no_fin & sdv["DESC"].str.contains(RX_DIVIDA) & sdv["DESC"].str.contains(RX_SD_VERBO) \
        & ~sdv["DESC"].str.contains(RX_JUROS)
    sdv = topo(sdv[e_jur | e_pri]).copy()
    m["servico_div_arrend_misto"] = np.nan
    if dfc.empty:
        m["juros_pagos"] = m["principal_pago"] = m["servico_divida_pago"] = np.nan
    else:
        sdv["classe"] = np.where(~sdv["DESC"].str.contains(RX_JUROS), "principal",
                                 np.where(sdv["DESC"].str.contains(RX_SD_MISTO), "misto", "juros"))
        c_arr = sdv["DESC"].str.contains(RX_ARR_SD)
        for c in ("juros", "principal", "misto"):
            registrar(sdv[(sdv["classe"] == c) & ~c_arr], f"servico_divida:{c}")
            registrar(sdv[(sdv["classe"] == c) & c_arr], f"servico_divida:{c}+arrendamento")
        total = float(-sdv["VL"].sum()) if len(sdv) else 0.0
        if c_arr.any():
            m["servico_div_arrend_misto"] = total
            m["juros_pagos"] = m["principal_pago"] = m["servico_divida_pago"] = np.nan
            notas.append("dívida e arrendamento nas mesmas linhas da DFC: só os índices com arrendamento no "
                         "denominador são calculáveis")
        else:
            m["servico_divida_pago"] = total
            if (sdv["classe"] == "misto").any():
                m["juros_pagos"] = m["principal_pago"] = np.nan
                notas.append("principal e juros da dívida em linha única: só o serviço total é confiável")
            else:
                m["juros_pagos"] = float(-sdv.loc[sdv["classe"] == "juros", "VL"].sum())
                m["principal_pago"] = float(-sdv.loc[sdv["classe"] == "principal", "VL"].sum())
        if not len(sdv):
            notas.append("nenhum pagamento de juros/principal de dívida identificado na DFC (assumido 0)")

    # Fluxo de caixa operacional. fco_reportado = total da DFC (6.01). fco = FCO comparável: sem juros e
    # dividendos/JCP pagos ou recebidos que a empresa classifica no operacional (valores com o sinal da DFC
    # em fco_juros_no_fco e fco_div_no_fco), para que FCO ≈ EBITDA − capital de giro − IR/CSLL pagos.
    m["fco_reportado"] = conta(sub, dfc_demo, "6.01") if not dfc.empty else np.nan
    jd = dfc[dfc["CD_CONTA"].str.startswith("6.01.")].copy()
    jd["classe"] = [classe_jur_div_fco(d, v, c.startswith("6.01.01"))
                    for d, v, c in zip(jd["DESC"], jd["VL"], jd["CD_CONTA"])]
    jd = topo(jd[jd["classe"].notna()])
    for c in ("juros", "dividendos"):
        registrar(jd[jd["classe"] == c], f"fco_excluido:{c}")
    if dfc.empty:
        m["fco_juros_no_fco"] = m["fco_div_no_fco"] = np.nan
    else:
        m["fco_juros_no_fco"] = float(jd.loc[jd["classe"] == "juros", "VL"].sum())
        m["fco_div_no_fco"] = float(jd.loc[jd["classe"] == "dividendos", "VL"].sum())
    m["fco"] = m["fco_reportado"] - m["fco_juros_no_fco"] - m["fco_div_no_fco"]
    # FCL (definição própria) = FCO comparável − capex orgânico líquido − arrendamento pago, contado uma vez:
    # o pago fora do FCO + os juros de arrendamento retirados do FCO acima
    pag_fora_fco = 0.0
    if len(pg):
        pag_fora_fco = -pg.loc[~pg["CD_CONTA"].str.startswith("6.01"), "VL"].sum()
    arr_j = jd[(jd["classe"] == "juros") & jd["DESC"].str.contains(RX_ARR_SD)]
    arr_misto = arr_j["DESC"].str.contains(RX_ARR_MISTO)
    pag_fora_fco += -arr_j.loc[~arr_misto, "VL"].sum()
    if arr_misto.any():
        notas.append("juros de dívida e arrendamento na mesma linha do FCO: retirados do FCO, mas a parte de "
                     "arrendamento não é descontada no FCL (FCL pode estar superestimado)")
    m["fcl_pos_arrend"] = m["fco"] - m["capex_organico_liquido"] - pag_fora_fco

    ifrs16 = pos_ifrs16
    m["regime_contabil"] = "IFRS 16" if ifrs16 else "IAS 17"
    if not ifrs16:
        m["ebitda_pre_ifrs16"] = m["ebitda"]
    elif not np.isnan(m["pagamento_arrendamento"]):
        m["ebitda_pre_ifrs16"] = m["ebitda"] - m["pagamento_arrendamento"]
    else:
        m["ebitda_pre_ifrs16"] = np.nan
        notas.append("pagamento de arrendamento não identificado")

    # Construção (concessões): valores identificados; o ajuste só é aplicado no universo (ajustar_construcao)
    m.update(construcao(sub, m["receita"], registrar))
    m.update(componentes_eletricos(sub, dfc, dfc_demo, registrar))
    for k, rep in REPORTADOS.items():
        m[rep] = m.get(k)

    derivadas(m)
    m["notas"] = "; ".join(notas)
    return m


def _nz(x):
    """NaN para ausente ou zero (linha que existe na DFP, mas vem zerada)."""
    return x if pd.notna(x) and abs(x) > 1e-9 else np.nan


def construcao(sub: pd.DataFrame, receita_dre, registrar=None) -> dict:
    """Receita e custo de construção de infraestrutura de concessão (ICPC 01), em R$ mm, positivos.
    Ordem de busca:
      receita: DRE (subconta de 3.01 com "construção") > DVA (7.01.xx com "construção", só quando a receita
               líquida da DRE a comporta: DRE − construção entre 70% e 105% das vendas brutas da DVA, e DRE acima
               das vendas brutas ou custo de construção nos insumos da DVA) > custo + margem da DFC
      custo:   DRE (subconta de 3.02) > receita − margem da DFC > DVA (insumo 7.02.xx) > receita (sem margem)
      margem da DFC: ajuste não-caixa "margem de construção" (6.01.xx), com sinal trocado
    """
    def linhas(demo, rx_cod):
        d = sub[(sub["DEMO"] == demo) & sub["CD_CONTA"].str.match(rx_cod) & sub["DESC"].str.contains(RX_CONSTR)
                & ~sub["DESC"].str.contains(RX_CONSTR_EXC)]
        return topo(d)

    def soma(rows):
        return _nz(rows["VL"].sum()) if len(rows) else np.nan

    r_dre_l, c_dre_l = linhas("DRE", r"^3\.01\."), linhas("DRE", r"^3\.02\.")
    r_dva_l, c_dva_l = linhas("DVA", r"^7\.01\."), linhas("DVA", r"^7\.02\.")
    r_dre, c_dre = soma(r_dre_l), soma(c_dre_l)
    r_dva, c_dva = soma(r_dva_l), soma(c_dva_l)
    c_dre = -c_dre if pd.notna(c_dre) else np.nan
    c_dva = -c_dva if pd.notna(c_dva) else np.nan
    dfc = sub[sub["DEMO"].isin(["DFC_MI", "DFC_MD"]) & sub["CD_CONTA"].str.match(r"^6\.01\.")]
    mg_l = topo(dfc[dfc["DESC"].str.contains(RX_MARGEM_CONSTR) & ~dfc["DESC"].str.contains(RX_MARGEM_CONSTR_EXC)])
    mg = mg_l["VL"].sum() if len(mg_l) else np.nan
    mg = -mg if pd.notna(mg) else np.nan  # na DFC a margem é deduzida do lucro (valor negativo)

    # DVA: a receita de construção de ativos próprios nem sempre está na DRE (ex.: ferrovias que constroem o
    # próprio imobilizado). Só vale quando a receita líquida da DRE a comporta.
    if pd.notna(r_dva):
        vendas = conta(sub, "DVA", "7.01.01")
        # a construção cabe na receita da DRE (receita líquida − construção = 70% a 105% das vendas brutas) e há
        # evidência de que passa pelo resultado: receita líquida acima das vendas brutas ou custo de construção
        # nos insumos da DVA
        ok = pd.notna(receita_dre) and pd.notna(vendas) and vendas > 0 and \
            0.70 <= (receita_dre - r_dva) / vendas <= 1.05 and (receita_dre > vendas or pd.notna(c_dva))
        if not ok:
            r_dva = c_dva = np.nan
    fonte = []
    rc, cc = r_dre, c_dre
    if pd.notna(rc):
        fonte.append("receita DRE")
        if registrar:
            registrar(r_dre_l, "receita_construcao")
    elif pd.notna(r_dva):
        rc = r_dva
        fonte.append("receita DVA")
        if registrar:
            registrar(r_dva_l, "receita_construcao")
    if pd.notna(cc):
        fonte.append("custo DRE")
        if registrar:
            registrar(c_dre_l, "custo_construcao")
    elif pd.notna(c_dva) and pd.notna(rc) and pd.isna(mg):  # com margem na DFC, custo = receita − margem
        cc = c_dva
        fonte.append("custo DVA")
        if registrar:
            registrar(c_dva_l, "custo_construcao")
    if pd.notna(mg) and registrar:
        registrar(mg_l, "margem_construcao_dfc")
    m0 = mg if pd.notna(mg) else 0.0
    if pd.isna(rc) and pd.notna(cc):
        rc = cc + m0
        fonte.append("receita = custo + margem DFC" if pd.notna(mg) else "receita = custo (sem margem)")
    elif pd.notna(rc) and pd.isna(cc):
        cc = rc - m0
        fonte.append("custo = receita − margem DFC" if pd.notna(mg) else "custo = receita (sem margem)")
    elif pd.isna(rc) and pd.notna(mg):
        fonte.append("só margem DFC")
    return {"receita_construcao": rc, "custo_construcao": cc, "margem_construcao_dfc": mg,
            "fonte_construcao": "; ".join(fonte)}


def ajustar_construcao(m: dict, ativo: bool) -> dict:
    """Retira a construção da concessão dos resultados (só quando ativo=True, concessões do modelo intangível).
    Receita líquida e CPV saem pelo valor da construção; lucro bruto, EBIT e EBITDA saem pela margem
    (receita − custo); lucro líquido sai pela margem líquida de IR/CS (34%). Sem receita de construção
    identificada, mas com margem na DFC, só a margem sai dos lucros (a receita fica a reportada).
    Idempotente: parte sempre dos campos *_reportado*."""
    for k, rep in REPORTADOS.items():
        if pd.isna(m.get(rep)) and pd.notna(m.get(k)):
            m[rep] = m[k]
        m[k] = m.get(rep)
    m["margem_construcao"] = 0.0
    if not ativo:
        return m
    rc, cc, mg = m.get("receita_construcao"), m.get("custo_construcao"), m.get("margem_construcao_dfc")
    rc = rc if rc is not None and pd.notna(rc) else np.nan
    cc = cc if cc is not None and pd.notna(cc) else np.nan
    if pd.notna(rc) and pd.notna(cc):
        marg = rc - cc
        for k, v in (("receita", rc), ("cpv", cc)):
            if pd.notna(m.get(k)):
                m[k] = m[k] - v
    elif mg is not None and pd.notna(mg):
        marg = mg
    else:
        return m
    m["margem_construcao"] = marg
    if marg:
        for k in ("lucro_bruto", "ebit", "ebitda", "ebitda_pre_ifrs16"):
            if pd.notna(m.get(k)):
                m[k] = m[k] - marg
        for k in ("lucro_liquido", "lucro_controladores"):
            if pd.notna(m.get(k)):
                m[k] = m[k] - marg * (1 - IR_CS_CONSTRUCAO)
    return m


def componentes_eletricos(sub: pd.DataFrame, dfc: pd.DataFrame, dfc_demo: str, registrar=None) -> dict:
    """Peças do EBITDA ajustado do setor elétrico (busca textual; aplicadas só às empresas de eletricas.yaml):
      ativo_concessao_saldo: saldo do ativo de contrato/concessão (BP, 1.01 + 1.02), R$ mm
      ativo_fin_concessao_saldo: parte "ativo financeiro de concessão" desse saldo (bonificação, na geração)
      bonificacao_saldo:     saldo com "bonificação pela outorga" no nome (BP)
      custo_implantacao:     custo de construção/implementação de infraestrutura (DRE 3.02.xx > ajuste da DFC >
                             DVA 7.02.xx; com transmissão no nome, quando houver), positivo
      custo_implant_om:      linha do CPV que junta implantação e O&M (ex.: ISA), menos O&M estimado pela DVA
                             (insumos de terceiros 7.02.02 + pessoal 7.08.01) — aproximação
      pis_cofins_diferidos:  PIS/COFINS diferidos somados de volta na DFC (não-caixa)
      equivalencia_patrimonial: DRE 3.04.06 (sai do EBITDA ajustado; o regulatório consolidado não a inclui)
      vnr_atualizacao:       atualização do ativo financeiro indenizável da DFC (ganho não-caixa = positivo)"""
    out = {}
    bpa = sub[(sub["DEMO"] == "BPA") & sub["CD_CONTA"].str.match(r"^1\.0[12]\.")]
    bonif = topo(bpa[bpa["DESC"].str.contains(RX_BONIF)])
    ac = bpa[bpa["DESC"].str.contains(RX_AC_SALDO) & ~bpa["DESC"].str.contains(RX_AC_SALDO_EXC)]
    ac = topo(ac[~ac["CD_CONTA"].isin(bonif["CD_CONTA"])])
    if registrar:
        registrar(ac, "eletrico:ativo_concessao_saldo")
        registrar(bonif, "eletrico:bonificacao_saldo")
    out["ativo_concessao_saldo"] = ac["VL"].sum() if len(ac) else np.nan
    fin = ac[ac["DESC"].str.contains(RX_AC_FIN)]
    out["ativo_fin_concessao_saldo"] = fin["VL"].sum() if len(ac) else np.nan
    out["bonificacao_saldo"] = bonif["VL"].sum() if len(bonif) else np.nan

    def implant(demo, prefixo, om=False):
        d = sub[(sub["DEMO"] == demo) & sub["CD_CONTA"].str.match(prefixo) & sub["DESC"].str.contains(RX_IMPLANT)
                & ~sub["DESC"].str.contains(RX_IMPLANT_EXC)]
        d = d[d["DESC"].str.contains(RX_IMPLANT_OM) == om]
        if d["DESC"].str.contains("transmiss").any():   # empresa com geração e transmissão: só a da transmissão
            d = d[d["DESC"].str.contains("transmiss")]
        return topo(d)
    v, fonte, ci = np.nan, "", None
    for demo, pref, nome in (("DRE", r"^3\.02\.", "DRE"), ("DFC_MI", r"^6\.01\.0[13]\.", "DFC (ajuste não-caixa)"),
                             ("DVA", r"^7\.02\.", "DVA")):
        ci = implant(demo, pref)
        if demo == "DFC_MI":
            ci = ci[ci["VL"] > 0]   # custo somado de volta ao lucro
        v = _nz(ci["VL"].sum()) if len(ci) else np.nan
        if pd.notna(v):
            fonte = nome
            break
    out["custo_implantacao"] = abs(v) if pd.notna(v) else np.nan
    out["fonte_custo_implantacao"] = f"custo de implantação {fonte}" if pd.notna(v) else ""
    if registrar and pd.notna(v):
        registrar(ci, "eletrico:custo_implantacao")
    cm = implant("DRE", r"^3\.02\.", om=True)
    vm = _nz(cm["VL"].sum()) if len(cm) else np.nan
    if pd.notna(vm):
        om = np.nansum([conta(sub, "DVA", "7.02.02"), conta(sub, "DVA", "7.08.01") * -1])
        out["custo_implant_om"] = max(abs(vm) - abs(om), 0.0)
        if registrar:
            registrar(cm, "eletrico:custo_implantacao_com_om")
    else:
        out["custo_implant_om"] = np.nan

    eq = sub[(sub["DEMO"] == "DRE") & sub["CD_CONTA"].str.match(r"^3\.04\.") & sub["DESC"].str.contains(RX_EQUIV)]
    eq = topo(eq)
    out["equivalencia_patrimonial"] = eq["VL"].sum() if len(eq) else 0.0

    if dfc_demo == "DFC_MI" and not dfc.empty:
        aj = dfc[dfc["CD_CONTA"].str.match(r"^6\.01\.0[13]\.")]
        # verbo (atualização, valor justo...) exigido fora de 6.01.01; nos ajustes não-caixa basta o nome do ativo
        vn = topo(aj[aj["DESC"].str.contains(RX_VNR) & ~aj["DESC"].str.contains(RX_VNR_EXC)
                     & (aj["DESC"].str.contains(RX_VNR_VERBO) | aj["CD_CONTA"].str.startswith("6.01.01."))])
        pc = topo(aj[aj["DESC"].str.contains(RX_PIS_DIF)])
        if registrar:
            registrar(vn, "eletrico:vnr_atualizacao")
            registrar(pc, "eletrico:pis_cofins_diferidos")
        out["vnr_atualizacao"] = float(-vn["VL"].sum()) if len(vn) else 0.0  # na DFC o ganho é deduzido do lucro
        out["vnr_identificado"] = bool(len(vn))
        out["pis_cofins_diferidos"] = float(pc["VL"].sum()) if len(pc) else 0.0
        op = dfc[dfc["CD_CONTA"].str.match(r"^6\.01\.0[12]\.")]
        ad = topo(op[op["DESC"].str.contains(RX_AC_DFC) & ~op["DESC"].str.contains(RX_AC_DFC_EXC)])
        if registrar:
            registrar(ad, "eletrico:var_ativo_concessao_dfc")
        out["var_ativo_concessao_dfc"] = float(-ad["VL"].sum()) if len(ad) else np.nan
    else:
        out["vnr_atualizacao"], out["vnr_identificado"], out["pis_cofins_diferidos"] = np.nan, False, np.nan
        out["var_ativo_concessao_dfc"] = np.nan
    return out


CAMPOS_TRANSM = ("ativo_concessao_saldo", "var_ativo_concessao", "custo_implantacao")


def ajustar_eletrico(m: dict, cfg: dict | None) -> dict:
    """EBITDA ajustado do setor elétrico (aproximação do EBITDA regulatório consolidado), por tipo de empresa:
      transmissão: EBITDA IFRS + custo de construção − Δ ativo de contrato/concessão + PIS/COFINS diferidos. A receita
                   de construção, a remuneração e a correção monetária do ativo saem pelo Δ; entra a RAP faturada.
                   Sem construção identificada e ativo em queda (concessão madura), custo = 0.
      VNR:         − atualização do ativo financeiro indenizável (distribuição; não-caixa).
      bonificação: − Δ saldo da bonificação pela outorga (remuneração na receita, recebimento reduz o saldo); sem conta
                   própria, usa a parte "ativo financeiro de concessão" do saldo.
      Todos:       − equivalência patrimonial (o EBITDA regulatório consolidado não a inclui).
    Os Δ e os custos são fluxos do período (var_* vêm do universo, por diferença de saldos). Sem cfg: NaN."""
    m["ebitda_ajustado"], m["fonte_ajuste_eletrico"] = np.nan, ""
    aj = (cfg or {}).get("ajustes") or []
    if "transmissao" not in aj:   # saldo e custo de implantação só fazem sentido (e só aparecem) nas transmissoras
        for k in CAMPOS_TRANSM:
            m[k] = np.nan
    if not cfg:
        return m
    e = m.get("ebitda")
    if e is None or pd.isna(e):
        return m

    def ok_(x):
        return x is not None and pd.notna(x) and x != 0
    tipo, val, fonte, ok = cfg.get("tipo", ""), float(e), [], True
    va, vf = m.get("var_ativo_concessao"), m.get("var_ativo_fin_concessao")
    vb = m.get("var_bonificacao")
    if "bonificacao" in aj and (vb is None or pd.isna(vb)) and vf is not None and pd.notna(vf) and vf:
        vb = vf   # bonificação registrada como "ativo financeiro de concessão": sai do Δ da transmissão
        va = va - vf if va is not None and pd.notna(va) else va   # (a versão pela DFC já exclui essa parte)
        fonte.append("bonificação = ativo financeiro de concessão do BP")
    if "transmissao" in aj:
        cc, orig = m.get("custo_implantacao"), m.get("fonte_custo_implantacao") or "custo de implantação"
        if not ok_(cc):
            cc, orig = m.get("custo_construcao"), m.get("fonte_construcao") or "construção"
        if not ok_(cc):
            cc, orig = m.get("custo_implant_om"), "≈ CPV de implantação + O&M − O&M estimado pela DVA"
        if not ok_(cc):
            ca = m.get("capex_ativo_contrato")
            cc, orig = (ca, "≈ investimento em ativo de contrato na DFC") if ok_(ca) and ca > 0 else (np.nan, "")
        if (cc is None or pd.isna(cc)) and va is not None and pd.notna(va) and va <= 0:
            cc, orig = 0.0, "sem construção identificada; ativo em queda (assumido 0)"
        aq = m.get("aquisicoes_participacoes")
        com_aquis = aq is not None and pd.notna(aq) and aq > 0
        vd = m.get("var_ativo_concessao_dfc")
        usa_dfc = com_aquis and cfg.get("dfc_confiavel") and vd is not None and pd.notna(vd)
        if usa_dfc:   # período com compra de SPEs: Δ operacional pela DFC (sem a consolidação)
            va = vd
        if va is None or pd.isna(va):
            ok = False
            fonte.append("Δ ativo de contrato não identificado (saldo do BP ou do ano anterior ausente)")
        elif cc is None or pd.isna(cc):
            ok = False
            fonte.append("custo de construção da transmissão não identificado")
        else:
            pc = m.get("pis_cofins_diferidos")
            pc = pc if pc is not None and pd.notna(pc) else 0.0
            val += cc - va + pc
            fonte.append(f"+ custo de construção ({orig}) − Δ ativo de contrato"
                         + (" + PIS/COFINS diferidos" if pc else ""))
            if usa_dfc:
                fonte.append("período com aquisição de participações: Δ do ativo pela DFC (sem a consolidação de SPEs)")
            elif com_aquis:
                fonte.append("ATENÇÃO: período com aquisição de participações — o Δ do ativo de contrato pode incluir "
                             "a consolidação de SPEs (EBITDA ajustado subestimado)")
    if "vnr" in aj:
        v = m.get("vnr_atualizacao")
        if ok_(v):
            val -= v
            fonte.append("− atualização do ativo financeiro indenizável (VNR)")
        else:
            fonte.append("VNR sem linha própria na DFC (sem ajuste)")
    if "bonificacao" in aj:
        if vb is not None and pd.notna(vb):
            val -= vb
            fonte.append("− Δ bonificação pela outorga")
        else:
            fonte.append("bonificação pela outorga não identificada (sem ajuste)")
    eq = m.get("equivalencia_patrimonial")
    if ok_(eq):
        val -= eq
        fonte.append("− equivalência patrimonial")
    if tipo == "integrada":
        fonte.append("transmissão não ajustada (consolidado mistura distribuição e transmissão)")
    if tipo in ("geracao", "holding") and not aj and not ok_(eq):
        fonte.append("sem ajuste (EBITDA IFRS)")
    if cfg.get("confirmar"):
        fonte.append("classificação a confirmar")
    m["ebitda_ajustado"] = val if ok else np.nan
    m["fonte_ajuste_eletrico"] = f"{tipo}: " + "; ".join(fonte)
    return m


def _div(a, b):
    return a / b if b not in (0, None) and not pd.isna(a) and not pd.isna(b) else np.nan


def derivadas(m: dict, anual: bool = True) -> dict:
    """Métricas derivadas (recalculadas em cada período). anual=False (trimestre isolado):
    alavancagem e ROE ficam vazios, porque fluxo de 3 meses não é comparável a estoque."""
    m["div_liq_com_arrend"] = m["divida_bruta_ex_arrend"] + m["passivo_arrendamento"] - m["caixa"]  # NaN se arrend. não identificado
    m["div_liq_sem_arrend"] = m["divida_bruta_ex_arrend"] - m["caixa"]
    m["ncav"] = m["ativo_circulante"] - m["passivo_circulante"] - m["passivo_nao_circulante"]
    m["margem_bruta"] = _div(m["lucro_bruto"], m["receita"])
    m["margem_ebitda"] = _div(m["ebitda"], m["receita"])
    m["margem_ebitda_pre_ifrs16"] = _div(m["ebitda_pre_ifrs16"], m["receita"])
    m["margem_liquida"] = _div(m["lucro_liquido"], m["receita"])
    m["capex_org_receita"] = _div(m["capex_organico"], m["receita"])
    m["capex_org_da"] = _div(m["capex_organico"], m["da"])
    # Cobertura da dívida (fluxos do mesmo período). IR = IR + CSLL pagos (DFC); arrendamento pago =
    # principal + juros (DFC); sob IAS 17 (antes de 2019) o aluguel já está no EBITDA: não identificado = 0.
    def _g(k):
        v = m.get(k)
        return v if v is not None and pd.notna(v) else np.nan

    def _pos(x):
        return x if pd.notna(x) and x > 0 else np.nan
    arr_pg = _g("pagamento_arrendamento")
    if pd.isna(arr_pg) and m.get("regime_contabil") == "IAS 17":
        arr_pg = 0.0
    rf = _g("resultado_financeiro")
    m["desp_fin_liquida"] = -rf if pd.notna(rf) and rf < 0 else np.nan  # vazio com resultado financeiro >= 0
    m["ebitda_menos_ir"] = _g("ebitda") - _g("ir_csll_pagos")
    m["caixa_servico_divida"] = m["ebitda_menos_ir"] - arr_pg
    # capex da cobertura = capex orgânico (6.02) + compra de frota no FCO líquida do valor contábil vendido
    cf = _g("capex_frota_liquido")
    m["capex_cobertura"] = _g("capex_organico") + (cf if pd.notna(cf) else 0.0)
    m["caixa_servico_pos_capex"] = m["caixa_servico_divida"] - m["capex_cobertura"]
    m["ebitda_ir_capex"] = m["ebitda_menos_ir"] - m["capex_cobertura"]
    mx = _g("servico_div_arrend_misto")
    if pd.notna(mx):  # arrendamento dentro das linhas de dívida da DFC
        m["servico_divida_c_arrend"] = mx + (arr_pg if pd.notna(arr_pg) else 0.0)
    else:
        m["servico_divida_c_arrend"] = _g("servico_divida_pago") + arr_pg
    m["cob_ebit_res_fin"] = _div(_g("ebit"), m["desp_fin_liquida"])
    m["cob_juros_caixa"] = _div(m["caixa_servico_divida"], _pos(_g("juros_pagos")))
    m["dscr_ebitda"] = _div(m["caixa_servico_divida"], _pos(_g("servico_divida_pago")))
    m["dscr_pos_capex"] = _div(m["caixa_servico_pos_capex"], _pos(_g("servico_divida_pago")))
    m["dscr_capex_arrend_den"] = _div(m["ebitda_ir_capex"], _pos(m["servico_divida_c_arrend"]))
    m["dscr_ebitda_arrend_den"] = _div(m["ebitda_menos_ir"], _pos(m["servico_divida_c_arrend"]))
    ea = _g("ebitda_ajustado")
    m["caixa_servico_ajustado"] = ea - _g("ir_csll_pagos") - arr_pg
    m["dscr_ebitda_ajustado"] = _div(m["caixa_servico_ajustado"], _pos(_g("servico_divida_pago")))
    m["margem_ebitda_ajustado"] = _div(ea, m["receita"])
    if anual:
        m["div_liq_ebitda_ajustado"] = _div(m["div_liq_com_arrend"], _pos(ea))
        m["div_liq_ebitda_ifrs16"] = _div(m["div_liq_com_arrend"], m["ebitda"])
        m["div_liq_ebitda_pre_ifrs16"] = _div(m["div_liq_sem_arrend"], m["ebitda_pre_ifrs16"])
        m["roe_pl_final"] = _div(m["lucro_liquido"], m["patrimonio_liquido"])
    else:
        m["div_liq_ebitda_ifrs16"] = m["div_liq_ebitda_pre_ifrs16"] = m["roe_pl_final"] = np.nan
        m["div_liq_ebitda_ajustado"] = np.nan
    return m


METRICAS = [  # (chave, rótulo, formato)
    ("receita", "Receita líquida (R$ mm)", "num"),
    ("lucro_bruto", "Lucro bruto (R$ mm)", "num"),
    ("ebit", "EBIT (R$ mm)", "num"),
    ("da", "D&A (R$ mm)", "num"),
    ("impairment", "Impairment - perda/reversão ao valor recuperável (R$ mm)", "num"),
    ("ebitda", "EBITDA reportado/IFRS 16 (R$ mm)", "num"),
    ("pagamento_arrendamento", "Pagamentos de arrendamento - DFC (R$ mm)", "num"),
    ("ebitda_pre_ifrs16", "EBITDA pré-IFRS 16 (R$ mm)", "num"),
    ("lucro_liquido", "Lucro líquido (R$ mm)", "num"),
    ("fco_reportado", "Fluxo de caixa operacional reportado - DFC 6.01 (R$ mm)", "num"),
    ("fco_juros_no_fco", "Juros pagos/recebidos dentro do FCO, líquido, sinal da DFC (R$ mm)", "num"),
    ("fco_div_no_fco", "Dividendos/JCP pagos/recebidos dentro do FCO, líquido, sinal da DFC (R$ mm)", "num"),
    ("fco", "FCO comparável, sem juros e dividendos (R$ mm)", "num"),
    ("capex_imobilizado", "Capex imobilizado (R$ mm)", "num"),
    ("capex_intangivel", "Capex intangível (R$ mm)", "num"),
    ("capex_imob_intang_combinado", "Capex imob.+intang. em linha única (R$ mm)", "num"),
    ("capex_ativo_contrato", "Capex ativo de contrato/concessão (R$ mm)", "num"),
    ("capex_organico", "Capex orgânico total (R$ mm)", "num"),
    ("vendas_imob_intang", "Vendas de imobilizado/intangível (R$ mm)", "num"),
    ("capex_organico_liquido", "Capex orgânico líquido de vendas (R$ mm)", "num"),
    ("aquisicoes_participacoes", "Aquisições de controladas/coligadas (R$ mm)", "num"),
    ("capex_total", "Capex total c/ aquisições (R$ mm)", "num"),
    ("capex_total_liquido", "Capex total líquido de vendas (R$ mm)", "num"),
    ("ativo_contrato_no_fco", "Info: ativo de contrato no FCO (R$ mm)", "num"),
    ("fcl_pos_arrend", "FCL pós-arrendamento (R$ mm)", "num"),
    ("caixa", "Caixa + aplicações CP (R$ mm)", "num"),
    ("divida_bruta_ex_arrend", "Dívida bruta ex-arrendamento (R$ mm)", "num"),
    ("passivo_arrendamento", "Passivo de arrendamento (R$ mm)", "num"),
    ("div_liq_com_arrend", "Dívida líquida c/ arrendamento (R$ mm)", "num"),
    ("div_liq_sem_arrend", "Dívida líquida s/ arrendamento (R$ mm)", "num"),
    ("patrimonio_liquido", "Patrimônio líquido (R$ mm)", "num"),
    ("ncav", "NCAV - Graham (R$ mm)", "num"),
    ("margem_bruta", "Margem bruta", "pct"),
    ("margem_ebitda", "Margem EBITDA IFRS 16", "pct"),
    ("margem_ebitda_pre_ifrs16", "Margem EBITDA pré-IFRS 16", "pct"),
    ("margem_liquida", "Margem líquida", "pct"),
    ("roe_pl_final", "ROE (LL / PL final)", "pct"),
    ("capex_org_receita", "Capex orgânico / receita", "pct"),
    ("capex_org_da", "Capex orgânico / D&A", "mult"),
    ("div_liq_ebitda_ifrs16", "DL c/ arrend / EBITDA IFRS 16", "mult"),
    ("div_liq_ebitda_pre_ifrs16", "DL s/ arrend / EBITDA pré-IFRS 16", "mult"),
    ("resultado_financeiro", "Resultado financeiro líquido (R$ mm)", "num"),
    ("ir_csll_pagos", "IR + CSLL pagos - DFC (R$ mm)", "num"),
    ("juros_pagos", "Juros pagos da dívida, ex-arrend. - DFC (R$ mm)", "num"),
    ("principal_pago", "Principal amortizado, ex-arrend. - DFC (R$ mm)", "num"),
    ("servico_divida_pago", "Serviço da dívida pago (juros + principal) (R$ mm)", "num"),
    ("capex_frota_liquido", "Capex de frota no FCO, líq. do valor contábil vendido (R$ mm)", "num"),
    ("cob_ebit_res_fin", "EBIT / resultado financeiro líquido", "mult"),
    ("cob_juros_caixa", "(EBITDA - IR - arrend.) / juros pagos", "mult"),
    ("dscr_ebitda", "(EBITDA - IR - arrend.) / (juros + principal)", "mult"),
    ("dscr_pos_capex", "(EBITDA - IR - capex org. - arrend.) / (juros + principal)", "mult"),
    ("dscr_capex_arrend_den", "(EBITDA - IR - capex org.) / (juros + principal + arrend.)", "mult"),
    ("dscr_ebitda_arrend_den", "(EBITDA - IR) / (juros + principal + arrend.)", "mult"),
]


# ------------------------------------------------------------------- coleta
def coletar_grupo(nome_grupo: str, cfg: dict, cad: pd.DataFrame, b3, anos: list[int], n_anos: int):
    print(f"\n=== Grupo: {nome_grupo} ===")
    empresas = resolver_empresas(cfg, cad, b3)
    todos = {c for e in empresas for c in e["cnpjs"]}
    ano_max = max(anos)
    arq_anos = [a for a in anos if a >= ano_max - n_anos]  # +1 ano p/ reapresentações
    recentes = {ano_max, ano_max - 1}
    brutos = []
    for a in arq_anos:
        print(f"  lendo DFP {a}...")
        d = ler_dfp(a, todos, forcar=(a in recentes and not OFFLINE))
        if not d.empty:
            brutos.append(d)
    if not brutos:
        raise SystemExit("Nenhum dado encontrado para o grupo.")
    df = preparar(pd.concat(brutos, ignore_index=True))

    linhas, aud, reapres, cobertura = [], [], [], []
    for emp in empresas:
        de = df[df["CNPJ_NUM"].isin(emp["cnpjs"])]
        resultados = {}  # exercicio -> dict
        for (cnpj, dt_ref, arq, ordem), g in de.groupby(["CNPJ_NUM", "DT_REFER", "ARQ_ANO", "ORDEM"]):
            base = "con" if ((g["BASE"] == "con") & (g["DEMO"] == "DRE")).any() else "ind"
            g = g[g["BASE"] == base]
            if g.empty or g["DT_FIM"].isna().all():
                continue
            dt_fim = g["DT_FIM"].max()
            exerc = dt_fim.year
            tipo = "reapresentado" if ordem == "penultimo" else "original"
            ctx = {"empresa": emp["nome"], "exercicio": exerc, "tipo": tipo,
                   "dfp_ano": arq, "cnpj": cnpj, "dt_fim": dt_fim}
            aud_tmp = []
            met = calcular(g, aud_tmp, ctx)
            met.update({"base": "consolidado" if base == "con" else "individual",
                        "cnpj": cnpj, "dt_fim": dt_fim, "dfp_ano": arq,
                        "denominacao": g["DENOM_CIA"].iloc[0]})
            slot = resultados.setdefault(exerc, {})
            # cadeia de CNPJs: em conflito, prevalece o CNPJ mais recente da lista
            atual = slot.get(tipo)
            if atual is None or emp["cnpjs"].index(cnpj) >= emp["cnpjs"].index(atual[0]["cnpj"]):
                slot[tipo] = (met, aud_tmp)

        exercicios = sorted(resultados)[-n_anos:]
        for ex in exercicios:
            slot = resultados[ex]
            final_tipo = "reapresentado" if "reapresentado" in slot else "original"
            met, a = slot[final_tipo]
            aud.extend(x for x in a)
            linhas.append({"empresa": emp["nome"], "exercicio": ex, "fonte_valor":
                           f"{final_tipo} (DFP {met['dfp_ano']})", **met})
            if "reapresentado" in slot and "original" in slot:
                o, r = slot["original"][0], slot["reapresentado"][0]
                for k, rot, _ in METRICAS:
                    vo, vr = o.get(k), r.get(k)
                    if pd.notna(vo) and pd.notna(vr) and abs(vr - vo) > LIMIAR_REAPRES * max(abs(vo), 1e-9):
                        reapres.append({"empresa": emp["nome"], "exercicio": ex, "metrica": rot,
                                        "original": vo, "reapresentado": vr, "diferenca": vr - vo})

        # cobertura e motivo do corte
        cad_e = cad[cad["CNPJ_NUM"].isin(emp["cnpjs"])]
        motivos = []
        if exercicios:
            ini, fim = exercicios[0], exercicios[-1]
            if len(exercicios) < n_anos:
                faltam = sorted(set(range(ini, fim + 1)) - set(exercicios))
                if faltam:
                    motivos.append(f"exercícios faltantes: {faltam}")
                dt_reg = pd.to_datetime(cad_e["DT_REG"], errors="coerce").min() if "DT_REG" in cad_e else pd.NaT
                if pd.notna(dt_reg) and dt_reg.year >= ini - 1:
                    motivos.append(f"registro na CVM em {dt_reg.date()}")
                elif ini > ano_max - n_anos + 1:
                    motivos.append(f"sem DFP estruturada antes de {ini} (verificar ENET/PDF)")
            canc = cad_e[cad_e.get("SIT", pd.Series(dtype=str)).map(norm).str.contains("cancel", na=False)]
            if not canc.empty and fim < ano_max - 1:
                motivos.append(f"registro cancelado em {canc['DT_CANCEL'].iloc[0]}")
            meses = {resultados[x][next(iter(resultados[x]))][0]["dt_fim"].month for x in exercicios}
            obs = "" if meses == {12} else f"exercício social termina no mês {sorted(meses)}: não alinhado ao ano-calendário"
            cobertura.append({"empresa": emp["nome"], "ticker": emp.get("ticker", ""), "cnpjs": ", ".join(emp["cnpjs"]),
                              "inicio_serie": ini, "fim_serie": fim, "anos_disponiveis": len(exercicios),
                              "motivo_corte": "; ".join(motivos) if motivos else
                              ("janela completa" if len(exercicios) >= n_anos else "série mais curta: verificar"),
                              "regime_contabil": ", ".join(sorted({resultados[x][next(iter(resultados[x]))][0]['regime_contabil'] for x in exercicios})),
                              "observacao": "; ".join(filter(None, [obs, emp["nota"]]))})
        else:
            cobertura.append({"empresa": emp["nome"], "ticker": emp.get("ticker", ""), "cnpjs": ", ".join(emp["cnpjs"]),
                              "anos_disponiveis": 0, "motivo_corte": "nenhuma DFP estruturada encontrada"})

    base_longa = pd.DataFrame(linhas)
    auditoria = pd.DataFrame(aud)
    if not auditoria.empty:
        chaves = set(zip(base_longa["empresa"], base_longa["exercicio"], base_longa["fonte_valor"].str.split(" ").str[0]))
        auditoria = auditoria[[k in chaves for k in zip(auditoria["empresa"], auditoria["exercicio"], auditoria["tipo"])]]
        auditoria = auditoria.drop(columns=["dt_fim"])
    escrever_saidas(nome_grupo, base_longa, pd.DataFrame(cobertura), pd.DataFrame(reapres), auditoria, n_anos)


# ------------------------------------------------------------------- Excel
def escrever_saidas(grupo, base, cob, reap, aud, n_anos):
    from openpyxl.styles import Alignment, Font, PatternFill
    from openpyxl.utils import get_column_letter

    SAIDAS.mkdir(exist_ok=True)
    rot = {k: r for k, r, _ in METRICAS}
    fmt = {r: f for _, r, f in METRICAS}
    painel = (base.melt(id_vars=["empresa", "exercicio"], value_vars=[k for k, _, _ in METRICAS],
                        var_name="metrica", value_name="valor")
              .assign(metrica=lambda d: d["metrica"].map(rot)))
    painel["valor"] = pd.to_numeric(painel["valor"], errors="coerce")
    ordem = {r: i for i, (_, r, _) in enumerate(METRICAS)}
    wide = painel.pivot_table(index=["empresa", "metrica"], columns="exercicio", values="valor", aggfunc="first")
    wide = wide.reset_index().sort_values(["empresa", "metrica"], key=lambda s: s.map(ordem) if s.name == "metrica" else s)
    wide.columns = [str(c) for c in wide.columns]
    mediana = painel.pivot_table(index="metrica", columns="exercicio", values="valor", aggfunc="median").reset_index()
    mediana = mediana.sort_values("metrica", key=lambda s: s.map(ordem))
    mediana.columns = [str(c) for c in mediana.columns]

    leia = pd.DataFrame({"Item": [
        "Grupo", "Gerado em", "Fonte dos dados", "Universo de empresas", "Janela", "Valores", "Escolha da versão",
        "EBITDA", "EBITDA pré-IFRS 16", "Pagamentos de arrendamento", "Passivo de arrendamento",
        "Dívida bruta ex-arrendamento", "Caixa", "NCAV", "Alavancagem",
        "Capex orgânico", "Vendas de imobilizado/intangível", "Aquisições", "Capex total", "Ativo de contrato no FCO", "FCO comparável", "FCL pós-arrendamento", "Capex / D&A", "Lacunas", "Aviso"],
        "Descrição": [
        grupo, datetime.now().strftime("%Y-%m-%d %H:%M"),
        "CVM Dados Abertos — DFP (dados.cvm.gov.br/dados/CIA_ABERTA/DOC/DFP) e cadastro de companhias",
        "Classificação setorial e CNPJ da página de empresas listadas da B3 (filtro_b3 no grupos.yaml) + empresas manuais. "
        "A lista B3 só tem empresas negociadas hoje: deslistadas ficam fora (viés de sobrevivência) salvo inclusão manual",
        f"Últimos {n_anos} exercícios disponíveis por empresa; se menos, maior período possível (aba cobertura)",
        "R$ milhões; consolidado quando existe, senão individual (coluna 'base' na base_longa)",
        "Reapresentado (penúltimo exercício da DFP seguinte) quando existe; original guardado na aba reapresentacoes",
        "Definição própria: EBIT (DRE 3.05) + D&A (ajustes da DFC indireta, busca textual)",
        "Definição própria: EBITDA − pagamentos de arrendamento (principal + juros) da DFC. Antes de 2019 (IAS 17) = EBITDA",
        "Busca textual na DFC (exceto ajustes 6.01.01). Se há linha de total e componentes, usa só o total",
        "Busca textual no passivo (2.01/2.02), só contas de nível mais alto para evitar dupla contagem",
        "Empréstimos e financiamentos (2.01.04 + 2.02.01) menos arrendamento classificado dentro dessas contas",
        "Caixa e equivalentes (1.01.01) + aplicações financeiras (1.01.02)",
        "Ativo circulante − passivo circulante − passivo não circulante (Graham)",
        "Pares consistentes: DL c/ arrend ÷ EBITDA IFRS 16; DL s/ arrend ÷ EBITDA pré-IFRS 16",
        "Saídas de caixa na DFC de investimento (6.02) para imobilizado, intangível e ativo de contrato/concessão (busca textual; positivo = saída)",
        "Entradas na DFC de investimento por venda/alienação de imobilizado e intangível (positivo = entrada); capex líquido = capex − vendas",
        "Saídas na DFC de investimento para aquisição/aporte em controladas, coligadas e joint ventures. Sem linha no ano = 0 (marcado em 'notas')",
        "Capex orgânico + aquisições (bruto e líquido de vendas de imobilizado/intangível)",
        "Informativo: variação de ativo de contrato classificada no operacional (concessões). Não somado ao capex para evitar dupla contagem",
        "Definição própria: FCO (6.01) sem juros e dividendos/JCP pagos ou recebidos classificados no operacional "
        "(diferença para o EBITDA = capital de giro + IR/CSLL pagos). FCO reportado guardado em fco_reportado",
        "Definição própria: FCO comparável − capex orgânico líquido de vendas − arrendamento pago (principal + juros), contado uma vez",
        "Definição própria: proxy de quanto do capex excede a reposição (>1x sugere expansão)",
        "Vazio = não identificado (nunca zero). Ver colunas 'notas' e aba auditoria",
        "Métricas derivadas são cálculo próprio do script, não números reportados pelas companhias. Conferir aba auditoria."]})

    caminho = SAIDAS / f"{grupo}.xlsx"
    with pd.ExcelWriter(caminho, engine="openpyxl") as xw:
        leia.to_excel(xw, sheet_name="LEIA-ME", index=False)
        wide.to_excel(xw, sheet_name="painel", index=False)
        mediana.to_excel(xw, sheet_name="mediana_setor", index=False)
        cob.to_excel(xw, sheet_name="cobertura", index=False)
        (reap if not reap.empty else pd.DataFrame({"info": ["sem reapresentações acima de 0,5%"]})
         ).to_excel(xw, sheet_name="reapresentacoes", index=False)
        (aud if not aud.empty else pd.DataFrame({"info": ["sem linhas de busca textual"]})
         ).to_excel(xw, sheet_name="auditoria", index=False)
        base.drop(columns=["dt_fim"], errors="ignore").to_excel(xw, sheet_name="base_longa", index=False)

        cab = PatternFill("solid", start_color="1F3864")
        for ws in xw.book.worksheets:
            for row in ws.iter_rows():
                for c in row:
                    c.font = Font(name="Arial", size=9, bold=c.row == 1, color="FFFFFF" if c.row == 1 else "000000")
                    if c.row == 1:
                        c.fill = cab
                        c.alignment = Alignment(wrap_text=True, vertical="center")
            ws.freeze_panes = "A2" if ws.title in ("LEIA-ME", "cobertura", "reapresentacoes", "auditoria") else "C2" if ws.title == "painel" else "B2"
            for i, col in enumerate(ws.columns, 1):
                larg = max(len(str(c.value)) if c.value is not None else 0 for c in col)
                ws.column_dimensions[get_column_letter(i)].width = min(max(10, larg + 2), 90)
        for nome, col_met in (("painel", 2), ("mediana_setor", 1)):
            ws = xw.book[nome]
            for row in ws.iter_rows(min_row=2):
                f = fmt.get(row[col_met - 1].value, "num")
                nf = {"num": '#,##0.0;(#,##0.0);"-"', "pct": '0.0%;(0.0%);"-"', "mult": '0.00"x";(0.00"x");"-"'}[f]
                for c in row[col_met:]:
                    c.number_format = nf
    base.drop(columns=["dt_fim"], errors="ignore").to_csv(SAIDAS / f"{grupo}_base.csv", index=False, sep=";", decimal=",")
    print(f"  -> {caminho}")


# --------------------------------------------------------------------- main
def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)
    d = sub.add_parser("descobrir", help="busca companhias no cadastro CVM por nome (regex)")
    d.add_argument("padrao")
    st = sub.add_parser("setores", help="lista setor/subsetor/segmento da B3 com nº de empresas")
    st.add_argument("--atualizar-b3", action="store_true")
    fl = sub.add_parser("filtrar", help="lista empresas B3 de um setor/subsetor/segmento (regex, sem acento)")
    fl.add_argument("--setor"); fl.add_argument("--subsetor"); fl.add_argument("--segmento")
    fl.add_argument("--todas", action="store_true", help="inclui emissores sem cotação")
    fl.add_argument("--atualizar-b3", action="store_true")
    u = sub.add_parser("universo", help="coleta TODAS as empresas classificadas na B3: anual, trimestral e LTM")
    u.add_argument("--anos", type=int, default=10)
    u.add_argument("--atualizar-b3", action="store_true")
    u.add_argument("--limite", type=int, default=0, help="processa só N empresas (teste)")
    c = sub.add_parser("coletar", help="coleta métricas dos grupos do grupos.yaml")
    c.add_argument("--grupo", help="nome do grupo (padrão: todos)")
    c.add_argument("--anos", type=int, default=10)
    c.add_argument("--config", default="grupos.yaml")
    c.add_argument("--atualizar-b3", action="store_true")
    a = ap.parse_args()

    CACHE.mkdir(parents=True, exist_ok=True)
    if a.cmd == "descobrir":
        descobrir(a.padrao)
        return
    if a.cmd == "setores":
        cmd_setores(a.atualizar_b3)
        return
    if a.cmd == "filtrar":
        cmd_filtrar(a)
        return
    if a.cmd == "universo":
        import universo
        universo.rodar(a.anos, a.atualizar_b3, a.limite)
        return
    cfg = yaml.safe_load(open(a.config, encoding="utf-8"))["grupos"]
    grupos = [a.grupo] if a.grupo else list(cfg)
    cad = carregar_cadastro()
    precisa_b3 = any(cfg[g].get("filtro_b3") for g in grupos if g in cfg)
    b3 = carregar_b3(a.atualizar_b3) if precisa_b3 else None
    anos = anos_dfp_disponiveis()
    print(f"DFPs disponíveis: {anos[0]}–{anos[-1]}")
    for g in grupos:
        if g not in cfg:
            sys.exit(f"Grupo '{g}' não existe em {a.config}")
        coletar_grupo(g, cfg[g], cad, b3, anos, a.anos)


if __name__ == "__main__":
    main()
