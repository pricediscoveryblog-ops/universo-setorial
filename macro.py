"""
macro.py — Price Discovery
Séries macro do Brasil para a aba "Macro" do app, cobrindo a mesma janela da base de empresas
mais um ano extra, usado só como base de cálculo (variação em 12 meses e anual do primeiro ano exibido).

Fontes:
  IPCA   — Banco Central, SGS série 433 (variação % mensal; o IBGE é a fonte primária)
  PIB    — IBGE, SIDRA tabela 5932 (taxas de variação do índice de volume trimestral, PIB a preços de mercado)
           e tabela 1620 (índice de volume trimestral sem ajuste sazonal, média 1995 = 100: nível para o PIB nominal)
  Câmbio — Banco Central, SGS série 1 (PTAX venda, R$ por US$, diária)
  Focus  — Banco Central, Sistema Expectativas de Mercado (API Olinda, ExpectativasMercadoAnuais, base de cálculo 0
           = todos os respondentes): estatísticas das projeções anuais, reduzidas a um registro por semana
           (último dia da semana encerrada na sexta, que é o dado do relatório Focus publicado na segunda seguinte)

Saídas em saidas/macro/: ipca_mensal.csv, pib_trimestral.csv, pib_volume_trimestral.csv, cambio_diario.csv,
focus_anual.csv, metadados.json

Regras de robustez:
  - a API do SGS recusa janelas acima de 10 anos: as consultas são fatiadas em janelas de até 3 anos;
  - o IBGE revisa trimestres passados do PIB: a série é regravada inteira a cada execução;
  - se uma fonte falhar ou devolver algo implausível, o arquivo anterior é mantido.
"""
import json
import sys
import time
from datetime import date, datetime, timedelta
from pathlib import Path

import pandas as pd
import requests
from urllib.parse import quote

ANOS = 11                                   # mesma janela da base de empresas (hoje: 2015 em diante)
SAIDA = Path("saidas/macro")
CAB = {"User-Agent": "Mozilla/5.0 (compatible; pricediscovery-bot)", "Accept": "application/json"}
SGS = "https://api.bcb.gov.br/dados/serie/bcdata.sgs.{cod}/dados"
SIDRA_PIB = "https://apisidra.ibge.gov.br/values/t/5932/n1/all/v/all/p/all/c11255/90707"
FOCUS = "https://olinda.bcb.gov.br/olinda/servico/Expectativas/versao/v1/odata/ExpectativasMercadoAnuais"
FOCUS_INDICADORES = ["IPCA", "IGP-M", "IPCA Administrados", "PIB Total", "Câmbio", "Selic",
                     "Resultado primário", "Dívida líquida do setor público"]
SIDRA_VOL = "https://apisidra.ibge.gov.br/values/t/1620/n1/all/v/583/p/all/c11255/90707"


def inicio_coleta(hoje: date) -> date:
    """1º de janeiro do ano anterior à janela exibida (ex.: em 2026 exibe 2015+ e coleta 2014+)."""
    return date(hoje.year - ANOS - 1, 1, 1)


def get_json(url: str, params: dict | None = None, tentativas: int = 4):
    ult = None
    for i in range(tentativas):
        try:
            r = requests.get(url, params=params, headers=CAB, timeout=90)
            if r.status_code == 200:
                return r.json()
            ult = f"HTTP {r.status_code}: {r.text[:150]}"
        except (requests.RequestException, ValueError) as e:
            ult = str(e)
        time.sleep(5 * (i + 1))
    raise RuntimeError(f"{url} falhou após {tentativas} tentativas ({ult})")


def janelas(ini: date, fim: date, anos: int = 3):
    a = ini
    while a <= fim:
        b = min(date(a.year + anos, 1, 1) - timedelta(days=1), fim)
        yield a, b
        a = b + timedelta(days=1)


def sgs(cod: int, ini: date, fim: date) -> pd.DataFrame:
    partes = []
    for a, b in janelas(ini, fim):
        dados = get_json(SGS.format(cod=cod), {"formato": "json", "dataInicial": a.strftime("%d/%m/%Y"),
                                               "dataFinal": b.strftime("%d/%m/%Y")})
        if isinstance(dados, list) and dados:
            partes.append(pd.DataFrame(dados))
    if not partes:
        return pd.DataFrame(columns=["data", "valor"])
    df = pd.concat(partes, ignore_index=True)
    df["data"] = pd.to_datetime(df["data"], format="%d/%m/%Y")
    df["valor"] = pd.to_numeric(df["valor"], errors="coerce")
    return df.dropna().drop_duplicates("data").sort_values("data").reset_index(drop=True)


# ----------------------------------------------------------------------------- séries
def coletar_ipca(ini: date, fim: date) -> pd.DataFrame:
    df = sgs(433, ini, fim).rename(columns={"valor": "ipca_mensal"})
    df["mes"] = df["data"].dt.to_period("M").astype(str)
    df = df[["mes", "ipca_mensal"]]
    meses = pd.period_range(df["mes"].iloc[0], df["mes"].iloc[-1], freq="M").astype(str)
    if len(meses) != len(df):
        raise ValueError(f"IPCA com meses faltando ({len(df)} de {len(meses)})")
    if not df["ipca_mensal"].between(-5, 10).all():
        raise ValueError("IPCA mensal fora da faixa plausível (−5% a 10%)")
    return df


MAPA_PIB = [("imediatamente anterior", "pib_tri_vs_tri_anterior_dessaz"),
            ("quatro trimestres", "pib_acum_4_tri"),
            ("ao longo do ano", "pib_acum_no_ano"),
            ("ano anterior", "pib_tri_vs_mesmo_tri_ano_anterior")]   # por último: as outras também citam "ano"


def parse_sidra_pib(dados: list) -> pd.DataFrame:
    cab, linhas = dados[0], dados[1:]
    k_var = next(k for k, v in cab.items() if v == "Variável")
    k_per = next(k for k, v in cab.items() if v.startswith("Trimestre (Código)"))
    reg = []
    for l in linhas:
        nome = l[k_var].lower()
        col = next((c for chave, c in MAPA_PIB if chave in nome), None)
        if col:
            reg.append({"cod": l[k_per], "col": col, "v": pd.to_numeric(l["V"], errors="coerce")})
    if not reg:
        raise ValueError("SIDRA 5932 sem as variáveis esperadas (o IBGE pode ter mudado os nomes)")
    df = pd.DataFrame(reg).pivot_table(index="cod", columns="col", values="v", aggfunc="first").reset_index()
    df["ano"] = df["cod"].str[:4].astype(int)
    df["tri"] = df["cod"].str[-2:].astype(int)
    df["trimestre"] = df["tri"].astype(str) + "T" + df["ano"].astype(str).str[2:]
    cols = [c for _, c in MAPA_PIB if c in df.columns]
    return df[["ano", "tri", "trimestre"] + cols].sort_values(["ano", "tri"]).reset_index(drop=True)


def coletar_pib(ini: date) -> pd.DataFrame:
    df = parse_sidra_pib(get_json(SIDRA_PIB))
    df = df[df["ano"] >= ini.year].reset_index(drop=True)
    if df.empty or df.drop(columns=["ano", "tri", "trimestre"]).abs().gt(30).any().any():
        raise ValueError("PIB vazio ou com taxas implausíveis (>30%)")
    return df


def parse_sidra_volume(dados: list) -> pd.DataFrame:
    cab, linhas = dados[0], dados[1:]
    k_per = next(k for k, v in cab.items() if v.startswith("Trimestre (Código)"))
    df = pd.DataFrame({"cod": [l[k_per] for l in linhas],
                       "indice_volume": pd.to_numeric([l["V"] for l in linhas], errors="coerce")}).dropna()
    df["ano"] = df["cod"].str[:4].astype(int)
    df["tri"] = df["cod"].str[-2:].astype(int)
    df["trimestre"] = df["tri"].astype(str) + "T" + df["ano"].astype(str).str[2:]
    return df[["ano", "tri", "trimestre", "indice_volume"]].sort_values(["ano", "tri"]).reset_index(drop=True)


def coletar_pib_volume(ini: date) -> pd.DataFrame:
    df = parse_sidra_volume(get_json(SIDRA_VOL))
    df = df[df["ano"] >= ini.year].reset_index(drop=True)
    if df.empty or not df["indice_volume"].between(50, 1000).all():
        raise ValueError("índice de volume do PIB vazio ou implausível")
    if len(df) != (df["ano"].iloc[-1] - df["ano"].iloc[0]) * 4 + df["tri"].iloc[-1] - df["tri"].iloc[0] + 1:
        raise ValueError("índice de volume do PIB com trimestres faltando")
    return df


def coletar_cambio(ini: date, fim: date) -> pd.DataFrame:
    df = sgs(1, ini, fim).rename(columns={"valor": "ptax_venda"})
    if df.empty or not df["ptax_venda"].between(0.5, 20).all():
        raise ValueError("PTAX vazia ou fora da faixa plausível")
    df["data"] = df["data"].dt.date
    return df


def _focus_pagina(indicador: str, ini: date, fim: date) -> list:
    regs, pulo, passo = [], 0, 10000
    filtro = (f"Indicador eq '{indicador}' and baseCalculo eq 0 and "
              f"Data ge '{ini.isoformat()}' and Data le '{fim.isoformat()}'")
    while True:
        q = {"$format": "json", "$filter": filtro, "$orderby": "Data", "$top": str(passo), "$skip": str(pulo),
             "$select": "Indicador,IndicadorDetalhe,Data,DataReferencia,Media,Mediana,"
                        "DesvioPadrao,Minimo,Maximo,numeroRespondentes"}
        # a API OData do BCB quer o "$" literal e espaços como %20: a URL é montada aqui, não pelo requests
        dados = get_json(FOCUS + "?" + "&".join(f"{k}={quote(v, safe=chr(39) + ',')}" for k, v in q.items()))
        lote = dados.get("value", []) if isinstance(dados, dict) else []
        regs += lote
        if len(lote) < passo:
            return regs
        pulo += passo


def semanal_focus(df: pd.DataFrame) -> pd.DataFrame:
    """Um registro por semana (semana encerrada na sexta): o último dia com dado, que é o do relatório Focus."""
    sem = df["data"].dt.to_period("W-FRI")
    ult = df.groupby(sem)["data"].transform("max")
    return df[df["data"] == ult].sort_values(["data", "indicador", "ano_ref"]).reset_index(drop=True)


def coletar_focus(ini: date, fim: date) -> pd.DataFrame:
    """Expectativas anuais do Focus. Incremental: com arquivo anterior, relê só as últimas 3 semanas e junta."""
    p = SAIDA / "focus_anual.csv"
    ant = pd.read_csv(p, parse_dates=["data"]) if p.exists() else pd.DataFrame()
    desde = (ant["data"].max().date() - timedelta(days=21)) if len(ant) else ini
    regs, faltando = [], []
    for ind in FOCUS_INDICADORES:
        n0 = len(regs)
        for a, b in janelas(desde, fim):
            regs += _focus_pagina(ind, a, b)
        if len(regs) == n0:
            faltando.append(ind)
    if faltando:
        print(f"  Focus: sem dados para {faltando} (nome do indicador pode ter mudado na API)")
    if not regs:
        raise ValueError("Focus: a API não devolveu registros")
    d = pd.DataFrame(regs)
    det = d["IndicadorDetalhe"].fillna("").astype(str).str.strip() if "IndicadorDetalhe" in d else ""
    novo = pd.DataFrame({
        "data": pd.to_datetime(d["Data"]),
        "indicador": d["Indicador"].where(det == "", d["Indicador"] + " — " + det),
        "ano_ref": pd.to_numeric(d["DataReferencia"], errors="coerce"),
        "mediana": pd.to_numeric(d["Mediana"], errors="coerce"),
        "media": pd.to_numeric(d["Media"], errors="coerce"),
        "desvio_padrao": pd.to_numeric(d["DesvioPadrao"], errors="coerce"),
        "minimo": pd.to_numeric(d["Minimo"], errors="coerce"),
        "maximo": pd.to_numeric(d["Maximo"], errors="coerce"),
        "respondentes": pd.to_numeric(d["numeroRespondentes"], errors="coerce"),
    }).dropna(subset=["ano_ref", "mediana"])
    novo["ano_ref"] = novo["ano_ref"].astype(int)
    if len(ant):
        ant = ant[ant["data"] < pd.Timestamp(desde)]
        novo = pd.concat([ant, novo], ignore_index=True)
    novo = novo.drop_duplicates(["data", "indicador", "ano_ref"], keep="last")
    out = semanal_focus(novo)
    out["data"] = out["data"].dt.date
    return out


# ----------------------------------------------------------------------------- gravação
def gravar(nome: str, df: pd.DataFrame, chave: str) -> str:
    """Grava se a série nova não encolher (>5% menor que a anterior indica resposta parcial da API)."""
    p = SAIDA / nome
    if p.exists():
        ant = pd.read_csv(p)
        if len(df) < 0.95 * len(ant):
            return f"mantido o arquivo anterior ({len(df)} linhas novas < {len(ant)} anteriores)"
    df.to_csv(p, index=False)
    return f"ok, {len(df)} linhas, último {chave} = {df[chave].iloc[-1]}"


def main() -> int:
    SAIDA.mkdir(parents=True, exist_ok=True)
    hoje = date.today()
    ini = inicio_coleta(hoje)
    tarefas = {
        "ipca_mensal.csv": (lambda: coletar_ipca(ini, hoje), "mes"),
        "pib_trimestral.csv": (lambda: coletar_pib(ini), "trimestre"),
        "pib_volume_trimestral.csv": (lambda: coletar_pib_volume(ini), "trimestre"),
        "cambio_diario.csv": (lambda: coletar_cambio(ini, hoje), "data"),
        "focus_anual.csv": (lambda: coletar_focus(ini, hoje), "data"),
    }
    meta_p = SAIDA / "metadados.json"
    meta = json.loads(meta_p.read_text()) if meta_p.exists() else {}
    falhas_sem_arquivo = 0
    for nome, (fn, chave) in tarefas.items():
        try:
            msg = gravar(nome, fn(), chave)
            if msg.startswith("ok"):
                meta[nome] = {"atualizado_em": datetime.now().isoformat(timespec="seconds"), "status": msg}
        except Exception as e:                     # noqa: BLE001 — uma fonte fora do ar não derruba as outras
            msg = f"FALHOU: {e}"
            if not (SAIDA / nome).exists():
                falhas_sem_arquivo += 1
        print(f"{nome}: {msg}")
    meta["janela"] = {"inicio_coleta": ini.isoformat(), "inicio_exibicao": f"{ini.year + 1}-01-01", "anos": ANOS}
    meta_p.write_text(json.dumps(meta, ensure_ascii=False, indent=2))
    return 1 if falhas_sem_arquivo else 0


if __name__ == "__main__":
    sys.exit(main())
