"""
macro.py — Price Discovery
Séries macro do Brasil para a aba "Macro" do app, cobrindo a mesma janela da base de empresas (10 anos)
mais um ano extra, usado só como base de cálculo (variação em 12 meses e anual do primeiro ano exibido).

Fontes:
  IPCA   — Banco Central, SGS série 433 (variação % mensal; o IBGE é a fonte primária)
  PIB    — IBGE, SIDRA tabela 5932 (taxas de variação do índice de volume trimestral, PIB a preços de mercado)
  Câmbio — Banco Central, SGS série 1 (PTAX venda, R$ por US$, diária)

Saídas em saidas/macro/: ipca_mensal.csv, pib_trimestral.csv, cambio_diario.csv, metadados.json

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

ANOS = 10                                   # mesma janela da base de empresas
SAIDA = Path("saidas/macro")
CAB = {"User-Agent": "Mozilla/5.0 (compatible; pricediscovery-bot)", "Accept": "application/json"}
SGS = "https://api.bcb.gov.br/dados/serie/bcdata.sgs.{cod}/dados"
SIDRA_PIB = "https://apisidra.ibge.gov.br/values/t/5932/n1/all/v/all/p/all/c11255/90707"


def inicio_coleta(hoje: date) -> date:
    """1º de janeiro do ano anterior à janela exibida (ex.: em 2026 exibe 2016+ e coleta 2015+)."""
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


def coletar_cambio(ini: date, fim: date) -> pd.DataFrame:
    df = sgs(1, ini, fim).rename(columns={"valor": "ptax_venda"})
    if df.empty or not df["ptax_venda"].between(0.5, 20).all():
        raise ValueError("PTAX vazia ou fora da faixa plausível")
    df["data"] = df["data"].dt.date
    return df


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
        "cambio_diario.csv": (lambda: coletar_cambio(ini, hoje), "data"),
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
