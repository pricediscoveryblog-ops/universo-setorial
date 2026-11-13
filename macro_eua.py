"""
macro_eua.py — Price Discovery
Séries dos EUA para as abas "Juros (Treasuries)" e "Macro EUA" do app (modo EUA), na mesma janela da base
brasileira (macro.py) mais um ano extra, usado só como base de cálculo das variações do primeiro ano exibido.

Fonte única: FRED (Federal Reserve Bank of St. Louis), download público em CSV (fredgraph.csv), sem chave de API.
Cada série é republicação da fonte primária:
  Curva nominal — U.S. Treasury, yields de vencimento constante (DGS1MO ... DGS30), % a.a.
  Curva real    — U.S. Treasury, TIPS de vencimento constante (DFII5 ... DFII30), % a.a.
  Fed funds     — Federal Reserve, taxa efetiva (DFF), % a.a.
  CPI           — BLS, CPI-U cheio e núcleo, sem ajuste sazonal (CPIAUCNS, CPILFENS), índice 1982-84 = 100
  PIB           — BEA, PIB real (GDPC1, US$ bi encadeados de 2017, SAAR), PIB nominal (GDP, US$ bi, SAAR) e
                  taxa oficial t/t anualizada (A191RL1Q225SBEA)
  Dólar         — Federal Reserve (H.10): índice amplo nominal do dólar (DTWEXBGS, jan/2006 = 100) e
                  R$ por US$ (DEXBZUS)

Saídas em saidas/eua_macro/: treasury_diario.csv.gz, cpi_mensal.csv, pib_trimestral.csv, dolar_diario.csv,
metadados.json

Regras de robustez (as mesmas do macro.py):
- cada série é baixada separadamente, com novas tentativas; uma fonte fora do ar não derruba as outras;
- o BEA revisa trimestres passados: os arquivos são regravados inteiros a cada execução;
- se a série nova encolher (>5% menor que a anterior) ou vier implausível, o arquivo anterior é mantido;
- meses sem CPI publicado (out/2025, shutdown do governo: o BLS não divulgou) ficam com o índice interpolado
  geometricamente entre os meses vizinhos e marcados na coluna 'interpolado'.
"""
import io
import json
import sys
import time
from datetime import date, datetime
from pathlib import Path

import numpy as np
import pandas as pd
import requests

ANOS = 11  # mesma janela do macro.py (hoje: 2015 em diante, coletando desde 2014)
SAIDA = Path("saidas/eua_macro")
FRED = "https://fred.stlouisfed.org/graph/fredgraph.csv"
CAB = {"User-Agent": "Mozilla/5.0 (compatible; pricediscovery-bot)", "Accept": "text/csv"}

# vértice (anos) -> série FRED
NOMINAL = {1 / 12: "DGS1MO", 0.25: "DGS3MO", 0.5: "DGS6MO", 1: "DGS1", 2: "DGS2", 3: "DGS3", 5: "DGS5",
           7: "DGS7", 10: "DGS10", 20: "DGS20", 30: "DGS30"}
REAL = {5: "DFII5", 7: "DFII7", 10: "DFII10", 20: "DFII20", 30: "DFII30"}


def rotulo_vertice(anos: float) -> str:
    """Nome de coluna do vértice: 1m, 3m, 6m, 1a, 2a ..."""
    return f"{round(anos * 12)}m" if anos < 1 else f"{anos:g}a"


def inicio_coleta(hoje: date) -> date:
    return date(hoje.year - ANOS - 1, 1, 1)


def fred(serie: str, ini: date, tentativas: int = 4) -> pd.Series:
    """Série FRED como pd.Series (índice = data). Valores ausentes ('.' ou vazio) viram NaN."""
    ult = None
    for i in range(tentativas):
        try:
            r = requests.get(FRED, params={"id": serie, "cosd": ini.isoformat()}, headers=CAB, timeout=90)
            if r.status_code == 200 and r.text.startswith("observation_date"):
                df = pd.read_csv(io.StringIO(r.text), na_values=[".", ""])
                s = pd.Series(pd.to_numeric(df.iloc[:, 1], errors="coerce").to_numpy(),
                              index=pd.to_datetime(df.iloc[:, 0]), name=serie)
                if s.notna().sum() == 0:
                    raise ValueError(f"{serie} sem valores")
                return s.sort_index()
            ult = f"HTTP {r.status_code}: {r.text[:120]!r}"
        except (requests.RequestException, ValueError) as e:
            ult = str(e)
        time.sleep(5 * (i + 1))
    raise RuntimeError(f"FRED {serie} falhou após {tentativas} tentativas ({ult})")


# ----------------------------------------------------------------------------- séries
def coletar_treasury(ini: date) -> pd.DataFrame:
    """Curva nominal e real por pregão. Colunas: nom_<vértice>, real_<vértice>, fed_funds (% a.a.)."""
    cols = {}
    for anos, s in NOMINAL.items():
        cols[f"nom_{rotulo_vertice(anos)}"] = fred(s, ini)
    for anos, s in REAL.items():
        cols[f"real_{rotulo_vertice(anos)}"] = fred(s, ini)
    df = pd.DataFrame(cols)
    # pregão = dia com a curva nominal publicada (feriados de títulos vêm vazios no FRED)
    nom = [c for c in df.columns if c.startswith("nom_")]
    df = df[df[nom].notna().sum(axis=1) >= len(nom) - 2]
    ff = fred("DFF", ini)  # diária, inclusive fins de semana: alinha no último valor até o pregão
    df["fed_funds"] = ff.reindex(df.index, method="ffill")
    if df.empty or not df[nom].stack().between(-2, 25).all():
        raise ValueError("curva nominal vazia ou fora da faixa plausível (−2% a 25%)")
    rr = [c for c in df.columns if c.startswith("real_")]
    if not df[rr].stack().between(-5, 15).all():
        raise ValueError("curva real fora da faixa plausível (−5% a 15%)")
    df.index.name = "data"
    return df.reset_index().assign(data=lambda x: x["data"].dt.date)


def interpolar_mensal(s: pd.Series) -> tuple[pd.Series, pd.Series]:
    """Completa meses sem dado com interpolação geométrica (log-linear) do índice. Devolve (série, máscara)."""
    s = s.copy()
    s.index = s.index.to_period("M")
    s = s.reindex(pd.period_range(s.index.min(), s.index.max(), freq="M"))
    # meses vazios só no fim (ainda não divulgados) não entram
    ult = s.last_valid_index()
    s = s[:ult]
    falta = s.isna()
    if falta.any():
        s = np.exp(np.log(s).interpolate(limit_area="inside"))
    return s, falta


def coletar_cpi(ini: date) -> pd.DataFrame:
    cheio, f1 = interpolar_mensal(fred("CPIAUCNS", ini))
    nucleo, f2 = interpolar_mensal(fred("CPILFENS", ini))
    df = pd.DataFrame({"cpi_indice": cheio, "core_indice": nucleo})
    df["interpolado"] = f1.reindex(df.index, fill_value=False) | f2.reindex(df.index, fill_value=False)
    df = df.dropna(subset=["cpi_indice"])
    if (df["interpolado"].rolling(3).sum() >= 3).any():
        raise ValueError("CPI com 3+ meses seguidos sem dado: não interpola")
    var = df["cpi_indice"].pct_change().dropna() * 100
    if df.empty or not df["cpi_indice"].between(100, 1000).all() or not var.between(-5, 5).all():
        raise ValueError("CPI vazio ou implausível")
    df.index.name = "mes"
    out = df.reset_index()
    out["mes"] = out["mes"].astype(str)
    return out.round({"cpi_indice": 3, "core_indice": 3})


def coletar_pib(ini: date) -> pd.DataFrame:
    real, nominal, saar = fred("GDPC1", ini), fred("GDP", ini), fred("A191RL1Q225SBEA", ini)
    df = pd.DataFrame({"pib_real": real, "pib_nominal": nominal, "pib_real_saar_bea": saar}).dropna(
        subset=["pib_real", "pib_nominal"])
    per = df.index.to_period("Q")
    df.insert(0, "ano", per.year)
    df.insert(1, "tri", per.quarter)
    df.insert(2, "trimestre", [f"{q}T{str(a)[2:]}" for a, q in zip(per.year, per.quarter)])
    esperado = (df["ano"].iloc[-1] - df["ano"].iloc[0]) * 4 + df["tri"].iloc[-1] - df["tri"].iloc[0] + 1
    if df.empty or len(df) != esperado:
        raise ValueError("PIB vazio ou com trimestres faltando")
    if not df["pib_real"].between(5_000, 100_000).all() or not df["pib_nominal"].between(5_000, 200_000).all():
        raise ValueError("PIB fora da faixa plausível (US$ bi)")
    return df.reset_index(drop=True)


def coletar_dolar(ini: date) -> pd.DataFrame:
    df = pd.DataFrame({"dolar_amplo": fred("DTWEXBGS", ini), "brl_usd": fred("DEXBZUS", ini)})
    df = df.dropna(how="all")
    if df.empty or not df["dolar_amplo"].dropna().between(50, 250).all() \
            or not df["brl_usd"].dropna().between(0.5, 20).all():
        raise ValueError("dólar vazio ou fora da faixa plausível")
    df.index.name = "data"
    return df.reset_index().assign(data=lambda x: x["data"].dt.date)


# ----------------------------------------------------------------------------- gravação
def gravar(nome: str, df: pd.DataFrame, chave: str) -> str:
    """Grava se a série nova não encolher (>5% menor que a anterior indica resposta parcial)."""
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
        "treasury_diario.csv.gz": (lambda: coletar_treasury(ini), "data"),
        "cpi_mensal.csv": (lambda: coletar_cpi(ini), "mes"),
        "pib_trimestral.csv": (lambda: coletar_pib(ini), "trimestre"),
        "dolar_diario.csv": (lambda: coletar_dolar(ini), "data"),
    }
    meta_p = SAIDA / "metadados.json"
    meta = json.loads(meta_p.read_text()) if meta_p.exists() else {}
    falhas_sem_arquivo = 0
    for nome, (fn, chave) in tarefas.items():
        try:
            msg = gravar(nome, fn(), chave)
            if msg.startswith("ok"):
                meta[nome] = {"atualizado_em": datetime.now().isoformat(timespec="seconds"), "status": msg}
        except Exception as e:  # noqa: BLE001 — uma fonte fora do ar não derruba as outras
            msg = f"FALHOU: {e}"
            if not (SAIDA / nome).exists():
                falhas_sem_arquivo += 1
        print(f"{nome}: {msg}")
    meta["janela"] = {"inicio_coleta": ini.isoformat(), "inicio_exibicao": f"{ini.year + 1}-01-01", "anos": ANOS}
    meta_p.write_text(json.dumps(meta, ensure_ascii=False, indent=2))
    return 1 if falhas_sem_arquivo else 0


if __name__ == "__main__":
    sys.exit(main())
