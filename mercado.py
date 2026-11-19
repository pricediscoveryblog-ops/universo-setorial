"""
mercado.py — Price Discovery Scanning
Dados de mercado complementares para o app (saidas/mercado/):

1) proventos.parquet — proventos e eventos societários de cada empresa listada, da página pública de empresas
   listadas da B3 (GetListedSupplementCompany):
     - dinheiro: dividendos, JCP (valor BRUTO por ação, antes do IR) e rendimentos, por ISIN e data-com;
     - ações: desdobramentos, grupamentos e bonificações (fator informado pela B3), por ISIN e data-com.
   Os campos originais da B3 ficam gravados (colunas *_b3) ao lado dos convertidos.
2) acionistas.parquet — posição acionária do Formulário de Referência (CVM Dados Abertos, item 15.1), versão mais
   recente de cada companhia, com a cadeia de controle (ID do acionista e do acionista relacionado) e quantidades
   ON/PN; distribuicao_capital.parquet — ações em circulação (free float) informadas no FRE.

Uso: python mercado.py [proventos|acionistas|tudo]   (padrão: tudo)
"""
import sys
import time
import zipfile
from datetime import date
from pathlib import Path

import numpy as np
import pandas as pd

import coletor_setorial as cs

SAIDA = Path("saidas/mercado")
FRE = "https://dados.cvm.gov.br/dados/CIA_ABERTA/DOC/FRE/DADOS/fre_cia_aberta_{ano}.zip"


def _num(s) -> float:
    """'1.234,5678' / '0,12345678' / '100.00' -> float."""
    if s is None:
        return np.nan
    t = str(s).strip()
    if not t:
        return np.nan
    if "," in t:
        t = t.replace(".", "").replace(",", ".")
    try:
        return float(t)
    except ValueError:
        return np.nan


def _data(s):
    return pd.to_datetime(str(s or "").strip()[:10], format="%d/%m/%Y", errors="coerce")


def raizes_listadas() -> pd.DataFrame:
    """Empresas com ação negociada em bolsa (cache da lista da B3 gerado pelo coletor setorial)."""
    b3 = pd.read_csv(cs.B3_CACHE, sep=";", dtype=str).fillna("")
    b3 = b3[(b3["tickers"] != "") & (b3["issuingCompany"] != "")]
    b3["cnpj"] = b3["cnpj"].map(cs.so_digitos)
    return b3.drop_duplicates("issuingCompany")[["issuingCompany", "cnpj", "tradingName", "tickers"]]


# ------------------------------------------------------------------ 1) proventos (B3)
def proventos() -> pd.DataFrame:
    emp = raizes_listadas()
    linhas, falhas = [], []
    print(f"proventos: {len(emp)} emissores", flush=True)
    for i, r in enumerate(emp.itertuples(), 1):
        d = cs._b3_get("GetListedSupplementCompany", {"issuingCompany": r.issuingCompany, "language": "pt-br"})
        if isinstance(d, list):
            d = d[0] if d else {}
        if not isinstance(d, dict):
            falhas.append(r.issuingCompany)
            continue
        for p in d.get("cashDividends") or []:
            linhas.append({"raiz": r.issuingCompany, "cnpj": r.cnpj, "tipo": "dinheiro",
                           "evento": str(p.get("label") or "").strip().upper(),
                           "isin": str(p.get("isinCode") or "").strip(),
                           "classe_b3": str(p.get("assetIssued") or "").strip(),
                           "data_com": _data(p.get("lastDatePrior")), "data_aprovacao": _data(p.get("approvedOn")),
                           "data_pagamento": _data(p.get("paymentDate")),
                           "valor": _num(p.get("rate")), "fator_b3": np.nan,
                           "referente": str(p.get("relatedTo") or "").strip(),
                           "valor_b3": str(p.get("rate") or ""), "fator_b3_txt": "",
                           "obs": str(p.get("remarks") or "").strip()})
        for p in d.get("stockDividends") or []:
            linhas.append({"raiz": r.issuingCompany, "cnpj": r.cnpj, "tipo": "acoes",
                           "evento": str(p.get("label") or "").strip().upper(),
                           "isin": str(p.get("isinCode") or "").strip(),
                           "classe_b3": str(p.get("assetIssued") or "").strip(),
                           "data_com": _data(p.get("lastDatePrior")), "data_aprovacao": _data(p.get("approvedOn")),
                           "data_pagamento": pd.NaT, "valor": np.nan, "fator_b3": _num(p.get("factor")),
                           "referente": "", "valor_b3": "", "fator_b3_txt": str(p.get("factor") or ""),
                           "obs": str(p.get("remarks") or "").strip()})
        if i % 50 == 0:
            print(f"  {i}/{len(emp)} ({len(linhas)} eventos)", flush=True)
        time.sleep(0.25)
    df = pd.DataFrame(linhas)
    if df.empty:
        raise SystemExit("B3 não devolveu proventos: arquivo anterior mantido")
    df = df.drop_duplicates(["raiz", "tipo", "evento", "isin", "data_com", "valor", "fator_b3"])
    df = df.sort_values(["raiz", "data_com", "isin"]).reset_index(drop=True)
    print(f"  -> {len(df)} eventos ({(df.tipo == 'dinheiro').sum()} em dinheiro, {(df.tipo == 'acoes').sum()} em "
          f"ações); falhas: {len(falhas)} {falhas[:15]}")
    print(df[df.tipo == "acoes"].groupby("evento")["fator_b3"].describe().to_string())
    return df


# ------------------------------------------------------------------ 2) acionistas (FRE)
def _col(df, *ps, ex=()):
    return next((c for c in df.columns if all(p in cs.norm(c) for p in ps) and not any(x in cs.norm(c) for x in ex)),
                None)


def _ultima_versao(df: pd.DataFrame) -> pd.DataFrame:
    c_dt, c_ver = _col(df, "data", "referencia"), _col(df, "versao")
    if c_dt and c_ver:
        df = df.assign(_v=pd.to_numeric(df[c_ver], errors="coerce"))
        df = df[df[c_dt] == df.groupby("_cnpj")[c_dt].transform("max")]
        df = df[df["_v"] == df.groupby("_cnpj")["_v"].transform("max")].drop(columns="_v")
    return df


def acionistas() -> tuple[pd.DataFrame, pd.DataFrame]:
    cnpjs = set(raizes_listadas()["cnpj"])
    pos, dist = [], []
    hoje = date.today().year
    for ano in (hoje, hoje - 1, hoje - 2):
        try:
            p = cs.baixar(FRE.format(ano=ano), cs.CACHE / f"fre_cia_aberta_{ano}.zip", forcar=ano >= hoje - 1)
        except Exception as e:  # noqa: BLE001
            print(f"  ! FRE {ano}: {e}")
            continue
        with zipfile.ZipFile(p) as z:
            for nome in z.namelist():
                n = nome.lower()
                alvo = pos if ("posicao_acionaria" in n and "classe" not in n) else \
                    dist if "distribuicao_capital" in n else None
                if alvo is None:
                    continue
                df = pd.read_csv(z.open(nome), sep=";", encoding="latin-1", dtype=str)
                c_cia = _col(df, "cnpj", "companhia")
                if not c_cia:
                    print(f"  ! {nome}: sem coluna de CNPJ ({list(df.columns)[:8]})")
                    continue
                df["_cnpj"] = df[c_cia].map(cs.so_digitos)
                df = df[df["_cnpj"].isin(cnpjs)]
                if len(df):
                    alvo.append(_ultima_versao(df).assign(ano_fre=ano))
                print(f"  FRE {ano} {nome}: {len(df)} linhas das listadas", flush=True)
    out = []
    for partes in (pos, dist):
        if not partes:
            out.append(pd.DataFrame())
            continue
        a = pd.concat(partes, ignore_index=True)
        a = a[a["ano_fre"] == a.groupby("_cnpj")["ano_fre"].transform("max")]   # FRE mais recente de cada uma
        out.append(a.rename(columns={"_cnpj": "cnpj"}).astype(str))
    print(f"  -> acionistas: {len(out[0])} linhas, {out[0]['cnpj'].nunique() if len(out[0]) else 0} empresas; "
          f"distribuição do capital: {len(out[1])} linhas")
    return out[0], out[1]


def main(qual: str = "tudo"):
    SAIDA.mkdir(parents=True, exist_ok=True)
    if qual in ("proventos", "tudo"):
        try:
            proventos().to_parquet(SAIDA / "proventos.parquet", index=False)
        except SystemExit as e:
            print(f"  ! {e}")
    if qual in ("acionistas", "tudo"):
        a, d = acionistas()
        if len(a):
            a.to_parquet(SAIDA / "acionistas.parquet", index=False)
        if len(d):
            d.to_parquet(SAIDA / "distribuicao_capital.parquet", index=False)


if __name__ == "__main__":
    main(sys.argv[1] if len(sys.argv) > 1 else "tudo")
