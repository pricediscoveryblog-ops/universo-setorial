"""
aluguel.py — Price Discovery
Resumo diário do aluguel de ações (custo do short) a partir dos arquivos do repositório coletor-aluguel-b3
(B3: empréstimos registrados e posições em aberto). Uma linha por ticker e dia.

Uso: python aluguel.py <pasta dados do coletor-aluguel-b3>
Saída: saidas/aluguel/resumo.csv.gz
"""
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ORIG = Path(sys.argv[1] if len(sys.argv) > 1 else "aluguel_repo/dados")
SAIDA = Path("saidas/aluguel")


def ler(pasta: str) -> pd.DataFrame:
    arqs = sorted((ORIG / pasta).rglob("*.parquet"))
    if not arqs:
        print(f"  ! nenhum arquivo em {ORIG / pasta}")
        return pd.DataFrame()
    df = pd.concat([pd.read_parquet(a).assign(_data_arquivo=a.stem) for a in arqs], ignore_index=True)
    print(f"  {pasta}: {len(arqs)} arquivos, {len(df)} linhas")
    return df


def num(s: pd.Series) -> pd.Series:
    if s.dtype == object:
        s = s.astype(str).str.replace(".", "", regex=False).str.replace(",", ".", regex=False) \
            if s.astype(str).str.contains(",").any() else s
    return pd.to_numeric(s, errors="coerce")


def data_ref(df: pd.DataFrame) -> pd.Series:
    """Data do pregão: a do nome do arquivo gravado pelo coletor (AAAA-MM-DD.parquet). O campo RptDt vem em
    formato variável entre arquivos; só é usado (lido como dia/mês/ano) quando o nome não tem data."""
    f = pd.to_datetime(df["_data_arquivo"], errors="coerce", format="%Y-%m-%d")
    if "RptDt" in df:
        d = pd.to_datetime(df["RptDt"].astype(str), errors="coerce", format="mixed", dayfirst=True)
        f = f.fillna(d)
    return f


def resumo() -> pd.DataFrame:
    emp = ler("emprestimos_registrados")
    pos = ler("posicoes_em_aberto")
    partes = []
    if not emp.empty:
        e = pd.DataFrame({"data": data_ref(emp),
                          "ticker": emp["TckrSymb"].astype(str).str.strip()})
        for c in ("ValCtrctsDay", "QtyCtrctsDay", "BRLValue", "DnrAvrgRate", "TkrAvrgRate", "DnrMinRate", "DnrMaxRate",
                  "TkrMinRate", "TkrMaxRate"):
            e[c] = num(emp[c]) if c in emp.columns else np.nan
        # vários mercados (eletrônico/balcão) no mesmo dia: taxa média ponderada pela quantidade de ativos
        e["peso"] = e["ValCtrctsDay"].fillna(0)
        e["tk_x"] = e["TkrAvrgRate"] * e["peso"]
        e["dn_x"] = e["DnrAvrgRate"] * e["peso"]
        g = e.groupby(["data", "ticker"]).agg(
            peso=("peso", "sum"), tk_x=("tk_x", "sum"), dn_x=("dn_x", "sum"),
            taxa_tomador_min=("TkrMinRate", "min"), taxa_tomador_max=("TkrMaxRate", "max"),
            taxa_doador_min=("DnrMinRate", "min"), taxa_doador_max=("DnrMaxRate", "max"),
            qtd_ativos_contratados_dia=("ValCtrctsDay", "sum"), contratos_dia=("QtyCtrctsDay", "sum"),
            valor_contratado_dia=("BRLValue", "sum"),
            tk_simples=("TkrAvrgRate", "mean"), dn_simples=("DnrAvrgRate", "mean"))
        g["taxa_tomador_media"] = np.where(g["peso"] > 0, g["tk_x"] / g["peso"].where(g["peso"] > 0), g["tk_simples"])
        g["taxa_doador_media"] = np.where(g["peso"] > 0, g["dn_x"] / g["peso"].where(g["peso"] > 0), g["dn_simples"])
        partes.append(g.drop(columns=["peso", "tk_x", "dn_x", "tk_simples", "dn_simples"]))
    if not pos.empty:
        p = pd.DataFrame({"data": data_ref(pos),
                          "ticker": pos["TckrSymb"].astype(str).str.strip(),
                          "saldo_alugado_qtd": num(pos["StockBalance"]) if "StockBalance" in pos else np.nan,
                          "saldo_alugado_rs": num(pos["Balance"]) if "Balance" in pos else np.nan})
        partes.append(p.groupby(["data", "ticker"])[["saldo_alugado_qtd", "saldo_alugado_rs"]].sum(min_count=1))
    if not partes:
        return pd.DataFrame()
    out = partes[0].join(partes[1], how="outer") if len(partes) == 2 else partes[0]
    return out.reset_index().sort_values(["ticker", "data"])


if __name__ == "__main__":
    r = resumo()
    if r.empty:
        sys.exit("Nenhum dado de aluguel encontrado.")
    SAIDA.mkdir(parents=True, exist_ok=True)
    r.to_csv(SAIDA / "resumo.csv.gz", index=False, compression="gzip")
    print(f"  -> {SAIDA}/resumo.csv.gz ({len(r)} linhas, {r['ticker'].nunique()} tickers, "
          f"{r['data'].min().date()} a {r['data'].max().date()})")
