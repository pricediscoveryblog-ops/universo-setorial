"""
dividas.py — Price Discovery
Debêntures por empresa listada (incluindo as emitidas por controladas), para comparar com a dívida do balanço.

Fontes:
- SND (debentures.com.br): cadastro de todas as debêntures — CNPJ do emissor, emissão, vencimento, quantidade em
  mercado, valor nominal atualizado (VNA), indexador e taxa de emissão.
- ANBIMA (mercado secundário de debêntures, arquivo diário): taxa indicativa, PU, % do PU par e duration.
- CVM, Formulário de Referência (item 9.1.c, participação em sociedades): CNPJs das controladas de cada empresa.

Saídas (saidas/dividas/):
- debentures.csv.gz : uma linha por debênture ativa ligada a uma empresa da base (cnpj do grupo)
- metadados.json
"""
import io
import json
import re
import sys
import zipfile
from datetime import date, timedelta
from pathlib import Path

import numpy as np
import pandas as pd
import requests

SND = ("https://www.debentures.com.br/exploreosnd/consultaadados/emissoesdedebentures/"
       "caracteristicas_e.asp?tip_deb=&ativo=&op_exc=False")
ANBIMA = "https://www.anbima.com.br/informacoes/merc-sec-debentures/arqs/db{d:%y%m%d}.txt"
FRE = "https://dados.cvm.gov.br/dados/CIA_ABERTA/DOC/FRE/DADOS/fre_cia_aberta_{ano}.zip"
SAIDA = Path("saidas/dividas")
CACHE = Path("cache_cvm")
CAB = {"User-Agent": "Mozilla/5.0 (compatible; pricediscovery-bot)"}


def so_digitos(x) -> str:
    return re.sub(r"\D", "", str(x or "")).zfill(14) if re.sub(r"\D", "", str(x or "")) else ""


def num(x) -> float:
    s = str(x or "").strip()
    if not s or s in ("-", "--"):
        return np.nan
    if "," in s:
        s = s.replace(".", "").replace(",", ".")
    try:
        return float(s)
    except ValueError:
        return np.nan


def norm(s) -> str:
    import unicodedata
    return unicodedata.normalize("NFKD", str(s)).encode("ascii", "ignore").decode().lower()


# ------------------------------------------------------------------ SND
def ler_snd() -> pd.DataFrame:
    r = requests.get(SND, headers=CAB, timeout=180)
    r.raise_for_status()
    t = r.content.decode("latin-1")
    linhas = t.splitlines()
    i = next(k for k, l in enumerate(linhas) if l.startswith("Codigo do Ativo"))
    df = pd.read_csv(io.StringIO("\n".join(linhas[i:])), sep="\t", dtype=str)
    df.columns = [c.strip() for c in df.columns]
    col = lambda *ps: next((c for c in df.columns if all(p in norm(c) for p in ps)), None)
    out = pd.DataFrame({
        "codigo": df[col("codigo do ativo")].str.strip(),
        "emissor": df[col("empresa")].str.strip(),
        "cnpj_emissor": df[col("cnpj")].map(so_digitos),
        "situacao": df[col("situacao")].str.strip(),
        "data_emissao": pd.to_datetime(df[col("data de emissao")], format="%d/%m/%Y", errors="coerce"),
        "vencimento": pd.to_datetime(df[col("data de vencimento")], format="%d/%m/%Y", errors="coerce"),
        "quantidade_mercado": pd.to_numeric(df[col("quantidade em mercado")].astype(str).str.replace(r",\d*$", "", regex=True).str.replace(r"[^\d]", "",
                                                                                                   regex=True),
                                            errors="coerce"),          # quantidade é inteira ("150.000" = 150 mil)
        "vna": df[col("valor nominal atual")].map(num),
        "indice": df[col("indice")].str.strip(),
        "percentual_indice": df[col("percentual multiplicador")].map(num),
        "taxa_emissao": df[col("juros criterio novo", "taxa")].map(num),
        "incentivada": df[col("lei 12.431")].str.strip().eq("S"),
    })
    out = out[out["situacao"].str.contains("Registrad", na=False) & (out["quantidade_mercado"] > 0)]
    print(f"  SND: {len(out)} debêntures ativas com quantidade em mercado")
    return out


# ------------------------------------------------------------------ ANBIMA
def ler_anbima(d: date) -> pd.DataFrame | None:
    r = requests.get(ANBIMA.format(d=d), headers=CAB, timeout=60)
    if r.status_code != 200 or b"@" not in r.content:
        return None
    linhas = r.content.decode("latin-1").splitlines()
    i = next((k for k, l in enumerate(linhas) if l.startswith("C") and "@" in l and "Taxa Indicativa" in l), None)
    if i is None:
        return None
    df = pd.read_csv(io.StringIO("\n".join(linhas[i:])), sep="@", dtype=str)
    df.columns = [c.strip() for c in df.columns]
    col = lambda *ps: next((c for c in df.columns if all(p in norm(c) for p in ps)), None)
    return pd.DataFrame({
        "codigo": df[col("codigo")].str.strip(),
        "indice_correcao": df[col("indice")].str.strip(),
        "taxa_indicativa": df[col("taxa indicativa")].map(num),
        "pu": df["PU"].map(num) if "PU" in df.columns else np.nan,
        "pct_pu_par": df[col("pu par")].map(num),
        "duration_du": df[col("duration")].map(num),
        "ref_ntnb": df[col("refer")].str.strip() if col("refer") else "",
        "data_anbima": d,
    })


def anbima_ate(d: date, max_dias: int = 10) -> pd.DataFrame | None:
    """Arquivo da ANBIMA do dia útil mais recente até d."""
    for k in range(max_dias):
        x = d - timedelta(days=k)
        if x.weekday() >= 5:
            continue
        try:
            a = ler_anbima(x)
        except Exception as e:  # noqa: BLE001
            print(f"  ! ANBIMA {x}: {e}")
            a = None
        if a is not None and len(a):
            print(f"  ANBIMA {x}: {len(a)} debêntures com taxa indicativa")
            return a
    return None


# ------------------------------------------------------------------ controladas (FRE 9.1.c)
def ler_controladas(cnpjs: set[str]) -> dict[str, set[str]]:
    """CNPJ da empresa listada -> CNPJs das controladas, do FRE mais recente disponível."""
    grupos: dict[str, set[str]] = {}
    hoje = date.today()
    for ano in (hoje.year, hoje.year - 1):
        p = CACHE / f"fre_cia_aberta_{ano}.zip"
        try:
            import time
            velho = not p.exists() or (ano == hoje.year and time.time() - p.stat().st_mtime > 7 * 86400)
            if velho:                          # FRE do ano corrente: atualiza no máximo uma vez por semana
                CACHE.mkdir(exist_ok=True)
                r = requests.get(FRE.format(ano=ano), headers=CAB, timeout=300)
                r.raise_for_status()
                p.write_bytes(r.content)
            with zipfile.ZipFile(p) as z:
                arq = [n for n in z.namelist() if "participacao_sociedade" in n and "valorizacao" not in n]
                if not arq:
                    print(f"  ! FRE {ano}: arquivo de participações não encontrado")
                    continue
                df = pd.read_csv(z.open(arq[0]), sep=";", encoding="latin-1", dtype=str)
        except Exception as e:  # noqa: BLE001
            print(f"  ! FRE {ano}: {e}")
            continue
        cols = list(df.columns)
        c_cia = next((c for c in cols if "cnpj" in norm(c) and "compan" in norm(c)), None)
        c_soc = next((c for c in cols if "cnpj" in norm(c) and c != c_cia), None)
        c_tipo = next((c for c in cols if "tipo" in norm(c) and "socied" in norm(c)), None)
        c_pct = next((c for c in cols if "particip" in norm(c) and ("%" in c or "percent" in norm(c) or "emissor" in norm(c))), None)
        if not (c_cia and c_soc):
            print(f"  ! FRE {ano}: colunas inesperadas {cols[:15]}")
            continue
        df["_cia"], df["_soc"] = df[c_cia].map(so_digitos), df[c_soc].map(so_digitos)
        df = df[df["_cia"].isin(cnpjs) & (df["_soc"] != "")]
        ctrl = pd.Series(True, index=df.index)
        if c_tipo:
            ctrl = df[c_tipo].map(norm).str.contains("controlad") & ~df[c_tipo].map(norm).str.contains("coligad")
        if c_pct:
            ctrl = ctrl | (df[c_pct].map(num) > 50)
        n0 = sum(len(v) for v in grupos.values())
        for cia, soc in zip(df.loc[ctrl, "_cia"], df.loc[ctrl, "_soc"]):
            if cia not in grupos or ano == hoje.year or soc not in grupos[cia]:
                grupos.setdefault(cia, set()).add(soc)
        print(f"  FRE {ano}: {sum(len(v) for v in grupos.values()) - n0} vínculos empresa–controlada "
              f"(colunas: {c_cia} / {c_soc} / {c_tipo} / {c_pct})")
    return grupos


# ------------------------------------------------------------------ montagem
def main():
    cob = pd.read_csv("saidas/universo/cobertura.csv", dtype={"cnpj": str})
    cob["cnpj"] = cob["cnpj"].map(so_digitos)
    fin_path = Path("saidas/universo/financeiras.csv.gz")
    cnpjs = set(cob["cnpj"])
    if fin_path.exists():
        cnpjs |= set(pd.read_csv(fin_path, dtype={"cnpj": str})["cnpj"].map(so_digitos))
    grupos = ler_controladas(cnpjs)
    dono = {c: c for c in cnpjs}
    for cia, socs in grupos.items():
        for s in socs:
            dono.setdefault(s, cia)          # se a controlada também é listada, ela fica com as próprias dívidas
    snd = ler_snd()
    snd["cnpj_grupo"] = snd["cnpj_emissor"].map(dono)
    snd = snd[snd["cnpj_grupo"].notna()].copy()
    snd["via_controlada"] = snd["cnpj_grupo"] != snd["cnpj_emissor"]
    anb = anbima_ate(date.today())
    if anb is not None:
        snd = snd.merge(anb, on="codigo", how="left")
    snd["saldo_vna_rs_mm"] = snd["quantidade_mercado"] * snd["vna"] / 1e6
    snd["saldo_pu_par_rs_mm"] = np.where(snd.get("pct_pu_par").notna() & snd.get("pu").notna(),
                                         snd["quantidade_mercado"] * snd["pu"] / (snd["pct_pu_par"] / 100) / 1e6,
                                         np.nan) if anb is not None else np.nan
    snd["saldo_rs_mm"] = snd["saldo_pu_par_rs_mm"].fillna(snd["saldo_vna_rs_mm"]) if anb is not None else snd["saldo_vna_rs_mm"]
    snd["duration_anos"] = snd["duration_du"] / 252 if "duration_du" in snd else np.nan
    SAIDA.mkdir(parents=True, exist_ok=True)
    snd.sort_values(["cnpj_grupo", "vencimento"]).to_csv(SAIDA / "debentures.csv.gz", index=False,
                                                         compression="gzip", date_format="%Y-%m-%d")
    meta = {"gerado_em": date.today().isoformat(), "debentures": int(len(snd)),
            "empresas": int(snd["cnpj_grupo"].nunique()), "via_controladas": int(snd["via_controlada"].sum()),
            "data_anbima": str(anb["data_anbima"].iloc[0]) if anb is not None else None}
    (SAIDA / "metadados.json").write_text(json.dumps(meta, ensure_ascii=False, indent=2))
    print(f"  -> {SAIDA}/debentures.csv.gz: {meta}")


if __name__ == "__main__":
    main()
