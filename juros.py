"""
juros.py — Price Discovery
Estrutura a termo das taxas de juros (ETTJ) da ANBIMA: curva prefixada, curva real (IPCA) e inflação implícita.
A ANBIMA disponibiliza gratuitamente só os últimos 5 dias úteis; este script roda todo dia útil e acumula
o histórico em saidas/juros/.

Saídas:
  saidas/juros/parametros.csv   — parâmetros Svensson por data e curva (PREFIXADOS, IPCA)
  saidas/juros/vertices.csv.gz  — vértices publicados (dias úteis; IPCA, PRE e implícita, % a.a.)
"""
import io
import re
import sys
from datetime import date, timedelta
from pathlib import Path

import pandas as pd
import requests

URL = "https://www.anbima.com.br/informacoes/est-termo/CZ-down.asp"
SAIDA = Path("saidas/juros")
CAB = {"User-Agent": "Mozilla/5.0 (compatible; pricediscovery-bot)",
       "Referer": "https://www.anbima.com.br/informacoes/est-termo/default.asp"}


def num(x: str) -> float:
    x = (x or "").strip().replace(".", "").replace(",", ".") if re.search(r",", x or "") else (x or "").strip()
    try:
        return float(x)
    except ValueError:
        return float("nan")


def baixar(d: date) -> tuple[list[dict], list[dict]] | None:
    r = requests.post(URL, data={"Idioma": "PT", "saida": "csv", "Dt_Ref": d.strftime("%d/%m/%Y")},
                      headers=CAB, timeout=60)
    if r.status_code != 200:
        return None
    t = r.content.decode("latin-1")
    linhas = [l.strip() for l in t.splitlines()]
    if not linhas or not re.match(r"\d{2}/\d{2}/\d{4};Beta", linhas[0]):
        return None                                     # sem dados para a data (feriado, fim de semana, ainda não publicado)
    dref = pd.to_datetime(linhas[0].split(";")[0], format="%d/%m/%Y").date()
    params, verts, em_vert = [], [], False
    for l in linhas[1:]:
        p = l.split(";")
        if p[0] in ("PREFIXADOS", "IPCA") and len(p) >= 7:
            params.append({"data": dref, "curva": p[0], "b1": num(p[1]), "b2": num(p[2]), "b3": num(p[3]),
                           "b4": num(p[4]), "l1": num(p[5]), "l2": num(p[6])})
        elif p[0].lower().startswith("vertices"):
            em_vert = True
        elif em_vert:
            if not l or not re.match(r"[\d.]+$", p[0]):
                em_vert = False
                continue
            verts.append({"data": dref, "du": int(p[0].replace(".", "")), "ipca": num(p[1]) if len(p) > 1 else float("nan"),
                          "pre": num(p[2]) if len(p) > 2 else float("nan"),
                          "implicita": num(p[3]) if len(p) > 3 else float("nan")})
    return params, verts


def main(dias: int = 10):
    SAIDA.mkdir(parents=True, exist_ok=True)
    arq_p, arq_v = SAIDA / "parametros.csv", SAIDA / "vertices.csv.gz"
    hist_p = pd.read_csv(arq_p, parse_dates=["data"]) if arq_p.exists() else pd.DataFrame()
    hist_v = pd.read_csv(arq_v, parse_dates=["data"]) if arq_v.exists() else pd.DataFrame()
    novos_p, novos_v = [], []
    hoje = date.today()
    for k in range(dias):
        d = hoje - timedelta(days=k)
        if d.weekday() >= 5:
            continue
        try:
            res = baixar(d)
        except Exception as e:  # noqa: BLE001
            print(f"  ! {d}: {e}")
            continue
        if not res:
            print(f"  {d}: sem curva publicada")
            continue
        p, v = res
        novos_p += p
        novos_v += v
        print(f"  {d}: {len(p)} curvas, {len(v)} vértices")
    if not novos_p:
        sys.exit("Nenhuma curva obtida da ANBIMA.")
    p = pd.concat([hist_p, pd.DataFrame(novos_p)], ignore_index=True)
    p["data"] = pd.to_datetime(p["data"])
    p = p.drop_duplicates(["data", "curva"], keep="last").sort_values(["data", "curva"])
    v = pd.concat([hist_v, pd.DataFrame(novos_v)], ignore_index=True)
    v["data"] = pd.to_datetime(v["data"])
    v = v.drop_duplicates(["data", "du"], keep="last").sort_values(["data", "du"])
    p.to_csv(arq_p, index=False, date_format="%Y-%m-%d")
    v.to_csv(arq_v, index=False, compression="gzip", date_format="%Y-%m-%d")
    print(f"  -> {arq_p} ({p['data'].nunique()} datas, {p['data'].min().date()} a {p['data'].max().date()}), {arq_v}")


if __name__ == "__main__":
    main()
