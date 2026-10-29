"""
entregas_rad.py — Price Discovery
Entregas de ITR e DFP em tempo quase real, pela consulta pública de documentos da CVM (ENET/RAD).
Complementa os pacotes de Dados Abertos (atualizados com cerca de uma semana de defasagem) no calendário de resultados.

Saída: saidas/universo/entregas_rad.csv.gz (cnpj, documento, dt_refer, dt_receb, link)
"""
import io
import re
from datetime import date, timedelta
from pathlib import Path

import pandas as pd
import requests

PAGINA = "https://www.rad.cvm.gov.br/ENET/frmConsultaExternaCVM.aspx"
LISTAR = PAGINA + "/ListarDocumentos"
VER = "https://www.rad.cvm.gov.br/ENET/frmGerenciaPaginaFRE.aspx?NumeroSequencialDocumento={n}&CodigoTipoInstituicao=1"
CADASTRO = "https://dados.cvm.gov.br/dados/CIA_ABERTA/CAD/DADOS/cad_cia_aberta.csv"
SAIDA = Path("saidas/universo/entregas_rad.csv.gz")
CATEGORIAS = {"EST_3": "ITR", "EST_4": "DFP"}
CAB = {"User-Agent": "Mozilla/5.0 (compatible; pricediscovery-bot)"}


def cd_cvm_para_cnpj() -> dict[int, str]:
    r = requests.get(CADASTRO, headers=CAB, timeout=120)
    r.raise_for_status()
    c = pd.read_csv(io.BytesIO(r.content), sep=";", encoding="latin-1", dtype=str, usecols=["CNPJ_CIA", "CD_CVM"])
    c["cnpj"] = c["CNPJ_CIA"].str.replace(r"\D", "", regex=True).str.zfill(14)
    c["cd"] = pd.to_numeric(c["CD_CVM"], errors="coerce")
    return dict(zip(c.dropna(subset=["cd"])["cd"].astype(int), c.dropna(subset=["cd"])["cnpj"]))


def listar(s: requests.Session, categoria: str, de: date, ate: date) -> list[list[str]]:
    corpo = {"dataDe": de.strftime("%d/%m/%Y"), "dataAte": ate.strftime("%d/%m/%Y"), "empresa": "",
             "setorAtividade": "-1", "categoriaEmissor": "-1", "situacaoEmissor": "-1", "tipoParticipante": "1",
             "dataReferencia": "", "categoria": categoria, "periodo": "2", "horaIni": "", "horaFim": "",
             "palavraChave": "", "ultimaDtRef": "false", "tipoEmpresa": "0", "token": "", "versaoCaptcha": ""}
    r = s.post(LISTAR, json=corpo, headers={**CAB, "Content-Type": "application/json; charset=utf-8"}, timeout=120)
    r.raise_for_status()
    d = r.json().get("d", {})
    if d.get("temErro"):
        raise RuntimeError(d.get("msgErro"))
    return [reg.split("$&") for reg in (d.get("dados") or "").split("&*") if reg]


def interpretar(regs: list[list[str]], doc: str, mapa: dict[int, str]) -> pd.DataFrame:
    linhas = []
    for c in regs:
        if len(c) < 9:
            continue
        txt = lambda i: re.sub(r"<[^>]+>", "", c[i]).strip()
        cd = pd.to_numeric(re.sub(r"\D", "", txt(0)), errors="coerce")
        ref = re.match(r"(\d{8})", txt(5))
        rec = re.search(r"(\d{2}/\d{2}/\d{4} \d{2}:\d{2})", txt(6))
        if pd.isna(cd) or not ref or not rec or "ativo" not in txt(7).lower():
            continue
        seq = re.search(r"NumeroSequencialDocumento=(\d+)", " ".join(c[9:]))
        linhas.append({"cnpj": mapa.get(int(cd)), "documento": doc,
                       "dt_refer": pd.to_datetime(ref.group(1), format="%Y%m%d").date(),
                       "dt_receb": pd.to_datetime(rec.group(1), format="%d/%m/%Y %H:%M"),
                       "link": VER.format(n=seq.group(1)) if seq else ""})
    return pd.DataFrame(linhas)


def main(dias: int = 130):
    mapa = cd_cvm_para_cnpj()
    s = requests.Session()
    s.get(PAGINA, headers=CAB, timeout=60)           # cookie de sessão
    hoje = date.today()
    partes = []
    for cat, doc in CATEGORIAS.items():
        # janelas de 30 dias: a consulta pública limita o volume por requisição
        ini = hoje - timedelta(days=dias)
        while ini <= hoje:
            fim = min(ini + timedelta(days=29), hoje)
            try:
                df = interpretar(listar(s, cat, ini, fim), doc, mapa)
                partes.append(df)
                print(f"  {doc} {ini:%d/%m}–{fim:%d/%m}: {len(df)} entregas")
            except Exception as e:  # noqa: BLE001
                print(f"  ! {doc} {ini:%d/%m}–{fim:%d/%m}: {e}")
            ini = fim + timedelta(days=1)
    if not partes:
        raise SystemExit("Nenhuma entrega obtida.")
    out = pd.concat(partes, ignore_index=True).dropna(subset=["cnpj"])
    out = out.sort_values("dt_receb").drop_duplicates(["cnpj", "documento", "dt_refer"])
    SAIDA.parent.mkdir(parents=True, exist_ok=True)
    out.to_csv(SAIDA, index=False, compression="gzip")
    print(f"  -> {SAIDA}: {len(out)} entregas, {out['cnpj'].nunique()} empresas, "
          f"última em {out['dt_receb'].max()}")


if __name__ == "__main__":
    main()
