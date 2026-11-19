# Coletor setorial — Price Discovery

Coleta métricas de DRE, balanço e DFC das DFPs da CVM (Dados Abertos) para grupos
setoriais definidos em `grupos.yaml`. As empresas de cada grupo vêm da classificação
setorial da B3 (com CNPJ da própria B3) e/ou de inclusões manuais. Janela: últimos 10
exercícios por empresa, ou o maior período disponível, com o motivo do corte na aba `cobertura`.

## Estrutura do repositório
```
coletor_setorial.py
grupos.yaml
requirements.txt
.github/workflows/coletor_setorial.yml
cache_b3/empresas_b3.csv   (gerado e versionado: lista B3 com CNPJ e setor)
saidas/                    (gerado: <grupo>.xlsx e <grupo>_base.csv)
cache_cvm/                 (gerado: ZIPs da CVM; NÃO versionar -> .gitignore)
```

## Uso
```
pip install -r requirements.txt
python coletor_setorial.py setores --atualizar-b3          # 1ª vez: monta a lista B3 (~15-20 min)
python coletor_setorial.py filtrar --setor "petroleo"       # confere quem entra no filtro
python coletor_setorial.py coletar --grupo distribuidoras_combustiveis
python coletor_setorial.py descobrir "nome"                  # busca no cadastro CVM (não listadas)
```
O cache da B3 vale 30 dias (`B3_CACHE_DIAS`); depois disso só as empresas novas têm o
detalhe baixado de novo. `--atualizar-b3` força a atualização completa.

## Universo (todas as empresas)
`python coletor_setorial.py universo` processa todas as empresas com classificação setorial
na B3 (exceto setor financeiro) e grava em `saidas/universo/`:
- `base.csv.gz` — uma linha por empresa × período, com `tipo` = anual | trimestral | ltm.
  Trimestres: fluxos por diferença dos acumulados (2T = 6M − 3M, 3T = 9M − 6M, 4T = anual − 9M);
  balanço = posição na data. LTM = soma de 4 trimestres consecutivos + balanço do último.
- `cobertura.csv` — anos e trimestres disponíveis por empresa e motivo de séries curtas.
- `auditoria.csv.gz` — linhas da DFC/balanço capturadas por busca textual.
- `metadados.json` — data de geração e contagens.
- `diagnostico_dfc.csv.gz` — linhas da DFC dos anos em que a busca não achou D&A, capex ou
  pagamento de arrendamento (para calibrar os filtros).

Concessões do modelo do intangível (segmentos rodovias, saneamento, apoio e armazenagem, ferrovias e hidrovias;
`SEG_CONSTRUCAO` em `universo.py`): a receita de construção (ICPC 01) sai da receita e do CPV, e a margem de
construção sai do lucro bruto, EBIT, EBITDA e (líquida de 34% de IR/CS) do lucro líquido. Fonte: subconta da DRE →
DVA 7.01.03 → custo da DRE + margem da DFC. Reportados em `*_reportado(a)`; origem em `fonte_construcao`.

`sucessoes.yaml` emenda séries de empresas que trocaram de CNPJ (ex.: JBS S.A. → JBS N.V.).
Se o CNPJ informado pela B3 não existir no cadastro da CVM, o universo usa o CNPJ de mesma raiz.

O workflow roda o universo todo dia 20 (e manualmente em Actions → Run workflow → modo).

## Abas do Excel (modo grupos)
LEIA-ME · painel · mediana_setor · cobertura (com ticker) · reapresentacoes · auditoria · base_longa.

## Mercado (cotações, proventos, VM/EV diário e acionistas) — nov/2026
- `multiplos.py` (dentro do universo): grava `saidas/mercado/precos_<ano>.parquet` (fechamento bruto do COTAHIST desde
  2015, ON/PN/units) e `vm_ev_<ano>.parquet` (valor de mercado diário e EV com/sem arrendamento, ponto no tempo: dívida
  líquida e minoritários do último ITR/DFP divulgado em cada data). Particionado por ano para o Git só regravar o ano corrente.
- `mercado.py` (workflow `mercado`, dias úteis 20h): `proventos.parquet` (B3: histórico de proventos em dinheiro por
  espécie + complemento recente por ISIN + desdobramentos/grupamentos/bonificações), `acionistas.parquet` e
  `distribuicao_capital.parquet` (FRE da CVM, posição acionária com cadeia de controle e free float).
- Fator de ações da B3: desdobramento/bonificação em % de ações novas (100 = 2 por 1); grupamento = ações resultantes por
  ação antiga (0,1 = 10 por 1). Conferido contra o salto de preço na data ex (MGLU, WEGE, BBAS, ITUB, BBDC, CMIG).

## Dois apps
- **Price Discovery Scanning BR**: `app/app.py` — https://pricediscovery-scanning-br.streamlit.app
- **Price Discovery Scanning USA**: `app/app_eua.py` (roda o mesmo `app.py` com `MERCADO_FIXO = "EUA"`) —
  https://pricediscovery-scanning-usa.streamlit.app. Lê o coletor-eua público pelo raw do GitHub; cotações do gráfico
  via yfinance.
- Seções da aba Empresa em `app/mercado_aba.py`; textos de modelo de negócio em `app/curadoria/<RAIZ>.yaml`.
