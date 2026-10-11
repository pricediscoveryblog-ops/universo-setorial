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

`sucessoes.yaml` emenda séries de empresas que trocaram de CNPJ (ex.: JBS S.A. → JBS N.V.).
Se o CNPJ informado pela B3 não existir no cadastro da CVM, o universo usa o CNPJ de mesma raiz.

O workflow roda o universo todo dia 20 (e manualmente em Actions → Run workflow → modo).

## Abas do Excel (modo grupos)
LEIA-ME · painel · mediana_setor · cobertura (com ticker) · reapresentacoes · auditoria · base_longa.
