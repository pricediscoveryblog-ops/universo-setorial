# Curadoria: modelo de negócio e breve histórico

Um arquivo por empresa, com o nome da raiz do ticker na B3 (ex.: `VULC.yaml`) ou o ticker americano (ex.: `AAPL.yaml`).
Campos: `modelo_negocio` (texto), `segmentos` (lista), `historico` (lista de `{ano, fato}`), `fontes` (lista de
`{nome, url}`) e `revisado_em`. O app mostra a seção no fim da aba Empresa; sem arquivo, aparece "em curadoria".
Texto de curadoria do Price Discovery, redigido a partir das fontes listadas em cada arquivo.
