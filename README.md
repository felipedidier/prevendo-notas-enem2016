# Descobrindo as melhores notas de matemática do ENEM 2016


Projeto final do curso de Data Science da [Awari](https://awari.com.br/). O objetivo desse projeto é realizar todos os passos importantes de uma trilha de Data Science.

Será criado um modelo de **previsão da nota da prova matemática de quem participou do ENEM 2016 **.


## Tópicos


- Ambiente
- Dataset
- Problemática
- Código
- Coleta e importação de dados
- Preparação dos dados
- Análise exploratória
- Modelagem


## Ambiente


O código do projeto foi implementado majoritariamente em Python 3 no ambiente do Google Colab.


## Dataset

O projeto utilizará o [dataset](https://raw.githubusercontent.com/felipedidier/prevendo-notas-enem2016/master/train.csv) do Desafio de Resultados do ENEM 2016. 

Este arquivo apresenta informações socioeconômicas preenchidas pelos participantes do ENEM 2016, assim como as respostas e desempenho em cada competência da prova. Todas as colunas de dados presentes nessa base podem ser consultadas [clicando aqui](https://s3-us-west-1.amazonaws.com/acceleration-assets-highway/data-science/dicionario-de-dados.zip).

## Problemática

O contexto do desafio gira em torno dos resultados do ENEM 2016. O objetivo dessa modelagem preditiva das notas dos participantes é procurar entender mais as 

Muitas universidades brasileiras utilizam o ENEM para selecionar seus futuros alunos e alunas. Isto é feito com uma média ponderada das notas das provas de matemática, ciências da natureza, linguagens e códigos, ciências humanas e redação. 

No arquivo train.csv será criado um modelo para prever nota da prova de matemática (coluna **NU_NOTA_MT**) de quem participou do ENEM 2016. 

```
# Isto está formatado como código
```
## Observações

Será utilizado Python 3.

**Arquivo principal**: modelagemPythonFINAL.ipynb
