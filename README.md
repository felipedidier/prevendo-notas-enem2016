# Projeto: Prevendo as notas de matemática do ENEM
<img src="https://img.shields.io/badge/Status-Completed-brightgreen"/>

Projeto final do curso de Data Science da [Awari](https://awari.com.br/). O objetivo desse projeto é realizar todos os passos importantes de uma trilha de Data Science.

Será criado um modelo de **previsão da nota da prova matemática de quem participou do ENEM 2016 :books:**.

> teste

## Tópicos


* [Ambiente](#ambiente)
- Dataset
- Problemática
- Descrição
- Conclusão


## Ambiente


O código do projeto foi implementado majoritariamente em Python 3 no ambiente do Google Colab.


## Dataset

O projeto utilizará o [dataset](https://raw.githubusercontent.com/felipedidier/prevendo-notas-enem2016/master/train.csv) do Desafio de Resultados do ENEM 2016. 

Este arquivo apresenta informações socioeconômicas preenchidas pelos participantes do ENEM 2016, assim como as respostas e desempenho em cada competência da prova, entre outras informações. Todas as colunas de dados presentes nessa base podem ser consultadas [clicando aqui](https://s3-us-west-1.amazonaws.com/acceleration-assets-highway/data-science/dicionario-de-dados.zip).

## Problemática

O contexto do desafio gira em torno dos resultados do ENEM 2016. O objetivo dessa modelagem preditiva das notas dos participantes é procurar entender melhor quais os fatores característicos que mais influenciam no desempenho escolar de um participante e assim. 

Muitas universidades brasileiras utilizam o ENEM para selecionar seus futuros alunos e alunas. Isto é feito com uma média ponderada das notas das provas de matemática, ciências da natureza, linguagens e códigos, ciências humanas e redação. 

No arquivo train.csv será criado um modelo para prever nota da prova de matemática (coluna **NU_NOTA_MT**) de quem participou do ENEM 2016. 

## Descrição

A documentação desse projeto acompanhou os passos de um projeto em Data Science, passando por todas as etapas necessárias:

1. Aquisição de Dados
2. Preparação de Dados e Análise Exploratória
4. Feature Engineering
5. Modelagem
6. Avaliação
7. Deploy

### Aquisição de dados

Como dito [anteriormente](#dataset), os dados foram coletados do dataset do Desafio de Resultados do ENEM 2016. Para coletar os dados do arquivo .CSV foi utilizada a biblioteca pandas, que estrutura os dados de forma tabular em linhas e colunas, DataFrame.

```python
## Bibliotecas utilizadas
import pandas as pd

## Conversão .CSV para DataFrame
df = pd.read_csv('https://raw.githubusercontent.com/felipedidier/prevendo-notas-enem2016/master/train.csv',encoding='utf-8-sig')
```

### Preparação dos Dados e Análise Exploratória

#### Shape do DataFrame
Inicialmente verificou-se a presença de 13730 linhas e 167 colunas utilizando ```df.shape```.

#### Pré-seleção de variáveis
Antes de dar procedimento na análise via código, as variáveis do DataFrame foram analisadas manualmente e foram selecionadas as que faziam sentido para a problemática em questão. Das 167 variáveis iniciais, 56 foram selecionadas.

```python
## Variáveis selecionadas
selected_feat = ['NU_IDADE', 'SG_UF_RESIDENCIA', 'TP_SEXO', 'TP_ESTADO_CIVIL', 'TP_COR_RACA', 'TP_NACIONALIDADE', 'TP_ST_CONCLUSAO', 'TP_ESCOLA', 'IN_TREINEIRO', 'IN_BAIXA_VISAO', 'IN_CEGUEIRA', 'IN_SURDEZ', 'IN_DEFICIENCIA_AUDITIVA', 'IN_SURDO_CEGUEIRA', 'IN_DEFICIENCIA_FISICA', 'IN_DEFICIENCIA_MENTAL', 'IN_DEFICIT_ATENCAO', 'IN_DISLEXIA', 'IN_DISCALCULIA', 'IN_AUTISMO', 'IN_VISAO_MONOCULAR', 'IN_OUTRA_DEF', 'IN_GESTANTE', 'IN_LACTANTE', 'IN_IDOSO', 'TP_PRESENCA_MT', 'NU_NOTA_CN', 'NU_NOTA_CH', 'NU_NOTA_LC', 'NU_NOTA_MT', 'Q006', 'Q007', 'Q008', 'Q009', 'Q010', 'Q011', 'Q012', 'Q013', 'Q014', 'Q015', 'Q016', 'Q017', 'Q018', 'Q019', 'Q020', 'Q021', 'Q022', 'Q023', 'Q024', 'Q025', 'Q026', 'Q042', 'Q043', 'Q045', 'Q047', 'Q048']

## Atualização do DataFrame somente com as 56 features
df = df[selected_feat]
```

#### Tipos das variáveis
Com o método ```df.info()```, que retorna a coluna e o tipo de dado presente nela, foi possível constatar que o tipo todas as colunas estavam com o tipo adequado para prosseguir na análise.

#### Dados nulos
O metódo ```df.isna().sum()``` permite visualizar a quantidade de dados nulos em cada coluna. Ao aplicar o método, foi constatado que as colunas ```[TP_ESTADO_CIVIL, NU_NOTA_CN, NU_NOTA_CH, NU_NOTA_LC, NU_NOTA_MT]``` apresentavam dados nulos.

No caso das colunas iniciadas com ```NU_NOTA```, os valores nulos ocorrem devido os participantes não estarem presentes no dia de realização da prova, ou seja, devem ser desconsiderados. A coluna ```TP_PRESENCA_MT``` utiliza de informação binária para representar os anulos presentes (1) e ausentes (0) na prova de Matemática. Dessa forma, para filtrar somente os alunos presentes:

```python
df = df[df['TP_PRESENCA_MT']==1]
```

Como a prova de Ciências Humanas é realizada em outro dia, alguns dados nulos ainda ficaram presentes nas variáveis ```NU_NOTA_CH``` e ```NU_NOTA_LC```. Para esses casos, o filtro foi aplicado utilizando o próprio método ```.isna()```:

```python
df = df[~df['NU_NOTA_CH'].isna()==True]
```

Como não há justificativa para os valores nulos na coluna ```TP_ESTADO_CIVIL```, os valores foram removidos.

```python
df = df[df['TP_ESTADO_CIVIL'].isna()==False]
```

Após essa preparação dos dados, o DataFrame apresenta 9781 linhas e 56 colunas.

#### Análise das colunas de PcD

É muito importante analisar a performance de pessoas com deficiência no ENEM, pois o exame deve ser acessível e factível de ser realizado por qualquer indivíduo. Nesse projeto especificamente não será aprofundada essa questão devido a baixa quantidade de participantes que apresentem alguma deficiência no dataset (apenas 0,5%). Como a amostra dessas features é muito baixa, essas features serão desconsideradas no projeto atual.

```python
IN = ['IN_BAIXA_VISAO', 'IN_CEGUEIRA', 'IN_SURDEZ', 'IN_DEFICIENCIA_AUDITIVA', 'IN_SURDO_CEGUEIRA', 'IN_DEFICIENCIA_FISICA', 'IN_DEFICIENCIA_MENTAL', 'IN_DEFICIT_ATENCAO', 'IN_DISLEXIA', 'IN_DISCALCULIA', 'IN_AUTISMO', 'IN_VISAO_MONOCULAR', 'IN_OUTRA_DEF']

df.drop(columns=IN, inplace=True)
```

Após essa preparação dos dados, o DataFrame apresenta 9781 linhas e 43 colunas.

#### Visualização dos dados

O histograma foi utilizado para analisar o comportamento das variáveis. As bibliotecas utilizadas para esse processo foram:

```python
## Bibliotecas utilizadas
import matplotlib.pyplot as plt
import seaborn as sns
sns.set_style(style='darkgrid')
```

Como era necessário visualizar o histograma de 43 variáveis, foi utilizado o método de ```.subplots()```.

```python
fig, axes = plt.subplots(nrows = 12, ncols = 5)
fig.set_size_inches(20,50)

for i, col in enumerate(selected_feat_eda):
  sns.histplot(df[col], ax = axes[i//5,i%5], bins= 10, stat='density', kde=True)
```

![histograma](https://iharsh234.github.io/WebApp/images/demo/demo_landing.JPG)

Diante da necessidade de manipulação, limpeza e visualização de dados, as bibliotecas abaixo foram importadas:

```python
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
```
