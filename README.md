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
3. Feature Engineering
4. Modelagem
5. Avaliação
6. Deploy

### 1. Aquisição de dados

Como dito [anteriormente](#dataset), os dados foram coletados do dataset do Desafio de Resultados do ENEM 2016. Para coletar os dados do arquivo .CSV foi utilizada a biblioteca pandas, que estrutura os dados de forma tabular em linhas e colunas, DataFrame.

```python
## Bibliotecas utilizadas
import pandas as pd

## Conversão .CSV para DataFrame
df = pd.read_csv('https://raw.githubusercontent.com/felipedidier/prevendo-notas-enem2016/master/train.csv',encoding='utf-8-sig')
```

### 2. Preparação dos Dados e Análise Exploratória

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

Após esse processo, a coluna ```TP_PRESENCA_MT``` não é mais necessária, portanto foi utilizado o método ```.drop()``` para retirá-la do dataframe.
Após essa preparação dos dados, o DataFrame apresenta 9781 linhas e 55 colunas.

#### Análise das colunas de PcD

É muito importante analisar a performance de pessoas com deficiência no ENEM, pois o exame deve ser acessível e factível de ser realizado por qualquer indivíduo. Nesse projeto especificamente não será aprofundada essa questão devido a baixa quantidade de participantes que apresentem alguma deficiência no dataset (apenas 0,5%). Como a amostra dessas features é muito baixa, essas features serão desconsideradas no projeto atual.

```python
IN = ['IN_BAIXA_VISAO', 'IN_CEGUEIRA', 'IN_SURDEZ', 'IN_DEFICIENCIA_AUDITIVA', 'IN_SURDO_CEGUEIRA', 'IN_DEFICIENCIA_FISICA', 'IN_DEFICIENCIA_MENTAL', 'IN_DEFICIT_ATENCAO', 'IN_DISLEXIA', 'IN_DISCALCULIA', 'IN_AUTISMO', 'IN_VISAO_MONOCULAR', 'IN_OUTRA_DEF']

df.drop(columns=IN, inplace=True)
```

Após essa preparação dos dados, o DataFrame apresenta 9781 linhas e 42 colunas.

#### Visualização dos dados

O histograma foi utilizado para analisar o comportamento das variáveis. As bibliotecas utilizadas para esse processo foram:

```python
## Bibliotecas utilizadas
import matplotlib.pyplot as plt
import seaborn as sns
sns.set_style(style='darkgrid')
```

Como era necessário visualizar o histograma de 42 variáveis, foi utilizado o método de ```.subplots()```.

```python
fig, axes = plt.subplots(nrows = 12, ncols = 5)
fig.set_size_inches(20,50)

for i, col in enumerate(selected_feat_eda):
  sns.histplot(df[col], ax = axes[i//5,i%5], bins= 10, stat='density', kde=True)
```
![histograma](https://github.com/felipedidier/prevendo-notas-enem2016/blob/master/images/histograma.png?raw=true)

Nota-se a presença de muitas variáveis categóricas no dataset. Analisando os gráficos, chama a atenção a distribuição assimétrica da feature ```NU_IDADE``` que representa a idade dos participantes. Para esse caso em específico, é realizada uma análise mais aprofundada utilizando boxplot.

```python
sns.boxplot(df['NU_IDADE'])
```
![boxplot](https://github.com/felipedidier/prevendo-notas-enem2016/blob/master/images/boxplot.png?raw=true)

Nota-se no boxplot a presença de alguns outliers que serão analisados.
Os outliers, utilizando o método ```z_socre```, representam todos os participantes com idade igual ou acima de 33 anos, o equivalente a 570 registros de dados (5,8% dos registros totais).

```python
def z_score_remove(df,col):
  z = np.abs(stats.zscore(df[col]))
  return df[(z < 2)], df[(z >= 2)]
  
df1, df2 = z_score_remove(df, 'NU_IDADE')
df2.shape # (570, 43)
df2['NU_IDADE'].value_counts().index.sort_values()[0] # 33
```

Devido a participação dos dados (5,8%), os outliers não foram removidos por representar um valor já significante dos dados.

Por fim, monta-se um heatmap prévio para visualizar a correlação entre as colunas do tipo numéricas existentes. De inicio, vê-se que as únicas correlações interessantes com ```NU_NOTAS_MT``` são as demais colunas ```NU_NOTAS```.

```python
sns.heatmap(df.corr())
```
![heatmap1](https://github.com/felipedidier/prevendo-notas-enem2016/blob/master/images/heatmap_pre.png?raw=true)

### 3. Feature Engineering

#### Treino e Teste

É realizada a separação dos dados em treino e teste. Para isso, importa-se a seguinte biblioteca:

```python
## Bibliotecas utilizadas
from sklearn.model_selection import train_test_split
```

Para a quebra do dataframe em treino e teste foi escolhida a proporção de 80/20, onde 80% da base será treino e 20% será teste.

```python
train, test = train_test_split(df, test_size=0.2, random_state=42)
X_train, y_train = train.drop(columns="NU_NOTA_MT"), train["NU_NOTA_MT"]
X_test, y_test = test.drop(columns="NU_NOTA_MT"), test["NU_NOTA_MT"]
```
As grandezas de cada DataFrame:
```
train: (7824, 42)
test: (1957, 42)
```

#### Transformação de features

As features serão agrupadas em 03 grupos:

- Variáveis numéricas ```feat_num```: a transformação dessas variáveis será realizada com Z-scale;
- Variáveis categóricas ```feat_cat1```: a transformação dessas variáveis será realizada com Ordinal Enconder;
- Variáveis categóricas ```feat_cat2```: a transformação dessas variáveis será realizada com OneHot Enconder.

```python
feat_num = ['NU_IDADE', 'NU_NOTA_MT', 'NU_NOTA_CN', 'NU_NOTA_CH', 'NU_NOTA_LC']
feat_cat1 = ['TP_SEXO', 'Q006', 'Q007', 'Q008', 'Q009', 'Q010', 'Q011', 'Q012', 'Q013', 'Q014', 'Q015', 'Q016', 'Q017', 'Q018', 'Q019', 'Q020', 'Q021', 'Q022', 'Q023', 'Q024', 'Q025', 'Q026', 'Q042', 'Q043', 'Q045', 'Q047', 'Q048']
feat_cat2 = ['SG_UF_RESIDENCIA', 'TP_ESTADO_CIVIL', 'TP_COR_RACA', 'TP_NACIONALIDADE', 'TP_ST_CONCLUSAO']
```

Para esse momento, serão utilizados alguns métodos da biblioteca sklearn e category_encoders.

```python
## Bibliotecas utilizadas
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from category_encoders import OrdinalEncoder 
```

#### Variáveis Númericas

Para as variáveis numéricas, será utilizada a transformação Standard Scaler (Z-scale).

```python
z_score = StandardScaler()
def std_z_scale(df,col):
  df[col] = z_score.fit_transform(df[[col]])

for col in feat_num:
  std_z_scale(train, col)
  std_z_scale(test, col)
```

Após transformação, os resultados obtidos foram:

```python
fig, axes = plt.subplots(nrows = 3, ncols = 2)
fig.set_size_inches(20,20)

for i, col in enumerate(feat_num):
  sns.histplot(train[col], ax = axes[i//2,i%2], bins= 10, stat='density', kde=True)
```
![histograma2](https://github.com/felipedidier/prevendo-notas-enem2016/blob/master/images/histplot2.png?raw=true)

#### Variáveis Categóricas

Devido o formato dos dados, serão adotados modelos de transformações distintos para atender cada feature. Serão utilizadas as transformações OneHot Encoder e Ordinal Encoder.

Para as features do grupo ```feat_cat1```, será utilizada a transformação Ordinal Encoder, que substituirá os dados alfabéticos por valores numéricos.

```python
## Ordinal Encoder
oe = OrdinalEncoder()

maplist = [{'mapping': {'A':1,'B':2,'C':3,'D':4,'E':5,'F':6,'G':7,'H':8,'I':9,'J':10,'K':11,'L':12,'M':13,'N':14,'O':15,'P':16,'Q':17}}]

def ordinal_encoder(df):
  df[feat_cat1] = oe.fit_transform(df[feat_cat1])
  
ordinal_encoder(train)
ordinal_encoder(test)
```

Para as features do grupo ```feat_cat2```, será utilizada a transformação OneHot Encoder, que para cada valor distinto de cada feature criará uma nova feature com valores binários.

```python
## OneHot Encoder
one_hot = OneHotEncoder()

## train
for col in feat_cat2:
  temp = pd.DataFrame(one_hot.fit_transform(train[[col]]).toarray(),columns=one_hot.get_feature_names([col]))
  train = train.drop(columns=col).join(temp)

## test
for col in feat_cat2:
  temp = pd.DataFrame(one_hot.fit_transform(test[[col]]).toarray(),columns=one_hot.get_feature_names([col]))
  test = test.drop(columns=col).join(temp)
```

As grandezas de cada DataFrame:
```
train: (7824, 83)
test: (1957, 83)
```

Após as transformações é possível visualizar a correlação entre todas as features envolvidas.

```python
plt.figure(figsize=(20,20))
sns.heatmap(train.corr())
```

![heatmap2](https://github.com/felipedidier/prevendo-notas-enem2016/blob/master/images/heatmap_pos.png?raw=true)

Em uma rápida análise, consegue-se perceber que há correlações interessantes sendo formadas entre ```NU_NOTA_MT``` e algumas features, como por exemplo ```Q006```, ```Q014```, ```Q047```, entre outras.

De uma forma alternativa, consegue-se visualizar a partir do gráfico abaixo as features que apresentam maior correlação com ```NU_NOTA_MT``` em ordem decrescente.

```python
train.corr()['NU_NOTA_MT'].sort_values(ascending=False).head(30).plot(kind='bar')
```
![grafic1](https://github.com/felipedidier/prevendo-notas-enem2016/blob/master/images/grafic1.png?raw=true)

#### Selecionando as melhores features

A seleção de features será realizada com o Select K Bests utilizando o f_classif, que se baseia na análise de variância com F-Tests.

```python
## Bibliotecas utilizadas
from sklearn.feature_selection import SelectKBest, f_classif
```

O primeiro passo é saber qual o score de cada feature para poder definir quantas features serão selecionadas. Para facilitar o processo, o ```train```é dividido em ```df_features``` e ```df_target```.

```python
df_features = train.drop(columns='NU_NOTA_MT')
df_target = train['NU_NOTA_MT']
```
Todas as features são selecionadas para obter o score de cada uma.

```python
fs = SelectKBest(score_func=f_classif, k='all')
fs.fit(df_features,df_target)

scores = pd.DataFrame(data=fs.scores_, columns=['score'])
scores.sort_values(by='score', ascending=False, inplace=True).reset_index(inplace=True)
scores = scores.sort_values(by='index')
scores['FEATURE'] = train.drop(columns='NU_NOTA_MT').columns
scores.sort_values(by='score', ascending=False, inplace=True)
```

Visualizando o resultado em gráfico para cada feature:

```python
plt.figure(figsize=(30,5))
sns.barplot(scores['index'], scores.score,  order=scores['index'], color='lightgreen')
plt.xticks(rotation=90)
plt.show()
``` 


