# Projeto: Prevendo as notas de matemática do ENEM
<img src="https://img.shields.io/badge/Status-Completed-brightgreen"/>

Projeto final do curso de Data Science da [Awari](https://awari.com.br/). O objetivo desse projeto é realizar todos os passos importantes de uma trilha de Data Science.

Será criado um modelo de **previsão da nota da prova matemática de quem participou do ENEM 2016 :books:**.

> Você pode acessar o código [clicando aqui](https://github.com/felipedidier/prevendo-notas-enem2016/blob/master/Prevendo_notas_ENEM.ipynb).

## Tópicos


* [Ambiente](#ambiente)
* [Dataset](#dataset)
* [Problemática](#problemática)
* [Descrição](#descrição)
* [Conclusão](#conclusão)


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
5. Pipeline

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

![feat_score](https://github.com/felipedidier/prevendo-notas-enem2016/blob/master/images/feature_score.png?raw=true)

A partir dos scores obtidos, é utilizado o P-value, que é interpretado no contexto como o nível de significância pré-escolhido, comumente definido por 5% (0,05). Também pode ser considerado como um nível de confiança de 95%. O P-value ajudará a definir quantas features serão escolhidas.

```python
scores_select = scores[scores['index'].isin(np.where(fs.pvalues_>0.05)[0])]
score_qttd = scores_select.shape[0]
print(f"Foram selecionadas {score_qttd} features de {train.shape[1]}.")
```

O resultado printado foi ```Foram selecionadas 38 features de 83.```. Com essa informação, realiza-se novamente o processo definindo a quantidade de 38 features (armazenado na variável ```score_qttd```).

```python
fs = SelectKBest(score_func=f_classif, k=score_qttd)
fs.fit(df_features,df_target)
cols = fs.get_support(indices=True)
df_features_new = df_features.iloc[:,cols]
train_fs = pd.DataFrame(fs.transform(df_features))

select_columns = []
for col in df_features_new.columns:
  select_columns.append(col)

train = train[select_columns]

feat_final = []
for col in train.columns:
  feat_final.append(col)

test = test[feat_final]
```

### 4. Modelagem

Selecionadas todas as features que serão utilizadas no modelo de previsão, partirmos para a etapa mais importante do projeto, onde serão definidos os modelos de Machine Learning para previsão das notas de matemática do ENEM 2016.

Primeiramente, algumas bibliotecas necessitam ser importadas:

```python
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import r2_score, mean_squared_error, accuracy_score
```

Para previsão do modelo foi utilizado Random Forest Regression (FRF), que possui uma abordagem mais precisa e robusta para mudanças nas features quando comparada com uma única árvore de regressão. A Random Forest utiliza de um conjunto de árvores de decisão aleatórias a fim de minimizar o overfitting de cada modelo individual.

```python
rfr = RandomForestRegressor()
```

Para se ter o melhor ajuste dos hiperparâmetros da RFR, foi utilizado o método de seleção ```GridSearchCV```, que recebe de entrada alguns valores de hiperparâmetros e roda vários modelos de RFR para encontrar qual possui a melhor performance. Dependendo da quantidade de registros no código, essa etapa pode ser um pouco demorada.

```python
parameters = {
    "n_estimators":[10, 50, 100, 250],
    "max_depth":[8, 10, 12]
}

cv = GridSearchCV(rfr,parameters,cv=5)

cv.fit(train.drop(columns='NU_NOTA_MT'),y_train.values.ravel())
```

Após executado, é possível conferir a performance de cada modelo.

```python
def display(results):
    print(f'Best parameters are: {results.best_params_}')
    print("\n")
    mean_score = results.cv_results_['mean_test_score']
    std_score = results.cv_results_['std_test_score']
    params = results.cv_results_['params']
    for mean,std,params in zip(mean_score,std_score,params):
        print(f'{round(mean,3)} + or -{round(std,3)} for the {params}')

display(cv)
```

Onde o é expresso por:

```
Best parameters are: {'max_depth': 8, 'n_estimators': 100}


0.451 + or -0.035 for the {'max_depth': 8, 'n_estimators': 10}
0.464 + or -0.031 for the {'max_depth': 8, 'n_estimators': 50}
0.465 + or -0.031 for the {'max_depth': 8, 'n_estimators': 100}
0.464 + or -0.032 for the {'max_depth': 8, 'n_estimators': 250}
0.443 + or -0.037 for the {'max_depth': 10, 'n_estimators': 10}
0.459 + or -0.033 for the {'max_depth': 10, 'n_estimators': 50}
0.46 + or -0.032 for the {'max_depth': 10, 'n_estimators': 100}
0.462 + or -0.032 for the {'max_depth': 10, 'n_estimators': 250}
0.429 + or -0.031 for the {'max_depth': 12, 'n_estimators': 10}
0.455 + or -0.033 for the {'max_depth': 12, 'n_estimators': 50}
0.459 + or -0.032 for the {'max_depth': 12, 'n_estimators': 100}
0.46 + or -0.032 for the {'max_depth': 12, 'n_estimators': 250}
```

Definido o modelo com os hiperparâmetros de melhor performance, realiza-se a previsão com a base ```test```.

```python
y_pred = cv.predict(test.drop(columns='NU_NOTA_MT'))
```

Para mensurar o resultado obtido, calcula-se o RMSE (erro médio quadrado). Quanto mais próximo de zero, melhor o resultado obtido.

```python
rmse_test = mean_squared_error(y_test, y_pred)**(1/2)
print(rmse_test) # 73.26
```

Para visualizar a comparação do valor previsto com o valor real, utilizou um gráfico scatter.

```python
sns.scatterplot(x=y_test, y=y_pred)
plt.show()
```
![model_prevreal](https://github.com/felipedidier/prevendo-notas-enem2016/blob/master/images/model_prevreal.png?raw=true)

### 5. Pipeline

Pipeline é uma série de etapas padronizadas que organiza o processo de transformação e modelagem doi projeto. As bibliotecas utilizadas seguem abaixo:

```python
## Bibliotecas utilizadas
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer, TransformedTargetRegressor

import pickle as pk
```

Novamente realiza-se a separação em treino e test. ```df_new``` foi uma cópia realizada do ```df``` antes das transformações de features.

```python
train, test = train_test_split(df_new, test_size=0.2, random_state=42)
X_train, y_train = train.drop(columns='NU_NOTA_MT'), train[target]
X_test, y_test = test.drop(columns='NU_NOTA_MT'), test[target]
```

Os métodos de transformação de features são importados, assim como as próprias features.

```python
z_score = StandardScaler()
oe = OrdinalEncoder()
one_hot = OneHotEncoder()
```

```
feat_num = ['NU_IDADE', 'NU_NOTA_CN', 'NU_NOTA_CH', 'NU_NOTA_LC']
feat_cat1 = ['TP_SEXO', 'Q006', 'Q007', 'Q008', 'Q009', 'Q010', 'Q011', 'Q012', 'Q013', 'Q014', 'Q015', 'Q016', 'Q017', 'Q018', 'Q019', 'Q020', 'Q021', 'Q022', 'Q023', 'Q024', 'Q025', 'Q026', 'Q042', 'Q043', 'Q045', 'Q047', 'Q048', 'IN']
feat_cat2 = ['SG_UF_RESIDENCIA', 'TP_ESTADO_CIVIL', 'TP_COR_RACA', 'TP_NACIONALIDADE', 'TP_ST_CONCLUSAO']
```

Com o auxilio de ```ColumTransformer```, é construído o preprocessamento e transformado em DataFrame.

```python
preprocess = ColumnTransformer(
                    [   
                        ('StdTransf', z_score, feat_num),
                        ('Ordinal', oe, feat_cat1),
                        ('OneHot', one_hot, feat_cat2),
                        
                    ], remainder='passthrough')

dt_tf = preprocess.fit_transform(X_train, y_train)
pd.DataFrame(data = dt_tf)
```

Utilizando do ```TransformedTargetRegressor```, é construído o modelo de Machine Learning selecionado, o Random Forest Regressor.

```python
model = TransformedTargetRegressor(regressor=RandomForestRegressor(n_estimators=250, bootstrap=True, criterion='mse', max_depth=8, max_features='auto', random_state=42), transformer = z_score)
```

Possuindo o preprocessamento e o modelo definidos, constroi-se o Pipeline.

```python
pipe = Pipeline([("pre", preprocess), ("tree", model)])
pipe.fit(X_train, y_train)
```

Com o Pipeline rodado, agora pode-se testar se os resultados ficam próximos aos resultados obtidos na etapa 05.

```python
pred_value = pipe.predict(X_test)

rmse_dt = mean_squared_error(y_test, pred_value) ** (1/2)

print("Test set RMSE of dt: {:.3f}".format(rmse_dt)) # 73.2
print(f"Test set R2 of dt {r2_score(y_test, pred_value):.3f}") # 0.466
```

#### Oportunidade para Deploy

Com o Pipeline construído é possível gerar um arquivo pickle para implementações de Deploy.

```python
pk.dump(pipe, open('model_rfr.pkl', 'wb'))
```

## Conclusão

Percebe-se que as características socioeconômicas dos participantes, que são definidas principalmente pelas features de Q006 à Q047, possuem considerável influência nos resultados da prova de matemática, o que deixa claro que alunos com menos acesso à educação, renda familiar mais baixa, dificuldade de acesso à internet e computadores necessitam de maior atenção e investimento por parte dos órgãos público para que tenhamos um modelo de ensino mais acessível e democrático.
