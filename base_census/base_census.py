import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import plotly.express as px
import seaborn as sns
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import LabelEncoder, OneHotEncoder, StandardScaler


class BaseSensus:
    def execute():
        base_census = pd.read_csv("census.csv")

        # print(
        #     base_census.describe()
        # )  # método util pra observar os dados e procurar valores inconsistentes, como valores negativos

        # print(
        #     base_census.isnull().sum()
        # )  # soma a quantidade de valores nulos/faltantes pra checar se a base de dados possui valores faltantes/nulos

        # print(np.unique(base_census["income"], return_counts=True))

        # sns.countplot(x=base_census["income"])  # gera o plot especificado na variavel x, necessario usar com a chamada do plt.show() abaixo

        # plt.hist(x=base_census["age"])  # gera o histograma especificado na variavel x

        # plt.show()  # exibe o plot do seaborn/pyplot etc

        # grafico = px.treemap(base_census, path=["workclass", "age"]) # agrupa os dados em uma relação de workclass e age

        # grafico = px.treemap(base_census, path=["occupation", "relationship", "age"])

        # grafico = px.parallel_categories(
        #     base_census, dimensions=["occupation", "relationship"]
        # )  # grafico para visualização paralela dos relacionamentos citados

        # grafico.show()

        # seleciona as features/previsores, das colunas 0 até 13, 14 não incluso. O ":" ao inicio seleciona todos os dados. O ".values" ao final converte para um array do NumPy
        # print(base_census.columns) # retorna quantidade de colunas
        X_census = base_census.iloc[:, 0:14].values

        # seleciona a classe, ou seja, a coluna da tabela que possui o resultado do conjunto de features/previsores, no caso desse database, o salário
        y_census = base_census.iloc[:, 14].values

        # instanciação do objeto LabelEncoder, que sera usado para transformar dados em strings do database em formato no numerico, para aplicação das formulas
        # matematicas dos algoritmos de aprendizado de maquina

        # label_encoder_workclass = LabelEncoder()
        # label_encoder_education = LabelEncoder()
        # label_encoder_marital = LabelEncoder()
        # label_encoder_occupation = LabelEncoder()
        # label_encoder_relationship = LabelEncoder()
        # label_encoder_country = LabelEncoder()
        # label_encoder_race = LabelEncoder()
        # label_encoder_sex = LabelEncoder()

        # converte todos os registros da coluna 1 (workclass), da base de dados, antes strings, para númericos
        # X_census[:, 1] = label_encoder_workclass.fit_transform(X_census[:, 1])
        # # faz o mesmo processos para as demais colunas do database
        # X_census[:, 3] = label_encoder_education.fit_transform(X_census[:, 3])
        # X_census[:, 5] = label_encoder_marital.fit_transform(X_census[:, 5])
        # X_census[:, 6] = label_encoder_occupation.fit_transform(X_census[:, 6])
        # X_census[:, 7] = label_encoder_relationship.fit_transform(X_census[:, 7])
        # X_census[:, 8] = label_encoder_race.fit_transform(X_census[:, 8])
        # X_census[:, 9] = label_encoder_sex.fit_transform(X_census[:, 9])
        # X_census[:, 13] = label_encoder_country.fit_transform(X_census[:, 13])

        # transforma as colunas que são strings, passadas na lista com seus respectivos indices, em novas colunas, gerando uma logica dessa forma:
        # digamos que a coluna cor no database possui tres cores que se alternam nos registros: vermelho, azul e rosa. As novas colunas ficariam assim:
        # para uma linha de registro com a cor azul, cria-se 3 novas colunas com os valores: 0 1 0. Sendo o primeiro 0 a cor vermelha, o segundo pra azul e a segunda pra cor rosa.
        # remainder='passthrough' serve para não deletar as colunas que possuem as strings após criar as novas colunas convertidas

        onehotencoder_census = ColumnTransformer(
            transformers=[("OneHot", OneHotEncoder(), [1, 3, 5, 6, 7, 8, 9, 13])],
            remainder="passthrough",
        )

        # aplica o método de conversão acima a variavel instanciada com as features/previsores do data base, gerando um novo array com as novas colunas convertidas de strings para numeros
        # esse é o método recomendado para o tratamento de strings no data base, impedindo problemas de peso maior entre diversos registros por conta do
        # indice gerado pelo metodo anterior, que pode fazer com que alguns algoritmos interpretem valores de indice mais alto como mais relevantes
        X_census = onehotencoder_census.fit_transform(X_census).toarray()

        print(
            X_census.shape
        )  # pode-se observar que a base de dados agora possui 108 colunas, ao invés das 14 anteriores
