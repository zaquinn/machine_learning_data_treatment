import pickle

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import plotly.express as px
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler


class BaseCredit:
    def execute():
        base_credit = pd.read_csv("credit_data.csv")

        # plt.hist(x=base_credit["loan"])

        # plt.show()

        # grafico = px.scatter_matrix(
        #     base_credit, dimensions=["age", "income", "loan"], color="default"
        # )
        # grafico.show()

        base_credit.loc[
            base_credit["age"] < 0, "age"
        ] = 40.92  # seta valor médio para os valores negativos

        # print(base_credit.head(27))

        base_credit["age"].fillna(base_credit["age"].mean(), inplace=True)
        base_credit.loc[
            pd.isnull(base_credit["age"])
        ]  # seta valor médio para valores nulos

        X_credit = base_credit.iloc[:, 1:4].values  # seleciona as features

        # print(X_credit[:, 0].min(), X_credit[:, 1].min(), X_credit[:, 2].min())

        y_credit = base_credit.iloc[
            :, 4
        ].values  # seleciona o atributo de classe, no caso a coluna default, que determina se pagou ou não a divida

        scaler_credit = StandardScaler()
        X_credit = scaler_credit.fit_transform(X_credit)

        # print(X_credit[:, 0].min(), X_credit[:, 1].min(), X_credit[:, 2].min())

        # print(X_credit[:, 0].max(), X_credit[:, 1].max(), X_credit[:, 2].max())

        # divide a base de dados e suas colunas de atributos previsores em bases de treinamento e teste, assim como a classe(coluna de resultados) dessas bases de dados
        # parametro test_size define a porcentagem da base de dados que sera separada para testes,nesse caso, 25%
        # random_state=0 faz com que os dados selecionados e divididos sejam sempre os mesmos em toda execução
        (
            X_credit_treinamento,
            X_credit_teste,
            y_credit_treinamento,
            y_credit_teste,
        ) = train_test_split(X_credit, y_credit, test_size=0.25, random_state=0)

        # utiliza da biblioteca pickle para gerar os arquivos com as bases de dados de treinamento e teste já pré-processadas
        # evitando ter que realizar todo o pré-processamento acima a cada execução
        with open("credit.pkl", mode="wb") as f:
            pickle.dump(
                [
                    X_credit_treinamento,
                    y_credit_treinamento,
                    X_credit_teste,
                    y_credit_teste,
                ],
                f,
            )
