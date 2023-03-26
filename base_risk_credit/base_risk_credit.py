import pickle

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import plotly.express as px
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import GaussianNB
from sklearn.preprocessing import LabelEncoder, StandardScaler


class BaseRiskCredit:
    def data_pre_processment():
        base_risco_credito = pd.read_csv("risco_credito.csv")

        print(type(base_risco_credito))

        # separando atributos previsores
        X_risco_credito = base_risco_credito.iloc[:, 0:4].values

        # separando atributo classificador
        y_risco_credito = base_risco_credito.iloc[:, 4].values

        # instanciação dos LabelEncoder, para converter registros que são strings em números
        label_encoder_historia = LabelEncoder()
        label_encoder_divida = LabelEncoder()
        label_encoder_garantia = LabelEncoder()
        label_encoder_renda = LabelEncoder()

        # conversão dos valores nas colunas que possuem registros em strings para números
        X_risco_credito[:, 0] = label_encoder_historia.fit_transform(
            X_risco_credito[:, 0]
        )
        X_risco_credito[:, 1] = label_encoder_divida.fit_transform(
            X_risco_credito[:, 1]
        )
        X_risco_credito[:, 2] = label_encoder_garantia.fit_transform(
            X_risco_credito[:, 2]
        )
        X_risco_credito[:, 3] = label_encoder_renda.fit_transform(X_risco_credito[:, 3])

        # gera arquivo pré-processado
        with open("risco_credito.pkl", "wb") as f:
            pickle.dump([X_risco_credito, y_risco_credito], f)

    def execute_algorithm():
        # importação do algoritmo naive bayes
        naive_risco_credito = GaussianNB()

        # instanciando o arquivo .pkl salvo com os dados ja pre-processados
        base_risco_credito = pd.read_pickle("risco_credito.pkl")

        # separando as features do arquivo .pkl pre-processado
        X_risco_credito = base_risco_credito[0]

        # separando as classes
        y_risco_credito = base_risco_credito[1]

        # geração da tabela de probabilidades, passando como parametro os atributos previsores e o atributo classificador
        # é aqui que o algoritmo é treinado com os dados
        naive_risco_credito.fit(X_risco_credito, y_risco_credito)

        # executa a previsão baseado em novas entradas de dados
        # dados 1: historia boa(0), divida alta(0), garantias nenhuma(1), renda > 35(2)
        # dados 2: historia ruim(2), divida alta(0), garantias adequada(0), renda < 15(0)
        previsao = naive_risco_credito.predict([[0, 0, 1, 2], [2, 0, 0, 0]])

        print(previsao)
