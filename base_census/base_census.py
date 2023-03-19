import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import plotly.express as px
import seaborn as sns
from sklearn.preprocessing import StandardScaler


class BaseSensus:
    def execute():
        base_census = pd.read_csv("census.csv")

        print(
            base_census.describe()
        )  # método util pra observar os dados e procurar valores inconsistentes, como valores negativos

        print(
            base_census.isnull().sum()
        )  # soma a quantidade de valores nulos/faltantes pra checar se a base de dados possui valores faltantes/nulos
