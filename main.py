import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error

def load_data():
    
    br_2018_10 = pd.read_csv("br_2018_10.csv")
    br_2018_38 = pd.read_csv("br_2018_38.csv")  
    br_2020_10 = pd.read_csv("br_2020_10.csv")
    br_2020_38 = pd.read_csv("br_2020_38.csv")
    br_2021_10 = pd.read_csv("br_2021_10.csv")
    br_2021_38 = pd.read_csv("br_2021_38.csv")
    br_2022_10 = pd.read_csv("br_2022_10.csv")
    br_2022_38 = pd.read_csv("br_2022_38.csv")
    br_2023_10 = pd.read_csv("br_2023_10.csv")
    br_2023_38 = pd.read_csv("br_2023_38.csv")
    br_2024_10 = pd.read_csv("br_2024_10.csv")
    br_2024_38 = pd.read_csv("br_2024_38.csv")
    br_2025_10 = pd.read_csv("br_2025_10.csv")
    br_2025_38 = pd.read_csv("br_2025_38.csv")

    ano_2018 = pd.merge(br_2018_10,br_2018_38, on="Clube", suffixes=("_10", "_38"))
    ano_2020 = pd.merge(br_2020_10,br_2020_38, on="Clube", suffixes=("_10", "_38"))
    ano_2021 = pd.merge(br_2021_10,br_2021_38, on="Clube", suffixes=("_10", "_38"))
    ano_2022 = pd.merge(br_2022_10,br_2022_38, on="Clube", suffixes=("_10", "_38"))
    ano_2023 = pd.merge(br_2023_10,br_2023_38, on="Clube", suffixes=("_10", "_38"))
    ano_2024 = pd.merge(br_2024_10,br_2024_38, on="Clube", suffixes=("_10", "_38"))
    ano_2025 = pd.merge(br_2025_10,br_2025_38, on="Clube", suffixes=("_10", "_38"))

    ano_2018["Ano"] = 2018
    ano_2020["Ano"] = 2020
    ano_2021["Ano"] = 2021
    ano_2022["Ano"] = 2022
    ano_2023["Ano"] = 2023
    ano_2024["Ano"] = 2024
    ano_2025["Ano"] = 2025 

    anos = [ano_2018, ano_2020, ano_2021, ano_2022, ano_2023, ano_2024] #Tirei 2025 para teste  

    for ano in anos:
        nan = ano.isna().any().any()
        if nan:
            print(ano.columns)
    
    anos_final = pd.concat(anos)
    anos_final.to_csv("brasileirao_2018_2024.csv", index=False)

    X = np.array(anos_final[["Posição_10", "Vitorias_10", "Derrotas_10", "Empates_10", "Gols feitos_10", "Gols sofridos_10", "Pontos_10"]])
    y=np.array(anos_final["Posição_38"])
    X_teste = np.array(ano_2025[["Posição_10", "Vitorias_10", "Derrotas_10", "Empates_10", "Gols feitos_10", "Gols sofridos_10", "Pontos_10"]])
    y_teste = np.array(ano_2025["Posição_38"])

    return X, y, X_teste, y_teste, ano_2025["Clube"]

X, y, X_teste, y_teste, times_2025 = load_data() 

model = LinearRegression()
model.fit(X,y)
erro= mean_absolute_error(y_teste, model.predict(X_teste))
resultado = model.predict(X_teste)
ordem = np.argsort(resultado)
resultado_ordenado = resultado[ordem]
clubes_ordenados = times_2025.values[ordem]

for clube, posicao in zip(clubes_ordenados, resultado_ordenado):
    print(clube, posicao)

