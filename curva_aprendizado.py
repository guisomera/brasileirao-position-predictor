from main import tabela
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error
import pandas as pd

COLUNAS = ["Posição_10", "Vitorias_10", "Empates_10",
           "Gols feitos_10", "Gols sofridos_10", "qualidade_elenco"]
           
anos = [2018, 2020, 2021, 2022, 2023, 2024]

def curva_aprendizado(ano_teste, tabela):
    curva_de_aprendizado = []
    anos_treino = []
    temp_treino=0
    for ano in anos:
        anos_treino.append(ano)
        temp_treino += 1 
        treino = tabela[tabela["Ano"].isin(anos_treino)]
        teste = tabela[tabela["Ano"] == ano_teste]
        modelo = LinearRegression()
        modelo.fit(treino[COLUNAS], treino["Posição_38"]) #X_treino e Y_treino
        mae_treino= mean_absolute_error(treino["Posição_38"], modelo.predict(treino[COLUNAS]))
        mae_teste= mean_absolute_error(teste["Posição_38"], modelo.predict(teste[COLUNAS]))
        curva_de_aprendizado.append({"temporadas treinadas": temp_treino, "mae_treino": mae_treino, "mae_teste": mae_teste})
        
    return pd.DataFrame(curva_de_aprendizado)

print(curva_aprendizado(2025, tabela))