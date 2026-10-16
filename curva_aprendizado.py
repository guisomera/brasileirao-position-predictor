from main import tabela
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error
import pandas as pd

COLUNAS = ["Posição_10", "Vitorias_10", "Empates_10",
           "Gols feitos_10", "Gols sofridos_10", "qualidade_elenco"]

anos = [2018, 2020, 2021, 2022, 2023, 2024, 2025]


def curva_aprendizado(ano_teste, tabela):
    resultados = []
    anos_treino = []
    anos_disponiveis = [a for a in anos if a != ano_teste]  # tira o ano de teste

    teste = tabela[tabela["Ano"] == ano_teste]
    mae_baseline = mean_absolute_error(teste["Posição_38"], teste["Posição_10"])  # "ninguém muda"

    for ano in anos_disponiveis:
        anos_treino.append(ano)
        treino = tabela[tabela["Ano"].isin(anos_treino)]

        modelo = LinearRegression()
        modelo.fit(treino[COLUNAS], treino["Posição_38"])

        mae_treino = mean_absolute_error(treino["Posição_38"], modelo.predict(treino[COLUNAS]))
        mae_teste = mean_absolute_error(teste["Posição_38"], modelo.predict(teste[COLUNAS]))

        resultados.append({
            "ano_teste": ano_teste,
            "temporadas treinadas": len(anos_treino),
            "mae_treino": mae_treino,
            "mae_teste": mae_teste,
            "mae_baseline": mae_baseline,
        })

    return pd.DataFrame(resultados)


todos = [curva_aprendizado(ano_teste, tabela) for ano_teste in anos]
resultado = pd.concat(todos)

# curva média (cada ponto = média de 7 testes)
print(resultado.groupby("temporadas treinadas")[["mae_treino", "mae_teste", "mae_baseline"]].mean())

# detalhe por ano de teste, só com 6 temporadas treinadas
print(resultado[resultado["temporadas treinadas"] == 6][["ano_teste", "mae_teste", "mae_baseline"]])