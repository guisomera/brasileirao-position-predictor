import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error

COLUNAS = ["Posição_10", "Vitorias_10", "Empates_10",
           "Gols feitos_10", "Gols sofridos_10", "qualidade_elenco"]

def montar_tabela():
    anos = []
    for ano in [2018, 2020, 2021, 2022, 2023, 2024, 2025]:
        r10 = pd.read_csv(f"br_{ano}_10.csv")
        r38 = pd.read_csv(f"br_{ano}_38.csv")
        df = pd.merge(r10, r38, on="Clube", suffixes=("_10", "_38"))
        df["Ano"] = ano
        anos.append(df)
    return pd.concat(anos, ignore_index=True)

tabela = montar_tabela()

resultados = []
for ano_teste in [2018, 2020, 2021, 2022, 2023, 2024, 2025]:
    treino = tabela[tabela["Ano"] != ano_teste]
    teste = tabela[tabela["Ano"] == ano_teste]

    model = LinearRegression()
    model.fit(treino[COLUNAS], treino["Posição_38"])

    mae_modelo = mean_absolute_error(teste["Posição_38"], model.predict(teste[COLUNAS]))
    mae_baseline = mean_absolute_error(teste["Posição_38"], teste["Posição_10"])

    resultados.append((ano_teste, mae_modelo, mae_baseline))
    print(f"Ano: {ano_teste} | Modelo: {mae_modelo:.2f} | Baseline: {mae_baseline:.2f}"
          f" | treino={len(treino)} teste={len(teste)}")
    if ano_teste == 2024:
        t = teste.sort_values("Posição_10")   # ordena do 1º ao 20º da rodada 10
        previsao = model.predict(t[COLUNAS])
        for clube, p10, p38, prev in zip(t["Clube"], t["Posição_10"], t["Posição_38"], previsao):
            print(f"{clube:20} rodada10={p10:.0f} final={p38:.0f} modelo={prev:.1f}")

modelo_medio = np.mean([r[1] for r in resultados])
baseline_medio = np.mean([r[2] for r in resultados])
vitorias = sum(1 for r in resultados if r[1] < r[2])

print(f"\nMédia modelo: {modelo_medio:.2f} | Média baseline: {baseline_medio:.2f}")
print(f"Modelo ganhou do baseline em {vitorias} de {len(resultados)} anos")
