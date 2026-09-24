# Brasileirão Position Predictor

Modelo de regressão linear que tenta prever a posição final de um time no Campeonato Brasileiro Série A, usando apenas o desempenho dele nas 10 primeiras rodadas.

Projeto feito como prática da Especialização em Aprendizado de Máquina (Coursera, Andrew Ng) e do livro *Aprendizado de Máquina com Scikit-Learn, Keras e TensorFlow* — primeira vez usando Scikit-Learn e primeira regressão com múltiplas features de verdade.

## Contexto e pivô

A ideia original era outra: prever o **público médio** de um time a partir do seu desempenho, testando a hipótese de que "time que joga bem atrai mais torcedores". Depois de levantar datasets reais do Brasileirão, ficou claro que nenhuma fonte disponível trazia número de público por jogo — o dado que sustentava a pergunta simplesmente não existia nas mãos.

Em vez de abandonar o projeto, a pergunta foi trocada, mantendo o mesmo tipo de dado já levantado: **dado o desempenho de um time nas 10 primeiras rodadas, dá pra prever em que posição ele vai terminar o campeonato (rodada 38)?**

## Dataset

- Fonte: tabelas de classificação do Transfermarkt, raspadas e formatadas em CSV com ajuda do Claude Cowork.
- Anos usados: 2018, 2020, 2021, 2022, 2023, 2024 (treino) e 2025 (teste). 2019 foi descartado por ter uma temporada fora do padrão (Flamengo campeão com vantagem incomum sobre o 2º colocado).
- Cada linha representa **um time em uma temporada** (20 times × 7 temporadas = 140 linhas).
- Tamanho do dataset dimensionado pela regra prática de 10–15 linhas por feature (7 features → mínimo de ~105–130 linhas).

## Features (X) e alvo (y)

**Entradas (dados da rodada 10):**
- Posição
- Vitórias
- Derrotas
- Empates
- Gols feitos
- Gols sofridos
- Pontos

**Alvo:** Posição final (rodada 38)

Ficaram de fora do X, por decisão deliberada:
- **Vitórias_38, Empates_38, Derrotas_38, Pontos_38** — vazamento de dado. Essas colunas são calculadas a partir do próprio resultado que o modelo deveria prever, então incluí-las seria dar a resposta pronta pro modelo.
- **Saldo de gols** — informação redundante: SG = Gols feitos − Gols sofridos, já representados como features separadas. Manter o SG também não agregaria informação nova.
- **Nome do clube** — usado só como chave para juntar as tabelas de rodada 10 e 38 do mesmo ano (join), nunca entrou como feature do modelo.

## Metodologia

1. Merge de rodada 10 com rodada 38 **por ano**, usando o nome do clube como chave, evitando misturar temporadas diferentes.
2. Concatenação dos anos de treino (2018–2024) em um único dataset.
3. **Divisão temporal** ao invés de `train_test_split` aleatório: a temporada de 2025 inteira foi reservada como teste, nunca vista pelo modelo durante o treino. Essa escolha evita que o modelo seja avaliado com times muito parecidos aos de treino (elencos mudam pouco de uma temporada pra outra), tornando o teste mais rigoroso do que um split aleatório teria sido.
4. Treino com `LinearRegression` do Scikit-Learn.
5. Avaliação com **MAE** (Mean Absolute Error), escolhido em vez de MSE por manter a mesma unidade do problema (posições), permitindo uma leitura direta do erro.

## Resultado

**MAE = 2.73** — em média, o modelo erra por cerca de 3 posições na tabela.

O modelo acerta bem a direção geral: o time com a menor posição prevista (Palmeiras) e o com a maior (Juventude) bateram com os extremos reais da tabela de 2025, com pequenas trocas entre vizinhos próximos (ex: Palmeiras e Flamengo trocados no topo, Sport e Juventude trocados no fim). No meio da tabela o erro é maior — esperado, já que só 10 rodadas (26% do campeonato) carregam pouca informação sobre reviravoltas de rendimento que costumam definir o meio da tabela no Brasileirão.

## Aprendizados

- Antes de escolher a ferramenta, é preciso garantir que o dado que sustenta a pergunta existe — a primeira ideia do projeto morreu por falta de dado, não por falha de modelagem.
- Vazamento de dado (data leakage) pode ser sutil: colunas calculadas a partir do próprio alvo (Pontos, Vitórias da rodada 38) pareciam boas features à primeira vista.
- Escolher a métrica certa (MAE vs. MSE) muda a interpretabilidade do resultado, não só o valor do erro.
- `argsort` (numpy) permite transformar previsões contínuas em um ranking, resolvendo o descompasso entre "regressão dá números soltos" e "eu queria uma tabela ordenada".
- Regra prática de linhas por feature (10–15:1) ajuda a dimensionar quanto dado é necessário antes de sair coletando.
- 
