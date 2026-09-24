# Brasileirão Position Predictor

A linear regression model that tries to predict a team's final league position in the Brazilian Série A, using only its performance in the first 10 matchdays.

Built as hands-on practice for the Machine Learning Specialization (Coursera, Andrew Ng) and the book *Hands-On Machine Learning with Scikit-Learn, Keras, and TensorFlow* — first time using Scikit-Learn and the first regression with real multiple features.

## Context and pivot

The original idea was different: predict a team's **average attendance** based on its performance, testing the hypothesis that "a team playing well attracts more fans." After gathering real Brasileirão datasets, it became clear that no available source had per-match attendance numbers — the data needed to support the question simply wasn't there.

Instead of dropping the project, the question was changed while keeping the same kind of data already collected: **given a team's performance in the first 10 matchdays, can we predict what position it will finish in (matchday 38)?**

## Dataset

- Source: Transfermarkt standings tables, scraped and formatted into CSV with help from Claude Cowork.
- Years used: 2018, 2020, 2021, 2022, 2023, 2024 (training) and 2025 (test). 2019 was excluded for being an outlier season (Flamengo won the title with an unusually large margin over the runner-up).
- Each row represents **one team in one season** (20 teams × 7 seasons = 140 rows).
- Dataset size was sized using the rule of thumb of 10–15 rows per feature (7 features → minimum of ~105–130 rows).

## Features (X) and target (y)

**Inputs (matchday 10 data):**
- Position
- Wins
- Losses
- Draws
- Goals scored
- Goals conceded
- Points

**Target:** Final position (matchday 38)

Deliberately left out of X:
- **Wins_38, Draws_38, Losses_38, Points_38** — data leakage. These columns are derived directly from the very outcome the model should predict, so including them would hand the model the answer.
- **Goal difference** — redundant information: GD = goals scored − goals conceded, already represented as separate features. Keeping GD as well would add no new information.
- **Club name** — used only as a key to join the matchday 10 and matchday 38 tables for the same year, never used as a model feature.

## Methodology

1. Merge matchday 10 with matchday 38 **per year**, using club name as the key, to avoid mixing different seasons.
2. Concatenate the training years (2018–2024) into a single dataset.
3. **Temporal split** instead of a random `train_test_split`: the entire 2025 season was held out as the test set, never seen by the model during training. This avoids evaluating the model on teams too similar to the ones it trained on (squads change little from one season to the next), making the test more rigorous than a random split would be.
4. Train with Scikit-Learn's `LinearRegression`.
5. Evaluate with **MAE** (Mean Absolute Error), chosen over MSE because it keeps the same unit as the problem (positions), allowing a direct reading of the error.

## Results

**MAE = 2.73** — on average, the model is off by about 3 positions in the table.

The model gets the overall direction right: the team with the lowest predicted position (Palmeiras) and the one with the highest (Juventude) matched the real extremes of the 2025 table, with minor swaps between close neighbors (e.g., Palmeiras and Flamengo swapped at the top, Sport and Juventude swapped at the bottom). The error is larger in the middle of the table — expected, since only 10 matchdays (26% of the season) carry little information about the performance swings that usually decide the middle of the Brasileirão table.

## Takeaways

- Before picking the tool, make sure the data behind the question actually exists — the original project idea died from lack of data, not from a modeling failure.
- Data leakage can be subtle: columns derived from the target itself (matchday-38 points, wins) looked like good features at first glance.
- Picking the right metric (MAE vs. MSE) changes how interpretable the result is, not just its value.
- `argsort` (numpy) turns continuous predictions into a ranking, bridging the gap between "regression outputs raw numbers" and "I wanted an ordered table."
- The rows-per-feature rule of thumb (10–15:1) helps size how much data is needed before going out to collect it.
