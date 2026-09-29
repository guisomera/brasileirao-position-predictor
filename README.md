# Brasileirão Position Predictor

A linear regression model that tries to predict a team's final league position in the Brazilian Série A, using its performance in the first 10 matchdays plus a measure of squad quality.

Built as hands-on practice for the Machine Learning Specialization (Coursera, Andrew Ng) and the book *Hands-On Machine Learning with Scikit-Learn, Keras, and TensorFlow* — first time using Scikit-Learn and the first regression with real multiple features.

## Context and pivot

The original idea was different: predict a team's **average attendance** based on its performance, testing the hypothesis that "a team playing well attracts more fans." After gathering real Brasileirão datasets, it became clear that no available source had per-match attendance numbers — the data needed to support the question simply wasn't there.

Instead of dropping the project, the question was changed while keeping the same kind of data already collected: **given a team's performance in the first 10 matchdays, can we predict what position it will finish in (matchday 38)?**

## Dataset

- Source: Transfermarkt standings tables and squad market values, scraped and formatted into CSV with help from Claude Cowork.
- Years used: 2018, 2020, 2021, 2022, 2023, 2024, 2025 — all seven used in cross-validation (see Methodology). 2019 was excluded for being an outlier season (Flamengo won the title with an unusually large margin over the runner-up), a manual decision not yet re-tested against the current validation pipeline.
- Each row represents **one team in one season** (20 teams × 7 seasons = 140 rows).
- Dataset size was sized using the rule of thumb of 10–15 rows per feature (7 features → minimum of ~105–130 rows).

## Features (X) and target (y)

**Inputs (matchday 10 data):**
- Position
- Wins
- Draws
- Goals scored
- Goals conceded
- **Squad market value** (`qualidade_elenco`) — total squad value at the start of the season, normalized as each team's share of that season's total (`team value ÷ sum of all 20 teams`), to correct for market inflation across different years

**Target:** Final position (matchday 38)

Deliberately left out of X:
- **Wins_38, Draws_38, Losses_38, Points_38** — data leakage. These columns are derived directly from the very outcome the model should predict, so including them would hand the model the answer.
- **Points and Losses (matchday 10)** — tested and removed: exact mathematical redundancy (`Points = 3×Wins + Draws`, `Losses = 10 − Wins − Draws`). Removing both left MAE unchanged, confirming they added no independent signal.
- **Goal difference** — redundant information: GD = goals scored − goals conceded, already represented as separate features.
- **Club name** — used only as a key to join the matchday 10, matchday 38, and market value tables for the same year, never used as a model feature.

## Methodology

1. Merge matchday 10 with matchday 38 **per year**, using club name as the key, to avoid mixing different seasons.
2. Concatenate all seven seasons into a single dataset.
3. **Leave-one-year-out cross-validation** instead of a single fixed test set: each of the 7 seasons takes a turn as the test set while the model trains on the other 6, then the years rotate. This replaced an earlier single-split design (train on 2018–2024, test only on 2025), which turned out to measure only one season's luck rather than the model's general behavior.
4. Train with Scikit-Learn's `LinearRegression`.
5. Evaluate with **MAE** (Mean Absolute Error), chosen over MSE because it keeps the same unit as the problem (positions), allowing a direct reading of the error.
6. Compare against a naive baseline every year: predicting that a team finishes in the same position it held at matchday 10.

## Results

| Year | Model MAE | Baseline MAE |
|------|-----------|---------------|
| 2018 | 3.53 | 4.40 |
| 2020 | 3.27 | 2.70 |
| 2021 | 3.00 | 3.65 |
| 2022 | 2.75 | 3.60 |
| 2023 | 2.56 | 2.40 |
| 2024 | 3.48 | 4.70 |
| 2025 | 2.64 | 2.50 |
| **Average** | **3.04** | **3.42** |

The model beats the baseline on average and wins outright in **4 of 7 seasons**. The pattern by season matters more than the average: the model tends to win in seasons where the table shuffled a lot between matchday 10 and 38 (baseline MAE above ~3.5) and tends to lose in "quiet" seasons where matchday 10 already closely resembled the final table (baseline MAE below ~2.7).

Looking at individual teams in 2024 (a "shuffled" season): the model correctly pulled Athletico-PR down from 4th to a predicted ~7th (it actually finished 17th) and kept Flamengo near 1st (it finished 3rd). It also moved Corinthians up from 18th toward a predicted ~12th (it actually finished 7th) — the right direction, but not enough magnitude. This suggests the squad-value feature carries real signal, particularly for expensive squads underperforming, though it's a read from a handful of teams in one season, not a proven effect.

## Takeaways

- Before picking the tool, make sure the data behind the question actually exists — the original project idea died from lack of data, not from a modeling failure.
- Data leakage can be subtle: columns derived from the target itself (matchday-38 points, wins) looked like good features at first glance.
- Exact mathematical redundancy (Points from Wins+Draws) is easy to test directly by dropping the column and checking whether MAE moves — no guesswork needed.
- A single train/test split on one season measures that season as much as it measures the model. Leave-one-year-out cross-validation was necessary to tell "the model is good" apart from "2025 happened to be a good year for it."
- Normalizing feature scale doesn't change predictions or MAE in plain `LinearRegression` — the model already compensates for scale through its coefficients. This only matters for algorithms like gradient descent or regularized regression.
- Adding a feature that "makes intuitive sense" (squad market value) doesn't guarantee a big MAE drop — it moved the average from 2.73 to 3.04 only in the sense of a more honest, cross-validated number, while genuinely helping in specific, identifiable cases (expensive squads collapsing).
- The rows-per-feature rule of thumb (10–15:1) helps size how much data is needed before going out to collect it.

## What I'd do differently starting over

- Set up cross-validation from day one, instead of trusting a single 20-row test season.
- Treat squad quality as a context feature conceptually separate from matchday-10 performance features from the start, rather than bolting it on later.
- Test the 2019 exclusion formally against the validation pipeline instead of leaving it as a one-off manual call.

## Known limitations

- 140 rows is a small dataset for a sport with this much year-to-year variability (baseline MAE alone ranges from 2.4 to 4.7 depending on the season).
- Linear regression can't see mid-season signings, injuries, or coaching changes — all plausible drivers behind cases like Corinthians' 2024 turnaround.
- 2019 was excluded by a manual, un-validated decision.
