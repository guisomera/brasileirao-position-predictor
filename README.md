# Brasileirão Position Predictor

A linear regression model that tries to predict a team's final league position in the Brazilian Série A, using its performance in the first 10 matchdays plus a measure of squad quality.

Built as hands-on practice for the Machine Learning Specialization (Coursera, Andrew Ng) and the book *Hands-On Machine Learning with Scikit-Learn, Keras, and TensorFlow* — first time using Scikit-Learn and the first regression with real multiple features.

## Context and pivot

The original idea was different: predict a team's average attendance based on its performance, testing the hypothesis that "a team playing well attracts more fans." After gathering real Brasileirão datasets, it became clear that no available source had per-match attendance numbers — the data needed to support the question simply wasn't there.

Instead of dropping the project, the question was changed while keeping the same kind of data already collected: given a team's performance in the first 10 matchdays, can we predict what position it will finish in (matchday 38)?

## Dataset

- **Source:** Transfermarkt standings tables and squad market values, scraped and formatted into CSV with help from Claude Cowork.
- **Years used:** 2018, 2020, 2021, 2022, 2023, 2024, 2025 — all seven used in cross-validation (see Methodology). 2019 was excluded for being an outlier season (Flamengo won the title with an unusually large margin over the runner-up), a manual decision not yet re-tested against the current validation pipeline.
- Each row represents one team in one season (20 teams × 7 seasons = 140 rows).
- Dataset size was sized using the rule of thumb of 10–15 rows per feature (6 features → minimum of ~60–90 rows; 120 training rows per fold comfortably clears it).

### Features (X) and target (y)

Inputs (matchday 10 data):

- Position
- Wins
- Draws
- Goals scored
- Goals conceded
- Squad market value (`qualidade_elenco`) — total squad value, normalized as each team's share of that season's total (team value ÷ sum of all 20 teams), to correct for market inflation across different years

**Target:** Final position (matchday 38)

Deliberately left out of X:

- **Wins_38, Draws_38, Losses_38, Points_38** — data leakage. These columns are derived directly from the very outcome the model should predict, so including them would hand the model the answer.
- **Points and Losses (matchday 10)** — tested and removed: exact mathematical redundancy (Points = 3×Wins + Draws, Losses = 10 − Wins − Draws). Removing both left MAE unchanged, confirming they added no independent signal.
- **Goal difference** — redundant information: GD = goals scored − goals conceded, already represented as separate features.
- **Club name** — used only as a key to join the matchday 10, matchday 38, and market value tables for the same year, never used as a model feature.

## Methodology

1. Merge matchday 10 with matchday 38 per year, using club name as the key, to avoid mixing different seasons.
2. Concatenate all seven seasons into a single dataset.
3. **Leave-one-year-out cross-validation** instead of a single fixed test set: each of the 7 seasons takes a turn as the test set while the model trains on the other 6, then the years rotate. This replaced an earlier single-split design (train on 2018–2024, test only on 2025), which turned out to measure only one season's luck rather than the model's general behavior.
4. Train with Scikit-Learn's `LinearRegression`.
5. Evaluate with **MAE** (Mean Absolute Error), chosen over MSE because it keeps the same unit as the problem (positions), allowing a direct reading of the error.
6. Compare against a naive baseline every year: predicting that a team finishes in the same position it held at matchday 10 ("nobody moves").

## Results

| Year | Model MAE | Baseline MAE |
|---|---|---|
| 2018 | 3.53 | 4.40 |
| 2020 | 3.27 | 2.70 |
| 2021 | 3.00 | 3.65 |
| 2022 | 2.75 | 3.60 |
| 2023 | 2.56 | 2.40 |
| 2024 | 3.48 | 4.70 |
| 2025 | 2.64 | 2.50 |
| **Average** | **3.04** | **3.42** |

The model beats the baseline on average (by ~0.4 positions) and wins outright in 4 of 7 seasons. The pattern by season matters more than the average: the model tends to win in seasons where the table shuffled a lot between matchday 10 and 38 (baseline MAE above ~3.5) and tends to lose in "quiet" seasons where matchday 10 already closely resembled the final table (baseline MAE below ~2.7: 2020, 2023, 2025).

Looking at individual teams in 2024 (a "shuffled" season): the model correctly pulled Athletico-PR down from 4th to a predicted ~7th (it actually finished 17th) and kept Flamengo near 1st (it finished 3rd). It also moved Corinthians up from 18th toward a predicted ~12th (it actually finished 7th) — the right direction, but not enough magnitude. This suggests the squad-value feature carries real signal, particularly for expensive squads underperforming, though it's a read from a handful of teams in one season, not a proven effect.

## Diagnosing the ~3-position error

After the results above, the question became: *why is the error stuck around 3?* Is it overfitting, underfitting, too little data, missing features, or is football simply unpredictable at matchday 10? Each hypothesis was tested separately.

### 1. Overfitting check

Test-year rows were added to the training set and the error re-measured. If the model were overfitting, the error on data it had already seen would have dropped far below the error on unseen data. It didn't: the error stayed the same (±0.1). Across all 7 folds, the gap between unseen-year error and training error ranged from −0.41 to +0.74 with mixed signs, which is noise, not a systematic overfit.

### 2. Learning curve (does more data help?)

Trained on 1 to 6 seasons, always testing on a season never seen, repeated for all 7 test years and averaged (each point = mean of 7 tests):

| Seasons trained | Train MAE | Test MAE | Baseline MAE |
|---|---|---|---|
| 1 | 2.59 | 4.10 | 3.42 |
| 2 | 3.14 | 3.24 | 3.42 |
| 3 | 3.01 | 3.10 | 3.42 |
| 4 | 2.86 | 3.00 | 3.42 |
| 5 | 2.92 | 3.06 | 3.42 |
| 6 | 2.90 | 3.04 | 3.42 |

- With 1 season (20 rows for 7 weights) the model is worse than the baseline: it fits that single season and fails on new ones.
- From 2 seasons on it beats the baseline, and from ~4 seasons on **both curves flatten and sit side by side** (~2.9 train / ~3.0 test).
- Training error doesn't go down either: the model sees the answers and still misses by ~3 positions. So more rows of the same kind will not help this model; the limit is in the information carried by the columns (or in randomness), not in the amount of data.

> Caveat: to build the curve, the model was allowed to train on future seasons to predict past ones (e.g. train on 2025, test on 2018). This is fine for diagnosis but is kinder to the model than real life. The "official" single-split number remains the 2025 test.

### 3. Is the linear model too simple? (decision tree test)

To test whether a more flexible model could extract patterns a straight-line formula can't, `DecisionTreeRegressor` was run with the same leave-one-year-out setup at different depths (`max_depth` limits how many chained yes/no questions the tree can ask):

| Model | Train MAE | Test MAE (avg of 7 years) | Train–test gap | Years beating the baseline |
|---|---|---|---|---|
| Linear regression | 2.90 | 3.04 | 0.1 | 4 of 7 |
| Tree, max_depth=3 | 2.66 | 3.93 | 1.27 | 2 of 7 |
| Tree, max_depth=4 | 2.23 | 3.85 | 1.61 | 3 of 7 |
| Tree, max_depth=5 | 1.65 | 3.82 | 2.17 | 2 of 7 |
| Tree, no limit | 0.00 | 4.06 | 4.06 | 2 of 7 |
| Baseline ("nobody moves") | — | 3.42 | — | — |

- The unrestricted tree memorizes the 120 training rows (train MAE = 0.00) and does *worse* than the naive baseline on unseen seasons: textbook overfitting.
- Restricting depth shrinks the gap but the test error never gets near the linear model's 3.04. The differences among depths (3.82–3.93) are within the noise of a 7-season evaluation.
- Pre-registered criterion: the tree would need to beat the linear model by well over 0.3 on average **and** win in at least 5 of 7 years to conclude the straight line was too simple. It lost in every configuration.

### Conclusion

| Hypothesis | Verdict |
|---|---|
| Overfitting (linear model) | Ruled out: train and test errors are almost equal |
| Not enough seasons | Unlikely: the learning curve is flat from ~4 seasons |
| Model too simple | Unlikely: a flexible model (tree) did not improve on it |
| Missing features | Open, but nothing else in scope is available for this project |
| Football is unpredictable from matchday-10 data | Strongest hypothesis |

With these inputs, a ~3-position average error looks close to the ceiling, and the ~0.4 improvement over "nobody moves" is probably most of what's available. Large single-team misses (e.g. Athletico-PR 4th → 17th in 2024, Fortaleza 11th → 4th) are not detectable from wins, goals and table position at matchday 10.

## Takeaways

- Before picking the tool, make sure the data behind the question actually exists — the original project idea died from lack of data, not from a modeling failure.
- Data leakage can be subtle: columns derived from the target itself (matchday-38 points, wins) looked like good features at first glance.
- Exact mathematical redundancy (Points from Wins+Draws) is easy to test directly by dropping the column and checking whether MAE moves — no guesswork needed.
- A single train/test split on one season measures that season as much as it measures the model. Leave-one-year-out cross-validation was necessary to tell "the model is good" apart from "2025 happened to be a good year for it."
- A number only means something next to a reference: the MAE of ~3 became interpretable only after comparing it with the "nobody moves" baseline.
- Reading a learning curve (train vs. test error as data grows) separates "needs more data" from "needs more information" — here it pointed to the latter.
- Giving the model more flexibility (decision tree) is a cheap way to ask whether the problem is the model or the data. If the flexible model only memorizes, the limit is likely in the data.
- Normalizing feature scale doesn't change predictions or MAE in plain `LinearRegression` — the model already compensates for scale through its coefficients. This only matters for algorithms like gradient descent or regularized regression.
- Adding a feature that "makes intuitive sense" (squad market value) doesn't guarantee a big MAE drop — it moved the average from 2.73 to 3.04 only in the sense of a more honest, cross-validated number, while genuinely helping in specific, identifiable cases (expensive squads collapsing).
- The rows-per-feature rule of thumb (10–15:1) helps size how much data is needed before going out to collect it.

## What I'd do differently starting over

- Set up cross-validation from day one, instead of trusting a single 20-row test season.
- Build the "nobody moves" baseline before the first model.
- Treat squad quality as a context feature conceptually separate from matchday-10 performance features from the start, rather than bolting it on later.
- Test the 2019 exclusion formally against the validation pipeline instead of leaving it as a one-off manual call.

## Known limitations

- 140 rows is a small dataset for a sport with this much year-to-year variability (baseline MAE alone ranges from 2.4 to 4.7 depending on the season). Differences of ~0.1–0.3 between models are within noise.
- Linear regression can't see mid-season signings, injuries, or coaching changes — all plausible drivers behind cases like Corinthians' 2024 turnaround. Those data are not available at the scope of this project.
- 2019 was excluded by a manual, un-validated decision.
- The squad-value timestamp is uncertain. A scraping-assistant check suggested Transfermarkt's "season N" squad page may show values from roughly January of N−1 (about a year before the season). This was inferred by comparing data, not from official documentation, and has not been verified. If true, the feature would not "see the future", but it would be a stale signal.
- The tree experiment used `max_depth` values chosen by looking at test error. Since no depth won, this doesn't change the conclusion, but it would be a bias in a case where one did.
- The learning-curve diagnosis trains on future seasons to predict past ones.
