# Results: Scenario B, accelerated tests → ageing at use conditions

**Question.** A new cell type is tested only at harsh conditions, and each test stops once the cell
reaches 90 % SOH (a few days to weeks). Can we forecast the *whole* ageing curve at a milder use
condition that was never tested?

**Fixed before the first run** (`knee_analogs/scenario_b.py`):
- **Split, by test conditions only** (never by how the cells actually aged; middle conditions left out):

  | type | accelerated (seen until 90 % SOH) | use (never seen; forecast) |
  |---|---|---|
  | LSD | charge >= 1.5 C or upper voltage 4.3 V (36 cells) | charge 0.5 C, 4.2 V (38) |
  | Mohtat (NMC) | -5 and 45 °C (14) | 25 °C (7) |
  | Tongji NCA | 35 and 45 °C (31) | 25 °C (35) |
  | Tongji NCM | 35 and 45 °C (32) | 25 °C (23) |
  | Stanford matrix (NCA) | charge >= 0.75 C (66) | charge <= 0.3 C (118) |

  Not used: Severson (every cell is fast-charged), Stanford dynamic and Tongji D3 (no stress axis).
- **Score:** absolute SOH error (percentage points) over each use cell's measured curve up to 3 × its
  true t90; mean per cell, per type, then over the 5 types. Bootstrap CIs resample cells within types.
  The main table uses the cells that reach 90 % (184 of 221), where every method has a forecast.
- **Methods:**
  - **accel_as_is:** no correction; the use cell ages like the accelerated cells.
  - **speed_regression:** log t90 against 1000/T, log C-rates, voltage window and depth of discharge,
    fitted on the accelerated cells (ridge).
  - Both use one **shared shape:** the accelerated cells' mean normalised curve up to 90 %, and beyond it
    their shape-only library forecasts from the *other* datasets.
  - **LLM** (`llm_trial/scenario_b_llm.py`; Opus 5.5 and Haiku 5.5, blinded, one prompt per type): the
    EFC at which SOH reaches 97.5, 95, ..., 70 % for each target condition.
    - **data:** chemistry, the accelerated tests (per condition, the EFC at 99, 98, ..., 90 % SOH) and
      a neutral mechanism note;
    - **know:** the same without any test data.
- **References (not zero-shot):**
  - **oracle_speed:** the shared shape with the cell's *true* t90;
  - **use_database:** the mean curve of the other use cells of the same type (leave-one-out).
- **Leakage test:** a use cell's forecast does not change when its own data, or the accelerated cells'
  data after 90 %, change (`tests/test_knee_analogs.py`).

**Cost:** 32 calls, $1.28 (Opus $1.23, Haiku $0.05); 3 repeats: 98 calls, $3.80.

## Results

Error (SOH points) on the common points:

| method | LSD | Mohtat | Tongji NCA | Tongji NCM | Stanford matrix | **mean** |
|---|---|---|---|---|---|---|
| oracle_speed (true t90; reference) | 1.44 | 1.60 | 1.75 | 1.02 | 2.30 | **1.62** |
| use_database (reference) | 1.82 | 3.43 | 2.62 | 0.98 | 2.90 | **2.35** |
| Opus, data: LLM speed + shared shape (post hoc) | 2.43 | 5.47 | 4.68 | 2.60 | 4.17 | **3.87** |
| Haiku, data | 2.30 | 5.36 | 4.75 | 2.28 | 10.60 | **5.06** |
| Haiku, know | 6.55 | 10.34 | 3.93 | 3.58 | 2.42 | 5.36 |
| Opus, data | 7.58 | 5.46 | 4.21 | 3.18 | 7.63 | 5.61 |
| Opus, know | 13.66 | 10.52 | 3.48 | 1.30 | 3.14 | 6.42 |
| speed_regression | 2.95 | 7.05 | 4.01 | 2.69 | 36.29 | 10.60 |
| accel_as_is | 30.18 | 6.13 | 2.88 | 1.28 | 16.67 | 11.43 |

Predicted vs true t90 (median factor, 1 = exact):

| method | LSD | Mohtat | Tongji NCA | Tongji NCM | Stanford matrix |
|---|---|---|---|---|---|
| accel_as_is | 7.31 | 1.74 | 1.37 | 1.15 | 3.36 |
| speed_regression | 1.44 | 2.00 | 2.27 | 2.26 | 5.26 |
| Opus, data | 1.32 | 1.60 | 2.82 | 2.21 | 1.33 |
| Opus, know | 1.72 | 6.59 | 1.76 | 1.29 | 1.43 |
| Haiku, data | 1.43 | 1.55 | 2.96 | 2.13 | 2.36 |
| Haiku, know | 2.00 | 5.45 | 2.38 | 2.85 | 1.27 |

True use / accelerated t90 (median): LSD 8.2, Stanford matrix 2.9, Mohtat 1.6, Tongji NCA 0.85,
Tongji NCM 1.07.

Comparisons (difference in error, 95 % CI):

| comparison | difference | 95 % CI |
|---|---|---|
| speed_regression − accel_as_is | −1.59 | [−2.54, −0.61] |
| Opus data − speed_regression | −4.49 | [−5.56, −3.36] |
| Haiku data − speed_regression | −5.05 | [−5.90, −4.20] |
| Opus data − Opus know | −1.15 | [−1.96, −0.38] |
| Haiku data − Haiku know | −0.49 | [−0.89, −0.09] |
| Haiku data − Opus data | −0.55 | [−1.07, −0.05] |
| Opus data − use_database | +3.03 | [+2.50, +3.55] |
| *post hoc:* Opus data speed + shared shape − Opus data | −1.62 | [−1.94, −1.31] |
| *post hoc:* Opus data speed + shared shape − use_database | +1.42 | [+1.03, +1.80] |

## What this says

1. **The shape transfers; the speed does not.** With the true t90, the accelerated cells' shape plus
   the library forecasts the use curve to 1.62 points, better than a database of use-condition cells
   (2.35). Everything hinges on how much slower the cell ages at use conditions.
2. **Fitting speed against the test conditions fails outside the tested range.** It works on LSD,
   whose accelerated conditions bracket the use condition (t90 within a factor of 1.4). On the
   Stanford matrix, the use charge rate (0.2-0.3 C) lies far below the accelerated ones (0.75-1 C);
   the fit is off by a factor of 5, and its curve even runs below 0 % SOH.
3. **"Accelerated" was not always faster.** The Tongji cells aged about as fast at 35-45 °C as at
   25 °C. Mohtat ages faster at both -5 and 45 °C than at 25 °C, so a straight-line fit on temperature
   cannot work there. A rule like "hotter means faster" would mislead.
4. **The LLM halves the error of the data-only methods** (about 5 vs 10.6 points). It estimates the
   speed-up from chemistry and mechanisms. With the test data it gets the speed within a factor of
   1.3-1.6 on LSD, Mohtat and (Opus) the Stanford matrix.
5. **The test data help on single runs, but not reliably (see Repeats).** On the first run, data
   improved Opus by 1.15 points and Haiku by 0.49. Over 4 runs the gain shrinks to 0.53 and 0.12 and
   its CI includes 0. By type: on Mohtat, where cold and heat both speed up ageing, the data matter
   (5.4 vs 10.4 on the first run). On the Stanford matrix, where the use condition lies far outside
   the tested range, knowledge alone did better: the models anchor on the fast tests.
6. **The LLMs invent knees.** Opus gets LSD's speed almost right, but forecasts an early, steep knee
   (SOH 70 % by 1.4 × t90) that the cells do not show. Both models do the same on the Stanford matrix
   after 1.5 × t90. Taking only Opus's speed with the shared shape (post hoc) gives 4.35 over 4 runs,
   against 6.40 for its own curves. The best split of labour: the LLM judges the speed, the data give
   the shape.
7. **Haiku's own curves beat Opus's** (4.67 vs 6.40 over 4 runs), because it forecasts milder knees.
   Opus judges the speed more consistently (see Repeats), so with the shared shape Opus is better
   (4.35 vs 5.18).
8. **Still far from a use-condition database** (+1.9 points for the best variant). The method is a
   first estimate before long tests, not a replacement for them.

## Repeats: how much does a single run vary?

Each model and condition was asked 3 more times with identical prompts (96 calls, $3.80; two
incomplete Haiku answers were re-asked). `scripts/scenario_b_repeats.py`. Fixed before the repeats
were seen: score every run the same way, and combine runs by the median over the 4 runs of each
forecast EFC.

Error (SOH points, mean over types):

| | single runs: mean ± sd | range | median of 4 runs |
|---|---|---|---|
| Opus, data | 6.80 ± 1.06 | 5.61-8.17 | 6.40 |
| Opus, know | 6.47 ± 0.56 | 6.03-7.26 | 6.52 |
| Haiku, data | 5.12 ± 0.59 | 4.71-5.97 | 4.67 |
| Haiku, know | 4.94 ± 0.47 | 4.29-5.36 | 4.57 |
| *post hoc:* Opus data speed + shared shape | 4.41 ± 0.38 | 3.87-4.72 | **4.35** |
| *post hoc:* Haiku data speed + shared shape | 5.64 ± 0.52 | 4.95-6.11 | 5.18 |

Spread of the predicted t90 across the 4 runs (max / min, median over target conditions):

| | LSD | Mohtat | Tongji NCA | Tongji NCM | Stanford matrix |
|---|---|---|---|---|---|
| Opus, data | 1.09 | 1.25 | 1.24 | 1.07 | 1.58 |
| Opus, know | 1.18 | 1.40 | 1.25 | 1.21 | 1.48 |
| Haiku, data | 1.59 | 2.58 | 1.40 | 1.34 | 2.04 |
| Haiku, know | 2.23 | 2.18 | 2.34 | 1.55 | 2.00 |

Median-of-runs comparisons (95 % CI):
- Opus data − Opus know: −0.53 [−1.27, +0.21];
- Haiku data − Haiku know: −0.12 [−0.38, +0.17];
- Haiku data − Opus data: −1.65 [−2.08, −1.21];
- Opus data − speed_regression: −3.80 [−4.84, −2.69];
- *post hoc:* Opus data speed + shared shape − Opus data: −1.87 [−2.25, −1.50];
- *post hoc:* the same − use_database: +1.86 [+1.48, +2.26].

**What the repeats change**
- **A single run is not enough.** Opus's first run was its best of four (5.61; the others 6.7-8.2).
  On the Stanford matrix its error varies by 5 points between runs. With one prompt per type, one
  answer sets a whole type's forecast.
- **The gain from the test data is not robust:** with runs combined, it is within the noise for both
  models.
- **What holds up:**
  - the LLM beats fitting speed against conditions (−3.8 points);
  - Opus's speed is stable across runs (within a factor of 1.1-1.6), while Haiku's varies by up to
    2.6×;
  - Opus's speed with the shared shape is the best and most stable variant (4.35, sd 0.38). That
    combination is still post hoc.

## Scenario B+: adding a short test at the use condition (`scripts/scenario_b_early.py`)

**Idea.** Speed was the missing piece. So also test the use condition itself, briefly: until SOH
first reaches 98 % (primary) or 99 %. That takes a median 11-23 % of the time to 90 % at 98 %, and
5-15 % at 99 %. Mohtat is the exception: its first check-up already comes at 31 % of t90, so it has no
real short test.

**Fixed before the first run** (`knee_analogs/scenario_b.py`, `forecast_early`). The use cell is seen
up to its first point at or below the stopping level, and only its later points are scored, up to
3 × its true t90.
- **early_accel:** t90 = the use cell's time to 98 % × the accelerated cells' median ratio t90 / t98.
  This assumes the early fade is a scaled copy of the whole curve.
- **early_library:** the same ratio, taken from every library cell of the other datasets.
- **early_accel+llm:** the geometric mean of the early_accel t90 and Opus's t90 (data, median of 4 runs).
- **early_linear:** the use cell's own straight-line trend.

The curves use the shared shape, as before.

Error (SOH points; mean over types, common points):

| method | stop at 98 % | stop at 99 % |
|---|---|---|
| oracle_speed (true t90; reference) | 1.78 | 1.76 |
| use_database (reference) | 2.77 | 2.68 |
| early_library | **4.62** | 7.18 |
| early_accel+llm | 5.00 | **5.49** |
| early_linear | 5.01 | 6.60 |
| early_accel | 5.84 | 8.13 |
| Opus, data, median of runs (no short test) | 7.56 | 7.23 |
| speed_regression (no short test) | 12.85 | 12.33 |

The no-short-test rows differ from the main table because only later points are scored here.

At 98 % (95 % CI):
- early_accel − speed_regression: −7.01 [−8.40, −5.64];
- early_accel − early_library: +1.23 [+0.78, +1.64];
- early_accel − early_linear: +0.72 [−0.33, +1.79];
- early_accel+llm − early_accel: −0.85 [−1.58, −0.22];
- early_accel − use_database: +3.07 [+2.54, +3.63].

**Why the short test helps less than expected.** The first 2 % of fade is not a scaled copy of the
rest of the curve, and that differs between accelerated and use conditions. At use conditions the cells
lose their first 2 % relatively quickly (break-in, early SEI growth) and then fade slowly. At
accelerated conditions the main stress dominates from the start.

Median t90 / t98 (and t90 / t95), true values:

| type | accelerated | use |
|---|---|---|
| LSD | 4.8 (1.9) | 9.3 (2.3) |
| Stanford matrix | 3.6 (1.7) | 8.0 (2.5) |
| Mohtat | 6.9 (2.5) | 6.3 (2.4) |
| Tongji NCA | 6.4 (2.8) | 4.4 (1.8) |
| Tongji NCM | 9.4 (3.1) | 7.2 (2.4) |

So scaling the early test by the accelerated cells' ratio is off by up to a factor of 2 (t90 factor 1.3-2.2).

**What this says**
- **A short use test cuts the error of the condition-only methods by more than half** (5-6 vs 12.9). It
  also beats the LLM alone (early_accel − Opus: −1.76 [−2.67, −0.82]). But it does not reach the
  database (+3 points).
- **The accelerated tests do not tell how to scale the early fade.** The library's ratio
  (early_library) does better than the type's own accelerated cells. A straight line through the
  short test does as well. "Same curve, different speed" holds for the shape after 90 %, not for the
  start of life.
- **The LLM and the short test err differently.** Averaging their t90s helps (−0.85 at 98 %, −2.94 at
  99 %), and the average is the most robust to a very short test.
- **Ideas for next time, not tested here:**
  - stop the short test later (t90 / t95 is closer between accelerated and use than t90 / t98);
  - scale by the late slope of the short test instead of the time to 98 %;
  - use voltage-based degradation modes to separate break-in from steady fade.

### Second round: a later stop and slope scaling (`scripts/scenario_b_early2.py`)

Both were fixed before this run. The ideas came from the first round.
- **Slope scaling** (slope_accel, slope_library): t90 = the use cell's time to 98 % + the time from 98 %
  to 90 %. That second part is estimated from the fade rate between 99 and 98 % (the end of the short
  test, after the break-in), times the reference cells' median ratio of the two.
- **A later stop at 95 %** with every method, the slope window then being 96 → 95 %.

Error (SOH points; mean over types, common points):

| method | stop at 98 % (median 19 % of t90) | stop at 95 % (median 48 % of t90) |
|---|---|---|
| oracle_speed (true t90; reference) | 1.78 | 1.83 |
| use_database (reference) | 2.77 | 3.18 |
| slope_library | **3.89** | 2.78 |
| slope_accel | 4.53 | **2.67** |
| slope_accel+llm | 4.51 | 3.94 |
| early_library | 4.62 | 3.20 |
| early_linear (own straight line) | 5.01 | **2.64** |
| early_accel | 5.84 | 3.75 |

Comparisons (95 % CI):

| comparison | stop at 98 % | stop at 95 % |
|---|---|---|
| slope_accel − early_accel | −1.33 [−1.78, −0.95] | −1.10 [−1.30, −0.92] |
| slope_library − early_library | −0.82 [−1.46, −0.29] | −0.42 [−0.66, −0.23] |
| slope_accel − slope_library | +0.71 [+0.30, +1.12] | −0.07 [−0.26, +0.11] |
| slope_accel − early_linear | −0.61 [−1.85, +0.60] | −0.00 [−0.74, +0.63] |
| slope_accel+llm − slope_accel | +0.00 [−0.41, +0.38] | +1.32 [+0.90, +1.72] |
| slope_accel − use_database | +1.76 [+1.30, +2.20] | −0.51 [−0.87, −0.17] |

Predicted t90 vs true (median factor) at 98 %: slope_library 1.2-1.5, slope_accel 1.2-1.9; at 95 %:
both 1.1-1.3.

**What this says**
- **Scaling by the late slope works better than scaling by time,** at both stops (−1.3 and −1.1). It
  skips the break-in, which is what broke the first rule. This supports the hypothesis in a
  narrower form: after the break-in, the fade does scale between conditions.
- **With a test to 98 % (about a fifth of the time to 90 %), the best is 3.9 SOH points** (slope scaled
  by the library). The type's own accelerated cells do slightly worse than the library here (+0.71).
  That is still 1.1 points short of a use-condition database.
- **With a test to 95 %, the cell's own data dominate.** At about half the time to 90 %, slope scaling
  (2.67) and the plain straight line (2.64) both beat a use-condition database (3.18 on these points,
  which are later in life). The accelerated tests and the library add nothing measurable over the
  straight line.
- **The LLM no longer helps once the cell's own slope is known.** It is neutral at 98 % and worse at 95 %.
- **The practical trade-off is error vs test time:**

  | information | test time, as a share of t90 at use | error |
  |---|---|---|
  | accelerated tests only, LLM speed + data shape | ~0 | ~4.4 |
  | + short use test to 98 % (slope scaling) | ~0.2 | ~3.9 |
  | + use test to 95 % | ~0.5 | ~2.7 |
  | full use-condition database (other cells) | long | 2.8-3.2 |

  The no-test row is measured on all points from the start of life, the others only after their stop,
  so the rows are not strictly comparable.

**Caveats:** cells resampled within only 5 types; Mohtat has no real short test; the first two methods
cannot be told apart statistically from the straight line.

**Caveats**
- **Four runs per model and condition** measure the spread, but not every source of variation: the
  prompt wording and the choice of a single prompt per type are fixed.
- **Five types only;** the per-type CIs say nothing about how a new type would fare.
- **The public datasets may be known to the models,** especially the Stanford and Tongji papers.
  "Knowledge only" doing well on the Stanford matrix could be recall.
- **The post-hoc combination was chosen after seeing the LLM curves.** It needs confirmation on new data.

Files:
- `reports/library/scenario_b_by_type.csv`, `scenario_b_overall.csv`, `scenario_b_t90.csv`,
  `scenario_b_ci.csv`, `scenario_b_repeats.csv`, `scenario_b_repeats_t90.csv`, `scenario_b_repeats_ci.csv`;
- `reports/library/scenario_b_early.csv`, `scenario_b_early_ci.csv`, `scenario_b_early_t90.csv`,
  `scenario_b_early2.csv`, `scenario_b_early2_ci.csv`, `scenario_b_early2_t90.csv`;
- `reports/library/figs/scenario_b_examples.png`, `scenario_b_early_examples.png`.

Prompts and answers stay in `data/library/llm/scenario_b/` (not committed).
