# Results: forecasting ageing without an ageing database of the cell type

**Idea being tested.** Building a long-term ageing database for every new cell type takes months. If
lithium-ion cells all age through the same mechanisms (SEI growth, lithium plating, loss of active
material), and mostly differ in *how fast*, then a short test down to 90 % state of health (SOH) plus
knowledge from *other* cell types might be enough to forecast the rest of the curve.

**Library** (`scripts/build_library.py` → `data/library/`, not committed): 766 cells from 6 datasets as
SOH (capacity / initial) against equivalent full cycles (EFC), with chemistry and conditions.

| dataset | chemistry | cells | capacity series |
|---|---|---|---|
| LSD first life | Ni-based / graphite (not stated), 2.4 Ah | 86 | C/5 check-ups |
| Mohtat | NMC / graphite pouch, 5 Ah | 21 | C/20 check-ups |
| Severson 2019 (3 batches) | LFP / graphite, 1.1 Ah | 135 | every cycle |
| Stanford aging matrix | NCA / graphite+SiOx, 4.84 Ah | 302 | 0.2C RPT every 100-200 cycles |
| Stanford dynamic cycling | NCA / SiOx-graphite (EV) | 92 | every cycle |
| Tongji (Zhu 2022) | NCA, NCM, NCM+NCA 18650 | 130 | every cycle |

## 1. Do cells age along one curve at different speeds? (`scripts/universality.py`)

For every cell: t_x = EFC at which SOH first reaches x %. If "same mechanisms, different speed" held
exactly, every curve against t / t90 would be the same and t80 / t90 would be one number.

| dataset | t80 / t90, median [IQR] | t70 / t90 |
|---|---|---|
| Severson (LFP) | 1.15 [1.13-1.18] | - |
| LSD | 1.54 [1.37-1.75] | - |
| Stanford matrix (NCA) | 2.01 [1.69-2.43] | 2.72 [2.06-3.78] |
| Mohtat (NMC) | 2.43 [2.07-2.73] | 3.33 |
| Tongji | 3.07 [1.64-3.32] | 3.86 |
| Stanford dynamic (NCA, EV) | 3.60 [3.20-3.99] | 7.55 |

- **Speed carries most of the information.**
  - t90 alone spans a factor of 7 across cells (10th-90th percentile); t80 / t90 spans 2.8.
  - On a log scale, t90 explains 76 % of the variance of t80 over all 491 cells that reached 80 %.
  - Within a dataset it explains 81-98 % (Tongji 40 %, Stanford dynamic 71 %).
- **The shape after 90 % is not universal.** It depends on chemistry and conditions
  (`reports/library/figs/universality.png`):
  - LFP falls off a cliff right after 90 %.
  - The NCA EV cells keep decelerating.
  - The Stanford matrix and Tongji contain both a slow straight path and a "sudden death" path.
  - So the hypothesis holds for the time scale, not for the shape: there are a few shape families.
- Median curves beyond ~1.5 t90 are biased toward long-lived cells (only they are still measured).

## 2. Scenario A: forecast from the cell's own curve down to 90 % (`scripts/zero_shot_study.py`)

**Fixed before the first run** (`knee_analogs/zero_shot.py`):
- **Query:** each cell is seen only until its first measurement at or below 90 % SOH, where a test would
  stop. Its future after that point, up to 3 × t90, is scored. 675 cells.
- **Memory:** cells of the *other five datasets* only, so no ageing data of the query's cell type.
- **Methods:**
  - **library analogs:** the 10 most similar pre-90 % shapes (SOH at 10 %, 20 %, ..., 90 % of t90),
    optionally plus temperature and C-rates. Their futures are followed in time normalised by t90, and
    the band is their 10-90 %.
  - **the library's median curve:** one universal curve;
  - **the cell's own linear and power-law trends;**
  - **Chronos-2 and Chronos-Bolt-base:** time-series foundation models, zero-shot on the cell's own curve
    (64 points), 10-90 % quantile band. Run locally from cached weights, no download.
- **Upper bound (not zero-shot):** analogs drawn from the query's own dataset.
- **Score:** absolute SOH error in percentage points, mean per cell, then per dataset, then the mean over
  the 6 datasets (each dataset counts once), on the points all methods forecast.
- **Correction during this step:** a leakage test found that the first point below 90 % was used to place
  t90 but also scored as "future". It now counts as seen, and only later points are scored. The numbers
  below are after the fix; before it, the analogs and Chronos-2 looked ~0.15 points better and the linear
  trend slightly worse.

| method | ageing data of the same type? | error (SOH points) | 1-1.5 t90 | 1.5-2 t90 | 2-3 t90 | band covers actual |
|---|---|---|---|---|---|---|
| **analogs from the same dataset (upper bound)** | yes | **0.97** | 0.61 | 1.05 | 1.63 | 64 % |
| library analogs, shape only | no | **1.94** | 1.35 | 1.96 | 2.96 | 45 % |
| Chronos-2 | no | **1.95** | 1.16 | 1.80 | 3.20 | **79 %** |
| own linear trend | no | **1.99** | 1.40 | 1.57 | 2.32 | - |
| library analogs, shape + conditions (pre-registered primary) | no | 2.41 | 1.48 | 2.29 | 3.81 | 39 % |
| library median curve | no | 2.59 | 1.85 | 2.74 | 3.04 | - |
| Chronos-Bolt-base | no | 2.63 | 1.39 | 2.22 | 4.81 | 72 % |
| own power law | no | 2.96 | 1.83 | 3.04 | 4.52 | - |

Per dataset (error, SOH points):

| dataset | same-dataset analogs | library, shape | Chronos-2 | own linear | own power law |
|---|---|---|---|---|---|
| LSD | 1.12 | **1.59** | 1.79 | 2.09 | 2.68 |
| Mohtat | 1.52 | 1.75 | 2.25 | 1.58 | **0.92** |
| Severson (LFP) | 0.61 | **2.08** | 2.46 | 3.48 | 3.79 |
| Stanford dynamic | 0.92 | 3.32 | 2.05 | **1.99** | 3.79 |
| Stanford matrix | 1.09 | 1.49 | 1.44 | **1.17** | 1.56 |
| Tongji | 0.57 | **1.41** | 1.74 | 1.64 | 5.00 |

Bootstrap 95 % CIs (cells resampled within datasets, mean over datasets):
- **Pre-registered primary (shape + conditions):**
  - worse than shape only: +0.55 [+0.39, +0.71];
  - worse than Chronos-2: +0.45 [+0.18, +0.71];
  - worse than the own linear trend: +0.59 [+0.36, +0.82];
  - better than the power law (−0.51 [−0.79, −0.24]) and Chronos-Bolt (−0.34 [−0.60, −0.08]).
- **Comparisons chosen after seeing the table:**
  - shape-only analogs vs Chronos-2: −0.04 [−0.26, +0.19];
  - shape-only analogs vs own linear trend: +0.00 [−0.14, +0.16];
  - Chronos-2 vs own linear trend: +0.09 [−0.15, +0.32];
  - shape-only analogs vs same-dataset analogs: +0.97 [+0.82, +1.12].

**What this says**

1. **Without any ageing data of the cell type, the error doubles:** ~1.95 vs ~0.97 SOH points over the
   next 2 × t90. That is the price of skipping the database.
2. **No zero-shot method beats the cell's own straight-line trend on average.** Library analogs,
   Chronos-2 and the linear trend all sit at ~1.95 (CIs include 0). They win in different places:
   - **library analogs** where another chemistry has the right shape: LFP's cliff after 90 %
     (2.1 vs 3.5 for the trend), LSD, Tongji;
   - **the trend** where fade stays straight: the Stanford matrix and the Stanford EV cells, whose
     decelerating shape the library lacks (analogs 3.3);
   - **a power law** on Mohtat (0.92).

   A rule that knew *which* method to trust per cell would beat all of them. That choice needs
   understanding of the mechanism, which is where an LLM could add something.
3. **Chronos-2's band is well calibrated** (79 % inside its 10-90 % band). The analog bands are too
   narrow (39-45 %).
4. **Raw conditions do not transfer across chemistries.** Adding temperature and C-rate to the retrieval
   makes it worse: 2 C does not mean the same for LFP and NCA. Conditions need a physical translation
   (an Arrhenius factor, a plating-risk index), not raw numbers.

## 3. Scenario A with an LLM (`llm_trial/zero_shot_llm.py`)

**Fixed before any API call:**
- **Model:** `claude-opus-5-5`, adaptive thinking, effort medium, structured JSON output.
- **Prompts:** 10 query cells of one dataset per prompt, labelled Q1..Q10, with no dataset or cell names.
  The analogs come only from other datasets, so no query's future appears in any prompt (checked by an
  assert). Each query's curve is given as SOH at 0, 10, ..., 100 % of t90 plus the stopping point.
- **Answer:** SOH at u = t / t90 = 1.25, 1.5, 1.75, 2, 2.5 and 3, as 10 / 50 / 90 % quantiles, plus one
  sentence of reasoning. It is interpolated to the measured points and scored like every other method.
- **Conditions:**
  - **A:** the numbers plus the 10 shape-only library analogs (their pre-90 % shape and their future);
  - **B:** A plus the chemistry and test conditions of the query and of each analog, and a short
    neutral note on degradation mechanisms;
  - **C:** B without the analogs: the curve, chemistry, conditions and mechanism knowledge only.
- **Fixed after the run, before scoring** (harness reliability only):
  - In 15 of 219 prompts the model answered for Q1 only, and one answer hit the token limit. Answers
    that do not cover every cell are kept in `incomplete/` and the prompt is asked again, unchanged.
  - 3 condition-A prompts were refused by the API's safety classifier (category "reasoning_extraction")
    and went through on a plain retry, without rewording.
  - All 707 cells now have a forecast in each condition.
- **Cost:** $20.2 in total (2.41 M input and 0.53 M output tokens, including retries).
- **Comparisons fixed before the LLM results were seen** (`scripts/zero_shot_study.py`):
  - A vs the shape-only analogs, B vs A, B vs C, B and C vs the linear trend and Chronos-2;
  - a combination rule: the equal-weight mean of the three tied zero-shot methods (shape-only analogs,
    Chronos-2, linear trend), with and without LLM B. The three were picked from the table in section 2,
    i.e. after seeing the non-LLM results.

**Results** (same points as section 2; the non-LLM numbers are unchanged):

| method | error (SOH points) | 1-1.5 t90 | 1.5-2 t90 | 2-3 t90 | band covers actual |
|---|---|---|---|---|---|
| analogs from the same dataset (upper bound) | 0.97 | 0.61 | 1.05 | 1.63 | 64 % |
| **mean of analogs, Chronos-2, linear trend and LLM B** | **1.39** | 0.94 | 1.39 | 2.01 | - |
| mean of analogs, Chronos-2 and linear trend | 1.46 | 1.02 | 1.48 | 2.05 | - |
| **LLM B: curve + analogs + chemistry, conditions, mechanisms** | **1.51** | 0.83 | 1.50 | 2.59 | 75 % |
| LLM C: curve + chemistry, conditions, mechanisms (no analogs) | 1.62 | 0.85 | 1.67 | 3.12 | 78 % |
| LLM A: curve + analogs | 1.75 | 1.06 | 1.85 | 2.88 | 66 % |
| library analogs, shape only | 1.94 | 1.35 | 1.96 | 2.96 | 45 % |
| Chronos-2 | 1.95 | 1.16 | 1.80 | 3.20 | 79 % |
| own linear trend | 1.99 | 1.40 | 1.57 | 2.32 | - |

Per dataset (error, SOH points):

| dataset | same-dataset analogs | library, shape | linear trend | LLM A | LLM B | LLM C | mean of 3 | mean of 3 + LLM B |
|---|---|---|---|---|---|---|---|---|
| LSD | 1.12 | 1.59 | 2.09 | 1.54 | **1.13** | 1.32 | 1.70 | 1.46 |
| Mohtat | 1.52 | 1.75 | 1.58 | 1.67 | 1.45 | 1.46 | 1.49 | **1.33** |
| Severson (LFP) | 0.61 | 2.08 | 3.48 | 1.46 | 1.14 | **1.02** | 1.47 | 1.32 |
| Stanford dynamic | 0.92 | 3.32 | **1.99** | 3.36 | 3.21 | 3.60 | 2.03 | 2.21 |
| Stanford matrix | 1.09 | 1.49 | 1.17 | 1.30 | 1.14 | 1.26 | 1.12 | **1.10** |
| Tongji | 0.57 | 1.41 | 1.64 | 1.18 | 0.97 | 1.05 | 0.95 | **0.92** |

Pre-registered comparisons (difference in error, bootstrap 95 % CI, cells resampled within datasets):

| comparison | difference | 95 % CI | reading |
|---|---|---|---|
| LLM A − library analogs | −0.19 | [−0.31, −0.08] | the LLM reads the same analogs better than their weighted mean |
| LLM B − LLM A | −0.38 | [−0.54, −0.24] | chemistry, conditions and mechanism knowledge help |
| LLM B − LLM C | −0.07 | [−0.16, +0.02] | the analogs add little once that knowledge is given |
| LLM B − own linear trend | −0.32 | [−0.46, −0.19] | |
| LLM B − Chronos-2 | −0.47 | [−0.68, −0.24] | |
| LLM C − own linear trend | −0.25 | [−0.39, −0.11] | knowledge alone, no ageing data at all |
| mean of 3 + LLM B − mean of 3 | −0.07 | [−0.11, −0.05] | |
| mean of 3 − own linear trend | −0.55 | [−0.67, −0.43] | |

**What this says**

1. **The LLM is the first zero-shot method that clearly beats the cell's own straight line.** With the
   chemistry, conditions and mechanism note it reaches 1.51 SOH points, against ~1.95 for everything else.
   That closes about 45 % of the gap to a same-type database (0.97). The gain is largest in the first
   half of the horizon (0.83 vs 1.16-1.40 up to 1.5 t90).
2. **What it needs is the physical context, not the analogs.** Removing the analogs costs only 0.07
   (CI includes 0). Removing the chemistry and conditions costs 0.38. This fits the hypothesis "same
   mechanisms, different speed": the model maps "LFP, fast charge" or "NCA, 45 °C" to a shape family,
   which raw-number retrieval could not do (section 2, point 4).
3. **Its bands are reasonably calibrated** (66-78 % inside a nominal 80 % band), close to Chronos-2 and
   much better than the analogs.
4. **It fails where the curve keeps decelerating.** On the Stanford EV cells, every LLM condition expects
   too much fade (mean bias −4 to −5 points). It allows for a silicon-driven knee that does not come.
   The linear trend is better there (1.99 vs 3.2).
5. **A simple average is a strong competitor.** Averaging the analogs, Chronos-2 and the linear trend gives
   1.46, about as good as the LLM, at no cost. Adding the LLM to the average gives the best result, 1.39.
   These three methods err in different directions (e.g. on Severson, Chronos-2 forecasts too low and the
   linear trend too high).

### 3b. The same prompts on a small, cheap model (Claude Haiku 5.5)

`python llm_trial/zero_shot_llm.py --model haiku`: identical prompts, settings and scoring. One condition-C
answer covered only Q1 and was re-asked. **Cost: $0.78**, against $20.2 for Opus 5.5 (about 0.1 cent per
cell). Comparisons fixed before the Haiku results were seen.

| | Opus 5.5 | Haiku 5.5 | best non-LLM (linear trend) |
|---|---|---|---|
| A: curve + analogs | 1.75 | 2.00 | 1.99 |
| B: + chemistry, conditions, mechanisms | **1.51** | **1.71** | |
| C: B without analogs | 1.62 | 1.81 | |
| B, first 1-1.5 t90 | 0.83 | 1.00 | 1.40 |
| B band covers actual | 75 % | 66 % | - |

Per dataset, condition B (SOH points): LSD 1.47 (Opus 1.13), Mohtat 1.18 (1.45), Severson 1.51 (1.14),
Stanford EV 3.61 (3.21), Stanford matrix 1.31 (1.14), Tongji 1.20 (0.97).

| comparison | difference | 95 % CI |
|---|---|---|
| Haiku B − Opus B | +0.22 | [+0.08, +0.35] |
| Haiku C − Opus C | +0.26 | [+0.12, +0.41] |
| Haiku A − library analogs | +0.06 | [−0.03, +0.14] |
| Haiku B − Haiku A | −0.26 | [−0.41, −0.11] |
| Haiku B − Haiku C | −0.12 | [−0.29, +0.05] |
| Haiku B − own linear trend | −0.11 | [−0.31, +0.10] |
| Haiku C − own linear trend | −0.00 | [−0.21, +0.20] |

- **The same pattern, about half the gain.** Chemistry and mechanism context helps Haiku as it helps Opus
  (−0.26). The analogs again add little. Haiku B gets 0.28 of the 0.48 points Opus B gains over the
  linear trend.
- **Overall, Haiku is not clearly better than the linear trend** (CI includes 0). It is clearly better
  in the near term (1.00 vs 1.40 up to 1.5 t90), and on LSD (1.47 vs 2.09) and Mohtat.
- **Given only numbers and analogs (A), Haiku adds nothing** over the analogs' weighted mean. Opus did
  (−0.19).
- **The bigger model's edge is largest on Severson (LFP) and LSD.** That fits "more knowledge"
  but also "more memorised literature": a smaller model is less likely to have memorised a specific
  paper. Haiku keeping half the gain suggests that at least part of it is general chemistry knowledge,
  not recall.

**Caveats**
- **Memorisation cannot be ruled out.** Severson, the Stanford sets, Tongji and Mohtat are public and
  widely analysed. The prompts never name a dataset, but chemistry plus conditions (e.g. LFP 1.1 Ah with
  fast charging) can identify one. Condition C doing well on Severson (1.02) could be knowledge of LFP
  in general or recall of that paper. LSD is the cleanest case, if it is unpublished: LLM B 1.13 vs 1.59
  for the analogs and 2.09 for the trend.
- **One run per condition, no repeats:** the run-to-run spread of the LLM is not measured.
- **The prompts and raw answers stay in `data/library/llm/`** (not committed): they contain the per-cell
  curves.

## Not yet done

- **Scenario B** (accelerated tests → use conditions) is now in
  [results_scenario_b.md](results_scenario_b.md).
- **A clean test on unpublished data** (Chronos and LLMs may have seen the public datasets), and repeats
  of the LLM run to measure its spread.

Files:
- `reports/library/universality_*.csv`, `zero_shot_overall.csv`, `zero_shot_by_dataset.csv`,
  `zero_shot_ci.csv`;
- `reports/library/figs/universality.png`, `zero_shot_examples.png`;
- `reports/library/figs/zero_shot_llm_examples.png` (one typical cell per dataset) and
  `zero_shot_llm_errors.png` (error by method and horizon), from `scripts/fig_zero_shot_llm.py`.

The per-point forecasts and Chronos outputs stay in `data/library/` (not committed).
