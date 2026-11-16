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

## Not yet done

- **Scenario B, accelerated → use conditions.** Only accelerated cells of a type observed to 90 %;
  forecast cells of the same type at milder conditions, without observing them.
- **LLM conditions** (numbers + analogs; + chemistry and mechanism knowledge; + operating history),
  to be costed before any API call.
- **A clean test on unpublished data:** Chronos and LLMs may have seen public datasets.

Files:
- `reports/library/universality_*.csv`, `zero_shot_overall.csv`, `zero_shot_by_dataset.csv`,
  `zero_shot_ci.csv`;
- `reports/library/figs/universality.png`, `zero_shot_examples.png`.

The per-point forecasts and Chronos outputs stay in `data/library/` (not committed).
