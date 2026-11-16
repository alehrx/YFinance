# Results: forecasting the whole future ageing trajectory

**Question.** Instead of one number (a knee cycle, or the cycle where capacity reaches 80%), can the
analogs forecast a cell's **entire future capacity curve** from its first N cycles, with an honest
band, and do better than simple alternatives?

**What is forecast.** The state of health (SOH, capacity relative to the reference) at **every later
check-up** of the cell, scored against everything measured after N, at all horizons the data allow.
- **LSD:** C/5 check-ups, every 5th cycle, up to 1,180 cycles ahead.
- **Mohtat:** C/20 check-ups, about every 20-40 equivalent full cycles, up to ~180 equivalent full
  cycles ahead.

Code: `knee_analogs/trajectory.py` (method, fixed before the first run), `scripts/trajectory_study.py`
(runner); leakage test in `tests/` (changing a cell's data after N changes none of its forecasts; a
cell is never its own analog).

## Method (pre-registered)

- **Inputs at N:** the same early-cycle features as Test A, plus the cell's SOH points up to N. Its
  current SOH (the *anchor*) is a straight line through its points in (N/2, N], evaluated at N.
- **Analogs (primary: matched state).** The 5 most similar other cells in early-feature space
  (scaling fit on the other cells only, inverse-distance weights), chosen among cells measured beyond N.
  - *Matched state* (primary): each analog is aligned at the point where it had the query's current SOH
    and followed from there.
  - *Same cycle*: each analog's change after cycle N is added to the query's anchor.

  The forecast is the weighted mean where ≥ 3 analogs have data; the band is the min-max of the
  analogs.
- **Baselines:**
  - the cell's own straight-line trend over (N/2, N];
  - the cell's own power law (1 − SOH = a·t^b);
  - the average future of all other cells, and of other cells with the same protocol;
  - ridge regression of the future SOH change on the features, at 30 future cycles.
- **Score:** absolute SOH error in percentage points at every check-up after N. Each cell's mean is
  taken first, then the mean over cells. Methods are compared on the check-ups every method can
  forecast. The analogs cover 92-94% of LSD's future check-ups and 70-91% of Mohtat's (they need
  3 analogs measured that far). Bootstrap 95% CIs over cells.

## LSD first life (86 cells, 16 protocols)

All 86 cells, including the 11 without a clear knee, since no knee is needed here.

Mean SOH error (% points) over each cell's whole measured future:

| first N cycles | cells | analogs (matched state) | analogs (same cycle) | ridge regression | same-protocol average | average of all other cells | own linear trend | own power law |
|---|---|---|---|---|---|---|---|---|
| 25 | 85 | 1.38 | 1.41 | 1.26 | 1.26 | 3.33 | 4.78 | 51 |
| 50 | 86 | **1.21** | 1.22 | 1.15 | 1.24 | 2.94 | 3.42 | 10.7 |
| 100 | 73 | 1.21 | 1.23 | 1.18 | 1.25 | 2.59 | 2.64 | 4.5 |
| 200 | 58 | 1.18 | 1.14 | 1.11 | 1.23 | 2.42 | 2.22 | 2.8 |

From 50 cycles, by how far ahead:

| cycles ahead | 1-146 | 146-336 | 336-631 | 631-1086 |
|---|---|---|---|---|
| analogs (matched state) | 0.49 | 1.07 | 1.44 | 1.68 |
| own linear trend | 0.92 | 2.62 | 6.31 | 10.96 |
| average of all other cells | 1.89 | 2.38 | 3.01 | 2.15 |

- **The whole future is forecast to about 1.2 SOH points** on average, and still to ~1.7 points
  600-1,100 cycles ahead.
- **Against the cell's own trend: far better.**
  - The linear trend misses the knee, so its error grows with the horizon (to 11 points).
  - The power law is unstable on 25-50 cycles.
  - Analogs minus own linear trend: −2.2 points [−2.9, −1.6] at N = 50.
- **Against the population average: far better** (−1.8 [−2.1, −1.5]). Matching to similar cells matters.
- **Against ridge regression and the same-protocol average: no difference.**
  - N = 50: +0.06 [−0.10, +0.22] and −0.04 [−0.19, +0.10].
  - As with the knee, much of what the analogs do here is find the protocol, and a regression on the
    same features gets there too.
- **Matched state vs same cycle:** practically identical (differences ≤ 0.04 points).
- **The band is too narrow.** The min-max of 5 analogs holds the measured future 63-67% of the time.
  It is a range of plausible paths, not a calibrated 90% interval.
- **Where it fails.** Cells that fade faster or slower than every look-alike. Example: cell40 (P12),
  forecast 5.7 points too optimistic (`reports/lsd1/figs/trajectory_first50.png`).

## Mohtat NMC pouch (21 cells, each under its own condition)

Here N is in equivalent full cycles. Only 13-16 cells have enough check-ups by N to be forecast, and
no two share a condition, so these numbers are indicative only.

| first N EFC | cells | analogs (matched state) | analogs (same cycle) | ridge regression | average of all other cells | own linear trend | own power law |
|---|---|---|---|---|---|---|---|
| 75 | 13 | 1.54 | 1.31 | 1.60 | 1.38 | 2.68 | **0.97** |
| 100 | 15 | 1.61 | 1.53 | 1.66 | 1.83 | 1.94 | **1.29** |
| 150 | 16 | 1.24 | 1.29 | 1.51 | 1.77 | 1.15 | 1.22 |

- **The cell's own power-law trend is best or tied.** Analogs minus power law: +0.56 [+0.13, +1.00]
  at N = 75, +0.30 [−0.10, +0.72] at N = 100, 0.0 at N = 150.
- **Why the power law works here:** most Mohtat cells fade smoothly with no knee within the test, and
  at −5 °C the fade even slows down.
- **The analogs help only against the population average at N = 150** (−0.52 [−0.93, −0.15]). With
  one cell per condition there is rarely a true look-alike.
- **Band:** covers 33-67% of the measured points.

## Bottom line

1. **Forecasting the whole curve works where there are precedents.** On LSD, from 50 cycles, the analogs
   forecast the next 1,000 cycles to about 1.2 SOH points, including the knee where similar cells had
   one. A trend fitted to the cell's own early data misses the knee and drifts off (up to 11 points).
2. **But it is not unique to analogs.** Ridge regression on the same features and the same-protocol
   average are just as accurate on LSD. The value of the analogs is that the forecast is a set of real,
   explainable trajectories ("cells 12, 40 and 47 went this way") with a band, not that it is more
   accurate.
3. **Without precedents, the cell's own trend wins.** On Mohtat (every cell its own condition, mostly
   knee-free fade), the simplest physically shaped fit, a power law, is as good or better.
4. **The bands need calibration** (63-67% coverage on LSD) before they can be read as uncertainty.

Outputs:
- `reports/<dataset>/trajectory_summary.csv`: errors by method, N and horizon;
- `reports/<dataset>/trajectory_ci.csv`: bootstrap CIs;
- `reports/<dataset>/figs/trajectory_first<N>.png`: error vs horizon and four example cells.

The per-check-up forecasts are written to `trajectory_first<N>.csv` but not committed (they contain
the measured curves).
