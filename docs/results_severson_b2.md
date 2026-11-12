# Results: Severson 2019, batch 2017-06-30 (preliminary)

**Question.** Can a "numeric RAG" predict the **knee** of a battery's capacity-fade curve? The method
finds other cells whose early cycles looked similar and uses when *their* knees came.

**Data.** Severson et al. 2019 (Nature Energy), batch 2017-06-30: A123 LFP/graphite 18650 cells,
1.1 Ah, under fast-charging protocols. The files came from a public GitHub mirror
(`github.com/Txiaoxiao/LSTM-Neural-Network-for-Time-Series-Prediction`, folder `data/`;
`python scripts/fetch_severson_b2.py` downloads them to `data/severson_b2/`). They hold one
summary row per cycle: discharge capacity QD, internal resistance, average temperature and charge
time. There are no voltage curves, so the ΔQ(V) features from the paper are not available. Of 48
files, one is an exact duplicate (cell47 = cell0) and five cells continue from batch 1 with
incomplete history; both are excluded, leaving **42 cells**.

**Knee.** The knee is the breakpoint of a continuous two-segment linear fit to the smoothed capacity
curve up to 80% end of life (Bacon-Watts style). Every detection was checked by eye
(`reports/severson_b2/figs/knees_detected.png`). 39 cells have their knee between cycles 325 and 405.
Three short-lived cells have it between cycles 80 and 160.

**Validation.** Leave-one-cell-out: each cell is predicted using only the other 41. The duplicate
file was removed for exactly this reason; it would have matched itself.

## Test A: predict the knee cycle from the first N cycles

Mean absolute error in cycles (`reports/severson_b2/testA_summary.csv`):

| method | first 50 | first 100 | first 150 | first 200 |
|---|---|---|---|---|
| average of the other cells | 45 | 45 | 45 | 45 |
| regression on charging protocol | 51 | 51 | 51 | 51 |
| regression on early-cycle features (ridge, log target) | 41 | 36 | 48 | 29 |
| **analogs: 5 most similar cells by early-cycle features** | **28** | **28** | **23** | **22** |
| analogs: 5 most similar cells by curve shape | 37 | 36 | 33 | 32 |

- **Typical cells.** On the 39 cells with knees between cycles 325 and 405, the analogs average 20
  cycles of error from 100 cycles, against 32 for regression and for the average. The bootstrap
  95% CI of the gap excludes zero.
- **Range.** The range spanned by the 5 matched cells contains the actual knee 62-69% of the time.
- **End of life (80% capacity) from 100 cycles.** Analogs 30 cycles (9.2%), regression 38, average 42.

**What it does and doesn't do.**
- **It identifies the group.** Analogs win because they work out which group a cell belongs to
  (typical vs fast early fade) and predict that group's typical knee, without the wild guesses
  regression makes.
- **It does not rank cells within the typical group.** The rank correlation between predicted and
  actual knee among the 39 typical cells is about 0 from 100 cycles (+0.14 from 200). Regression
  does no better (+0.07 / +0.36).
- **It fails on cells without precedent.** The three short-lived cells are badly predicted (cell1:
  predicted 235, actual 79), because only two other cells behave like them. Analogs cannot predict
  cases without precedent.

## Test B: early warning during life

At checkpoints every 50 cycles, the method predicts the cycles left until the knee. It matches the
cell's current state (capacity lost so far, fade rate over the last 50 cycles, change in fade rate,
resistance and temperature drift, protocol) to snapshots of other cells at any cycle.

| | average error, cycles left until knee | AUC for "knee within 50 cycles" |
|---|---|---|
| cycle count only | **20** | 0.91 |
| regression on state | 33 | 0.96 |
| **analogs on state** | 27 (but **7 at cycle 350**) | **0.98** |

- **Early in life,** the cycle count is the best guide, because most knees fall in a narrow window.
- **Close to the knee,** the analogs pick up the start of the accelerating fade and become clearly
  the best (7 cycles of error at cycle 350, against 15).
- **Adding the cycle number to the state** brings all three methods to 20-23 cycles of error on
  average; AUC is 0.97 for both analogs and regression.

## Conclusions and next steps

1. Analog matching is competitive with or better
   than regression and gives an interpretable answer ("your cell looks like cells 34, 22 and 17,
   whose knees came at cycles 372-401") with a range.
2. With summary data only, it mainly identifies the group a cell belongs to and detects knee onset
   late in life. Finer within-group prediction needs richer inputs:
   - the voltage-curve features (ΔQ(V) between cycles, dQ/dV peaks), which in Severson's paper
     carried most of the early signal;
   - more cells, especially more of the rare fast-fading kind;
   - several batches or protocols, to test whether it transfers to new conditions.
3. The LLM step (explaining matches, combining with test notes) was not run here: no API key in
   this environment.

Run it on another dataset with `python run_study.py --data <folder>`. The folder needs one CSV
per cell with `cycle` and `QD`, plus optionally `IR`, `Tavg`, `chargetime` and `charge_policy`.
Other formats need a small adapter in `knee_analogs/core.py::load_cells`.
