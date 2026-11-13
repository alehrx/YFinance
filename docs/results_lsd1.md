# Results: LSD first life (86 cells, 16 protocols)

**Question.** Can a "numeric RAG" predict the **knee** of a cell's capacity-fade curve on a dataset with
many cycling protocols, and does it do more than recognise the protocol?

**Data.** `C:\Users\alexa\Datasets Batteries\LSD 1st life`: 86 Arbin CSV files, one cell each, 10 s
logging of voltage, current, capacity and temperature at every cycle. There is no description file, so
everything below was measured from the files:
- **Cell.** One cell type, 2.0 or 3.0-4.2/4.3 V. The protocol currents are multiples of 0.48 A
  (= C/5 of **2.4 Ah**), so nominal capacity is taken as 2.4 Ah. Chemistry not stated (the voltage window
  suggests a Ni-based positive electrode with graphite).
- **Cycle.** CC-CV charge ending at 0.048 A (C/50), 15 min rest, discharge, 15 min rest, 25-30 °C.
- **Check-up.** Every 5th cycle: C/2 CC-CV charge to 4.2 V, 30 min rest, **C/5 discharge to 3.0 V**,
  30 min rest. These are the same for every protocol.
- **16 protocols** (`condition` P01-P16; 2-8 cells each) vary:
  - charge current: 0.5, 1, 1.5 or 2C;
  - discharge current: 0.2, 0.5, 1, 2 or 3C;
  - upper voltage: 4.2 or 4.3 V;
  - lower cut-off: 3.0 or 2.0 V;
  - pulsed discharge in P16.
- **Life** (to 80% of the initial C/5 capacity): 72-1363 cycles. Charge rate dominates: 2C charge
  gives ~75 cycles, 0.5C charge with 1C discharge ~900.

**Adapter.** `scripts/convert_lsd.py` → `data/lsd1/` (not committed):
- per-cell table of the ageing cycles (`cycle`, `QD`, `QC`, `Tavg`, `Tmax`, `chargetime`, `condition`);
- protocol descriptors `proto_chg_C`, `proto_dis_C`, `proto_vmax`, `proto_vmin` and `proto_pulsed`,
  measured on the first cycles and known before the test starts;
- the C/5 check-ups in `checkups/`;
- Q(V) curves and the raw check-up discharges in `curves/*.npz`.

There are no duplicate or near-identical cells (`near_duplicates.csv` is empty).

## Knee: where and how it is measured

`reports/lsd1/figs/knee_check.png` shows every cell. Checked by eye before any prediction was run:
- **Ageing-cycle capacity is not usable at high rates.** At 2-3C discharge (P07, P08) and with pulses
  (P16), it shows rate artefacts and an early drop. The two-segment fit put the "knee" at that drop:
  cell20 at cycle 117 although the cell bends near cycle 900.
- **Fix 1: measure on the check-ups.** Knee and end of life are now measured on the **C/5 check-ups**,
  the same rate and 3.0-4.2 V window for every protocol (`knee_from: "checkup"`, sparse points, no
  smoothing). For 1C protocols this changes the knee by only a few percent.
- **Fix 2: the fade must accelerate.** Only breakpoints where fade gets faster count as a knee
  (`knee_accelerating_only`). Without this, cells with an early fast drop got their "knee" at cycle 70.
- **Fix 3: exclude cells without a clear knee.** The rule, set from the curves only: fade after the
  breakpoint at least 1.3× faster than before, breakpoint between 20% and 95% of end of life. **11 cells
  fail** and are excluded (reasons in `scripts/convert_lsd.py`, `exclude_cells`):
  - **all six P06 cells** (2C charge, 1C discharge): capacity falls almost linearly from the first cycle
    to end of life, so there is no knee to predict (five fail the 1.3× test; cell13's best breakpoint is
    at cycle 12, 17% of its life);
  - cell60 (P01) and cell24/cell25 (P08/P09): nearly linear;
  - cell22 and cell31: the break sits in the last 3% before end of life.

**75 cells** remain, with knees between cycles 48 and 1189 (median 0.68 of end of life). All reached
end of life.

**Settings**, fixed before the first prediction (`data/lsd1/dataset.json`, written by the adapter):
- end of life at 80% of each cell's initial check-up capacity, since the 4.3 V protocols start ~8% higher;
- Test A from the first 25, 50, 100 and 200 cycles, headline at **50**, because the shortest cells reach
  end of life after ~72 cycles;
- Test B checkpoints at 50-1000 cycles, a 25-cycle fade-rate window, and "knee within 50 cycles".

## Test A: knee from the first N cycles (leave-one-cell-out)

Mean absolute error in cycles (MAPE in brackets), 75 cells (74 at N = 25, 73 at N = 200):

| method | first 25 | first 50 | first 100 | first 200 |
|---|---|---|---|---|
| average of the other cells | 294 (120%) | 299 (121%) | 299 (121%) | 300 (118%) |
| **average of the other cells with the same protocol** | **40 (14%)** | **41 (13%)** | **41 (13%)** | 41 (13%) |
| regression on protocol descriptors | 90 (26%) | 89 (26%) | 89 (26%) | 92 (27%) |
| ridge regression on early-cycle features | 94 (26%) | 96 (27%) | 99 (27%) | 97 (25%) |
| analogs: 5 most similar cells, early-cycle features | 66 (21%) | 53 (17%) | 45 (15%) | 40 (12%) |
| analogs: features + curve shape | 47 (15%) | 48 (14%) | 58 (19%) | **37 (10%)** |
| analogs: curve shape only | 202 | 210 | 201 | 154 |

- **Analogs beat ridge regression** by 28-57 cycles. The bootstrap 95% CI of the gap excludes zero at
  every N: −43 cycles [−66, −20] at N = 50.
- **They do not beat the same-protocol average.**
  - At N = 25 they are worse (+26 [8, 45]).
  - At N = 50-200 they are level: +13 [−1, 28], +5 [−6, 17], −2 [−11, 7].
  - Here the protocol is known in advance and largely sets the life, so most of what the analogs do is
    find cells of the same protocol.
- **Range.** The range of the 5 matched cells' knees contains the actual knee 69-77% of the time.
- **Ranking.** Across all cells the rank correlation is 0.95 (Tukey's fences flag no outliers, so the
  "main group" is all 75 cells), but that only says that protocols differ. **Within a protocol**
  (predicted and actual knee both centred on the protocol's mean), the analogs reach +0.07 to +0.24
  (+0.30 to +0.38 with curve shape) and ridge regression −0.06 to −0.16. Some within-protocol signal,
  but weak.
- **By life (N = 50):**
  - short-lived cells (knee < 150): analogs 26 cycles (27%), same-protocol average 17 (20%);
  - knee 150-500: 37 (14%) vs 34 (13%);
  - knee > 500: 94 (11%) vs 69 (8%).

  The analogs lose most on P07 and P08, whose 2-4 cells have very tight knees (3-5% spread) but
  resemble other long-lived protocols in their first 50 cycles, and on P14 (0.2C discharge).
- **End of life** from the first 50 cycles: same-protocol average 49 cycles (10.6%), analogs 55 (11.9%),
  regression 83 (16.3%).

### Unseen protocol: leave one protocol out (N = 50, 15 protocols)

| method | knee error | MAPE | range covers actual |
|---|---|---|---|
| average of all other cells | 314 | 133% | - |
| **regression on protocol descriptors** | **133** | **41%** | - |
| ridge regression on early-cycle features | 181 | 61% | - |
| analogs: early-cycle features | 205 | 79% | 48% |

With the protocol's own cells removed, **analogs fail**: their range covers the knee only 48% of the
time. Regression on the five protocol numbers extrapolates best. Analogs need a precedent.

## Test B: early warning (cycles left until the knee)

Checkpoints at 50-1000 cycles, 285 checks on cells not yet past their knee.

| checkpoint | checks | analogs on state | regression on state | cycle count, all cells | **cycle count, same protocol** |
|---|---|---|---|---|---|
| 50 | 73 | 185 | 155 | 303 | **40** |
| 100 | 56 | 128 | 147 | 307 | **47** |
| 200 | 44 | 112 | 111 | 284 | **48** |
| 300 | 37 | 113 | 84 | 227 | **54** |
| 400 | 27 | 102 | 77 | 118 | **61** |
| 600 | 25 | 93 | 77 | 107 | **66** |
| 800 | 20 | 101 | 83 | 71 | **34** |
| all | 285 | 130 | 117 | 238 | **48** |

AUC for "knee within 50 cycles":

| method | AUC |
|---|---|
| analogs | 0.94 |
| regression | 0.95 |
| cycle count, same protocol | 0.95 |
| cycle count, all cells | 0.51 |

- **The state alone isn't enough.** The state is capacity lost, fade rate over the last 25 cycles,
  change in fade rate, temperature drift and the protocol descriptors. Matching it across protocols
  whose time scales differ tenfold does not tell how many cycles are left.
- **Cycle count within the protocol is far better** (48 cycles of error against 117-130).
- The near-knee advantage that analogs showed on Severson (7 cycles at cycle 350) does not appear here.
- Leaving out a whole protocol, the analogs' AUC drops to 0.89 and regression's to 0.94.

## Voltage-curve features (step 6)

`knee_analogs/voltage_features.py` computes, from cycles ≤ N only:
- **ΔQ(V) of the ageing discharges** (cycle N vs cycle 10, as in Severson et al.): log variance,
  minimum and mean;
- **ΔQ(V) of the C/5 check-ups** (latest ≤ N vs first);
- **dQ/dV (ICA) of the check-ups**:
  - height and position of the main peak (3.73 V; falls from ~10 to ~6 Ah/V over life),
  - height of the 3.52 V peak,
  - capacity in the 3.85-4.15 V shoulder;
- **dV/dQ (DVA)**: capacity before and after the dV/dQ valley and the valley height, as changes from
  the first check-up.

`scripts/compare_voltage_features.py`, same leave-one-cell-out folds (`reports/lsd1/voltage_features_comparison.csv`):

| features (N = 50) | analogs | regression | within-protocol rank, analogs |
|---|---|---|---|
| summary only | 53 | 96 | 0.13 |
| + ΔQ(V) ageing cycles | 56 | 91 | 0.11 |
| + ΔQ(V) check-ups | 56 | 95 | 0.15 |
| + dQ/dV, dV/dQ | 70 | 93 | 0.10 |
| + all voltage features | 66 | 92 | 0.13 |
| protocol + voltage features only | 72 | 86 | 0.00 |

- **Analogs: no gain.**
  - At N = 100-200 the voltage features change the error by less than 1 cycle (CIs include 0).
  - At N = 25-50, adding the dQ/dV/DVA features makes the analogs **worse** (+13 to +26 cycles, CI
    excludes 0): with equal feature weights, early-life peak changes that are mostly noise dilute the
    distance.
- **Regression: a small gain.** Check-up ΔQ(V) and ICA/DVA gain 6-8 cycles, with CIs excluding 0 in three
  cases (ΔQ(V) of the check-ups at N = 25 and 100, ICA/DVA at N = 100), but it stays at ~90 cycles.
- **Ranking within a protocol** does not improve with any set.
- **Interpretation.** In the first 25-200 cycles, the voltage curves carry little information about the
  knee *beyond the protocol* in this dataset. A degradation-mode fit (loss of lithium inventory, loss of
  active material) is a different, physically constrained use of the same curves and is left for Phase 3.

## Conclusions

1. **Analogs beat regression clearly but do not beat the obvious baseline.** On a multi-protocol dataset,
   averaging the other cells of the same protocol is as good from 50 cycles on, and better from 25.
   The analogs' value here is that they find the right protocol group without being told.
2. **Within a protocol, prediction is weak for every method:**
   - analogs: rank correlation up to +0.24 (+0.38 combined with curve shape);
   - regression: about 0.
   Cell-to-cell variation within a protocol (5-25% in knee cycle) is mostly not foreseeable from the
   first 25-200 cycles of summary or voltage-curve data.
3. **Analogs cannot transfer to an unseen protocol.** Regression on protocol descriptors does far
   better there. This is the cross-dataset question of Phase 2 in miniature.
4. **The knee depends on the protocol.** 2C charge (P06) gives linear fade with no knee. High-rate
   discharge capacity hides the knee, which only the C/5 check-ups show.

## Process notes

- **Added after the first run:** the same-protocol average baseline, the within-protocol rank
  correlation and the same-protocol cycle-count baseline for Test B. They were added because the
  headline rank correlation (0.95) and the cycle-count baseline (AUC 0.51) were clearly driven by
  differences between protocols. No setting, feature or exclusion was changed after any prediction.
- **Still to confirm:** cell chemistry and nominal capacity (inferred: 2.4 Ah), the meaning of the
  protocol IDs, and whether this dataset is published.
- `LSD 2nd life` continues the same 86 batteries (same IDs and protocols) from ~70% to 34-56%. It is not
  an independent dataset and is not used here.

## Files

- **Code:**
  - `scripts/convert_lsd.py` (adapter and settings);
  - `scripts/check_knees.py` (knee review and exclusion rule);
  - `run_study.py --data data/lsd1` (Tests A and B);
  - `scripts/compare_voltage_features.py` and `knee_analogs/voltage_features.py` (step 6).
- **Outputs** in `reports/lsd1/`:
  - `knees.csv` and `knee_check.csv`;
  - `testA_detail.csv`, with the bootstrap CIs in `testA_analogs_minus_*_ci.csv`;
  - the leave-one-protocol-out tables;
  - `testB_*.csv`;
  - `voltage_features_comparison.csv`;
  - figures in `figs/`.
