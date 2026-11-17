# battery-knee-analogs

Can we predict the **knee** of a lithium-ion cell's capacity-fade curve by finding other cells
whose early life looked similar, and looking at when *their* knees came?

This is a nearest-neighbour "analog" method, also called a numeric RAG (retrieval over numbers
instead of text). It is tested leave-one-cell-out against regression and simple baselines.

## How it works

1. **Per cell, per cycle:** discharge capacity, plus resistance, temperature and charge time when available.
2. **Knee:** the breakpoint of a continuous two-segment linear fit to the smoothed capacity curve,
   up to 80% end of life. It is checked by eye on a plot of every cell.
3. **Test A, early prediction:** from the first N cycles, compute summary features (capacity lost,
   fade rate, curvature, resistance and temperature drift, charging protocol). Find the 5 most
   similar *other* cells and predict the knee from theirs. The output is a cycle estimate plus the
   range spanned by the matched cells.
4. **Test B, early warning:** at checkpoints during life, match the cell's current state to
   snapshots of other cells at any cycle, and predict the cycles left until the knee.
5. **Baselines:** the average of the other cells, regression on the charging protocol, and ridge
   regression on the same early-cycle features.

## Results so far

| dataset | cells | knee error from first 100 cycles, analogs / regression / average | details |
|---|---|---|---|
| Severson 2019, batch 2017-06-30 (LFP/graphite, fast charging) | 42 | 28 / 36 / 45 cycles | [docs/results_severson_b2.md](docs/results_severson_b2.md) |
| LSD first life (2.4 Ah, 16 protocols, knee from C/5 check-ups) | 75 | 45 / 99 / 299 cycles; same-protocol average 41 | [docs/results_lsd1.md](docs/results_lsd1.md) |

In short:
- Analogs beat regression and the average, mostly by identifying which group a cell belongs to.
- Within the main group, no method ranks cells yet.
- Cells without similar peers are missed.
- Close to the knee, analogs give the sharpest early warning (AUC 0.98 for "knee within 50 cycles").
- **LSD, 16 protocols:** analogs beat regression by about half, but only match the average of the other
  cells with the same protocol, which the protocol makes easy to find. Within a protocol, no method ranks
  cells well. Voltage-curve features (ΔQ(V), dQ/dV, DVA) do not help the analogs. On an unseen protocol
  the analogs fail and regression on the protocol settings does best.
- **Whole future trajectory** (not one number): from 50 cycles, the analogs forecast each LSD cell's
  entire future capacity curve to about 1.2 SOH points, up to 1,000 cycles ahead. That is far better
  than extrapolating the cell's own trend, but no better than ridge regression or the same-protocol
  average. On Mohtat (21 cells, each under its own condition) the cell's own power-law trend is as good
  or better. See [docs/results_trajectory.md](docs/results_trajectory.md).
- **Without an ageing database of the cell type** (766 cells, 6 datasets, LFP / NCA / NMC):
  - **Speed carries most of the information:** the time to reach 90 % SOH explains 76 % of the variance
    of the time to 80 %. The curve's shape after 90 % differs by chemistry.
  - **Zero-shot forecast:** seeing a cell only down to 90 % SOH, analogs from *other* datasets,
    Chronos-2 and even the cell's own straight-line trend all forecast the rest to ~1.95 SOH points,
    against ~0.97 with a same-type database.
  - **LLM, zero-shot** (`claude-opus-5-5`, blinded): given the curve to 90 %, the chemistry, the
    conditions and analogs from other datasets, it forecasts the rest to 1.51 SOH points, beating
    every non-LLM zero-shot method (−0.32 vs the linear trend, CI [−0.46, −0.19]). Most of the gain
    comes from chemistry and mechanism knowledge, not the analogs: without analogs it scores 1.62.
    Averaging the three tied methods (1.46) does about as well. It is wrong on cells that keep
    decelerating, and the data are public, so it may have seen them.
  - **Small model:** the same prompts on Claude Haiku 5.5 ($0.78 vs $20) give 1.71: the same pattern
    and about half the gain. It is clearly better than the trend only in the near term (1.00 vs 1.40 up
    to 1.5 t90).
  See [docs/results_zero_shot.md](docs/results_zero_shot.md).
- **Accelerated tests → use conditions** (Scenario B, 5 cell types): seeing only harsh-condition cells
  down to 90 % SOH, the shape transfers (1.6 SOH points given the true speed), but the speed does not:
  fitting it against conditions errs by 10.6 points. An LLM judging the speed-up from chemistry and
  mechanisms halves that (4.6-6.5 over 4 runs per model). The LLM's speed with the data's shape gives
  4.35, against 2.35 for a use-condition database. Single LLM runs vary a lot; the gain from showing it
  the test data is within that noise.
  Adding a short test of the use cell itself (to 98 % SOH, 11-23 % of the time to 90 %) brings the
  error to 4.6-5.0. The start of life is not a scaled copy of the rest of the curve; scaling by the fade
  rate at the end of the short test (after the break-in) gives 3.9. A test to 95 % (half the time to
  90 %) gives 2.7, as good as a use-condition database, but then a straight line does as well. See [docs/results_scenario_b.md](docs/results_scenario_b.md).
- **LLM trial:** a blinded LLM reading the same analogs ties overall (27 vs 28 cycles). It is worse on
  typical cells but far better on cells with no precedent, because it extrapolates beyond the
  analogs. Adding chemistry and mechanism context did not change the numbers. See the LLM section in
  [docs/results_severson_b2.md](docs/results_severson_b2.md).

## Quick start

```bash
pip install -r requirements.txt
python scripts/fetch_severson_b2.py                 # public example data -> data/severson_b2/
python run_study.py --data data/severson_b2         # -> reports/severson_b2/ (tables + figures)
python -m pytest -q tests
```

## Using your own datasets

```bash
python -m knee_analogs.inspect_datasets "C:\Users\alexa\Datasets Batteries" > datasets_overview.txt
```

This gives a read-only overview: file types, columns or variables, and the first rows. `run_study.py`
expects a folder with one CSV per cell containing `cycle` and `QD` (discharge capacity, Ah),
plus optionally `IR`, `Tavg`, `chargetime`, `charge_policy` and `condition`. Other formats need a
small adapter that writes that layout to `data/<dataset>/`.

The adapter also writes `data/<dataset>/dataset.json` with the per-dataset settings
(`knee_analogs.core.DatasetSpec`). Without the file, the Severson batch-2 settings are used.
Set them from the dataset's description, before looking at any result:

| key | meaning | Severson default |
|---|---|---|
| `nominal_ah`, `eol_fraction` | end of life = `eol_fraction` x `nominal_ah` | 1.1, 0.8 |
| `min_start_fraction` | skip cells starting below this fraction of nominal (`null` keeps all) | 0.945 |
| `sparse` | capacity only at check-ups: no smoothing, knee at least 3 check-ups from either end, end of life interpolated | false |
| `early_cycles`, `main_n` | Test A: predict from the first N cycles; N for the headline and figures | 50-200, 100 |
| `checkpoints`, `state_window`, `warning_window` | Test B checkpoints, fade-rate window, "knee within" window (cycles) | 100-350, 50, 50 |
| `eol_basis` | `"nominal"` or `"initial"`: end of life relative to nominal or to each cell's own initial capacity | nominal |
| `knee_from` | `"QD"` or `"checkup"`: measure knee and end of life on the per-cycle capacity or on `checkups/<cell>.csv` | QD |
| `knee_accelerating_only` | only breakpoints where the fade gets faster count as a knee | false |
| `exclude_cells` | `{cell: reason}` for cells left out, decided from the capacity curves only | {} |

`run_study.py` also writes `knees.csv` (knee, end of life, whether it was reached),
`near_duplicates.csv`, `testA_detail.csv` (error overall, on the main group and on outliers, rank
correlation within the main group and within each condition, range coverage) and the bootstrap CIs of
analogs minus regression and minus the average of same-condition cells.
When at least two conditions have two or more cells each, it also runs leave-one-condition-out.
`prompts/local_session.md` is a ready-made prompt for a Claude Code session on your own computer.

## Layout

```
knee_analogs/
  core.py               loading (de-duplicates files), smoothing, knee detection, early-cycle features
  evaluate.py           leave-one-cell-out tests A and B, baselines, summaries
  inspect_datasets.py   read-only overview of a dataset folder
  voltage_features.py   ΔQ(V), dQ/dV and DVA features from adapter-written curves
  trajectory.py         whole-future capacity-curve forecasts (analogs and baselines)
  zero_shot.py          cross-dataset forecasts from a cell's curve down to 90 % SOH
  scenario_b.py         accelerated tests to 90 % SOH -> curves at untested use conditions
run_study.py            runs everything for one dataset folder, writes reports/<dataset>/
llm_trial/              blinded LLM trials: knee prompts (Severson), zero_shot_llm.py, scenario_b_llm.py
scripts/                data download helpers, dataset adapters (convert_lsd.py, convert_mohtat.py),
                        knee review (check_knees.py), voltage-feature comparison, trajectory_study.py,
                        build_library.py, universality.py, zero_shot_study.py, chronos_forecast.py,
                        fig_zero_shot_llm.py, scenario_b_study.py, scenario_b_repeats.py,
                        scenario_b_early.py, scenario_b_early2.py, fig_scenario_b_tradeoff.py,
                        fig_scenario_b_predictions.py, scenario_b_anchor.py
docs/                   results write-ups per dataset
reports/<dataset>/      tables (.csv) and figures (.png/.pdf)
tests/                  synthetic-cell tests: knee detection, de-duplication, no self-matching
data/                   local data (not committed)
```
