I want to test whether a nearest-neighbour "analog" method (a numeric RAG) can predict the **knee** of lithium-ion battery capacity-fade curves using my own datasets in `C:\Users\alexa\Datasets Batteries`: first within one dataset, then **across datasets** (memory built from some datasets, tested on another).

## Starting point (already built and tested)
Repo `https://github.com/alehrx/YFinance` (being renamed to `battery-knee-analogs`; the old URL keeps redirecting), branch `claude/sp500-news-metrics-rag-8lqacw`. Clone it if it isn't here yet, and read `README.md` and `docs/results_severson_b2.md` first. The relevant files:
- `knee_analogs/core.py`:
  - `load_cells(folder)`: one CSV per cell with `cycle`, `QD` (discharge Ah), and optionally `IR`, `Tavg`, `chargetime`, `charge_policy`. It drops duplicate files by content hash.
  - `knee_point()`: Bacon-Watts-style two-segment fit on the smoothed capacity curve, up to 80% end of life.
  - `early_features()`, `curve_vector()`, `parse_policy()`.
- `knee_analogs/evaluate.py`: leave-one-cell-out tests.
  - Test A: predict the knee from the first N cycles. Compares analogs (5 most similar cells) with ridge regression, protocol-only regression and the plain average.
  - Test B: early warning, matching a cell's current state to snapshots of other cells to predict cycles left until the knee.
- `run_study.py --data <folder>`: runs everything and writes `reports/<folder name>/*.csv` and figures.
- `knee_analogs/inspect_datasets.py`: read-only overview of a dataset folder.
- `scripts/fetch_severson_b2.py`: downloads the public example data. Use it as a reference dataset in Phase 2 if useful.

Results on the public Severson 2019 batch 2017-06-30 (42 LFP cells, per-cycle summary only):
- From 100 cycles, analogs average 28 cycles of error against 36 for regression and 45 for the average.
- Among typical cells, no method ranks cells (rank correlation about 0). The analogs mostly identify which group a cell belongs to.
- Cells with no similar peers are badly predicted.
- Analogs are best close to the knee (7 cycles of error at cycle 350).
- Knees there fall at about 93-96% of initial capacity.

## Ground rules (all phases)
- Read only inside `C:\Users\alexa\Datasets Batteries` and the repo; **ask me before reading anything else**. Never open pickle files.
- Keep raw data untouched. Converted data goes under `data/<dataset>/`, which git ignores; never commit data.
- No leakage:
  - a cell is never matched to its own cycles;
  - check for duplicated or near-identical cells before testing;
  - fit scaling, regression and feature weights only on the cells the test is allowed to see;
  - don't tune settings on test results.
- Keep me posted after each step. Stop and ask when a format is unclear, rather than guessing.

## Phase 1: one dataset
1. Run `python -m knee_analogs.inspect_datasets "C:\Users\alexa\Datasets Batteries"` and summarise each dataset for me:
   - chemistry and format if identifiable;
   - nominal capacity;
   - number of cells, and how many were cycled past the knee;
   - cycles per cell;
   - test conditions (temperature, C-rates, depth of discharge);
   - what is measured, and how often (every cycle or periodic check-ups);
   - whether voltage curves exist.
2. Propose **one** dataset to start with: the most cells past the knee, with per-cycle capacity; voltage curves are a bonus. Wait for my OK.
3. Write an adapter to the per-cell table `load_cells` expects.
   - If capacity is only measured in periodic check-ups, adapt the smoothing and knee detection to sparse points.
   - Make nominal capacity and the end-of-life threshold per dataset, not the hard-coded `NOMINAL_AH = 1.1`. Use the datasheet value if known, else the cell's initial capacity.
4. Plot every cell with its detected knee and end of life, and look at the plots. Report wrong or ambiguous knees, then fix the detection or exclude those cells with a stated reason.
5. Run Test A and Test B leave-one-cell-out. Report:
   - error in cycles and %;
   - the bootstrap 95% CI of the analogs-vs-regression difference;
   - how often the matched cells' range contains the actual knee;
   - the **rank correlation within the main group**, not only overall error.

   If the dataset has several conditions, also leave out a whole condition at a time.
6. If voltage curves exist, add Severson-style ΔQ(V) features (Q(V) at cycle N minus an early cycle: variance, minimum, mean) and dQ/dV peak position and height, then compare with summary-only features.
7. Write `docs/results_<dataset>.md` and add a row to the results table in `README.md`. Commit.

## Phase 2: across datasets
Do this once at least two datasets have working adapters, ideally three or more. Include the Severson data if that helps.

8. Make the features **scale-free**, so cells from different datasets can be compared:
   - capacity as a fraction of initial capacity;
   - fade rates per cycle relative to initial capacity;
   - curvature;
   - resistance growth relative to its start;
   - condition descriptors (temperature, average charge and discharge C-rate, depth of discharge, chemistry flag).

   Add targets that don't depend on absolute cycle counts:
   - **state of health at the knee**;
   - **cycles to the knee as a multiple of the cycles needed to lose the first 2% of capacity**;
   - for early warning, "knee within the next 10% of elapsed life".

   Keep the raw-cycle target too, for comparison.
9. First, a descriptive check: plot, per dataset, the distributions of state of health at the knee and of the scale-free fade features. If the knee sits at very different states of health, say so before modelling.
10. **Leave-one-dataset-out:** the memory and the regression contain all cells of the other datasets; predict every cell of the held-out dataset. Compare:
    - analogs on raw features;
    - analogs on scale-free features;
    - analogs on features standardised within each dataset;
    - ridge regression on the same features;
    - the held-out dataset's own average, a baseline that peeks at the target dataset's mean, as a reference;
    - the within-dataset leave-one-cell-out result as an upper bound.
11. **Few-shot:** add k = 0, 1, 3, 5, 10 cells of the held-out dataset to the memory and evaluate on its remaining cells. Repeat with several random draws of those k cells. For regression, refit with them included. Plot error against k. This answers "how many cells of a new test campaign must run to end of life before predictions become useful".
12. Do the same for Test B (early warning), where transfer is more plausible because the fade-rate upturn and resistance rise are mechanism-linked.
13. Write `docs/results_cross_dataset.md`, covering:
    - which targets and features transfer and which don't;
    - the few-shot curves;
    - an honest bottom line on whether training on one dataset and testing on another works, and under what conditions (same chemistry? same temperature range?).

    Commit the code as reusable functions in `knee_analogs/`, for example `cross_dataset.py`, with a runner script and tests for the leakage rules.
