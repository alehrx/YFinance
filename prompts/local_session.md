I want to test whether nearest-neighbour "analogs" (a numeric RAG), and an LLM reading those analogs, can predict the **knee** of lithium-ion battery capacity-fade curves using my own datasets in `C:\Users\alexa\Datasets Batteries`. First within one dataset, then **across datasets**, then with an **LLM given extra information** (DVA, operating history).

## Setup
Repo `https://github.com/alehrx/YFinance` (being renamed to `battery-knee-analogs`; the old URL keeps redirecting). The work is on branch `claude/sp500-news-metrics-rag-8lqacw`:

```
git fetch origin claude/sp500-news-metrics-rag-8lqacw
git checkout claude/sp500-news-metrics-rag-8lqacw
pip install -r requirements.txt
```

Clone the repo first if this folder isn't a clone of it. Then read `README.md` and `docs/results_severson_b2.md`. The relevant files:
- `knee_analogs/core.py`:
  - `load_cells(folder)`: one CSV per cell with `cycle`, `QD` (discharge Ah), and optionally `IR`, `Tavg`, `chargetime`, `charge_policy`. It drops duplicate files by content hash.
  - `knee_point()`: Bacon-Watts-style two-segment fit on the smoothed capacity curve, up to 80% end of life.
  - `early_features()`, `curve_vector()`, `parse_policy()`.
- `knee_analogs/evaluate.py`: leave-one-cell-out tests.
  - Test A: predict the knee from the first N cycles. Compares analogs (5 most similar cells) with ridge regression, protocol-only regression and the plain average.
  - Test B: early warning, matching a cell's current state to snapshots of other cells to predict cycles left until the knee.
- `run_study.py --data <folder>`: runs everything and writes `reports/<folder name>/*.csv` and figures.
- `knee_analogs/inspect_datasets.py`: read-only overview of a dataset folder.
- `llm_trial/`: the blinded LLM trial.
  - `build_trial.py` builds anonymized prompts in folds: each fold's query cells are matched only to cells outside the fold, so no query's knee appears in another query's analog list.
  - `score_trial.py`, `plot_trial.py`.
- `scripts/fetch_severson_b2.py`: downloads the public reference dataset.

## What we already know (Severson 2019 batch 2017-06-30, 42 LFP cells, per-cycle summary, no voltage curves)
- **Knee from the first 100 cycles:** analogs 28 cycles of error, regression 36, average 45.
  - Among typical cells, no method ranks cells (rank correlation about 0). The analogs mostly identify which group a cell belongs to.
  - Cells without similar peers are missed.
- **Early warning:** analogs are best close to the knee (7 cycles of error at cycle 350; AUC 0.98 for "knee within 50 cycles"). Early in life, the cycle count alone is as good.
- **Blinded LLM trial** (sub-agents reading anonymized numbers plus 8 analogs per cell):
  - Ties overall (27 vs 28 cycles): worse on typical cells (24 vs 18), much better on the 3 short-lived cells (63 vs 150), because it extrapolates beyond its analogs. One false alarm.
  - Adding chemistry, protocol physics and mechanism knowledge changed the explanations, not the numbers.
  - The 10-90% intervals were well calibrated (76-81% coverage).
  - The LLM recognized the public dataset, so those numbers are optimistic. **My unpublished data is the clean test.**

## Ground rules (all phases)
- Read only inside `C:\Users\alexa\Datasets Batteries` and the repo; **ask me before reading anything else**. Never open pickle files.
- Keep raw data untouched. Converted data goes under `data/<dataset>/`, which git ignores; never commit data.
- No leakage:
  - a cell is never matched to its own cycles;
  - check for duplicated or near-identical cells before testing;
  - fit scaling, regression and feature weights only on the cells the test may see;
  - in batched LLM prompts, never show a query cell's knee as another query's analog;
  - don't tune settings on test results.
- Before any step that calls the Anthropic API, tell me the estimated number of calls and cost, and wait for my OK. `ANTHROPIC_API_KEY` is set in my environment.
- Keep me posted after each step. Stop and ask when a format is unclear, rather than guessing.

## Phase 1: one dataset
1. Run `python -m knee_analogs.inspect_datasets "C:\Users\alexa\Datasets Batteries"` and summarise each dataset:
   - chemistry and format;
   - nominal capacity;
   - number of cells, and how many were cycled past the knee;
   - cycles per cell;
   - conditions (temperature, C-rates, depth of discharge, rests);
   - what is measured, and how often;
   - whether there are voltage curves and **slow check-up cycles (C/20-C/10) usable for DVA/ICA**.
2. Propose **one** dataset to start with: the most cells past the knee, with per-cycle capacity; check-up curves are a bonus. Wait for my OK.
3. Write an adapter to the per-cell table `load_cells` expects.
   - Make nominal capacity and the end-of-life threshold per dataset, not the hard-coded `NOMINAL_AH = 1.1`.
   - If capacity is only measured at check-ups, adapt the smoothing and knee detection to sparse points.
4. Plot every cell with its detected knee and end of life, and look at the plots. Report wrong or ambiguous knees, then fix the detection or exclude those cells with a reason.
5. Run Test A and Test B leave-one-cell-out. Report:
   - error in cycles and %;
   - the bootstrap 95% CI of the analogs-vs-regression difference;
   - how often the matched cells' range contains the actual knee;
   - the **rank correlation within the main group**.

   If there are several conditions, also leave out a whole condition at a time.
6. If voltage curves exist, add ΔQ(V) features (Q(V) at cycle N minus an early cycle: variance, minimum, mean) and dQ/dV / DVA peak features, then compare with summary-only features.
7. Write `docs/results_<dataset>.md`, add a row to the README results table, and commit.

## Phase 2: across datasets (once two or more datasets have working adapters)
8. **Scale-free features:**
   - capacity as a fraction of initial capacity;
   - relative fade rates and curvature;
   - relative resistance growth;
   - condition descriptors (temperature, C-rates, depth of discharge, chemistry flag).

   **Scale-free targets:**
   - state of health at the knee;
   - cycles to the knee divided by the cycles needed to lose the first 2%;
   - "knee within the next 10% of elapsed life".

   Keep the raw-cycle target for comparison.
9. **Descriptive check first:** compare, per dataset, the distributions of state of health at the knee and of the scale-free fade features. Say so if they differ a lot.
10. **Leave-one-dataset-out.** Compare:
    - analogs on raw features;
    - analogs on scale-free features;
    - analogs on features standardised within each dataset;
    - ridge regression;
    - the held-out dataset's own average (a reference that peeks at its mean);
    - the within-dataset leave-one-cell-out result as an upper bound.
11. **Few-shot:** add k = 0, 1, 3, 5, 10 target cells to the memory, with several random draws, and plot error against k.
12. Repeat for early warning (Test B).
13. Write `docs/results_cross_dataset.md` with an honest bottom line on when transfer works. Put the code in `knee_analogs/cross_dataset.py`, with a runner and leakage tests.

## Phase 3: an LLM with extra information (on my unpublished data)
14. Compute these in code, not by the LLM:
    - **DVA/ICA features** from the slow check-ups, aged vs fresh: peak positions and heights. If feasible, also loss of lithium inventory and loss of active material on each electrode, from a standard degradation-mode fit; explain the method you choose.
    - **Operating-history aggregates:**
      - time at temperature;
      - charge C-rate distribution, and charge current at high state of charge combined with low temperature (lithium-plating exposure);
      - rest time at high state of charge and high temperature (SEI / calendar ageing);
      - depth of discharge;
      - Ah throughput.
15. Extend `llm_trial/` to call the Anthropic API directly. Use the official `anthropic` Python SDK, model `claude-opus-5-5`, structured JSON output, and responses cached on disk. Use the same anonymized, fold-based design as `build_trial.py`. Conditions, each adding one thing:
    - **A** numbers + analogs;
    - **B** + chemistry and protocol physics;
    - **C** + operating-history aggregates;
    - **D** + DVA degradation modes;
    - **E** + a few short paper excerpts that I approve (propose candidates, e.g. on knee pathways and DVA-based degradation modes). Never include results about the test cells.
16. Score each condition against the numeric analogs and regression on the same folds:
    - error overall, on typical cells and on outlier cells;
    - interval calibration;
    - rank correlation.

    Also add **regression or analogs using the same engineered features** (e.g. a plating-exposure index), to check whether simple code captures what the LLM uses.
17. Pre-register one combination rule before looking at results: use the LLM when the query lies outside its analogs' range, otherwise the analogs. Evaluate it once.
18. Write `docs/results_llm_<dataset>.md` and commit, including the prompts, raw responses and scores.
