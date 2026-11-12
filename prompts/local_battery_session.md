I want to test whether a "numeric RAG" (nearest-neighbour analogs) can predict the **knee** of lithium-ion battery capacity-fade curves, using my own datasets in `C:\Users\alexa\Datasets Batteries`.

## Starting point (already built and tested)
Repo `https://github.com/alehrx/YFinance`, branch `claude/sp500-news-metrics-rag-8lqacw`. Clone it if it isn't here yet, and read `BATTERY_RAG.md` first. The relevant files:
- `battery_rag/core.py`:
  - `load_cells(folder)`: one CSV per cell with `cycle`, `QD` (discharge Ah), and optionally `IR`, `Tavg`, `chargetime`, `charge_policy`. It drops duplicate files by content hash.
  - `knee_point()`: Bacon-Watts-style two-segment fit on the smoothed capacity curve, up to 80% end of life.
  - `early_features()`, `curve_vector()`, `parse_policy()`.
- `battery_rag/evaluate.py`: leave-one-cell-out tests.
  - Test A: predict the knee from the first N cycles. Compares analogs (5 most similar cells) with ridge regression, protocol-only regression and the plain average.
  - Test B: early warning, matching a cell's current state to snapshots of other cells to predict cycles left until the knee.
- `run_battery.py --data <folder>`: runs everything and writes `reports/battery/*.csv` and figures.
- `battery_rag/inspect_datasets.py`: read-only overview of a dataset folder.

Results on the public Severson 2019 batch 2017-06-30 (42 LFP cells, per-cycle summary only):
- From 100 cycles, analogs average 28 cycles of error against 36 for regression and 45 for the average.
- Among typical cells, no method ranks cells (rank correlation about 0). The analogs mostly identify which group a cell belongs to.
- Cells with no similar peers are badly predicted.
- Analogs are best close to the knee (7 cycles of error at cycle 350).

## What to do
1. Run `python -m battery_rag.inspect_datasets "C:\Users\alexa\Datasets Batteries"` and summarise the datasets for me: chemistry if identifiable, number of cells, cycles per cell, what is measured per cycle, and whether voltage curves or check-up tests exist. **Ask me before reading anything outside that folder.** Never open pickle files.
2. Propose **one** dataset to start with. Prefer the one with the most cells cycled past the knee and per-cycle capacity data; voltage curves are a bonus. Wait for my OK.
3. Write an adapter that converts it to the per-cell table `load_cells` expects. Keep my raw data untouched and write converted files under `data/cache/<dataset>/`, which git ignores. If capacity is only measured in periodic check-ups, adapt the smoothing and knee detection to sparse points instead of forcing per-cycle logic.
4. Plot every cell's capacity curve with its detected knee and 80% end of life, and look at the plots before going further. Report cells where the knee detection is wrong or ambiguous, and fix the detection or exclude those cells with a stated reason. If nominal capacity is not 1.1 Ah, set it correctly (`NOMINAL_AH` in core.py).
5. Run Test A and Test B with leave-one-cell-out. Rules:
   - a cell may never be matched to its own cycles;
   - check for duplicated or near-identical cells (same file under two names) before testing;
   - if the dataset has several temperatures or protocols, also test leaving out a whole condition, to see whether it transfers;
   - report error in cycles and %, the bootstrap 95% CI of the analogs-vs-regression difference, how often the matched cells' range contains the actual knee, and the **rank correlation within the main group**, not only overall error.
6. If voltage curves exist, add Severson-style ΔQ(V) features (difference between the discharge Q(V) curve at cycle N and an early cycle: variance, minimum, mean) and dQ/dV peak position and height. Re-run and compare with the summary-only features.
7. Write the results into `BATTERY_RAG.md` as a new section with tables and figures. Be honest about what does not work; don't tune settings on the test results. Commit to the same branch, but do not commit raw or converted data.

Keep me posted after each step. Stop and ask if a dataset's format is unclear, rather than guessing.
