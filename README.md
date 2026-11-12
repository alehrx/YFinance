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

In short:
- Analogs beat regression and the average, mostly by identifying which group a cell belongs to.
- Within the main group, no method ranks cells yet.
- Cells without similar peers are missed.
- Close to the knee, analogs give the sharpest early warning (AUC 0.98 for "knee within 50 cycles").
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
plus optionally `IR`, `Tavg`, `chargetime` and `charge_policy`. Other formats need a small adapter
that writes that layout to `data/<dataset>/`. `prompts/local_session.md` is a ready-made prompt for
a Claude Code session on your own computer to do this.

## Layout

```
knee_analogs/
  core.py               loading (de-duplicates files), smoothing, knee detection, early-cycle features
  evaluate.py           leave-one-cell-out tests A and B, baselines, summaries
  inspect_datasets.py   read-only overview of a dataset folder
run_study.py            runs everything for one dataset folder, writes reports/<dataset>/
llm_trial/              blinded LLM trial: build prompts, score, plot
scripts/                data download helpers
docs/                   results write-ups per dataset
reports/<dataset>/      tables (.csv) and figures (.png/.pdf)
tests/                  synthetic-cell tests: knee detection, de-duplication, no self-matching
data/                   local data (not committed)
```
