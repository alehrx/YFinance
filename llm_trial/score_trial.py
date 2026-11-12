"""Score the blinded LLM knee predictions against numeric baselines on the same folds.

    python llm_trial/score_trial.py <key_dir> <responses_dir>

key_dir holds truth.json and baselines.csv from build_trial.py; responses_dir holds
response_A0.txt ... response_B2.txt (the raw text each predictor returned).
"""
import json
import re
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import spearmanr

key, resp = Path(sys.argv[1]), Path(sys.argv[2])
truth = json.load(open(key / "truth.json"))
B = pd.read_csv(key / "baselines.csv").set_index("id")

recognized = {}
for cond in ("A", "B"):
    rows = []
    for f in range(3):
        p = resp / f"response_{cond}{f}.txt"
        if not p.exists():
            continue
        txt = p.read_text()
        arr = json.loads(re.search(r"\[.*\]", txt, re.S).group(0))
        rows += arr
        m = re.search(r"RECOGNIZED:\s*(.*)", txt)
        recognized[f"{cond}{f}"] = m.group(1).strip() if m else "?"
    if rows:
        d = pd.DataFrame(rows).set_index("id")
        B[f"llm_{cond}"] = d["knee"].astype(float)
        B[f"llm_{cond}_lo"] = d["lo"].astype(float)
        B[f"llm_{cond}_hi"] = d["hi"].astype(float)
        B[f"llm_{cond}_why"] = d["why"]

methods = ["mean", "regression", "analogs_k8"] + [c for c in ("llm_A", "llm_B") if c in B]
rng = np.random.default_rng(0)
idx = rng.integers(0, len(B), (4000, len(B)))
typ = B["actual"] > 300
rows = []
for m in methods:
    e = (B[m] - B["actual"]).abs()
    r = {"method": m, "n": int(B[m].notna().sum()), "MAE": e.mean(), "median_AE": e.median(),
         "MAE_typical": e[typ].mean(), "MAE_short_lived": e[~typ].mean(),
         "rank_corr_typical": spearmanr(B.loc[typ, m], B.loc[typ, "actual"]).statistic}
    lo, hi = (f"{m}_lo", f"{m}_hi")
    if lo in B:
        r["interval_covers_%"] = ((B[lo] <= B["actual"]) & (B["actual"] <= B[hi])).mean() * 100
        r["interval_width"] = (B[hi] - B[lo]).median()
    if m != "analogs_k8":
        d = (e - (B["analogs_k8"] - B["actual"]).abs()).to_numpy()[idx].mean(1)
        r["vs_analogs_MAE_diff_CI"] = f"{d.mean():+.1f} [{np.percentile(d, 2.5):+.1f}, {np.percentile(d, 97.5):+.1f}]"
    rows.append(r)
res = pd.DataFrame(rows).set_index("method")
pd.set_option("display.width", 220)
print(res.round(2).to_string())
print("\nrecognized dataset?", recognized)
short = B[~typ][["cell", "actual"] + methods]
print("\nshort-lived cells:\n" + short.round(0).to_string())
B.to_csv(resp / "scored.csv")
res.to_csv(resp / "summary.csv")
