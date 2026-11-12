"""Build blinded LLM prompts (conditions A and B) and numeric baselines on identical folds.

Knee prediction from the first 100 cycles, Severson batch 2017-06-30 (42 cells).
3 folds of 14 query cells; each fold's memory = the other 28 cells only (so no query cell's knee
can appear in another query's analog list within the same prompt).
"""
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.linear_model import RidgeCV

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
from knee_analogs.core import knee_point, load_cells  # noqa: E402
from knee_analogs.evaluate import _scale, cell_table  # noqa: E402

N, K, FOLDS, SEED = 100, 8, 3, 7
OUT = Path(__file__).resolve().parent / "work"
OUT.mkdir(exist_ok=True)

cells = load_cells(str(ROOT / "data" / "severson_b2"))
T, _ = cell_table(cells, N)
feats = [c for c in T.columns if c not in ("knee", "eol")]
rng = np.random.default_rng(SEED)
# stratified folds: sort by knee, deal cells round-robin in random order within each block of 3
order = T["knee"].sort_values().index.to_list()
fold_of = {}
for b in range(0, len(order), FOLDS):
    blk = order[b:b + FOLDS]
    for f, c in zip(rng.permutation(FOLDS), blk):
        fold_of[c] = int(f)
T["fold"] = [fold_of[c] for c in T.index]


def curve_points(d):
    x, y = d["cycle"].to_numpy(float), d["QDs"].to_numpy(float)
    q0 = np.median(y[(x >= 2) & (x <= 10)])
    return [float(round((np.interp(c, x, y) - q0) * 1000, 1)) for c in range(10, N + 1, 10)]


def row_text(r, curve, physical):
    if physical:
        hi_soc = r["C2"] if r["Q1"] < 80 else r["C1"]
        return (f"charge {r['C1']:.2f}C from 0 to {r['Q1']:.0f}% SOC then {r['C2']:.2f}C to 80% "
                f"(current in the high-SOC window: {hi_soc:.2f}C; time-averaged 0-80%: {r['C_avg']:.2f}C); "
                f"capacity at start {r['q_start']:.4f} Ah; capacity change since start (mAh) at cycles 10..100: {curve}; "
                f"fade rate cycles 1-100 {r['slope_all']:.3f} mAh/cycle, cycles 75-100 {r['slope_late']:.3f} mAh/cycle; "
                f"curvature {r['curv']:.3f}; internal resistance {r['IR_start']*1000:.2f} mOhm, change {r['IR_delta']*1000:+.3f} mOhm; "
                f"average cell temperature {r['Tavg_start']:.1f} C, change {r['Tavg_delta']:+.2f} C; "
                f"charge time {r['chargetime_start']:.2f} min")
    return (f"p1 {r['C1']:.2f}, p2 {r['Q1']:.0f}, p3 {r['C2']:.2f}, p4 {r['C_avg']:.2f}; "
            f"x_start {r['q_start']:.4f}; trajectory (relative units x1000) at steps 10..100: {curve}; "
            f"s_all {r['slope_all']:.3f}, s_late {r['slope_late']:.3f}, curv {r['curv']:.3f}; "
            f"r0 {r['IR_start']*1000:.2f}, dr {r['IR_delta']*1000:+.3f}; t0 {r['Tavg_start']:.1f}, dt {r['Tavg_delta']:+.2f}; "
            f"u0 {r['chargetime_start']:.2f}")


HEADER = {
    "A": """You are given anonymized numerical records of devices. For each QUERY device you see measurements from its first
100 steps; the task is to predict the step at which its degradation curve shows a "knee" (the point where the slow,
roughly linear decline turns into a much faster decline). For each query you also get the 8 most similar devices from a
reference set (by a numeric distance) with their KNOWN knee step. Fields: p1-p4 are operating parameters; x_start is the
initial level; trajectory is the change of the level since the start, sampled every 10 steps; s_all and s_late are decline
rates over steps 1-100 and 75-100; curv is curvature; r0/dr, t0/dt, u0 are auxiliary sensor readings (start and change).
Reason only from these numbers.""",
    "B": """You are a lithium-ion battery degradation expert. The cells are commercial LFP/graphite 18650 cells (1.1 Ah nominal)
cycled in a 30 C chamber with two-step constant-current fast charging (C1 from 0 to Q1% state of charge, then C2 up to 80%
SOC, then a 1C CC-CV to full), and discharged at 4C. For each QUERY cell you see its first 100 cycles; predict the cycle at
which its capacity-fade curve shows the KNEE (the transition from slow, roughly linear fade to rapid fade). For each query you
also get the 8 most similar cells from a reference set (by a numeric distance on these features) with their KNOWN knee cycle.
Use your knowledge of degradation mechanisms (for example lithium plating when charging at high current at high SOC,
heating, SEI growth, loss of active material, how early fade rate and curvature relate to later knees) together with the
analog cells. Do not try to recall published results for specific cells; reason from the data and mechanisms.""",
}
FOOTER = """
Rules: do NOT use any tools (no file reads, no searches, no code). Answer only from this text.
Return ONLY a JSON array, one object per query, in the same order:
[{"id": "Q..", "knee": <predicted knee cycle/step, number>, "lo": <10th percentile>, "hi": <90th percentile>,
  "why": "<one short sentence>"}, ...]
After the JSON array, add one line: RECOGNIZED: <yes/no> - whether you believe you recognize this exact dataset from
the literature, and if yes which one."""

truth, prompts, base_rows = {}, {}, []
ids = {}
for f in range(FOLDS):
    q_cells = T.index[T["fold"] == f].tolist()
    kb = T[T["fold"] != f]
    Xkb = _scale(kb[feats], kb[feats])
    ridge = RidgeCV(alphas=np.logspace(-2, 3, 30)).fit(Xkb, np.log(kb["knee"]))
    order_q = rng.permutation(q_cells)
    blocks = {"A": [], "B": []}
    for j, c in enumerate(order_q, 1):
        qid = f"Q{f}{j:02d}"
        ids[qid] = c
        truth[qid] = float(T.loc[c, "knee"])
        xq = _scale(kb[feats], T.loc[[c], feats])[0]
        dist = np.sqrt(((Xkb - xq) ** 2).mean(1))
        nn = np.argsort(dist)[:K]
        w = 1 / (dist[nn] + 1e-6); w /= w.sum()
        base_rows.append({"id": qid, "cell": c, "fold": f, "actual": truth[qid],
                          "analogs_k8": float(np.exp(w @ np.log(kb["knee"].iloc[nn]))),
                          "analogs_k8_lo": float(kb["knee"].iloc[nn].min()), "analogs_k8_hi": float(kb["knee"].iloc[nn].max()),
                          "regression": float(np.exp(ridge.predict(xq[None])[0])),
                          "mean": float(np.exp(np.log(kb["knee"]).mean()))})
        for cond in ("A", "B"):
            phys = cond == "B"
            lines = [f"QUERY {qid}: " + row_text(T.loc[c], curve_points(cells[c]), phys)]
            for a, i in enumerate(nn, 1):
                ac = kb.index[i]
                lines.append(f"  analog {a}: knee at {'cycle' if phys else 'step'} {kb.loc[ac, 'knee']:.0f} | "
                             + row_text(kb.loc[ac], curve_points(cells[ac]), phys))
            blocks[cond].append("\n".join(lines))
    for cond in ("A", "B"):
        prompts[f"{cond}{f}"] = HEADER[cond] + "\n\n" + "\n\n".join(blocks[cond]) + "\n" + FOOTER

for k, p in prompts.items():
    (OUT / f"prompt_{k}.txt").write_text(p)
pd.DataFrame(base_rows).to_csv(OUT / "baselines.csv", index=False)
json.dump({"truth": truth, "ids": ids}, open(OUT / "truth.json", "w"), indent=1)
print({k: len(p) for k, p in prompts.items()})
print("folds:", T.groupby("fold")["knee"].describe()[["count", "min", "50%", "max"]].to_string())
