"""Zero-shot Chronos forecasts of SOH curves. Runs in the Python environment that has Chronos
(``anaconda3/envs/phm_env``); reads a context file written by scripts/zero_shot_study.py and writes
the forecasts. No download: models load from the local cache only.

    <phm_env python> scripts/chronos_forecast.py --context ctx.csv --out fc.csv --model amazon/chronos-2

Context: columns id, step, efc, soh (equally spaced EFC per id). Output: id, efc, q10, q50, q90 for the
next ``--steps`` steps at the same EFC spacing.
"""

from __future__ import annotations

import argparse
import os

os.environ.setdefault("HF_HUB_OFFLINE", "1")

import numpy as np
import pandas as pd
import torch
from chronos import BaseChronosPipeline


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--context", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--model", default="amazon/chronos-2")
    ap.add_argument("--steps", type=int, default=128)
    ap.add_argument("--batch", type=int, default=64)
    args = ap.parse_args()
    ctx = pd.read_csv(args.context)
    pipe = BaseChronosPipeline.from_pretrained(args.model, device_map="cpu", local_files_only=True)
    ids = list(dict.fromkeys(ctx["id"]))
    rows = []
    for i in range(0, len(ids), args.batch):
        chunk = ids[i:i + args.batch]
        series = [torch.tensor(ctx.loc[ctx["id"] == c, "soh"].to_numpy(np.float32)) for c in chunk]
        q, _ = pipe.predict_quantiles(series, prediction_length=args.steps, quantile_levels=[0.1, 0.5, 0.9])
        for c, qc in zip(chunk, q):
            qc = np.asarray(qc.squeeze().cpu() if hasattr(qc, "cpu") else qc).reshape(args.steps, 3)
            e = ctx.loc[ctx["id"] == c, "efc"].to_numpy()
            dt = e[1] - e[0]
            efc = e[-1] + dt * np.arange(1, args.steps + 1)
            rows.append(pd.DataFrame({"id": c, "efc": efc, "q10": qc[:, 0], "q50": qc[:, 1], "q90": qc[:, 2]}))
        print(f"{min(i + args.batch, len(ids))}/{len(ids)}", flush=True)
    pd.concat(rows).to_csv(args.out, index=False)


if __name__ == "__main__":
    main()
