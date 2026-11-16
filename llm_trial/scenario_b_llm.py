"""Blinded LLM forecasts for Scenario B (knee_analogs/scenario_b.py): the whole ageing curve of a cell
type at use conditions that were never tested, from accelerated tests of the same type stopped at 90 % SOH.
Fixed before any API call:

* One prompt per cell type (at most 15 target conditions each), effort ``medium``, structured JSON, the same
  settings and refusal handling as llm_trial/zero_shot_llm.py; responses cached on disk.
* Target conditions are labelled U1, U2, ...; no dataset, paper or cell name appears. Only accelerated cells
  contribute numbers (asserted); a use cell's data never enters a prompt.
* Answer: for each target condition, the EFC at which SOH first reaches 97.5, 95, ..., 70 %, as 10 / 50 / 90 %
  quantiles, plus one sentence of reasoning.

Conditions:
* ``data``: chemistry and format, the accelerated tests (per condition: number of cells and the median EFC
  at which SOH reached 99, 98, ..., 90 %, with the range of t90), the mechanism note and the target conditions.
* ``know``: the same without the accelerated tests: chemistry, format, mechanism note and target conditions.

    python llm_trial/scenario_b_llm.py --model opus --dry-run
    python llm_trial/scenario_b_llm.py --model opus --conditions data know
    python llm_trial/scenario_b_llm.py --model opus --repeat 1      # the same prompts asked again
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "llm_trial"))
from knee_analogs import scenario_b as b  # noqa: E402
from knee_analogs import zero_shot as z  # noqa: E402
from zero_shot_llm import MECHANISMS, MODELS, SCHEMA, conditions_text  # noqa: E402

EFFORT = "medium"
LEVELS = [97.5, 95.0, 92.5, 90.0, 87.5, 85.0, 82.5, 80.0, 77.5, 75.0, 72.5, 70.0]
SEEN_LEVELS = [0.99, 0.98, 0.97, 0.96, 0.95, 0.94, 0.93, 0.92, 0.91, 0.90]
PER_PROMPT = 15
WORK = ROOT / "data" / "library" / "llm" / "scenario_b"

SYSTEM = """You forecast the capacity fade of lithium-ion cells at use conditions that have never been \
tested. State of health (SOH) = capacity / initial capacity; time is in equivalent full cycles (EFC).

For each target condition, give the EFC at which SOH first reaches 97.5, 95, 92.5, 90, 87.5, 85, 82.5, 80, \
77.5, 75, 72.5 and 70 %, each as a 10th, 50th and 90th percentile (p10 <= p50 <= p90; EFC increases as \
SOH falls). Give your best extrapolation even for levels far in the future. Real cells can fade steadily, \
slow down, or accelerate sharply (a "knee"). Give one short sentence of reasoning per condition. Work from \
the information given; do not assume the cells come from any particular published dataset."""


def accelerated_block(M, curves, acc, F):
    """Per accelerated condition: n cells, median EFC at SOH 99..90 %, t90 range. Only accelerated cells."""
    assert (M.loc[acc, "role"] == "accelerated").all()
    rows = []
    for c in acc:
        t, s, t90 = b.seen_curve(curves, c)
        rows.append({"key": b.cond_key(M.loc[c]), "c": c, "t90": t90,
                     **{lv: z.t_at(t, s, level=lv) for lv in SEEN_LEVELS}})
    R = pd.DataFrame(rows)
    lines = ["Accelerated tests of this cell type (each stopped when SOH first reached 90 %). Per condition: "
             "number of cells; median EFC at which SOH reached " + ", ".join(f"{lv * 100:.0f}" for lv in SEEN_LEVELS)
             + " %; range of t90 over the cells:"]
    for key, D in R.groupby("key", sort=True):
        m = M.loc[D["c"].iloc[0]]
        med = [np.nanmedian(D[lv]) for lv in SEEN_LEVELS]
        lines.append(f"- {conditions_text(m)}: {len(D)} cells; " + " ".join(f"{v:.0f}" for v in med)
                     + f"; t90 {D['t90'].min():.0f}-{D['t90'].max():.0f} EFC")
    return "\n".join(lines)


def build_prompts(cond):
    meta, curves = z.load_library()
    M = b.assign_types(meta)
    F = b.features(M)
    prompts = []
    for typ, D in M.groupby("type"):
        acc = [c for c in D.index[D["role"] == "accelerated"] if np.isfinite(z.t_at(*curves[c]))]
        use = D[D["role"] == "use"].copy()
        use["key"] = [b.cond_key(r) for _, r in use.iterrows()]
        targets = use.groupby("key").head(1)  # one representative row per target condition
        r0 = D.iloc[0]
        head = f"Cell type: {r0['chemistry']}" + (f", nominal {r0['nominal_ah']:g} Ah" if np.isfinite(
            r0.get("nominal_ah", np.nan)) else "") + "."
        acc_txt = accelerated_block(M, curves, acc, F) if cond == "data" else \
            "No ageing data of this cell type is available."
        keys = list(targets["key"])
        for i in range(0, len(keys), PER_PROMPT):
            chunk = keys[i:i + PER_PROMPT]
            labels = {k: f"U{j + 1}" for j, k in enumerate(chunk)}
            tgt = "\n".join(f"- {labels[k]}: {conditions_text(targets[targets['key'] == k].iloc[0])}" for k in chunk)
            text = "\n\n".join([head, acc_txt, MECHANISMS,
                                "Target use conditions (never tested); forecast each:\n" + tgt])
            for c in use.index:  # no use cell's numbers: its id never appears and no curve of it is read
                assert c.split("__")[1] not in text
            prompts.append({"type": typ, "labels": labels, "text": text,
                            "key": hashlib.sha256((cond + EFFORT + SYSTEM + text).encode()).hexdigest()[:16]})
    return prompts


def call(client, model, text):
    resp = client.beta.messages.create(
        model=model, max_tokens=16000, system=SYSTEM, messages=[{"role": "user", "content": text}],
        output_config={"effort": EFFORT, "format": {"type": "json_schema", "schema": SCHEMA}},
        betas=["server-side-fallback-2026-07-01"], fallbacks="default",
    )
    if resp.stop_reason == "refusal":
        raise RuntimeError(f"refusal: {resp.stop_details}")
    out = next(x.text for x in resp.content if x.type == "text")
    return {"json": json.loads(out), "model": resp.model, "stop_reason": resp.stop_reason,
            "usage": {"input": resp.usage.input_tokens, "output": resp.usage.output_tokens}}


def complete(rec, labels) -> bool:
    got = {e["cell"].strip(): e for e in rec["json"]["forecasts"]}
    return rec["stop_reason"] != "max_tokens" and set(got) == set(labels.values()) and all(
        all(len(got[l][q]) == len(LEVELS) for q in ("p10", "p50", "p90")) for l in labels.values())


def run(name, model, price, prompts, workers=4):
    import anthropic

    d = WORK / "raw" / name
    d.mkdir(parents=True, exist_ok=True)
    todo = [p for p in prompts if not (d / f"{p['key']}.json").exists()]
    client = anthropic.Anthropic(default_headers={"Accept-Encoding": "gzip, deflate"})  # see zero_shot_llm.py
    spent = [0, 0]

    def one(p):
        r = call(client, model, p["text"])
        rec = {"prompt": p["text"], "system": SYSTEM, "labels": p["labels"], "type": p["type"], **r}
        if not complete(rec, p["labels"]):
            (d / "incomplete").mkdir(exist_ok=True)
            (d / "incomplete" / f"{p['key']}_{len(list((d / 'incomplete').glob(p['key'] + '_*')))}.json").write_text(
                json.dumps(rec, indent=1), encoding="utf-8")
            raise RuntimeError(f"incomplete answer; {r['usage']} tokens billed")
        (d / f"{p['key']}.json").write_text(json.dumps(rec, indent=1), encoding="utf-8")
        return r["usage"]

    with ThreadPoolExecutor(workers) as ex:
        futs = {ex.submit(one, p): p for p in todo}
        for i, f in enumerate(as_completed(futs), 1):
            try:
                u = f.result()
                spent[0] += u["input"]; spent[1] += u["output"]
            except Exception as e:  # failed prompts are retried on the next run
                print(f"  failed {futs[f]['type']}: {type(e).__name__}: {str(e)[:200]}", flush=True)
            print(f"  {name}: {i}/{len(todo)} calls, ${spent[0] * price[0] / 1e6 + spent[1] * price[1] / 1e6:.2f}",
                  flush=True)


def collect(name, prompts):
    rows, reasons = [], []
    for p in prompts:
        f = WORK / "raw" / name / f"{p['key']}.json"
        if not f.exists():
            continue
        rec = json.loads(f.read_text(encoding="utf-8"))
        back = {v: k for k, v in p["labels"].items()}
        for e in rec["json"]["forecasts"]:
            key = back[e["cell"].strip()]
            q = np.sort(np.vstack([e["p10"], e["p50"], e["p90"]]).astype(float), axis=0)
            for j, lv in enumerate(LEVELS):
                rows.append({"type": p["type"], "key": key, "soh": lv / 100, "q10": q[0, j], "q50": q[1, j],
                             "q90": q[2, j]})
            reasons.append({"type": p["type"], "key": key, "reason": e["reason"]})
    return pd.DataFrame(rows), pd.DataFrame(reasons)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--model", choices=list(MODELS), default="opus")
    ap.add_argument("--conditions", nargs="+", default=["data", "know"])
    ap.add_argument("--dry-run", action="store_true")
    ap.add_argument("--repeat", type=int, default=0, help="repeat number (1, 2, ...): same prompts, asked again")
    args = ap.parse_args()
    model, p_in, p_out, _ = MODELS[args.model]
    for cond in args.conditions:
        name = f"{args.model}_{cond}" + (f"_r{args.repeat}" if args.repeat else "")
        prompts = build_prompts(cond)
        if args.repeat:  # a new cache key, so the same prompt is asked again
            for p in prompts:
                p["key"] = hashlib.sha256((p["key"] + f"repeat{args.repeat}").encode()).hexdigest()[:16]
        n_tok = sum(len(SYSTEM + p["text"]) for p in prompts) / 3.5
        print(f"{name}: {len(prompts)} prompts, {sum(len(p['labels']) for p in prompts)} target conditions, "
              f"~{n_tok / 1e3:.0f}k input tokens (~${n_tok * p_in / 1e6:.2f} input)", flush=True)
        if args.dry_run:
            (WORK / "dry_run").mkdir(parents=True, exist_ok=True)
            (WORK / "dry_run" / f"example_{cond}.txt").write_text(SYSTEM + "\n\n---\n\n" + prompts[0]["text"],
                                                                 encoding="utf-8")
            continue
        run(name, model, (p_in, p_out), prompts)
        fc, rs = collect(name, prompts)
        if len(fc):
            out = WORK / "repeats" if args.repeat else WORK  # repeats stay out of the main table
            out.mkdir(parents=True, exist_ok=True)
            fc.to_csv(out / f"{name}.csv", index=False)
            rs.to_csv(WORK / "raw" / f"{name}_reasons.csv", index=False)
            print(f"  {name}: forecasts for {fc[['type', 'key']].drop_duplicates().shape[0]} target conditions")


if __name__ == "__main__":
    main()
