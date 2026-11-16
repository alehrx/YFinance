"""Blinded LLM forecasts for Scenario A (knee_analogs/zero_shot.py): each cell is seen only until its
first measurement at or below 90 % SOH; the LLM forecasts the rest of its curve. No ageing data of the
cell's own type is shown. Fixed before any API call:

* Model ``claude-opus-5-5``, adaptive thinking, effort ``medium``, structured JSON output; server-side
  refusal fallback (``fallbacks: "default"``). Responses are cached on disk (data/library/llm/<condition>/),
  so a prompt is never billed twice.
* Prompts hold 10 query cells of the *same* dataset; their analogs come only from *other* datasets, so no
  query's future can appear in any prompt. Datasets and cell ids are never named (Q1.., R1..).
* The query curve: SOH (%) at 0, 10, ..., 100 % of t90 plus the stopping point, with EFC.
* Forecast: SOH (%) at u = t / t90 = 1.25, 1.5, 1.75, 2, 2.5, 3, as 10 / 50 / 90 % quantiles, plus one
  sentence of reasoning.

Conditions (each adds or removes one thing):
* ``A``: numbers + 10 analogs (the shape-only library analogs of knee_analogs/zero_shot.py: their
  pre-90 % shape and their normalised future). No chemistry, no conditions.
* ``B``: A + chemistry and test conditions of the query and of each analog, and a short neutral note on
  degradation mechanisms; instructed to use mechanism knowledge.
* ``C``: B without analogs: the query's own curve, its chemistry and conditions, and mechanism knowledge.

    python llm_trial/zero_shot_llm.py --dry-run                 # build prompts, estimate tokens and cost
    python llm_trial/zero_shot_llm.py --conditions A B C --pilot  # one prompt per condition
    python llm_trial/zero_shot_llm.py --conditions A B C          # everything (cached prompts are free)
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
from knee_analogs import zero_shot as z  # noqa: E402

MODEL = "claude-opus-5-5"
EFFORT = "medium"
PRICE_IN, PRICE_OUT = 4.0, 20.0  # $ per million tokens (Claude Opus 5.5)
U_GRID = [1.25, 1.5, 1.75, 2.0, 2.5, 3.0]
PER_PROMPT = 10
WORK = ROOT / "data" / "library" / "llm"

SYSTEM = """You forecast the capacity fade of lithium-ion cells. Each query cell was aged until its state \
of health (SOH = capacity / initial capacity) first fell to 90 %; that test then stopped. Forecast the rest \
of its SOH curve.

Time is in equivalent full cycles (EFC). t90 is the EFC at which the cell first reached 90 % SOH. Forecast \
SOH (in %) at the normalised times u = t / t90 = 1.25, 1.5, 1.75, 2.0, 2.5 and 3.0, each as a 10th, 50th and \
90th percentile (p10 <= p50 <= p90). Real cells can keep a steady fade, slow down, or accelerate sharply \
(a "knee"); SOH can fall far below 80 % by 3 t90. Give one short sentence of reasoning per cell. Work from \
the numbers given; do not assume the cells come from any particular published dataset."""

MECHANISMS = """Background on degradation mechanisms (general knowledge, not about these cells):
- SEI growth consumes lithium inventory; faster at high temperature and high state of charge, and usually \
decelerating over time (roughly with the square root of time).
- Lithium plating on the negative electrode: promoted by low temperature, high charge current and high \
upper voltage; it consumes lithium quickly and can start or be amplified as the negative electrode loses \
active material, which can produce an accelerating fade (knee).
- Loss of active material (particle cracking, contact loss, silicon expansion in Si-containing anodes, \
cathode degradation at high voltage or wide depth of discharge) can shift the electrode balance and \
precede a knee.
- Higher discharge rates and temperatures change heat generation and can accelerate or redistribute ageing.
Use this knowledge, together with the chemistry and conditions, to judge whether each cell's fade will \
continue, slow down or accelerate after 90 % SOH."""


def fmt(v, nd=1):
    return "n/a" if v is None or not np.isfinite(v) else f"{v:.{nd}f}"


def conditions_text(m: pd.Series) -> str:
    parts = [f"chemistry {m['chemistry']}"]
    if np.isfinite(m.get("nominal_ah", np.nan)):
        parts.append(f"nominal {m['nominal_ah']:g} Ah")
    for k, lab, unit in (("T", "temperature", " °C"), ("chg_C", "charge", "C"), ("dis_C", "discharge", "C")):
        if np.isfinite(m.get(k, np.nan)):
            parts.append(f"{lab} {m[k]:g}{unit}")
    if np.isfinite(m.get("dod", np.nan)):
        parts.append(f"depth of discharge {m['dod'] * 100:.0f} %")
    if np.isfinite(m.get("vmax", np.nan)) and np.isfinite(m.get("vmin", np.nan)):
        parts.append(f"voltage window {m['vmin']:g}-{m['vmax']:g} V")
    return ", ".join(parts)


def query_block(label, q, Q, meta, curves, analogs, cond):
    t, s = curves[q]
    t90, t_stop = Q.loc[q, "t90"], Q.loc[q, "t_stop"]
    pts = [(f * t90, np.interp(f * t90, t, s)) for f in np.linspace(0, 1, 11)]
    if t_stop > t90:
        pts.append((t_stop, float(s[t <= t_stop][-1])))
    lines = [f"### {label}", f"t90 = {t90:.1f} EFC. Seen curve (EFC: SOH %): "
             + "; ".join(f"{e:.0f}: {v * 100:.2f}" for e, v in pts)]
    if cond in ("B", "C"):
        lines.append("Cell: " + conditions_text(meta.loc[q]))
    if cond in ("A", "B"):
        lines.append("Reference cells (other cell types, fully aged). Shape = SOH % at 0.1, 0.2, ..., 0.9 t90; "
                     "future = SOH % at u = 1.25, 1.5, 1.75, 2, 2.5, 3 (n/a = not measured that far):")
        for j, a in enumerate(analogs, 1):
            ta, sa = curves[a]
            t90a = z.t_at(ta, sa)
            shape = [np.interp(f * t90a, ta, sa) * 100 for f in z.FRACS]
            fut = [np.interp(u * t90a, ta, sa, left=np.nan, right=np.nan) * 100 for u in U_GRID]
            line = (f"- R{j}: shape " + " ".join(fmt(v) for v in shape) + " | future "
                    + " ".join(fmt(v) for v in fut))
            if cond == "B":
                line += f" | {conditions_text(meta.loc[a])}"
            lines.append(line)
    return "\n".join(lines)


SCHEMA = {
    "type": "object",
    "properties": {
        "forecasts": {
            "type": "array",
            "items": {
                "type": "object",
                "properties": {
                    "cell": {"type": "string"},
                    "p10": {"type": "array", "items": {"type": "number"}},
                    "p50": {"type": "array", "items": {"type": "number"}},
                    "p90": {"type": "array", "items": {"type": "number"}},
                    "reason": {"type": "string"},
                },
                "required": ["cell", "p10", "p50", "p90", "reason"],
                "additionalProperties": False,
            },
        }
    },
    "required": ["forecasts"],
    "additionalProperties": False,
}


def build_prompts(cond: str):
    meta, curves = z.load_library()
    Q = z.cell_table(meta, curves)
    Q = Q[(Q["n_seen"] >= 4) & (Q["n_future"] >= 1)]
    F = pd.read_csv(ROOT / "data" / "library" / "zero_shot_forecasts.csv", usecols=["id", "lib_analogs"])
    analogs = F.drop_duplicates("id").set_index("id")["lib_analogs"].str.split(",")
    prompts = []
    for ds, D in Q.groupby("dataset"):
        ids = sorted(D.index)
        for k in range(0, len(ids), PER_PROMPT):
            chunk = ids[k:k + PER_PROMPT]
            labels = {q: f"Q{i + 1}" for i, q in enumerate(chunk)}
            for q in chunk:  # leakage guard: no analog from the query's dataset
                assert all(meta.loc[a, "dataset"] != ds for a in analogs[q])
            body = "\n\n".join(query_block(labels[q], q, Q, meta, curves, analogs[q], cond) for q in chunk)
            intro = "" if cond == "A" else MECHANISMS + "\n\n"
            text = (intro + f"Forecast these {len(chunk)} cells. Return one entry per cell, with 'cell' set to "
                    "its label (Q1, Q2, ...) and six values in each of p10, p50, p90 (for u = 1.25, 1.5, 1.75, "
                    "2, 2.5, 3).\n\n" + body)
            key = hashlib.sha256((MODEL + EFFORT + SYSTEM + text).encode()).hexdigest()[:16]
            prompts.append({"key": key, "dataset": ds, "labels": labels, "text": text})
    return prompts, Q


def call(client, text):
    resp = client.beta.messages.create(
        model=MODEL, max_tokens=16000, system=SYSTEM,
        messages=[{"role": "user", "content": text}],
        output_config={"effort": EFFORT, "format": {"type": "json_schema", "schema": SCHEMA}},
        betas=["server-side-fallback-2026-07-01"], fallbacks="default",
    )
    if resp.stop_reason == "refusal":
        raise RuntimeError(f"refusal: {resp.stop_details}")
    out = next(b.text for b in resp.content if b.type == "text")
    return {"json": json.loads(out), "model": resp.model, "stop_reason": resp.stop_reason,
            "usage": {"input": resp.usage.input_tokens, "output": resp.usage.output_tokens}}


def complete(rec) -> bool:
    """An answer counts only if it forecasts every cell of the prompt and was not cut off."""
    got = {e["cell"].strip() for e in rec["json"]["forecasts"]}
    return rec["stop_reason"] != "max_tokens" and got == set(rec["labels"].values())


def run(cond, prompts, pilot=False, workers=4):
    import anthropic

    d = WORK / cond
    d.mkdir(parents=True, exist_ok=True)
    for f in d.glob("*.json"):  # answers cached before the completeness check
        rec = json.loads(f.read_text(encoding="utf-8"))
        if not complete(rec):
            (d / "incomplete").mkdir(exist_ok=True)
            f.replace(d / "incomplete" / f"{f.stem}_0.json")
            print(f"  {cond}: set aside incomplete answer {f.stem}", flush=True)
    todo = [p for p in prompts if not (d / f"{p['key']}.json").exists()]
    if pilot:
        todo = todo[:1]
    # gzip only: the Brotli 1.0.9 in Anaconda base is too old for the SDK's HTTP decoder
    client = anthropic.Anthropic(default_headers={"Accept-Encoding": "gzip, deflate"})
    spent = [0, 0]

    def one(p):
        r = call(client, p["text"])
        rec = {"prompt": p["text"], "system": SYSTEM, "labels": p["labels"], "dataset": p["dataset"], **r}
        if not complete(rec):  # keep the raw answer, but ask again on the next run
            (d / "incomplete").mkdir(exist_ok=True)
            n = len(list((d / "incomplete").glob(f"{p['key']}_*.json")))
            (d / "incomplete" / f"{p['key']}_{n}.json").write_text(json.dumps(rec, indent=1), encoding="utf-8")
            raise RuntimeError(f"incomplete answer ({len(r['json']['forecasts'])} of {len(p['labels'])} cells, "
                               f"stop {r['stop_reason']}); {r['usage']} tokens billed")
        (d / f"{p['key']}.json").write_text(json.dumps(rec, indent=1), encoding="utf-8")
        return r["usage"]

    with ThreadPoolExecutor(workers) as ex:
        futs = {ex.submit(one, p): p for p in todo}
        for i, f in enumerate(as_completed(futs), 1):
            try:
                u = f.result()
                spent[0] += u["input"]; spent[1] += u["output"]
            except Exception as e:  # keep going; failed prompts are retried on the next run
                print(f"  failed {futs[f]['key']}: {type(e).__name__}: {str(e)[:200]}", flush=True)
            cost = spent[0] * PRICE_IN / 1e6 + spent[1] * PRICE_OUT / 1e6
            print(f"  {cond}: {i}/{len(todo)} calls, {spent[0]} in / {spent[1]} out tokens, ${cost:.2f}", flush=True)
    return spent


def collect(cond, prompts, Q):
    """Forecast table in the Chronos format: id, efc, q10, q50, q90 (plus u = 1 at 90 %)."""
    rows, reasons, bad = [], [], 0
    for p in prompts:
        f = WORK / cond / f"{p['key']}.json"
        if not f.exists():
            continue
        rec = json.loads(f.read_text(encoding="utf-8"))
        back = {v: k for k, v in p["labels"].items()}
        for e in rec["json"]["forecasts"]:
            q = back.get(e["cell"].strip())
            arr = [np.asarray(e[k], float) for k in ("p10", "p50", "p90")]
            if q is None or any(len(a) != len(U_GRID) for a in arr):
                bad += 1
                continue
            lo, mid, hi = (np.sort(np.vstack(arr), axis=0) / 100)  # enforce p10 <= p50 <= p90
            t90 = Q.loc[q, "t90"]
            rows.append(pd.DataFrame({"id": q, "efc": t90 * np.r_[1.0, U_GRID], "q10": np.r_[0.9, lo],
                                      "q50": np.r_[0.9, mid], "q90": np.r_[0.9, hi]}))
            reasons.append({"id": q, "condition": cond, "reason": e["reason"]})
    if bad:
        print(f"  {cond}: {bad} malformed entries skipped")
    return (pd.concat(rows) if rows else pd.DataFrame()), pd.DataFrame(reasons)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--conditions", nargs="+", default=["A", "B", "C"])
    ap.add_argument("--dry-run", action="store_true")
    ap.add_argument("--pilot", action="store_true")
    args = ap.parse_args()
    total_in = 0
    for cond in args.conditions:
        prompts, Q = build_prompts(cond)
        n_tok = sum(len(SYSTEM + p["text"]) for p in prompts) / 3.5  # rough chars-per-token
        total_in += n_tok
        print(f"{cond}: {len(prompts)} prompts, {sum(len(p['labels']) for p in prompts)} cells, "
              f"~{n_tok / 1e3:.0f}k input tokens (~${n_tok * PRICE_IN / 1e6:.2f} input)")
        if args.dry_run:
            (WORK / "dry_run").mkdir(parents=True, exist_ok=True)
            (WORK / "dry_run" / f"example_{cond}.txt").write_text(SYSTEM + "\n\n---\n\n" + prompts[0]["text"],
                                                                 encoding="utf-8")
            continue
        run(cond, prompts, pilot=args.pilot)
        fc, rs = collect(cond, prompts, Q)
        if len(fc):
            fc.to_csv(WORK / f"llm_{cond}.csv", index=False)
            rs.to_csv(WORK / f"llm_{cond}_reasons.csv", index=False)
            print(f"  {cond}: forecasts for {fc['id'].nunique()} cells")
    if args.dry_run:
        print(f"total ~{total_in / 1e3:.0f}k input tokens; output (incl. thinking) is the larger cost: "
              f"~3-8k tokens per call x $20/M")


if __name__ == "__main__":
    main()
