"""LLM learner: Claude predicts P(up) over the next 1 / 7 / 20 trading days from
(a) today's metrics + news-theme profile and (b) retrieved historical analogs with their
known outcomes (in-context learning). Training happens in two ways:

1. Retrieval memory: the analogs shown are labelled past days (only ones whose outcome
   was already known on the query date).
2. Playbook: on the validation period Claude predicts, sees its own errors, and writes
   a short rulebook ("when the analogs say X but VIX is spiking, ..."). The playbook is
   frozen and given to every held-out prediction.

Leakage: prompts carry no dates, no index level and no raw headlines (unless
``include_headlines=True``) because the model knows market history up to its training
cutoff. Distinctive situations (e.g. a pandemic-headline surge) can still be recognized,
so a held-out period *after* the model's training cutoff is the only fully clean test.

Requires ``pip install anthropic`` and credentials (ANTHROPIC_API_KEY or ``ant auth login``).
"""

from __future__ import annotations

import hashlib
import json
import threading
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import numpy as np
import pandas as pd

from .features import THEMES

MODEL = "claude-opus-5-5"
HORIZONS = (1, 7, 20)

SYSTEM = """You are a quantitative market analyst forecasting the direction of a broad US equity index.

For each query you get:
- the index's current technical / valuation metrics and a profile of news themes
  (share of the week's headlines about each theme, and the change vs the prior 3 months);
- historical ANALOG days that looked similar, each with the index return that followed
  over the next 1, 7 and 20 trading days;
- the unconditional base rates of an up move for each horizon;
- possibly a PLAYBOOK of lessons learned from your own past forecast errors.

Dates and index levels are deliberately hidden. Do not try to identify the period; reason
only from the data given. Analog outcomes are noisy: weigh how close and how consistent
they are, compare with the base rate, and say where today differs from them. Return
calibrated probabilities: most of the time they should be within roughly 0.35-0.75, and
far from the base rate only with strong, consistent evidence."""

SCHEMA = {
    "type": "object",
    "properties": {
        "p_up_1d": {"type": "number"},
        "p_up_7d": {"type": "number"},
        "p_up_20d": {"type": "number"},
        "rationale": {"type": "string"},
    },
    "required": ["p_up_1d", "p_up_7d", "p_up_20d", "rationale"],
    "additionalProperties": False,
}

PLAYBOOK_SYSTEM = """You are reviewing your own track record forecasting an equity index from
metrics and historical analogs. Distil concise, testable lessons that would have improved
calibration and accuracy on these cases. Prefer rules that reference the metrics given
(e.g. VIX, drawdown, distance to moving averages, analog agreement, news-theme surges)."""

_METRIC_FMT = [
    ("ret_1d", "1d", "pct"), ("ret_5d", "5d", "pct"), ("ret_1m", "1m", "pct"), ("ret_3m", "3m", "pct"),
    ("ret_1y", "1y", "pct"), ("px_vs_ma50", "px/MA50", "pct"), ("px_vs_ma200", "px/MA200", "pct"),
    ("ma50_vs_ma200", "MA50/MA200", "pct"), ("pos_52w", "52w-pos", "f2"), ("dd_52w", "DD", "pct"),
    ("vol_1m", "vol", "pct"), ("rsi14", "RSI", "f0"), ("pe_ttm", "P/E", "f1"), ("cape", "CAPE", "f1"),
    ("rate_10y", "10y", "f2"), ("vix", "VIX", "f1"),
]


def fmt_metrics(row: pd.Series) -> str:
    out = []
    for c, name, kind in _METRIC_FMT:
        v = row.get(c, np.nan)
        if pd.isna(v):
            continue
        out.append(f"{name} {v * 100:+.1f}%" if kind == "pct" else
                   f"{name} {v:.{kind[1]}f}")
    return ", ".join(out)


def fmt_themes(row: pd.Series, min_share=0.02, min_nov=0.02) -> str:
    parts = []
    for t in THEMES:
        sh, nv = row.get(f"theme_{t}", np.nan), row.get(f"theme_{t}_nov", np.nan)
        if pd.isna(sh):
            continue
        if sh >= min_share or (pd.notna(nv) and abs(nv) >= min_nov):
            parts.append(f"{t} {sh * 100:.0f}%" + ("" if pd.isna(nv) else f" ({nv * 100:+.0f}pp)"))
    if not parts and pd.notna(row.get("theme_war", np.nan)):
        return "no notable theme"
    return "; ".join(parts) if parts else "n/a"


def build_prompt(query: pd.Series, analogs: pd.DataFrame, base_rates: dict, playbook: str = "",
                 headlines: str = "") -> str:
    lines = []
    if playbook:
        lines += ["PLAYBOOK (lessons from your validation-period errors):", playbook.strip(), ""]
    lines += ["BASE RATES of an up move: " + ", ".join(f"{h}d {base_rates[h]:.0%}" for h in HORIZONS), ""]
    lines += [f"QUERY: {fmt_metrics(query)}", f"QUERY news themes: {fmt_themes(query)}"]
    if headlines:
        lines += [f"QUERY headlines: {headlines[:1500]}"]
    lines += ["", f"ANALOGS ({len(analogs)}, most similar first; outcome = next 1d / 7d / 20d):"]
    for j, (_, r) in enumerate(analogs.iterrows(), 1):
        out = " / ".join(f"{r[f'fwd_ret_{h}'] * 100:+.1f}%" for h in HORIZONS)
        lines.append(f"A{j}: outcome {out} | {fmt_metrics(r)} | themes: {fmt_themes(r)}")
    up = {h: (analogs[f"fwd_ret_{h}"] > 0).mean() for h in HORIZONS}
    lines += ["", "Analog up-share: " + ", ".join(f"{h}d {up[h]:.0%}" for h in HORIZONS),
              "", "Give P(up) for the next 1, 7 and 20 trading days."]
    return "\n".join(lines)


class ClaudePredictor:
    """Calls Claude with structured output; every response is cached on disk (resumable, and
    re-running an evaluation costs nothing). ``mock=True`` swaps in a deterministic stand-in
    that returns the analog vote, to test the pipeline without an API key."""

    def __init__(self, model: str = MODEL, effort: str = "low", cache_path: str | Path | None = None,
                 max_workers: int = 8, mock: bool = False):
        self.model, self.effort, self.mock, self.max_workers = model, effort, mock, max_workers
        self.cache_path = Path(cache_path) if cache_path else None
        self.cache: dict[str, dict] = {}
        self._lock = threading.Lock()
        self.usage = {"input_tokens": 0, "output_tokens": 0, "calls": 0}
        if self.cache_path and self.cache_path.exists():
            for line in self.cache_path.read_text().splitlines():
                rec = json.loads(line)
                self.cache[rec["key"]] = rec["out"]
        if not mock:
            import anthropic

            self.client = anthropic.Anthropic()

    def _key(self, system: str, prompt: str) -> str:
        tag = "mock" if self.mock else f"{self.model}|{self.effort}"
        return hashlib.sha256(f"{tag}\n{system}\n{prompt}".encode()).hexdigest()

    def _call(self, system: str, prompt: str, schema: dict | None, max_tokens: int, effort: str) -> dict:
        resp = self.client.beta.messages.create(
            model=self.model,
            max_tokens=max_tokens,
            system=[{"type": "text", "text": system, "cache_control": {"type": "ephemeral"}}],
            messages=[{"role": "user", "content": prompt}],
            output_config={"effort": effort, **({"format": {"type": "json_schema", "schema": schema}}
                                                 if schema else {})},
            # If the model declines, the API re-runs the request on a fallback model.
            betas=["server-side-fallback-2026-07-01"],
            fallbacks="default",
        )
        with self._lock:
            self.usage["input_tokens"] += resp.usage.input_tokens
            self.usage["output_tokens"] += resp.usage.output_tokens
            self.usage["calls"] += 1
        if resp.stop_reason == "refusal":
            raise RuntimeError(f"declined: {getattr(resp.stop_details, 'category', None)}")
        text = "".join(b.text for b in resp.content if b.type == "text")
        return json.loads(text) if schema else {"text": text}

    @staticmethod
    def _mock(prompt: str) -> dict:
        line = next(l for l in prompt.splitlines() if l.startswith("Analog up-share"))
        vals = [float(x.split()[1].rstrip("%")) / 100 for x in line.split(":")[1].split(",")]
        base = next(l for l in prompt.splitlines() if l.startswith("BASE RATES"))
        bvals = [float(x.split()[1].rstrip("%")) / 100 for x in base.split(":")[1].split(",")]
        p = [0.5 * v + 0.5 * b for v, b in zip(vals, bvals)]
        return {"p_up_1d": p[0], "p_up_7d": p[1], "p_up_20d": p[2], "rationale": "mock: analog vote"}

    def ask(self, prompt: str, system: str = SYSTEM, schema: dict | None = SCHEMA,
            max_tokens: int = 4000, effort: str | None = None) -> dict:
        key = self._key(system, prompt)
        if key in self.cache:
            return self.cache[key]
        if self.mock:
            out = self._mock(prompt) if schema is SCHEMA else {"text": "(mock playbook)"}
        else:
            out = self._call(system, prompt, schema, max_tokens, effort or self.effort)
        with self._lock:
            self.cache[key] = out
            if self.cache_path:
                self.cache_path.parent.mkdir(parents=True, exist_ok=True)
                with self.cache_path.open("a") as f:
                    f.write(json.dumps({"key": key, "out": out}) + "\n")
        return out

    def predict_many(self, prompts: dict) -> dict:
        """prompts: {key -> prompt}. Returns {key -> response dict}; failures are skipped."""
        results = {}

        def work(item):
            k, p = item
            try:
                return k, self.ask(p)
            except Exception as e:  # keep going; report at the end
                return k, {"error": f"{type(e).__name__}: {e}"}

        with ThreadPoolExecutor(self.max_workers) as ex:
            for n, (k, out) in enumerate(ex.map(work, prompts.items()), 1):
                results[k] = out
                if n % 25 == 0:
                    print(f"    {n}/{len(prompts)} LLM calls done")
        errs = [v["error"] for v in results.values() if "error" in v]
        if errs:
            print(f"    {len(errs)} LLM calls failed, e.g. {errs[0]}")
        return {k: v for k, v in results.items() if "error" not in v}

    def learn_playbook(self, records: pd.DataFrame, max_cases: int = 120) -> str:
        """records: one row per validation query with columns
        metrics_text, themes_text, analog_up_{h}, llm_p_{h}, actual_{h}."""
        r = records.sample(min(len(records), max_cases), random_state=0).sort_index()
        lines = []
        for j, (_, x) in enumerate(r.iterrows(), 1):
            res = " ".join(
                f"{h}d: analogs {x[f'analog_up_{h}']:.0%} you {x[f'llm_p_{h}']:.0%} actual {x[f'actual_{h}'] * 100:+.1f}%"
                for h in HORIZONS)
            lines.append(f"C{j}: {x['metrics_text']} | themes: {x['themes_text']} || {res}")
        stats = []
        for h in HORIZONS:
            up = records[f"actual_{h}"] > 0
            stats.append(f"{h}d: base {up.mean():.0%}, your accuracy {((records[f'llm_p_{h}'] >= .5) == up).mean():.0%}, "
                         f"analog-vote accuracy {((records[f'analog_up_{h}'] >= .5) == up).mean():.0%}, "
                         f"always-up accuracy {up.mean():.0%}")
        prompt = ("Validation-period summary:\n" + "\n".join(stats) +
                  "\n\nCases (your forecast vs what happened):\n" + "\n".join(lines) +
                  "\n\nWrite a PLAYBOOK of at most 12 numbered rules (under 350 words) for future "
                  "forecasts: when to trust the analog vote, when to lean toward the base rate, "
                  "and which metric/theme patterns preceded errors. Output only the rules.")
        out = self.ask(prompt, system=PLAYBOOK_SYSTEM, schema=None, max_tokens=16000, effort="high")
        return out["text"].strip()
