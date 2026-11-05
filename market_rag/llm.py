"""Optional generation step: let Claude read the retrieved analogs and make the call.

Requires ``pip install anthropic`` and credentials (ANTHROPIC_API_KEY or ``ant auth login``).
The numeric RAG prediction works without it.
"""

from __future__ import annotations

import json

import pandas as pd

from .features import describe_row
from .rag import Prediction

MODEL = "claude-opus-5-5"

SYSTEM = """You are a cautious quantitative market analyst. You are given today's snapshot of an
equity index (technical and valuation metrics plus recent headlines) and a set of historical
analog days retrieved because they looked similar, each with what the index did over the
following horizon. Base-rate: the index rises over this horizon in roughly the stated share
of all historical windows. Weigh the analogs, note where today differs from them, and
give a calibrated probability. Do not be more confident than the evidence supports."""

SCHEMA = {
    "type": "object",
    "properties": {
        "p_up": {"type": "number", "description": "probability the index is higher after the horizon"},
        "expected_return_pct": {"type": "number"},
        "stance": {"type": "string", "enum": ["bullish", "neutral", "bearish"]},
        "key_drivers": {"type": "array", "items": {"type": "string"}},
        "rationale": {"type": "string"},
    },
    "required": ["p_up", "expected_return_pct", "stance", "key_drivers", "rationale"],
    "additionalProperties": False,
}


def build_prompt(query_row: pd.Series, query_news: str, pred: Prediction, horizon: int,
                 base_rate: float, max_analogs: int = 12) -> str:
    nb = pred.neighbors.sort_values("weight", ascending=False).head(max_analogs)
    lines = [
        f"Horizon: {horizon} trading days. Historical base rate of a gain: {base_rate:.0%}.",
        f"Retrieval-only estimate: P(up)={pred.p_up:.0%}, weighted mean return {pred.exp_ret:+.2%}.",
        "",
        f"TODAY ({pred.date.date()}): {describe_row(query_row)}",
        f"Headlines: {query_news[:1500] or '(none)'}",
        "",
        "ANALOGS (most similar first):",
    ]
    for d, r in nb.iterrows():
        lines.append(f"- {d.date()} | weight {r['weight']:.2f} | outcome {r[f'fwd_ret_{horizon}']:+.2%} | "
                     f"{describe_row(r)}" + (f" | news: {r['headlines'][:200]}" if r.get("headlines") else ""))
    return "\n".join(lines)


def llm_call(prompt: str, model: str = MODEL) -> dict:
    import anthropic

    client = anthropic.Anthropic()
    resp = client.beta.messages.create(
        model=model,
        max_tokens=8000,
        system=SYSTEM,
        messages=[{"role": "user", "content": prompt}],
        output_config={"effort": "medium", "format": {"type": "json_schema", "schema": SCHEMA}},
        # Server-side fallback: if the model declines, the API retries on a fallback model.
        betas=["server-side-fallback-2026-07-01"],
        fallbacks="default",
    )
    if resp.stop_reason == "refusal":
        raise RuntimeError(f"model declined: {getattr(resp.stop_details, 'category', None)}")
    text = next((b.text for b in resp.content if b.type == "text"), "{}")
    return json.loads(text)
