"""Train / validation / held-out-test evaluation for 1-, 7- and 20-day direction.

    train  : labels known before ``val_start``      -> scaling, metric weights, logistic fit
    val    : [val_start, val_end)                   -> tune RAG (alpha, k); LLM writes its playbook
    test   : [val_end, test_end]  (never touched)   -> all reported numbers

During the test the parameters are frozen. The retrieval memory keeps growing with days
whose outcome became known before each query date, as it would in live use.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score

from .llm import HORIZONS, ClaudePredictor, build_prompt, fmt_metrics, fmt_themes
from .rag import MarketRAG, RAGConfig


def _test_index(kb, start, end, h, step=1):
    y = kb[f"fwd_ret_{h}"]
    ok = (kb.index >= pd.Timestamp(start)) & (kb.index <= pd.Timestamp(end)) & y.notna().to_numpy()
    return np.flatnonzero(ok)[::step]


def rag_predict(kb, h, features, val_start, val_end, test_end, use_news=False, step=1):
    cfg = RAGConfig(horizon=h, features=features, use_news=use_news, min_coverage=0.3,
                    k_grid=(10, 20, 40), alpha_grid=(0.0, 0.25, 0.5) if use_news else (0.0,))
    m = MarketRAG(cfg).fit(kb, train_end=pd.Timestamp(val_end), val_start=pd.Timestamp(val_start))
    rows = []
    for i in _test_index(kb, val_end, test_end, h, step):
        p = m.predict_index(i)
        if p is not None:
            rows.append((kb.index[i], p.p_up, p.exp_ret, m.y[i]))
    df = pd.DataFrame(rows, columns=["date", "p_up", "exp_ret", "actual"]).set_index("date")
    return df, m


def logit_predict(kb, h, features, val_end, test_end, step=1):
    y = kb[f"fwd_ret_{h}"]
    tr = (pd.to_datetime(kb[f"label_date_{h}"]) < pd.Timestamp(val_end)) & y.notna()
    feats = [c for c in features if kb.loc[tr, c].notna().sum() >= 200]
    X = kb[feats]
    med, sd = X[tr].median(), X[tr].std().replace(0, 1)
    Z = ((X - med) / sd).fillna(0).clip(-5, 5)
    clf = LogisticRegression(C=0.05, max_iter=3000).fit(Z[tr], (y[tr] > 0).astype(int))
    qi = _test_index(kb, val_end, test_end, h, step)
    p = clf.predict_proba(Z.iloc[qi])[:, 1]
    return pd.DataFrame({"p_up": p, "exp_ret": p - 0.5, "actual": y.iloc[qi].to_numpy()}, index=kb.index[qi])


def evaluate(preds: pd.DataFrame, base_rate: float, n_boot: int = 1000, block: int = 20, seed: int = 0) -> dict:
    """Accuracy, always-up accuracy, AUC with a moving-block bootstrap 95% CI, Brier skill."""
    a = preds["actual"].to_numpy()
    up = (a > 0).astype(int)
    p = preds["p_up"].to_numpy()
    out = dict(n=len(p), up_share=up.mean(), acc=((p >= 0.5) == up).mean(), always_up=up.mean(),
               brier_skill=1 - np.mean((p - up) ** 2) / np.mean((base_rate - up) ** 2))
    if 0 < up.sum() < len(up):
        out["auc"] = roc_auc_score(up, p)
        rng = np.random.default_rng(seed)
        nb = max(1, len(p) // block)
        aucs, accs = [], []
        for _ in range(n_boot):
            starts = rng.integers(0, max(1, len(p) - block), nb)
            idx = (starts[:, None] + np.arange(block)).ravel()
            idx = idx[idx < len(p)]
            if 0 < up[idx].sum() < len(idx):
                aucs.append(roc_auc_score(up[idx], p[idx]))
                accs.append(((p[idx] >= 0.5) == up[idx]).mean() - up[idx].mean())
        out["auc_lo"], out["auc_hi"] = np.percentile(aucs, [2.5, 97.5])
        out["acc_minus_alwaysup_lo"], out["acc_minus_alwaysup_hi"] = np.percentile(accs, [2.5, 97.5])
    return out


# ------------------------------------------------------------------------------- LLM
def llm_predict(kb, features, val_start, val_end, test_end, predictor: ClaudePredictor, step=5,
                max_val=80, max_test=300, k=15, include_headlines=False, use_playbook=True,
                playbook: str | None = None, log=print):
    """Returns (test_predictions {h: DataFrame}, playbook, validation records)."""
    retr = MarketRAG(RAGConfig(horizon=max(HORIZONS), features=features, use_news=False, k=k,
                               tune=False, min_coverage=0.3)).fit(kb, train_end=pd.Timestamp(val_end))
    tr = pd.to_datetime(kb[f"label_date_{max(HORIZONS)}"]) < pd.Timestamp(val_start)
    base = {h: float((kb.loc[tr, f"fwd_ret_{h}"] > 0).mean()) for h in HORIZONS}

    def queries(start, end, max_n):
        qi = _test_index(kb, start, end, max(HORIZONS), step)
        if len(qi) > max_n:  # spread evenly over the period
            qi = qi[np.linspace(0, len(qi) - 1, max_n).astype(int)]
        out = {}
        for i in qi:
            pr = retr.predict_index(i)
            if pr is None:
                continue
            an = kb.loc[pr.neighbors.index]
            an = an.loc[pr.neighbors.sort_values("weight", ascending=False).index]
            out[kb.index[i]] = (kb.iloc[i], an)
        return out

    def run(qs, pb):
        prompts = {d: build_prompt(row, an, base, pb, row["news_text"] if include_headlines else "")
                   for d, (row, an) in qs.items()}
        return prompts, predictor.predict_many(prompts)

    val_q = queries(val_start, val_end, max_val)
    records = None
    if use_playbook and playbook is None:
        log(f"  LLM validation pass: {len(val_q)} queries")
        _, val_out = run(val_q, "")
        recs = []
        for d, (row, an) in val_q.items():
            if d not in val_out:
                continue
            r = {"date": d, "metrics_text": fmt_metrics(row), "themes_text": fmt_themes(row)}
            for h in HORIZONS:
                r[f"analog_up_{h}"] = (an[f"fwd_ret_{h}"] > 0).mean()
                r[f"llm_p_{h}"] = val_out[d][f"p_up_{h}d"]
                r[f"actual_{h}"] = row[f"fwd_ret_{h}"]
            recs.append(r)
        records = pd.DataFrame(recs).set_index("date")
        playbook = predictor.learn_playbook(records)
        log("  playbook learned:\n" + "\n".join("    " + l for l in playbook.splitlines()))
    pb = playbook if use_playbook else ""

    test_q = queries(val_end, test_end, max_test)
    log(f"  LLM test pass: {len(test_q)} queries")
    prompts, test_out = run(test_q, pb)
    preds = {}
    for h in HORIZONS:
        rows = [(d, test_out[d][f"p_up_{h}d"], kb.loc[d, f"fwd_ret_{h}"],
                 (an[f"fwd_ret_{h}"] > 0).mean()) for d, (_, an) in test_q.items() if d in test_out]
        df = pd.DataFrame(rows, columns=["date", "p_up", "actual", "analog_vote"]).set_index("date")
        df["exp_ret"] = df["p_up"] - 0.5
        preds[h] = df.dropna(subset=["actual"])
    example = next(iter(prompts.values()), "")
    return preds, playbook or "", records, base, example
