"""The retriever / predictor.

Knowledge base row  = (metrics snapshot, news digest) at date t  ->  label fwd_ret_h(t)
Query               = (metrics snapshot, news digest) at date q
Retrieval           = most similar KB rows whose label was already known at q
                      (label_date < q), de-duplicated so analogs come from distinct episodes
Generation          = similarity-weighted outcome distribution (+ optional LLM narrative)

What is *learned* (refit on each training window):
  * robust scaling of every metric (median / IQR),
  * per-metric relevance weights (|rank correlation| with the forward return),
  * a news embedding space (TF-IDF -> truncated SVD / LSA) fit on training headlines,
  * the mixing weight between news and metrics similarity (alpha) and the number
    of analogs (k), chosen on a time-ordered validation slice of the training window.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import pandas as pd
from scipy.stats import spearmanr

from .features import FEATURE_COLUMNS, NEWS_COLUMNS


@dataclass
class RAGConfig:
    horizon: int = 21                 # forward window in trading days (21 ~ 1 month)
    k: int = 20                       # number of analogs
    alpha: float = 0.0                # weight of news similarity vs metric similarity
    min_gap: int = 10                 # min trading days between two analogs (distinct episodes)
    temperature: float = 0.5          # softmax temperature on z-scored similarity
    weighting: str = "ic"             # "ic" (learned) or "uniform" metric weights
    features: list[str] = field(default_factory=lambda: list(FEATURE_COLUMNS))
    svd_dims: int = 64
    prior_strength: float = 4.0       # shrink P(up) toward the base rate
    tune: bool = True
    alpha_grid: tuple = (0.0, 0.25, 0.5, 0.75)
    k_grid: tuple = (10, 20, 40)
    val_frac: float = 0.25
    min_coverage: float = 0.5         # drop metrics missing in more than half the training rows
    use_news: bool = True             # build the news embedding and allow alpha > 0


@dataclass
class Prediction:
    date: pd.Timestamp
    exp_ret: float
    p_up: float
    median_ret: float
    neighbors: pd.DataFrame
    alpha: float
    k: int


class MarketRAG:
    def __init__(self, cfg: RAGConfig | None = None):
        self.cfg = cfg or RAGConfig()
        self.text_model = None

    # ------------------------------------------------------------------ fitting
    def fit(self, kb: pd.DataFrame, train_end: pd.Timestamp | None = None,
            val_start: pd.Timestamp | None = None) -> "MarketRAG":
        """Learn from rows whose label is known before ``train_end``. If ``val_start`` is
        given, (alpha, k) are tuned on queries in [val_start, train_end) instead of the
        last ``val_frac`` of the window."""
        self.val_start = pd.Timestamp(val_start) if val_start is not None else None
        cfg = self.cfg
        h = cfg.horizon
        self.kb = kb
        self.y = kb[f"fwd_ret_{h}"].to_numpy(float)
        ld = pd.to_datetime(kb[f"label_date_{h}"])
        self.label_ns = np.where(ld.isna().to_numpy(), np.iinfo(np.int64).max,
                                 ld.to_numpy(dtype="datetime64[ns]").astype(np.int64))
        self.date_ns = kb.index.to_numpy(dtype="datetime64[ns]").astype(np.int64)
        self.pos = np.arange(len(kb))
        train_end = pd.Timestamp(train_end) if train_end is not None else kb.index[-1] + pd.Timedelta(days=1)
        self.train_end = train_end
        train = (self.label_ns < train_end.value) & ~np.isnan(self.y)
        if train.sum() < 200:
            raise ValueError(f"Only {train.sum()} labelled rows before {train_end.date()}")
        self.train_mask = train
        tr = kb.loc[train]

        # 1) metrics: robust scale + learned relevance weights
        # News features only exist for news-era days, so they are exempt from the coverage
        # rule and just need enough training rows.
        feats = [c for c in cfg.features if c in kb and (
            tr[c].notna().mean() >= cfg.min_coverage if c not in NEWS_COLUMNS else tr[c].notna().sum() >= 200)]
        self.features = feats
        self.is_news = np.array([c in NEWS_COLUMNS for c in feats])
        med = tr[feats].median()
        iqr = tr[feats].quantile(0.75) - tr[feats].quantile(0.25)
        sd = tr[feats].std()
        scale = iqr.where(iqr > 1e-9, sd).where(lambda x: x > 1e-9, 1.0)
        self.center, self.scale = med, scale
        if cfg.weighting == "ic":
            w = []
            for c in feats:
                ok = tr[c].notna()
                rho = spearmanr(tr.loc[ok, c], tr.loc[ok, f"fwd_ret_{h}"]).statistic if ok.sum() > 50 else 0
                w.append(abs(np.nan_to_num(rho)) + 0.02)
            w = np.array(w)
            w = w / w.mean()
        else:
            w = np.ones(len(feats))
        self.weights = pd.Series(w, index=feats)
        self.Xn = self._scale(kb[feats].to_numpy(float))

        # 2) news: LSA embedding fit on training headlines only
        texts = kb["news_text"].fillna("") if "news_text" in kb else pd.Series("", index=kb.index)
        self.has_text = (texts.str.len() > 0).to_numpy()
        self.Et = None
        if cfg.use_news and (self.has_text & train).sum() >= 100:
            from sklearn.decomposition import TruncatedSVD
            from sklearn.feature_extraction.text import TfidfVectorizer

            vec = TfidfVectorizer(stop_words="english", ngram_range=(1, 2), min_df=3,
                                  max_df=0.5, max_features=60000, sublinear_tf=True)
            Xtr = vec.fit_transform(texts[self.has_text & train])
            svd = TruncatedSVD(n_components=min(cfg.svd_dims, Xtr.shape[1] - 1), random_state=0)
            svd.fit(Xtr)
            self.text_model = (vec, svd)
            self.Et = np.zeros((len(kb), svd.n_components))
            self.Et[self.has_text] = self._embed(texts[self.has_text].tolist())

        self.base_rate = float((self.y[train] > 0).mean())
        self.alpha, self.k = (cfg.alpha if self.Et is not None else 0.0), cfg.k
        if cfg.tune:
            self._tune()
        return self

    def _scale(self, X: np.ndarray) -> np.ndarray:
        return (X - self.center.to_numpy()) / self.scale.to_numpy() * np.sqrt(self.weights.to_numpy())

    def _embed(self, texts: list[str]) -> np.ndarray:
        vec, svd = self.text_model
        E = svd.transform(vec.transform(texts))
        return E / (np.linalg.norm(E, axis=1, keepdims=True) + 1e-12)

    def _tune(self):
        """Pick (alpha, k) on the last ``val_frac`` of the training window (walk-forward inside).

        The greedy de-duplicated ranking is prefix-consistent, so one retrieval with the
        largest k serves every smaller k."""
        cfg = self.cfg
        idx = np.flatnonzero(self.train_mask)
        if self.val_start is not None:
            start = int(np.searchsorted(self.kb.index.values, np.datetime64(self.val_start)))
        else:
            start = idx[int(len(idx) * (1 - cfg.val_frac))]
        qs = idx[(idx >= start)][:: max(cfg.horizon // 2, 1)]
        alphas = cfg.alpha_grid if self.Et is not None else (0.0,)
        kmax = max(cfg.k_grid)
        best, best_score = (self.alpha, self.k), -np.inf
        for a in alphas:
            preds = {k: [] for k in cfg.k_grid}
            acts = []
            for i in qs:
                if a > 0 and not self.has_text[i]:
                    continue
                r = self._retrieve(self.Xn[i], self._et(i), self.date_ns[i], a, kmax, self.pos[i])
                if r is None or len(r[0]) < kmax:
                    continue
                acts.append(self.y[i])
                for k in cfg.k_grid:
                    sc = r[2][:k]
                    w = np.exp((sc - sc.max()) / cfg.temperature)
                    preds[k].append(float((w / w.sum()) @ self.y[r[0][:k]]))
            if len(acts) < 20:
                continue
            for k in cfg.k_grid:
                score = spearmanr(preds[k], acts).statistic
                if np.isfinite(score) and score > best_score:
                    best, best_score = (a, k), score
        self.alpha, self.k = best
        self.val_ic = best_score

    def _et(self, i):
        return self.Et[i] if (self.Et is not None and self.has_text[i]) else None

    # ------------------------------------------------------------------ retrieval
    def _retrieve(self, xn, et, as_of_ns, alpha, k, self_pos=None):
        cand = self.label_ns < as_of_ns
        cand &= ~np.isnan(self.y)
        use_text = alpha > 0 and et is not None and self.Et is not None
        if use_text:
            cand_t = cand & self.has_text
            if cand_t.sum() >= 3 * k:
                cand = cand_t
            else:
                use_text = False
        ci = np.flatnonzero(cand)
        if len(ci) < k:
            return None
        d = self._distance(self.Xn[ci], xn)
        s_num = -d
        fin = np.isfinite(s_num)
        z = np.full(len(ci), -np.inf)
        z[fin] = (s_num[fin] - s_num[fin].mean()) / (s_num[fin].std() + 1e-12)
        sims_txt = np.full(len(ci), np.nan)
        if use_text:
            sims_txt = self.Et[ci] @ et
            zt = (sims_txt - sims_txt.mean()) / (sims_txt.std() + 1e-12)
            z = np.where(fin, (1 - alpha) * z + alpha * zt, -np.inf)
        order = np.argsort(-z)
        chosen = []
        for j in order:
            if not np.isfinite(z[j]):
                break
            p = ci[j]
            if self_pos is not None and abs(p - self_pos) < self.cfg.min_gap:
                continue
            if all(abs(p - ci[c]) >= self.cfg.min_gap for c in chosen):
                chosen.append(j)
                if len(chosen) == k:
                    break
        if not chosen:
            return None
        chosen = np.array(chosen)
        sc = z[chosen]
        w = np.exp((sc - sc.max()) / self.cfg.temperature)
        w /= w.sum()
        return ci[chosen], w, sc, -s_num[chosen], sims_txt[chosen]

    def _distance(self, X: np.ndarray, xn: np.ndarray) -> np.ndarray:
        """NaN-aware weighted distance. Metric dims: mean squared diff over shared dims
        (candidate needs at least half of them). News dims: only if the query has news; a
        candidate without news gets the median news distance (neither near nor far)."""
        m, nw = ~self.is_news, self.is_news & ~np.isnan(xn)
        dm = X[:, m] - xn[m]
        vm = ~np.isnan(dm)
        nm = vm.sum(1)
        d2 = np.where(vm, dm, 0) ** 2 @ np.ones(dm.shape[1]) / np.maximum(nm, 1)
        d2[nm < 0.5 * m.sum()] = np.inf
        if nw.any():
            dn = X[:, nw] - xn[nw]
            vn = ~np.isnan(dn)
            nn = vn.sum(1)
            d2n = np.where(vn, dn, 0) ** 2 @ np.ones(dn.shape[1]) / np.maximum(nn, 1)
            has = nn >= 0.5 * nw.sum()
            if has.any():
                d2n[~has] = np.median(d2n[has])
                d2 = (d2 * m.sum() + d2n * nw.sum()) / (m.sum() + nw.sum())
        return np.sqrt(d2)

    def _predict_encoded(self, xn, et, as_of_ns, alpha, k, self_pos=None):
        r = self._retrieve(xn, et, as_of_ns, alpha, k, self_pos)
        if r is None:
            return None
        rows, w = r[0], r[1]
        y = self.y[rows]
        exp_ret = float(w @ y)
        n_eff = 1.0 / (w ** 2).sum()
        p_raw = float(w @ (y > 0))
        m = self.cfg.prior_strength
        p_up = (n_eff * p_raw + m * self.base_rate) / (n_eff + m)
        return exp_ret, p_up, r

    # ------------------------------------------------------------------ public API
    def predict_index(self, i: int) -> Prediction | None:
        """Predict for KB row i using only analogs whose outcome was known on that date."""
        return self._package(self.kb.index[i], self.Xn[i], self._et(i), self.date_ns[i], self.pos[i])

    def predict_snapshot(self, features: pd.Series, news_text: str = "",
                         as_of: pd.Timestamp | None = None) -> Prediction | None:
        """Predict for an arbitrary snapshot (e.g. today's live metrics + Yahoo headlines)."""
        as_of = pd.Timestamp(as_of) if as_of is not None else pd.Timestamp.now().normalize()
        x = np.array([features.get(c, np.nan) for c in self.features], float)
        xn = self._scale(x[None])[0]
        et = self._embed([news_text])[0] if (news_text and self.text_model) else None
        return self._package(as_of, xn, et, as_of.value + 1, None)

    def _package(self, date, xn, et, as_of_ns, self_pos):
        out = self._predict_encoded(xn, et, as_of_ns, self.alpha, self.k, self_pos)
        if out is None:
            return None
        exp_ret, p_up, (rows, w, sc, dist, tsim) = out
        h = self.cfg.horizon
        nb = self.kb.iloc[rows][self.features + ["close", f"fwd_ret_{h}"]].copy()
        nb.insert(0, "weight", w)
        nb.insert(1, "metric_dist", dist)
        nb.insert(2, "news_sim", tsim)
        if "news_text" in self.kb:
            nb["headlines"] = self.kb["news_text"].iloc[rows].fillna("").str.slice(0, 300).to_numpy()
        y = self.y[rows]
        return Prediction(date=pd.Timestamp(date), exp_ret=exp_ret, p_up=p_up,
                          median_ret=float(np.median(y)), neighbors=nb, alpha=self.alpha, k=self.k)

    def summary(self) -> str:
        w = self.weights.sort_values(ascending=False)
        top = ", ".join(f"{k}={v:.2f}" for k, v in w.head(8).items())
        return (f"trained to {self.train_end.date()} | base P(up)={self.base_rate:.2f} | "
                f"alpha={self.alpha} k={self.k} | val IC={getattr(self, 'val_ic', float('nan')):.3f} | "
                f"top metric weights: {top}")
