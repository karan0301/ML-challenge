"""Decision layer: choose, per Source-1 entity, which candidates to output.

Metric per S1:  F0.5 = 1.25*TP / (0.25*|truth| + |pred|)   (1.0 if truth and pred are both empty).

`select_expected_f` picks, for every S1, the prefix (top-k by probability) that maximises the EXPECTED F0.5
under the model's match probabilities (candidates treated as independent Bernoulli variables; a Poisson-binomial DP
gives the distribution of TP and of the matches left outside the prefix). k = 0 (empty prediction) is a first-class
option, so singletons are handled by the same rule: an S1 whose candidates all look unlikely gets an empty list.
`extra_missing` models true matches that blocking never surfaced (they count in |truth| but can never be hit).

A global probability threshold is available as a simpler baseline (see train.scan_thresholds).
"""
import numpy as np
import polars as pl


def _pmf(P: np.ndarray) -> np.ndarray:
    """Poisson-binomial pmf for every row of P (n, K). Returns (n, K+1)."""
    n, K = P.shape
    out = np.zeros((n, K + 1), np.float64)
    out[:, 0] = 1.0
    for j in range(K):
        p = P[:, j:j + 1]
        out[:, 1:j + 2] = out[:, 1:j + 2] * (1 - p) + out[:, 0:j + 1] * p
        out[:, 0:1] = out[:, 0:1] * (1 - p)
    return out


def best_prefix(P: np.ndarray, extra_missing: float = 0.0) -> np.ndarray:
    """P: (n, K) probabilities, each row sorted descending (padded with 0). Returns best k per row."""
    n, K = P.shape
    ks = np.arange(K + 1)
    E = np.zeros((n, K + 1))
    pre = [np.zeros((n, K + 1)) for _ in range(K + 1)]
    suf = [None] * (K + 1)
    # prefix pmf per k, suffix pmf per k
    cur = np.zeros((n, K + 1)); cur[:, 0] = 1.0
    pre[0] = cur.copy()
    for j in range(K):
        nxt = cur * (1 - P[:, j:j + 1])
        nxt[:, 1:] += cur[:, :-1] * P[:, j:j + 1]
        cur = nxt; pre[j + 1] = cur.copy()
    cur = np.zeros((n, K + 2)); cur[:, 0] = 1.0
    if extra_missing > 0:                       # matches blocking missed: an extra Bernoulli(min(extra,1)) outside every prefix
        q = min(extra_missing, 0.99)
        cur[:, 1] = q; cur[:, 0] = 1 - q
    suf[K] = cur.copy()
    for j in range(K - 1, -1, -1):
        nxt = cur * (1 - P[:, j:j + 1])
        nxt[:, 1:] += cur[:, :-1] * P[:, j:j + 1]
        cur = nxt; suf[j] = cur.copy()
    a = np.arange(K + 1)[:, None]; b = np.arange(K + 2)[None, :]
    for k in range(K + 1):
        if k == 0:
            E[:, 0] = suf[0][:, 0]              # empty prediction is right iff the S1 has no matches at all
            continue
        W = 1.25 * a / (0.25 * (a + b) + k)     # (K+1, K+2)
        E[:, k] = np.einsum("na,ab,nb->n", pre[k], W, suf[k])
    return E.argmax(1)


def select_expected_f(surv: pl.DataFrame, K: int = 14, gamma: float = 1.0, extra_missing: float = 0.0,
                      p_col: str = "p") -> pl.DataFrame:
    """surv: (s1, src, m, p). Returns the selected rows. gamma>1 sharpens probabilities (more conservative)."""
    d = surv.sort(["s1", p_col], descending=[False, True]).with_columns(
        pl.int_range(pl.len()).over("s1").alias("_r"))
    d = d.filter(pl.col("_r") < K)
    s1_ids = d["s1"].unique(maintain_order=True)
    idx = d.select(pl.col("s1").rle_id().alias("g"), "_r", p_col)
    n = int(idx["g"].max()) + 1
    P = np.zeros((n, K))
    P[idx["g"].to_numpy(), idx["_r"].to_numpy()] = np.clip(idx[p_col].to_numpy().astype(np.float64), 0, 1) ** gamma
    kbest = np.zeros(n, np.int64)
    step = 100_000
    for lo in range(0, n, step):
        kbest[lo:lo + step] = best_prefix(P[lo:lo + step], extra_missing)
    kb = kbest[idx["g"].to_numpy()]
    return d.filter(pl.Series(idx["_r"].to_numpy() < kb)).drop("_r")
