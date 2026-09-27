"""Candidate generation driver + blocking evaluation.

generate_candidates():  for every query S1 row, run per-(country, source) retrieval and return one long
                        frame  (s1_row, src, pool_row, score, rank)  with row indices into the cached tables.
evaluate_blocking():    candidate recall / size statistics against ground truth.
"""
import time

import numpy as np
import polars as pl

from . import config
from .blocking import BlockingConfig, PoolIndex, retrieve
from .data_loader import read_ground_truth, gt_pairs
from .preprocessing import cache_path


def load_cache(split: str, cols=None):
    out = {}
    for i in (1, 2, 3):
        df = pl.read_parquet(cache_path(split, i), columns=cols)
        out[i] = df
    return out


def truth_table(split="train") -> pl.DataFrame:
    """(s1, src, m) numeric ids of every true pair."""
    gt = read_ground_truth(config.TRAIN_DIR / "train_ground_truth.tsv")
    p = gt_pairs(gt)
    return p.select(pl.col("source1_entity_id").str.slice(3).cast(pl.UInt32).alias("s1"),
                    pl.col("matched_id").str.slice(1, 1).cast(pl.UInt8).alias("src"),
                    pl.col("matched_id").str.slice(3).cast(pl.UInt32).alias("m"))


def _view_frame(r, c, s, vi):
    """(q_row, pool_row, score) sorted by q_row,-score  ->  frame with sc{vi}, rk{vi}."""
    df = pl.DataFrame({"q": r, "p": c, "sc": s})
    return df.with_columns((pl.int_range(pl.len()).over("q") + 1).cast(pl.UInt16).alias("rk")).rename(
        {"sc": f"sc{vi}", "rk": f"rk{vi}"})


class CountryRetriever:
    """Inverted indexes (S2 and S3) of one country's pool, kept alive so queries can be streamed in batches."""

    def __init__(self, country, pools, cfg, workers=8, verbose=True):
        self.country, self.cfg, self.workers, self.verbose = country, cfg, workers, verbose
        self.p_rows, self.idx = {}, {}
        for src in (2, 3):
            t = time.time()
            self.p_rows[src] = np.flatnonzero((pools[src]["country"].cast(pl.Utf8) == country).to_numpy())
            self.idx[src] = PoolIndex(pools[src][self.p_rows[src]], cfg)
            if verbose:
                print(f"  {country:7s} S{src}: pool={len(self.p_rows[src]):>9,} index built in {time.time() - t:.0f}s", flush=True)

    def retrieve(self, q: pl.DataFrame) -> pl.DataFrame:
        """q: query frame (all rows must be of this country). Returns
        (s1_row u32 [row of q], src u8, pool_row u32 [row of the FULL source frame], sc<i>, rk<i>, rank, score)
        sorted by (s1_row, src, rank, -score)."""
        nv = len(self.cfg.views)
        out = []
        for src in (2, 3):
            t = time.time()
            res = retrieve(self.idx[src], q, self.cfg, workers=self.workers)
            m = None
            for vi, (r, pc, v) in enumerate(res):
                f = _view_frame(r, pc, v, vi)
                m = f if m is None else m.join(f, on=["q", "p"], how="full", coalesce=True)
            if self.verbose:
                print(f"  {self.country:7s} S{src}: queries={q.height:>8,} retrieve {time.time() - t:5.0f}s  "
                      f"cands={m.height:,}", flush=True)
            pr = self.p_rows[src]
            out.append(m.select(pl.col("q").cast(pl.UInt32).alias("s1_row"),
                                pl.Series("pool_row", pr[m["p"].to_numpy()].astype(np.uint32)),
                                pl.lit(src, dtype=pl.UInt8).alias("src"),
                                *[pl.col(c) for c in m.columns if c[:2] in ("sc", "rk")]))
        cand = pl.concat(out, how="diagonal_relaxed")
        rk = [f"rk{i}" for i in range(nv)]; sc = [f"sc{i}" for i in range(nv)]
        cand = cand.with_columns(pl.min_horizontal(rk).alias("rank"), pl.max_horizontal(sc).alias("score"))
        return cand.sort(["s1_row", "src", "rank", "score"], descending=[False, False, False, True])


def generate_candidates(s1: pl.DataFrame, pools: dict, cfg: BlockingConfig,
                        countries=None, workers=8, verbose=True) -> pl.DataFrame:
    """Multi-pass blocking for ALL rows of s1 (rows = queries), country by country (country is a hard key).
    Every view (BlockingConfig.views) retrieves its own top-K per (S1, source); the UNION is the candidate set.
    Output columns: s1_row, src, pool_row, sc<i>/rk<i> (score/rank in view i; null if not retrieved there),
    rank (best rank over views), score (best score). Sorted by (s1_row, src, rank, -score)."""
    out = []
    cs = countries or s1["country"].cast(pl.Utf8).unique().to_list()
    for c in cs:
        q_rows = np.flatnonzero((s1["country"].cast(pl.Utf8) == c).to_numpy())
        R = CountryRetriever(c, pools, cfg, workers, verbose)
        df = R.retrieve(s1[q_rows])
        del R
        out.append(df.with_columns(pl.Series("s1_row", q_rows[df["s1_row"].to_numpy()].astype(np.uint32))))
    return pl.concat(out, how="diagonal_relaxed").sort(
        ["s1_row", "src", "rank", "score"], descending=[False, False, False, True])


def label_candidates(cand: pl.DataFrame, s1: pl.DataFrame, pools: dict, truth: pl.DataFrame) -> pl.DataFrame:
    """Adds column `y` (1 if the pair is in the ground truth)."""
    s1_eid = s1["eid"].gather(cand["s1_row"].to_numpy())
    # gather per source (preserving candidate order)
    m = np.zeros(cand.height, dtype=np.uint32)
    for src in (2, 3):
        mask = (cand["src"] == src).to_numpy()
        m[mask] = pools[src]["eid"].gather(cand["pool_row"].to_numpy()[mask]).to_numpy()
    c = cand.with_columns(pl.Series("s1", s1_eid.to_numpy()), pl.Series("m", m))
    t = truth.with_columns(pl.lit(1, dtype=pl.UInt8).alias("y"))
    return c.join(t, on=["s1", "src", "m"], how="left").with_columns(pl.col("y").fill_null(0))


def evaluate_blocking(cand_labeled: pl.DataFrame, s1_rows: np.ndarray, s1: pl.DataFrame,
                      truth: pl.DataFrame, tag="", verbose=True):
    """Candidate recall (against ALL true pairs of the queried S1s) and candidate-set size stats."""
    q_eids = s1["eid"].gather(s1_rows)
    n_true = truth.join(pl.DataFrame({"s1": q_eids}), on="s1", how="semi").height
    n_q = len(s1_rows)
    pos = cand_labeled.filter(pl.col("y") == 1)
    res = {"recall_union": pos.height / n_true, "avg_cands": cand_labeled.height / n_q}
    for c in [c for c in cand_labeled.columns if c.startswith("rk")]:
        res[f"recall_{c}"] = pos.filter(pl.col(c).is_not_null()).height / n_true
    sz = cand_labeled.group_by("s1_row").len()["len"]
    res.update(median=float(sz.median()), p95=float(sz.quantile(0.95)), max=int(sz.max()))
    if verbose:
        print(f"[{tag}] queries={n_q:,} true_pairs={n_true:,} | recall(union)={res['recall_union']:.4f} "
              f"avg cands/S1={res['avg_cands']:.1f} median={res['median']:.0f} p95={res['p95']:.0f} max={res['max']}")
        print("   per-view recall: " + "  ".join(f"{k}={v:.4f}" for k, v in res.items() if k.startswith("recall_rk")))
        for k in (1, 2, 3, 5, 8, 10):
            got = pos.filter(pl.col("rank") <= k).height / n_true
            print(f"   best-rank<={k}: {got:.4f}", end="")
        print()
    return res
