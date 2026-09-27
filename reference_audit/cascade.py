"""Two-stage candidate cascade shared by training and inference.

  stage 0  retrieval        IDF-token blocking, top-`top_k` per (S1, source)            (blocking.py)
  stage 1  learned pruning  lightweight LightGBM ranker M1 keeps the best `keep` per S1  -> candidate_pairs.tsv
  stage 2  final matcher    LightGBM M2 scores ONLY the survivors                        -> matching_results.tsv

candidate_pairs.tsv is exactly the set of survivors that M2 scores; matching_results.tsv is a subset of it.
Both stages use the same pair features; M2 additionally sees M1's per-S1 context (M1 probability, rank, the
S1's best / summed M1 probability), which tells it how many matches to expect.
"""
import numpy as np
import polars as pl

from .features import feature_chunks

M2_EXTRA = ["m1_p", "m1_rank", "m1_p_max", "m1_p_sum", "m1_ratio_max", "m1_p2", "m1_n_over_half"]


def stage1_chunks(cand, s1, pools, tables, m1, feat_cols, keep, chunk=400_000, p_min=0.0):
    """Yield (lo, hi, survivor_mask, F_survivors_with_m1_features, p1_all) per chunk.
    p1_all is M1's probability for every wide candidate of the chunk (needed for recall-vs-K tables)."""
    for lo, hi, F in feature_chunks(cand, s1, pools, tables, chunk):
        X = F.select(feat_cols).to_numpy()
        p1 = m1.predict(X, num_threads=10).astype(np.float32)
        yield lo, hi, *_survivors(cand[lo:hi], F, p1, keep, p_min), p1


def _survivors(c, F, p1, keep, p_min):
    d = pl.DataFrame({"s": c["s1_row"], "p": p1, "i": np.arange(len(p1))}).with_columns(
        pl.col("p").rank("ordinal", descending=True).over("s").alias("rk"),
        pl.col("p").max().over("s").alias("pmax"), pl.col("p").sum().over("s").alias("psum"),
        (pl.col("p") > 0.5).sum().over("s").alias("nhalf"))
    p2 = d.group_by("s").agg(pl.col("p").sort(descending=True).get(1, null_on_oob=True).alias("p2"))
    d = d.join(p2, on="s", how="left").sort("i")
    mask = ((d["rk"] <= keep) & (d["p"] >= p_min)).to_numpy()
    E = pl.DataFrame({
        "m1_p": d["p"], "m1_rank": d["rk"].cast(pl.Float32), "m1_p_max": d["pmax"], "m1_p_sum": d["psum"],
        "m1_ratio_max": d["p"] / d["pmax"].clip(1e-9), "m1_p2": d["p2"].fill_null(0.0),
        "m1_n_over_half": d["nhalf"].cast(pl.Float32),
    }).cast(pl.Float32)
    return mask, pl.concat([F, E], how="horizontal").filter(pl.Series(mask))
