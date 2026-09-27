"""Official metric: macro-averaged F0.5 over Source-1 entities (singletons included).

Per S1:  F0.5 = 1.25*TP / (0.25*|truth| + |pred|)     (identical to 1.25*P*R / (0.25*P + R))
         no truth and no prediction -> 1.0 ;  prediction on a singleton -> 0.0 ;  truth but no prediction -> 0.0
"""
import numpy as np
import polars as pl

BETA2 = 0.25


def per_s1_f05(pred: pl.DataFrame, truth: pl.DataFrame, s1_ids: pl.Series) -> pl.DataFrame:
    """pred / truth: frames with columns (s1, src, m). s1_ids: every S1 id being evaluated.
    Returns one row per S1: (s1, tp, n_pred, n_true, f05)."""
    base = pl.DataFrame({"s1": s1_ids})
    t = truth.join(base, on="s1", how="semi")
    p = pred.join(base, on="s1", how="semi")
    n_true = t.group_by("s1").len().rename({"len": "n_true"})
    n_pred = p.group_by("s1").len().rename({"len": "n_pred"})
    tp = (p.join(t, on=["s1", "src", "m"], how="inner").group_by("s1").len().rename({"len": "tp"}))
    r = (base.join(n_true, on="s1", how="left").join(n_pred, on="s1", how="left")
         .join(tp, on="s1", how="left").with_columns(pl.col(["n_true", "n_pred", "tp"]).fill_null(0)))
    return r.with_columns(
        pl.when((pl.col("n_true") == 0) & (pl.col("n_pred") == 0)).then(1.0)
        .otherwise(1.25 * pl.col("tp") / (BETA2 * pl.col("n_true") + pl.col("n_pred"))).alias("f05"))


def macro_f05(pred, truth, s1_ids) -> dict:
    r = per_s1_f05(pred, truth, s1_ids)
    tp, npred, ntrue = r["tp"].sum(), r["n_pred"].sum(), r["n_true"].sum()
    single = r.filter(pl.col("n_true") == 0)
    return {
        "macro_f05": float(r["f05"].mean()),
        "micro_precision": float(tp / npred) if npred else 1.0,
        "micro_recall": float(tp / ntrue) if ntrue else 1.0,
        "singleton_score": float(single["f05"].mean()) if single.height else float("nan"),
        "nonsingleton_score": float(r.filter(pl.col("n_true") > 0)["f05"].mean()),
        "n_s1": r.height, "n_pred": int(npred), "n_true": int(ntrue),
    }
