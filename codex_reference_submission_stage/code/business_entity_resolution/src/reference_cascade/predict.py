"""Inference on the test set (streaming, country by country, S1 batches):

    retrieval (top-K/src) -> features -> M1 prune -> candidate_pairs.tsv -> M2 -> threshold -> matching_results.tsv

    python -m src.predict --model cascade_v1 --threshold 0.6

candidate_pairs.tsv holds exactly the survivors that M2 scores; matching_results.tsv is a subset of them (asserted).
Every test S1 appears exactly once in both files (empty list when nothing survives / matches). Country is an
open label: whatever countries the test files contain (France included) are handled the same way.
"""
import argparse
import json
import time
from pathlib import Path

import lightgbm as lgb
import numpy as np
import polars as pl

from . import config
from .blocking import BlockingConfig
from .candidate_generation import CountryRetriever, load_cache
from .cascade import stage1_chunks
from .features import add_context_columns, build_df_tables
from .train import CACHE_COLS


def id_lists(df: pl.DataFrame, s1_ids: pl.Series, sort_col=None) -> pl.DataFrame:
    """df: (s1 eid, src, m[, p]) -> one row per test S1 (in file order) with comma-joined S2-/S3- ids."""
    d = df.with_columns((pl.lit("S") + pl.col("src").cast(pl.Utf8) + pl.lit("-") + pl.col("m").cast(pl.Utf8)).alias("id"))
    if sort_col:
        d = d.sort(sort_col, descending=True)
    agg = d.group_by("s1", maintain_order=True).agg(pl.col("id").str.join(","))
    return pl.DataFrame({"s1": s1_ids}).join(agg, on="s1", how="left").with_columns(pl.col("id").fill_null(""))


def write_tsv(ids: pl.DataFrame, path, col_name):
    out = ids.select((pl.lit("S1-") + pl.col("s1").cast(pl.Utf8)).alias("source1_entity_id"), pl.col("id").alias(col_name))
    out.write_csv(path, separator="\t", quote_style="never")


def resolve_conflicts(sel: pl.DataFrame) -> pl.DataFrame:
    """A pool record belongs to at most one S1 (verified on train ground truth: 0 shared matches):
    keep each S2/S3 record only for its highest-probability S1."""
    return sel.sort("p", descending=True).unique(subset=["src", "m"], keep="first", maintain_order=True)


def score_test(model_tag: str, workers=8, batch=150_000, split="test", limit=None, keep=None):
    """Runs the whole cascade; returns (s1 frame, survivors frame with columns s1, src, m, p2, p1)."""
    meta = json.load(open(config.MODELS_DIR / f"{model_tag}.json"))
    m1 = lgb.Booster(model_file=str(config.MODELS_DIR / f"{model_tag}_m1.txt"))
    m2 = lgb.Booster(model_file=str(config.MODELS_DIR / f"{model_tag}_m2.txt"))
    cfg = BlockingConfig.from_dict(meta["blocking"])
    keep = keep or meta["keep"]
    d = load_cache(split, CACHE_COLS)
    s1, pools = add_context_columns(d[1], {2: d[2], 3: d[3]})
    tables = build_df_tables(pl.concat([pools[2], pools[3]]))
    countries = sorted(s1["country"].cast(pl.Utf8).unique().to_list())
    print(f"{split}: S1={s1.height:,} S2={pools[2].height:,} S3={pools[3].height:,} countries={countries}", flush=True)

    parts, wide_total, t0 = [], 0, time.time()
    for c in countries:
        rows = np.flatnonzero((s1["country"].cast(pl.Utf8) == c).to_numpy())
        if limit:
            rows = rows[:limit]
        R = CountryRetriever(c, pools, cfg, workers)
        for b in range(0, len(rows), batch):
            qb = s1[rows[b:b + batch]]
            cand = R.retrieve(qb)
            wide_total += cand.height
            for lo, hi, mask, Fs, p1 in stage1_chunks(cand, qb, pools, tables, m1, meta["feat_cols"], keep):
                p2 = m2.predict(Fs.select(meta["m2_cols"]).to_numpy(), num_threads=10).astype(np.float32)
                cs = cand[lo:hi].filter(pl.Series(mask))
                m = np.zeros(cs.height, np.uint32)
                for src in (2, 3):
                    k = (cs["src"] == src).to_numpy()
                    m[k] = pools[src]["eid"].gather(cs["pool_row"].to_numpy()[k]).to_numpy()
                parts.append(pl.DataFrame({"s1": qb["eid"].gather(cs["s1_row"].to_numpy()), "src": cs["src"],
                                           "m": m, "p2": p2, "p1": Fs["m1_p"]}))
            print(f"[{c}] {min(b + batch, len(rows)):,}/{len(rows):,} S1 done  ({time.time() - t0:.0f}s)", flush=True)
        del R
    surv = pl.concat(parts)
    print(f"wide candidates {wide_total:,} -> survivors {surv.height:,} ({surv.height / s1.height:.1f}/S1)", flush=True)
    return s1, surv


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="cascade_v1")
    ap.add_argument("--threshold", type=float, required=True)
    ap.add_argument("--workers", type=int, default=8)
    ap.add_argument("--batch", type=int, default=150_000)
    ap.add_argument("--keep", type=int, default=None, help="candidates kept per S1 after M1 pruning (default: model meta)")
    ap.add_argument("--out", default=str(config.OUTPUT_DIR))
    ap.add_argument("--rescore", action="store_true", help="reuse artifacts/analysis/test_survivors.parquet")
    a = ap.parse_args()
    t0 = time.time()
    surv_path = config.ANALYSIS_DIR / "test_survivors.parquet"
    if a.rescore and surv_path.exists():
        surv = pl.read_parquet(surv_path)
        s1 = load_cache("test", ["eid"])[1]
    else:
        s1, surv = score_test(a.model, a.workers, a.batch, keep=a.keep)
        surv.write_parquet(surv_path)

    sel = resolve_conflicts(surv.filter(pl.col("p2") >= a.threshold).rename({"p2": "p"}))
    n_matched = sel["s1"].n_unique()
    print(f"threshold {a.threshold}: {sel.height:,} matches for {n_matched:,} of {s1.height:,} S1 "
          f"({1 - n_matched / s1.height:.2%} predicted singletons)")
    assert sel.join(surv, on=["s1", "src", "m"], how="anti").height == 0        # matches ⊂ candidates
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    write_tsv(id_lists(surv, s1["eid"], "p2"), out / "candidate_pairs.tsv", "candidate_entity_ids")
    write_tsv(id_lists(sel, s1["eid"], "p"), out / "matching_results.tsv", "matched_entity_ids")
    print(f"wrote {out}/candidate_pairs.tsv and matching_results.tsv  (total {time.time() - t0:.0f}s)")


if __name__ == "__main__":
    main()
