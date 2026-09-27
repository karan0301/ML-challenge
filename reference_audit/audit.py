"""Phase A: data audit. Prints + saves artifacts/analysis/audit_<split>.txt

    python -m src.audit            # from business_entity_resolution/
"""
import sys
import time

import polars as pl

from . import config
from .data_loader import load_split, gt_pairs

pl.Config.set_tbl_rows(40)
pl.Config.set_tbl_cols(20)
pl.Config.set_fmt_str_lengths(60)

_out = []


def P(*a):
    s = " ".join(str(x) for x in a)
    print(s, flush=True)
    _out.append(s)


def describe_source(name, df):
    P(f"\n=== {name}: {df.height:,} rows")
    for c in ("business_name", "business_address", "country"):
        empty = int((df[c] == "").sum())
        P(f"  {c:17s} empty={empty:>9,} ({empty / df.height:6.2%})  "
          f"len mean={df[c].str.len_chars().mean():.1f} "
          f"p50={df[c].str.len_chars().quantile(0.5):.0f} "
          f"p95={df[c].str.len_chars().quantile(0.95):.0f} max={df[c].str.len_chars().max()}")
    P(f"  duplicate entity_id: {df.height - df['entity_id'].n_unique():,}")
    dup_full = df.height - df.select(["business_name", "business_address", "country"]).n_unique()
    P(f"  exact duplicate (name,addr,country) rows: {dup_full:,} ({dup_full / df.height:.2%})")
    dup_name = df.height - df.select(["business_name", "country"]).n_unique()
    P(f"  duplicate (name,country): {dup_name:,} ({dup_name / df.height:.2%})")
    dup_addr = df.filter(pl.col("business_address") != "").select(["business_address", "country"])
    P(f"  duplicate non-empty (addr,country): {dup_addr.height - dup_addr.n_unique():,}")
    P("  country dist:")
    P(df.group_by("country").len().sort("len", descending=True).with_columns(
        (pl.col("len") / df.height).round(4).alias("frac")).to_pandas().to_string(index=False))
    nonascii = df.with_columns(
        pl.col("business_name").str.contains(r"[^\x00-\x7F]").alias("name_nonascii"),
        pl.col("business_address").str.contains(r"[^\x00-\x7F]").alias("addr_nonascii"),
        pl.col("business_name").str.contains(r"[ऀ-ॿ]").alias("name_devanagari"),
        pl.col("business_address").str.contains(r"[ऀ-ॿ]").alias("addr_devanagari"))
    P("  non-ASCII / Devanagari share by country:")
    P(nonascii.group_by("country").agg(
        pl.col("name_nonascii").mean().round(4), pl.col("addr_nonascii").mean().round(4),
        pl.col("name_devanagari").mean().round(4), pl.col("addr_devanagari").mean().round(4)
    ).to_pandas().to_string(index=False))
    P("  empty address share by country:")
    P(df.group_by("country").agg((pl.col("business_address") == "").mean().round(4).alias("addr_empty")
                                 ).to_pandas().to_string(index=False))


def audit(split):
    t = time.time()
    d = load_split(split)
    P(f"##### {split.upper()}  (loaded in {time.time() - t:.0f}s)")
    for k in ("s1", "s2", "s3"):
        describe_source(f"{split}_{k}", d[k])
    if split != "train":
        return d
    s1, s2, s3, gt = d["s1"], d["s2"], d["s3"], d["gt"]
    P("\n=== ground truth")
    P(f"  rows={gt.height:,}  S1 rows={s1.height:,}  same ids: "
      f"{set(gt['source1_entity_id'].to_list()) == set(s1['entity_id'].to_list())}")
    g = gt.with_columns(
        pl.when(pl.col("matched_entity_ids") == "").then(0)
          .otherwise(pl.col("matched_entity_ids").str.count_matches(",") + 1).alias("n_match"),
        pl.col("matched_entity_ids").str.count_matches("S2-").alias("n_s2"),
        pl.col("matched_entity_ids").str.count_matches("S3-").alias("n_s3"))
    n = g.height
    P(f"  singletons (0 matches): {(g['n_match'] == 0).sum():,} ({(g['n_match'] == 0).mean():.2%})")
    P("  matches-per-S1 distribution:")
    P(g.group_by("n_match").len().sort("n_match").with_columns(
        (pl.col("len") / n).round(4).alias("frac")).to_pandas().to_string(index=False))
    P(f"  mean matches/S1 {g['n_match'].mean():.2f}; S2/S1 {g['n_s2'].mean():.2f}; S3/S1 {g['n_s3'].mean():.2f}")
    P("  n_s2 dist:"); P(g.group_by("n_s2").len().sort("n_s2").to_pandas().to_string(index=False))
    P("  n_s3 dist:"); P(g.group_by("n_s3").len().sort("n_s3").to_pandas().to_string(index=False))
    pairs = gt_pairs(gt)
    P(f"  total positive pairs: {pairs.height:,};  unique matched ids {pairs['matched_id'].n_unique():,}")
    multi = pairs.height - pairs["matched_id"].n_unique()
    P(f"  matched ids appearing under >1 S1: {multi:,}")
    s2_ids, s3_ids = set(s2["entity_id"].to_list()), set(s3["entity_id"].to_list())
    mid = pairs["matched_id"].to_list()
    bad = sum(1 for m in mid if m not in s2_ids and m not in s3_ids)
    P(f"  matched ids not present in S2/S3 train files: {bad:,}")
    P(f"  S2 rows that match some S1: {pairs.filter(pl.col('matched_id').str.starts_with('S2-'))['matched_id'].n_unique():,} / {s2.height:,}")
    P(f"  S3 rows that match some S1: {pairs.filter(pl.col('matched_id').str.starts_with('S3-'))['matched_id'].n_unique():,} / {s3.height:,}")

    # per-country match structure + country agreement of matches
    s1c = s1.select(pl.col("entity_id").alias("source1_entity_id"), pl.col("country").alias("c1"),
                    pl.col("business_name").alias("n1"), pl.col("business_address").alias("a1"))
    gc = g.join(s1c, on="source1_entity_id")
    P("  per-country: n_S1, singleton rate, mean matches:")
    P(gc.group_by("c1").agg(pl.len().alias("n"), (pl.col("n_match") == 0).mean().round(4).alias("singleton"),
                            pl.col("n_match").mean().round(3).alias("mean_match"),
                            pl.col("n_s2").mean().round(3).alias("mean_s2"),
                            pl.col("n_s3").mean().round(3).alias("mean_s3")).to_pandas().to_string(index=False))
    allb = pl.concat([s2.select("entity_id", pl.col("country").alias("c2"),
                                pl.col("business_name").alias("n2"), pl.col("business_address").alias("a2")),
                      s3.select("entity_id", pl.col("country").alias("c2"),
                                pl.col("business_name").alias("n2"), pl.col("business_address").alias("a2"))])
    pj = pairs.join(s1c, on="source1_entity_id").join(allb, left_on="matched_id", right_on="entity_id")
    P(f"  joined positive pairs: {pj.height:,}")
    P(f"  match country == S1 country: {(pj['c1'] == pj['c2']).mean():.4%}")
    pj = pj.with_columns(pl.col("matched_id").str.slice(0, 2).alias("src"))
    P("  identical raw name  (S1 vs match) by src/country:")
    P(pj.group_by(["src", "c1"]).agg(
        (pl.col("n1") == pl.col("n2")).mean().round(4).alias("name_identical"),
        (pl.col("a1") == pl.col("a2")).mean().round(4).alias("addr_identical"),
        (pl.col("a2") == "").mean().round(4).alias("match_addr_empty"),
        pl.col("n2").str.contains(r"[^\x00-\x7F]").mean().round(4).alias("match_name_nonascii")
    ).sort(["src", "c1"]).to_pandas().to_string(index=False))
    P("\n  sample positive pairs (US):")
    for r in pj.filter(pl.col("c1") == "US").sample(12, seed=1).iter_rows(named=True):
        P(f"   [{r['src']}] {r['n1']!r} | {r['a1']!r}\n        -> {r['n2']!r} | {r['a2']!r}")
    P("\n  sample positive pairs (India):")
    for r in pj.filter(pl.col("c1") == "India").sample(14, seed=1).iter_rows(named=True):
        P(f"   [{r['src']}] {r['n1']!r} | {r['a1']!r}\n        -> {r['n2']!r} | {r['a2']!r}")
    return d


if __name__ == "__main__":
    splits = sys.argv[1:] or ["train", "test"]
    for s in splits:
        audit(s)
        (config.ANALYSIS_DIR / f"audit_{s}.txt").write_text("\n".join(_out))
        _out.clear()
