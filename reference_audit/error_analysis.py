import sys, numpy as np, polars as pl
sys.path.insert(0, ".")
from src import config
from src.candidate_generation import truth_table
tag = sys.argv[1] if len(sys.argv) > 1 else "cascade_v1"
V = pl.read_parquet(config.ANALYSIS_DIR / f"{tag}_V_survivors.parquet").select("s1", "src", "m", "y", pl.col("p2").alias("p"))
cols = ["eid", "country", "business_name", "business_address", "name_core", "addr_norm"]
s1 = pl.read_parquet(config.CACHE_DIR / "train_s1.parquet", columns=cols)
pool = pl.concat([pl.read_parquet(config.CACHE_DIR / f"train_s{i}.parquet", columns=cols).with_columns(pl.lit(i, dtype=pl.UInt8).alias("src")) for i in (2, 3)])
q = V.join(s1.rename({c: c + "1" for c in cols if c != "eid"}).rename({"eid": "s1"}), on="s1") \
     .join(pool.rename({c: c + "2" for c in cols if c != "eid"}).rename({"eid": "m"}), on=["m", "src"])
q = q.with_columns(pl.col("country1").cast(pl.Utf8))
print("n pairs", q.height)
# ownership of false positives: is the FP pool record a true match of ANOTHER S1?
truth = truth_table()
owned = truth.select("src", "m", pl.col("s1").alias("owner"))
fp = q.filter((pl.col("y") == 0) & (pl.col("p") >= 0.7)).join(owned, on=["src", "m"], how="left")
print(f"false positives (p>=0.7): {fp.height}  | pool record owned by another S1: {fp['owner'].is_not_null().mean():.3f}  | unowned decoy: {fp['owner'].is_null().mean():.3f}")
fn = q.filter((pl.col("y") == 1) & (pl.col("p") < 0.7))
tp = q.filter((pl.col("y") == 1) & (pl.col("p") >= 0.7))
print(f"false negatives (y=1, p<0.7): {fn.height} of {fn.height+tp.height} reachable positives ({fn.height/(fn.height+tp.height):.3%})")
for nm, df in (("FN", fn), ("TP", tp)):
    print(nm, "addr2 empty:", round((df['addr_norm2'] == '').mean(), 3), " name non-latin-ish (name_core2 has no vowel-ish overlap):",
          "country split:", df.group_by("country1").len().sort("country1").to_dicts())
print("FN p distribution:", np.round(np.quantile(fn['p'].to_numpy(), [0.1, .25, .5, .75, .9]), 3))
print("FN by pool addr empty:", fn.group_by(pl.col('addr_norm2') == '').len().to_dicts())
print("\n--- sample FN (missed true matches)")
for r in fn.sample(22, seed=3).iter_rows(named=True):
    print(f"p={r['p']:.2f} [{r['country1']} S{r['src']}] {r['business_name1']!r} | {r['business_address1']!r}\n         -> {r['business_name2']!r} | {r['business_address2']!r}")
print("\n--- sample FP (wrong merges)")
for r in fp.sample(14, seed=3).iter_rows(named=True):
    print(f"p={r['p']:.2f} owned={r['owner'] is not None} [{r['country1']} S{r['src']}] {r['business_name1']!r} | {r['business_address1']!r}\n         -> {r['business_name2']!r} | {r['business_address2']!r}")
