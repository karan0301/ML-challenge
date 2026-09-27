import sys, numpy as np, polars as pl
sys.path.insert(0, ".")
from src import config
from src.candidate_generation import truth_table
from src.decision import select_expected_f
from src.evaluate import macro_f05
tag = sys.argv[1] if len(sys.argv) > 1 else "cascade_v1"
V = pl.read_parquet(config.ANALYSIS_DIR / f"{tag}_V_survivors.parquet").select("s1", "src", "m", "y", pl.col("p2").alias("p"))
s1_ids = V["s1"].unique()
truth = truth_table().join(pl.DataFrame({"s1": s1_ids}), on="s1", how="semi")
def ev(sel): 
    r = macro_f05(sel.select("s1", "src", "m"), truth, s1_ids); return r
base = V.filter(pl.col("p") >= 0.70)
r = ev(base); print(f"threshold 0.70          F0.5 {r['macro_f05']:.4f}  P {r['micro_precision']:.4f} R {r['micro_recall']:.4f} singleton {r['singleton_score']:.4f}")
for gamma in (0.8, 1.0, 1.2, 1.5):
    for extra in (0.0, 0.03, 0.06):
        sel = select_expected_f(V, K=14, gamma=gamma, extra_missing=extra)
        r = ev(sel)
        print(f"expected-F g={gamma} miss={extra:.2f} F0.5 {r['macro_f05']:.4f}  P {r['micro_precision']:.4f} R {r['micro_recall']:.4f} singleton {r['singleton_score']:.4f}  n={sel.height}")
