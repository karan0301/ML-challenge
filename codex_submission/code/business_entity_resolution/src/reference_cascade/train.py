"""Train + validate the candidate cascade (see cascade.py).

    python -m src.train --tag cascade_v1

S1-level folds (disjoint Source-1 entities, so no entity leakage and M2 never sees data M1 was fitted on):
    A   fits M1 (light pruner)                  ES  early stopping for M1 and M2
    B   fits M2 on M1's survivors               V   held-out validation, scored against the FULL ground truth
Hard negatives come from retrieval itself: every non-matching retrieved candidate (same country, high token
overlap) is a negative, so the models learn exactly the confusions blocking produces.
Validation candidates go through the same blocking + pruning code as the test set.
"""
import argparse
import json
import time

import lightgbm as lgb
import numpy as np
import polars as pl

from . import config
from .blocking import BlockingConfig, View
from .candidate_generation import generate_candidates, label_candidates, load_cache, truth_table
from .cascade import M2_EXTRA, stage1_chunks
from .evaluate import macro_f05
from .features import add_context_columns, build_df_tables, feature_chunks
from .utils import log_experiment

CACHE_COLS = ["eid", "country", "name_full", "name_core", "name_alnum", "name_sorted",
              "addr_norm", "addr_nums"]

M1_PARAMS = dict(objective="binary", learning_rate=0.1, num_leaves=63, min_data_in_leaf=200, feature_fraction=0.8,
                 bagging_fraction=0.7, bagging_freq=1, lambda_l2=5.0, num_threads=10, verbose=-1)
M2_PARAMS = dict(objective="binary", learning_rate=0.05, num_leaves=127, min_data_in_leaf=100, feature_fraction=0.8,
                 bagging_fraction=0.8, bagging_freq=1, lambda_l2=5.0, num_threads=10, verbose=-1)


def sample_s1(s1: pl.DataFrame, sizes: dict, seed: int):
    """Disjoint random S1 row-index sets."""
    perm = np.random.default_rng(seed).permutation(s1.height)
    out, off = {}, 0
    for name, n in sizes.items():
        out[name] = np.sort(perm[off:off + n]); off += n
    return out


def scan_thresholds(q, lab, prob, truth, ths=np.arange(0.10, 0.96, 0.05), conflict=True):
    """macro F0.5 on the S1 set `q` for each probability threshold. lab needs (s1, src, m)."""
    pool_m = lab.select("s1", "src", "m", pl.Series("p", prob))
    tr = truth.join(pl.DataFrame({"s1": q["eid"]}), on="s1", how="semi")
    rows = []
    for th in ths:
        sel = pool_m.filter(pl.col("p") >= th)
        if conflict:   # a pool record belongs to at most one S1: keep its best-scoring S1
            sel = sel.sort("p", descending=True).unique(subset=["src", "m"], keep="first")
        r = macro_f05(sel.select("s1", "src", "m"), tr, q["eid"])
        r["threshold"] = round(float(th), 2)
        rows.append(r)
    return rows


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n-a", type=int, default=100_000)
    ap.add_argument("--n-b", type=int, default=150_000)
    ap.add_argument("--n-es", type=int, default=30_000)
    ap.add_argument("--n-valid", type=int, default=100_000)
    ap.add_argument("--top-k", type=int, default=40, help="retrieval depth per S1 per source")
    ap.add_argument("--keep", type=int, default=14, help="survivors per S1 after M1 pruning")
    ap.add_argument("--seed", type=int, default=config.SEED)
    ap.add_argument("--tag", default="cascade_v1")
    ap.add_argument("--workers", type=int, default=8)
    a = ap.parse_args()
    t0 = time.time()

    d = load_cache("train", CACHE_COLS)
    truth = truth_table()
    s1_full, pools = add_context_columns(d[1], {2: d[2], 3: d[3]})
    cfg = BlockingConfig(views=[View("A", True, 1.0, a.top_k)])
    tables = build_df_tables(pl.concat([pools[2], pools[3]]))
    print(f"loaded + df tables {time.time() - t0:.0f}s", flush=True)

    sets = sample_s1(s1_full, {"A": a.n_a, "B": a.n_b, "ES": a.n_es, "V": a.n_valid}, a.seed)
    all_rows = np.sort(np.concatenate(list(sets.values())))
    q = s1_full[all_rows]
    t = time.time()
    import hashlib
    key = hashlib.md5(json.dumps([a.n_a, a.n_b, a.n_es, a.n_valid, a.seed, cfg.to_dict()], sort_keys=True).encode()).hexdigest()[:10]
    wide_path = config.CACHE_DIR / f"wide_{key}.parquet"
    if wide_path.exists():
        lab = pl.read_parquet(wide_path)
        cand = lab.drop("s1", "m", "y")
        print(f"[retrieval] loaded cached wide candidates {wide_path.name}", flush=True)
    else:
        cand = generate_candidates(q, pools, cfg, workers=a.workers)
        lab = label_candidates(cand, q, pools, truth)
        lab.write_parquet(wide_path, compression="zstd")
    assert (lab["s1_row"].to_numpy() == cand["s1_row"].to_numpy()).all()
    t_block = time.time() - t
    print(f"[retrieval] {q.height:,} S1 -> {cand.height:,} wide candidates, {int(lab['y'].sum()):,} positives, "
          f"{t_block:.0f}s", flush=True)

    owner = np.full(q.height, -1, np.int8)                      # which fold each q row belongs to
    names = ["A", "B", "ES", "V"]
    for i, nm in enumerate(names):
        owner[np.searchsorted(all_rows, sets[nm])] = i
    pair_owner = owner[cand["s1_row"].to_numpy()]
    y_all = lab["y"].to_numpy()

    # ------------------------------------------------------------------ M1 (fold A, ES)
    def wide_features(fold):
        m = pair_owner == names.index(fold)
        c = cand.filter(pl.Series(m))
        F = pl.concat([f for _, _, f in feature_chunks(c, q, pools, tables)])
        return c, F, y_all[m]

    t = time.time()
    cA, FA, yA = wide_features("A")
    cE, FE, yE = wide_features("ES")
    feat_cols = FA.columns
    t_feat = time.time() - t
    print(f"[features] fold A {FA.height:,} rows, ES {FE.height:,} rows  {t_feat:.0f}s", flush=True)
    t = time.time()
    m1 = lgb.train(M1_PARAMS, lgb.Dataset(FA.to_numpy(), yA, feature_name=feat_cols), num_boost_round=1500,
                   valid_sets=[lgb.Dataset(FE.to_numpy(), yE)],
                   callbacks=[lgb.early_stopping(40), lgb.log_evaluation(100)])
    t_m1 = time.time() - t
    print(f"[M1] {m1.best_iteration} trees, {t_m1:.0f}s", flush=True)
    m1.save_model(str(config.MODELS_DIR / f"{a.tag}_m1.txt"))
    del FA, FE

    # ------------------------------------------------------------------ stage-1 pruning for B, ES, V
    stage = {}
    wideV = {}
    for fold in ("B", "ES", "V"):
        m = pair_owner == names.index(fold)
        c = cand.filter(pl.Series(m)); l = lab.filter(pl.Series(m))
        Fs, keepmask, p1s = [], [], []
        base = 0
        for lo, hi, mask, Fsurv, p1 in stage1_chunks(c, q, pools, tables, m1, feat_cols, a.keep):
            Fs.append(Fsurv); keepmask.append(mask); p1s.append(p1)
        mask = np.concatenate(keepmask)
        stage[fold] = (c.filter(pl.Series(mask)), l.filter(pl.Series(mask)), pl.concat(Fs))
        sv = stage[fold]
        pl.concat([sv[1].select("s1", "src", "m", "y", "s1_row").with_columns(
            pl.Series("country", q["country"].cast(pl.Utf8).gather(sv[1]["s1_row"].to_numpy()))), sv[2]],
            how="horizontal").write_parquet(config.CACHE_DIR / f"{a.tag}_{fold}_surv.parquet", compression="zstd")
        if fold == "V":
            wideV = dict(y=l["y"].to_numpy(), p1=np.concatenate(p1s), s1_row=c["s1_row"].to_numpy())
        print(f"[stage1] fold {fold}: {c.height:,} -> {int(mask.sum()):,} survivors "
              f"({mask.sum() / len(np.unique(c['s1_row'].to_numpy())):.1f}/S1)", flush=True)

    # recall of the pruned candidate set as a function of K (on V, M1 ranking)
    n_true_V = truth.join(pl.DataFrame({"s1": q["eid"].gather(np.searchsorted(all_rows, sets["V"]))}),
                          on="s1", how="semi").height
    wv = pl.DataFrame({"s": wideV["s1_row"], "y": wideV["y"], "p": wideV["p1"]}).with_columns(
        pl.col("p").rank("ordinal", descending=True).over("s").alias("rk"))
    n_qV = len(sets["V"])
    print("\nM1 pruning: candidate recall vs kept-per-S1 (validation)")
    rec_table = {}
    for k in (4, 6, 8, 10, 12, 14, 16, 20, 30, 80):
        r = wv.filter((pl.col("rk") <= k) & (pl.col("y") == 1)).height / n_true_V
        rec_table[k] = r
        print(f"   keep {k:>3}: recall {r:.4f}")

    # ------------------------------------------------------------------ M2 (fold B survivors)
    cols2 = feat_cols + M2_EXTRA
    cB, lB, FB = stage["B"]; cE2, lE2, FE2 = stage["ES"]
    t = time.time()
    m2 = lgb.train({**M2_PARAMS, "seed": a.seed},
                   lgb.Dataset(FB.select(cols2).to_numpy(), lB["y"].to_numpy(), feature_name=cols2),
                   num_boost_round=3000,
                   valid_sets=[lgb.Dataset(FE2.select(cols2).to_numpy(), lE2["y"].to_numpy())],
                   callbacks=[lgb.early_stopping(50), lgb.log_evaluation(100)])
    t_m2 = time.time() - t
    print(f"[M2] {m2.best_iteration} trees, {t_m2:.0f}s", flush=True)
    m2.save_model(str(config.MODELS_DIR / f"{a.tag}_m2.txt"))
    json.dump({"blocking": cfg.to_dict(), "feat_cols": feat_cols, "m2_cols": cols2, "keep": a.keep,
               "m1_trees": m1.best_iteration, "m2_trees": m2.best_iteration},
              open(config.MODELS_DIR / f"{a.tag}.json", "w"), indent=1)
    imp = sorted(zip(cols2, m2.feature_importance("gain")), key=lambda x: -x[1])
    (config.METRICS_DIR / f"{a.tag}_importance.txt").write_text("\n".join(f"{k}\t{v:.0f}" for k, v in imp))

    # ------------------------------------------------------------------ validation
    cV, lV, FV = stage["V"]
    qV = q[np.searchsorted(all_rows, sets["V"])]
    prob = m2.predict(FV.select(cols2).to_numpy(), num_threads=10)
    from sklearn.metrics import average_precision_score, roc_auc_score
    cand_recall = int(lV["y"].sum()) / n_true_V
    sizes = lV.group_by("s1_row").len()["len"]
    print(f"\nVALID: {n_qV:,} S1 | final candidate recall {cand_recall:.4f} | candidates/S1 "
          f"avg {lV.height / n_qV:.1f} median {sizes.median():.0f} max {sizes.max()}")
    print(f"pair AUC {roc_auc_score(lV['y'], prob):.5f}  AP {average_precision_score(lV['y'], prob):.5f}")
    lV = lV.with_columns(pl.col("s1_row"))
    res = scan_thresholds(qV, lV, prob, truth, conflict=False)
    resc = scan_thresholds(qV, lV, prob, truth, conflict=True)
    print("thr   F0.5(noconf)  F0.5(conf)  prec   rec   singleton")
    for r, rc in zip(res, resc):
        print(f"{r['threshold']:.2f}   {r['macro_f05']:.4f}       {rc['macro_f05']:.4f}     "
              f"{rc['micro_precision']:.4f} {rc['micro_recall']:.4f} {rc['singleton_score']:.4f}")
    best = max(resc, key=lambda r: r["macro_f05"])
    print(f"\nBEST: threshold {best['threshold']}  macro F0.5 {best['macro_f05']:.4f}  "
          f"(precision {best['micro_precision']:.4f} recall {best['micro_recall']:.4f})")
    json.dump({"tag": a.tag, "candidate_recall": cand_recall, "avg_candidates": lV.height / n_qV,
               "recall_vs_keep": rec_table, "scan_noconf": res, "scan_conf": resc, "best": best,
               "t_block": t_block, "t_feat": t_feat, "t_m1": t_m1, "t_m2": t_m2},
              open(config.METRICS_DIR / f"{a.tag}_valid.json", "w"), indent=1, default=float)
    # keep V survivors (+ scores) for error analysis
    lV.with_columns(pl.Series("p2", prob)).write_parquet(config.ANALYSIS_DIR / f"{a.tag}_V_survivors.parquet")
    log_experiment(experiment_id=a.tag,
                   blocking_strategy=f"idf-token(addr uni+bi, pair-keys, num-del; name uni+glued) top{a.top_k}/src -> M1 prune keep {a.keep}",
                   candidate_recall=round(cand_recall, 4), avg_candidates=round(lV.height / n_qV, 1),
                   median_candidates=float(sizes.median()), reduction_ratio="", model="LightGBM cascade M1->M2",
                   features=len(cols2), threshold=best["threshold"], precision=round(best["micro_precision"], 4),
                   recall=round(best["micro_recall"], 4), **{"macro_F0.5": round(best["macro_f05"], 4)},
                   training_time_s=round(t_block + t_feat + t_m1 + t_m2),
                   notes=f"folds A={a.n_a} B={a.n_b} ES={a.n_es} V={a.n_valid}; S1-level; conflict resolution")


if __name__ == "__main__":
    main()
