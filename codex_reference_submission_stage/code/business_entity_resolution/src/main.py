"""End-to-end runner for the Amazon ML Challenge entity-resolution solution."""

from __future__ import annotations

import argparse
import json
import sys
from collections import defaultdict
from pathlib import Path

import duckdb
import joblib
import numpy as np
import pandas as pd
from lightgbm import LGBMClassifier

from blocking import build_candidates
from evaluation import macro_metrics, truth_from_rows
from features import FEATURE_COLUMNS, make_features

SEED = 2026
TRAIN_BUCKETS = "1,2,3,4,5"
VALID_BUCKET = 0
MODEL_FEATURE_COLUMNS = ["exact_name_block", "exact_address_block", "rare_shared_tokens", "rare_score", "name_length_ratio", "address_length_ratio", "name_jaro_winkler", "address_jaro_winkler", "name_edit_similarity", "address_edit_similarity", "target_address_missing"]


def connect(db_path: Path):
    con = duckdb.connect(str(db_path), read_only=False)
    con.execute("SET memory_limit='11GB'")
    con.execute("SET threads=6")
    return con


def add_training_labels(con, ground_truth: Path) -> None:
    path = str(ground_truth.resolve()).replace("'", "''")
    con.execute(
        "CREATE OR REPLACE TABLE labels AS "
        "SELECT source1_entity_id, target_id AS candidate_entity_id "
        "FROM read_csv(?, delim='\\t', header=true, all_varchar=true), "
        "UNNEST(string_split(matched_entity_ids, ',')) AS u(target_id) "
        "WHERE matched_entity_ids <> ''",
        [path],
    )
    con.execute("CREATE INDEX labels_idx ON labels(source1_entity_id, candidate_entity_id)")


def candidate_recall(con) -> dict[str, float]:
    positive = con.execute("SELECT count(*) FROM labels").fetchone()[0]
    covered = con.execute(
        "SELECT count(*) FROM labels l JOIN candidates c USING(source1_entity_id, candidate_entity_id)"
    ).fetchone()[0]
    total_pairs = con.execute("SELECT count(*) FROM candidates").fetchone()[0]
    return {"positive_links": int(positive), "covered_positive_links": int(covered),
            "candidate_recall": covered / positive, "candidate_pairs": int(total_pairs)}


def _pair_query(bucket_clause: str, labeled: bool) -> str:
    label = "COALESCE(CASE WHEN l.candidate_entity_id IS NULL THEN 0 ELSE 1 END, 0) AS label," if labeled else ""
    join = "LEFT JOIN labels l ON c.source1_entity_id=l.source1_entity_id AND c.candidate_entity_id=l.candidate_entity_id" if labeled else ""
    return (
        "SELECT c.source1_entity_id, c.candidate_entity_id, " + label +
        "c.exact_name_block, c.exact_address_block, c.rare_shared_tokens, c.rare_score, "
        "least(length(s.name_norm), length(t.name_norm))::FLOAT / greatest(1, greatest(length(s.name_norm), length(t.name_norm))) AS name_length_ratio, "
        "least(length(s.address_norm), length(t.address_norm))::FLOAT / greatest(1, greatest(length(s.address_norm), length(t.address_norm))) AS address_length_ratio, "
        "jaro_winkler_similarity(s.name_norm, t.name_norm) AS name_jaro_winkler, "
        "jaro_winkler_similarity(s.address_norm, t.address_norm) AS address_jaro_winkler, "
        "1.0 - levenshtein(s.name_norm, t.name_norm)::FLOAT / greatest(1, greatest(length(s.name_norm), length(t.name_norm))) AS name_edit_similarity, "
        "1.0 - levenshtein(s.address_norm, t.address_norm)::FLOAT / greatest(1, greatest(length(s.address_norm), length(t.address_norm))) AS address_edit_similarity, "
        "CASE WHEN t.address_norm='' THEN 1.0 ELSE 0.0 END AS target_address_missing "
        "FROM candidates c JOIN source1 s ON c.source1_entity_id=s.entity_id "
        "JOIN targets t ON c.candidate_entity_id=t.entity_id " + join + " " + bucket_clause
    )


def features_for_query(con, query: str, output_path: Path, batch_size: int = 250_000) -> pd.DataFrame:
    """Extract features incrementally and persist one parquet artifact for reproducibility."""
    output_path.parent.mkdir(parents=True, exist_ok=True)
    batches = []
    reader = con.execute(query).fetch_record_batch(rows_per_batch=batch_size)
    for batch in reader:
        raw = batch.to_pandas()
        base = raw[["source1_entity_id", "candidate_entity_id", *MODEL_FEATURE_COLUMNS]].copy()
        base[MODEL_FEATURE_COLUMNS] = base[MODEL_FEATURE_COLUMNS].astype(np.float32)
        if "label" in raw:
            base["label"] = raw["label"].astype(np.int8)
        batches.append(base)
    if not batches:
        return pd.DataFrame(columns=["source1_entity_id", "candidate_entity_id", *MODEL_FEATURE_COLUMNS])
    frame = pd.concat(batches, ignore_index=True)
    frame.to_parquet(output_path, index=False)
    return frame


def train_and_validate(data_dir: Path, work_dir: Path, db_path: Path) -> dict:
    con = connect(db_path)
    add_training_labels(con, data_dir / "train" / "train_ground_truth.tsv")
    blocking_stats = candidate_recall(con)
    train_query = _pair_query(f"WHERE hash(c.source1_entity_id) % 100 IN ({TRAIN_BUCKETS})", True)
    valid_query = _pair_query(f"WHERE hash(c.source1_entity_id) % 100 = {VALID_BUCKET}", True)
    train = features_for_query(con, train_query, work_dir / "train_features.parquet")
    valid = features_for_query(con, valid_query, work_dir / "valid_features.parquet")
    con.close()

    model = LGBMClassifier(
        objective="binary", n_estimators=500, learning_rate=0.06, num_leaves=48,
        min_child_samples=80, subsample=0.85, colsample_bytree=0.90, reg_lambda=2.0,
        random_state=SEED, n_jobs=6, verbosity=-1,
    )
    model.fit(train[MODEL_FEATURE_COLUMNS], train.label)
    valid["score"] = model.predict_proba(valid[MODEL_FEATURE_COLUMNS])[:, 1]
    con = connect(db_path)
    valid_ids = [r[0] for r in con.execute(f"SELECT entity_id FROM source1 WHERE hash(entity_id) % 100 = {VALID_BUCKET}").fetchall()]
    truth_rows = con.execute(
        f"SELECT source1_entity_id, candidate_entity_id FROM labels WHERE hash(source1_entity_id) % 100 = {VALID_BUCKET}"
    ).fetchall()
    truth = truth_from_rows(truth_rows)
    con.close()
    best = None
    experiments = []
    for threshold in np.arange(0.50, 0.981, 0.02):
        selected = valid[valid.score >= threshold]
        predicted = defaultdict(set)
        for row in selected[["source1_entity_id", "candidate_entity_id"]].itertuples(index=False):
            predicted[row.source1_entity_id].add(row.candidate_entity_id)
        metrics = macro_metrics(dict(predicted), truth, valid_ids)
        metrics["threshold"] = float(round(threshold, 2))
        experiments.append(metrics)
        if best is None or metrics["macro_f0_5"] > best["macro_f0_5"]:
            best = metrics
    joblib.dump(model, work_dir / "model.joblib")
    with (work_dir / "validation_scores.json").open("w", encoding="utf-8") as handle:
        json.dump({"blocking": blocking_stats, "thresholds": experiments, "best": best}, handle, indent=2)
    return {"blocking": blocking_stats, "best": best, "train_pairs": len(train), "valid_pairs": len(valid)}


def fit_final_model(data_dir: Path, work_dir: Path, db_path: Path) -> None:
    """Refit on deterministic 60% training buckets after threshold selection."""
    con = connect(db_path)
    train_query = _pair_query(f"WHERE hash(c.source1_entity_id) % 100 IN ({TRAIN_BUCKETS})", True)
    final = features_for_query(con, train_query, work_dir / "final_train_features.parquet")
    con.close()
    model = LGBMClassifier(
        objective="binary", n_estimators=500, learning_rate=0.06, num_leaves=48,
        min_child_samples=80, subsample=0.85, colsample_bytree=0.90, reg_lambda=2.0,
        random_state=SEED, n_jobs=6, verbosity=-1,
    )
    model.fit(final[MODEL_FEATURE_COLUMNS], final.label)
    joblib.dump(model, work_dir / "final_model.joblib")


def infer(data_dir: Path, work_dir: Path, output_dir: Path, db_path: Path, threshold: float) -> dict:
    con = connect(db_path)
    model = joblib.load(work_dir / "final_model.joblib")
    con.execute("DROP TABLE IF EXISTS predictions")
    con.execute("CREATE TABLE predictions(source1_entity_id VARCHAR, candidate_entity_id VARCHAR)")
    query = _pair_query("", False)
    reader = con.execute(query).fetch_record_batch(rows_per_batch=500_000)
    count = 0
    for batch in reader:
        raw = batch.to_pandas()
        raw[MODEL_FEATURE_COLUMNS] = raw[MODEL_FEATURE_COLUMNS].astype(np.float32)
        scores = model.predict_proba(raw[MODEL_FEATURE_COLUMNS])[:, 1]
        keep = raw.loc[scores >= threshold, ["source1_entity_id", "candidate_entity_id"]]
        if not keep.empty:
            con.register("prediction_batch", keep)
            con.execute("INSERT INTO predictions SELECT source1_entity_id, candidate_entity_id FROM prediction_batch")
            con.unregister("prediction_batch")
        count += len(raw)
    output_dir.mkdir(parents=True, exist_ok=True)
    match_path = str((output_dir / "matching_results.tsv").resolve()).replace("'", "''")
    candidate_path = str((output_dir / "candidate_pairs.tsv").resolve()).replace("'", "''")
    con.execute(
        "COPY (SELECT s.entity_id AS source1_entity_id, "
        "coalesce(string_agg(DISTINCT p.candidate_entity_id, ',' ORDER BY p.candidate_entity_id), '') AS matched_entity_ids "
        "FROM source1 s LEFT JOIN predictions p ON s.entity_id=p.source1_entity_id "
        "GROUP BY s.entity_id ORDER BY s.entity_id) "
        f"TO '{match_path}' (HEADER, DELIMITER '\\t')"
    )
    con.execute(
        "COPY (SELECT s.entity_id AS source1_entity_id, "
        "coalesce(string_agg(DISTINCT c.candidate_entity_id, ',' ORDER BY c.candidate_entity_id), '') AS candidate_entity_ids "
        "FROM source1 s LEFT JOIN candidates c ON s.entity_id=c.source1_entity_id "
        "GROUP BY s.entity_id ORDER BY s.entity_id) "
        f"TO '{candidate_path}' (HEADER, DELIMITER '\\t')"
    )
    stats = con.execute(
        "SELECT (SELECT count(*) FROM source1), (SELECT count(*) FROM predictions), "
        "(SELECT count(*) FROM source1 s LEFT JOIN predictions p ON s.entity_id=p.source1_entity_id GROUP BY s.entity_id HAVING count(p.candidate_entity_id)=0)"
    ).fetchone()
    con.close()
    return {"test_entities": int(stats[0]), "candidate_pairs": count,
            "predicted_matches": int(stats[1]), "predicted_singletons": int(stats[2])}


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-dir", type=Path, default=Path("dataset"))
    parser.add_argument("--work-dir", type=Path, default=Path("artifacts"))
    parser.add_argument("--output-dir", type=Path, default=Path("output"))
    parser.add_argument("--token-df-limit", type=int, default=120)
    parser.add_argument("command", choices=["block-train", "train", "fit-final", "block-test", "infer", "all"])
    args = parser.parse_args()
    args.work_dir.mkdir(parents=True, exist_ok=True)
    train_db = args.work_dir / "train_blocking.duckdb"
    test_db = args.work_dir / "test_blocking.duckdb"
    if args.command in {"block-train", "all"}:
        train_db = build_candidates(args.data_dir, "train", args.work_dir, args.token_df_limit)
        print(f"TRAIN_BLOCKING_DB={train_db}")
    if args.command in {"train", "all"}:
        result = train_and_validate(args.data_dir, args.work_dir, train_db)
        print(json.dumps(result, indent=2))
    if args.command in {"fit-final", "all"}:
        fit_final_model(args.data_dir, args.work_dir, train_db)
    if args.command in {"block-test", "all"}:
        test_db = build_candidates(args.data_dir, "test", args.work_dir, args.token_df_limit)
        print(f"TEST_BLOCKING_DB={test_db}")
    if args.command in {"infer", "all"}:
        scores = json.loads((args.work_dir / "validation_scores.json").read_text(encoding="utf-8"))
        result = infer(args.data_dir, args.work_dir, args.output_dir, test_db, scores["best"]["threshold"])
        (args.work_dir / "inference_stats.json").write_text(json.dumps(result, indent=2), encoding="utf-8")
        print(json.dumps(result, indent=2))
    return 0


if __name__ == "__main__":
    sys.exit(main())
