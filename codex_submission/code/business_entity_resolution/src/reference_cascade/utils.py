"""Small shared helpers: timing + experiment tracker."""
import csv
import time
from contextlib import contextmanager

from . import config

EXP_FIELDS = ["experiment_id", "blocking_strategy", "candidate_recall", "avg_candidates", "median_candidates",
              "reduction_ratio", "model", "features", "threshold", "precision", "recall", "macro_F0.5",
              "training_time_s", "inference_time_s", "notes"]


@contextmanager
def timer(label="", out=None):
    t = time.time()
    yield
    dt = time.time() - t
    if out is not None:
        out[label] = dt
    print(f"[{label}] {dt:.1f}s", flush=True)


def log_experiment(**row):
    """Append one row to experiments/experiments.csv (creates header on first write)."""
    p = config.EXPERIMENTS_CSV
    new = not p.exists()
    with open(p, "a", newline="") as f:
        w = csv.DictWriter(f, fieldnames=EXP_FIELDS, extrasaction="ignore")
        if new:
            w.writeheader()
        w.writerow({k: row.get(k, "") for k in EXP_FIELDS})
