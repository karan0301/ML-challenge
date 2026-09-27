"""Project-wide paths and constants."""
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]          # business_entity_resolution/
# Organizer data lives next to this package by default; override with BER_DATA_DIR.
import os
DATA_DIR = Path(os.environ.get(
    "BER_DATA_DIR", ROOT.parent / "student_resource" / "dataset"))
TRAIN_DIR = DATA_DIR / "train"
TEST_DIR = DATA_DIR / "test"

ARTIFACTS = ROOT / "artifacts"
MODELS_DIR = ARTIFACTS / "models"
METRICS_DIR = ARTIFACTS / "metrics"
ANALYSIS_DIR = ARTIFACTS / "analysis"
CACHE_DIR = ARTIFACTS / "cache"          # parquet caches of normalised tables (git-ignored)
OUTPUT_DIR = ROOT / "output"
EXPERIMENTS_CSV = ROOT / "experiments" / "experiments.csv"

for _d in (MODELS_DIR, METRICS_DIR, ANALYSIS_DIR, CACHE_DIR, OUTPUT_DIR):
    _d.mkdir(parents=True, exist_ok=True)

SEED = 42
BETA = 0.5
