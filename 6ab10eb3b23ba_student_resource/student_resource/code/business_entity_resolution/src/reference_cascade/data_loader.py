"""TSV loading. Everything is tab-separated; polars is used because the sources
are 5M+ rows each. All columns are read as strings (no NA inference, no quoting)."""
from pathlib import Path

import polars as pl

from . import config

_SCHEMA = {"entity_id": pl.Utf8, "business_name": pl.Utf8,
           "business_address": pl.Utf8, "country": pl.Utf8}


def read_source(path: Path) -> pl.DataFrame:
    """Read one source TSV. Empty fields become empty strings, not nulls, so that
    'missing address' is an explicit, countable state."""
    return pl.read_csv(path, separator="\t", schema_overrides=_SCHEMA,
                       quote_char=None, null_values=[], infer_schema=False,
                       missing_utf8_is_empty_string=True)


def load_split(split: str) -> dict:
    """split in {'train','test'} -> {'s1','s2','s3'} DataFrames (+ 'gt' for train)."""
    d = config.TRAIN_DIR if split == "train" else config.TEST_DIR
    out = {f"s{i}": read_source(d / f"{split}_source{i}.tsv") for i in (1, 2, 3)}
    if split == "train":
        out["gt"] = read_ground_truth(d / "train_ground_truth.tsv")
    return out


def read_ground_truth(path: Path) -> pl.DataFrame:
    return pl.read_csv(path, separator="\t", quote_char=None, infer_schema=False,
                       missing_utf8_is_empty_string=True)


def gt_pairs(gt: pl.DataFrame) -> pl.DataFrame:
    """Explode the ground truth into (source1_entity_id, matched_id) rows."""
    return (gt.filter(pl.col("matched_entity_ids") != "")
              .with_columns(pl.col("matched_entity_ids").str.split(","))
              .explode("matched_entity_ids")
              .rename({"matched_entity_ids": "matched_id"}))
