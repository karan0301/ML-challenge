"""Raw TSV -> normalised parquet cache (one file per split/source).

    python -m src.preprocessing train test
"""
import sys
import time

import polars as pl

from . import config
from .data_loader import read_source
from .normalization import normalise_frame

CHUNK = 1_000_000


def cache_path(split: str, i: int):
    return config.CACHE_DIR / f"{split}_s{i}.parquet"


def build_cache(split: str, sources=(1, 2, 3)):
    d = config.TRAIN_DIR if split == "train" else config.TEST_DIR
    for i in sources:
        t = time.time()
        raw = read_source(d / f"{split}_source{i}.tsv")
        parts = []
        for off in range(0, raw.height, CHUNK):
            n = normalise_frame(raw.slice(off, CHUNK))
            n = n.with_columns(pl.col("entity_id").str.slice(3).cast(pl.UInt32).alias("eid"),
                               pl.col("country").cast(pl.Categorical)).drop("entity_id")
            parts.append(n)
        out = pl.concat(parts)
        out.write_parquet(cache_path(split, i), compression="zstd")
        print(f"{split} s{i}: {out.height:,} rows, {time.time() - t:.0f}s", flush=True)


if __name__ == "__main__":
    for s in sys.argv[1:] or ["train", "test"]:
        build_cache(s)
