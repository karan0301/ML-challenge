"""Pair features used by the MIT-licensed LightGBM classifier."""

from __future__ import annotations

import numpy as np
import pandas as pd
from rapidfuzz import fuzz

from preprocessing import basic_normalize, jaccard, normalize_with_abbreviations, numeric_tokens, token_set


FEATURE_COLUMNS = [
    "name_exact", "name_compact_exact", "name_ratio", "name_token_sort", "name_token_set",
    "name_jaccard", "name_overlap_min", "name_length_ratio", "address_exact", "address_ratio",
    "address_token_sort", "address_token_set", "address_jaccard", "address_overlap_min",
    "address_length_ratio", "numeric_jaccard", "country_equal", "target_address_missing",
]


def make_features(pairs: pd.DataFrame) -> pd.DataFrame:
    """Create deterministic similarity features for a batch of candidate pairs."""
    rows = []
    for pair in pairs.itertuples(index=False):
        sn = basic_normalize(pair.s1_name)
        tn = basic_normalize(pair.target_name)
        sa = normalize_with_abbreviations(pair.s1_address, "address")
        ta = normalize_with_abbreviations(pair.target_address, "address")
        sn_alt = normalize_with_abbreviations(pair.s1_name, "name")
        tn_alt = normalize_with_abbreviations(pair.target_name, "name")
        sn_tokens, tn_tokens = token_set(sn_alt), token_set(tn_alt)
        sa_tokens, ta_tokens = token_set(sa), token_set(ta)
        name_common = len(sn_tokens & tn_tokens)
        address_common = len(sa_tokens & ta_tokens)
        rows.append((
            float(sn_alt == tn_alt), float(sn_alt.replace(" ", "") == tn_alt.replace(" ", "")),
            fuzz.ratio(sn_alt, tn_alt) / 100.0, fuzz.token_sort_ratio(sn_alt, tn_alt) / 100.0,
            fuzz.token_set_ratio(sn_alt, tn_alt) / 100.0, jaccard(sn_tokens, tn_tokens),
            name_common / max(1, min(len(sn_tokens), len(tn_tokens))),
            min(len(sn_alt), len(tn_alt)) / max(1, max(len(sn_alt), len(tn_alt))),
            float(sa == ta and bool(sa)), fuzz.ratio(sa, ta) / 100.0,
            fuzz.token_sort_ratio(sa, ta) / 100.0, fuzz.token_set_ratio(sa, ta) / 100.0,
            jaccard(sa_tokens, ta_tokens), address_common / max(1, min(len(sa_tokens), len(ta_tokens))),
            min(len(sa), len(ta)) / max(1, max(len(sa), len(ta))),
            jaccard(numeric_tokens(sa), numeric_tokens(ta)), float(pair.s1_country == pair.target_country),
            float(not ta),
        ))
    return pd.DataFrame(rows, columns=FEATURE_COLUMNS, dtype=np.float32)
