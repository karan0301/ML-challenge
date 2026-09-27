"""Text normalization helpers for the challenge's noisy business records."""

from __future__ import annotations

import re
import unicodedata
from typing import Iterable


_WS = re.compile(r"\s+")
_NON_ALNUM = re.compile(r"[^a-z0-9]+")
_ADDRESS_REPLACEMENTS = {
    "rd": "road", "st": "street", "ave": "avenue", "blvd": "boulevard",
    "hwy": "highway", "ln": "lane", "dr": "drive", "apt": "apartment",
    "ste": "suite", "fl": "floor", "nagar": "nagar", "marg": "marg",
}
_NAME_REPLACEMENTS = {
    "corp": "corporation", "inc": "incorporated", "co": "company",
    "ltd": "limited", "pvt": "private", "intl": "international",
}


def basic_normalize(value: object) -> str:
    """Unicode-fold and tokenize text without deleting potentially useful words."""
    text = "" if value is None else str(value)
    text = unicodedata.normalize("NFKD", text).encode("ascii", "ignore").decode("ascii")
    text = text.lower().replace("&", " and ")
    return _WS.sub(" ", _NON_ALNUM.sub(" ", text)).strip()


def normalize_with_abbreviations(value: object, kind: str) -> str:
    """Return a second, conservative normalization with common abbreviations expanded."""
    tokens = basic_normalize(value).split()
    replacements = _ADDRESS_REPLACEMENTS if kind == "address" else _NAME_REPLACEMENTS
    return " ".join(replacements.get(token, token) for token in tokens)


def token_set(value: str) -> set[str]:
    return set(value.split()) if value else set()


def numeric_tokens(value: str) -> set[str]:
    return {token for token in value.split() if any(ch.isdigit() for ch in token)}


def jaccard(left: Iterable[str], right: Iterable[str]) -> float:
    left_set, right_set = set(left), set(right)
    union = left_set | right_set
    return len(left_set & right_set) / len(union) if union else 1.0
