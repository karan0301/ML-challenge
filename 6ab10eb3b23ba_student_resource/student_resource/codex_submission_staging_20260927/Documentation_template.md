# ML Challenge 2026: Business Entity Resolution Solution

**Team Name:** Codex  
**Submission Date:** 2026-09-27

## 1. Executive Summary

This solution uses external-memory multi-pass blocking followed by a LightGBM pair classifier. It uses only the supplied TSV data. The included candidate TSV is a compact validated fallback containing every predicted link; the reproducible code retains the full blocking-candidate generation implementation.

## 2. Methodology

### 2.1 Problem Analysis

Training contains 2,206,821 Source-1 records, 5,034,616 Source-2 records, and 5,285,603 Source-3 records. Source-2 and Source-3 have 168,967 and 175,916 missing addresses. A deterministic 17,220-Source-1 positive sample found country agreement for all links, exact normalized name agreement for 27.8%, exact normalized address agreement for 8.4%, name-token overlap for 85.5%, and address-token overlap for 95.6%. Test includes France, so country is an open-set string, never a fixed list.

### 2.2 Solution Strategy

1. Read all inputs as tab-separated files.
2. Preserve raw fields and derive normalized name/address forms.
3. Build country-scoped candidates using exact normalized fields and rare normalized tokens.
4. Train LightGBM on deterministic Source-1 hash buckets.
5. Select an all-matches-above-threshold policy using macro F0.5.

**Approach Type:** Blocking + classifier.  
**Core innovation:** Disk-backed blocking with candidate-level inverse-frequency evidence.

## 3. Candidate Generation

Blocking unions exact country/name, exact country/address, and country/token pairs where target token document frequency is at most 120. For each candidate, the pipeline stores exact-block flags, distinct rare-token count, and inverse-square-root document-frequency score. Training generated 89,586,058 candidates and covered 4,662,332 of 7,638,365 labeled links, candidate recall 0.61038. This measured recall ceiling is the principal limitation. The included candidate file is compact because the full test candidate audit export exceeded the available runtime; it contains every final match and passes the official format/ID validator.

## 4. Matching Model

The final model is LightGBM 4.6.0 binary classification, an MIT-licensed gradient-boosted tree model. Features are exact normalized name/address flags, rare-token count/weight, name and address length ratios, Jaro-Winkler similarities, normalized Levenshtein similarities, and target-address-missingness. `features.py` also contains token-set/Jaccard/RapidFuzz experiment features. The selected threshold is 0.56 from a macro F0.5 sweep.

## 5. Validation and Results

The training split is Source-1 level: hash buckets 1-5 of 100 (5%) train the classifier and bucket 0 (1%) is untouched validation. Empty truth/prediction sets score 1.0, matching the challenge singleton rule. The final validation results are:

| Experiment | Candidate recall | Macro precision | Macro recall | Macro F0.5 |
| --- | ---: | ---: | ---: | ---: |
| Evidence-only baseline | 0.61038 | 0.47768 | 0.29166 | 0.39145 |
| + Jaro-Winkler and edit similarities | 0.61038 | 0.75618 | 0.52519 | 0.65904 |

The winning configuration has false-merge rate 0.08752. False positives are usually records sharing rare location or business-name tokens but with insufficient true identity evidence. False negatives include transformations whose shared tokens exceed the blocker frequency limit, abbreviations/transliterations, and missing/noisy addresses.

## 6. Inference, Reproducibility, and Compliance

`src/main.py` runs `block-train`, `train`, `fit-final`, `block-test`, and `infer`. Full commands are in `code/business_entity_resolution/README.md`. Inference predicts all candidates over the selected threshold, preserving valid multi-match outcomes and leaving empty lists for predicted singletons. The pipeline uses fixed seed 2026, only supplied TSVs, and no business databases, web search, geocoding, APIs, or external data augmentation. DuckDB, RapidFuzz, and LightGBM are permissively licensed; the final model is far below the 8B parameter limit.
