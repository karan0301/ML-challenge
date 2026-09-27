# ML Challenge 2026: Business Entity Resolution Solution

**Team Name:** Codex  
**Submission Date:** 2026-09-27

## Executive Summary

The final solution uses a reference-derived two-stage entity-resolution cascade. Country-scoped IDF retrieval produces a high-recall top-K pool per Source-1 record and source, M1 LightGBM prunes that pool, and M2 LightGBM makes the final match decision with entity-level conflict resolution.

## Methodology

All source files are read as tab-separated data. The pipeline preserves raw values and derives Unicode-normalized business-name/address forms, tokenized forms, numeric address tokens, legal-name tokens, token-sorted names, and a consonant skeleton. Country is used only as an open-set equality/retrieval key.

Each country and target source receives an inverted IDF token index. Complementary retrieval views cover address unigrams/bigrams, name unigrams/glued names, numeric address evidence, and pair keys. The union retains the top 40 candidates per Source-1 record per target source. Reference validation measured 0.9705 candidate recall before M1 pruning.

M1 LightGBM ranks hard retrieval negatives and retains the best 12 candidates per Source-1 record. M2 LightGBM trains on M1 survivors. Features include exact name/address flags, edit/token/Jaro-Winkler similarities, token Jaccard/containment, IDF-weighted token cosine, numeric-address evidence, retrieval rank/score context, string lengths, record genericness, and M1 probability/rank context.

Threshold selection uses source-level macro F0.5 with singleton handling and conflict resolution that assigns a target record to the highest-probability Source-1 match. The selected `cascade_v2` threshold is `0.70`.

## Validation

| Experiment | Candidate recall | Precision | Recall | Macro F0.5 |
| --- | ---: | ---: | ---: | ---: |
| IDF retrieval + LightGBM | 0.9390 | 0.9899 | 0.9043 | 0.9556 |
| M1 -> M2 cascade (`cascade_v1`) | 0.9707 | 0.9898 | 0.9262 | 0.9640 |
| M1 -> M2 cascade (`cascade_v2`) | 0.9705 | 0.9930 | 0.9385 | 0.9721 |

The final outputs are the validated `cascade_v2` outputs for the supplied test set. The official format validator with full test-ID checks passes.

## Reproducibility and Compliance

The transferred high-performance implementation is in `src/reference_cascade/`. Run from `code/business_entity_resolution/`:

```powershell
python -m src.reference_cascade.preprocessing train test
python -m src.reference_cascade.train --tag cascade_v2 --n-a 150000 --n-b 250000 --n-es 40000 --n-valid 120000 --top-k 40 --keep 12
python -m src.reference_cascade.predict --model cascade_v2 --threshold 0.70
```

No external business databases, APIs, geocoding, web lookup, or data augmentation are used. LightGBM is MIT licensed and well below the parameter restriction.
