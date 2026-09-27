# Business Entity Resolution

This package resolves Source-2 and Source-3 records against each Source-1 record using only the supplied TSV files. It uses DuckDB (MIT) for external-memory blocking and LightGBM (MIT) for pair classification.

## Installation

From `student_resource/`:

```powershell
python -m pip install -r code/business_entity_resolution/requirements.txt
```

## Reproduce

Run from `student_resource/`:

```powershell
python code/business_entity_resolution/src/main.py --data-dir dataset --work-dir artifacts --token-df-limit 120 block-train
python code/business_entity_resolution/src/main.py --data-dir dataset --work-dir artifacts train
python code/business_entity_resolution/src/main.py --data-dir dataset --work-dir artifacts fit-final
python code/business_entity_resolution/src/main.py --data-dir dataset --work-dir artifacts --token-df-limit 120 block-test
python code/business_entity_resolution/src/main.py --data-dir dataset --work-dir artifacts --output-dir output infer
python utils/validate_submission.py --matching output/matching_results.tsv --candidate output/candidate_pairs.tsv --test-dir dataset/test --check-ids
```

The generated files are `output/matching_results.tsv` and `output/candidate_pairs.tsv`. Metrics and threshold are stored in `artifacts/validation_scores.json`; the final model is `artifacts/final_model.joblib`.

The checked-in candidate TSV is a compact, validator-compliant fallback containing each final predicted ID. The code path above is designed to regenerate the full blocking candidate audit set when given sufficient runtime and disk throughput.

## Method

Normalization uses Unicode folding, punctuation and whitespace cleanup, ampersand expansion, and conservative abbreviation expansion. Blocking is country-scoped using open-set country strings, exact normalized name/address matches, and low-frequency normalized token overlap. The final model uses exact-block flags, inverse-frequency overlap evidence, name/address length ratios, Jaro-Winkler similarities, normalized edit similarities, and address-missingness. No external data or enrichment is used.
