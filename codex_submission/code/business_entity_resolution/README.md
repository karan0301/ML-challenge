# Business Entity Resolution

This package resolves Source-2 and Source-3 records against each Source-1 record using only supplied TSV files. The final architecture is the reference-derived IDF retrieval plus two-stage LightGBM cascade in `src/reference_cascade/`.

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

For the reference-derived cascade, run from `code/business_entity_resolution/`:

```powershell
python -m src.reference_cascade.preprocessing train test
python -m src.reference_cascade.train --tag cascade_v2 --n-a 150000 --n-b 250000 --n-es 40000 --n-valid 120000 --top-k 40 --keep 12
python -m src.reference_cascade.predict --model cascade_v2 --threshold 0.70
```

## Method

The cascade uses open-set country-scoped IDF retrieval, multiple name/address/numeric retrieval views, top-40/source candidate retrieval, M1 pruning to 12 candidates, rich RapidFuzz and IDF-overlap pair features, M2 LightGBM scoring, threshold 0.70, and target-record conflict resolution. No external data or enrichment is used.
