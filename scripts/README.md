# Scripts

## Scope

The public scripts in this directory implement the completed **Natural Primary**
pipeline: candidate preparation, visual intervention, quality-control imports,
Primary question construction, frozen inference, and stage-wise analysis. The
project is closed as of August 2026.

Scripts preserved under
[`archive/development/`](../archive/development/) document positive controls,
feasibility work, editing pilots, and Controlled/Four-image development. They
are not maintained Final reproduction entry points.

## Final Pipeline

1. **Candidate extraction** —
   [`attribute_binding/extract_attribute_binding_candidates_v0_2.py`](attribute_binding/extract_attribute_binding_candidates_v0_2.py)
   extracts object–color pair candidates from the source scene metadata.
2. **Candidate pool and Main preparation** —
   [`attribute_binding/build_main_candidate_pool.py`](attribute_binding/build_main_candidate_pool.py)
   builds the review pool, while
   [`data/prepare_main_experiment_v1_1.py`](data/prepare_main_experiment_v1_1.py)
   freezes the held-out Main source and audits development overlap.
3. **Segmentation** —
   [`attribute_binding/run_segmentation_v1_1.py`](attribute_binding/run_segmentation_v1_1.py)
   creates deterministic bbox-prompted SAM masks and cutouts in frozen order.
4. **Basic Mask QC** —
   [`data/import_mask_qc_v1_1.py`](data/import_mask_qc_v1_1.py) and
   [`data/validate_mask_results_v1_1.py`](data/validate_mask_results_v1_1.py)
   import human mask decisions and validate generated or reviewed records.
5. **Semantic-purity QC** —
   [`data/build_mask_semantic_purity_review_assets_v1_1.py`](data/build_mask_semantic_purity_review_assets_v1_1.py),
   [`data/import_mask_semantic_purity_qc_v1_1.py`](data/import_mask_semantic_purity_qc_v1_1.py),
   and [`data/validate_mask_semantic_purity_qc_v1_1.py`](data/validate_mask_semantic_purity_qc_v1_1.py)
   support human review of non-target foreground contamination.
6. **Pre-edit technical eligibility** —
   [`data/derive_pre_edit_technical_eligibility_v1_1.py`](data/derive_pre_edit_technical_eligibility_v1_1.py)
   applies the frozen gate, and
   [`data/validate_pre_edit_technical_eligibility_v1_1.py`](data/validate_pre_edit_technical_eligibility_v1_1.py)
   verifies its membership and attrition contract.
7. **Natural color swap** —
   [`attribute_binding/run_color_swap_v1_1.py`](attribute_binding/run_color_swap_v1_1.py)
   creates edited cutouts and Natural Edited images from accepted masks.
8. **Natural Edit QC** —
   [`data/import_natural_edit_qc_v1_1.py`](data/import_natural_edit_qc_v1_1.py)
   imports human edit decisions; [`data/validate_color_edit_results_v1_1.py`](data/validate_color_edit_results_v1_1.py)
   validates generation and reviewed states.
9. **Primary Q/A construction** —
   [`data/build_primary_qa_v1_2.py`](data/build_primary_qa_v1_2.py) and
   [`data/validate_primary_qa_v1_2.py`](data/validate_primary_qa_v1_2.py)
   produce and validate the 91-candidate, 546-query Natural Primary domain.
10. **Pre-inference freeze** —
    [`data/freeze_primary_pre_inference_v1_2.py`](data/freeze_primary_pre_inference_v1_2.py)
    records and verifies the Primary data and question contract.
11. **Primary inference** —
    [`inference/run_primary_vlm_v1_2.py`](inference/run_primary_vlm_v1_2.py)
    runs one pinned model at a time using the shared contract in
    [`inference/primary_contract_v1_2.py`](inference/primary_contract_v1_2.py).
12. **Inference validation** —
    [`data/validate_primary_inference_results_v1_2.py`](data/validate_primary_inference_results_v1_2.py)
    checks result membership, parsing, model provenance, and frozen query order.
13. **CRF and five-stage analysis** —
    [`analysis/analyze_primary_attribute_binding_v1_2.py`](analysis/analyze_primary_attribute_binding_v1_2.py)
    computes Primary metrics and mutually exclusive first-failure stages.
14. **Natural failure-structure analysis** —
    [`analysis/analyze_natural_failure_structure_v1_2.py`](analysis/analyze_natural_failure_structure_v1_2.py)
    performs the descriptive post-hoc analysis. Its exact rerun is only partially
    supported because not all private visual lineage assets are redistributed.

[`shared/create_review_contact_sheets.py`](shared/create_review_contact_sheets.py)
is a general human-review visualization utility. Identity-field and schema
validators under `data/` preserve additional pipeline contracts.

## Important Runtime Notes

The inference runner and result validator retain a historical default pointing
to a superseded config. For Final results, always pass the authoritative config
explicitly:

```bash
python scripts/inference/run_primary_vlm_v1_2.py \
  --mode primary \
  --config configs/attribute_binding/primary_inference_v1.2_parser_amendment1.json \
  --input-manifest data/manifests/attribute_binding/primary_queries_v1.2.jsonl \
  --model-key MODEL_KEY \
  --output OUTPUT.jsonl

python scripts/data/validate_primary_inference_results_v1_2.py \
  --config configs/attribute_binding/primary_inference_v1.2_parser_amendment1.json \
  --query-manifest data/manifests/attribute_binding/primary_queries_v1.2.jsonl \
  --result-manifest OUTPUT.jsonl \
  --model-key MODEL_KEY
```

`MODEL_KEY` must be one of the keys recorded in the authoritative config. Model
access, external images, recreated Python environments, and suitable GPU compute
are required for an inference rerun.

The Primary analysis script also retains historical processed-data defaults.
Use the published layout explicitly and write reproduced outputs to a new
directory rather than overwriting the archive:

```bash
python scripts/analysis/analyze_primary_attribute_binding_v1_2.py \
  --query-manifest data/manifests/attribute_binding/primary_queries_v1.2.jsonl \
  --config configs/attribute_binding/primary_inference_v1.2_parser_amendment1.json \
  --results-dir results/primary_v1.2/inference \
  --output-dir reproduced_results/primary_v1.2
```

## Tests

- [`tests/inference/test_primary_parser_v1_2.py`](../tests/inference/test_primary_parser_v1_2.py)
  checks strict and parser-amended option handling plus prompt rendering.
- [`tests/protocol/test_main_segmentation_ordering_v1_2.py`](../tests/protocol/test_main_segmentation_ordering_v1_2.py)
  checks frozen Main ordering and development subset ordering.
- [`tests/fixtures/primary_inference_technical_smoke_v1.2.jsonl`](../tests/fixtures/primary_inference_technical_smoke_v1.2.jsonl)
  is a development-only technical smoke fixture, not a confirmatory result.

## Development Scripts

Development scripts are preserved in
[`archive/development/scripts/`](../archive/development/scripts/). Their old
paths, defaults, and assumptions may not match the public layout. In particular,
`audit_main_contract_v1_2.py` is retained there as historical protocol tooling,
not as a Final runtime dependency.
