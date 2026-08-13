# Manifest Schema v1.1

Status: `PRE-INFERENCE DRAFT`

This document defines the data contracts for candidate reclassification, matched
sample construction, quality control, primary questions, answer keys, and model
outputs. JSONL is the canonical storage format. CSV exports are views for human
review only.

## 1. General Conventions

- `schema_version`: `1.1`
- `protocol_version`: `1.1_PRE_INFERENCE_DRAFT`
- IDs are immutable, unique strings.
- Paths are repository-relative POSIX paths.
- Bounding boxes use `[x, y, width, height]`, normalized to `[0, 1]`.
- Masks and images must have SHA-256 checksums before freeze.
- Missing values use JSON `null`; empty strings are not missing values.
- Controlled and Natural tracks must reference the same original and edited
  object cutout asset IDs.
- Development-exposed data must never enter the main evaluation population.

## 2. Candidate Reclassification Manifest

Canonical file:

`data/manifests/attribute_binding/candidates_v1.1.jsonl`

One row represents one of the 293 curation-pilot candidates.

### Required Fields

| Field | Type | Allowed values or meaning |
| --- | --- | --- |
| `schema_version` | string | `1.1` |
| `candidate_id` | string | Immutable candidate identifier |
| `source_dataset` | string | For example, `GQA` |
| `source_image_id` | string | Original dataset image ID |
| `source_image_path` | string | Repository-relative path |
| `development_exposed` | boolean | Must be `false` for main candidates |
| `object_a.label` | string | Human-readable object label |
| `object_a.reference` | string | Unambiguous prompt reference |
| `object_a.target_color` | string | `red`, `blue`, `green`, or `yellow` |
| `object_a.bbox` | number[4] | Normalized bounding box |
| `object_b.label` | string | Human-readable object label |
| `object_b.reference` | string | Unambiguous prompt reference |
| `object_b.target_color` | string | `red`, `blue`, `green`, or `yellow` |
| `object_b.bbox` | number[4] | Normalized bounding box |
| `semantic_status` | string | `valid`, `invalid`, `human_review` |
| `semantic_reason` | string/null | Required for `invalid` or `human_review` |
| `technical_status` | string | `directly_editable`, `segmentation_test_required`, `failed_after_test` |
| `technical_reason` | string/null | Required for `failed_after_test` |
| `technical_criterion_scores.material_suitability` | integer | Pre-edit expected-risk score: 0, 1, or 2 |
| `technical_criterion_scores.expected_mask_separability` | integer | Pre-edit expected-risk score: 0, 1, or 2 |
| `technical_criterion_scores.sufficient_visible_area` | integer | Pre-edit expected-risk score: 0, 1, or 2 |
| `technical_criterion_scores.swap_feasibility` | integer | Pre-edit expected-risk score: 0, 1, or 2 |
| `technical_criterion_scores.expected_identity_preservability` | integer | Pre-edit expected-risk score: 0, 1, or 2 |
| `mask_qc_status` | string/null | `not_tested` or null before mask QC |
| `edit_qc_status` | string/null | `not_tested` or null before edit QC |
| `object_identity_preservation_status` | string/null | `not_tested` or null before post-edit human QC |
| `difficulty` | object | Boolean metadata listed below |
| `review.review_status` | string | `pending`, `reviewed`, `adjudication_required` |
| `review.reviewer_ids` | string[] | Blinded reviewer IDs |
| `review.reviewed_at` | string/null | ISO 8601 timestamp |
| `review.notes` | string/null | Concise audit note |

### Semantic Reason Vocabulary

Only the following values can make a candidate semantically invalid:

- `label_mismatch`
- `nonunique_or_unresolvable_reference`
- `object_unidentifiable`
- `target_color_indeterminate`

Difficulty alone must not be used as a semantic-invalid reason.

### Difficulty Metadata

Every field is required and boolean:

- `small_object`
- `partial_occlusion`
- `thin_structure`
- `complex_boundary`
- `overlap`
- `crowded_background`
- `same_owner`
- `part_whole`
- `patterned_surface`
- `text_or_logo`
- `low_color_contrast`
- `distant_object`
- `atypical_color`

### Candidate-State Invariants

1. `failed_after_test` requires a recorded segmentation/edit test ID. It cannot
   be assigned from visual impression alone.
2. `semantic_status=invalid` prevents entry into the editing pilot and main
   candidate queue.
3. `semantic_status=human_review` remains unresolved and cannot enter the frozen
   main queue.
4. Previous `KEEP`, `BORDERLINE`, and `REJECT` labels are provenance only and
   cannot populate v1.1 status fields automatically.
5. Pre-edit score `2` means expected favorable, not verified success.
6. Pre-edit actual-QC statuses must be `not_tested` or null.
7. An edited image requires tested edit and identity statuses; identity `fail`
   requires a note.
8. Final matched QC cannot pass with any applicable actual-QC status other than
   `pass`.

## 3. Editing Pilot and QC Manifest

Canonical file:

`data/manifests/attribute_binding/editing_pilot_qc_v1.1.jsonl`

One row represents one candidate tested in the 30-50 sample development-only
editing pilot.

Required fields:

- `pilot_test_id`
- `candidate_id`
- `segmentation_method`
- `segmentation_model_revision`
- `edit_method`
- `edit_config_hash`
- `mask_a_path`, `mask_a_sha256`
- `mask_b_path`, `mask_b_sha256`
- `original_cutout_a_path`, `original_cutout_a_sha256`
- `original_cutout_b_path`, `original_cutout_b_sha256`
- `edited_cutout_a_path`, `edited_cutout_a_sha256`
- `edited_cutout_b_path`, `edited_cutout_b_sha256`
- `qc.target_mask_color_change`
- `qc.outside_mask_pixel_preservation`
- `qc.original_color_residue`
- `qc.mask_leakage`
- `qc.boundary_artifact`
- `mask_qc_status`: `pass`, `fail`, `human_review`, or `not_tested`
- `edit_qc_status`: `pass`, `fail`, `human_review`, or `not_tested`
- `object_identity_preservation_status`: `pass`, `fail`, `human_review`, or `not_tested`
- `object_identity_preservation_note`
- `object_identity_preservation_reviewer`
- `object_identity_preservation_timestamp`
- `qc.prompt_answer_validity`
- `qc.overall_decision`
- `qc.reviewer_ids`
- `qc.notes`

Numeric thresholds and the exact pass function remain unresolved until the
editing pilot is reviewed and the QC rule is frozen. Atypical edited colors are
not a rejection reason by themselves.

## 4. Frozen Main Source

Authoritative file:

processed/attribute_binding/main_experiment/v1.1/preparation/main_input_manifest_v1.1.jsonl

Required execution fields include candidate_id, source_image_id, source pair
identity, object labels and colors, original_image_path, semantic and technical
status, frozen_order, and overlap status.

Rules:

- exactly 157 rows at the current freeze
- frozen_order is unique and contiguous from 1 through 157
- selection_rank and selection_seed are pilot provenance and are not required
  for main execution
- membership and order are immutable
- all rows are processed; no replacement or target-160 replenishment

## 5. Main Dataset-State Manifest

A future main construction manifest must distinguish:

- primary_natural_valid: Natural Original and Natural Edited pass all applicable
  semantic, intervention, QC, and generated question-answer validity requirements
- controlled_matched_valid: a primary_natural_valid sample whose Controlled
  Original and Controlled Edited also pass secondary QC

Natural Edit and identity QC create the pool eligible for Primary question
construction. The primary_natural_valid state is assigned only after
Original/Edited question-answer generation and validation. Controlled validity
does not determine Primary validity.

Each row records identity and provenance, shared masks and original/edited
cutouts, Natural image records, optional Controlled image records, per-image QC,
primary_natural_valid, primary_exclusion_reasons, controlled_matched_valid,
controlled_exclusion_reasons, and QC/config/checksum provenance.

The legacy final_matched_valid field may remain only in historical development
artifacts where it means four-image validity. New main outputs must not use it
as the Primary inclusion state.

## 6. Question Manifests

Primary query manifest:

data/manifests/attribute_binding/primary_queries_v1.2.jsonl

There must be exactly six Primary rows per primary_natural_valid sample:

- Natural Original: Object A color, Object B color, binding
- Natural Edited: Object A color, Object B color, binding

Required fields include query_id, sample_id, image ID/path/checksum, track,
state, analysis_role, task_type, prompt version/text, option IDs/values/order,
correct option, balancing provenance, response alphabet, and answer-key
reference.

Primary invariants:

- analysis_role is primary and track is natural
- color questions contain exactly red, blue, green, and yellow
- binding choices are exactly Original Binding and Swapped Binding
- Original Binding is correct for Original
- Swapped Binding is correct for Edited
- six independent stateless rows exist per included sample
- prompt text does not reveal state or the paired image

Controlled questions use a separate secondary manifest or analysis_role =
secondary_matched_condition. Their wording and option order are yoked to the
corresponding Natural questions and are never counted among the six Primary
responses.

## 7. Diagnostic Manifest

Canonical file:

`data/manifests/attribute_binding/diagnostics_v1.1.jsonl`

Additional required fields:

- `diagnostic_family`: `pre_randomized` or `post_hoc_oracle`
- `diagnostic_type`: `no_image`, `shuffled_image`, `direct_change_selection`,
  `ground_truth_attribute_oracle`, or `self_extracted_attribute_oracle`
- `selection_rule_version`
- `selection_seed`
- `source_query_ids`
- `matched_success_control_id`
- `oracle_text_source`

Pre-randomized diagnostics are fixed before main inference. Post-hoc Oracle
diagnostics are selected after main inference only through the pre-specified
rules and are not confirmatory main results.

## 8. Model Run Manifest and Results

Run-level fields:

- `run_id`
- `model_family`
- `model_id`
- `model_revision`
- `processor_revision`
- `inference_config_path`
- `inference_config_sha256`
- `prompt_manifest_sha256`
- `sample_manifest_sha256`
- `decoding_method`
- `temperature`
- `max_new_tokens`
- `image_resolution_policy`
- `parser_version`
- `invalid_handling`
- `started_at`
- `completed_at`

One result row per query:

- `run_id`
- `query_id`
- `sample_id`
- `model_id`
- `raw_response`
- `parsed_option_id`
- `parse_status`: `valid` or `invalid`
- `invalid_reason`
- `is_correct`
- `latency_ms`
- `runtime_error`

Missing, malformed, or out-of-alphabet responses are `invalid` and counted as
incorrect in primary accuracy metrics.

## 9. Freeze Record

Before inference, create:

`data/manifests/attribute_binding/FROZEN_PRE_INFERENCE_v1.1.json`

It must include:

- protocol and schema versions
- UTC freeze timestamp
- Git commit if available
- SHA-256 for every protocol, manifest, prompt, QC rule, and inference config
- included matched-sample count
- included query count
- model panel and exact revisions
- unresolved deviations, which must be empty for freeze
- human approver names or IDs

Any post-freeze correction creates a new version and an auditable deviation
record; frozen files are never silently overwritten.
