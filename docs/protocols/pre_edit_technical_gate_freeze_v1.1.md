# Pre-edit Technical Gate Freeze v1.1

## Scope

This document freezes the technical eligibility rule used after human mask QC and before color editing for Completion-25 and the main study. It is a dataset-construction gate, not an estimate of VLM performance and not a guarantee that editing will succeed.

The authoritative configuration is `configs/attribute_binding/pre_edit_technical_gate_v1.1.json`. It was frozen after Mini-10 and before Completion-25. Completion results may not be used to tune its thresholds or add rules.

## Object-side rules

Both object A and object B must pass all four rules:

1. Human semantic-purity QC reports no material non-target mask inclusion.
2. `selected_color_ratio >= 0.05`.
3. The circular distance between the two objects' observed source hues is at least `45.0` degrees.
4. `low_saturation_ratio <= 0.50`.

The selected-color ratio is the fraction of mask pixels within 55 degrees of the object's canonical original-color center and with HSV saturation at least 45 and value at least 30. Low-saturation ratio is the fraction of mask pixels with HSV saturation below 45. Observed source hue is the saturation-weighted circular mean over selected source-color pixels. Pair hue distance is the minimum circular distance between the observed source hues of A and B.

`candidate_pre_edit_eligible = object_a_eligible AND object_b_eligible`.

Human statuses `fail`, `human_review`, and `not_tested` block editing. Excluded candidates remain in the candidate-level attrition manifest. No replacement, reselection, resampling, or order change is permitted.

## Semantic-purity QC

The enhanced review asset contains seven views per candidate:

- full original image;
- object A mask overlay;
- object B mask overlay;
- object A alpha cutout on RGB(224,224,224);
- object B alpha cutout on RGB(224,224,224);
- enlarged object A bbox crop with mask boundary;
- enlarged object B bbox crop with mask boundary.

The reviewer records per-object semantic-purity status, non-target inclusion, contamination labels, reviewer, and note. Material inclusion of skin, arm, hand, neck, face, hair, an adjacent object, background region, or another non-target region is a failure. There is no automatic all-pass import mode for this review.

## Mini-10 development audit

The frozen gate was applied retrospectively in read-only mode. It retained the six candidates that passed Natural Edit QC and excluded exactly the four human-confirmed failures:

| Candidate | Trigger |
| --- | --- |
| `maincand_0011` | `selected_color_ratio_below_0_05` |
| `maincand_0020` | `non_target_mask_inclusion` |
| `maincand_0082` | `hue_distance_below_45` |
| `maincand_0179` | `low_saturation_above_0_50` |

Counts: start 10, eligible 6, excluded 4, multiple-rule exclusions 0. False exclusions among the six successful Mini candidates: 0. This agreement is developmental evidence only; the thresholds are not claimed to be generally optimal.

Authoritative audit outputs:

- `processed/attribute_binding/editing_pilot/v1.1/analysis/mini_pre_edit_gate_audit_v1.1.jsonl`
- `processed/attribute_binding/editing_pilot/v1.1/analysis/mini_pre_edit_gate_audit_summary_v1.1.json`

## Post-freeze policy

Passing the gate does not assign `edit_qc_status`, `object_identity_preservation_status`, or four-image QC. Natural Edit and Four-image human QC remain mandatory. If a fatal implementation error requires a change, all Completion outputs from this version must be discarded and the pipeline restarted under a new version; partial reuse is prohibited.
