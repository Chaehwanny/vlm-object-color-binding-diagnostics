# Main Experiment Protocol v1.2 Pre-Inference Amendment

## Status

- Amended: 2026-08-03
- Research design: frozen
- Experiment protocol: v1.2 pre-inference amendment complete
- Phase: Main Data Production - in progress
- Segmentation preflight: complete
- Next substage: actual main segmentation for 157 frozen candidates
- Amendment timing: before main segmentation assets, questions, or VLM outputs
- Frozen main source: 157 candidates in frozen_order
- Preserves all v1.1 development artifacts and historical results

## Research Question

After a VLM correctly answers the individual color questions and the binding
question for the Original image, and correctly identifies both edited object
colors, does it still fail to select the correct object-color correspondence in
the Edited image?

한국어 연구 질문:

VLM이 원본의 두 객체 색상과 원본 결합을 올바르게 처리하고, 편집 후 두
객체의 새 색상까지 정확히 확인한 경우에도 편집된 객체-색상 대응을 잘못
선택하는 잔여 실패가 존재하는가?

The study measures forced-choice response correctness, not internal model
modules. Preferred terms are edited rebinding failure and residual rebinding
failure. Change-tracking failure is reserved for an explicit paired-image task.

## Empirical Story

The frozen left/right positive control had full consistency of 13/16 for
Qwen2.5-VL, 13/16 for LLaVA-OneVision, and 15/16 for InternVL2.5. Its historical
conditions remain Baseline True, Baseline False, Irrelevant True, Image
Conflict, and Flip Control. It motivates the pipeline but is not a color main
outcome.

Color development revealed that apparent binding errors could co-occur with
incorrect Edited individual-color responses. Residual failures after the
development atomic/control gate were 0/2 in feasibility and 0/3, 0/5, and 0/5
for Qwen, LLaVA, and InternVL in the interim set. These small denominators are
motivation only. They motivate CRF as the primary conditional endpoint.

## Primary Experiment

Primary images are natural_original and natural_edited. Each receives three
independent stateless questions: Object A color (four-way), Object B color
(four-way), and binding (two-way Original Binding versus Swapped Binding).
The Primary contract is 2 images x 3 questions = 6 responses per sample.

| Image | Original Binding | Swapped Binding |
| --- | --- | --- |
| Natural Original | correct (Baseline True) | incorrect (Baseline False) |
| Natural Edited | incorrect (Image Conflict) | correct (Edit Control) |

The four names describe the conceptual truth mapping. They do not create four
Yes/No questions.

## Analysis Hierarchy

- Primary: Natural Original and Natural Edited
- Secondary matched condition: Controlled Original and Controlled Edited
- Secondary analysis: difficulty-conditioned error decomposition
- Visual-dependence diagnostic: No-image and Shuffled-image
- Post-hoc diagnostic: Ground-Truth and Self-Extracted Attribute Oracles

Controlled uses the same original and edited cutouts as Natural. It is a
context-reduced composite, not a pure background intervention.
Natural-Controlled differences are secondary and non-causal.

## Primary Eligibility

Primary Natural validity requires semantic validity, no development overlap,
segmentation success, basic and semantic-purity mask QC, the frozen pre-edit
gate, Natural Edited generation, Natural Edit and identity QC, and valid
Original/Edited questions and answers.

Natural Edit QC determines the pool eligible for Primary question
construction; primary_natural_valid is finalized only after the corresponding
Original/Edited questions and answers are generated and validated. Controlled
generation and QC define only the secondary controlled_matched_valid subset and
never determine Primary inclusion.

Use two distinct states:

- primary_natural_valid: Primary Natural pair is valid
- controlled_matched_valid: corresponding Controlled pair also passes secondary QC

## Frozen Source And Stopping Rule

Process all 157 held-out candidates in frozen_order. Record:

- N_source = 157
- N_primary_natural_valid = candidates passing the frozen Natural pipeline
- N_controlled_matched_valid = Primary candidates also passing Controlled QC

Do not replace, add, reorder, restore a target of 160, or sample from model
results. The authoritative source is
processed/attribute_binding/main_experiment/v1.1/preparation/main_input_manifest_v1.1.jsonl
with SHA-256
48e6158a20b22e3014096a31abd9e8654758e909ef7542fedef87cb92b10be98.

## Frozen Intervention Pipeline

Development-frozen segmentation, QC, pre-edit gate, color-swap algorithm, and
thresholds remain unchanged. Main technical failures are attrition.

Primary production follows: frozen source, segmentation, basic mask QC,
semantic-purity QC, frozen pre-edit gate, Natural color swap, Natural Edit and
identity QC, Natural-pair QC-valid pool, Primary question/answer generation and
validation, then finalization of N_primary_natural_valid.

Controlled branches from the Natural-pair QC-valid pool, reuses the same
cutouts and edit, and produces N_controlled_matched_valid only for secondary
analysis.

## Metrics And Error Hierarchy

Primary metrics are OBA, EBA, PBC, and CRF. CRF is the primary conditional
metric. Eligibility requires correct Original A color, Original B color,
Original binding, Edited A color, and Edited B color. Failure is incorrect
Edited binding among eligible cases. Always report failures / eligible N;
eligible N of zero is not_estimable.

Classify each Natural sample-model response into exactly one:

1. Original atomic color failure
2. Original binding failure after atomic pass
3. Edited atomic color failure after original pass
4. Conditional Rebinding Failure
5. Full paired pass

Korean reporting uses 개별 색상 확인 실패 for the evaluated outcome. Dataset
semantic and QC failures are not model error categories.

## Pre-Result Interpretation

- Scenario A: repeated CRF supports residual Edited rebinding failure after
  correct individual-color checks.
- Scenario B: CRF becomes rare after atomic conditioning, suggesting many
  apparent binding errors co-occurred with preceding color-check errors.
- Scenario C: models have different atomic-error and CRF profiles despite
  similar final binding accuracy.

Do not introduce a new favorable Main scenario after observing results.

## Claim Boundary

Describe behavioral results on a GQA-derived semantic-valid and
intervention-valid diagnostic subset. Do not claim universal VLM weakness,
internal-module causation, GQA-wide representativeness, or a pure causal
background effect.
