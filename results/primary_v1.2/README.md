# Primary v1.2 Results

## Authoritative Result Set

The files ending in `*_parser_amendment1.jsonl` are the authoritative final
inference results:

- [`inference/qwen3_vl_8b_instruct_parser_amendment1.jsonl`](inference/qwen3_vl_8b_instruct_parser_amendment1.jsonl)
- [`inference/internvl3_5_8b_instruct_parser_amendment1.jsonl`](inference/internvl3_5_8b_instruct_parser_amendment1.jsonl)
- [`inference/gemma3_12b_it_parser_amendment1.jsonl`](inference/gemma3_12b_it_parser_amendment1.jsonl)
- [`inference/llava_onevision2_8b_instruct_parser_amendment1.jsonl`](inference/llava_onevision2_8b_instruct_parser_amendment1.jsonl)

Each file contains 546 responses over 91 Natural Original–Edited pairs. For
every model, all 546 responses are valid, 0 are parser-invalid, and 0 have
inference errors. Final Main is Natural Primary only.

## Analysis Outputs

Seven analysis artifacts are published:

- [`analysis/primary_model_metrics_parser_amendment1.json`](analysis/primary_model_metrics_parser_amendment1.json)
  is the structured authoritative model-level metric and five-stage summary.
- [`analysis/primary_model_metrics_parser_amendment1.csv`](analysis/primary_model_metrics_parser_amendment1.csv)
  is its compact tabular form.
- [`analysis/primary_candidate_outcomes_parser_amendment1.jsonl`](analysis/primary_candidate_outcomes_parser_amendment1.jsonl)
  records six-question correctness, CRF eligibility, and first-failure stage for
  every model–candidate pair.
- [`analysis/natural_failure_structure_parser_amendment1.json`](analysis/natural_failure_structure_parser_amendment1.json)
  contains the post-hoc descriptive failure-structure analysis and its policy
  metadata.
- [`analysis/natural_color_transition_parser_amendment1.csv`](analysis/natural_color_transition_parser_amendment1.csv)
  summarizes Original-to-Edited color-response transitions and failure types.
- [`analysis/natural_edited_binding_error_breakdown_parser_amendment1.csv`](analysis/natural_edited_binding_error_breakdown_parser_amendment1.csv)
  lists Edited Binding errors and the correctness of their prerequisite
  questions.
- [`analysis/natural_cross_model_overlap_parser_amendment1.json`](analysis/natural_cross_model_overlap_parser_amendment1.json)
  reports cross-model overlap for selected behavioral outcome sets.

## Main Result

**Conditional Rebinding Failure (CRF)** is evaluated only when Original Object A
color, Original Object B color, Original binding, Edited Object A color, and
Edited Object B color are all correct. CRF occurs when Edited binding is then
wrong.

| Model | Edited Binding Errors | CRF |
|---|---:|---:|
| Qwen3-VL-8B | 7/91 | 0/48 |
| InternVL3.5-8B | 6/91 | 0/49 |
| Gemma-3-12B | 14/91 | 0/27 |
| LLaVA-OneVision-2-8B | 4/91 | 0/49 |

No residual rebinding failures were observed among prerequisite-eligible
samples under the evaluated Natural setting. Because the eligible subsets range
from 27 to 49 samples, rare residual failures cannot be ruled out. These results
also do not establish a causal distinction between perception and binding
mechanisms.

## Five-Stage Decomposition

The mutually exclusive stages are:

1. Original individual-color failure.
2. Original binding failure after both Original colors pass.
3. Edited individual-color failure after all Original prerequisites pass.
4. Conditional rebinding failure after all first five questions pass.
5. All six questions correct.

| Model | Stage 1 | Stage 2 | Stage 3 | Stage 4 | Stage 5 |
|---|---:|---:|---:|---:|---:|
| Qwen3-VL-8B | 13 | 0 | 30 | 0 | 48 |
| InternVL3.5-8B | 15 | 0 | 27 | 0 | 49 |
| Gemma-3-12B | 29 | 1 | 34 | 0 | 27 |
| LLaVA-OneVision-2-8B | 15 | 1 | 26 | 0 | 49 |

Failures appeared more frequently at prerequisite individual-color response
stages than at the residual rebinding stage. This is a behavioral decomposition,
not a claim about internal architecture or causal mechanism.

## Parser Amendment

The initial Qwen run used a strict exact-letter parser. A one-time amendment
accepted an unambiguous option identifier at the beginning of a response while
continuing to reject ambiguous choices. The amendment was gold-independent and
was applied to all four models before task-level metrics were computed.

The pre-amendment strict-parser Qwen result is provenance only. The rationale is
documented in
[`../../docs/protocols/primary_response_parser_amendment_v1.2.md`](../../docs/protocols/primary_response_parser_amendment_v1.2.md).

## Sanitized Publication Metadata

The public copies of
`analysis/primary_model_metrics_parser_amendment1.json` and
`analysis/natural_failure_structure_parser_amendment1.json` normalize local
absolute provenance paths to repository-relative paths. Metric values, counts,
stage assignments, source checksums, model IDs, candidate IDs, and query IDs
were not changed. The original research copies remain preserved outside this
public staging archive, so the two public-copy SHA256 values may differ solely
because of path normalization.
