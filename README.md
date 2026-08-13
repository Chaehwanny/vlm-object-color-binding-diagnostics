# Diagnosing Object–Color Binding Errors in Vision-Language Models

*Minimal Visual Intervention and Stage-Wise Error Decomposition*

**Status: Closed — August 2026**

This repository is a public technical archive of a completed diagnostic study of
object–color responses in vision-language models (VLMs). It preserves the final
Natural-image experiment, frozen contracts, model responses, analysis outputs,
and selected development history. It is not an active benchmark or an ongoing
model-evaluation project.

## Overview

A wrong object–color binding answer can arise in more than one way. A model may
identify both colors correctly but associate them with the wrong objects, or it
may already have failed to identify one or both object colors. Final-answer
accuracy alone does not distinguish these cases.

This study therefore asks:

> When a VLM gives a wrong object–color binding answer, did it bind the correct
> colors to the wrong objects, or had it already failed to identify one or both
> object colors?

The narrower confirmatory question is whether binding-only failures remain when
all prerequisite color and original-binding judgments are correct.

The project began with a left/right relation positive-control study, followed by
object–color feasibility and interim experiments. Development results showed
that some apparent binding errors coincided with simpler object/color response
errors. This motivated a shift from reporting only final binding accuracy to
diagnosing the first stage at which each response pattern failed. Development
materials are retained separately in [`archive/development/`](archive/development/).

## Final Study Design

The Final Main experiment is **Natural Primary only**. It contains:

- 91 Natural Original–Edited matched pairs;
- a controlled color-swap intervention applied to two identified objects;
- 6 independent multiple-choice questions per pair;
- 546 queries per model; and
- 2,184 responses across four models.

For each pair, the model answered three questions about the Original image and
three corresponding questions about the Edited image:

| State | Question 1 | Question 2 | Question 3 |
|---|---|---|---|
| Original | Object A color | Object B color | Object–color binding |
| Edited | Object A color | Object B color | Object–color binding |

The Edited image exchanged the target colors while retaining the natural scene.
Questions were evaluated independently and inference was stateless. All 2,184
records were parser-valid; there were no invalid responses or inference errors.

### Models

- `Qwen/Qwen3-VL-8B-Instruct`
- `OpenGVLab/InternVL3_5-8B-Instruct`
- `google/gemma-3-12b-it`
- `lmms-lab-encoder/LLaVA-OneVision-2-8B-Instruct`

Pinned revisions and decoding settings are recorded in
[`configs/attribute_binding/primary_inference_v1.2_parser_amendment1.json`](configs/attribute_binding/primary_inference_v1.2_parser_amendment1.json)
and the corresponding [`frozen/`](frozen/) metadata.

### Scope Exclusions

Controlled images were explored during development but were not executed as
part of Final Main. No-image, Shuffled-image, and Oracle conditions were also not
executed. Controlled/Four-image files under `archive/development/` are
development artifacts, not confirmatory Main results.

## Evaluation Protocol

### Conditional Rebinding Failure

**Conditional Rebinding Failure (CRF)** isolates a residual failure pattern. A
sample is CRF-eligible only when the model correctly answers the first five
questions:

1. Original Object A color;
2. Original Object B color;
3. Original binding;
4. Edited Object A color; and
5. Edited Object B color.

CRF occurs when the final Edited Binding answer is wrong despite those five
prerequisite judgments being correct. This is a behavioral conditional metric,
not an assertion that every eligible error would represent a unique internal or
"true" binding mechanism.

### Five-Stage Decomposition

Each candidate is assigned to one mutually exclusive first-failure stage:

1. **Original individual-color failure** — at least one Original color answer is
   wrong.
2. **Original binding failure after Original colors pass** — both Original
   colors are correct, but Original binding is wrong.
3. **Edited individual-color failure after Original prerequisites pass** — the
   Original colors and binding are correct, but at least one Edited color answer
   is wrong.
4. **Conditional rebinding failure** — the first five answers are correct, but
   Edited binding is wrong.
5. **All six questions correct** — the full Original–Edited pair passes.

## Main Results

| Model | Edited Binding Errors | CRF |
|---|---:|---:|
| Qwen3-VL-8B | 7/91 | 0/48 |
| InternVL3.5-8B | 6/91 | 0/49 |
| Gemma-3-12B | 14/91 | 0/27 |
| LLaVA-OneVision-2-8B | 4/91 | 0/49 |

No residual rebinding failures were observed among prerequisite-eligible
samples under the evaluated Natural setting. The eligible denominators ranged
from 27 to 49, so these observations do not rule out rare residual failures or
generalize beyond the evaluated intervention and sample domain.

### Stage-Wise Error Counts

| Model | Stage 1 | Stage 2 | Stage 3 | Stage 4 | Stage 5 |
|---|---:|---:|---:|---:|---:|
| Qwen3-VL-8B | 13 | 0 | 30 | 0 | 48 |
| InternVL3.5-8B | 15 | 0 | 27 | 0 | 49 |
| Gemma-3-12B | 29 | 1 | 34 | 0 | 27 |
| LLaVA-OneVision-2-8B | 15 | 1 | 26 | 0 | 49 |

Failures were observed more frequently at prerequisite individual-color
response stages than at the residual rebinding stage. This describes where
errors first appeared behaviorally; it does not establish a causal distinction
between perception and binding mechanisms, nor does it show that editing caused
the color-response errors.

Full model-level metrics and candidate outcomes are available in
[`results/primary_v1.2/analysis/`](results/primary_v1.2/analysis/).

## What We Learned

Final-answer accuracy can obscure the stage at which an error first appears. A
diagnostic evaluation benefits from:

- separating prerequisite object/color judgments from the final relational
  answer;
- assigning each sample to a mutually exclusive first-failure stage;
- validating the visual intervention before model evaluation; and
- keeping semantic validity, technical editability, and observed model
  difficulty as separate concepts.

The contribution of this archive is therefore methodological as well as
empirical: it records a pipeline for locating behavioral failures without
claiming to identify their internal architectural cause.

## Parser Amendment

Files ending in `*_parser_amendment1.jsonl` are the authoritative final
inference results. The initial strict parser accepted only exact option-letter
responses and rejected many otherwise explicit leading choices. A one-time,
gold-independent parser amendment was frozen before metric computation and
applied uniformly to all four models.

The pre-amendment strict-parser Qwen result is retained only as provenance. The
rationale and unchanged experimental contract are documented in
[`docs/protocols/primary_response_parser_amendment_v1.2.md`](docs/protocols/primary_response_parser_amendment_v1.2.md).

## Repository Structure

| Path | Contents |
|---|---|
| [`configs/`](configs/) | Final inference, editing, segmentation, and analysis configurations |
| [`schemas/`](schemas/) | Machine-readable manifest and result contracts |
| [`scripts/`](scripts/) | Final data, intervention, inference, validation, and analysis entry points |
| [`tests/`](tests/) | Parser and metric contract tests |
| [`docs/`](docs/) | Governing protocol and supporting specifications |
| [`data/`](data/) | Final 91-candidate status and 546-query manifests |
| [`requirements/`](requirements/) | Recorded Python environments for the model stacks |
| [`frozen/`](frozen/) | Pre-inference and parser-amended inference-stack freeze metadata |
| [`processed/`](processed/) | Public lineage and supporting machine-readable records |
| [`results/primary_v1.2/`](results/primary_v1.2/) | Authoritative amended inference and final analysis outputs |
| [`archive/development/`](archive/development/) | Positive controls, feasibility work, and pipeline-development history |

Historical documents are preserved as contemporaneous research records and may
contain status language such as "in progress." The repository-level project
status at the top of this README is authoritative.

## Reproducibility

This archive supports different levels of reproduction rather than claiming
complete end-to-end reproducibility.

**Supported from published artifacts**

- inspection of final manifests, responses, metrics, and freeze metadata;
- parser unit validation;
- Primary pre-inference freeze verification; and
- reproduction of Primary CRF and five-stage metrics from published query and
  response artifacts.

**Requires environment recreation**

- exact inference-stack verification using the recorded model revisions and
  Python environments.

**Requires external data and compute environments**

- rerunning inference for all four models. Gemma additionally requires
  appropriate Hugging Face model access.

**Partially supported**

- reproduction of the Natural post-hoc failure analysis. Final outputs are
  published, but not every private visual lineage artifact is redistributed.

**Not publicly reproducible end to end**

- the complete segmentation and color-editing pipeline. GQA/Visual Genome
  images, masks, cutouts, edited images, contact sheets, and model weights are
  not included because of size, licensing, and research-record constraints.

The recorded environments are
[`requirements/primary-vlm-tf457.txt`](requirements/primary-vlm-tf457.txt) and
[`requirements/primary-vlm-llava.txt`](requirements/primary-vlm-llava.txt).
They document the executed stacks but are not presented as complete
hardware/CUDA lockfiles.

## Development Archive

The development archive preserves the path from the left/right positive control
through color feasibility, interim, smoke, mini, and completion stages. These
materials explain how the diagnostic design and quality-control workflow were
developed. They should not be combined with the 91-pair Final Main results or
treated as additional confirmatory samples.

## Limitations

- Final Main contains 91 Natural pairs from one object–color intervention
  setting.
- CRF-eligible subsets contain only 27–49 samples depending on the model.
- The behavioral decomposition does not identify causal perception or binding
  mechanisms.
- Controlled, No-image, Shuffled-image, and Oracle conditions were not executed
  in Final Main.
- Raw source and generated visual assets are not redistributed.

## Project Closure

This project was closed in August 2026 after the Primary Natural analysis
answered the scoped research question. The repository is maintained as a
technical research archive rather than an active benchmark or ongoing VLM
project.
