# Diagnosing Object–Color Binding Errors in Vision-Language Models

> **Project Status: Closed — August 2026**
> This repository archives a completed research project on behavioral error diagnosis in Vision-Language Models (VLMs).

## Overview

Vision-Language Models can make mistakes on questions that require matching visual attributes to the correct objects.

However, a wrong answer to an object–color binding question does not necessarily mean that the model failed at **binding** itself.

For example, if a model incorrectly answers a question about a *blue car and red truck*, it may have:

* misidentified the color of the car,
* misidentified the color of the truck,
* failed to recognize one of the objects,
* or correctly recognized both colors but incorrectly associated them with the objects.

This project investigates a simple question:

> **When a VLM appears to make an object–color binding error, did the failure actually occur at the binding stage, or had the model already failed to recognize the individual object colors?**

To study this, we used **minimal visual interventions** that swap the colors of two objects while preserving their identity, position, and surrounding scene as much as possible.

---

## Research Progression

The project evolved through several stages:

```text
Visual relation change
        ↓
Left/right positive control
        ↓
Object–color intervention
        ↓
Atomic color checks
        ↓
Failure-stage analysis
        ↓
Conditional Rebinding Failure (CRF)
        ↓
Natural Primary evaluation
```

The original goal was to test whether VLMs fail to update their judgments after visual relations are changed.

A left/right positive-control experiment showed that models generally handled simple horizontal flips well when they had correctly understood the original relation.

The study then moved to the more fine-grained problem of **object–color binding**.

During pilot experiments, apparent binding errors were observed. However, closer inspection showed that some of these errors were already accompanied by failures in individual color judgments.

This observation changed the focus of the project from:

> **Does the model fail at object–color binding?**

to:

> **Where does the observed error first appear?**

---

## Minimal Object–Color Intervention

For each example, two objects with different colors were selected.

Example:

```text
Original
Red car + Blue truck

        ↓ color swap

Edited
Blue car + Red truck
```

The goal was to change only the object–color correspondence while preserving the rest of the visual scene as much as possible.

The final editing pipeline consisted of:

```text
Object Segmentation
        ↓
Mask Quality Check
        ↓
Semantic-Purity Check
        ↓
Editability Check
        ↓
Color Swap
        ↓
Human Edit Quality Check
```

Development samples were kept separate from the final evaluation set.

---

## Final Dataset

The candidate pool was progressively reviewed and filtered:

```text
2,783 raw candidates
        ↓
293 manually reviewed candidates
        ↓
197 semantically valid candidates
        ↓
40 development-source images excluded
        ↓
157 frozen held-out candidates
        ↓
91 final Natural Original–Edited pairs
```

The final evaluation dataset contained:

* **91 Original–Edited image pairs**
* **6 independent questions per pair**
* **4 VLMs**
* **546 responses per model**
* **2,184 total model responses**

No invalid, missing, duplicated, or inference-error responses were observed in the final evaluation.

---

## Evaluated Models

The final Natural Primary evaluation used:

* `Qwen/Qwen3-VL-8B-Instruct`
* `OpenGVLab/InternVL3_5-8B-Instruct`
* `google/gemma-3-12b-it`
* `lmms-lab-encoder/LLaVA-OneVision-2-8B-Instruct`

---

## Evaluation Protocol

Each Original–Edited pair was evaluated using six independent questions.

### Original Image

1. Color of Object A
2. Color of Object B
3. Correct object–color binding

### Edited Image

4. New color of Object A
5. New color of Object B
6. Correct object–color binding after the color swap

The two individual color questions were evaluated independently from the binding question.

---

## Conditional Rebinding Failure (CRF)

A wrong answer to the final binding question is **not automatically classified as a binding-specific failure**.

A sample becomes eligible for CRF analysis only when the model correctly answers all five preceding questions:

```text
Original Object A color  ✓
Original Object B color  ✓
Original binding         ✓
Edited Object A color    ✓
Edited Object B color    ✓
                         ↓
Edited binding           ✗  → CRF
```

Therefore:

> **CRF measures whether an additional binding failure remains after all prerequisite judgments have been answered correctly.**

---

## Main Results

### Binding Questions Only

If we look only at the object–color binding questions, all four models made errors.

| Model                | Original Binding | Edited Binding | Both Correct |
| -------------------- | ---------------: | -------------: | -----------: |
| Qwen3-VL-8B          |            90/91 |          84/91 |        83/91 |
| InternVL3.5-8B       |            90/91 |          85/91 |        85/91 |
| Gemma-3-12B          |            87/91 |          77/91 |        73/91 |
| LLaVA-OneVision-2-8B |            90/91 |          87/91 |        86/91 |

There were **31 Edited Binding errors** across the four models.

If the analysis stopped here, it would be natural to conclude that the models showed object–color rebinding failures.

### After Prerequisite Conditioning

When individual color judgments and the Original binding judgment were taken into account:

| Model                | CRF-Eligible Samples |      CRF |
| -------------------- | -------------------: | -------: |
| Qwen3-VL-8B          |                   48 | **0/48** |
| InternVL3.5-8B       |                   49 | **0/49** |
| Gemma-3-12B          |                   27 | **0/27** |
| LLaVA-OneVision-2-8B |                   49 | **0/49** |

All 31 Edited Binding errors occurred in samples that did **not** satisfy the CRF prerequisite conditions.

In other words:

> **No additional Edited Binding-only failure was observed after all prerequisite judgments had been answered correctly in the evaluated Natural setting.**

This does **not** mean that VLMs have no object–attribute binding problems.

The result is limited to the evaluated dataset, models, and CRF-eligible subsets.

---

## Five-Stage Error Decomposition

Each sample was assigned to the earliest stage at which the model failed.

| Model       | Stage 1<br>Original Color | Stage 2<br>Original Binding | Stage 3<br>Edited Color | Stage 4<br>CRF | Stage 5<br>All Pass |
| ----------- | ------------------------: | --------------------------: | ----------------------: | -------------: | ------------------: |
| Qwen3       |                        13 |                           0 |                      30 |          **0** |                  48 |
| InternVL3.5 |                        15 |                           0 |                      27 |          **0** |                  49 |
| Gemma-3     |                        29 |                           1 |                      34 |          **0** |                  27 |
| LLaVA-OV2   |                        15 |                           1 |                      26 |          **0** |                  49 |

Across the four models, first failures were observed much more frequently during the individual-color stages than during the binding stages.

This is a **behavioral error decomposition** and should not be interpreted as evidence about the models' internal causal mechanisms.

---

## What We Learned

### 1. Final-task errors do not necessarily identify where the failure began

A wrong binding answer may already contain an earlier failure in individual attribute recognition.

Evaluating only the final task can therefore oversimplify error attribution.

### 2. Dataset validity and model difficulty should be separated

Small, occluded, or visually cluttered objects may be difficult for a model without being invalid evaluation samples.

Treating every difficult example as invalid can unintentionally produce an overly easy benchmark.

### 3. Visual interventions must themselves be validated

When images are modified, the experiment must distinguish between:

* a model failure, and
* an artifact introduced by the image-editing process.

Segmentation, mask validation, editability checks, and human quality control became essential parts of the study.

### 4. Cleaner data comes with a sample-size trade-off

Strict filtering improves confidence in each individual example, but it also reduces the number of usable samples.

This becomes especially important when the target phenomenon may be rare.

---

## Limitations

This project has several important limitations.

* The final evaluation contains 91 Natural Original–Edited pairs.
* The CRF-eligible subset ranges from 27 to 49 samples depending on the model.
* Rare residual binding failures therefore cannot be ruled out.
* Only object–color binding was evaluated.
* Other attributes or relations may behave differently.
* The visual interventions were applied to natural images rather than fully controlled synthetic scenes.
* The five-stage analysis describes observable model behavior, not the internal causal mechanism of the model.

Additional controlled, no-image, shuffled-image, and oracle-style analyses were considered during development but were **not executed** as part of the final project.

---

## Repository Structure

> **TODO — will be updated after the final public repository cleanup.**

The repository will contain the final versions of:

```text
configs/        Model and experiment configurations
src/            Core experiment code
scripts/        Reproduction and utility scripts
analysis/       CRF and stage-wise analysis
results/        Final tables and summaries
docs/           Research note and experiment documentation
data/           Dataset metadata / registry information
```

The final structure will be documented only after the archived research code has been reviewed and cleaned.

---

## Reproduction

> **TODO — will be added after the final code audit.**

The final release will document:

* environment setup,
* required dependencies,
* dataset preparation,
* image-editing pipeline,
* model inference,
* answer parsing,
* CRF computation,
* five-stage decomposition,
* and final table generation.

---

## Technical Research Note

A detailed Technical Research Note documenting the full research process, negative results, methodological lessons, and project-closure decision will be included in the final archive.

---

## Project Status

**Closed — August 2026**

The primary Natural evaluation has been completed.

No additional models, datasets, harder-example mining, controlled conditions, or new visual relations are currently planned.

This repository is maintained as a **research archive and reproducibility record**, rather than as an actively developing benchmark.

---

## License

> **TODO — license will be selected after reviewing the redistribution requirements of the datasets and research artifacts used in this project.**
