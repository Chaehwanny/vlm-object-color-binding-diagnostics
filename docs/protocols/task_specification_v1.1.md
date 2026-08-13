# Task Specification v1.2 Amendment

## Status And Unit

- Status: PRE-INFERENCE AMENDED
- Primary unit: one Natural Original/Edited pair
- Primary responses: 6 per sample
- Inference mode: independent, stateless query calls

Primary task types are object_a_color, object_b_color, and binding_choice.

## Primary Image-State Matrix

| Image role | Object A color | Object B color | Correct binding |
| --- | --- | --- | --- |
| Natural Original | color_a_original | color_b_original | Original Binding |
| Natural Edited | color_b_original | color_a_original | Swapped Binding |

## Color Forced Choice

For each image, ask one four-way color question for each object. Display red,
blue, green, and yellow in the frozen deterministic balanced order. Store the
option order and correct option.

## Binding Forced Choice

For each image, ask one two-way question whose options are Original Binding and
Swapped Binding. Original Binding is correct for Natural Original; Swapped
Binding is correct for Natural Edited.

Baseline True, Baseline False, Image Conflict, and Edit Control are the
conceptual 2x2 truth mapping, not four separate Yes/No questions.

## Response Count

- Natural Original: A color + B color + binding = 3
- Natural Edited: A color + B color + binding = 3
- Primary total: 6 responses per sample

Controlled Original/Edited may receive the same questions for secondary
matched-condition analysis. Controlled responses are not Primary responses.

## Query Independence And Balancing

- Start a new conversation for every query.
- Do not reveal the paired image, track, state, or correctness.
- Reuse the audited deterministic option-order policy.
- Yoke corresponding Natural/Controlled wording and option order.
- Preserve raw outputs before parsing.
- Invalid responses count as incorrect and are reported separately.

## Diagnostics

No-image and Shuffled-image are visual-dependence diagnostics. Direct Change
Selection is the only task labeled change tracking. Attribute Oracles are
post-hoc behavioral diagnostics.
