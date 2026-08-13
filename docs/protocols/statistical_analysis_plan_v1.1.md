# Statistical Analysis Plan v1.2 Amendment

Status: PRE-INFERENCE AMENDED; MODEL RESULTS NOT INSPECTED

## 1. Primary Population

Primary inclusion requires semantic and overlap validity, segmentation and both
mask-QC passes, frozen pre-edit gate pass, Natural Edited generation, Natural
Edit and identity QC pass, and valid Original/Edited questions and answers.

Natural Edit and identity QC determine the pool eligible for Primary question
construction. Primary Natural validity is finalized only after the
corresponding Original/Edited questions and answers are generated and
validated. Controlled generation and Four-image QC are not Primary inclusion
conditions. Process all 157 frozen candidates in frozen_order; final Primary N
is the realized N_primary_natural_valid after question/answer validation.

## 2. Units

Descriptive accuracy uses responses within samples. Primary binding metrics use
the Natural Original/Edited pair within model. Bootstrap at matched-sample
level, preserving all six Primary responses. Model count does not multiply N.

## 3. Primary Endpoints

Report Natural OBA, EBA, PBC, and CRF for each model. CRF is the primary
conditional endpoint. Every estimate includes numerator, denominator,
percentage, and 95% interval. Zero eligible N is not_estimable.

CRF eligibility requires correct Original A color, Original B color, Original
binding, Edited A color, and Edited B color. CRF failure is incorrect Edited
binding among eligible samples.

## 4. Primary Error Decomposition

Use this mutually exclusive order:

1. Original atomic color failure
2. Original binding failure after atomic pass
3. Edited atomic color failure after original pass
4. Conditional Rebinding Failure
5. Full paired pass

Counts sum to N_primary_natural_valid for each model.

## 5. Uncertainty

Use Wilson 95% intervals for descriptive proportions. Use a 10,000-iteration
matched-sample bootstrap with seed 20260729 for paired summaries and secondary
Natural-Controlled differences. Never resample questions independently.

## 6. Pre-Result Scenarios

- A: repeated CRF indicates residual Edited rebinding failure after correct
  individual-color checks.
- B: rare CRF after conditioning indicates many apparent binding errors
  co-occurred with preceding individual-color errors.
- C: models differ in atomic and CRF profiles despite similar binding accuracy.

These interpretations are frozen before main VLM results.

## 7. Secondary Analyses

Controlled is secondary and includes only controlled_matched_valid samples.
Report track metrics before Natural-Controlled differences. CRF paired
comparison uses the both-track common-eligible subset.

Difficulty analysis uses QC-passing mask area and frozen metadata. Difficulty
is not an inclusion, ordering, or sampling rule.

## 8. Diagnostics

No-image and Shuffled-image are fixed-seed visual-dependence diagnostics chosen
after Primary N is known and before model results are inspected. Oracle analyses
are pre-specified post-hoc behavioral diagnostics.

## 9. Claim Boundaries

Do not infer internal-module causation, universal VLM weakness, GQA-wide
representativeness, or a pure causal background effect. Describe a GQA-derived
semantic-valid and intervention-valid diagnostic subset.
