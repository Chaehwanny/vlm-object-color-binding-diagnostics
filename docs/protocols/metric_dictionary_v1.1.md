# Metric Dictionary v1.2 Amendment

## Notation

For each Primary Natural sample, A_o and B_o are Original individual-color
correctness, G_o is Original binding correctness, A_e and B_e are Edited
individual-color correctness, and G_e is Edited binding correctness. Invalid
responses have correctness zero and are separately counted.

## Primary Metrics

- OBA = sum(G_o) / N_primary_natural_valid
- EBA = sum(G_e) / N_primary_natural_valid
- PBC = sum(G_o AND G_e) / N_primary_natural_valid

CRF is the primary conditional metric:

- crf_eligible = A_o AND B_o AND G_o AND A_e AND B_e
- crf_failure = crf_eligible AND NOT G_e
- CRF = failures / eligible N

Always report the CRF numerator, eligible denominator, rate, and interval. If
eligible N is zero, report not_estimable.

## Exclusive Primary Error Hierarchy

Assign each sample-model Natural response to exactly one category:

1. original_atomic_color_failure
2. original_binding_failure_after_atomic_pass
3. edited_atomic_color_failure_after_original_pass
4. conditional_rebinding_failure
5. full_paired_pass

These are behavioral response patterns. Korean reporting uses 개별 색상 확인 실패
for the atomic categories.

## Secondary Controlled Metrics

For the controlled_matched_valid subset, compute the same metrics as secondary
results. Natural-Controlled gaps equal Natural minus Controlled. CRF uses
track-specific denominators and a both-track common-eligible paired comparison.
The gap is not a causal background effect.

## Reporting

Report N_source, N_primary_natural_valid, each model's CRF eligible N and
failure numerator, percentages and intervals, invalid count/rate, and all five
error-category counts. Bootstrap at matched-sample level; model count does not
multiply sample N.
