# Primary Response Parser Amendment v1.2

- Amendment timestamp: 2026-08-05T20:10:29+09:00
- Amendment ID: `parser_amendment1`
- Timing: after the first Qwen Primary run and before any OBA, EBA, PBC, or CRF computation

## Observation

The first Qwen run contained 546 structurally valid inference records: 290 were accepted by the strict parser, 256 were parser-invalid, and 0 had inference errors. Inspection of raw responses showed explicit leading choices followed by option text or a short explanation, such as `A. blue` or a multi-line option list beginning with `B. ...`.

## One-Time Correction

The historical `strict_option_letter_v1.2` parser and original Qwen result remain preserved as provenance. The amended parser, `leading_option_identifier_v1.2_amendment1`, deterministically reads a single allowed option identifier at the beginning of the response. It accepts the historical exact forms plus leading forms such as `A. text`, `A: text`, and `A because ...`.

Ambiguous forms such as `A or B`, `A/B`, `A, B`, `Maybe A`, and `A but maybe B` remain invalid. The row's `response_alphabet` is authoritative. The parser never consults the gold answer or evaluates option-text correctness, and the same rule applies to all four models.

## Unchanged Contract

This amendment does not change the 91 Primary candidates, 546 questions, images, prompts, options, option order, gold answers, reference policy, model panel, checkpoint revisions, preprocessing, or decoding. Raw model output remains preserved. The original Qwen result and the historical inference-stack freeze are not overwritten.
