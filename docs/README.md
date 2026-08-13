# Documentation

## Governing Final Documentation

The public protocol set is under [`protocols/`](protocols/):

- [`protocol_v1.2_PRE_INFERENCE_AMENDMENT.md`](protocols/protocol_v1.2_PRE_INFERENCE_AMENDMENT.md)
  defines the final Natural Primary research question, eligibility semantics,
  analysis hierarchy, and claim boundary.
- [`task_specification_v1.1.md`](protocols/task_specification_v1.1.md)
  specifies the image states, question roles, answer contract, and controls.
- [`manifest_schema_v1.1.md`](protocols/manifest_schema_v1.1.md)
  documents the broader manifest and provenance field model used during data
  construction.
- [`statistical_analysis_plan_v1.1.md`](protocols/statistical_analysis_plan_v1.1.md)
  records planned comparisons, paired analysis rules, uncertainty reporting,
  and diagnostic-analysis boundaries.
- [`metric_dictionary_v1.1.md`](protocols/metric_dictionary_v1.1.md)
  defines metric names and their behavioral interpretation.
- [`pre_edit_technical_gate_freeze_v1.1.md`](protocols/pre_edit_technical_gate_freeze_v1.1.md)
  freezes the development-derived pre-edit eligibility rule before Main
  production.
- [`primary_response_parser_amendment_v1.2.md`](protocols/primary_response_parser_amendment_v1.2.md)
  documents the one-time, gold-independent parser amendment and identifies the
  amended result set as authoritative.

The v1.2 amendment governs where it narrows or supersedes earlier v1.1 planning
language. The repository [root README](../README.md) is authoritative for final
scope, terminology, result wording, and project status.

## Historical Status Language

Historical documents are preserved as contemporaneous research records. Some
may say "in progress," describe planned secondary conditions, or use broader
pre-execution language because that was the state when they were frozen. The
repository-level status — **Closed, August 2026** — is authoritative.

In particular, Controlled/Four-image material describes development or planned
secondary work and is not part of Final Main. Final Main is Natural Primary
only. No-image, Shuffled-image, and Oracle conditions were not executed.

## Development Documentation

No separate public `archive/development/docs/` subtree is included. Public
development history is preserved through the scripts, configs, schemas, and
tests under [`../archive/development/`](../archive/development/). Those files are
provenance records, not governing Final protocol documents.
