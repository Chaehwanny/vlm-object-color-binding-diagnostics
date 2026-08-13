# Development Archive

> Files in this directory are preserved as development history and are not part
> of the Final Natural Primary result set.

The project is closed as of August 2026. This archive records how the research
question, data criteria, editing pipeline, controls, and metrics evolved before
the 91-pair Final Main experiment was frozen.

## Included Development Stages

The retained scripts, configs, schemas, and tests cover:

- a left/right relation positive-control study;
- object–color feasibility experiments;
- interim multi-model experiments;
- smoke, mini, and completion editing pilots;
- semantic reclassification and pre-edit technical triage;
- mask and Natural Edit QC workflow development;
- Controlled/Four-image generation and QC development;
- PACO candidate and mask exploration;
- CQ-IBF and earlier attribute-binding diagnostic development; and
- protocol and identity-field migration checks.

These materials are useful for understanding design decisions and discarded
alternatives. Their sample counts, metrics, and QC outcomes are development
evidence, not additional confirmatory observations.

## Scope Boundary

Final Main is **Natural Primary only**: 91 Natural Original–Edited pairs, six
questions per pair, and four evaluated VLMs. The presence of Controlled or
Four-image scripts and schemas here does not mean that Controlled was executed
as part of Final Main.

No-image, Shuffled-image, and Oracle conditions were not executed. Development
metrics must not be merged with or cited as confirmatory Primary results. The
authoritative parser-amended inference and analysis files are under
[`../../results/primary_v1.2/`](../../results/primary_v1.2/).

## Historical Nature

Development scripts are preserved for provenance rather than maintained as
public runtime entry points. Some commands, defaults, local assumptions, or
artifact paths may no longer match the public repository layout. The archive
also omits private images, masks, cutouts, contact sheets, and other local
research assets.

The historical `audit_main_contract_v1_2.py` utility is stored under
[`scripts/protocol/`](scripts/protocol/) because it depends on development-era
documents and is not a Final runtime dependency. The local GQA download helper
is excluded from public Git tracking and is intentionally not documented as an
execution path.

For the final research question, terminology, result table, and reproduction
boundary, use the repository [root README](../../README.md). For maintained
Final entry points, use [`../../scripts/`](../../scripts/).
