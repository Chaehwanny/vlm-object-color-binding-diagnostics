# Data and Manifests

## Published Primary Manifests

The Final Main domain is **Natural Primary only**. Its public machine-readable
manifests are under [`manifests/attribute_binding/`](manifests/attribute_binding/):

- [`primary_candidate_status_v1.2.jsonl`](manifests/attribute_binding/primary_candidate_status_v1.2.jsonl)
  contains the 91 final Natural Original–Edited pairs in Primary order, with
  candidate status, object references, image metadata, and lineage checksums.
- [`primary_queries_v1.2.jsonl`](manifests/attribute_binding/primary_queries_v1.2.jsonl)
  contains 546 questions: six independent questions for each of the 91 pairs.
- [`primary_generation_summary_v1.2.json`](manifests/attribute_binding/primary_generation_summary_v1.2.json)
  records the deterministic question-construction contract and output checksum.
- [`primary_validation_summary_v1.2.json`](manifests/attribute_binding/primary_validation_summary_v1.2.json)
  records the validated candidate/query counts, reference policy, option
  balancing, and validation status.

The six question roles are Original Object A color, Original Object B color,
Original binding, Edited Object A color, Edited Object B color, and Edited
binding. The manifest is a question and provenance record; it does not embed the
corresponding image bytes.

## Dataset Lineage

Selected machine-readable lineage is published because it is needed to inspect
the data flow and verify frozen contracts:

- [`../processed/attribute_binding/main_experiment/v1.1/preparation/main_input_manifest_v1.1.jsonl`](../processed/attribute_binding/main_experiment/v1.1/preparation/main_input_manifest_v1.1.jsonl)
  is the frozen 157-candidate Main source.
- [`../processed/attribute_binding/main_experiment/v1.1/preparation/main_input_manifest_v1.1.sha256`](../processed/attribute_binding/main_experiment/v1.1/preparation/main_input_manifest_v1.1.sha256)
  records its source checksum.
- [`../processed/attribute_binding/main_experiment/v1.1/color_editing/main_v1.1/color_edit_results_main_v1.1_reviewed.jsonl`](../processed/attribute_binding/main_experiment/v1.1/color_editing/main_v1.1/color_edit_results_main_v1.1_reviewed.jsonl)
  preserves the reviewed Natural color-edit lineage from which the 91 Primary
  pairs were derived.

These files preserve attrition and checksum provenance. They should not be
treated as additional evaluation samples beyond the 91-candidate Primary
manifest.

## External Visual Data

This repository does **not** redistribute:

- raw GQA or Visual Genome images;
- segmentation masks;
- object cutouts;
- Natural Edited images;
- review overlays or contact sheets; or
- model weights.

An image path or source identifier in a manifest is a reference, not evidence
that the corresponding asset is included. Obtaining and using GQA/Visual Genome
source material remains subject to the upstream dataset terms. Reconstructing
the complete segmentation and editing pipeline therefore requires external data
and is not supported end to end by this public archive.

## Privacy and Review Metadata

Frozen lineage may contain pseudonymous reviewer or tool identifiers used to
record human-QC provenance. These identifiers are research metadata, not access
credentials. Free-text review assets and private visual review material are not
published.

The authoritative project scope, terminology, and result interpretation are in
the repository [root README](../README.md).
