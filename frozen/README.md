# Frozen Contracts

This directory preserves three related but distinct freezes. A freeze manifest
records artifact paths, byte sizes, SHA256 values, ordered fingerprints, and
relevant contract metadata. `SHA256SUMS` provides a compact checksum record.

## 1. Primary Pre-Inference Freeze

[`attribute_binding_primary_v1.2/`](attribute_binding_primary_v1.2/) contains:

- `primary_freeze_manifest_v1.2.json`
- `SHA256SUMS`

This freeze fixes the 91 Natural Primary candidates, 546 queries, governing
protocol, question/config contracts, and published lineage used before model
inference. It is the authoritative data-and-question freeze for Final Main.

## 2. Historical Initial Inference Freeze

[`attribute_binding_primary_inference_v1.2/`](attribute_binding_primary_inference_v1.2/)
contains:

- `primary_inference_freeze_manifest_v1.2.json`
- `SHA256SUMS`

This is the original inference-stack freeze associated with the strict-parser
Qwen run. It is retained as parser-amendment provenance. It is not the
authoritative final inference stack and should not be used to identify final
results.

## 3. Parser-Amendment Inference Freeze

[`attribute_binding_primary_inference_v1.2_parser_amendment1/`](attribute_binding_primary_inference_v1.2_parser_amendment1/)
contains:

- `primary_inference_freeze_manifest_v1.2_parser_amendment1.json`
- `SHA256SUMS`

This is the authoritative final inference-stack freeze. It retains the same 91
candidates, 546 questions, model panel, checkpoint revisions, preprocessing,
prompt, options, gold answers, and decoding contract while recording the
gold-independent leading-option parser amendment. Final inference results use
the `*_parser_amendment1.jsonl` suffix.

The amendment rationale and preserved historical relationship are documented in
[`../docs/protocols/primary_response_parser_amendment_v1.2.md`](../docs/protocols/primary_response_parser_amendment_v1.2.md).

## Verification Scope

Freeze metadata supports inspection of artifact membership and checksums, but
exact verification is environment-dependent. The inference-stack freeze records
pinned model revisions, Python package evidence, relative virtual-environment
executables, `pip check` policy, CUDA capability, and model-specific imports.
Recreating those environments and obtaining model access may be necessary before
the verification script can pass.

The archive is therefore not presented as fully portable or one-command
reproducible. In particular:

- the Primary pre-inference freeze can be inspected from the published metadata;
- exact inference-stack verification requires environment recreation;
- four-model inference also requires external images, model access, and GPU
  compute; and
- the historical initial freeze is preserved to document provenance, not to
  supersede the parser-amended freeze.

Historical manifests remain byte-preserved records. Their status and paths
reflect the project state at the time they were created. The repository-level
status in the [root README](../README.md) — Closed, August 2026 — is
authoritative.
