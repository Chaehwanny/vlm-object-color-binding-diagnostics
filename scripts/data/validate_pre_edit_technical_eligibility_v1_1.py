#!/usr/bin/env python3
"""Validate pre-edit eligibility by independent deterministic re-derivation."""

from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
import sys
import tempfile
from pathlib import Path
from typing import Any


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--reviewed-mask-manifest", type=Path, required=True)
    p.add_argument("--original-subset-manifest", type=Path, required=True)
    p.add_argument("--candidate-pool", type=Path, required=True)
    p.add_argument("--segmentation-root", type=Path, required=True)
    p.add_argument("--project-root", type=Path, required=True)
    p.add_argument("--semantic-purity-review", type=Path, required=True)
    p.add_argument("--gate-config", type=Path, required=True)
    p.add_argument("--subset-name", required=True)
    p.add_argument("--eligible-manifest", type=Path)
    p.add_argument("--eligibility-manifest", type=Path, required=True)
    p.add_argument("--object-diagnostics", type=Path)
    p.add_argument("--summary", type=Path, required=True)
    p.add_argument("--expected-source-count", type=int, required=True)
    p.add_argument("--expected-segmentation-failure-count", type=int, required=True)
    p.add_argument("--expected-pixel-gate-input-count", type=int, required=True)
    p.add_argument("--expected-eligible-count", type=int, required=True)
    p.add_argument("--expected-excluded-id", action="append", default=[])
    p.add_argument("--audit-only", action="store_true")
    return p.parse_args()


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    with path.open(encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def main() -> None:
    args = parse_args()
    errors: list[str] = []
    eligibility = read_jsonl(args.eligibility_manifest)
    if not eligibility:
        errors.append("eligibility manifest is empty")
        timestamp = "missing"
    else:
        timestamps = {row.get("derivation_timestamp") for row in eligibility}
        if len(timestamps) != 1 or None in timestamps:
            errors.append("eligibility derivation timestamp is not unique")
        timestamp = next(iter(timestamps - {None}), "missing")
    source_before = sha256(args.reviewed_mask_manifest)

    derive_script = Path(__file__).with_name("derive_pre_edit_technical_eligibility_v1_1.py")
    with tempfile.TemporaryDirectory(prefix="pre_edit_gate_validate_") as temp_value:
        temp = Path(temp_value)
        expected_eligibility = temp / "eligibility.jsonl"
        expected_summary = temp / "summary.json"
        command = [
            sys.executable, derive_script.as_posix(),
            "--reviewed-mask-manifest", args.reviewed_mask_manifest.as_posix(),
            "--original-subset-manifest", args.original_subset_manifest.as_posix(),
            "--candidate-pool", args.candidate_pool.as_posix(),
            "--segmentation-root", args.segmentation_root.as_posix(),
            "--project-root", args.project_root.as_posix(),
            "--semantic-purity-review", args.semantic_purity_review.as_posix(),
            "--gate-config", args.gate_config.as_posix(),
            "--subset-name", args.subset_name,
            "--derivation-timestamp", timestamp,
            "--eligibility-output", expected_eligibility.as_posix(),
            "--summary-output", expected_summary.as_posix(),
        ]
        if args.audit_only:
            command.append("--audit-only")
        else:
            if args.eligible_manifest is None or args.object_diagnostics is None:
                errors.append("normal mode requires eligible manifest and object diagnostics")
            expected_eligible = temp / "eligible.jsonl"
            expected_objects = temp / "objects.jsonl"
            command += [
                "--eligible-output", expected_eligible.as_posix(),
                "--object-diagnostics-output", expected_objects.as_posix(),
            ]
        if not errors:
            completed = subprocess.run(command, text=True, capture_output=True)
            if completed.returncode:
                errors.append(f"deterministic re-derivation failed: {completed.stderr.strip()}")
            else:
                if args.eligibility_manifest.read_bytes() != expected_eligibility.read_bytes():
                    errors.append("candidate eligibility differs from deterministic re-derivation")
                if not args.audit_only:
                    if args.eligible_manifest.read_bytes() != expected_eligible.read_bytes():
                        errors.append("eligible subset differs from deterministic re-derivation")
                    if args.object_diagnostics.read_bytes() != expected_objects.read_bytes():
                        errors.append("object diagnostics differ from deterministic re-derivation")
                actual_summary = json.loads(args.summary.read_text(encoding="utf-8"))
                expected_summary_value = json.loads(expected_summary.read_text(encoding="utf-8"))
                for field in ("eligible_output_path", "object_diagnostics_output_path"):
                    actual_summary.pop(field, None)
                    expected_summary_value.pop(field, None)
                if actual_summary != expected_summary_value:
                    errors.append("summary differs from deterministic re-derivation")

    original_rows = read_jsonl(args.original_subset_manifest)
    source_rows = read_jsonl(args.reviewed_mask_manifest)
    semantic_rows = read_jsonl(args.semantic_purity_review)
    source_count = len(original_rows)
    source_by = {row.get("candidate_id"): row for row in source_rows}
    semantic_by = {row.get("candidate_id"): row for row in semantic_rows}
    original_ids_for_contract = [row.get("candidate_id") for row in original_rows]
    segmentation_success_ids = [
        cid for cid in original_ids_for_contract
        if source_by.get(cid, {}).get("mask_generation_status") == "generated"
    ]
    segmentation_failure_ids = [
        cid for cid in original_ids_for_contract
        if source_by.get(cid, {}).get("mask_generation_status") == "segmentation_failed"
    ]
    if len(source_rows) != source_count or set(source_by) != set(original_ids_for_contract):
        errors.append("reviewed mask manifest does not preserve frozen source membership")
    if len(segmentation_success_ids) + len(segmentation_failure_ids) != source_count:
        errors.append("reviewed mask manifest contains unknown generation status")
    if len(segmentation_failure_ids) != args.expected_segmentation_failure_count:
        errors.append("segmentation failure count mismatch")
    if len(semantic_by) != len(semantic_rows) or set(semantic_by) != set(segmentation_success_ids):
        errors.append("semantic review domain differs from segmentation-success candidates")

    basic_qc_failure_ids = [
        cid for cid in segmentation_success_ids
        if source_by[cid].get("mask_qc_status") != "pass"
    ]
    pixel_gate_ids = [
        cid for cid in segmentation_success_ids
        if source_by[cid].get("mask_qc_status") == "pass"
        and semantic_by.get(cid, {}).get("object_a_semantic_purity_status") == "pass"
        and semantic_by.get(cid, {}).get("object_b_semantic_purity_status") == "pass"
        and not semantic_by.get(cid, {}).get("non_target_inclusion_a")
        and not semantic_by.get(cid, {}).get("non_target_inclusion_b")
    ]
    if len(pixel_gate_ids) != args.expected_pixel_gate_input_count:
        errors.append("pixel-gate input count mismatch")

    eligible_rows = [row for row in eligibility if row.get("candidate_pre_edit_eligible")]
    excluded_rows = [row for row in eligibility if not row.get("candidate_pre_edit_eligible")]
    ids = [row.get("candidate_id") for row in eligibility]
    original_ids = original_ids_for_contract
    if source_count != args.expected_source_count or len(eligibility) != args.expected_source_count:
        errors.append("source/eligibility count mismatch")
    if len(eligible_rows) != args.expected_eligible_count:
        errors.append("eligible count mismatch")
    if len(ids) != len(set(ids)) or set(ids) != set(original_ids):
        errors.append("candidate duplication, addition, or omission")
    if ids != original_ids:
        errors.append("candidate order changed")
    excluded_ids = [row["candidate_id"] for row in excluded_rows]
    if args.expected_excluded_id and set(excluded_ids) != set(args.expected_excluded_id):
        errors.append("excluded candidate IDs differ from expectation")
    if any(not row.get("exclusion_reasons") for row in excluded_rows):
        errors.append("excluded candidate without attrition reason")
    if any(row.get("candidate_id") not in set(pixel_gate_ids) for row in eligible_rows):
        errors.append("eligible manifest contains a pre-gate attrition candidate")
    if not args.audit_only and args.object_diagnostics is not None:
        diagnostics = read_jsonl(args.object_diagnostics)
        diagnostic_pairs = [
            (row.get("candidate_id"), row.get("object_side")) for row in diagnostics
        ]
        expected_pairs = [
            (cid, side) for cid in pixel_gate_ids for side in ("a", "b")
        ]
        if diagnostic_pairs != expected_pairs:
            errors.append("object diagnostics do not exactly cover pixel-gate inputs")
    if sha256(args.reviewed_mask_manifest) != source_before:
        errors.append("source reviewed mask manifest changed during validation")

    report = {
        "subset_name": args.subset_name,
        "source_count": source_count,
        "segmentation_success_count": len(segmentation_success_ids),
        "segmentation_failure_count": len(segmentation_failure_ids),
        "semantic_review_count": len(semantic_rows),
        "basic_qc_failure_count": len(basic_qc_failure_ids),
        "pre_gate_qc_attrition_count": source_count - len(pixel_gate_ids),
        "pixel_gate_input_count": len(pixel_gate_ids),
        "pixel_gate_failure_count": len(pixel_gate_ids) - len(eligible_rows),
        "eligible_count": len(eligible_rows),
        "excluded_count": len(excluded_rows),
        "eligible_plus_excluded": len(eligible_rows) + len(excluded_rows),
        "duplicate_candidate_count": len(ids) - len(set(ids)),
        "new_candidate_count": len(set(ids) - set(original_ids)),
        "order_preserved": ids == original_ids,
        "excluded_candidate_ids": excluded_ids,
        "source_manifest_sha256": source_before,
        "gate_config_sha256": sha256(args.gate_config),
        "deterministic_rederivation_match": not errors,
        "validation_errors": len(errors),
        "errors": errors,
    }
    print(json.dumps(report, ensure_ascii=False, indent=2))
    if errors:
        sys.exit(1)


if __name__ == "__main__":
    main()
