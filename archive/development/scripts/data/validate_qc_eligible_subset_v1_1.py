#!/usr/bin/env python3
"""Validate QC eligibility lineage, attrition, exact rows, and frozen order."""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path
from typing import Any

REQUIRED_PASS = (
    "mask_qc_status",
    "edit_qc_status",
    "object_identity_preservation_status",
)
RULE_VERSION = "controlled_qc_eligibility_v1.1"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-reviewed-manifest", type=Path, required=True)
    parser.add_argument("--original-subset-manifest", type=Path, required=True)
    parser.add_argument("--eligible-manifest", type=Path, required=True)
    parser.add_argument("--eligibility-manifest", type=Path, required=True)
    parser.add_argument("--summary", type=Path, required=True)
    parser.add_argument("--subset-name", required=True)
    parser.add_argument("--expected-source-count", type=int, required=True)
    parser.add_argument("--expected-eligible-count", type=int, required=True)
    parser.add_argument("--expected-excluded-id", action="append", default=[])
    return parser.parse_args()


def sha256_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def read_jsonl_with_raw(path: Path) -> list[tuple[dict[str, Any], bytes]]:
    rows = []
    with path.open("rb") as handle:
        for raw in handle:
            content = raw.rstrip(b"\r\n")
            if content.strip():
                rows.append((json.loads(content.decode("utf-8")), content))
    return rows


def main() -> None:
    args = parse_args()
    errors: list[str] = []
    source = read_jsonl_with_raw(args.source_reviewed_manifest)
    original = read_jsonl_with_raw(args.original_subset_manifest)
    eligible = read_jsonl_with_raw(args.eligible_manifest)
    eligibility = read_jsonl_with_raw(args.eligibility_manifest)
    summary = json.loads(args.summary.read_text(encoding="utf-8"))

    source_ids = [row[0].get("candidate_id") for row in source]
    original_ids = [row[0].get("candidate_id") for row in original]
    eligible_ids = [row[0].get("candidate_id") for row in eligible]
    eligibility_ids = [row[0].get("candidate_id") for row in eligibility]
    source_by_id = {row[0].get("candidate_id"): row for row in source}
    eligibility_by_id = {row[0].get("candidate_id"): row[0] for row in eligibility}

    if len(source) != args.expected_source_count:
        errors.append(f"source count {len(source)} != {args.expected_source_count}")
    if len(original) != args.expected_source_count:
        errors.append(f"original count {len(original)} != {args.expected_source_count}")
    if len(eligibility) != args.expected_source_count:
        errors.append(f"eligibility count {len(eligibility)} != {args.expected_source_count}")
    if len(eligible) != args.expected_eligible_count:
        errors.append(f"eligible count {len(eligible)} != {args.expected_eligible_count}")
    for label, ids in (("source", source_ids), ("original", original_ids), ("eligible", eligible_ids), ("eligibility", eligibility_ids)):
        if None in ids or len(ids) != len(set(ids)):
            errors.append(f"{label} candidate IDs are missing or duplicated")
    if set(source_ids) != set(original_ids) or set(eligibility_ids) != set(original_ids):
        errors.append("source/original/eligibility candidate sets differ")
    if not set(eligible_ids).issubset(set(original_ids)):
        errors.append("eligible manifest contains a new candidate")

    expected_eligible_ids = []
    expected_excluded_ids = []
    for index, cid in enumerate(original_ids):
        source_row, source_raw = source_by_id.get(cid, ({}, b""))
        attrition = eligibility_by_id.get(cid, {})
        reasons = [f"{field}_not_pass" for field in REQUIRED_PASS if source_row.get(field) != "pass"]
        is_eligible = not reasons
        (expected_eligible_ids if is_eligible else expected_excluded_ids).append(cid)
        expected = {
            "original_subset": args.subset_name,
            "original_order_index": index,
            "controlled_eligible": is_eligible,
            "mask_qc_status": source_row.get("mask_qc_status"),
            "edit_qc_status": source_row.get("edit_qc_status"),
            "object_identity_preservation_status": source_row.get("object_identity_preservation_status"),
            "exclusion_reasons": reasons,
            "source_reviewed_manifest_path": args.source_reviewed_manifest.as_posix(),
            "source_reviewed_manifest_sha256": sha256_file(args.source_reviewed_manifest),
            "source_row_sha256": sha256_bytes(source_raw),
            "derivation_rule_version": RULE_VERSION,
        }
        for field, value in expected.items():
            if attrition.get(field) != value:
                errors.append(f"{cid}: {field} mismatch")
        if not attrition.get("derivation_timestamp"):
            errors.append(f"{cid}: missing derivation timestamp")
        if is_eligible and any(source_row.get(field) != "pass" for field in REQUIRED_PASS):
            errors.append(f"{cid}: eligible row has non-pass QC")
        if not is_eligible and not reasons:
            errors.append(f"{cid}: excluded row has no failure reason")

    if eligible_ids != expected_eligible_ids:
        errors.append("eligible candidate order differs from frozen original order")
    for row, raw in eligible:
        cid = row["candidate_id"]
        if cid not in source_by_id or raw != source_by_id[cid][1]:
            errors.append(f"{cid}: eligible row is not byte-identical to source reviewed row")
    if args.expected_excluded_id and set(args.expected_excluded_id) != set(expected_excluded_ids):
        errors.append(f"excluded IDs {expected_excluded_ids} differ from expected {args.expected_excluded_id}")
    if len(expected_eligible_ids) + len(expected_excluded_ids) != len(source):
        errors.append("eligible + excluded does not equal source count")

    summary_expected = {
        "source_count": len(source),
        "eligible_count": len(expected_eligible_ids),
        "excluded_count": len(expected_excluded_ids),
        "eligible_candidate_ids": expected_eligible_ids,
        "excluded_candidate_ids": expected_excluded_ids,
        "source_reviewed_manifest_sha256": sha256_file(args.source_reviewed_manifest),
        "eligible_output_sha256": sha256_file(args.eligible_manifest),
        "eligibility_manifest_sha256": sha256_file(args.eligibility_manifest),
        "candidate_reselection": False,
        "random_sampling": False,
        "source_manifest_unchanged": True,
    }
    for field, value in summary_expected.items():
        if summary.get(field) != value:
            errors.append(f"summary {field} mismatch")

    report = {
        "subset_name": args.subset_name,
        "source_count": len(source),
        "eligible_count": len(expected_eligible_ids),
        "excluded_count": len(expected_excluded_ids),
        "eligible_plus_excluded": len(expected_eligible_ids) + len(expected_excluded_ids),
        "eligible_candidate_ids": expected_eligible_ids,
        "excluded_candidate_ids": expected_excluded_ids,
        "duplicate_candidate_count": len(source_ids) - len(set(source_ids)),
        "new_candidate_count": len(set(eligible_ids) - set(original_ids)),
        "order_preserved": eligible_ids == expected_eligible_ids,
        "source_reviewed_manifest_sha256": sha256_file(args.source_reviewed_manifest),
        "validation_errors": len(errors),
        "errors": errors,
    }
    print(json.dumps(report, ensure_ascii=False, indent=2))
    if errors:
        sys.exit(1)


if __name__ == "__main__":
    main()
