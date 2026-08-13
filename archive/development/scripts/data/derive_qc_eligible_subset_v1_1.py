#!/usr/bin/env python3
"""Derive a Controlled QC-eligible subset without reselection or row mutation."""

from __future__ import annotations

import argparse
import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

RULE_VERSION = "controlled_qc_eligibility_v1.1"
REQUIRED_PASS = (
    "mask_qc_status",
    "edit_qc_status",
    "object_identity_preservation_status",
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-reviewed-manifest", type=Path, required=True)
    parser.add_argument("--original-subset-manifest", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--subset-name", required=True)
    parser.add_argument("--expected-source-count", type=int, required=True)
    parser.add_argument("--expected-eligible-count", type=int, required=True)
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
    rows: list[tuple[dict[str, Any], bytes]] = []
    with path.open("rb") as handle:
        for raw in handle:
            content = raw.rstrip(b"\r\n")
            if content.strip():
                rows.append((json.loads(content.decode("utf-8")), content))
    return rows


def atomic_raw_jsonl(path: Path, rows: list[bytes]) -> None:
    if path.exists():
        raise FileExistsError(f"Refusing to overwrite: {path}")
    temp = path.with_name(f".{path.name}.tmp")
    if temp.exists():
        raise FileExistsError(f"Stale temporary file: {temp}")
    with temp.open("xb") as handle:
        for row in rows:
            handle.write(row + b"\n")
    temp.replace(path)


def atomic_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    encoded = [json.dumps(row, ensure_ascii=False, separators=(",", ":")).encode("utf-8") for row in rows]
    atomic_raw_jsonl(path, encoded)


def atomic_json(path: Path, value: dict[str, Any]) -> None:
    if path.exists():
        raise FileExistsError(f"Refusing to overwrite: {path}")
    temp = path.with_name(f".{path.name}.tmp")
    if temp.exists():
        raise FileExistsError(f"Stale temporary file: {temp}")
    temp.write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    temp.replace(path)


def exclusion_reasons(row: dict[str, Any]) -> list[str]:
    return [f"{field}_not_pass" for field in REQUIRED_PASS if row.get(field) != "pass"]


def main() -> None:
    args = parse_args()
    source_sha_before = sha256_file(args.source_reviewed_manifest)
    source_rows = read_jsonl_with_raw(args.source_reviewed_manifest)
    original_rows = read_jsonl_with_raw(args.original_subset_manifest)

    if len(source_rows) != args.expected_source_count:
        raise ValueError(f"Source count is {len(source_rows)}, expected {args.expected_source_count}")
    if len(original_rows) != args.expected_source_count:
        raise ValueError(f"Original subset count is {len(original_rows)}, expected {args.expected_source_count}")

    source_ids = [row[0].get("candidate_id") for row in source_rows]
    original_ids = [row[0].get("candidate_id") for row in original_rows]
    if None in source_ids or len(source_ids) != len(set(source_ids)):
        raise ValueError("Source reviewed manifest requires unique candidate IDs")
    if None in original_ids or len(original_ids) != len(set(original_ids)):
        raise ValueError("Original subset manifest requires unique candidate IDs")
    if set(source_ids) != set(original_ids):
        raise ValueError("Source reviewed IDs differ from the frozen original subset IDs")
    if any(row[0].get("pilot_subset", args.subset_name) != args.subset_name for row in original_rows):
        raise ValueError("Original subset name mismatch")

    source_by_id = {row[0]["candidate_id"]: row for row in source_rows}
    source_path = args.source_reviewed_manifest.as_posix()
    timestamp = datetime.now(timezone.utc).isoformat()
    eligibility_rows: list[dict[str, Any]] = []
    eligible_raw_rows: list[bytes] = []

    for order_index, candidate_id in enumerate(original_ids):
        source_row, source_raw = source_by_id[candidate_id]
        reasons = exclusion_reasons(source_row)
        eligible = not reasons
        if eligible:
            eligible_raw_rows.append(source_raw)
        eligibility_rows.append(
            {
                "candidate_id": candidate_id,
                "original_subset": args.subset_name,
                "original_order_index": order_index,
                "controlled_eligible": eligible,
                "mask_qc_status": source_row.get("mask_qc_status"),
                "edit_qc_status": source_row.get("edit_qc_status"),
                "object_identity_preservation_status": source_row.get(
                    "object_identity_preservation_status"
                ),
                "exclusion_reasons": reasons,
                "source_reviewed_manifest_path": source_path,
                "source_reviewed_manifest_sha256": source_sha_before,
                "source_row_sha256": sha256_bytes(source_raw),
                "derivation_rule_version": RULE_VERSION,
                "derivation_timestamp": timestamp,
            }
        )

    eligible_count = sum(row["controlled_eligible"] for row in eligibility_rows)
    if eligible_count != args.expected_eligible_count:
        raise ValueError(f"Eligible count is {eligible_count}, expected {args.expected_eligible_count}")

    args.output_dir.mkdir(parents=True, exist_ok=True)
    eligible_path = args.output_dir / f"color_edit_results_{args.subset_name}_v1.1_controlled_eligible.jsonl"
    eligibility_path = args.output_dir / f"controlled_eligibility_{args.subset_name}_v1.1.jsonl"
    summary_path = args.output_dir / f"controlled_eligibility_summary_{args.subset_name}_v1.1.json"
    for path in (eligible_path, eligibility_path, summary_path):
        if path.exists():
            raise FileExistsError(f"Refusing to overwrite: {path}")

    atomic_raw_jsonl(eligible_path, eligible_raw_rows)
    atomic_jsonl(eligibility_path, eligibility_rows)
    source_sha_after = sha256_file(args.source_reviewed_manifest)
    if source_sha_before != source_sha_after:
        raise RuntimeError("Source reviewed manifest changed during derivation")

    summary = {
        "schema_version": "1.1",
        "subset_name": args.subset_name,
        "derivation_rule_version": RULE_VERSION,
        "derivation_timestamp": timestamp,
        "source_reviewed_manifest_path": source_path,
        "source_reviewed_manifest_sha256": source_sha_before,
        "original_subset_manifest_path": args.original_subset_manifest.as_posix(),
        "original_subset_manifest_sha256": sha256_file(args.original_subset_manifest),
        "source_count": len(source_rows),
        "eligible_count": eligible_count,
        "excluded_count": len(source_rows) - eligible_count,
        "eligible_candidate_ids": [row["candidate_id"] for row in eligibility_rows if row["controlled_eligible"]],
        "excluded_candidate_ids": [row["candidate_id"] for row in eligibility_rows if not row["controlled_eligible"]],
        "candidate_reselection": False,
        "random_sampling": False,
        "order_policy": "frozen_original_subset_order",
        "eligible_output_path": eligible_path.as_posix(),
        "eligible_output_sha256": sha256_file(eligible_path),
        "eligibility_manifest_path": eligibility_path.as_posix(),
        "eligibility_manifest_sha256": sha256_file(eligibility_path),
        "source_manifest_unchanged": True,
    }
    atomic_json(summary_path, summary)
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
