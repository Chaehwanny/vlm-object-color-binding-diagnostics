#!/usr/bin/env python3
"""Import explicit human four-image QC decisions for any frozen subset."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
from copy import deepcopy
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

CRITERIA = (
    "controlled_original_identity", "controlled_edited_identity",
    "original_color_binding", "edited_color_binding", "edited_cutout_reuse",
    "placement_consistency", "relative_scale_preservation", "overlap_and_clipping",
    "background_margin_alpha_quality", "four_image_prompt_answer_validity",
)
ALL_PASS_NOTE = (
    "Human review confirmed object identity, original and edited bindings, authoritative "
    "edited-cutout reuse, matched placement and relative scale, no overlap or clipping, "
    "acceptable neutral background/alpha boundaries, and unambiguous four-image answers."
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--generated-results", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--subset-name", required=True)
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--review-input", type=Path)
    group.add_argument("--confirm-all-pass", action="store_true")
    parser.add_argument("--reviewer")
    parser.add_argument("--review-timestamp")
    return parser.parse_args()


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    with path.open(encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def reviewed_at(value: str | None) -> str:
    if value is None:
        return datetime.now(timezone.utc).isoformat()
    parsed = datetime.fromisoformat(value.replace("Z", "+00:00"))
    if parsed.tzinfo is None:
        raise ValueError("--review-timestamp must include a timezone")
    return parsed.isoformat()


def atomic_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    if path.exists():
        raise FileExistsError(f"Refusing to overwrite: {path}")
    temp = path.with_name(f".{path.name}.tmp")
    if temp.exists():
        raise FileExistsError(f"Stale temporary file: {temp}")
    with temp.open("x", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")
    temp.replace(path)


def atomic_json(path: Path, value: dict[str, Any]) -> None:
    if path.exists():
        raise FileExistsError(f"Refusing to overwrite: {path}")
    temp = path.with_name(f".{path.name}.tmp")
    temp.write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    temp.replace(path)


def atomic_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    if path.exists():
        raise FileExistsError(f"Refusing to overwrite: {path}")
    fields = ("candidate_id", "controlled_qc_status", "four_image_qc_status", "reviewer", "review_timestamp", "review_note", "failure_reasons", *CRITERIA)
    temp = path.with_name(f".{path.name}.tmp")
    with temp.open("x", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for row in rows:
            writer.writerow({field: row[field] for field in fields})
    temp.replace(path)


def validate_generated(rows: list[dict[str, Any]], subset: str, root: Path) -> None:
    if not rows or len(rows) != len({row.get("candidate_id") for row in rows}):
        raise ValueError("Generated results require nonempty unique candidate IDs")
    for row in rows:
        cid = row["candidate_id"]
        expected = {"subset_name": subset, "controlled_generation_status": "generated", "controlled_qc_status": "not_tested", "four_image_qc_status": "not_tested", "mask_qc_status": "pass", "edit_qc_status": "pass", "object_identity_preservation_status": "pass", "failure_reasons": []}
        for field, value in expected.items():
            if row.get(field) != value:
                raise ValueError(f"{cid}: {field}={row.get(field)!r}; expected {value!r}")
        for path_field, checksum_field in (("controlled_original_path", "controlled_original_sha256"), ("controlled_edited_path", "controlled_edited_sha256"), ("four_image_contact_sheet_path", "four_image_contact_sheet_sha256")):
            path = root / row[path_field]
            expected_sha = row["asset_checksums"][checksum_field]
            if not path.is_file() or sha256(path) != expected_sha:
                raise ValueError(f"{cid}: missing/checksum-invalid asset: {path}")


def validate_reviews(reviews: list[dict[str, Any]], ids: set[str], subset: str) -> None:
    indexed = {row.get("candidate_id"): row for row in reviews}
    if len(reviews) != len(indexed) or set(indexed) != ids:
        raise ValueError("Review IDs must exactly match generated result IDs")
    allowed = {"pass", "fail", "human_review"}
    for row in reviews:
        cid = row["candidate_id"]
        if row.get("subset_name") != subset:
            raise ValueError(f"{cid}: subset mismatch")
        if row.get("controlled_qc_status") not in allowed or row.get("four_image_qc_status") not in allowed:
            raise ValueError(f"{cid}: aggregate review status is incomplete")
        criteria = row.get("criteria", {})
        if set(criteria) != set(CRITERIA) or any(value not in allowed for value in criteria.values()):
            raise ValueError(f"{cid}: criterion review is incomplete")
        reasons = row.get("failure_reasons", [])
        if row["four_image_qc_status"] == "pass" and reasons:
            raise ValueError(f"{cid}: pass forbids failure reasons")
        if row["four_image_qc_status"] == "fail" and not reasons:
            raise ValueError(f"{cid}: fail requires a reason")
        if not row.get("reviewer") or not row.get("review_timestamp"):
            raise ValueError(f"{cid}: reviewer and timestamp required")
        if row["four_image_qc_status"] != "pass" and not row.get("review_note"):
            raise ValueError(f"{cid}: non-pass review requires a note")


def main() -> None:
    args = parse_args()
    generated = read_jsonl(args.generated_results)
    validate_generated(generated, args.subset_name, args.output_dir)
    ids = {row["candidate_id"] for row in generated}
    if args.confirm_all_pass:
        reviewer = (args.reviewer or "").strip()
        if not reviewer:
            raise ValueError("--confirm-all-pass requires --reviewer")
        timestamp = reviewed_at(args.review_timestamp)
        reviews = [{"schema_version": "1.1", "subset_name": args.subset_name, "candidate_id": row["candidate_id"], "criteria": {criterion: "pass" for criterion in CRITERIA}, "controlled_qc_status": "pass", "four_image_qc_status": "pass", "failure_reasons": [], "review_note": ALL_PASS_NOTE, "reviewer": reviewer, "review_timestamp": timestamp} for row in generated]
    else:
        reviews = read_jsonl(args.review_input)
    validate_reviews(reviews, ids, args.subset_name)
    review_by_id = {row["candidate_id"]: row for row in reviews}
    reviewed_results = []
    for row in generated:
        review = review_by_id[row["candidate_id"]]
        reviewed = deepcopy(row)
        reviewed.update({"controlled_qc_status": review["controlled_qc_status"], "controlled_original_qc_status": review["controlled_qc_status"], "controlled_edited_qc_status": review["controlled_qc_status"], "four_image_qc_status": review["four_image_qc_status"], "failure_reasons": list(review["failure_reasons"]), "four_image_qc_criteria": dict(review["criteria"]), "four_image_qc_reviewer": review["reviewer"], "four_image_qc_timestamp": review["review_timestamp"], "four_image_qc_note": review.get("review_note")})
        reviewed_results.append(reviewed)
    prefix = args.subset_name
    paths = {
        "csv": args.output_dir / f"four_image_qc_review_{prefix}_v1.1.csv",
        "review": args.output_dir / f"four_image_qc_review_{prefix}_v1.1.jsonl",
        "reviewed": args.output_dir / f"controlled_generation_results_{prefix}_v1.1_reviewed.jsonl",
        "summary": args.output_dir / f"four_image_qc_summary_{prefix}_v1.1.json",
    }
    existing = [path for path in paths.values() if path.exists()]
    if existing:
        raise FileExistsError(f"Refusing to overwrite: {existing}")
    csv_rows = []
    for review in reviews:
        csv_rows.append({"candidate_id": review["candidate_id"], "controlled_qc_status": review["controlled_qc_status"], "four_image_qc_status": review["four_image_qc_status"], "reviewer": review["reviewer"], "review_timestamp": review["review_timestamp"], "review_note": review.get("review_note"), "failure_reasons": json.dumps(review["failure_reasons"]), **review["criteria"]})
    atomic_csv(paths["csv"], csv_rows)
    atomic_jsonl(paths["review"], reviews)
    atomic_jsonl(paths["reviewed"], reviewed_results)
    summary = {"schema_version": "1.1", "subset_name": args.subset_name, "candidate_count": len(generated), "reviewed_count": len(reviews), "controlled_qc_status_counts": dict(Counter(row["controlled_qc_status"] for row in reviews)), "four_image_qc_status_counts": dict(Counter(row["four_image_qc_status"] for row in reviews)), "failure_reason_count": sum(len(row["failure_reasons"]) for row in reviews), "asset_modifications": 0, "generated_results_sha256": sha256(args.generated_results), "outputs": {key: path.as_posix() for key, path in paths.items() if key != "summary"}}
    atomic_json(paths["summary"], summary)
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
