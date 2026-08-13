#!/usr/bin/env python3
"""Merge a completed blinded review CSV into the canonical provenance manifest."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Any


JUDGMENT_FIELDS = (
    "semantic_status",
    "semantic_invalid_reason",
    "technical_status",
    "difficulty_tags",
    "reviewer_id",
    "review_confidence",
    "review_note",
)

IMMUTABLE_REVIEW_FIELDS = (
    "source_image_id",
    "source_object_a_id",
    "source_object_b_id",
    "object_a_label",
    "object_b_label",
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--prepared-manifest", type=Path, required=True)
    parser.add_argument("--review-csv", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    return parser.parse_args()


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with path.open(encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            try:
                rows.append(json.loads(line))
            except json.JSONDecodeError as error:
                raise ValueError(f"{path}:{line_number}: {error}") from error
    return rows


def read_review_csv(path: Path) -> dict[str, dict[str, str]]:
    with path.open(encoding="utf-8-sig", newline="") as handle:
        rows = list(csv.DictReader(handle))
    indexed = {row["candidate_id"]: row for row in rows}
    if len(indexed) != len(rows):
        raise ValueError("Duplicate candidate_id values in review CSV.")
    return indexed


def blank_to_none(value: str | None) -> str | None:
    if value is None:
        return None
    stripped = value.strip()
    return stripped or None


def parse_tags(value: str | None) -> list[str]:
    if not value:
        return []
    return [
        tag.strip()
        for tag in value.replace(",", ";").split(";")
        if tag.strip()
    ]


def main() -> None:
    args = parse_args()
    if args.output.exists():
        raise FileExistsError(f"Refusing to overwrite existing output: {args.output}")

    prepared = read_jsonl(args.prepared_manifest)
    reviewed = read_review_csv(args.review_csv)
    prepared_ids = {row["candidate_id"] for row in prepared}
    if len(prepared) != 293 or len(prepared_ids) != 293:
        raise ValueError("Prepared manifest must contain 293 unique candidates.")
    if prepared_ids != set(reviewed):
        raise ValueError("Review CSV candidate coverage differs from prepared manifest.")

    output_rows: list[dict[str, Any]] = []
    for row in prepared:
        candidate_id = row["candidate_id"]
        review = reviewed[candidate_id]
        for field in IMMUTABLE_REVIEW_FIELDS:
            if str(review.get(field, "")).strip() != str(row[field]):
                raise ValueError(
                    f"{candidate_id}: immutable field changed in review CSV: {field}"
                )
        merged = dict(row)
        for field in JUDGMENT_FIELDS:
            if field == "difficulty_tags":
                merged[field] = parse_tags(review.get(field))
            else:
                merged[field] = blank_to_none(review.get(field))
        output_rows.append(merged)

    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("x", encoding="utf-8") as handle:
        for row in output_rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")

    print(
        json.dumps(
            {
                "rows": len(output_rows),
                "semantic_status_filled": sum(
                    row["semantic_status"] is not None for row in output_rows
                ),
                "technical_status_filled": sum(
                    row["technical_status"] is not None for row in output_rows
                ),
                "output": args.output.as_posix(),
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
