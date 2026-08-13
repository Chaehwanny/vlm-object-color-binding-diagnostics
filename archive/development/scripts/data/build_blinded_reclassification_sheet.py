#!/usr/bin/env python3
"""Create a human review CSV with previous curation judgments hidden."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Any


FIELDS = (
    "candidate_id",
    "source_image_id",
    "source_object_a_id",
    "source_object_b_id",
    "object_a_label",
    "object_a_original_color",
    "object_b_label",
    "object_b_original_color",
    "semantic_status",
    "semantic_invalid_reason",
    "technical_status",
    "difficulty_tags",
    "reviewer_id",
    "review_confidence",
    "review_note",
    "original_image_path",
    "preview_path",
)


def parse_args() -> argparse.Namespace:
    project_root = Path(__file__).resolve().parents[2]
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument(
        "--candidate-pool",
        type=Path,
        default=project_root
        / "processed/attribute_binding/candidates/main_v1/candidate_pool.jsonl",
    )
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


def main() -> None:
    args = parse_args()
    if args.output.exists():
        raise FileExistsError(
            f"Refusing to overwrite existing output: {args.output}"
        )
    rows = read_jsonl(args.input)
    candidate_rows = read_jsonl(args.candidate_pool)
    candidates = {row["candidate_id"]: row for row in candidate_rows}
    if len(rows) != 293:
        raise ValueError(f"Expected 293 rows, found {len(rows)}.")
    if {row["candidate_id"] for row in rows} != set(candidates):
        raise ValueError("Candidate IDs differ between input and candidate pool.")
    if any(
        row.get("semantic_status") is not None
        or row.get("technical_status") is not None
        for row in rows
    ):
        raise ValueError("Input is not a blank pre-review manifest.")

    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("x", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=FIELDS)
        writer.writeheader()
        for row in rows:
            output = {field: row.get(field) for field in FIELDS}
            candidate = candidates[row["candidate_id"]]
            output["object_a_original_color"] = candidate["object_a_color"]
            output["object_b_original_color"] = candidate["object_b_color"]
            output["difficulty_tags"] = ""
            writer.writerow(output)

    print(
        json.dumps(
            {
                "rows": len(rows),
                "previous_label_exposed": False,
                "previous_reason_codes_exposed": False,
                "output": args.output.as_posix(),
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
