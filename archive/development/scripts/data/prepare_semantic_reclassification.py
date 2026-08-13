#!/usr/bin/env python3
"""Prepare a blank, development-only semantic reclassification manifest."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Any


CRITERIA_FIELDS = (
    "object_a_clear",
    "object_b_clear",
    "independent_pair",
    "label_a_correct",
    "label_b_correct",
    "unique_reference",
    "color_a_clear",
    "color_b_clear",
    "canonical_color_pair",
    "material_suitable",
    "separable_masks",
    "sufficient_visible_area",
    "swap_feasible",
    "identity_preserved",
    "scene_plausibility",
)

REVIEW_FIELDS = (
    "candidate_id",
    "source_image_id",
    "source_object_a_id",
    "source_object_b_id",
    "source_pair_key",
    "object_a_label",
    "object_b_label",
    "previous_label",
    "previous_reason_codes",
    "semantic_status",
    "semantic_invalid_reason",
    "technical_status",
    "difficulty_tags",
    "reviewer_id",
    "review_confidence",
    "review_note",
    "development_exposure",
    "development_source",
    "original_image_path",
    "preview_path",
)


def parse_args() -> argparse.Namespace:
    project_root = Path(__file__).resolve().parents[2]
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--candidate-pool",
        type=Path,
        default=project_root
        / "processed/attribute_binding/candidates/main_v1/candidate_pool.jsonl",
    )
    parser.add_argument(
        "--previous-review",
        type=Path,
        default=project_root
        / "processed/attribute_binding/candidates/main_v1/agent_prescreen_v1.csv",
    )
    parser.add_argument(
        "--image-root-prefix",
        default="raw/GQA-Scene-Graph",
        help="Repository-relative prefix prepended to source_image_path.",
    )
    parser.add_argument(
        "--preview-root-prefix",
        default="processed/attribute_binding/candidates/main_v1",
        help="Repository-relative prefix prepended to preview_path.",
    )
    parser.add_argument("--output-jsonl", type=Path, required=True)
    parser.add_argument("--output-csv", type=Path, required=True)
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


def read_previous_review(path: Path) -> dict[str, dict[str, str]]:
    with path.open(encoding="utf-8", newline="") as handle:
        rows = list(csv.DictReader(handle))
    indexed = {row["candidate_id"]: row for row in rows}
    if len(indexed) != len(rows):
        raise ValueError(f"Duplicate candidate IDs in previous review: {path}")
    return indexed


def previous_reason_codes(row: dict[str, str]) -> list[str]:
    hard_reject_reason = row.get("hard_reject_reason", "")
    if hard_reject_reason:
        return [value for value in hard_reject_reason.split(";") if value]
    return [
        field
        for field in CRITERIA_FIELDS
        if row.get(field, "").isdigit() and int(row[field]) < 2
    ]


def repository_path(prefix: str, relative_path: str) -> str:
    return (Path(prefix) / relative_path).as_posix()


def build_rows(
    candidates: list[dict[str, Any]],
    previous: dict[str, dict[str, str]],
    image_root_prefix: str,
    preview_root_prefix: str,
) -> list[dict[str, Any]]:
    candidate_ids = {row["candidate_id"] for row in candidates}
    if len(candidates) != 293 or len(candidate_ids) != 293:
        raise ValueError("Candidate pool must contain 293 unique candidates.")
    if candidate_ids != set(previous):
        raise ValueError(
            "Candidate IDs differ between candidate pool and previous review."
        )

    output: list[dict[str, Any]] = []
    for candidate in candidates:
        candidate_id = candidate["candidate_id"]
        previous_row = previous[candidate_id]
        object_a_id = str(candidate["object_a_detection_index"])
        object_b_id = str(candidate["object_b_detection_index"])
        source_image_id = str(candidate["source_image_id"])
        pair_ids = sorted((object_a_id, object_b_id), key=int)
        output.append(
            {
                "candidate_id": candidate_id,
                "source_image_id": source_image_id,
                "source_object_a_id": object_a_id,
                "source_object_b_id": object_b_id,
                "source_pair_key": (
                    f"{source_image_id}:{pair_ids[0]}:{pair_ids[1]}"
                ),
                "object_a_label": candidate["object_a_label"],
                "object_b_label": candidate["object_b_label"],
                "previous_label": previous_row["final_label"],
                "previous_reason_codes": previous_reason_codes(previous_row),
                "semantic_status": None,
                "semantic_invalid_reason": None,
                "technical_status": None,
                "difficulty_tags": [],
                "reviewer_id": None,
                "review_confidence": None,
                "review_note": None,
                "development_exposure": True,
                "development_source": "curation_pilot_293",
                "original_image_path": repository_path(
                    image_root_prefix, candidate["source_image_path"]
                ),
                "preview_path": repository_path(
                    preview_root_prefix, candidate["preview_path"]
                ),
            }
        )
    return output


def ensure_new_output(path: Path) -> None:
    if path.exists():
        raise FileExistsError(f"Refusing to overwrite existing output: {path}")
    path.parent.mkdir(parents=True, exist_ok=True)


def write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    ensure_new_output(path)
    with path.open("x", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    ensure_new_output(path)
    with path.open("x", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=REVIEW_FIELDS)
        writer.writeheader()
        for row in rows:
            csv_row = dict(row)
            csv_row["previous_reason_codes"] = ";".join(
                row["previous_reason_codes"]
            )
            csv_row["difficulty_tags"] = ""
            writer.writerow(csv_row)


def main() -> None:
    args = parse_args()
    rows = build_rows(
        read_jsonl(args.candidate_pool),
        read_previous_review(args.previous_review),
        args.image_root_prefix,
        args.preview_root_prefix,
    )
    write_jsonl(args.output_jsonl, rows)
    write_csv(args.output_csv, rows)
    print(
        json.dumps(
            {
                "rows": len(rows),
                "semantic_status_filled": sum(
                    row["semantic_status"] is not None for row in rows
                ),
                "technical_status_filled": sum(
                    row["technical_status"] is not None for row in rows
                ),
                "development_exposure_true": sum(
                    row["development_exposure"] is True for row in rows
                ),
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
