#!/usr/bin/env python3
"""Validate prepared or human-completed semantic reclassification manifests."""

from __future__ import annotations

import argparse
import json
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any


REQUIRED_FIELDS = {
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
}

SEMANTIC_STATUSES = {"valid", "invalid", "human_review"}
SEMANTIC_INVALID_REASONS = {
    "label_mismatch",
    "nonunique_or_unresolvable_reference",
    "object_unidentifiable",
    "target_color_indeterminate",
}
TECHNICAL_STATUSES = {
    "directly_editable",
    "segmentation_test_required",
    "failed_after_test",
}
DIFFICULTY_TAGS = {
    "small_object",
    "partial_occlusion",
    "thin_structure",
    "complex_boundary",
    "overlap",
    "crowded_background",
    "same_owner",
    "part_whole",
    "patterned_surface",
    "text_or_logo",
    "low_color_contrast",
    "distant_object",
    "atypical_color",
}
PREVIOUS_LABELS = {"keep", "borderline", "reject"}
EXPECTED_PREVIOUS_COUNTS = {"keep": 11, "borderline": 10, "reject": 272}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument(
        "--mode", choices=("prepared", "semantic-reviewed", "reviewed"), default="prepared"
    )
    parser.add_argument(
        "--project-root",
        type=Path,
        default=Path(__file__).resolve().parents[2],
    )
    parser.add_argument("--summary-json", type=Path)
    parser.add_argument(
        "--allow-failed-after-test",
        action="store_true",
        help="Allow failed_after_test only after external test provenance audit.",
    )
    return parser.parse_args()


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with path.open(encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            try:
                row = json.loads(line)
            except json.JSONDecodeError as error:
                raise ValueError(f"{path}:{line_number}: {error}") from error
            missing = REQUIRED_FIELDS - set(row)
            if missing:
                raise ValueError(
                    f"{path}:{line_number}: missing fields {sorted(missing)}"
                )
            rows.append(row)
    return rows


def is_blank(value: Any) -> bool:
    return value is None or value == ""


def validate_common(
    rows: list[dict[str, Any]], project_root: Path
) -> list[str]:
    errors: list[str] = []
    if len(rows) != 293:
        errors.append(f"Expected 293 rows, found {len(rows)}.")

    candidate_ids = [row["candidate_id"] for row in rows]
    pair_keys = [row["source_pair_key"] for row in rows]
    if len(candidate_ids) != len(set(candidate_ids)):
        errors.append("Duplicate candidate_id values.")
    if len(pair_keys) != len(set(pair_keys)):
        errors.append("Duplicate source_pair_key values.")

    previous_counts = Counter(row["previous_label"] for row in rows)
    if set(previous_counts) - PREVIOUS_LABELS:
        errors.append(
            f"Unknown previous labels: {sorted(set(previous_counts) - PREVIOUS_LABELS)}"
        )
    if dict(previous_counts) != EXPECTED_PREVIOUS_COUNTS:
        errors.append(
            "Previous-label counts differ from frozen curation provenance: "
            f"{dict(previous_counts)}"
        )

    for row in rows:
        candidate_id = row["candidate_id"]
        if row["development_exposure"] is not True:
            errors.append(f"{candidate_id}: development_exposure must be true.")
        if row["development_source"] != "curation_pilot_293":
            errors.append(f"{candidate_id}: unexpected development_source.")
        if not isinstance(row["previous_reason_codes"], list):
            errors.append(f"{candidate_id}: previous_reason_codes must be a list.")
        if not isinstance(row["difficulty_tags"], list):
            errors.append(f"{candidate_id}: difficulty_tags must be a list.")
        for path_field in ("original_image_path", "preview_path"):
            path = project_root / row[path_field]
            if not path.is_file():
                errors.append(
                    f"{candidate_id}: missing {path_field}: {row[path_field]}"
                )
    return errors


def validate_prepared(rows: list[dict[str, Any]]) -> list[str]:
    errors: list[str] = []
    blank_fields = (
        "semantic_status",
        "semantic_invalid_reason",
        "technical_status",
        "reviewer_id",
        "review_confidence",
        "review_note",
    )
    for row in rows:
        candidate_id = row["candidate_id"]
        for field in blank_fields:
            if not is_blank(row[field]):
                errors.append(
                    f"{candidate_id}: {field} must be blank in prepared input."
                )
        if row["difficulty_tags"]:
            errors.append(
                f"{candidate_id}: difficulty_tags must be empty in prepared input."
            )
    return errors


def validate_reviewed(
    rows: list[dict[str, Any]], allow_failed_after_test: bool
) -> list[str]:
    errors: list[str] = []
    for row in rows:
        candidate_id = row["candidate_id"]
        semantic_status = row["semantic_status"]
        semantic_reason = row["semantic_invalid_reason"]
        technical_status = row["technical_status"]
        difficulty_tags = row["difficulty_tags"]

        if semantic_status not in SEMANTIC_STATUSES:
            errors.append(f"{candidate_id}: invalid semantic_status.")
        if semantic_status == "invalid":
            if semantic_reason not in SEMANTIC_INVALID_REASONS:
                errors.append(
                    f"{candidate_id}: invalid semantic_invalid_reason."
                )
        elif not is_blank(semantic_reason):
            errors.append(
                f"{candidate_id}: semantic_invalid_reason must be blank unless invalid."
            )

        if technical_status not in TECHNICAL_STATUSES:
            errors.append(f"{candidate_id}: invalid technical_status.")
        if (
            technical_status == "failed_after_test"
            and not allow_failed_after_test
        ):
            errors.append(
                f"{candidate_id}: failed_after_test requires a separate recorded "
                "segmentation-test provenance audit."
            )

        unknown_tags = set(difficulty_tags) - DIFFICULTY_TAGS
        if unknown_tags:
            errors.append(
                f"{candidate_id}: unknown difficulty tags {sorted(unknown_tags)}"
            )
        if is_blank(row["reviewer_id"]):
            errors.append(f"{candidate_id}: reviewer_id is required.")
        if is_blank(row["review_confidence"]):
            errors.append(f"{candidate_id}: review_confidence is required.")
    return errors



def validate_semantic_reviewed(rows: list[dict[str, Any]]) -> list[str]:
    errors: list[str] = []
    for row in rows:
        candidate_id = row["candidate_id"]
        semantic_status = row["semantic_status"]
        semantic_reason = row["semantic_invalid_reason"]
        if semantic_status not in SEMANTIC_STATUSES:
            errors.append(f"{candidate_id}: invalid semantic_status.")
        if semantic_status == "invalid":
            if semantic_reason not in SEMANTIC_INVALID_REASONS:
                errors.append(f"{candidate_id}: invalid semantic_invalid_reason.")
        elif not is_blank(semantic_reason):
            errors.append(
                f"{candidate_id}: semantic_invalid_reason must be blank unless invalid."
            )
        if not is_blank(row["technical_status"]):
            errors.append(
                f"{candidate_id}: technical_status must remain blank at semantic-only stage."
            )
        unknown_tags = set(row["difficulty_tags"]) - DIFFICULTY_TAGS
        if unknown_tags:
            errors.append(
                f"{candidate_id}: unknown difficulty tags {sorted(unknown_tags)}"
            )
        if is_blank(row["reviewer_id"]):
            errors.append(f"{candidate_id}: reviewer_id is required.")
        if row["review_confidence"] not in {"high", "medium", "low"}:
            errors.append(f"{candidate_id}: invalid review_confidence.")
    return errors

def summarize(rows: list[dict[str, Any]], mode: str) -> dict[str, Any]:
    previous_counts = Counter(row["previous_label"] for row in rows)
    summary: dict[str, Any] = {
        "row_count": len(rows),
        "unique_candidate_count": len({row["candidate_id"] for row in rows}),
        "unique_source_image_count": len(
            {row["source_image_id"] for row in rows}
        ),
        "development_exposure_true": sum(
            row["development_exposure"] is True for row in rows
        ),
        "previous_label_counts": dict(sorted(previous_counts.items())),
        "mode": mode,
    }
    if mode in {"semantic-reviewed", "reviewed"}:
        semantic_counts = Counter(row["semantic_status"] for row in rows)
        difficulty_counts = Counter(
            tag for row in rows for tag in row["difficulty_tags"]
        )
        valid_by_previous: dict[str, int] = defaultdict(int)
        for row in rows:
            if row["semantic_status"] == "valid":
                valid_by_previous[row["previous_label"]] += 1
        summary.update(
            {
                "semantic_status_counts": dict(sorted(semantic_counts.items())),
                "semantic_valid_by_previous_label": dict(
                    sorted(valid_by_previous.items())
                ),
                "recovered_valid_from_previous_reject": valid_by_previous[
                    "reject"
                ],
                "difficulty_tag_counts": dict(
                    sorted(difficulty_counts.items())
                ),
            }
        )
        if mode == "reviewed":
            technical_counts = Counter(row["technical_status"] for row in rows)
            summary["technical_status_counts"] = dict(
                sorted(technical_counts.items())
            )
    else:
        summary.update(
            {
                "semantic_status_filled": sum(
                    not is_blank(row["semantic_status"]) for row in rows
                ),
                "technical_status_filled": sum(
                    not is_blank(row["technical_status"]) for row in rows
                ),
            }
        )
    return summary


def write_summary(path: Path, summary: dict[str, Any]) -> None:
    if path.exists():
        raise FileExistsError(f"Refusing to overwrite existing output: {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(summary, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )


def main() -> None:
    args = parse_args()
    rows = read_jsonl(args.input)
    errors = validate_common(rows, args.project_root)
    if args.mode == "prepared":
        errors.extend(validate_prepared(rows))
    elif args.mode == "semantic-reviewed":
        errors.extend(validate_semantic_reviewed(rows))
    else:
        errors.extend(
            validate_reviewed(rows, args.allow_failed_after_test)
        )

    if errors:
        raise ValueError("\n".join(errors))

    summary = summarize(rows, args.mode)
    if args.summary_json:
        write_summary(args.summary_json, summary)
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
