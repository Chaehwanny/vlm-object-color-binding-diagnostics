#!/usr/bin/env python3
"""Validate pre-edit expectations separately from post-edit identity QC."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any


EXPECTED_FIELDS = (
    "material_suitability",
    "expected_mask_separability",
    "sufficient_visible_area",
    "swap_feasibility",
    "expected_identity_preservability",
)
LEGACY_FIELDS = {
    "material_suitable",
    "separable_masks",
    "swap_feasible",
    "identity_preserved",
}
QC_STATUSES = {"pass", "fail", "human_review", "not_tested"}
TESTED_QC_STATUSES = {"pass", "fail", "human_review"}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument(
        "--stage",
        choices=("pre-edit", "post-edit", "final-matched"),
        required=True,
    )
    parser.add_argument("--summary-json", type=Path)
    return parser.parse_args()


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    with path.open(encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def is_blank(value: Any) -> bool:
    return value is None or value == ""


def edited_paths(row: dict[str, Any]) -> list[str]:
    paths: list[str] = []
    for field in (
        "edited_image_path",
        "natural_edited_image_path",
        "controlled_edited_image_path",
        "edited_cutout_a_path",
        "edited_cutout_b_path",
    ):
        if row.get(field):
            paths.append(str(row[field]))
    images = row.get("images")
    if isinstance(images, dict):
        for image in images.values():
            if (
                isinstance(image, dict)
                and image.get("state") == "edited"
                and image.get("path")
            ):
                paths.append(str(image["path"]))
    return paths


def mask_paths(row: dict[str, Any]) -> list[str]:
    return [
        str(row[field])
        for field in ("mask_a_path", "mask_b_path")
        if row.get(field)
    ]


def final_qc_pass(row: dict[str, Any]) -> bool:
    return any(
        (
            row.get("four_image_qc_pass") is True,
            row.get("matched_sample_qc") == "pass",
            row.get("final_matched_qc_status") == "pass",
            row.get("primary_inclusion_status") == "included",
        )
    )


def validate_pre_edit(row: dict[str, Any]) -> list[str]:
    errors: list[str] = []
    candidate_id = row.get("candidate_id", "<missing>")
    scores = row.get("technical_criterion_scores")
    if not isinstance(scores, dict):
        return [f"{candidate_id}: missing technical_criterion_scores."]
    missing = set(EXPECTED_FIELDS) - set(scores)
    if missing:
        errors.append(
            f"{candidate_id}: missing expected fields {sorted(missing)}."
        )
    legacy = LEGACY_FIELDS & set(scores)
    if legacy:
        errors.append(
            f"{candidate_id}: legacy result-like fields remain {sorted(legacy)}."
        )
    for field in EXPECTED_FIELDS:
        if field in scores and scores[field] not in {0, 1, 2}:
            errors.append(f"{candidate_id}: {field} must be 0, 1, or 2.")

    if row.get("technical_status") == "directly_editable" and any(
        scores.get(field) != 2 for field in EXPECTED_FIELDS
    ):
        errors.append(
            f"{candidate_id}: directly_editable requires five scores of 2."
        )
    if row.get("technical_status") == "failed_after_test":
        record = row.get("technical_test_record")
        if not isinstance(record, dict):
            errors.append(
                f"{candidate_id}: failed_after_test requires technical_test_record."
            )
        else:
            for field in ("test_id", "test_type", "failure_reason"):
                if is_blank(record.get(field)):
                    errors.append(
                        f"{candidate_id}: technical_test_record.{field} required."
                    )

    identity_status = row.get("object_identity_preservation_status")
    if identity_status not in {None, "", "not_tested"}:
        errors.append(
            f"{candidate_id}: pre-edit identity result must be not_tested or null."
        )
    for field in ("mask_qc_status", "edit_qc_status"):
        if row.get(field) not in {None, "", "not_tested"}:
            errors.append(
                f"{candidate_id}: pre-edit {field} must be not_tested or null."
            )
    if edited_paths(row):
        errors.append(f"{candidate_id}: edited assets present in pre-edit manifest.")
    return errors


def validate_post_edit(
    row: dict[str, Any], require_final_pass: bool
) -> list[str]:
    errors = validate_pre_edit_expectations_only(row)
    candidate_id = row.get("candidate_id", row.get("sample_id", "<missing>"))
    edit_assets = edited_paths(row)
    masks = mask_paths(row)
    identity_status = row.get("object_identity_preservation_status")
    edit_status = row.get("edit_qc_status")
    mask_status = row.get("mask_qc_status")

    for field, status in (
        ("mask_qc_status", mask_status),
        ("edit_qc_status", edit_status),
        ("object_identity_preservation_status", identity_status),
    ):
        if status is not None and status not in QC_STATUSES:
            errors.append(f"{candidate_id}: invalid {field}: {status!r}.")

    if edit_assets:
        if edit_status not in TESTED_QC_STATUSES:
            errors.append(
                f"{candidate_id}: edited image requires tested edit_qc_status."
            )
        if identity_status not in TESTED_QC_STATUSES:
            errors.append(
                f"{candidate_id}: edited image requires tested identity status."
            )
    if masks and mask_status not in TESTED_QC_STATUSES:
        errors.append(
            f"{candidate_id}: mask assets require tested mask_qc_status."
        )
    if identity_status == "fail" and is_blank(
        row.get("object_identity_preservation_note")
    ):
        errors.append(f"{candidate_id}: identity fail requires a note.")
    if identity_status in TESTED_QC_STATUSES:
        if is_blank(row.get("object_identity_preservation_reviewer")):
            errors.append(f"{candidate_id}: tested identity requires reviewer.")
        if is_blank(row.get("object_identity_preservation_timestamp")):
            errors.append(f"{candidate_id}: tested identity requires timestamp.")

    if require_final_pass or final_qc_pass(row):
        for field, status in (
            ("mask_qc_status", mask_status),
            ("edit_qc_status", edit_status),
            ("object_identity_preservation_status", identity_status),
        ):
            if status != "pass":
                errors.append(
                    f"{candidate_id}: final matched QC requires {field}=pass."
                )
    return errors


def validate_pre_edit_expectations_only(
    row: dict[str, Any],
) -> list[str]:
    errors: list[str] = []
    candidate_id = row.get("candidate_id", row.get("sample_id", "<missing>"))
    scores = row.get("technical_criterion_scores")
    if scores is None:
        return errors
    if not isinstance(scores, dict):
        return [f"{candidate_id}: technical_criterion_scores must be an object."]
    legacy = LEGACY_FIELDS & set(scores)
    if legacy:
        errors.append(
            f"{candidate_id}: legacy pre-edit fields remain {sorted(legacy)}."
        )
    return errors


def main() -> None:
    args = parse_args()
    rows = read_jsonl(args.input)
    errors: list[str] = []
    for row in rows:
        if args.stage == "pre-edit":
            errors.extend(validate_pre_edit(row))
        else:
            errors.extend(
                validate_post_edit(
                    row, require_final_pass=args.stage == "final-matched"
                )
            )
    summary = {
        "stage": args.stage,
        "row_count": len(rows),
        "error_count": len(errors),
        "errors": errors,
    }
    if args.summary_json:
        if args.summary_json.exists():
            raise FileExistsError(
                f"Refusing to overwrite output: {args.summary_json}"
            )
        args.summary_json.parent.mkdir(parents=True, exist_ok=True)
        args.summary_json.write_text(
            json.dumps(summary, ensure_ascii=False, indent=2) + "\n",
            encoding="utf-8",
        )
    if errors:
        raise ValueError("\n".join(errors))
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
