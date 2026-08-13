#!/usr/bin/env python3
"""Migrate legacy pre-edit technical names without changing judgments."""

from __future__ import annotations

import argparse
import hashlib
import json
from collections import Counter
from pathlib import Path
from typing import Any


FIELD_MAP = {
    "material_suitable": "material_suitability",
    "separable_masks": "expected_mask_separability",
    "sufficient_visible_area": "sufficient_visible_area",
    "swap_feasible": "swap_feasibility",
    "identity_preserved": "expected_identity_preservability",
}

EDIT_RESULT_PATH_FIELDS = {
    "edited_image_path",
    "natural_edited_image_path",
    "controlled_edited_image_path",
    "edited_cutout_a_path",
    "edited_cutout_b_path",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--report", type=Path, required=True)
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


def ensure_new(paths: tuple[Path, ...]) -> None:
    existing = [path for path in paths if path.exists()]
    if existing:
        raise FileExistsError(f"Refusing to overwrite outputs: {existing}")
    for path in paths:
        path.parent.mkdir(parents=True, exist_ok=True)


def has_edit_result(row: dict[str, Any]) -> bool:
    if any(row.get(field) for field in EDIT_RESULT_PATH_FIELDS):
        return True
    images = row.get("images")
    if isinstance(images, dict):
        for image in images.values():
            if (
                isinstance(image, dict)
                and image.get("state") == "edited"
                and image.get("path")
            ):
                return True
    return False


def main() -> None:
    args = parse_args()
    ensure_new((args.output, args.report))
    rows = read_jsonl(args.input)
    if len(rows) != 293:
        raise ValueError(f"Expected 293 rows, found {len(rows)}.")

    before_semantic = Counter(row["semantic_status"] for row in rows)
    before_technical = Counter(
        (row["semantic_status"], row["technical_status"]) for row in rows
    )
    cannot_migrate: list[dict[str, Any]] = []
    migrated: list[dict[str, Any]] = []
    source_digest = sha256(args.input)

    for source in rows:
        candidate_id = source["candidate_id"]
        scores = source.get("technical_criterion_scores")
        reasons = source.get("technical_reason_codes", [])
        row_errors: list[str] = []

        if not isinstance(scores, dict):
            row_errors.append("missing_technical_criterion_scores")
        else:
            missing = set(FIELD_MAP) - set(scores)
            if missing:
                row_errors.append(f"missing_legacy_fields:{sorted(missing)}")
            for field in FIELD_MAP:
                if field in scores and scores[field] not in {0, 1, 2}:
                    row_errors.append(f"nonordinal_score:{field}")
        if source.get("technical_assessment_basis") != (
            "model_blind_prescreen_criteria"
        ):
            row_errors.append("unexpected_assessment_basis")
        if has_edit_result(source):
            row_errors.append("edit_result_provenance_present")
        if source.get("object_identity_preservation_status") not in {
            None,
            "",
            "not_tested",
        }:
            row_errors.append("actual_identity_result_present")

        if row_errors:
            cannot_migrate.append(
                {"candidate_id": candidate_id, "reasons": row_errors}
            )
            continue

        row = dict(source)
        row["technical_criterion_scores"] = {
            FIELD_MAP[field]: scores[field] for field in FIELD_MAP
        }
        row["technical_reason_codes"] = [
            FIELD_MAP.get(reason, reason) for reason in reasons
        ]
        row["mask_qc_status"] = "not_tested"
        row["edit_qc_status"] = "not_tested"
        row["object_identity_preservation_status"] = "not_tested"
        row["object_identity_preservation_note"] = None
        row["object_identity_preservation_reviewer"] = None
        row["object_identity_preservation_timestamp"] = None
        row["technical_field_migration"] = {
            "migration_version": "technical_identity_fields_v1.1",
            "source_manifest": args.input.as_posix(),
            "source_manifest_sha256": source_digest,
            "field_map": FIELD_MAP,
            "values_changed": False,
            "technical_status_changed": False,
            "meaning": "pre_edit_expected_risk_not_verified_success",
        }
        migrated.append(row)

    if cannot_migrate:
        report = {
            "status": "human_approval_required",
            "input_rows": len(rows),
            "migrated_rows": 0,
            "cannot_migrate_count": len(cannot_migrate),
            "cannot_migrate": cannot_migrate,
        }
        args.report.write_text(
            json.dumps(report, ensure_ascii=False, indent=2) + "\n",
            encoding="utf-8",
        )
        raise ValueError(
            f"{len(cannot_migrate)} rows cannot be migrated automatically."
        )

    after_semantic = Counter(row["semantic_status"] for row in migrated)
    after_technical = Counter(
        (row["semantic_status"], row["technical_status"]) for row in migrated
    )
    if before_semantic != after_semantic or before_technical != after_technical:
        raise AssertionError("Migration changed semantic or technical judgments.")

    with args.output.open("x", encoding="utf-8") as handle:
        for row in migrated:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")

    report = {
        "status": "migrated",
        "migration_version": "technical_identity_fields_v1.1",
        "source_manifest": args.input.as_posix(),
        "source_manifest_sha256": source_digest,
        "input_rows": len(rows),
        "migrated_rows": len(migrated),
        "cannot_migrate_count": 0,
        "cannot_migrate_candidate_ids": [],
        "field_map": FIELD_MAP,
        "semantic_counts_before": dict(before_semantic),
        "semantic_counts_after": dict(after_semantic),
        "semantic_valid_technical_before": {
            status: count
            for (semantic, status), count in before_technical.items()
            if semantic == "valid"
        },
        "semantic_valid_technical_after": {
            status: count
            for (semantic, status), count in after_technical.items()
            if semantic == "valid"
        },
        "post_edit_status_initialization": {
            "mask_qc_status": "not_tested",
            "edit_qc_status": "not_tested",
            "object_identity_preservation_status": "not_tested",
        },
        "segmentation_executed": False,
        "editing_executed": False,
        "controlled_generation_executed": False,
        "vlm_inference_executed": False,
    }
    args.report.write_text(
        json.dumps(report, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
