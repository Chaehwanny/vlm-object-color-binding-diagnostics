#!/usr/bin/env python3
"""Audit the v1.1 pre-edit/post-edit identity-field separation."""

from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path
from typing import Any


OFFICIAL_FIELDS = {
    "material_suitability",
    "expected_mask_separability",
    "sufficient_visible_area",
    "swap_feasibility",
    "expected_identity_preservability",
}
DOCUMENT_REQUIREMENTS = {
    "docs/protocols/protocol_v1.1_PRE_INFERENCE_DRAFT.md": (
        "expected_identity_preservability",
        "object_identity_preservation_status",
        "expected favorable, not verified success",
    ),
    "docs/protocols/semantic_reclassification_rubric_v1.1_PRE_REVIEW.md": (
        "expected_mask_separability",
        "swap_feasibility",
        "not verified success",
    ),
    "docs/protocols/manifest_schema_v1.1.md": (
        "technical_criterion_scores.expected_identity_preservability",
        "object_identity_preservation_reviewer",
        "Final matched QC cannot pass",
    ),
    "docs/protocols/execution_plan_v1.1.md": (
        "expected_identity_preservability",
        "object_identity_preservation_status=pass",
        "pre-edit expected scores cannot populate this result",
    ),
    "docs/protocols/protocol_v1.1_changelog.md": (
        "identity_preserved",
        "expected_identity_preservability",
        "concept/schema correction",
    ),
    "docs/protocols/technical_triage_identity_field_revision_v1.1.md": (
        "Official Pre-Edit Fields",
        "Official Post-Edit QC Fields",
        "Human Approval",
    ),
}


def parse_args() -> argparse.Namespace:
    root = Path(__file__).resolve().parents[2]
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--project-root", type=Path, default=root)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--summary-json", type=Path)
    return parser.parse_args()


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    with path.open(encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def main() -> None:
    args = parse_args()
    errors: list[str] = []
    for relative, phrases in DOCUMENT_REQUIREMENTS.items():
        path = args.project_root / relative
        if not path.is_file():
            errors.append(f"Missing document: {relative}")
            continue
        text = path.read_text(encoding="utf-8")
        for phrase in phrases:
            if phrase not in text:
                errors.append(f"{relative}: missing required phrase: {phrase}")

    rows = read_jsonl(args.manifest)
    if len(rows) != 293:
        errors.append(f"Expected 293 manifest rows, found {len(rows)}.")
    semantic = Counter(row.get("semantic_status") for row in rows)
    valid_technical = Counter(
        row.get("technical_status")
        for row in rows
        if row.get("semantic_status") == "valid"
    )
    if semantic != Counter({"valid": 197, "invalid": 96}):
        errors.append(f"Unexpected semantic distribution: {dict(semantic)}")
    if valid_technical != Counter(
        {"segmentation_test_required": 176, "directly_editable": 21}
    ):
        errors.append(
            f"Unexpected semantic-valid technical distribution: "
            f"{dict(valid_technical)}"
        )

    for row in rows:
        candidate_id = row.get("candidate_id", "<missing>")
        scores = row.get("technical_criterion_scores")
        if not isinstance(scores, dict) or set(scores) != OFFICIAL_FIELDS:
            errors.append(f"{candidate_id}: nonofficial technical fields.")
        if row.get("technical_status") == "failed_after_test":
            errors.append(f"{candidate_id}: unexpected failed_after_test.")
        for field in (
            "mask_qc_status",
            "edit_qc_status",
            "object_identity_preservation_status",
        ):
            if row.get(field) not in {None, "", "not_tested"}:
                errors.append(f"{candidate_id}: pre-edit {field} is tested.")

    summary = {
        "document_count": len(DOCUMENT_REQUIREMENTS),
        "manifest_rows": len(rows),
        "semantic_counts": dict(semantic),
        "semantic_valid_technical_counts": dict(valid_technical),
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
