#!/usr/bin/env python3
"""Assign conservative pre-segmentation technical statuses for protocol v1.1."""

from __future__ import annotations

import argparse
import csv
import json
from collections import Counter
from pathlib import Path
from typing import Any


TECHNICAL_EVIDENCE_FIELDS = {
    "material_suitability": "material_suitable",
    "expected_mask_separability": "separable_masks",
    "sufficient_visible_area": "sufficient_visible_area",
    "swap_feasibility": "swap_feasible",
    "expected_identity_preservability": "identity_preserved",
}

TECHNICAL_CRITERIA = tuple(TECHNICAL_EVIDENCE_FIELDS)


VISUAL_TEST_OVERRIDES = {
    "maincand_0031": ["visual_thin_structure", "visual_partial_occlusion"],
    "maincand_0072": ["visual_partial_occlusion"],
    "maincand_0082": ["visual_small_visible_region", "visual_clutter"],
    "maincand_0092": ["visual_background_like_region", "visual_thin_structure"],
    "maincand_0104": ["visual_partial_occlusion"],
    "maincand_0118": ["visual_background_like_region"],
    "maincand_0123": ["visual_overlap", "visual_complex_boundary"],
    "maincand_0130": ["visual_small_object", "visual_partial_occlusion"],
    "maincand_0156": ["visual_repeated_instances"],
    "maincand_0179": ["visual_repeated_instances", "visual_partial_occlusion"],
    "maincand_0195": ["visual_partial_occlusion", "visual_low_contrast"],
    "maincand_0199": ["visual_small_object", "visual_distant_object"],
    "maincand_0243": ["visual_thin_structure", "visual_partial_occlusion"],
    "maincand_0251": ["visual_semantic_anomaly", "visual_partial_occlusion"],
    "maincand_0255": ["visual_background_like_region"],
    "maincand_0267": ["visual_thin_structure"],
    "maincand_0290": ["visual_semantic_anomaly", "visual_small_visible_region"],
}

SEMANTIC_VISUAL_OVERRIDES = {
    "maincand_0251": "label_mismatch",
    "maincand_0290": "object_unidentifiable",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--semantic-manifest", type=Path, required=True)
    parser.add_argument("--technical-evidence", type=Path, required=True)
    parser.add_argument("--output-jsonl", type=Path, required=True)
    parser.add_argument("--output-csv", type=Path, required=True)
    parser.add_argument("--summary-json", type=Path, required=True)
    return parser.parse_args()


def ensure_new(paths: list[Path]) -> None:
    existing = [path for path in paths if path.exists()]
    if existing:
        raise FileExistsError(f"Refusing to overwrite outputs: {existing}")
    for path in paths:
        path.parent.mkdir(parents=True, exist_ok=True)


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    with path.open(encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def main() -> None:
    args = parse_args()
    outputs = [args.output_jsonl, args.output_csv, args.summary_json]
    ensure_new(outputs)

    rows = read_jsonl(args.semantic_manifest)
    with args.technical_evidence.open(encoding="utf-8", newline="") as handle:
        evidence_rows = {
            row["candidate_id"]: row for row in csv.DictReader(handle)
        }

    if len(rows) != 293 or len(evidence_rows) != 293:
        raise ValueError(
            f"Expected 293 rows in each input; found {len(rows)} and "
            f"{len(evidence_rows)}."
        )

    output_rows: list[dict[str, Any]] = []
    for source in rows:
        candidate_id = source["candidate_id"]
        if candidate_id not in evidence_rows:
            raise ValueError(f"Missing technical evidence for {candidate_id}.")
        evidence = evidence_rows[candidate_id]
        scores = {
            official: int(evidence[legacy])
            for official, legacy in TECHNICAL_EVIDENCE_FIELDS.items()
        }
        uncertain_criteria = [
            field for field, score in scores.items() if score < 2
        ]
        visual_reasons = VISUAL_TEST_OVERRIDES.get(candidate_id, [])

        row = dict(source)
        if candidate_id in SEMANTIC_VISUAL_OVERRIDES:
            row["semantic_status"] = "invalid"
            row["semantic_invalid_reason"] = SEMANTIC_VISUAL_OVERRIDES[candidate_id]
            row["semantic_reaudit_version"] = "direct_editable_visual_audit_v1.1"
            row["semantic_reaudit_note"] = (
                "An obvious semantic anomaly was discovered while visually "
                "auditing the provisional directly-editable subset."
            )
        row["technical_status"] = (
            "directly_editable"
            if not uncertain_criteria and not visual_reasons
            else "segmentation_test_required"
        )
        row["technical_assessment_version"] = "technical_triage_v1.1.1"
        row["technical_assessment_basis"] = "model_blind_pre_edit_expected_risk_criteria"
        row["technical_criterion_scores"] = scores
        row["technical_reason_codes"] = uncertain_criteria + visual_reasons
        row["technical_reviewer_id"] = (
            "codex_rule_plus_visual_audit"
            if visual_reasons
            else "codex_rule_application"
        )
        row["technical_note"] = (
            "All frozen pre-segmentation technical criteria scored 2."
            if not uncertain_criteria and not visual_reasons
            else "One or more frozen technical criteria require an actual "
            "segmentation test; this is not a recorded edit failure."
        )
        row["mask_qc_status"] = "not_tested"
        row["edit_qc_status"] = "not_tested"
        row["object_identity_preservation_status"] = "not_tested"
        row["object_identity_preservation_note"] = None
        row["object_identity_preservation_reviewer"] = None
        row["object_identity_preservation_timestamp"] = None
        output_rows.append(row)

    with args.output_jsonl.open("x", encoding="utf-8") as handle:
        for row in output_rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")

    fields = list(dict.fromkeys(key for row in output_rows for key in row))
    with args.output_csv.open("x", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for source in output_rows:
            row = dict(source)
            for field in (
                "previous_reason_codes",
                "difficulty_tags",
                "technical_reason_codes",
            ):
                row[field] = ";".join(row[field])
            row["technical_criterion_scores"] = json.dumps(
                row["technical_criterion_scores"], sort_keys=True
            )
            writer.writerow(row)

    valid_rows = [
        row for row in output_rows if row["semantic_status"] == "valid"
    ]
    summary = {
        "schema_version": "1.1",
        "input_rows": len(output_rows),
        "semantic_valid_rows": len(valid_rows),
        "technical_status_all_rows": dict(
            Counter(row["technical_status"] for row in output_rows)
        ),
        "technical_status_semantic_valid": dict(
            Counter(row["technical_status"] for row in valid_rows)
        ),
        "technical_reason_counts_semantic_valid": dict(
            Counter(
                reason
                for row in valid_rows
                for reason in row["technical_reason_codes"]
            )
        ),
        "failed_after_test_count": sum(
            row["technical_status"] == "failed_after_test"
            for row in output_rows
        ),
        "rule": {
            "directly_editable": (
                "all five frozen technical criterion scores equal 2"
            ),
            "segmentation_test_required": (
                "at least one frozen technical criterion score is below 2"
            ),
            "failed_after_test": "prohibited before recorded mask test",
            "criteria": list(TECHNICAL_CRITERIA),
            "visual_override_count": len(VISUAL_TEST_OVERRIDES),
            "semantic_visual_override_count": len(SEMANTIC_VISUAL_OVERRIDES),
            "previous_final_label_used": False,
            "model_results_used": False,
        },
    }
    args.summary_json.write_text(
        json.dumps(summary, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
