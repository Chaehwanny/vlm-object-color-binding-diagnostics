#!/usr/bin/env python3
"""Build and validate the frozen Completion attrition summary."""

from __future__ import annotations

import argparse
import hashlib
import json
from collections import Counter
from pathlib import Path
from typing import Any


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--completion-manifest", type=Path, required=True)
    parser.add_argument("--segmentation-results", type=Path, required=True)
    parser.add_argument("--reviewed-mask-results", type=Path, required=True)
    parser.add_argument("--semantic-purity-review", type=Path, required=True)
    parser.add_argument("--pre-edit-eligibility", type=Path, required=True)
    parser.add_argument("--color-edit-results", type=Path, required=True)
    parser.add_argument("--reviewed-color-edit-results", type=Path, required=True)
    parser.add_argument("--controlled-eligibility", type=Path, required=True)
    parser.add_argument("--controlled-results", type=Path, required=True)
    parser.add_argument("--reviewed-controlled-results", type=Path, required=True)
    parser.add_argument("--gate-config", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
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


def ids(rows: list[dict[str, Any]], label: str) -> list[str]:
    values = [row.get("candidate_id") for row in rows]
    if None in values or len(values) != len(set(values)):
        raise ValueError(f"{label}: missing or duplicate candidate_id")
    return values  # type: ignore[return-value]


def require_subsequence(parent: list[str], child: list[str], label: str) -> None:
    expected = [candidate_id for candidate_id in parent if candidate_id in set(child)]
    if child != expected:
        raise ValueError(f"{label}: candidates are new, duplicated, or out of frozen order")


def atomic_json(path: Path, value: dict[str, Any]) -> None:
    if path.exists():
        raise FileExistsError(f"Refusing to overwrite: {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp")
    if temporary.exists():
        raise FileExistsError(f"Stale temporary file: {temporary}")
    temporary.write_text(
        json.dumps(value, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    temporary.replace(path)


def main() -> None:
    args = parse_args()
    paths = {
        "completion_manifest": args.completion_manifest,
        "segmentation_results": args.segmentation_results,
        "reviewed_mask_results": args.reviewed_mask_results,
        "semantic_purity_review": args.semantic_purity_review,
        "pre_edit_eligibility": args.pre_edit_eligibility,
        "color_edit_results": args.color_edit_results,
        "reviewed_color_edit_results": args.reviewed_color_edit_results,
        "controlled_eligibility": args.controlled_eligibility,
        "controlled_results": args.controlled_results,
        "reviewed_controlled_results": args.reviewed_controlled_results,
        "gate_config": args.gate_config,
    }
    rows = {name: read_jsonl(path) for name, path in paths.items() if name != "gate_config"}
    row_ids = {name: ids(value, name) for name, value in rows.items()}
    frozen = row_ids["completion_manifest"]
    if len(frozen) != 25:
        raise ValueError(f"Completion manifest must contain 25 candidates, found {len(frozen)}")
    if row_ids["segmentation_results"] != frozen or row_ids["reviewed_mask_results"] != frozen:
        raise ValueError("Segmentation and reviewed-mask manifests must preserve all 25 frozen rows")
    if row_ids["semantic_purity_review"] != frozen or row_ids["pre_edit_eligibility"] != frozen:
        raise ValueError("Semantic-purity and pre-edit attrition manifests must preserve all 25 rows")

    pre_edit_eligible = [
        row["candidate_id"] for row in rows["pre_edit_eligibility"]
        if row.get("candidate_pre_edit_eligible") is True
    ]
    require_subsequence(frozen, pre_edit_eligible, "pre-edit eligible")
    if row_ids["color_edit_results"] != pre_edit_eligible:
        raise ValueError("Color-edit results do not exactly match pre-edit eligible candidates")
    if row_ids["reviewed_color_edit_results"] != pre_edit_eligible:
        raise ValueError("Reviewed color-edit results do not preserve pre-edit eligible candidates")

    controlled_eligible = [
        row["candidate_id"] for row in rows["controlled_eligibility"]
        if row.get("controlled_eligible") is True
    ]
    require_subsequence(pre_edit_eligible, controlled_eligible, "Controlled eligible")
    if row_ids["controlled_results"] != controlled_eligible:
        raise ValueError("Controlled results do not exactly match Controlled eligible candidates")
    if row_ids["reviewed_controlled_results"] != controlled_eligible:
        raise ValueError("Reviewed Controlled results do not preserve Controlled eligible candidates")

    semantic_pass = {
        row["candidate_id"] for row in rows["semantic_purity_review"]
        if row.get("object_a_semantic_purity_status") == "pass"
        and row.get("object_b_semantic_purity_status") == "pass"
    }
    natural_pass = {
        row["candidate_id"] for row in rows["reviewed_color_edit_results"]
        if row.get("edit_qc_status") == "pass"
        and row.get("object_identity_preservation_status") == "pass"
    }
    final_pass = {
        row["candidate_id"] for row in rows["reviewed_controlled_results"]
        if row.get("controlled_qc_status") == "pass"
        and row.get("four_image_qc_status") == "pass"
    }
    reason_counts: Counter[str] = Counter()
    multiple_rules = 0
    for row in rows["pre_edit_eligibility"]:
        if row.get("candidate_pre_edit_eligible") is True:
            continue
        reasons = set(row.get("triggered_rules", []))
        for reason in (
            "non_target_mask_inclusion",
            "selected_color_ratio_below_0_05",
            "hue_distance_below_45",
            "low_saturation_above_0_50",
        ):
            if reason in reasons:
                reason_counts[reason] += 1
        if len(reasons) > 1:
            multiple_rules += 1

    summary = {
        "schema_version": "1.1",
        "summary_version": "completion_attrition_summary_v1.1",
        "subset_name": "completion",
        "completion_start_count": len(frozen),
        "segmentation_generated_count": len(rows["segmentation_results"]),
        "mask_semantic_purity_pass_count": len(semantic_pass),
        "pre_edit_gate_eligible_count": len(pre_edit_eligible),
        "pre_edit_gate_excluded_count": len(frozen) - len(pre_edit_eligible),
        "color_edit_generated_count": len(rows["color_edit_results"]),
        "natural_edit_qc_pass_count": len(natural_pass),
        "controlled_eligible_count": len(controlled_eligible),
        "controlled_generated_count": len(rows["controlled_results"]),
        "four_image_qc_pass_count": len(final_pass),
        "final_matched_valid_count": len(final_pass),
        "gate_exclusion_reason_counts_overlapping": {
            "non_target_mask_inclusion": reason_counts["non_target_mask_inclusion"],
            "selected_color_ratio_below_0_05": reason_counts["selected_color_ratio_below_0_05"],
            "hue_distance_below_45": reason_counts["hue_distance_below_45"],
            "low_saturation_above_0_50": reason_counts["low_saturation_above_0_50"],
            "multiple_rules": multiple_rules,
        },
        "frozen_candidate_ids": frozen,
        "pre_edit_eligible_candidate_ids": pre_edit_eligible,
        "controlled_eligible_candidate_ids": controlled_eligible,
        "final_matched_valid_candidate_ids": [cid for cid in frozen if cid in final_pass],
        "candidate_reselection": False,
        "candidate_replacement": False,
        "order_preserved": True,
        "source_files": {
            name: {"path": path.as_posix(), "sha256": sha256(path)}
            for name, path in paths.items()
        },
        "validation_errors": 0,
    }
    atomic_json(args.output, summary)
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
