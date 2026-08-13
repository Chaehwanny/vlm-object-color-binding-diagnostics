#!/usr/bin/env python3
"""Freeze post-edit validity decisions for mini color-swap candidates."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Any


DECISIONS = {
    "mini_review_007": ("keep", "all_edit_quality_gates_pass"),
    "mini_review_024": ("revise_edit", "edited_red_target_appears_pink"),
    "mini_review_029": (
        "reject_sample",
        "object_identity_color_ambiguous_and_mask_coverage_below_threshold",
    ),
    "mini_review_038": ("keep", "all_edit_quality_gates_pass"),
}


def parse_args() -> argparse.Namespace:
    root = Path(__file__).resolve().parents[2]
    parser = argparse.ArgumentParser(description="Freeze mini post-edit audit.")
    parser.add_argument(
        "--edit-dir",
        type=Path,
        default=root / "processed/attribute_binding/edits/mini_color_swap_edits_v0.1",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=root / "processed/attribute_binding/audits/mini_edit_audit_v0.1",
    )
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    with path.open("r", encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")


def main() -> None:
    args = parse_args()
    if args.output_dir.exists() and any(args.output_dir.iterdir()) and not args.overwrite:
        raise FileExistsError(f"Output directory is not empty: {args.output_dir}")

    rows = read_jsonl(args.edit_dir / "edit_manifest.jsonl")
    row_ids = {row["segmentation_id"] for row in rows}
    if row_ids != set(DECISIONS):
        raise ValueError("Decision IDs do not match the edit manifest")

    args.output_dir.mkdir(parents=True, exist_ok=True)
    accepted: list[dict[str, Any]] = []
    revisions: list[dict[str, Any]] = []
    fields = [
        "segmentation_id",
        "object_a_before",
        "object_b_before",
        "object_a_after",
        "object_b_after",
        "selected_color_fraction_a",
        "selected_color_fraction_b",
        "outside_edit_pixels_identical",
        "post_edit_decision",
        "post_edit_reason",
        "comparison_path",
    ]
    with (args.output_dir / "edit_audit_decisions.csv").open(
        "w", encoding="utf-8", newline=""
    ) as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for row in rows:
            decision, reason = DECISIONS[row["segmentation_id"]]
            object_a, object_b = row["object_a"], row["object_b"]
            writer.writerow(
                {
                    "segmentation_id": row["segmentation_id"],
                    "object_a_before": f"{object_a['color']} {object_a['label']}",
                    "object_b_before": f"{object_b['color']} {object_b['label']}",
                    "object_a_after": f"{object_b['color']} {object_a['label']}",
                    "object_b_after": f"{object_a['color']} {object_b['label']}",
                    "selected_color_fraction_a": row["selected_color_fraction_a"],
                    "selected_color_fraction_b": row["selected_color_fraction_b"],
                    "outside_edit_pixels_identical": row["outside_edit_pixels_identical"],
                    "post_edit_decision": decision,
                    "post_edit_reason": reason,
                    "comparison_path": str(args.edit_dir / row["comparison_path"]),
                }
            )
            enriched = {
                **row,
                "post_edit_decision": decision,
                "post_edit_reason": reason,
                "second_human_review_status": "pending",
            }
            if decision == "keep":
                accepted.append(enriched)
            elif decision == "revise_edit":
                revisions.append(enriched)

    write_jsonl(args.output_dir / "accepted_edits.jsonl", accepted)
    write_jsonl(args.output_dir / "revision_candidates.jsonl", revisions)
    summary = {
        "schema_version": "0.1",
        "reviewed_edits": len(rows),
        "accepted_current_edits": len(accepted),
        "revision_candidates": len(revisions),
        "rejected_samples": len(rows) - len(accepted) - len(revisions),
        "accepted_ids": [row["segmentation_id"] for row in accepted],
        "revision_ids": [row["segmentation_id"] for row in revisions],
        "model_outputs_used_for_decision": False,
        "second_human_review_required": True,
    }
    (args.output_dir / "summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
