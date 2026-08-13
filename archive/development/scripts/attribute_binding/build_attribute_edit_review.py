#!/usr/bin/env python3
"""Build the human quality-gate sheet for color-swap edits."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Any


def parse_args() -> argparse.Namespace:
    project_root = Path(__file__).resolve().parents[2]
    parser = argparse.ArgumentParser(description="Build edit review sheet.")
    parser.add_argument(
        "--manifest",
        type=Path,
        default=project_root
        / "processed/attribute_binding/edits/color_swap_edits_v0.2/edit_manifest.jsonl",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=project_root
        / "processed/attribute_binding/edits/color_swap_edits_v0.2/human_edit_review.csv",
    )
    return parser.parse_args()


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    with path.open("r", encoding="utf-8") as input_file:
        return [json.loads(line) for line in input_file]


def main() -> None:
    args = parse_args()
    fields = [
        "segmentation_id",
        "selection_role",
        "object_a_before",
        "object_b_before",
        "object_a_after",
        "object_b_after",
        "comparison_path",
        "mask_boundaries_pass_fail",
        "target_colors_clear_pass_fail",
        "neutral_regions_preserved_pass_fail",
        "identity_preserved_pass_fail",
        "background_unchanged_pass_fail",
        "before_answers_unambiguous_pass_fail",
        "after_answers_unambiguous_pass_fail",
        "final_keep_reject_pending",
        "reject_reason",
        "notes",
    ]
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w", encoding="utf-8", newline="") as output_file:
        writer = csv.DictWriter(output_file, fieldnames=fields)
        writer.writeheader()
        for row in read_jsonl(args.manifest):
            object_a = row["object_a"]
            object_b = row["object_b"]
            writer.writerow(
                {
                    "segmentation_id": row["segmentation_id"],
                    "selection_role": row["selection_role"],
                    "object_a_before": f"{object_a['color']} {object_a['label']}",
                    "object_b_before": f"{object_b['color']} {object_b['label']}",
                    "object_a_after": f"{object_b['color']} {object_a['label']}",
                    "object_b_after": f"{object_a['color']} {object_b['label']}",
                    "comparison_path": row["comparison_path"],
                    "final_keep_reject_pending": "pending",
                }
            )


if __name__ == "__main__":
    main()
