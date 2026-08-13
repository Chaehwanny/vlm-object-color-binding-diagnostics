#!/usr/bin/env python3
"""Freeze human validity decisions for PACO pixel-color candidates."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Any


DECISIONS = {
    "paco_color_001": ("reject", "near_duplicate_scene_family"),
    "paco_color_002": ("reject", "noncanonical_source_colors"),
    "paco_color_003": ("reject", "mask_content_color_contamination_and_logo"),
    "paco_color_004": ("reject", "source_red_appears_brown"),
    "paco_color_005": ("keep", "clear_distinct_objects_plausible_swap"),
    "paco_color_006": ("reject", "text_pattern_and_mask_content_contamination"),
    "paco_color_007": ("keep", "clear_distinct_objects_plausible_swap"),
    "paco_color_008": ("reject", "source_yellow_appears_olive"),
    "paco_color_009": ("keep", "clear_distinct_objects_plausible_swap"),
    "paco_color_010": ("reject", "severe_occlusion_and_small_target"),
}


def parse_args() -> argparse.Namespace:
    root = Path(__file__).resolve().parents[2]
    parser = argparse.ArgumentParser(description="Freeze PACO color audit.")
    parser.add_argument(
        "--screen-dir",
        type=Path,
        default=root / "processed/attribute_binding/reviews/paco_color_screen_v0.2",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=root / "processed/attribute_binding/audits/paco_color_audit_v0.1",
    )
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    with path.open("r", encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def main() -> None:
    args = parse_args()
    if args.output_dir.exists() and any(args.output_dir.iterdir()) and not args.overwrite:
        raise FileExistsError(f"Output directory is not empty: {args.output_dir}")
    rows = read_jsonl(args.screen_dir / "accepted_candidates.jsonl")
    if {row["segmentation_id"] for row in rows} != set(DECISIONS):
        raise ValueError("Decision IDs do not match PACO color candidates")

    args.output_dir.mkdir(parents=True, exist_ok=True)
    accepted = []
    fields = [
        "segmentation_id",
        "image_id",
        "object_a",
        "object_b",
        "color_pair",
        "audit_decision",
        "audit_reason",
        "preview_path",
    ]
    with (args.output_dir / "audit_decisions.csv").open(
        "w", encoding="utf-8", newline=""
    ) as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for row in rows:
            decision, reason = DECISIONS[row["segmentation_id"]]
            writer.writerow(
                {
                    "segmentation_id": row["segmentation_id"],
                    "image_id": row["image_id"],
                    "object_a": f"{row['object_a']['color']} {row['object_a']['label']}",
                    "object_b": f"{row['object_b']['color']} {row['object_b']['label']}",
                    "color_pair": row["color_pair"],
                    "audit_decision": decision,
                    "audit_reason": reason,
                    "preview_path": str(args.screen_dir / row["preview_path"]),
                }
            )
            if decision == "keep":
                accepted.append(
                    {
                        **row,
                        "selection_role": "paco_human_approved_edit_attempt",
                        "human_review_status": "keep_for_edit_attempt",
                        "human_review_reason": reason,
                        "development_data_only": True,
                    }
                )

    with (args.output_dir / "accepted_candidates.jsonl").open(
        "w", encoding="utf-8"
    ) as handle:
        for row in accepted:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")
    summary = {
        "schema_version": "0.1",
        "reviewed": len(rows),
        "accepted_for_edit_attempt": len(accepted),
        "rejected": len(rows) - len(accepted),
        "accepted_ids": [row["segmentation_id"] for row in accepted],
        "model_outputs_used_for_decision": False,
        "second_human_review_required": True,
    }
    (args.output_dir / "summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
