#!/usr/bin/env python3
"""Record human exclusions and freeze mini-set mask candidates."""

from __future__ import annotations

import argparse
import csv
import json
import shutil
from pathlib import Path
from typing import Any


EXCLUDED_IDS = {
    "mini_review_001",
    "mini_review_004",
    "mini_review_006",
    "mini_review_009",
    "mini_review_011",
    "mini_review_012",
    "mini_review_013",
    "mini_review_014",
    "mini_review_015",
    "mini_review_016",
    "mini_review_019",
    "mini_review_021",
    "mini_review_023",
    "mini_review_030",
    "mini_review_031",
    "mini_review_033",
    "mini_review_034",
    "mini_review_037",
}


def parse_args() -> argparse.Namespace:
    project_root = Path(__file__).resolve().parents[2]
    parser = argparse.ArgumentParser(description="Freeze human-approved mask candidates.")
    parser.add_argument(
        "--review-dir",
        type=Path,
        default=project_root / "processed/attribute_binding/reviews/mini_review_pool_v0.1",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=project_root / "processed/attribute_binding/segmentation/mini_mask_candidates_v0.1",
    )
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    with path.open("r", encoding="utf-8") as input_file:
        return [json.loads(line) for line in input_file]


def main() -> None:
    args = parse_args()
    if args.output_dir.exists() and any(args.output_dir.iterdir()) and not args.overwrite:
        raise FileExistsError(f"Output directory is not empty: {args.output_dir}. Use --overwrite.")

    rows = read_jsonl(args.review_dir / "review_pool.jsonl")
    known_ids = {row["mini_review_id"] for row in rows}
    unknown = EXCLUDED_IDS - known_ids
    if unknown:
        raise ValueError(f"Unknown excluded IDs: {sorted(unknown)}")

    args.output_dir.mkdir(parents=True, exist_ok=True)
    previews_dir = args.output_dir / "previews"
    previews_dir.mkdir(parents=True, exist_ok=True)
    accepted = []
    decision_fields = [
        "mini_review_id",
        "color_pair",
        "object_a",
        "object_b",
        "human_decision",
        "exclude_reason",
        "source_preview_path",
    ]
    with (args.output_dir / "human_decisions.csv").open(
        "w", encoding="utf-8", newline=""
    ) as decision_file:
        writer = csv.DictWriter(decision_file, fieldnames=decision_fields)
        writer.writeheader()
        for row in rows:
            is_excluded = row["mini_review_id"] in EXCLUDED_IDS
            writer.writerow(
                {
                    "mini_review_id": row["mini_review_id"],
                    "color_pair": row["color_pair"],
                    "object_a": f"{row['object_a']['color']} {row['object_a']['label']}",
                    "object_b": f"{row['object_b']['color']} {row['object_b']['label']}",
                    "human_decision": "reject" if is_excluded else "keep_for_mask",
                    "exclude_reason": "human_visual_review_reject" if is_excluded else "",
                    "source_preview_path": row["preview_path"],
                }
            )
            if is_excluded:
                continue
            source_preview = args.review_dir / row["preview_path"]
            target_preview = previews_dir / source_preview.name
            shutil.copy2(source_preview, target_preview)
            accepted.append(
                {
                    **row,
                    "segmentation_id": row["mini_review_id"],
                    "selection_role": "mini_human_approved_mask_attempt",
                    "selection_note": "Kept after model-blind human visual review.",
                    "human_review_status": "keep_for_mask",
                    "mask_candidate_preview_path": str(
                        target_preview.relative_to(args.output_dir)
                    ),
                }
            )

    with (args.output_dir / "mask_candidates.jsonl").open(
        "w", encoding="utf-8"
    ) as output_file:
        for row in accepted:
            output_file.write(json.dumps(row, ensure_ascii=False) + "\n")

    summary = {
        "schema_version": "0.1",
        "reviewed": len(rows),
        "human_rejected": len(EXCLUDED_IDS),
        "kept_for_mask": len(accepted),
        "model_outputs_used_for_decision": False,
        "excluded_ids": sorted(EXCLUDED_IDS),
    }
    (args.output_dir / "summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
