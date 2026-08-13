#!/usr/bin/env python3
"""Build the six-image mask-attempt package for binding feasibility."""

from __future__ import annotations

import argparse
import csv
import json
import shutil
from pathlib import Path
from typing import Any


PRIMARY_IDS = (
    "binding_v02_036",
    "binding_v02_038",
    "binding_v02_052",
    "binding_v02_030",
)

RESERVE_IDS = {}

REPLACEMENT_IDS = {
    "replacement_004": (
        "Provisional blue-green coverage candidate. Objects are on separate "
        "people, but garment masks and illumination require an edit-quality check."
    ),
}


def parse_args() -> argparse.Namespace:
    project_root = Path(__file__).resolve().parents[2]
    parser = argparse.ArgumentParser(description="Build segmentation candidates.")
    parser.add_argument(
        "--shortlist-dir",
        type=Path,
        default=project_root
        / "processed/attribute_binding/reviews/feasibility_shortlist_v0.2",
    )
    parser.add_argument(
        "--replacement-dir",
        type=Path,
        default=project_root
        / "processed/attribute_binding/reviews/replacement_review_pool_v0.2",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=project_root
        / "processed/attribute_binding/segmentation/segmentation_candidates_v0.2",
    )
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    with path.open("r", encoding="utf-8") as input_file:
        return [json.loads(line) for line in input_file]


def main() -> None:
    args = parse_args()
    if args.output_dir.exists() and any(args.output_dir.iterdir()) and not args.overwrite:
        raise FileExistsError(
            f"Output directory is not empty: {args.output_dir}. Use --overwrite."
        )

    shortlist_rows = {
        row["candidate_id"]: row
        for row in read_jsonl(args.shortlist_dir / "shortlist.jsonl")
    }
    replacement_rows = {
        row["replacement_id"]: row
        for row in read_jsonl(args.replacement_dir / "replacement_pool.jsonl")
    }

    args.output_dir.mkdir(parents=True, exist_ok=True)
    previews_dir = args.output_dir / "previews"
    previews_dir.mkdir(parents=True, exist_ok=True)
    output_rows: list[dict[str, Any]] = []

    for source_id in PRIMARY_IDS:
        source = shortlist_rows[source_id]
        output_rows.append(
            {
                **source,
                "segmentation_id": source_id,
                "selection_role": "primary_mask_attempt",
                "selection_note": source["precheck_note"],
                "source_preview": args.shortlist_dir / source["shortlist_preview_path"],
            }
        )

    for source_id, note in RESERVE_IDS.items():
        source = shortlist_rows[source_id]
        output_rows.append(
            {
                **source,
                "segmentation_id": source_id,
                "selection_role": "reserve_mask_attempt",
                "selection_note": note,
                "source_preview": args.shortlist_dir / source["shortlist_preview_path"],
            }
        )

    for source_id, note in REPLACEMENT_IDS.items():
        source = replacement_rows[source_id]
        output_rows.append(
            {
                **source,
                "segmentation_id": source_id,
                "selection_role": "accepted_replacement",
                "selection_note": note,
                "source_preview": args.replacement_dir / source["preview_path"],
            }
        )

    for row in output_rows:
        target = previews_dir / f"{row['segmentation_id']}.jpg"
        shutil.copy2(row.pop("source_preview"), target)
        row["segmentation_preview_path"] = str(target.relative_to(args.output_dir))

    with (args.output_dir / "segmentation_candidates.jsonl").open(
        "w", encoding="utf-8"
    ) as output_file:
        for row in output_rows:
            output_file.write(json.dumps(row, ensure_ascii=False) + "\n")

    fields = [
        "segmentation_id",
        "selection_role",
        "color_pair",
        "object_a_label",
        "object_a_color",
        "object_b_label",
        "object_b_color",
        "preview_path",
        "selection_note",
        "mask_a_quality_pass_fail",
        "mask_b_quality_pass_fail",
        "edited_color_a_pass_fail",
        "edited_color_b_pass_fail",
        "identity_preserved_pass_fail",
        "background_unchanged_pass_fail",
        "human_answer_unambiguous_pass_fail",
        "final_edit_gate_keep_reject_pending",
        "reject_reason",
        "notes",
    ]
    with (args.output_dir / "edit_quality_review.csv").open(
        "w", encoding="utf-8", newline=""
    ) as output_file:
        writer = csv.DictWriter(output_file, fieldnames=fields)
        writer.writeheader()
        for row in output_rows:
            writer.writerow(
                {
                    "segmentation_id": row["segmentation_id"],
                    "selection_role": row["selection_role"],
                    "color_pair": row["color_pair"],
                    "object_a_label": row["object_a"]["label"],
                    "object_a_color": row["object_a"]["color"],
                    "object_b_label": row["object_b"]["label"],
                    "object_b_color": row["object_b"]["color"],
                    "preview_path": row["segmentation_preview_path"],
                    "selection_note": row["selection_note"],
                    "final_edit_gate_keep_reject_pending": "pending",
                }
            )

    (args.output_dir / "README.md").write_text(
        """# Attribute-Binding Segmentation Candidates v0.2

This package contains five human-accepted feasibility edit candidates.

- Four candidates came from the visually screened feasibility shortlist.
- One accepted replacement adds blue-green coverage.
- The target is at least five edited samples that pass every quality-gate field
  in `edit_quality_review.csv`.
- Do not run VLM inference until the five-sample edited set is frozen.

The original shortlist and replacement pool remain unchanged as an audit trail.
""",
        encoding="utf-8",
    )
    print(
        f"Wrote {len(output_rows)} segmentation candidates to {args.output_dir}"
    )


if __name__ == "__main__":
    main()
