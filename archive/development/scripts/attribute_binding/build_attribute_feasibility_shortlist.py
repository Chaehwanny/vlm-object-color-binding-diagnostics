#!/usr/bin/env python3
"""Build the pre-mask shortlist for attribute-binding feasibility v0.2."""

from __future__ import annotations

import argparse
import csv
import json
import shutil
from pathlib import Path
from typing import Any


SHORTLIST = {
    "binding_v02_008": {
        "role": "excluded_pre_mask",
        "precheck_note": "Excluded by human review: backpack and jeans share one wearer and a physically entangled boundary.",
    },
    "binding_v02_015": {
        "role": "excluded_pre_mask",
        "precheck_note": "Excluded by human review: jacket and pants share one wearer and an adjacent garment boundary.",
    },
    "binding_v02_030": {
        "role": "primary",
        "precheck_note": "Clear yellow equipment box and blue jeans; distinct manufactured targets.",
    },
    "binding_v02_036": {
        "role": "primary",
        "precheck_note": "Clear green jacket and red umbrella; distinct colors and object identities.",
    },
    "binding_v02_042": {
        "role": "primary",
        "precheck_note": "Separate green car and yellow bus; whole-object color edit candidate.",
    },
    "binding_v02_052": {
        "role": "primary",
        "precheck_note": "Isolated yellow coat and red chair; clear labels and dominant colors.",
    },
    "binding_v02_038": {
        "role": "primary",
        "precheck_note": "Green bicycle and red shirt are clear; thin bicycle structure may challenge masking.",
    },
}


def parse_args() -> argparse.Namespace:
    project_root = Path(__file__).resolve().parents[2]
    parser = argparse.ArgumentParser(description="Build color-binding shortlist v0.2.")
    parser.add_argument(
        "--review-pool",
        type=Path,
        default=project_root
        / "processed/attribute_binding/reviews/review_pool_v0.2/review_pool.jsonl",
    )
    parser.add_argument(
        "--review-dir",
        type=Path,
        default=project_root / "processed/attribute_binding/reviews/review_pool_v0.2",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=project_root
        / "processed/attribute_binding/reviews/feasibility_shortlist_v0.2",
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

    source_rows = {
        row["candidate_id"]: row for row in read_jsonl(args.review_pool)
    }
    missing = sorted(set(SHORTLIST) - set(source_rows))
    if missing:
        raise KeyError(f"Shortlist IDs missing from review pool: {missing}")

    args.output_dir.mkdir(parents=True, exist_ok=True)
    previews_dir = args.output_dir / "previews"
    previews_dir.mkdir(parents=True, exist_ok=True)
    rows = []

    for candidate_id, selection in SHORTLIST.items():
        source = source_rows[candidate_id]
        preview_source = args.review_dir / source["preview_path"]
        preview_target = previews_dir / preview_source.name
        shutil.copy2(preview_source, preview_target)
        rows.append(
            {
                **source,
                "shortlist_role": selection["role"],
                "precheck_note": selection["precheck_note"],
                "shortlist_preview_path": str(
                    preview_target.relative_to(args.output_dir)
                ),
            }
        )

    with (args.output_dir / "shortlist.jsonl").open(
        "w", encoding="utf-8"
    ) as output_file:
        for row in rows:
            output_file.write(json.dumps(row, ensure_ascii=False) + "\n")

    fields = [
        "candidate_id",
        "shortlist_role",
        "color_pair",
        "object_a_label",
        "object_a_color",
        "object_b_label",
        "object_b_color",
        "preview_path",
        "precheck_note",
        "reviewer_1_data_valid_yes_no_uncertain",
        "reviewer_1_swap_plausible_yes_no_uncertain",
        "reviewer_1_mask_feasible_yes_no_uncertain",
        "reviewer_2_data_valid_yes_no_uncertain",
        "reviewer_2_swap_plausible_yes_no_uncertain",
        "reviewer_2_mask_feasible_yes_no_uncertain",
        "consensus_keep_for_mask_yes_no_pending",
        "consensus_exclude_reason",
        "notes",
    ]
    with (args.output_dir / "shortlist_review.csv").open(
        "w", encoding="utf-8", newline=""
    ) as output_file:
        writer = csv.DictWriter(output_file, fieldnames=fields)
        writer.writeheader()
        for row in rows:
            writer.writerow(
                {
                    "candidate_id": row["candidate_id"],
                    "shortlist_role": row["shortlist_role"],
                    "color_pair": row["color_pair"],
                    "object_a_label": row["object_a"]["label"],
                    "object_a_color": row["object_a"]["color"],
                    "object_b_label": row["object_b"]["label"],
                    "object_b_color": row["object_b"]["color"],
                    "preview_path": row["shortlist_preview_path"],
                    "precheck_note": row["precheck_note"],
                    "consensus_keep_for_mask_yes_no_pending": (
                        "no" if row["shortlist_role"] == "excluded_pre_mask" else "pending"
                    ),
                    "consensus_exclude_reason": (
                        "physical_entanglement_shared_owner"
                        if row["shortlist_role"] == "excluded_pre_mask"
                        else ""
                    ),
                }
            )

    (args.output_dir / "README.md").write_text(
        """# Attribute Binding Feasibility Shortlist v0.2

This is a pre-mask shortlist, not an accepted feasibility dataset.

- Five primary candidates advance to segmentation-mask feasibility.
- Two rejected candidates remain in the audit trail with exclusion reasons.
- Color-pair balance is not required for this five-image technical gate; it is
  enforced when constructing the 20-30 image mini development set.
- Selection used metadata filtering plus visual precheck only.
- Final inclusion requires human review and a valid segmentation/color edit.
- Model inference is prohibited until at least five edited pairs pass the
  editing quality gate.

Reviewers must fill shortlist_review.csv without consulting any future model
outputs. A primary candidate may be replaced by a new review-pool candidate for
data-quality reasons only.
""",
        encoding="utf-8",
    )
    print(
        f"Wrote {sum(row['shortlist_role'] == 'primary' for row in rows)} primary "
        f"and {sum(row['shortlist_role'] == 'excluded_pre_mask' for row in rows)} excluded candidates "
        f"to {args.output_dir}"
    )


if __name__ == "__main__":
    main()
