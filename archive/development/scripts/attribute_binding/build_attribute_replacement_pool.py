#!/usr/bin/env python3
"""Build a targeted blue-green/blue-red replacement review pool."""

from __future__ import annotations

import argparse
import csv
import json
from collections import Counter
from pathlib import Path
from typing import Any

from create_attribute_review_pool_v0_2 import (
    TIER_ORDER,
    canonical_label_pair,
    create_preview,
)


TARGET_PAIRS = ("blue_green", "blue_red")


def parse_args() -> argparse.Namespace:
    project_root = Path(__file__).resolve().parents[2]
    parser = argparse.ArgumentParser(description="Build replacement review pool.")
    parser.add_argument(
        "--input",
        type=Path,
        default=project_root
        / "processed/attribute_binding/candidates/gqa/candidates_color_pairs_v0.2.jsonl",
    )
    parser.add_argument(
        "--exclude-manifest",
        type=Path,
        default=project_root
        / "processed/attribute_binding/reviews/feasibility_shortlist_v0.2/shortlist.jsonl",
    )
    parser.add_argument(
        "--image-root",
        type=Path,
        default=project_root / "raw/GQA-Scene-Graph",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=project_root
        / "processed/attribute_binding/reviews/replacement_review_pool_v0.2",
    )
    parser.add_argument("--per-color-pair", type=int, default=12)
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    with path.open("r", encoding="utf-8") as input_file:
        return [json.loads(line) for line in input_file]


def rank_key(row: dict[str, Any]) -> tuple[Any, ...]:
    minimum_area = min(
        row["object_a"]["bbox_area"], row["object_b"]["bbox_area"]
    )
    return (
        TIER_ORDER[row["editability_priority"]],
        row["bbox_iou"] > 0,
        row["bbox_iou"],
        -minimum_area,
        row["image_id"],
    )


def select_rows(
    rows: list[dict[str, Any]], excluded_images: set[str], per_pair: int
) -> list[dict[str, Any]]:
    selected = []
    used_images = set(excluded_images)
    for color_pair in TARGET_PAIRS:
        pool = sorted(
            (row for row in rows if row["color_pair"] == color_pair),
            key=rank_key,
        )
        used_label_pairs: set[tuple[str, str]] = set()
        pair_rows = []
        for require_unique_labels in (True, False):
            for row in pool:
                if len(pair_rows) == per_pair:
                    break
                if row["image_path"] in used_images:
                    continue
                label_pair = canonical_label_pair(row)
                if require_unique_labels and label_pair in used_label_pairs:
                    continue
                pair_rows.append(row)
                used_images.add(row["image_path"])
                used_label_pairs.add(label_pair)
            if len(pair_rows) == per_pair:
                break
        if len(pair_rows) < per_pair:
            raise RuntimeError(
                f"Only selected {len(pair_rows)}/{per_pair} for {color_pair}"
            )
        selected.extend(pair_rows)
    return selected


def main() -> None:
    args = parse_args()
    if args.output_dir.exists() and any(args.output_dir.iterdir()) and not args.overwrite:
        raise FileExistsError(
            f"Output directory is not empty: {args.output_dir}. Use --overwrite."
        )
    candidates = read_jsonl(args.input)
    excluded_images = {
        row["image_path"] for row in read_jsonl(args.exclude_manifest)
    }
    selected = select_rows(candidates, excluded_images, args.per_color_pair)

    args.output_dir.mkdir(parents=True, exist_ok=True)
    previews_dir = args.output_dir / "previews"
    previews_dir.mkdir(parents=True, exist_ok=True)
    fields = [
        "replacement_id",
        "color_pair",
        "editability_priority",
        "image_path",
        "preview_path",
        "object_a_label",
        "object_a_color",
        "object_b_label",
        "object_b_color",
        "bbox_iou",
        "minimum_bbox_area",
        "human_status_keep_borderline_reject",
        "reject_reason",
        "notes",
    ]

    with (args.output_dir / "replacement_pool.jsonl").open(
        "w", encoding="utf-8"
    ) as manifest_file, (args.output_dir / "human_review.csv").open(
        "w", encoding="utf-8", newline=""
    ) as review_file:
        writer = csv.DictWriter(review_file, fieldnames=fields)
        writer.writeheader()
        for index, row in enumerate(selected, start=1):
            replacement_id = f"replacement_{index:03d}"
            preview_path = previews_dir / f"{replacement_id}.jpg"
            create_preview(row, args.image_root / row["image_path"], preview_path)
            enriched = {
                **row,
                "replacement_id": replacement_id,
                "preview_path": str(preview_path.relative_to(args.output_dir)),
                "selection_rule": {
                    "target_pairs": list(TARGET_PAIRS),
                    "shortlist_images_excluded": True,
                    "editability_tier_then_zero_iou_then_area": True,
                    "label_pair_diversity_preferred": True,
                },
            }
            manifest_file.write(json.dumps(enriched, ensure_ascii=False) + "\n")
            writer.writerow(
                {
                    "replacement_id": replacement_id,
                    "color_pair": row["color_pair"],
                    "editability_priority": row["editability_priority"],
                    "image_path": row["image_path"],
                    "preview_path": enriched["preview_path"],
                    "object_a_label": row["object_a"]["label"],
                    "object_a_color": row["object_a"]["color"],
                    "object_b_label": row["object_b"]["label"],
                    "object_b_color": row["object_b"]["color"],
                    "bbox_iou": row["bbox_iou"],
                    "minimum_bbox_area": min(
                        row["object_a"]["bbox_area"],
                        row["object_b"]["bbox_area"],
                    ),
                }
            )

    summary = {
        "schema_version": "0.2",
        "selected": len(selected),
        "selected_by_color_pair": dict(
            sorted(Counter(row["color_pair"] for row in selected).items())
        ),
        "selected_by_priority": dict(
            sorted(Counter(row["editability_priority"] for row in selected).items())
        ),
        "unique_images": len({row["image_path"] for row in selected}),
        "shortlist_images_excluded": len(excluded_images),
    }
    (args.output_dir / "summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
