#!/usr/bin/env python3
"""Extract PACO-LVIS object-mask pairs before pixel-based color screening."""

from __future__ import annotations

import argparse
import itertools
import json
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

from extract_paco_attribute_pairs import EDITABLE_CATEGORIES, bbox_iou, clean_label


def parse_args() -> argparse.Namespace:
    root = Path(__file__).resolve().parents[2]
    parser = argparse.ArgumentParser(description="Extract PACO object-mask pairs.")
    parser.add_argument(
        "--annotation",
        type=Path,
        default=root / "raw/PACO-LVIS/annotations/paco_lvis_v1_train.json",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=root / "processed/attribute_binding/candidates/paco/paco_mask_pairs_train_v0.1",
    )
    parser.add_argument("--min-mask-area", type=float, default=0.015)
    parser.add_argument("--min-bbox-area", type=float, default=0.02)
    parser.add_argument("--max-bbox-area", type=float, default=0.50)
    parser.add_argument("--max-iou", type=float, default=0.05)
    parser.add_argument("--max-candidates", type=int, default=500)
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.output_dir.exists() and any(args.output_dir.iterdir()) and not args.overwrite:
        raise FileExistsError(f"Output directory is not empty: {args.output_dir}")

    with args.annotation.open("r", encoding="utf-8") as handle:
        data = json.load(handle)

    images = {image["id"]: image for image in data["images"]}
    categories = {category["id"]: category for category in data["categories"]}
    objects_by_image: dict[int, list[dict[str, Any]]] = defaultdict(list)
    for ann in data["annotations"]:
        category = categories[ann["category_id"]]
        if category["supercategory"] != "OBJECT" or ann["id"] != ann["obj_ann_id"]:
            continue
        objects_by_image[ann["image_id"]].append(ann)

    pairs = []
    rejection_counts: Counter[str] = Counter()
    for image_id, annotations in objects_by_image.items():
        image = images[image_id]
        image_area = image["width"] * image["height"]
        label_counts = Counter(ann["category_id"] for ann in annotations)
        eligible = []
        for ann in annotations:
            raw_label = categories[ann["category_id"]]["name"]
            _, _, width, height = ann["bbox"]
            bbox_fraction = width * height / image_area
            mask_fraction = ann["area"] / image_area
            if raw_label not in EDITABLE_CATEGORIES:
                rejection_counts["noneditable_category"] += 1
                continue
            if label_counts[ann["category_id"]] != 1:
                rejection_counts["competing_same_label"] += 1
                continue
            if bbox_fraction < args.min_bbox_area:
                rejection_counts["bbox_too_small"] += 1
                continue
            if bbox_fraction > args.max_bbox_area:
                rejection_counts["bbox_too_large"] += 1
                continue
            if mask_fraction < args.min_mask_area:
                rejection_counts["mask_too_small"] += 1
                continue
            if not ann.get("segmentation"):
                rejection_counts["missing_segmentation"] += 1
                continue
            eligible.append(
                {
                    "annotation_id": ann["id"],
                    "category_id": ann["category_id"],
                    "label": clean_label(raw_label),
                    "raw_label": raw_label,
                    "bbox_xywh": ann["bbox"],
                    "bbox_area_fraction": bbox_fraction,
                    "mask_area_fraction": mask_fraction,
                }
            )

        image_pairs = []
        for object_a, object_b in itertools.combinations(eligible, 2):
            overlap = bbox_iou(object_a["bbox_xywh"], object_b["bbox_xywh"])
            if overlap > args.max_iou:
                rejection_counts["pair_iou_too_high"] += 1
                continue
            image_pairs.append(
                {
                    "schema_version": "0.1",
                    "source_dataset": "PACO-LVIS",
                    "source_split": args.annotation.stem.removeprefix("paco_lvis_v1_"),
                    "image_id": image_id,
                    "file_name": image["file_name"],
                    "coco_url": image.get("coco_url"),
                    "width": image["width"],
                    "height": image["height"],
                    "bbox_iou": overlap,
                    "object_a": object_a,
                    "object_b": object_b,
                    "selection_stage": "mask_geometry_before_pixel_color_screen",
                    "pixel_color_status": "pending",
                }
            )
        if image_pairs:
            image_pairs.sort(
                key=lambda row: (
                    -min(
                        row["object_a"]["mask_area_fraction"],
                        row["object_b"]["mask_area_fraction"],
                    ),
                    row["bbox_iou"],
                )
            )
            pairs.append(image_pairs[0])

    pairs.sort(
        key=lambda row: (
            -min(
                row["object_a"]["mask_area_fraction"],
                row["object_b"]["mask_area_fraction"],
            ),
            row["bbox_iou"],
            row["image_id"],
        )
    )
    total_pairs = len(pairs)
    pairs = pairs[: args.max_candidates]
    for index, row in enumerate(pairs, start=1):
        row["paco_mask_pair_id"] = f"paco_mask_pair_{index:05d}"

    args.output_dir.mkdir(parents=True, exist_ok=True)
    with (args.output_dir / "candidates.jsonl").open("w", encoding="utf-8") as handle:
        for row in pairs:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")

    summary = {
        "schema_version": "0.1",
        "annotation": str(args.annotation),
        "images": len(images),
        "object_annotations": sum(map(len, objects_by_image.values())),
        "candidate_images_before_cap": total_pairs,
        "candidates_written": len(pairs),
        "max_candidates": args.max_candidates,
        "filter_parameters": {
            "min_mask_area": args.min_mask_area,
            "min_bbox_area": args.min_bbox_area,
            "max_bbox_area": args.max_bbox_area,
            "max_iou": args.max_iou,
            "require_unique_object_label": True,
            "require_color_annotation": False,
        },
        "rejection_counts": dict(sorted(rejection_counts.items())),
        "model_outputs_used_for_decision": False,
    }
    (args.output_dir / "summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
