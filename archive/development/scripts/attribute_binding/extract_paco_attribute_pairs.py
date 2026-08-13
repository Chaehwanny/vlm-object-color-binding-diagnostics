#!/usr/bin/env python3
"""Extract model-blind color-binding pairs from PACO-LVIS annotations."""

from __future__ import annotations

import argparse
import itertools
import json
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any


CORE_COLORS = {"blue", "green", "red", "yellow"}

# Categories whose colors can ordinarily vary without changing object identity.
# This is intentionally conservative; visual review remains mandatory.
EDITABLE_CATEGORIES = {
    "ball",
    "basket",
    "bicycle",
    "bottle",
    "bowl",
    "bucket",
    "car_(automobile)",
    "chair",
    "crate",
    "cup",
    "drum_(musical_instrument)",
    "earphone",
    "fan",
    "glass_(drink_container)",
    "guitar",
    "handbag",
    "hat",
    "helmet",
    "jar",
    "kettle",
    "ladder",
    "lamp",
    "mug",
    "napkin",
    "pan_(for_cooking)",
    "pen",
    "pencil",
    "pillow",
    "plate",
    "plastic_bag",
    "scarf",
    "shoe",
    "slipper_(footwear)",
    "soap",
    "sponge",
    "spoon",
    "stool",
    "sweater",
    "table",
    "towel",
    "trash_can",
    "tray",
    "vase",
}


def parse_args() -> argparse.Namespace:
    root = Path(__file__).resolve().parents[2]
    parser = argparse.ArgumentParser(description="Extract PACO color-object pairs.")
    parser.add_argument(
        "--annotation",
        type=Path,
        default=root / "raw/PACO-LVIS/annotations/paco_lvis_v1_val.json",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=root / "processed/attribute_binding/candidates/paco/paco_pairs_val_v0.1",
    )
    parser.add_argument("--min-mask-area", type=float, default=0.015)
    parser.add_argument("--min-bbox-area", type=float, default=0.02)
    parser.add_argument("--max-bbox-area", type=float, default=0.50)
    parser.add_argument("--max-iou", type=float, default=0.05)
    parser.add_argument("--allow-nonplain", action="store_true")
    parser.add_argument("--allow-repeated-labels", action="store_true")
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def bbox_area_fraction(ann: dict[str, Any], image: dict[str, Any]) -> float:
    _, _, width, height = ann["bbox"]
    return float(width * height / (image["width"] * image["height"]))


def mask_area_fraction(ann: dict[str, Any], image: dict[str, Any]) -> float:
    return float(ann["area"] / (image["width"] * image["height"]))


def bbox_iou(a: list[float], b: list[float]) -> float:
    ax, ay, aw, ah = a
    bx, by, bw, bh = b
    ix1, iy1 = max(ax, bx), max(ay, by)
    ix2, iy2 = min(ax + aw, bx + bw), min(ay + ah, by + bh)
    intersection = max(0.0, ix2 - ix1) * max(0.0, iy2 - iy1)
    union = aw * ah + bw * bh - intersection
    return float(intersection / union) if union else 0.0


def clean_label(label: str) -> str:
    return label.replace("_(automobile)", "").replace(
        "_(drink_container)", ""
    ).replace("_(musical_instrument)", "").replace("_(footwear)", "")


def main() -> None:
    args = parse_args()
    if args.output_dir.exists() and any(args.output_dir.iterdir()) and not args.overwrite:
        raise FileExistsError(f"Output directory is not empty: {args.output_dir}")

    with args.annotation.open("r", encoding="utf-8") as handle:
        data = json.load(handle)

    images = {image["id"]: image for image in data["images"]}
    categories = {category["id"]: category for category in data["categories"]}
    attributes = {attribute["id"]: attribute["name"] for attribute in data["attributes"]}
    color_id_to_name = {
        attr_id: name for attr_id, name in attributes.items() if name in CORE_COLORS
    }
    plain_id = next(attr_id for attr_id, name in attributes.items() if name == "plain")

    object_annotations: dict[int, list[dict[str, Any]]] = defaultdict(list)
    for ann in data["annotations"]:
        category = categories[ann["category_id"]]
        if category["supercategory"] != "OBJECT" or ann["id"] != ann["obj_ann_id"]:
            continue
        object_annotations[ann["image_id"]].append(ann)

    eligible_by_image: dict[int, list[dict[str, Any]]] = defaultdict(list)
    rejection_counts: Counter[str] = Counter()
    for image_id, anns in object_annotations.items():
        image = images[image_id]
        label_counts = Counter(ann["category_id"] for ann in anns)
        for ann in anns:
            category = categories[ann["category_id"]]
            label = category["name"]
            bbox_fraction = bbox_area_fraction(ann, image)
            mask_fraction = mask_area_fraction(ann, image)
            reasons = []
            if label not in EDITABLE_CATEGORIES:
                reasons.append("noneditable_category")
            if not args.allow_repeated_labels and label_counts[ann["category_id"]] != 1:
                reasons.append("competing_same_label")
            if ann.get("unknown_color", 0):
                reasons.append("unknown_color")
            if len(ann.get("dom_color_ids", [])) != 1:
                reasons.append("not_single_dominant_color")
            elif ann["dom_color_ids"][0] not in color_id_to_name:
                reasons.append("noncore_color")
            if not args.allow_nonplain and plain_id not in ann.get("attribute_ids", []):
                reasons.append("not_plain")
            if bbox_fraction < args.min_bbox_area:
                reasons.append("bbox_too_small")
            if bbox_fraction > args.max_bbox_area:
                reasons.append("bbox_too_large")
            if mask_fraction < args.min_mask_area:
                reasons.append("mask_too_small")
            if not ann.get("segmentation"):
                reasons.append("missing_segmentation")
            if reasons:
                rejection_counts.update(set(reasons))
                continue
            eligible_by_image[image_id].append(
                {
                    "annotation_id": ann["id"],
                    "category_id": ann["category_id"],
                    "label": clean_label(label),
                    "raw_label": label,
                    "color": color_id_to_name[ann["dom_color_ids"][0]],
                    "bbox_xywh": ann["bbox"],
                    "bbox_area_fraction": bbox_fraction,
                    "mask_area_fraction": mask_fraction,
                }
            )

    pairs = []
    pair_id = 0
    for image_id, objects in eligible_by_image.items():
        image = images[image_id]
        for object_a, object_b in itertools.combinations(objects, 2):
            if object_a["color"] == object_b["color"]:
                rejection_counts["same_color_pair"] += 1
                continue
            overlap = bbox_iou(object_a["bbox_xywh"], object_b["bbox_xywh"])
            if overlap > args.max_iou:
                rejection_counts["pair_iou_too_high"] += 1
                continue
            pair_id += 1
            colors = sorted((object_a["color"], object_b["color"]))
            pairs.append(
                {
                    "schema_version": "0.1",
                    "paco_pair_id": f"paco_pair_{pair_id:05d}",
                    "source_dataset": "PACO-LVIS",
                    "source_split": args.annotation.stem.removeprefix("paco_lvis_v1_"),
                    "image_id": image_id,
                    "file_name": image["file_name"],
                    "coco_url": image.get("coco_url"),
                    "width": image["width"],
                    "height": image["height"],
                    "color_pair": "_".join(colors),
                    "bbox_iou": overlap,
                    "object_a": object_a,
                    "object_b": object_b,
                    "selection_stage": "annotation_only_model_blind_filter",
                    "human_review_status": "pending",
                }
            )

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
    for index, row in enumerate(pairs, start=1):
        row["paco_pair_id"] = f"paco_pair_{index:05d}"

    args.output_dir.mkdir(parents=True, exist_ok=True)
    with (args.output_dir / "candidates.jsonl").open("w", encoding="utf-8") as handle:
        for row in pairs:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")

    summary = {
        "schema_version": "0.1",
        "annotation": str(args.annotation),
        "images": len(images),
        "object_annotations": sum(map(len, object_annotations.values())),
        "images_with_two_or_more_eligible_objects": sum(
            len(objects) >= 2 for objects in eligible_by_image.values()
        ),
        "candidate_pairs": len(pairs),
        "candidate_images": len({row["image_id"] for row in pairs}),
        "color_pair_counts": dict(sorted(Counter(row["color_pair"] for row in pairs).items())),
        "filter_parameters": {
            "min_mask_area": args.min_mask_area,
            "min_bbox_area": args.min_bbox_area,
            "max_bbox_area": args.max_bbox_area,
            "max_iou": args.max_iou,
            "require_plain": not args.allow_nonplain,
            "require_unique_object_label": not args.allow_repeated_labels,
            "colors": sorted(CORE_COLORS),
        },
        "object_rejection_counts_nonexclusive": dict(sorted(rejection_counts.items())),
        "model_outputs_used_for_decision": False,
    }
    (args.output_dir / "summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
