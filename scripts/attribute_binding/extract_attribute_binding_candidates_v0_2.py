#!/usr/bin/env python3
"""Extract edit-feasible color-object binding candidates from GQA scene graphs."""

from __future__ import annotations

import argparse
import json
from collections import Counter
from itertools import combinations
from pathlib import Path
from typing import Any


TARGET_COLORS = {"red", "blue", "green", "yellow"}
KNOWN_COLORS = TARGET_COLORS | {
    "black",
    "white",
    "gray",
    "grey",
    "brown",
    "orange",
    "purple",
    "pink",
    "beige",
    "tan",
    "cyan",
    "teal",
    "gold",
    "silver",
}

EXCLUDED_LABEL_GROUPS = {
    "generic": {
        "object",
        "thing",
        "item",
        "container",
        "stuff",
        "material",
        "surface",
        "area",
        "structure",
        "region",
    },
    "background_region": {
        "sky",
        "skies",
        "grass",
        "field",
        "fields",
        "ground",
        "floor",
        "floors",
        "ceiling",
        "road",
        "roads",
        "sidewalk",
        "pavement",
        "snow",
        "sand",
        "dirt",
        "soil",
        "landscape",
        "background",
        "cloud",
        "clouds",
        "air",
    },
    "vegetation_or_natural_region": {
        "tree",
        "trees",
        "pine tree",
        "palm tree",
        "leaves",
        "leaf",
        "foliage",
        "bush",
        "bushes",
        "plant",
        "plants",
        "weeds",
        "hay",
    },
    "large_structure": {
        "wall",
        "walls",
        "building",
        "buildings",
        "roof",
        "roofs",
        "fence",
        "fences",
        "window",
        "windows",
        "door",
        "doors",
        "pole",
        "poles",
        "barrier",
    },
    "liquid_or_transparent": {
        "water",
        "lake",
        "ocean",
        "sea",
        "river",
        "glass",
        "wine glass",
        "mirror",
        "windshield",
    },
    "text_or_screen": {
        "text",
        "letter",
        "letters",
        "sign",
        "signs",
        "traffic sign",
        "display",
        "screen",
        "television",
        "tv",
        "monitor",
        "laptop",
        "computer",
        "phone",
        "cellphone",
        "cell phone",
        "book",
        "newspaper",
        "poster",
        "drawing",
    },
    "body_part": {
        "hair",
        "skin",
        "face",
        "head",
        "neck",
        "arm",
        "arms",
        "leg",
        "legs",
        "hand",
        "hands",
        "tail",
    },
    "animate_entity": {
        "person",
        "people",
        "man",
        "men",
        "woman",
        "women",
        "boy",
        "boys",
        "girl",
        "girls",
        "child",
        "children",
        "baby",
        "animal",
        "dog",
        "cat",
        "horse",
        "cow",
        "sheep",
        "bear",
        "pig",
        "bird",
        "elephant",
        "giraffe",
        "zebra",
    },
    "substance_or_ambiguous": {
        "paint",
        "plain",
    },
}

PREFERRED_EDITABLE_LABELS = {
    "shirt",
    "t-shirt",
    "blouse",
    "sweater",
    "jacket",
    "coat",
    "jersey",
    "dress",
    "skirt",
    "pants",
    "jeans",
    "shorts",
    "cap",
    "hat",
    "shoe",
    "shoes",
    "bag",
    "backpack",
    "suitcase",
    "chair",
    "bench",
    "couch",
    "sofa",
    "table",
    "cabinet",
    "dresser",
    "drawer",
    "bed",
    "pillow",
    "blanket",
    "rug",
    "curtain",
    "drapes",
    "towel",
    "tent",
    "box",
    "bowl",
    "cup",
    "plate",
    "vase",
    "lamp",
    "umbrella",
    "car",
    "truck",
    "bus",
    "train",
    "boat",
    "bike",
    "bicycle",
    "motorcycle",
    "racket",
    "tennis racket",
    "frisbee",
    "surfboard",
    "skis",
    "snowboard",
    "helmet",
    "fire hydrant",
    "traffic cone",
    "clock",
}


def parse_args() -> argparse.Namespace:
    project_root = Path(__file__).resolve().parents[2]
    parser = argparse.ArgumentParser(
        description="Extract GQA color-binding feasibility candidates v0.2."
    )
    parser.add_argument(
        "--input",
        type=Path,
        default=project_root / "raw/GQA-Scene-Graph/samples.json",
    )
    parser.add_argument(
        "--image-root",
        type=Path,
        default=project_root / "raw/GQA-Scene-Graph",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=project_root
        / "processed/attribute_binding/candidates/gqa/candidates_color_pairs_v0.2.jsonl",
    )
    parser.add_argument(
        "--summary-output",
        type=Path,
        default=project_root
        / "processed/attribute_binding/candidates/gqa/candidate_summary_v0.2.json",
    )
    parser.add_argument("--min-bbox-area", type=float, default=0.03)
    parser.add_argument("--max-bbox-area", type=float, default=0.40)
    parser.add_argument("--max-iou", type=float, default=0.10)
    parser.add_argument("--border-margin", type=float, default=0.005)
    return parser.parse_args()


def bbox_area(bbox: list[float]) -> float:
    return bbox[2] * bbox[3]


def valid_normalized_bbox(bbox: Any) -> bool:
    if not isinstance(bbox, list) or len(bbox) != 4:
        return False
    if not all(isinstance(value, (int, float)) for value in bbox):
        return False
    x, y, width, height = bbox
    return (
        width > 0
        and height > 0
        and x >= 0
        and y >= 0
        and x + width <= 1.01
        and y + height <= 1.01
    )


def bbox_touches_frame(bbox: list[float], margin: float) -> bool:
    x, y, width, height = bbox
    return (
        x <= margin
        or y <= margin
        or x + width >= 1 - margin
        or y + height >= 1 - margin
    )


def bbox_iou(first: list[float], second: list[float]) -> float:
    ax1, ay1, aw, ah = first
    bx1, by1, bw, bh = second
    ax2, ay2 = ax1 + aw, ay1 + ah
    bx2, by2 = bx1 + bw, by1 + bh
    overlap_width = max(0.0, min(ax2, bx2) - max(ax1, bx1))
    overlap_height = max(0.0, min(ay2, by2) - max(ay1, by1))
    intersection = overlap_width * overlap_height
    union = bbox_area(first) + bbox_area(second) - intersection
    return intersection / union if union else 0.0


def normalized_attributes(detection: dict[str, Any]) -> set[str]:
    values = detection.get("object_attributes", [])
    return {str(value).strip().lower() for value in values if str(value).strip()}


def sole_target_color(detection: dict[str, Any]) -> str | None:
    observed_colors = normalized_attributes(detection) & KNOWN_COLORS
    if len(observed_colors) != 1:
        return None
    color = next(iter(observed_colors))
    return color if color in TARGET_COLORS else None


def excluded_label_group(label: str) -> str | None:
    for group, labels in EXCLUDED_LABEL_GROUPS.items():
        if label in labels:
            return group
    return None


def semantically_overlapping_labels(first: str, second: str) -> bool:
    first_words = set(first.split())
    second_words = set(second.split())
    return first in second or second in first or bool(first_words & second_words)


def main() -> None:
    args = parse_args()
    with args.input.open("r", encoding="utf-8") as input_file:
        samples = json.load(input_file)["samples"]

    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.summary_output.parent.mkdir(parents=True, exist_ok=True)
    counts: Counter[str] = Counter()
    color_pairs: Counter[str] = Counter()
    priority_tiers: Counter[str] = Counter()
    accepted_labels: Counter[str] = Counter()

    with args.output.open("w", encoding="utf-8") as output_file:
        for sample_index, sample in enumerate(samples):
            detections = sample.get("detections", {}).get("detections", [])
            labels = [str(item.get("label", "")).strip().lower() for item in detections]
            label_counts = Counter(label for label in labels if label)
            eligible: list[dict[str, Any]] = []

            for detection_index, detection in enumerate(detections):
                counts["detections_seen"] += 1
                label = labels[detection_index]
                bbox = detection.get("bounding_box")
                if not label:
                    counts["rejected_empty_label"] += 1
                    continue
                excluded_group = excluded_label_group(label)
                if excluded_group:
                    counts[f"rejected_label_{excluded_group}"] += 1
                    continue
                if label_counts[label] != 1:
                    counts["rejected_nonunique_label"] += 1
                    continue
                color = sole_target_color(detection)
                if color is None:
                    counts["rejected_color_attribute"] += 1
                    continue
                if not valid_normalized_bbox(bbox):
                    counts["rejected_invalid_bbox"] += 1
                    continue
                area = bbox_area(bbox)
                if area < args.min_bbox_area:
                    counts["rejected_small_bbox"] += 1
                    continue
                if area > args.max_bbox_area:
                    counts["rejected_large_bbox"] += 1
                    continue
                if bbox_touches_frame(bbox, args.border_margin):
                    counts["rejected_frame_touching_bbox"] += 1
                    continue
                eligible.append(
                    {
                        "detection_index": detection_index,
                        "label": label,
                        "color": color,
                        "bbox_xywh_norm": bbox,
                        "bbox_area": area,
                        "attributes": sorted(normalized_attributes(detection)),
                        "preferred_editable_label": label in PREFERRED_EDITABLE_LABELS,
                    }
                )
                counts["eligible_detections"] += 1

            if len(eligible) < 2:
                continue
            image_path = args.image_root / str(sample["filepath"])
            if not image_path.is_file():
                counts["rejected_samples_missing_image"] += 1
                continue

            for first, second in combinations(eligible, 2):
                counts["eligible_pairs_before_pair_filters"] += 1
                if first["color"] == second["color"]:
                    counts["rejected_same_color"] += 1
                    continue
                if semantically_overlapping_labels(first["label"], second["label"]):
                    counts["rejected_semantic_label_overlap"] += 1
                    continue
                overlap = bbox_iou(first["bbox_xywh_norm"], second["bbox_xywh_norm"])
                if overlap > args.max_iou:
                    counts["rejected_high_iou"] += 1
                    continue

                preferred_count = sum(
                    object_row["preferred_editable_label"]
                    for object_row in (first, second)
                )
                priority_tier = {
                    2: "both_preferred",
                    1: "one_preferred",
                    0: "manual_review",
                }[preferred_count]
                color_pair = "_".join(sorted((first["color"], second["color"])))
                record = {
                    "schema_version": "0.2",
                    "source_dataset": "Voxel51/GQA-Scene-Graph",
                    "sample_index": sample_index,
                    "image_id": Path(sample["filepath"]).stem,
                    "image_path": sample["filepath"],
                    "object_a": first,
                    "object_b": second,
                    "color_pair": color_pair,
                    "bbox_iou": overlap,
                    "editability_priority": priority_tier,
                    "candidate_stage": "pre_segmentation_feasibility_v0.2",
                }
                output_file.write(json.dumps(record, ensure_ascii=False) + "\n")
                counts["accepted_pairs"] += 1
                color_pairs[color_pair] += 1
                priority_tiers[priority_tier] += 1
                accepted_labels[first["label"]] += 1
                accepted_labels[second["label"]] += 1

    summary = {
        "schema_version": "0.2",
        "target_colors": sorted(TARGET_COLORS),
        "thresholds": {
            "unique_labels": True,
            "single_recognized_color_attribute": True,
            "min_bbox_area": args.min_bbox_area,
            "max_bbox_area": args.max_bbox_area,
            "max_iou": args.max_iou,
            "border_margin": args.border_margin,
            "semantic_substring_or_token_overlap_excluded": True,
            "hard_excluded_label_groups": {
                group: sorted(labels)
                for group, labels in EXCLUDED_LABEL_GROUPS.items()
            },
        },
        "counts": dict(sorted(counts.items())),
        "accepted_by_color_pair": dict(sorted(color_pairs.items())),
        "accepted_by_editability_priority": dict(sorted(priority_tiers.items())),
        "top_accepted_labels": accepted_labels.most_common(50),
        "warning": (
            "Candidates remain pre-segmentation. Human review must establish dominant "
            "color, opacity, color-swap plausibility, mask feasibility, edit naturalness, "
            "and identity preservation before VLM inference."
        ),
    }
    args.summary_output.write_text(
        json.dumps(summary, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
