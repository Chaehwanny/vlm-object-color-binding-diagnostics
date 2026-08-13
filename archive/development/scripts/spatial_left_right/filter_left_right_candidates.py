#!/usr/bin/env python3
"""Apply geometric and ambiguity filters to extracted GQA left/right candidates."""

from __future__ import annotations

import argparse
import json
import sys
from collections import Counter
from pathlib import Path
from typing import Any


def parse_args() -> argparse.Namespace:
    project_root = Path(__file__).resolve().parents[2]
    parser = argparse.ArgumentParser(
        description="Filter GQA left/right relation candidates for the pilot study."
    )
    parser.add_argument(
        "--input",
        type=Path,
        default=project_root
        / "processed/spatial_left_right/candidates/candidates_left_right.jsonl",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=project_root
        / "processed/spatial_left_right/candidates/filtered_left_right_candidates.jsonl",
    )
    parser.add_argument(
        "--summary-output",
        type=Path,
        default=project_root
        / "processed/spatial_left_right/candidates/filter_summary.json",
    )
    parser.add_argument(
        "--image-root",
        type=Path,
        default=project_root / "raw/GQA-Scene-Graph",
    )
    parser.add_argument("--min-bbox-area", type=float, default=0.015)
    parser.add_argument("--min-center-gap", type=float, default=0.15)
    parser.add_argument("--max-iou", type=float, default=0.10)
    return parser.parse_args()


def valid_bbox(value: Any) -> bool:
    if not isinstance(value, list) or len(value) != 4:
        return False
    if not all(isinstance(number, (int, float)) for number in value):
        return False
    x, y, width, height = value
    return (
        0 <= x <= 1
        and 0 <= y <= 1
        and width > 0
        and height > 0
        and x + width <= 1.000001
        and y + height <= 1.000001
    )


def area(bbox: list[float]) -> float:
    return bbox[2] * bbox[3]


def center_x(bbox: list[float]) -> float:
    return bbox[0] + bbox[2] / 2


def iou(first: list[float], second: list[float]) -> float:
    first_x1, first_y1, first_width, first_height = first
    second_x1, second_y1, second_width, second_height = second
    first_x2, first_y2 = first_x1 + first_width, first_y1 + first_height
    second_x2, second_y2 = second_x1 + second_width, second_y1 + second_height

    intersection_width = max(0.0, min(first_x2, second_x2) - max(first_x1, second_x1))
    intersection_height = max(0.0, min(first_y2, second_y2) - max(first_y1, second_y1))
    intersection = intersection_width * intersection_height
    union = area(first) + area(second) - intersection
    return intersection / union if union else 0.0


def rejection_reason(record: dict[str, Any], args: argparse.Namespace) -> str | None:
    if record["source_label"] == record["target_label"]:
        return "same_source_target_label"
    if record["source_label_count"] != 1 or record["target_label_count"] != 1:
        return "non_unique_object_label"
    if len(record["target_detection_indices"]) != 1:
        return "unresolved_target_instance"

    source_bbox = record["source_bbox_xywh_norm"]
    target_bbox = record["target_bboxes_xywh_norm"][0]
    if not valid_bbox(source_bbox) or not valid_bbox(target_bbox):
        return "invalid_bbox"
    if area(source_bbox) < args.min_bbox_area or area(target_bbox) < args.min_bbox_area:
        return "small_bbox"

    horizontal_gap = abs(center_x(source_bbox) - center_x(target_bbox))
    if horizontal_gap < args.min_center_gap:
        return "small_horizontal_gap"
    if iou(source_bbox, target_bbox) > args.max_iou:
        return "high_iou"

    relation = record["relation"]
    if relation == "left_of" and center_x(source_bbox) >= center_x(target_bbox):
        return "relation_bbox_direction_mismatch"
    if relation == "right_of" and center_x(source_bbox) <= center_x(target_bbox):
        return "relation_bbox_direction_mismatch"

    image_path = args.image_root / str(record["image_path"])
    if not image_path.is_file():
        return "missing_image"
    return None


def main() -> None:
    args = parse_args()
    if not args.input.is_file():
        raise FileNotFoundError(f"Input file not found: {args.input}")

    thresholds = {
        "unique_source_and_target_labels": True,
        "min_bbox_area": args.min_bbox_area,
        "min_center_gap": args.min_center_gap,
        "max_iou": args.max_iou,
        "bbox_direction_must_match_scene_graph": True,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.summary_output.parent.mkdir(parents=True, exist_ok=True)

    counts: Counter[str] = Counter()
    accepted_relations: Counter[str] = Counter()
    seen: set[tuple[str, int, int, str]] = set()

    with args.input.open("r", encoding="utf-8") as input_file, args.output.open(
        "w", encoding="utf-8"
    ) as output_file:
        for line_number, line in enumerate(input_file, start=1):
            record = json.loads(line)
            counts["input"] += 1
            reason = rejection_reason(record, args)
            if reason:
                counts[f"rejected_{reason}"] += 1
                continue

            target_index = record["target_detection_indices"][0]
            dedupe_key = (
                str(record["image_path"]),
                int(record["source_detection_index"]),
                int(target_index),
                str(record["relation"]),
            )
            if dedupe_key in seen:
                counts["rejected_duplicate_relation"] += 1
                continue
            seen.add(dedupe_key)

            source_bbox = record["source_bbox_xywh_norm"]
            target_bbox = record["target_bboxes_xywh_norm"][0]
            record["target_detection_index"] = target_index
            record["target_bbox_xywh_norm"] = target_bbox
            record["geometry"] = {
                "source_center_x": center_x(source_bbox),
                "target_center_x": center_x(target_bbox),
                "horizontal_center_gap": abs(center_x(source_bbox) - center_x(target_bbox)),
                "iou": iou(source_bbox, target_bbox),
                "source_bbox_area": area(source_bbox),
                "target_bbox_area": area(target_bbox),
            }
            record["filter_version"] = "pilot_left_right_v1"
            record["filter_thresholds"] = thresholds
            output_file.write(json.dumps(record, ensure_ascii=False) + "\n")
            counts["accepted"] += 1
            accepted_relations[record["relation"]] += 1

            if line_number % 200000 == 0:
                print(
                    f"Processed {line_number:,} candidates; accepted {counts['accepted']:,}.",
                    file=sys.stderr,
                )

    summary = {
        "filter_version": "pilot_left_right_v1",
        "input_path": str(args.input),
        "output_path": str(args.output),
        "image_root": str(args.image_root),
        "thresholds": thresholds,
        "counts": dict(counts),
        "accepted_by_relation": dict(accepted_relations),
    }
    with args.summary_output.open("w", encoding="utf-8") as summary_file:
        json.dump(summary, summary_file, ensure_ascii=False, indent=2)
        summary_file.write("\n")

    print(
        f"Done. Accepted {counts['accepted']:,}/{counts['input']:,} candidates. "
        f"Summary: {args.summary_output}",
        file=sys.stderr,
    )


if __name__ == "__main__":
    main()
