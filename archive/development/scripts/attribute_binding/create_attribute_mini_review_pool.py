#!/usr/bin/env python3
"""Build a model-blind, pixel-screened review pool for the binding mini set."""

from __future__ import annotations

import argparse
import csv
import json
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

import numpy as np
from PIL import Image

from create_attribute_review_pool_v0_2 import TIER_ORDER, canonical_label_pair, create_preview


COLOR_CENTERS = {"red": 0.0, "yellow": 55.0, "green": 120.0, "blue": 220.0}
SEMANTIC_OVERLAP_GROUPS = (
    {"pants", "jeans", "shorts", "trousers"},
    {"bike", "bicycle"},
    {"sofa", "couch"},
    {"shirt", "t-shirt", "tee"},
)


def parse_args() -> argparse.Namespace:
    project_root = Path(__file__).resolve().parents[2]
    parser = argparse.ArgumentParser(description="Build mini-experiment review pool.")
    parser.add_argument(
        "--input",
        type=Path,
        default=project_root
        / "processed/attribute_binding/candidates/gqa/candidates_color_pairs_v0.2.jsonl",
    )
    parser.add_argument(
        "--image-root",
        type=Path,
        default=project_root / "raw/GQA-Scene-Graph",
    )
    parser.add_argument(
        "--exclude-manifest",
        type=Path,
        action="append",
        default=[
            project_root / "processed/attribute_binding/reviews/review_pool_v0.2/review_pool.jsonl",
            project_root
            / "processed/attribute_binding/reviews/replacement_review_pool_v0.2/replacement_pool.jsonl",
        ],
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=project_root / "processed/attribute_binding/reviews/mini_review_pool_v0.1",
    )
    parser.add_argument("--target-size", type=int, default=40)
    parser.add_argument("--base-per-pair", type=int, default=6)
    parser.add_argument("--min-color-fraction", type=float, default=0.30)
    parser.add_argument("--min-median-saturation", type=float, default=60.0)
    parser.add_argument("--hue-tolerance", type=float, default=50.0)
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    with path.open("r", encoding="utf-8") as input_file:
        return [json.loads(line) for line in input_file]


def angular_distance(values: np.ndarray, center: float) -> np.ndarray:
    return np.abs((values - center + 180.0) % 360.0 - 180.0)


def object_color_metrics(
    hsv: np.ndarray,
    image_size: tuple[int, int],
    object_row: dict[str, Any],
    hue_tolerance: float,
) -> dict[str, float]:
    width, height = image_size
    x, y, box_width, box_height = object_row["bbox_xywh_norm"]
    x0 = max(0, round(x * width))
    y0 = max(0, round(y * height))
    x1 = min(width, round((x + box_width) * width))
    y1 = min(height, round((y + box_height) * height))
    crop = hsv[y0:y1, x0:x1]
    if crop.size == 0:
        return {"target_color_fraction": 0.0, "median_saturation": 0.0, "mean_hue_distance": 180.0}

    hue = crop[..., 0].astype(np.float32) * (360.0 / 255.0)
    saturation = crop[..., 1]
    value = crop[..., 2]
    distance = angular_distance(hue, COLOR_CENTERS[object_row["color"]])
    selected = (distance <= hue_tolerance) & (saturation >= 45) & (value >= 30)
    if not selected.any():
        return {"target_color_fraction": 0.0, "median_saturation": 0.0, "mean_hue_distance": 180.0}
    weights = saturation[selected].astype(np.float32)
    return {
        "target_color_fraction": float(selected.mean()),
        "median_saturation": float(np.median(saturation[selected])),
        "mean_hue_distance": float(np.average(distance[selected], weights=weights)),
    }


def semantically_overlapping(label_a: str, label_b: str) -> bool:
    labels = {label_a.lower(), label_b.lower()}
    return any(labels <= group for group in SEMANTIC_OVERLAP_GROUPS)


def enrich_candidates(args: argparse.Namespace) -> tuple[list[dict[str, Any]], Counter[str]]:
    excluded_images = set()
    for path in args.exclude_manifest:
        excluded_images.update(row["image_path"] for row in read_jsonl(path))

    rejection_counts: Counter[str] = Counter()
    accepted = []
    for row in read_jsonl(args.input):
        if row["image_path"] in excluded_images:
            rejection_counts["previously_reviewed_image"] += 1
            continue
        if semantically_overlapping(row["object_a"]["label"], row["object_b"]["label"]):
            rejection_counts["semantic_overlap_group"] += 1
            continue
        with Image.open(args.image_root / row["image_path"]) as opened_image:
            hsv_image = opened_image.convert("HSV")
        hsv = np.asarray(hsv_image)
        metrics_a = object_color_metrics(hsv, hsv_image.size, row["object_a"], args.hue_tolerance)
        metrics_b = object_color_metrics(hsv, hsv_image.size, row["object_b"], args.hue_tolerance)
        if min(metrics_a["target_color_fraction"], metrics_b["target_color_fraction"]) < args.min_color_fraction:
            rejection_counts["low_bbox_color_fraction"] += 1
            continue
        if min(metrics_a["median_saturation"], metrics_b["median_saturation"]) < args.min_median_saturation:
            rejection_counts["low_bbox_color_saturation"] += 1
            continue
        accepted.append({**row, "bbox_color_metrics_a": metrics_a, "bbox_color_metrics_b": metrics_b})
    return accepted, rejection_counts


def rank_key(row: dict[str, Any]) -> tuple[Any, ...]:
    minimum_fraction = min(
        row["bbox_color_metrics_a"]["target_color_fraction"],
        row["bbox_color_metrics_b"]["target_color_fraction"],
    )
    minimum_saturation = min(
        row["bbox_color_metrics_a"]["median_saturation"],
        row["bbox_color_metrics_b"]["median_saturation"],
    )
    maximum_hue_distance = max(
        row["bbox_color_metrics_a"]["mean_hue_distance"],
        row["bbox_color_metrics_b"]["mean_hue_distance"],
    )
    minimum_area = min(row["object_a"]["bbox_area"], row["object_b"]["bbox_area"])
    return (
        TIER_ORDER[row["editability_priority"]],
        -minimum_fraction,
        -minimum_saturation,
        maximum_hue_distance,
        -minimum_area,
        row["image_id"],
    )


def select_review_pool(rows: list[dict[str, Any]], target_size: int, base_per_pair: int) -> list[dict[str, Any]]:
    grouped: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        grouped[row["color_pair"]].append(row)
    for pool in grouped.values():
        pool.sort(key=rank_key)

    selected = []
    used_images: set[str] = set()
    used_label_pairs: set[tuple[str, str]] = set()
    for color_pair in sorted(grouped):
        count = 0
        for row in grouped[color_pair]:
            if count == base_per_pair:
                break
            label_pair = canonical_label_pair(row)
            if row["image_path"] in used_images or label_pair in used_label_pairs:
                continue
            selected.append(row)
            used_images.add(row["image_path"])
            used_label_pairs.add(label_pair)
            count += 1

    maximum_pair_count = max(1, int(target_size * 0.40))
    pair_counts = Counter(row["color_pair"] for row in selected)
    for require_new_label_pair in (True, False):
        for row in sorted(rows, key=rank_key):
            if len(selected) == target_size:
                break
            label_pair = canonical_label_pair(row)
            if row["image_path"] in used_images:
                continue
            if pair_counts[row["color_pair"]] >= maximum_pair_count:
                continue
            if require_new_label_pair and label_pair in used_label_pairs:
                continue
            selected.append(row)
            used_images.add(row["image_path"])
            used_label_pairs.add(label_pair)
            pair_counts[row["color_pair"]] += 1
        if len(selected) == target_size:
            break
    if len(selected) != target_size:
        raise RuntimeError(f"Selected {len(selected)}/{target_size} mini review candidates")
    return selected


def main() -> None:
    args = parse_args()
    if args.output_dir.exists() and any(args.output_dir.iterdir()) and not args.overwrite:
        raise FileExistsError(f"Output directory is not empty: {args.output_dir}. Use --overwrite.")

    eligible, rejection_counts = enrich_candidates(args)
    selected = select_review_pool(eligible, args.target_size, args.base_per_pair)
    previews_dir = args.output_dir / "previews"
    args.output_dir.mkdir(parents=True, exist_ok=True)

    fields = [
        "mini_review_id", "color_pair", "editability_priority", "image_path", "preview_path",
        "object_a_label", "object_a_color", "object_a_bbox_color_fraction", "object_a_median_saturation",
        "object_b_label", "object_b_color", "object_b_bbox_color_fraction", "object_b_median_saturation",
        "both_objects_identifiable_yes_no_uncertain", "labels_visually_unique_yes_no_uncertain",
        "colors_canonical_and_clear_yes_no_uncertain", "objects_semantically_distinct_yes_no_uncertain",
        "objects_independently_segmentable_yes_no_uncertain", "no_strong_color_distractor_yes_no_uncertain",
        "bidirectional_swap_plausible_yes_no_uncertain", "post_edit_binding_likely_clear_yes_no_uncertain",
        "keep_for_mask_yes_no", "exclude_reason", "notes",
    ]
    with (args.output_dir / "review_pool.jsonl").open("w", encoding="utf-8") as manifest_file, (
        args.output_dir / "human_review.csv"
    ).open("w", encoding="utf-8", newline="") as review_file:
        writer = csv.DictWriter(review_file, fieldnames=fields)
        writer.writeheader()
        for index, row in enumerate(selected, start=1):
            review_id = f"mini_review_{index:03d}"
            preview_path = previews_dir / f"{review_id}.jpg"
            create_preview(row, args.image_root / row["image_path"], preview_path)
            enriched = {
                **row,
                "mini_review_id": review_id,
                "preview_path": str(preview_path.relative_to(args.output_dir)),
                "selection_stage": "model_blind_mini_human_review",
            }
            manifest_file.write(json.dumps(enriched, ensure_ascii=False) + "\n")
            writer.writerow({
                "mini_review_id": review_id,
                "color_pair": row["color_pair"],
                "editability_priority": row["editability_priority"],
                "image_path": row["image_path"],
                "preview_path": enriched["preview_path"],
                "object_a_label": row["object_a"]["label"],
                "object_a_color": row["object_a"]["color"],
                "object_a_bbox_color_fraction": row["bbox_color_metrics_a"]["target_color_fraction"],
                "object_a_median_saturation": row["bbox_color_metrics_a"]["median_saturation"],
                "object_b_label": row["object_b"]["label"],
                "object_b_color": row["object_b"]["color"],
                "object_b_bbox_color_fraction": row["bbox_color_metrics_b"]["target_color_fraction"],
                "object_b_median_saturation": row["bbox_color_metrics_b"]["median_saturation"],
            })

    summary = {
        "schema_version": "0.1",
        "model_outputs_used_for_selection": False,
        "previous_review_images_excluded": True,
        "eligible_after_pixel_screen": len(eligible),
        "selected_for_human_review": len(selected),
        "selected_by_color_pair": dict(sorted(Counter(row["color_pair"] for row in selected).items())),
        "selected_by_priority": dict(sorted(Counter(row["editability_priority"] for row in selected).items())),
        "rejection_counts": dict(sorted(rejection_counts.items())),
        "thresholds": {
            "minimum_bbox_target_color_fraction": args.min_color_fraction,
            "minimum_selected_color_median_saturation_0_255": args.min_median_saturation,
            "hue_tolerance_degrees": args.hue_tolerance,
        },
    }
    (args.output_dir / "summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    (args.output_dir / "README.md").write_text(
        """# Attribute Binding Mini Review Pool v0.1

Forty new, model-blind candidates for human review. All images previously used
in feasibility or replacement review were excluded. Automated bbox color
metrics are ranking aids only; human review determines advancement to SAM masks.

Do not run any VLM on this pool before the accepted mini set is frozen.
""",
        encoding="utf-8",
    )
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
