#!/usr/bin/env python3
"""Screen PACO object-mask pairs using canonical colors in mask pixels."""

from __future__ import annotations

import argparse
import json
import shutil
from pathlib import Path
from typing import Any

import numpy as np
from PIL import Image, ImageDraw


COLOR_CENTERS = {"red": 0.0, "yellow": 55.0, "green": 120.0, "blue": 220.0}
BOUNDARY_COLORS = {"a": (255, 0, 255), "b": (0, 255, 255)}


def parse_args() -> argparse.Namespace:
    root = Path(__file__).resolve().parents[2]
    parser = argparse.ArgumentParser(description="Screen PACO pair colors.")
    parser.add_argument(
        "--pair-manifest",
        type=Path,
        default=root
        / "processed/attribute_binding/candidates/paco/paco_mask_pairs_train_v0.1/candidates.jsonl",
    )
    parser.add_argument(
        "--annotation",
        type=Path,
        default=root / "raw/PACO-LVIS/annotations/paco_lvis_v1_train.json",
    )
    parser.add_argument(
        "--image-dir",
        type=Path,
        default=root / "raw/PACO-LVIS/candidate_images_v0.1",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=root / "processed/attribute_binding/reviews/paco_color_screen_v0.1",
    )
    parser.add_argument("--hue-tolerance", type=float, default=50.0)
    parser.add_argument("--min-saturation", type=int, default=60)
    parser.add_argument("--min-value", type=int, default=30)
    parser.add_argument("--min-color-coverage", type=float, default=0.70)
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    with path.open("r", encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def angular_distance(values: np.ndarray, center: float) -> np.ndarray:
    return np.abs((values - center + 180.0) % 360.0 - 180.0)


def polygon_mask(segmentation: list[list[float]], size: tuple[int, int]) -> np.ndarray:
    mask_image = Image.new("L", size, 0)
    draw = ImageDraw.Draw(mask_image)
    for polygon in segmentation:
        if len(polygon) >= 6:
            draw.polygon(list(zip(polygon[0::2], polygon[1::2])), fill=255)
    return np.asarray(mask_image) >= 128


def classify_color(
    hsv: np.ndarray,
    mask: np.ndarray,
    hue_tolerance: float,
    min_saturation: int,
    min_value: int,
) -> dict[str, Any]:
    hue = hsv[..., 0].astype(np.float32) * (360.0 / 255.0)
    saturation = hsv[..., 1]
    value = hsv[..., 2]
    valid = mask & (saturation >= min_saturation) & (value >= min_value)
    mask_pixels = max(int(mask.sum()), 1)
    color_names = list(COLOR_CENTERS)
    distances = np.stack(
        [angular_distance(hue, COLOR_CENTERS[color]) for color in color_names],
        axis=-1,
    )
    nearest_index = np.argmin(distances, axis=-1)
    nearest_distance = np.min(distances, axis=-1)
    metrics = {}
    for index, color in enumerate(color_names):
        selected = valid & (nearest_index == index) & (nearest_distance <= hue_tolerance)
        metrics[color] = {
            "coverage": float(selected.sum() / mask_pixels),
            "selected_pixels": int(selected.sum()),
        }
    dominant_color = max(metrics, key=lambda color: metrics[color]["coverage"])
    dominant_index = color_names.index(dominant_color)
    selected = (
        valid
        & (nearest_index == dominant_index)
        & (nearest_distance <= hue_tolerance)
    )
    metrics[dominant_color]["median_saturation"] = (
        float(np.median(saturation[selected])) if selected.any() else 0.0
    )
    metrics[dominant_color]["mean_hue_distance"] = (
        float(np.mean(angular_distance(hue[selected], COLOR_CENTERS[dominant_color])))
        if selected.any()
        else 180.0
    )
    return {
        "dominant_color": dominant_color,
        "dominant_coverage": metrics[dominant_color]["coverage"],
        "color_metrics": metrics,
    }


def mask_boundary(mask: np.ndarray) -> np.ndarray:
    interior = mask.copy()
    interior[1:, :] &= mask[:-1, :]
    interior[:-1, :] &= mask[1:, :]
    interior[:, 1:] &= mask[:, :-1]
    interior[:, :-1] &= mask[:, 1:]
    return mask & ~interior


def save_preview(
    image: Image.Image,
    mask_a: np.ndarray,
    mask_b: np.ndarray,
    row: dict[str, Any],
    path: Path,
) -> None:
    array = np.asarray(image).copy()
    array[mask_boundary(mask_a)] = BOUNDARY_COLORS["a"]
    array[mask_boundary(mask_b)] = BOUNDARY_COLORS["b"]
    preview = Image.fromarray(array)
    draw = ImageDraw.Draw(preview)
    lines = [
        f"A: {row['object_a']['color']} {row['object_a']['label']}",
        f"B: {row['object_b']['color']} {row['object_b']['label']}",
    ]
    y = 5
    for line, color in zip(lines, (BOUNDARY_COLORS["a"], BOUNDARY_COLORS["b"])):
        box = draw.textbbox((5, y), line)
        draw.rectangle((box[0] - 2, box[1] - 2, box[2] + 2, box[3] + 2), fill="black")
        draw.text((5, y), line, fill=color)
        y = box[3] + 7
    preview.save(path, quality=95)


def main() -> None:
    args = parse_args()
    if args.output_dir.exists() and any(args.output_dir.iterdir()) and not args.overwrite:
        raise FileExistsError(f"Output directory is not empty: {args.output_dir}")

    pairs = read_jsonl(args.pair_manifest)
    wanted_ids = {
        obj["annotation_id"] for row in pairs for obj in (row["object_a"], row["object_b"])
    }
    with args.annotation.open("r", encoding="utf-8") as handle:
        data = json.load(handle)
    annotations = {ann["id"]: ann for ann in data["annotations"] if ann["id"] in wanted_ids}

    masks_dir = args.output_dir / "masks"
    originals_dir = args.output_dir / "originals"
    previews_dir = args.output_dir / "previews"
    for directory in (masks_dir, originals_dir, previews_dir):
        directory.mkdir(parents=True, exist_ok=True)

    screened = []
    accepted = []
    for row in pairs:
        image_path = args.image_dir / f"{row['image_id']:012d}.jpg"
        image = Image.open(image_path).convert("RGB")
        hsv = np.asarray(image.convert("HSV"))
        ann_a = annotations[row["object_a"]["annotation_id"]]
        ann_b = annotations[row["object_b"]["annotation_id"]]
        mask_a = polygon_mask(ann_a["segmentation"], image.size)
        mask_b = polygon_mask(ann_b["segmentation"], image.size)
        color_a = classify_color(
            hsv, mask_a, args.hue_tolerance, args.min_saturation, args.min_value
        )
        color_b = classify_color(
            hsv, mask_b, args.hue_tolerance, args.min_saturation, args.min_value
        )
        passes = (
            color_a["dominant_color"] != color_b["dominant_color"]
            and color_a["dominant_coverage"] >= args.min_color_coverage
            and color_b["dominant_coverage"] >= args.min_color_coverage
        )
        enriched = {
            **row,
            "object_a": {**row["object_a"], "color": color_a["dominant_color"]},
            "object_b": {**row["object_b"], "color": color_b["dominant_color"]},
            "pixel_color_metrics_a": color_a,
            "pixel_color_metrics_b": color_b,
            "pixel_color_status": "pass" if passes else "reject",
        }
        screened.append(enriched)
        if not passes:
            continue

        candidate_id = f"paco_color_{len(accepted) + 1:03d}"
        mask_a_path = masks_dir / f"{candidate_id}_object_a.png"
        mask_b_path = masks_dir / f"{candidate_id}_object_b.png"
        original_path = originals_dir / f"{candidate_id}.jpg"
        preview_path = previews_dir / f"{candidate_id}_preview.jpg"
        Image.fromarray(mask_a.astype(np.uint8) * 255, mode="L").save(mask_a_path)
        Image.fromarray(mask_b.astype(np.uint8) * 255, mode="L").save(mask_b_path)
        shutil.copy2(image_path, original_path)
        accepted_row = {
            **enriched,
            "segmentation_id": candidate_id,
            "color_pair": "_".join(sorted((color_a["dominant_color"], color_b["dominant_color"]))),
            "mask_a_path": str(mask_a_path.relative_to(args.output_dir)),
            "mask_b_path": str(mask_b_path.relative_to(args.output_dir)),
            "original_path": str(original_path.relative_to(args.output_dir)),
            "preview_path": str(preview_path.relative_to(args.output_dir)),
            "human_review_status": "pending",
        }
        save_preview(image, mask_a, mask_b, accepted_row, preview_path)
        accepted.append(accepted_row)

    accepted.sort(
        key=lambda row: (
            -min(
                row["pixel_color_metrics_a"]["dominant_coverage"],
                row["pixel_color_metrics_b"]["dominant_coverage"],
            ),
            row["segmentation_id"],
        )
    )
    for index, row in enumerate(accepted, start=1):
        row["review_rank"] = index

    with (args.output_dir / "screened_pairs.jsonl").open("w", encoding="utf-8") as handle:
        for row in screened:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")
    with (args.output_dir / "accepted_candidates.jsonl").open(
        "w", encoding="utf-8"
    ) as handle:
        for row in accepted:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")

    from collections import Counter

    summary = {
        "schema_version": "0.1",
        "screened_pairs": len(screened),
        "accepted_pairs": len(accepted),
        "color_pair_counts": dict(
            sorted(Counter(row["color_pair"] for row in accepted).items())
        ),
        "parameters": {
            "hue_tolerance": args.hue_tolerance,
            "min_saturation": args.min_saturation,
            "min_value": args.min_value,
            "min_color_coverage": args.min_color_coverage,
        },
        "model_outputs_used_for_decision": False,
    }
    (args.output_dir / "summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
