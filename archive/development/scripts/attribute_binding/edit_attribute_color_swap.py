#!/usr/bin/env python3
"""Swap object-color hues inside reviewed SAM masks while preserving texture."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import numpy as np
from PIL import Image


COLOR_CENTERS_DEGREES = {
    "red": 0.0,
    "yellow": 55.0,
    "green": 120.0,
    "blue": 220.0,
}


def parse_args() -> argparse.Namespace:
    project_root = Path(__file__).resolve().parents[2]
    parser = argparse.ArgumentParser(description="Swap two object-color bindings.")
    parser.add_argument(
        "--mask-manifest",
        type=Path,
        default=project_root
        / "processed/attribute_binding/segmentation/segmentation_masks_v0.2/mask_manifest.jsonl",
    )
    parser.add_argument(
        "--mask-root",
        type=Path,
        default=project_root
        / "processed/attribute_binding/segmentation/segmentation_masks_v0.2",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=project_root
        / "processed/attribute_binding/edits/color_swap_edits_v0.2",
    )
    parser.add_argument("--hue-tolerance", type=float, default=55.0)
    parser.add_argument("--min-saturation", type=int, default=45)
    parser.add_argument("--min-value", type=int, default=30)
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    with path.open("r", encoding="utf-8") as input_file:
        return [json.loads(line) for line in input_file]


def angular_distance_degrees(values: np.ndarray, center: float) -> np.ndarray:
    return np.abs((values - center + 180.0) % 360.0 - 180.0)


def circular_mean_degrees(values: np.ndarray, weights: np.ndarray) -> float:
    radians = np.deg2rad(values)
    x = float(np.sum(np.cos(radians) * weights))
    y = float(np.sum(np.sin(radians) * weights))
    if x == 0.0 and y == 0.0:
        raise ValueError("Cannot compute circular mean from zero weights")
    return float(np.rad2deg(np.arctan2(y, x)) % 360.0)


def load_mask(path: Path) -> np.ndarray:
    return np.asarray(Image.open(path).convert("L")) >= 128


def select_color_pixels(
    hsv: np.ndarray,
    object_mask: np.ndarray,
    color: str,
    hue_tolerance: float,
    min_saturation: int,
    min_value: int,
) -> np.ndarray:
    hue_degrees = hsv[..., 0].astype(np.float32) * (360.0 / 255.0)
    saturation = hsv[..., 1]
    value = hsv[..., 2]
    return (
        object_mask
        & (angular_distance_degrees(hue_degrees, COLOR_CENTERS_DEGREES[color]) <= hue_tolerance)
        & (saturation >= min_saturation)
        & (value >= min_value)
    )


def shift_hue(
    hsv: np.ndarray, selected: np.ndarray, source_hue: float, target_hue: float
) -> None:
    hue_degrees = hsv[..., 0].astype(np.float32) * (360.0 / 255.0)
    delta = (target_hue - source_hue + 180.0) % 360.0 - 180.0
    hue_degrees[selected] = (hue_degrees[selected] + delta) % 360.0
    hsv[..., 0][selected] = np.rint(hue_degrees[selected] * (255.0 / 360.0)).astype(
        np.uint8
    )


def save_selected_mask(mask: np.ndarray, path: Path) -> None:
    Image.fromarray(mask.astype(np.uint8) * 255, mode="L").save(path)


def make_comparison(original: Image.Image, edited: Image.Image, path: Path) -> None:
    width, height = original.size
    comparison = Image.new("RGB", (width * 2, height), "white")
    comparison.paste(original, (0, 0))
    comparison.paste(edited, (width, 0))
    comparison.save(path, quality=95)


def main() -> None:
    args = parse_args()
    if args.output_dir.exists() and any(args.output_dir.iterdir()) and not args.overwrite:
        raise FileExistsError(
            f"Output directory is not empty: {args.output_dir}. Use --overwrite."
        )

    originals_dir = args.output_dir / "paired_originals"
    edits_dir = args.output_dir / "edited"
    selected_masks_dir = args.output_dir / "color_masks"
    comparisons_dir = args.output_dir / "comparisons"
    for directory in (originals_dir, edits_dir, selected_masks_dir, comparisons_dir):
        directory.mkdir(parents=True, exist_ok=True)

    output_rows = []
    for row in read_jsonl(args.mask_manifest):
        sample_id = row["segmentation_id"]
        original = Image.open(args.mask_root / row["original_path"]).convert("RGB")
        hsv_image = original.convert("HSV")
        hsv = np.asarray(hsv_image).copy()
        object_mask_a = load_mask(args.mask_root / row["mask_a_path"])
        object_mask_b = load_mask(args.mask_root / row["mask_b_path"])
        color_a = row["object_a"]["color"]
        color_b = row["object_b"]["color"]

        selected_a = select_color_pixels(
            hsv,
            object_mask_a,
            color_a,
            args.hue_tolerance,
            args.min_saturation,
            args.min_value,
        )
        selected_b = select_color_pixels(
            hsv,
            object_mask_b,
            color_b,
            args.hue_tolerance,
            args.min_saturation,
            args.min_value,
        )
        if not selected_a.any() or not selected_b.any():
            raise RuntimeError(f"No source-color pixels selected for {sample_id}")

        hue_degrees = hsv[..., 0].astype(np.float32) * (360.0 / 255.0)
        source_hue_a = circular_mean_degrees(
            hue_degrees[selected_a], hsv[..., 1][selected_a].astype(np.float32)
        )
        source_hue_b = circular_mean_degrees(
            hue_degrees[selected_b], hsv[..., 1][selected_b].astype(np.float32)
        )
        shift_hue(hsv, selected_a, source_hue_a, source_hue_b)
        shift_hue(hsv, selected_b, source_hue_b, source_hue_a)

        edited_array = np.asarray(Image.fromarray(hsv, mode="HSV").convert("RGB")).copy()
        original_array = np.asarray(original)
        changed_mask = selected_a | selected_b
        edited_array[~changed_mask] = original_array[~changed_mask]
        edited = Image.fromarray(edited_array, mode="RGB")
        original_pair_path = originals_dir / f"{sample_id}_original.png"
        edit_path = edits_dir / f"{sample_id}_color_swap.png"
        selected_a_path = selected_masks_dir / f"{sample_id}_color_a.png"
        selected_b_path = selected_masks_dir / f"{sample_id}_color_b.png"
        comparison_path = comparisons_dir / f"{sample_id}_before_after.jpg"
        original.save(original_pair_path)
        edited.save(edit_path)
        save_selected_mask(selected_a, selected_a_path)
        save_selected_mask(selected_b, selected_b_path)
        make_comparison(original, edited, comparison_path)

        object_pixels_a = max(int(object_mask_a.sum()), 1)
        object_pixels_b = max(int(object_mask_b.sum()), 1)
        enriched = {
            **row,
            "edit_method": "sam_mask_intersect_source_hue_then_swap_hue_means",
            "hue_tolerance_degrees": args.hue_tolerance,
            "min_saturation_0_255": args.min_saturation,
            "min_value_0_255": args.min_value,
            "source_hue_a_degrees": source_hue_a,
            "source_hue_b_degrees": source_hue_b,
            "selected_color_fraction_a": float(selected_a.sum() / object_pixels_a),
            "selected_color_fraction_b": float(selected_b.sum() / object_pixels_b),
            "changed_image_fraction": float(changed_mask.mean()),
            "outside_edit_pixels_identical": bool(
                np.array_equal(edited_array[~changed_mask], original_array[~changed_mask])
            ),
            "paired_original_path": str(original_pair_path.relative_to(args.output_dir)),
            "edited_path": str(edit_path.relative_to(args.output_dir)),
            "selected_color_mask_a_path": str(
                selected_a_path.relative_to(args.output_dir)
            ),
            "selected_color_mask_b_path": str(
                selected_b_path.relative_to(args.output_dir)
            ),
            "comparison_path": str(comparison_path.relative_to(args.output_dir)),
            "human_edit_review": "pending",
        }
        output_rows.append(enriched)
        print(
            f"{sample_id}: hue {source_hue_a:.1f}<->{source_hue_b:.1f}, "
            f"coverage {enriched['selected_color_fraction_a']:.3f}/"
            f"{enriched['selected_color_fraction_b']:.3f}"
        )

    with (args.output_dir / "edit_manifest.jsonl").open(
        "w", encoding="utf-8"
    ) as output_file:
        for row in output_rows:
            output_file.write(json.dumps(row, ensure_ascii=False) + "\n")

    (args.output_dir / "README.md").write_text(
        """# Attribute-Binding Color-Swap Edits v0.2

Each edit intersects the automatic object mask with source-color pixels, then
swaps the two objects' circular mean hue while preserving saturation, value,
texture, and neutral internal regions. Every comparison requires human review.

No sample is accepted solely from automatic mask scores or color coverage.
""",
        encoding="utf-8",
    )


if __name__ == "__main__":
    main()
