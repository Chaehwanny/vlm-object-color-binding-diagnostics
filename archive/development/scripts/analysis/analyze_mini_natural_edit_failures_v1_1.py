#!/usr/bin/env python3
"""Read-only pixel diagnostics for Mini-10 Natural color edits.

The script reads frozen source assets/manifests and writes analysis tables only.
It never writes image, segmentation, edit, QC, or benchmark manifests.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
from collections import Counter
from pathlib import Path
from typing import Any

import numpy as np
from PIL import Image
from scipy import ndimage

RULE_VERSION = "mini_natural_edit_failure_diagnostics_v1.1"
FAILED_IDS = {"maincand_0011", "maincand_0020", "maincand_0082", "maincand_0179"}
FAILURE_SIDE = {
    ("maincand_0011", "b"): "low_selected_pixel_coverage",
    ("maincand_0020", "b"): "human_confirmed_target_mask_contamination",
    ("maincand_0082", "a"): "near_source_target_hue_with_partial_selection",
    ("maincand_0179", "a"): "partial_spatial_selection_on_multiregion_surface",
}
FINAL_RECOMMENDATION = {
    "choice": "A",
    "algorithm_action": "retain_masked_observed_hue_mean_exchange_hsv_v1",
    "smoke_mini_rerun_required": False,
    "completion_tuning_allowed": False,
    "frozen_preflight_exclusion_rules": {
        "mask_semantic_purity": "exclude material non-target inclusion",
        "minimum_selected_color_ratio": 0.05,
        "minimum_observed_source_target_hue_distance_degrees": 45.0,
        "maximum_low_saturation_ratio": 0.50,
    },
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--generated-color-manifest", type=Path, required=True)
    parser.add_argument("--reviewed-color-manifest", type=Path, required=True)
    parser.add_argument("--mask-manifest", type=Path, required=True)
    parser.add_argument("--color-root", type=Path, required=True)
    parser.add_argument("--segmentation-root", type=Path, required=True)
    parser.add_argument("--project-root", type=Path, required=True)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--diagnostics-output", type=Path, required=True)
    parser.add_argument("--summary-output", type=Path, required=True)
    return parser.parse_args()


def read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    with path.open(encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def resolve(root: Path, value: str) -> Path:
    path = Path(value)
    return path if path.is_absolute() else root / path


def load_rgb(path: Path) -> np.ndarray:
    with Image.open(path) as image:
        return np.asarray(image.convert("RGB"))


def load_rgba(path: Path) -> np.ndarray:
    with Image.open(path) as image:
        return np.asarray(image.convert("RGBA"))


def load_mask(path: Path) -> np.ndarray:
    with Image.open(path) as image:
        return np.asarray(image.convert("L")) > 0


def rgb_to_hsv(rgb: np.ndarray) -> np.ndarray:
    return np.asarray(Image.fromarray(rgb, mode="RGB").convert("HSV"))


def hue_degrees(hsv: np.ndarray) -> np.ndarray:
    return hsv[..., 0].astype(np.float64) * (360.0 / 255.0)


def angular_distance(a: np.ndarray | float, b: np.ndarray | float) -> np.ndarray:
    return np.abs((np.asarray(a) - np.asarray(b) + 180.0) % 360.0 - 180.0)


def circular_mean(values: np.ndarray, weights: np.ndarray | None = None) -> float | None:
    if values.size == 0:
        return None
    radians = np.deg2rad(values)
    if weights is None:
        weights = np.ones(values.shape, dtype=np.float64)
    total = float(weights.sum())
    if total <= 0:
        return None
    x = float((np.cos(radians) * weights).sum() / total)
    y = float((np.sin(radians) * weights).sum() / total)
    return float(np.rad2deg(math.atan2(y, x)) % 360.0)


def distribution(values: np.ndarray) -> dict[str, float | None]:
    if values.size == 0:
        return {"mean": None, "median": None, "p10": None, "p90": None}
    values = values.astype(np.float64)
    return {
        "mean": float(values.mean()),
        "median": float(np.median(values)),
        "p10": float(np.percentile(values, 10)),
        "p90": float(np.percentile(values, 90)),
    }


def bbox_area(bbox: list[int]) -> int:
    return max(int(bbox[2]) - int(bbox[0]), 0) * max(int(bbox[3]) - int(bbox[1]), 0)


def component_selection(mask: np.ndarray, selected: np.ndarray) -> list[dict[str, Any]]:
    labels, count = ndimage.label(mask, structure=np.ones((3, 3), dtype=np.uint8))
    records = []
    for component_id in range(1, count + 1):
        component = labels == component_id
        area = int(component.sum())
        selected_count = int((component & selected).sum())
        records.append(
            {
                "component_id": component_id,
                "mask_pixels": area,
                "selected_pixels": selected_count,
                "selected_ratio": selected_count / area if area else 0.0,
            }
        )
    return sorted(records, key=lambda row: row["mask_pixels"], reverse=True)


def grid_coverage(mask: np.ndarray, selected: np.ndarray, grid: int = 4) -> float:
    ys, xs = np.where(mask)
    if not ys.size:
        return 0.0
    y0, y1, x0, x1 = ys.min(), ys.max() + 1, xs.min(), xs.max() + 1
    occupied = covered = 0
    for gy in range(grid):
        for gx in range(grid):
            ya = y0 + (y1 - y0) * gy // grid
            yb = y0 + (y1 - y0) * (gy + 1) // grid
            xa = x0 + (x1 - x0) * gx // grid
            xb = x0 + (x1 - x0) * (gx + 1) // grid
            cell_mask = mask[ya:yb, xa:xb]
            if cell_mask.any():
                occupied += 1
                if (selected[ya:yb, xa:xb] & cell_mask).any():
                    covered += 1
    return covered / occupied if occupied else 0.0


def selected_bbox_coverage(mask: np.ndarray, selected: np.ndarray) -> float:
    mask_y, mask_x = np.where(mask)
    selected_y, selected_x = np.where(selected)
    if not mask_y.size or not selected_y.size:
        return 0.0
    mask_area = (mask_y.max() - mask_y.min() + 1) * (mask_x.max() - mask_x.min() + 1)
    selected_area = (selected_y.max() - selected_y.min() + 1) * (selected_x.max() - selected_x.min() + 1)
    return float(selected_area / mask_area)


def selection_at_tolerance(
    hue: np.ndarray,
    saturation: np.ndarray,
    value: np.ndarray,
    mask: np.ndarray,
    center: float,
    tolerance: float,
    min_saturation: int,
    min_value: int,
) -> np.ndarray:
    return mask & (angular_distance(hue, center) <= tolerance) & (saturation >= min_saturation) & (value >= min_value)


def ratio(numerator: np.ndarray, denominator: np.ndarray) -> float:
    count = int(denominator.sum())
    return float((numerator & denominator).sum() / count) if count else 0.0


def atomic_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    if path.exists():
        raise FileExistsError(f"Refusing to overwrite: {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_name(f".{path.name}.tmp")
    with temp.open("x", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")
    temp.replace(path)


def atomic_json(path: Path, value: dict[str, Any]) -> None:
    if path.exists():
        raise FileExistsError(f"Refusing to overwrite: {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_name(f".{path.name}.tmp")
    temp.write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    temp.replace(path)


def main() -> None:
    args = parse_args()
    source_paths = (
        args.generated_color_manifest,
        args.reviewed_color_manifest,
        args.mask_manifest,
        args.config,
    )
    source_checksums_before = {path.as_posix(): sha256(path) for path in source_paths}
    generated_rows = read_jsonl(args.generated_color_manifest)
    reviewed_rows = read_jsonl(args.reviewed_color_manifest)
    mask_rows = read_jsonl(args.mask_manifest)
    config = read_json(args.config)
    generated = {row["candidate_id"]: row for row in generated_rows}
    reviewed = {row["candidate_id"]: row for row in reviewed_rows}
    masks = {row["candidate_id"]: row for row in mask_rows}
    if not (set(generated) == set(reviewed) == set(masks)) or len(generated) != 10:
        raise ValueError("Mini generated/reviewed/mask manifests must share exactly ten candidates")

    method = config["method"]
    centers = method["color_centers_degrees"]
    tolerance = float(method["source_hue_tolerance_degrees"])
    min_saturation = int(method["min_saturation_0_255"])
    min_value = int(method["min_value_0_255"])
    diagnostics: list[dict[str, Any]] = []

    for cid in sorted(generated):
        color = generated[cid]
        review = reviewed[cid]
        mask_record = masks[cid]
        original_path = resolve(args.project_root, color["original_image_path"])
        original_rgb = load_rgb(original_path)
        original_hsv = rgb_to_hsv(original_rgb)
        hue = hue_degrees(original_hsv)
        saturation = original_hsv[..., 1]
        value = original_hsv[..., 2]
        natural_edited = load_rgb(resolve(args.color_root, color["natural_edited_path"]))
        edited_hsv = rgb_to_hsv(natural_edited)
        edited_hue = hue_degrees(edited_hsv)

        for side in ("a", "b"):
            other = "b" if side == "a" else "a"
            object_record = mask_record[f"object_{side}"]
            mask = load_mask(resolve(args.segmentation_root, color[f"object_{side}_mask_path"]))
            selected = load_mask(resolve(args.color_root, color[f"selected_color_mask_{side}_path"]))
            original_cutout = load_rgba(resolve(args.color_root, color[f"object_{side}_original_cutout_path"]))
            edited_cutout = load_rgba(resolve(args.color_root, color[f"object_{side}_edited_cutout_path"]))
            source_hue = float(color["automatic_qc_metrics"]["pair"][f"source_hue_{side}_degrees"])
            target_hue = float(color["automatic_qc_metrics"]["pair"][f"source_hue_{other}_degrees"])
            mask_pixels = int(mask.sum())
            selected_pixels = int(selected.sum())
            alpha_valid = int((original_cutout[..., 3] > 0).sum())
            changed = np.any(natural_edited != original_rgb, axis=2)
            original_selected_mean = circular_mean(hue[selected], saturation[selected].astype(np.float64))
            edited_selected_mean = circular_mean(edited_hue[selected], edited_hsv[..., 1][selected].astype(np.float64))
            valid_hue_mask = mask & (saturation >= min_saturation) & (value >= min_value)
            target_agreement_selected = ratio(angular_distance(edited_hue, target_hue) <= tolerance, selected)
            target_agreement_mask = ratio(angular_distance(edited_hue, target_hue) <= tolerance, valid_hue_mask)
            original_residue_selected = ratio(
                (angular_distance(edited_hue, source_hue) <= tolerance)
                & (angular_distance(edited_hue, source_hue) < angular_distance(edited_hue, target_hue)),
                selected,
            )
            original_residue_mask = ratio(
                (angular_distance(edited_hue, source_hue) <= tolerance)
                & (angular_distance(edited_hue, source_hue) < angular_distance(edited_hue, target_hue)),
                valid_hue_mask,
            )
            selection_75 = selection_at_tolerance(hue, saturation, value, mask, centers[color[f"original_color_{side}"]], 75.0, min_saturation, min_value)
            selection_90 = selection_at_tolerance(hue, saturation, value, mask, centers[color[f"original_color_{side}"]], 90.0, min_saturation, min_value)
            components = component_selection(mask, selected)
            alpha_matches_mask = bool(np.array_equal(original_cutout[..., 3] > 0, mask) and np.array_equal(edited_cutout[..., 3] > 0, mask))
            natural_matches_cutout = bool(np.array_equal(natural_edited[mask], edited_cutout[..., :3][mask]))
            bbox_pixels = bbox_area(object_record["bbox_xyxy_pixels"])
            inferred = FAILURE_SIDE.get((cid, side), "none")
            if cid in FAILED_IDS and inferred == "none":
                inferred = "counterpart_object_failure"
            diagnostics.append(
                {
                    "schema_version": "1.1",
                    "diagnostic_rule_version": RULE_VERSION,
                    "candidate_id": cid,
                    "object_side": side,
                    "object_label": color[f"object_{side}_label"],
                    "original_color": color[f"original_color_{side}"],
                    "target_color": color[f"target_color_{side}"],
                    "human_qc_outcome": review["edit_qc_status"],
                    "object_identity_outcome": review["object_identity_preservation_status"],
                    "mask_pixel_count": mask_pixels,
                    "alpha_valid_cutout_pixel_count": alpha_valid,
                    "bbox_pixel_area": bbox_pixels,
                    "mask_area_over_bbox_area": mask_pixels / bbox_pixels if bbox_pixels else None,
                    "selected_color_pixel_count": selected_pixels,
                    "selected_color_pixel_ratio": selected_pixels / mask_pixels if mask_pixels else 0.0,
                    "selection_ratio_tolerance_75": float(selection_75.sum() / mask_pixels) if mask_pixels else 0.0,
                    "selection_ratio_tolerance_90": float(selection_90.sum() / mask_pixels) if mask_pixels else 0.0,
                    "original_hue_circular_mean_degrees": original_selected_mean,
                    "recorded_source_hue_degrees": source_hue,
                    "target_hue_degrees": target_hue,
                    "source_target_hue_distance_degrees": float(angular_distance(source_hue, target_hue)),
                    "edited_hue_circular_mean_degrees": edited_selected_mean,
                    "edited_mean_to_target_distance_degrees": None if edited_selected_mean is None else float(angular_distance(edited_selected_mean, target_hue)),
                    "before_after_hue_change_degrees": None if original_selected_mean is None or edited_selected_mean is None else float(angular_distance(original_selected_mean, edited_selected_mean)),
                    "expected_hue_shift_degrees": float((target_hue - source_hue + 180.0) % 360.0 - 180.0),
                    "saturation_distribution_mask": distribution(saturation[mask]),
                    "value_distribution_mask": distribution(value[mask]),
                    "low_saturation_pixel_ratio": float((saturation[mask] < min_saturation).mean()) if mask_pixels else 0.0,
                    "low_value_pixel_ratio": float((value[mask] < min_value).mean()) if mask_pixels else 0.0,
                    "original_color_residue_ratio_selected": original_residue_selected,
                    "original_color_residue_ratio_mask": original_residue_mask,
                    "target_color_agreement_ratio_selected": target_agreement_selected,
                    "target_color_agreement_ratio_mask": target_agreement_mask,
                    "component_selection_ratios": components,
                    "mask_component_count": len(components),
                    "selected_component_count": int(ndimage.label(selected, structure=np.ones((3, 3), dtype=np.uint8))[1]),
                    "spatial_coverage_grid_4x4": grid_coverage(mask, selected),
                    "selected_bbox_spatial_coverage": selected_bbox_coverage(mask, selected),
                    "changed_pixel_ratio_within_mask": float(changed[mask].mean()) if mask_pixels else 0.0,
                    "mean_abs_rgb_change_within_mask": float(np.abs(natural_edited.astype(np.int16) - original_rgb.astype(np.int16))[mask].mean()) if mask_pixels else 0.0,
                    "alpha_matches_mask": alpha_matches_mask,
                    "natural_composition_matches_edited_cutout": natural_matches_cutout,
                    "sam_predicted_iou": mask_record["automatic_metrics"][f"object_{side}"]["predicted_iou"],
                    "mask_contamination_indicator": "human_confirmed" if (cid, side) == ("maincand_0020", "b") else "not_observed",
                    "inferred_failure_mechanism": inferred,
                }
            )

    passed = [row for row in diagnostics if row["human_qc_outcome"] == "pass"]
    failed = [row for row in diagnostics if row["human_qc_outcome"] == "fail"]

    def aggregate(rows: list[dict[str, Any]], field: str) -> dict[str, float | None]:
        values = [float(row[field]) for row in rows if row[field] is not None]
        if not values:
            return {"median": None, "min": None, "max": None}
        return {"median": float(np.median(values)), "min": min(values), "max": max(values)}

    source_checksums_after = {path.as_posix(): sha256(path) for path in source_paths}
    if source_checksums_before != source_checksums_after:
        raise RuntimeError("A frozen source manifest/config changed during read-only analysis")
    summary = {
        "schema_version": "1.1",
        "diagnostic_rule_version": RULE_VERSION,
        "candidate_count": 10,
        "object_row_count": len(diagnostics),
        "human_pass_candidates": 6,
        "human_fail_candidates": 4,
        "failure_candidate_ids": sorted(FAILED_IDS),
        "source_checksums": source_checksums_before,
        "source_files_unchanged": True,
        "aggregate_by_candidate_outcome": {
            "pass": {
                "selected_color_pixel_ratio": aggregate(passed, "selected_color_pixel_ratio"),
                "changed_pixel_ratio_within_mask": aggregate(passed, "changed_pixel_ratio_within_mask"),
                "target_distance_after_edit": aggregate(passed, "edited_mean_to_target_distance_degrees"),
                "original_color_residue_ratio_mask": aggregate(passed, "original_color_residue_ratio_mask"),
                "spatial_coverage_grid_4x4": aggregate(passed, "spatial_coverage_grid_4x4"),
                "low_saturation_pixel_ratio": aggregate(passed, "low_saturation_pixel_ratio"),
            },
            "fail": {
                "selected_color_pixel_ratio": aggregate(failed, "selected_color_pixel_ratio"),
                "changed_pixel_ratio_within_mask": aggregate(failed, "changed_pixel_ratio_within_mask"),
                "target_distance_after_edit": aggregate(failed, "edited_mean_to_target_distance_degrees"),
                "original_color_residue_ratio_mask": aggregate(failed, "original_color_residue_ratio_mask"),
                "spatial_coverage_grid_4x4": aggregate(failed, "spatial_coverage_grid_4x4"),
                "low_saturation_pixel_ratio": aggregate(failed, "low_saturation_pixel_ratio"),
            },
        },
        "mechanism_counts": dict(Counter(row["inferred_failure_mechanism"] for row in diagnostics if row["inferred_failure_mechanism"] not in {"none", "counterpart_object_failure"})),
        "final_recommendation": FINAL_RECOMMENDATION,
        "analysis_scope": "developmental diagnostic only; no statistical generalization",
        "image_processing_executed": False,
    }
    atomic_jsonl(args.diagnostics_output, diagnostics)
    atomic_json(args.summary_output, summary)
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
