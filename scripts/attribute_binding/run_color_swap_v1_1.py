#!/usr/bin/env python3
"""Create deterministic edited cutouts and Natural Edited images for a frozen subset."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import platform
import random
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
from PIL import Image, ImageDraw, ImageOps, __version__ as PILLOW_VERSION


QC_CRITERIA = (
    "target_color_change",
    "original_color_residue",
    "outside_mask_preservation",
    "color_leakage",
    "boundary_quality",
    "object_identity_preservation",
    "texture_shading_pattern_preservation",
    "binding_swap_achieved",
    "prompt_answer_validity",
)


def parse_args() -> argparse.Namespace:
    root = Path(__file__).resolve().parents[2]
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-manifest", "--mask-results", dest="input_manifest", type=Path, required=True)
    parser.add_argument("--subset-name", required=True)
    parser.add_argument("--segmentation-root", type=Path, required=True)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--project-root", type=Path, default=root)
    parser.add_argument("--resume", action="store_true")
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


def angular_distance(values: np.ndarray, center: float) -> np.ndarray:
    return np.abs((values - center + 180.0) % 360.0 - 180.0)


def circular_mean(values: np.ndarray, weights: np.ndarray) -> float:
    radians = np.deg2rad(values)
    x = float(np.sum(np.cos(radians) * weights))
    y = float(np.sum(np.sin(radians) * weights))
    if abs(x) + abs(y) < 1e-12:
        raise ValueError("Cannot estimate source hue from zero circular weight.")
    return float(np.rad2deg(np.arctan2(y, x)) % 360.0)


def load_binary_mask(path: Path, expected_size: tuple[int, int]) -> np.ndarray:
    with Image.open(path) as image:
        array = np.asarray(image.convert("L"))
    if array.shape != (expected_size[1], expected_size[0]):
        raise ValueError(f"Mask/image size mismatch: {path}")
    if not set(np.unique(array).tolist()).issubset({0, 255}):
        raise ValueError(f"Mask is not binary: {path}")
    return array > 0


def select_source_pixels(
    hsv: np.ndarray,
    mask: np.ndarray,
    color: str,
    centers: dict[str, float],
    tolerance: float,
    min_saturation: int,
    min_value: int,
) -> np.ndarray:
    hue = hsv[..., 0].astype(np.float32) * (360.0 / 255.0)
    return (
        mask
        & (angular_distance(hue, centers[color]) <= tolerance)
        & (hsv[..., 1] >= min_saturation)
        & (hsv[..., 2] >= min_value)
    )


def shifted_rgb(
    original_rgb: np.ndarray,
    original_hsv: np.ndarray,
    selected: np.ndarray,
    source_hue: float,
    target_hue: float,
) -> np.ndarray:
    shifted_hsv = original_hsv.copy()
    hue = shifted_hsv[..., 0].astype(np.float32) * (360.0 / 255.0)
    delta = (target_hue - source_hue + 180.0) % 360.0 - 180.0
    hue[selected] = (hue[selected] + delta) % 360.0
    shifted_hsv[..., 0][selected] = np.rint(
        hue[selected] * (255.0 / 360.0)
    ).astype(np.uint8)
    converted = np.asarray(
        Image.fromarray(shifted_hsv, mode="HSV").convert("RGB")
    )
    result = original_rgb.copy()
    result[selected] = converted[selected]
    return result


def rgba_cutout(rgb: np.ndarray, mask: np.ndarray) -> Image.Image:
    rgba = np.zeros((*mask.shape, 4), dtype=np.uint8)
    rgba[..., :3] = rgb
    rgba[..., 3] = mask.astype(np.uint8) * 255
    return Image.fromarray(rgba, mode="RGBA")


def boundary(mask: np.ndarray) -> np.ndarray:
    padded = np.pad(mask, 1, constant_values=False)
    eroded = mask.copy()
    for y in range(3):
        for x in range(3):
            eroded &= padded[y : y + mask.shape[0], x : x + mask.shape[1]]
    return mask & ~eroded


def mean_abs_change(original: np.ndarray, edited: np.ndarray, area: np.ndarray) -> float:
    if not area.any():
        return 0.0
    delta = np.abs(edited.astype(np.int16) - original.astype(np.int16))
    return float(delta[area].mean())


def hue_metrics(
    edited_rgb: np.ndarray,
    selected: np.ndarray,
    source_hue: float,
    target_hue: float,
    tolerance: float,
) -> dict[str, float]:
    if not selected.any():
        return {
            "post_target_hue_distance_mean_degrees": 180.0,
            "original_color_residue_fraction": 1.0,
        }
    hsv = np.asarray(Image.fromarray(edited_rgb, mode="RGB").convert("HSV"))
    hue = hsv[..., 0].astype(np.float32) * (360.0 / 255.0)
    target_distance = angular_distance(hue[selected], target_hue)
    source_distance = angular_distance(hue[selected], source_hue)
    residue = (source_distance <= tolerance) & (source_distance < target_distance)
    return {
        "post_target_hue_distance_mean_degrees": float(target_distance.mean()),
        "original_color_residue_fraction": float(residue.mean()),
    }


def atomic_image(image: Image.Image, path: Path) -> None:
    if path.exists():
        raise FileExistsError(f"Refusing to overwrite: {path}")
    temporary = path.with_name(f".{path.name}.tmp")
    if temporary.exists():
        raise FileExistsError(f"Stale temporary file: {temporary}")
    image.save(temporary, format="PNG")
    temporary.replace(path)


def atomic_json(path: Path, value: dict[str, Any]) -> None:
    if path.exists():
        raise FileExistsError(f"Refusing to overwrite: {path}")
    temporary = path.with_name(f".{path.name}.tmp")
    if temporary.exists():
        raise FileExistsError(f"Stale temporary file: {temporary}")
    temporary.write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    temporary.replace(path)


def atomic_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    if path.exists():
        raise FileExistsError(f"Refusing to overwrite: {path}")
    temporary = path.with_name(f".{path.name}.tmp")
    if temporary.exists():
        raise FileExistsError(f"Stale temporary file: {temporary}")
    with temporary.open("x", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")
    temporary.replace(path)


def asset(path: Path, root: Path) -> dict[str, str]:
    return {"path": path.relative_to(root).as_posix(), "sha256": sha256(path)}


def contact_sheet(
    original: Image.Image,
    natural_edited: Image.Image,
    original_a: Image.Image,
    edited_a: Image.Image,
    original_b: Image.Image,
    edited_b: Image.Image,
) -> Image.Image:
    tiles = (
        ("Natural Original", original.convert("RGB")),
        ("Natural Edited", natural_edited.convert("RGB")),
        ("Object A Original", original_a),
        ("Object A Edited", edited_a),
        ("Object B Original", original_b),
        ("Object B Edited", edited_b),
    )
    tile_w, tile_h, label_h = 360, 270, 26
    canvas = Image.new("RGB", (tile_w * 2, (tile_h + label_h) * 3), "white")
    draw = ImageDraw.Draw(canvas)
    for index, (label, image) in enumerate(tiles):
        rgb = Image.new("RGB", image.size, "white")
        if image.mode == "RGBA":
            rgb.paste(image, mask=image.getchannel("A"))
        else:
            rgb.paste(image.convert("RGB"))
        fitted = ImageOps.contain(rgb, (tile_w, tile_h))
        x = (index % 2) * tile_w + (tile_w - fitted.width) // 2
        y0 = (index // 2) * (tile_h + label_h)
        y = y0 + label_h + (tile_h - fitted.height) // 2
        canvas.paste(fitted, (x, y))
        draw.text(((index % 2) * tile_w + 8, y0 + 6), label, fill="black")
    return canvas


def qc_template(record: dict[str, Any]) -> dict[str, Any]:
    return {
        "schema_version": "1.1",
        "subset_name": record["pilot_subset"],
        "candidate_id": record["candidate_id"],
        "criteria": {name: "not_tested" for name in QC_CRITERIA},
        "edit_qc_status": "not_tested",
        "object_identity_preservation_status": "not_tested",
        "failure_reasons": [],
        "review_note": None,
        "reviewer": None,
        "review_timestamp": None,
    }


def paths_for(root: Path, layout: dict[str, str], cid: str) -> dict[str, Path]:
    return {
        "original_a": root / layout["original_cutouts_dir"] / f"{cid}__object_a_original.png",
        "original_b": root / layout["original_cutouts_dir"] / f"{cid}__object_b_original.png",
        "edited_a": root / layout["edited_cutouts_dir"] / f"{cid}__object_a_edited.png",
        "edited_b": root / layout["edited_cutouts_dir"] / f"{cid}__object_b_edited.png",
        "selected_a": root / layout["selected_masks_dir"] / f"{cid}__object_a_selected.png",
        "selected_b": root / layout["selected_masks_dir"] / f"{cid}__object_b_selected.png",
        "natural": root / layout["natural_edited_dir"] / f"{cid}__natural_edited.png",
        "contact": root / layout["contact_sheets_dir"] / f"{cid}__before_after.png",
        "record": root / layout["records_dir"] / f"{cid}.json",
    }


def verify_resume(record: dict[str, Any], expected: dict[str, Any], root: Path) -> None:
    for field in ("candidate_id", "source_image_sha256", "mask_a_sha256", "mask_b_sha256", "color_swap_config_sha256"):
        if record.get(field) != expected.get(field):
            raise ValueError(f"Resume provenance mismatch: {record.get('candidate_id')} {field}")
    if record.get("generation_status") == "generated":
        for field in (
            "object_a_original_cutout_path", "object_b_original_cutout_path",
            "object_a_edited_cutout_path", "object_b_edited_cutout_path",
            "natural_edited_path", "contact_sheet_path",
        ):
            path = root / record[field]
            checksum_field = {
                "object_a_original_cutout_path": "original_cutout_a_sha256",
                "object_b_original_cutout_path": "original_cutout_b_sha256",
                "object_a_edited_cutout_path": "edited_cutout_a_sha256",
                "object_b_edited_cutout_path": "edited_cutout_b_sha256",
                "natural_edited_path": "natural_edited_sha256",
                "contact_sheet_path": "contact_sheet_sha256",
            }[field]
            if not path.is_file() or sha256(path) != record[checksum_field]:
                raise ValueError(f"Resume asset mismatch: {path}")


def main() -> None:
    args = parse_args()
    config = read_json(args.config)
    rows = read_jsonl(args.input_manifest)
    scope = config["scope"]
    if not rows:
        raise ValueError("Input manifest is empty.")
    if len({row["candidate_id"] for row in rows}) != len(rows):
        raise ValueError("Duplicate candidate IDs in reviewed mask manifest.")
    for row in rows:
        if row.get("pilot_subset") != args.subset_name:
            raise ValueError(f"Subset mismatch: {row['candidate_id']}")
        if row.get("mask_qc_status") != scope["required_mask_qc_status"]:
            raise ValueError(f"Mask QC not passed: {row['candidate_id']}")
        if row.get("edit_qc_status") != "not_tested" or row.get("object_identity_preservation_status") != "not_tested":
            raise ValueError(f"Post-edit state already populated: {row['candidate_id']}")
        if row.get("development_exposure", True) is not True:
            raise ValueError(f"Non-development candidate: {row['candidate_id']}")
        colors = {row["object_a"]["original_color"], row["object_b"]["original_color"]}
        if not colors.issubset(set(scope["allowed_colors"])) or len(colors) != 2:
            raise ValueError(f"Invalid color pair: {row['candidate_id']}")

    layout = config["output_layout"]
    top_files = {
        "metadata": args.output_dir / layout["run_metadata"],
        "manifest": args.output_dir / f"color_edit_results_{args.subset_name}_v1.1.jsonl",
        "qc": args.output_dir / f"natural_edit_qc_review_template_{args.subset_name}_v1.1.jsonl",
        "summary": args.output_dir / f"color_edit_run_summary_{args.subset_name}_v1.1.json",
    }
    if args.output_dir.exists() and any(args.output_dir.iterdir()) and not args.resume:
        raise FileExistsError(f"Output is nonempty; use verified --resume: {args.output_dir}")
    args.output_dir.mkdir(parents=True, exist_ok=True)
    for key in ("original_cutouts_dir", "edited_cutouts_dir", "natural_edited_dir", "selected_masks_dir", "contact_sheets_dir", "records_dir"):
        (args.output_dir / layout[key]).mkdir(parents=True, exist_ok=True)

    config_sha = sha256(args.config)
    input_sha = sha256(args.input_manifest)
    seed = int(config["seed"])
    random.seed(seed)
    np.random.seed(seed)
    metadata = {
        "schema_version": "1.1",
        "run_id": f"attribute_binding_color_swap_{args.subset_name}_v1.1",
        "created_at": datetime.now(timezone.utc).isoformat(),
        "seed": seed,
        "input_manifest": args.input_manifest.as_posix(),
        "input_manifest_sha256": input_sha,
        "config": args.config.as_posix(),
        "config_sha256": config_sha,
        "candidate_ids": [row["candidate_id"] for row in rows],
        "subset_name": args.subset_name,
        "method": config["method"],
    }
    if top_files["metadata"].exists():
        existing = read_json(top_files["metadata"])
        for field in ("seed", "input_manifest_sha256", "config_sha256", "candidate_ids"):
            if existing[field] != metadata[field]:
                raise ValueError(f"Resume metadata mismatch: {field}")
        metadata = existing
    else:
        atomic_json(top_files["metadata"], metadata)

    method = config["method"]
    centers = {key: float(value) for key, value in method["color_centers_degrees"].items()}
    results: list[dict[str, Any]] = []
    generated = resumed = failed = 0
    for row in rows:
        cid = row["candidate_id"]
        candidate_paths = paths_for(args.output_dir, layout, cid)
        source_path = resolve(args.project_root, row["source_image"]["path"])
        mask_a_path = args.segmentation_root / row["assets"]["mask_a"]["path"]
        mask_b_path = args.segmentation_root / row["assets"]["mask_b"]["path"]
        expected = {
            "candidate_id": cid,
            "source_image_sha256": sha256(source_path),
            "mask_a_sha256": sha256(mask_a_path),
            "mask_b_sha256": sha256(mask_b_path),
            "color_swap_config_sha256": config_sha,
        }
        if expected["source_image_sha256"] != row["source_image"]["sha256"]:
            raise ValueError(f"Source checksum mismatch: {cid}")
        if expected["mask_a_sha256"] != row["assets"]["mask_a"]["sha256"] or expected["mask_b_sha256"] != row["assets"]["mask_b"]["sha256"]:
            raise ValueError(f"Mask checksum mismatch: {cid}")
        if candidate_paths["record"].exists():
            if not args.resume:
                raise FileExistsError(candidate_paths["record"])
            record = read_json(candidate_paths["record"])
            verify_resume(record, expected, args.output_dir)
            results.append(record)
            resumed += 1
            continue
        unexpected = [path for name, path in candidate_paths.items() if name != "record" and path.exists()]
        if unexpected:
            raise FileExistsError(f"Partial assets without record for {cid}: {unexpected}")

        base = {
            "schema_version": "1.1",
            "run_id": metadata["run_id"],
            "subset_name": args.subset_name,
            "pilot_subset": args.subset_name,
            "candidate_id": cid,
            "source_image_id": row["source_image_id"],
            "object_a_label": row["object_a"]["label"],
            "object_b_label": row["object_b"]["label"],
            "original_color_a": row["object_a"]["original_color"],
            "original_color_b": row["object_b"]["original_color"],
            "target_color_a": row["object_b"]["original_color"],
            "target_color_b": row["object_a"]["original_color"],
            "original_image_path": row["source_image"]["path"],
            "object_a_mask_path": row["assets"]["mask_a"]["path"],
            "object_b_mask_path": row["assets"]["mask_b"]["path"],
            "color_swap_method": method["name"],
            "color_swap_config_sha256": config_sha,
            "source_image_sha256": expected["source_image_sha256"],
            "mask_a_sha256": expected["mask_a_sha256"],
            "mask_b_sha256": expected["mask_b_sha256"],
            "mask_qc_status": "pass",
            "edit_qc_status": "not_tested",
            "object_identity_preservation_status": "not_tested",
            "failure_reasons": [],
            "development_exposure": bool(row.get("development_exposure", True)),
        }
        for field in (
            "selection_seed",
            "semantic_status",
            "technical_status",
            "technical_status_before",
            "technical_status_after_mask_qc",
        ):
            if field in row:
                base[field] = row[field]
        try:
            with Image.open(source_path) as opened:
                original_image = opened.convert("RGB")
            original_rgb = np.asarray(original_image)
            hsv = np.asarray(original_image.convert("HSV"))
            mask_a = load_binary_mask(mask_a_path, original_image.size)
            mask_b = load_binary_mask(mask_b_path, original_image.size)
            overlap = int((mask_a & mask_b).sum())
            if overlap > config["automatic_diagnostics"]["maximum_ab_mask_overlap_pixels"]:
                raise ValueError(f"A/B masks overlap by {overlap} pixels")
            color_a, color_b = base["original_color_a"], base["original_color_b"]
            selected_a = select_source_pixels(hsv, mask_a, color_a, centers, float(method["source_hue_tolerance_degrees"]), int(method["min_saturation_0_255"]), int(method["min_value_0_255"]))
            selected_b = select_source_pixels(hsv, mask_b, color_b, centers, float(method["source_hue_tolerance_degrees"]), int(method["min_saturation_0_255"]), int(method["min_value_0_255"]))
            if not selected_a.any() or not selected_b.any():
                raise ValueError("Insufficient source-color pixels for one or both objects")
            hue = hsv[..., 0].astype(np.float32) * (360.0 / 255.0)
            source_hue_a = circular_mean(hue[selected_a], hsv[..., 1][selected_a].astype(np.float32))
            source_hue_b = circular_mean(hue[selected_b], hsv[..., 1][selected_b].astype(np.float32))
            edited_rgb_a = shifted_rgb(original_rgb, hsv, selected_a, source_hue_a, source_hue_b)
            edited_rgb_b = shifted_rgb(original_rgb, hsv, selected_b, source_hue_b, source_hue_a)
            original_cutout_a = rgba_cutout(original_rgb, mask_a)
            original_cutout_b = rgba_cutout(original_rgb, mask_b)
            edited_cutout_a = rgba_cutout(edited_rgb_a, mask_a)
            edited_cutout_b = rgba_cutout(edited_rgb_b, mask_b)
            natural_rgb = original_rgb.copy()
            natural_rgb[mask_a] = edited_rgb_a[mask_a]
            natural_rgb[mask_b] = edited_rgb_b[mask_b]
            natural_image = Image.fromarray(natural_rgb, mode="RGB")
            sheet = contact_sheet(original_image, natural_image, original_cutout_a, edited_cutout_a, original_cutout_b, edited_cutout_b)

            atomic_image(original_cutout_a, candidate_paths["original_a"])
            atomic_image(original_cutout_b, candidate_paths["original_b"])
            atomic_image(edited_cutout_a, candidate_paths["edited_a"])
            atomic_image(edited_cutout_b, candidate_paths["edited_b"])
            atomic_image(Image.fromarray(selected_a.astype(np.uint8) * 255, mode="L"), candidate_paths["selected_a"])
            atomic_image(Image.fromarray(selected_b.astype(np.uint8) * 255, mode="L"), candidate_paths["selected_b"])
            atomic_image(natural_image, candidate_paths["natural"])
            atomic_image(sheet, candidate_paths["contact"])

            changed = np.any(natural_rgb != original_rgb, axis=2)
            union = mask_a | mask_b
            outside_changed = int((changed & ~union).sum())
            metrics_a = hue_metrics(edited_rgb_a, selected_a, source_hue_a, source_hue_b, float(method["source_hue_tolerance_degrees"]))
            metrics_b = hue_metrics(edited_rgb_b, selected_b, source_hue_b, source_hue_a, float(method["source_hue_tolerance_degrees"]))
            metrics_a.update({"selected_mask_fraction": float(selected_a.sum() / max(mask_a.sum(), 1)), "target_mask_color_change_mean_abs_rgb": mean_abs_change(original_rgb, edited_rgb_a, mask_a), "boundary_change_mean_abs_rgb": mean_abs_change(original_rgb, edited_rgb_a, boundary(mask_a)), "alpha_consistent": True, "area_consistent": True})
            metrics_b.update({"selected_mask_fraction": float(selected_b.sum() / max(mask_b.sum(), 1)), "target_mask_color_change_mean_abs_rgb": mean_abs_change(original_rgb, edited_rgb_b, mask_b), "boundary_change_mean_abs_rgb": mean_abs_change(original_rgb, edited_rgb_b, boundary(mask_b)), "alpha_consistent": True, "area_consistent": True})
            flags: list[str] = []
            thresholds = config["automatic_diagnostics"]
            for side, metrics in (("a", metrics_a), ("b", metrics_b)):
                if metrics["selected_mask_fraction"] < thresholds["minimum_selected_mask_fraction_warning"]:
                    flags.append(f"object_{side}:low_selected_fraction")
                if metrics["post_target_hue_distance_mean_degrees"] > thresholds["maximum_post_target_hue_distance_warning_degrees"]:
                    flags.append(f"object_{side}:target_hue_distance_warning")
                if metrics["original_color_residue_fraction"] > thresholds["maximum_original_color_residue_warning_fraction"]:
                    flags.append(f"object_{side}:original_color_residue_warning")
            if outside_changed > thresholds["maximum_outside_mask_changed_pixels"]:
                flags.append("outside_mask_change_warning")
            record = {
                **base,
                "generation_status": "generated",
                "generation_error": None,
                "object_a_original_cutout_path": asset(candidate_paths["original_a"], args.output_dir)["path"],
                "object_b_original_cutout_path": asset(candidate_paths["original_b"], args.output_dir)["path"],
                "object_a_edited_cutout_path": asset(candidate_paths["edited_a"], args.output_dir)["path"],
                "object_b_edited_cutout_path": asset(candidate_paths["edited_b"], args.output_dir)["path"],
                "natural_edited_path": asset(candidate_paths["natural"], args.output_dir)["path"],
                "contact_sheet_path": asset(candidate_paths["contact"], args.output_dir)["path"],
                "selected_color_mask_a_path": asset(candidate_paths["selected_a"], args.output_dir)["path"],
                "selected_color_mask_b_path": asset(candidate_paths["selected_b"], args.output_dir)["path"],
                "original_cutout_a_sha256": sha256(candidate_paths["original_a"]),
                "original_cutout_b_sha256": sha256(candidate_paths["original_b"]),
                "edited_cutout_a_sha256": sha256(candidate_paths["edited_a"]),
                "edited_cutout_b_sha256": sha256(candidate_paths["edited_b"]),
                "natural_edited_sha256": sha256(candidate_paths["natural"]),
                "contact_sheet_sha256": sha256(candidate_paths["contact"]),
                "selected_color_mask_a_sha256": sha256(candidate_paths["selected_a"]),
                "selected_color_mask_b_sha256": sha256(candidate_paths["selected_b"]),
                "automatic_qc_metrics": {"object_a": metrics_a, "object_b": metrics_b, "pair": {"source_hue_a_degrees": source_hue_a, "source_hue_b_degrees": source_hue_b, "ab_mask_overlap_pixels": overlap, "outside_mask_changed_pixels": outside_changed, "outside_mask_pixel_preservation": float(1.0 - outside_changed / max((~union).sum(), 1)), "cross_object_color_leakage_pixels": int((changed & ~(selected_a | selected_b)).sum())}},
                "diagnostic_flags": flags,
                "generation_provenance": {"seed": seed, "input_manifest_sha256": input_sha, "config_sha256": config_sha, "python": platform.python_version(), "numpy": np.__version__, "pillow": PILLOW_VERSION, "device": "cpu", "dtype": "uint8", "shared_edited_cutouts_required_for_controlled": True},
            }
            generated += 1
        except Exception as exc:
            record = {
                **base,
                "generation_status": "failed",
                "generation_error": f"{type(exc).__name__}: {exc}",
                "object_a_original_cutout_path": None,
                "object_b_original_cutout_path": None,
                "object_a_edited_cutout_path": None,
                "object_b_edited_cutout_path": None,
                "natural_edited_path": None,
                "contact_sheet_path": None,
                "edited_cutout_a_sha256": None,
                "edited_cutout_b_sha256": None,
                "natural_edited_sha256": None,
                "automatic_qc_metrics": {},
                "diagnostic_flags": ["generation_failed_requires_review"],
                "generation_provenance": {"seed": seed, "input_manifest_sha256": input_sha, "config_sha256": config_sha, "device": "cpu"},
            }
            failed += 1
        atomic_json(candidate_paths["record"], record)
        results.append(record)

    results.sort(key=lambda item: item["candidate_id"])
    qc_rows = [qc_template(record) for record in results]
    if top_files["manifest"].exists():
        if not args.resume or read_jsonl(top_files["manifest"]) != results:
            raise FileExistsError(top_files["manifest"])
    else:
        atomic_jsonl(top_files["manifest"], results)
    if top_files["qc"].exists():
        if not args.resume or read_jsonl(top_files["qc"]) != qc_rows:
            raise FileExistsError(top_files["qc"])
    else:
        atomic_jsonl(top_files["qc"], qc_rows)
    summary = {
        "schema_version": "1.1",
        "run_id": metadata["run_id"],
        "status": "generation_complete_qc_not_tested",
        "candidate_count": len(results),
        "generated_count": sum(row["generation_status"] == "generated" for row in results),
        "failed_generation_count": sum(row["generation_status"] == "failed" for row in results),
        "generated_this_run": generated,
        "resumed_count": resumed,
        "mask_qc_status_counts": {"pass": len(results)},
        "edit_qc_status_counts": {"not_tested": len(results)},
        "object_identity_preservation_status_counts": {"not_tested": len(results)},
        "controlled_generation_executed": False,
        "question_manifest_generated": False,
        "vlm_inference_executed": False,
    }
    if top_files["summary"].exists():
        if not args.resume:
            raise FileExistsError(top_files["summary"])
    else:
        atomic_json(top_files["summary"], summary)
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
