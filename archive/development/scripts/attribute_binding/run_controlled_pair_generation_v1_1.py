#!/usr/bin/env python3
"""Generate deterministic Controlled Original/Edited pairs from shared cutouts."""

from __future__ import annotations

import argparse
import hashlib
import json
import platform
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
from PIL import Image, ImageDraw, ImageOps, __version__ as PILLOW_VERSION

CRITERIA = (
    "controlled_original_identity",
    "controlled_edited_identity",
    "original_color_binding",
    "edited_color_binding",
    "edited_cutout_reuse",
    "placement_consistency",
    "relative_scale_preservation",
    "overlap_and_clipping",
    "background_margin_alpha_quality",
    "four_image_prompt_answer_validity",
)


def parse_args() -> argparse.Namespace:
    root = Path(__file__).resolve().parents[2]
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-manifest", type=Path, required=True)
    parser.add_argument("--segmentation-root", type=Path, required=True)
    parser.add_argument("--color-edit-root", type=Path, required=True)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--project-root", type=Path, default=root)
    parser.add_argument("--subset-name", required=True)
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


def atomic_image(image: Image.Image, path: Path) -> None:
    if path.exists():
        raise FileExistsError(f"Refusing to overwrite: {path}")
    temp = path.with_name(f".{path.name}.tmp")
    if temp.exists():
        raise FileExistsError(f"Stale temporary file: {temp}")
    image.save(temp, format="PNG")
    temp.replace(path)


def atomic_json(path: Path, value: dict[str, Any]) -> None:
    if path.exists():
        raise FileExistsError(f"Refusing to overwrite: {path}")
    temp = path.with_name(f".{path.name}.tmp")
    if temp.exists():
        raise FileExistsError(f"Stale temporary file: {temp}")
    temp.write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    temp.replace(path)


def atomic_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    if path.exists():
        raise FileExistsError(f"Refusing to overwrite: {path}")
    temp = path.with_name(f".{path.name}.tmp")
    if temp.exists():
        raise FileExistsError(f"Stale temporary file: {temp}")
    with temp.open("x", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")
    temp.replace(path)


def alpha_bbox(image: Image.Image) -> tuple[int, int, int, int]:
    bbox = image.convert("RGBA").getchannel("A").getbbox()
    if bbox is None or bbox[2] <= bbox[0] or bbox[3] <= bbox[1]:
        raise ValueError("Empty cutout alpha channel.")
    return bbox


def crop_pair(original: Image.Image, edited: Image.Image) -> tuple[Image.Image, Image.Image, tuple[int, int, int, int]]:
    original = original.convert("RGBA")
    edited = edited.convert("RGBA")
    if original.size != edited.size:
        raise ValueError("Original and edited cutout canvas sizes differ.")
    alpha_original = np.asarray(original.getchannel("A"))
    alpha_edited = np.asarray(edited.getchannel("A"))
    if not np.array_equal(alpha_original, alpha_edited):
        raise ValueError("Original and edited cutout alpha channels differ.")
    bbox = alpha_bbox(original)
    return original.crop(bbox), edited.crop(bbox), bbox


def scaled(image: Image.Image, size: tuple[int, int]) -> Image.Image:
    if image.size == size:
        return image.copy()
    return image.resize(size, Image.Resampling.LANCZOS)


def placement_plan(
    sizes: dict[str, tuple[int, int]], centers_x: dict[str, float], config: dict[str, Any]
) -> tuple[dict[str, dict[str, Any]], float]:
    canvas = config["canvas"]
    width, height = int(canvas["width"]), int(canvas["height"])
    margin, gap = int(canvas["outer_margin_pixels"]), int(canvas["minimum_inter_object_gap_pixels"])
    available_width = width - 2 * margin - gap
    available_height = height - 2 * margin
    if available_width <= 0 or available_height <= 0:
        raise ValueError("Canvas margin/gap leaves no placement area.")
    total_source_width = sum(size[0] for size in sizes.values())
    max_source_height = max(size[1] for size in sizes.values())
    scale = min(1.0, available_width / total_source_width, available_height / max_source_height)
    if scale <= 0:
        raise ValueError("Nonpositive global scale factor.")
    dimensions = {
        side: (max(1, round(size[0] * scale)), max(1, round(size[1] * scale)))
        for side, size in sizes.items()
    }
    order = sorted(("a", "b"), key=lambda side: (centers_x[side], side))
    total_width = dimensions[order[0]][0] + gap + dimensions[order[1]][0]
    x = (width - total_width) // 2
    placements: dict[str, dict[str, Any]] = {}
    for index, side in enumerate(order):
        object_width, object_height = dimensions[side]
        y = (height - object_height) // 2
        placements[side] = {
            "x": x,
            "y": y,
            "width": object_width,
            "height": object_height,
            "natural_x_order": index,
        }
        x += object_width + gap
    for side, value in placements.items():
        if value["x"] < margin or value["y"] < margin:
            raise ValueError(f"Object {side} violates minimum outer margin.")
        if value["x"] + value["width"] > width - margin or value["y"] + value["height"] > height - margin:
            raise ValueError(f"Object {side} is clipped by the canvas.")
    return placements, float(scale)


def compose(
    cutouts: dict[str, Image.Image], placements: dict[str, dict[str, Any]],
    canvas_size: tuple[int, int], background: tuple[int, int, int]
) -> tuple[Image.Image, np.ndarray, dict[str, np.ndarray]]:
    canvas = Image.new("RGBA", canvas_size, (*background, 255))
    masks: dict[str, np.ndarray] = {}
    for side in ("a", "b"):
        p = placements[side]
        object_image = scaled(cutouts[side], (p["width"], p["height"]))
        canvas.alpha_composite(object_image, (p["x"], p["y"]))
        alpha = np.asarray(object_image.getchannel("A")) > 0
        full = np.zeros((canvas_size[1], canvas_size[0]), dtype=bool)
        full[p["y"] : p["y"] + p["height"], p["x"] : p["x"] + p["width"]] = alpha
        masks[side] = full
    return canvas.convert("RGB"), masks["a"] | masks["b"], masks


def tile(image: Image.Image, size: tuple[int, int]) -> Image.Image:
    return ImageOps.contain(image.convert("RGB"), size)


def contact_sheet(
    cid: str, row: dict[str, Any], natural_original: Image.Image, natural_edited: Image.Image,
    controlled_original: Image.Image, controlled_edited: Image.Image,
    cutouts: list[tuple[str, Image.Image]]
) -> Image.Image:
    tile_w, tile_h, label_h = 460, 330, 30
    footer_h = 190
    canvas = Image.new("RGB", (tile_w * 2, (tile_h + label_h) * 2 + footer_h), "white")
    draw = ImageDraw.Draw(canvas)
    main = (
        ("Natural Original", natural_original), ("Natural Edited", natural_edited),
        ("Controlled Original", controlled_original), ("Controlled Edited", controlled_edited),
    )
    for index, (label, image) in enumerate(main):
        x0 = (index % 2) * tile_w
        y0 = (index // 2) * (tile_h + label_h)
        shown = tile(image, (tile_w - 12, tile_h - 8))
        canvas.paste(shown, (x0 + (tile_w - shown.width) // 2, y0 + label_h + (tile_h - shown.height) // 2))
        draw.text((x0 + 8, y0 + 7), label, fill="black")
    footer_y = (tile_h + label_h) * 2
    info = (
        f"{cid} | A={row['object_a_label']} {row['original_color_a']}->{row['target_color_a']} | "
        f"B={row['object_b_label']} {row['original_color_b']}->{row['target_color_b']}"
    )
    draw.text((10, footer_y + 8), info, fill="black")
    preview_w, preview_h = 210, 140
    for index, (label, image) in enumerate(cutouts):
        shown = tile(image, (preview_w, preview_h - 20))
        x = 10 + index * 225
        y = footer_y + 38
        canvas.paste(shown, (x + (preview_w - shown.width) // 2, y + 20))
        draw.text((x, y), label, fill="black")
    return canvas


def placement_preview(
    canvas_size: tuple[int, int], masks: dict[str, np.ndarray], placements: dict[str, dict[str, Any]]
) -> Image.Image:
    array = np.full((canvas_size[1], canvas_size[0], 3), 224, dtype=np.uint8)
    array[masks["a"]] = np.array([255, 80, 200], dtype=np.uint8)
    array[masks["b"]] = np.array([50, 210, 255], dtype=np.uint8)
    image = Image.fromarray(array)
    draw = ImageDraw.Draw(image)
    for side in ("a", "b"):
        p = placements[side]
        draw.rectangle((p["x"], p["y"], p["x"] + p["width"] - 1, p["y"] + p["height"] - 1), outline="black", width=2)
        draw.text((p["x"] + 4, p["y"] + 4), side.upper(), fill="black")
    return image


def output_paths(root: Path, layout: dict[str, str], cid: str) -> dict[str, Path]:
    return {
        "controlled_original": root / layout["controlled_original_dir"] / f"{cid}__controlled_original.png",
        "controlled_edited": root / layout["controlled_edited_dir"] / f"{cid}__controlled_edited.png",
        "contact": root / layout["contact_sheets_dir"] / f"{cid}__four_image.png",
        "preview": root / layout["placement_previews_dir"] / f"{cid}__placement.png",
        "record": root / layout["records_dir"] / f"{cid}.json",
    }


def qc_template(row: dict[str, Any], subset: str) -> dict[str, Any]:
    return {
        "schema_version": "1.1", "subset_name": subset, "candidate_id": row["candidate_id"],
        "criteria": {criterion: "not_tested" for criterion in CRITERIA},
        "controlled_qc_status": "not_tested", "four_image_qc_status": "not_tested",
        "failure_reasons": [], "review_note": None, "reviewer": None, "review_timestamp": None,
    }


def main() -> None:
    args = parse_args()
    config = read_json(args.config)
    rows = read_jsonl(args.input_manifest)
    # Attrition-aware contract: the reviewed QC-eligible manifest owns the count.
    # Frozen source-subset counts are provenance, not a post-QC generation target.
    expected_count = len(rows)
    ids = [row.get("candidate_id") for row in rows]
    if not rows or len(ids) != len(set(ids)) or None in ids:
        raise ValueError("Input manifest must contain nonempty unique candidate IDs.")
    required = config["required_input_status"]
    for row in rows:
        for field, value in required.items():
            if row.get(field) != value:
                raise ValueError(f"{row['candidate_id']}: {field} must be {value}")
        for field in ("controlled_original_qc_status", "controlled_edited_qc_status", "four_image_qc_status"):
            if row.get(field, "not_tested") != "not_tested":
                raise ValueError(f"{row['candidate_id']}: {field} must be not_tested")

    layout = config["output_layout"]
    names = {
        "metadata": layout["run_metadata"],
        "manifest": layout["result_manifest_template"].format(subset=args.subset_name),
        "qc": layout["qc_template_template"].format(subset=args.subset_name),
        "summary": layout["run_summary_template"].format(subset=args.subset_name),
    }
    top = {key: args.output_dir / value for key, value in names.items()}
    if args.output_dir.exists() and any(args.output_dir.iterdir()) and not args.resume:
        raise FileExistsError(f"Output is nonempty; use verified --resume: {args.output_dir}")
    args.output_dir.mkdir(parents=True, exist_ok=True)
    for key in ("controlled_original_dir", "controlled_edited_dir", "contact_sheets_dir", "placement_previews_dir", "records_dir"):
        (args.output_dir / layout[key]).mkdir(parents=True, exist_ok=True)
    config_sha, input_sha = sha256(args.config), sha256(args.input_manifest)
    metadata = {
        "schema_version": "1.1", "run_id": f"controlled_generation_{args.subset_name}_v1.1",
        "created_at": datetime.now(timezone.utc).isoformat(), "subset_name": args.subset_name,
        "seed": config["seed"], "config_frozen": config["frozen"],
        "input_manifest": args.input_manifest.as_posix(), "input_manifest_sha256": input_sha,
        "config": args.config.as_posix(), "config_sha256": config_sha, "candidate_ids": ids,
        "python": platform.python_version(), "numpy": np.__version__, "pillow": PILLOW_VERSION,
    }
    if top["metadata"].exists():
        existing = read_json(top["metadata"])
        for field in ("subset_name", "input_manifest_sha256", "config_sha256", "candidate_ids"):
            if existing[field] != metadata[field]:
                raise ValueError(f"Resume metadata mismatch: {field}")
        metadata = existing
    else:
        atomic_json(top["metadata"], metadata)

    canvas_config = config["canvas"]
    canvas_size = (int(canvas_config["width"]), int(canvas_config["height"]))
    background = tuple(int(value) for value in canvas_config["background_rgb"])
    results: list[dict[str, Any]] = []
    generated_this_run = resumed = 0
    for source in rows:
        cid = source["candidate_id"]
        paths = output_paths(args.output_dir, layout, cid)
        input_paths = {
            "natural_original": resolve(args.project_root, source["original_image_path"]),
            "natural_edited": resolve(args.color_edit_root, source["natural_edited_path"]),
            "original_a": resolve(args.color_edit_root, source["object_a_original_cutout_path"]),
            "original_b": resolve(args.color_edit_root, source["object_b_original_cutout_path"]),
            "edited_a": resolve(args.color_edit_root, source["object_a_edited_cutout_path"]),
            "edited_b": resolve(args.color_edit_root, source["object_b_edited_cutout_path"]),
            "mask_a": resolve(args.segmentation_root, source["object_a_mask_path"]),
            "mask_b": resolve(args.segmentation_root, source["object_b_mask_path"]),
        }
        checksum_map = {
            "natural_original_sha256": source["source_image_sha256"],
            "natural_edited_sha256": source["natural_edited_sha256"],
            "original_cutout_a_sha256": source["original_cutout_a_sha256"],
            "original_cutout_b_sha256": source["original_cutout_b_sha256"],
            "edited_cutout_a_sha256": source["edited_cutout_a_sha256"],
            "edited_cutout_b_sha256": source["edited_cutout_b_sha256"],
            "mask_a_sha256": source["mask_a_sha256"], "mask_b_sha256": source["mask_b_sha256"],
        }
        for key, path in input_paths.items():
            if not path.is_file():
                raise FileNotFoundError(path)
            expected_key = {"natural_original": "natural_original_sha256", "natural_edited": "natural_edited_sha256", "original_a": "original_cutout_a_sha256", "original_b": "original_cutout_b_sha256", "edited_a": "edited_cutout_a_sha256", "edited_b": "edited_cutout_b_sha256", "mask_a": "mask_a_sha256", "mask_b": "mask_b_sha256"}[key]
            if sha256(path) != checksum_map[expected_key]:
                raise ValueError(f"{cid}: input checksum mismatch: {key}")
        if paths["record"].exists():
            if not args.resume:
                raise FileExistsError(paths["record"])
            record = read_json(paths["record"])
            if record.get("config_sha256") != config_sha or record.get("input_manifest_sha256") != input_sha:
                raise ValueError(f"{cid}: resume provenance mismatch")
            if record.get("controlled_generation_status") == "failed":
                results.append(record)
                resumed += 1
                continue
            for path_field, checksum_key in (("controlled_original_path", "controlled_original_sha256"), ("controlled_edited_path", "controlled_edited_sha256"), ("four_image_contact_sheet_path", "four_image_contact_sheet_sha256"), ("placement_preview_path", "placement_preview_sha256")):
                path = args.output_dir / record[path_field]
                if not path.is_file() or sha256(path) != record["asset_checksums"][checksum_key]:
                    raise ValueError(f"{cid}: resume output mismatch: {path}")
            results.append(record)
            resumed += 1
            continue
        partial = [path for name, path in paths.items() if name != "record" and path.exists()]
        if partial:
            raise FileExistsError(f"{cid}: partial outputs without record: {partial}")

        base = {
            "schema_version": "1.1", "run_id": metadata["run_id"], "subset_name": args.subset_name,
            "candidate_id": cid, "source_image_id": source["source_image_id"],
            "object_a_label": source["object_a_label"], "object_b_label": source["object_b_label"],
            "original_color_a": source["original_color_a"], "original_color_b": source["original_color_b"],
            "target_color_a": source["target_color_a"], "target_color_b": source["target_color_b"],
            "natural_original_path": source["original_image_path"], "natural_edited_path": source["natural_edited_path"],
            "object_a_original_cutout_path": source["object_a_original_cutout_path"], "object_b_original_cutout_path": source["object_b_original_cutout_path"],
            "object_a_edited_cutout_path": source["object_a_edited_cutout_path"], "object_b_edited_cutout_path": source["object_b_edited_cutout_path"],
            "canvas_width": canvas_size[0], "canvas_height": canvas_size[1], "background_rgb": list(background),
            "placement_method": config["placement"]["method"], "config_sha256": config_sha,
            "input_manifest_sha256": input_sha, "mask_qc_status": "pass", "edit_qc_status": "pass",
            "object_identity_preservation_status": "pass", "controlled_qc_status": "not_tested",
            "controlled_original_qc_status": "not_tested", "controlled_edited_qc_status": "not_tested",
            "four_image_qc_status": "not_tested", "failure_reasons": [],
            "development_exposure": bool(source.get("development_exposure", False)),
        }
        for optional in ("semantic_status", "technical_status", "technical_status_before", "technical_status_after_mask_qc"):
            if optional in source:
                base[optional] = source[optional]
        try:
            with Image.open(input_paths["natural_original"]) as image:
                natural_original = image.convert("RGB")
            with Image.open(input_paths["natural_edited"]) as image:
                natural_edited = image.convert("RGB")
            loaded = {}
            for key in ("original_a", "original_b", "edited_a", "edited_b"):
                with Image.open(input_paths[key]) as image:
                    loaded[key] = image.convert("RGBA")
            original_a, edited_a, bbox_a = crop_pair(loaded["original_a"], loaded["edited_a"])
            original_b, edited_b, bbox_b = crop_pair(loaded["original_b"], loaded["edited_b"])
            centers_x = {"a": (bbox_a[0] + bbox_a[2]) / 2, "b": (bbox_b[0] + bbox_b[2]) / 2}
            placements, scale_factor = placement_plan({"a": original_a.size, "b": original_b.size}, centers_x, config)
            placements["a"]["source_alpha_bbox"] = list(bbox_a)
            placements["b"]["source_alpha_bbox"] = list(bbox_b)
            controlled_original, original_union, original_masks = compose({"a": original_a, "b": original_b}, placements, canvas_size, background)
            controlled_edited, edited_union, edited_masks = compose({"a": edited_a, "b": edited_b}, placements, canvas_size, background)
            if not np.array_equal(original_union, edited_union):
                raise ValueError("Controlled Original/Edited alpha unions differ.")
            overlap_pixels = int((original_masks["a"] & original_masks["b"]).sum())
            if overlap_pixels:
                raise ValueError(f"Controlled objects overlap by {overlap_pixels} pixels.")
            original_array, edited_array = np.asarray(controlled_original), np.asarray(controlled_edited)
            changed = np.any(original_array != edited_array, axis=2)
            outside_changed = int((changed & ~original_union).sum())
            background_expected = np.asarray(background, dtype=np.uint8)
            background_mismatch_original = int(np.any(original_array != background_expected, axis=2)[~original_union].sum())
            background_mismatch_edited = int(np.any(edited_array != background_expected, axis=2)[~edited_union].sum())
            sheet = contact_sheet(cid, source, natural_original, natural_edited, controlled_original, controlled_edited, [("A original", original_a), ("A edited", edited_a), ("B original", original_b), ("B edited", edited_b)])
            preview = placement_preview(canvas_size, original_masks, placements)
            atomic_image(controlled_original, paths["controlled_original"])
            atomic_image(controlled_edited, paths["controlled_edited"])
            atomic_image(sheet, paths["contact"])
            atomic_image(preview, paths["preview"])
            output_checksums = {
                "controlled_original_sha256": sha256(paths["controlled_original"]),
                "controlled_edited_sha256": sha256(paths["controlled_edited"]),
                "four_image_contact_sheet_sha256": sha256(paths["contact"]),
                "placement_preview_sha256": sha256(paths["preview"]),
            }
            flags: list[str] = []
            diagnostics = config["automatic_diagnostics"]
            if scale_factor < diagnostics["minimum_global_scale_warning"]:
                flags.append("global_scale_warning")
            if outside_changed > diagnostics["maximum_non_object_difference_pixels"]:
                flags.append("non_object_difference_warning")
            record = {
                **base, "controlled_generation_status": "generated", "generation_error": None,
                "controlled_original_path": paths["controlled_original"].relative_to(args.output_dir).as_posix(),
                "controlled_edited_path": paths["controlled_edited"].relative_to(args.output_dir).as_posix(),
                "four_image_contact_sheet_path": paths["contact"].relative_to(args.output_dir).as_posix(),
                "placement_preview_path": paths["preview"].relative_to(args.output_dir).as_posix(),
                "placement_a": placements["a"], "placement_b": placements["b"], "global_scale_factor": scale_factor,
                "asset_checksums": {**checksum_map, **output_checksums},
                "authoritative_edited_cutout_reuse": {"object_a_path": source["object_a_edited_cutout_path"], "object_a_sha256": source["edited_cutout_a_sha256"], "object_b_path": source["object_b_edited_cutout_path"], "object_b_sha256": source["edited_cutout_b_sha256"], "recomputed_color_edit": False},
                "automatic_qc_metrics": {"object_overlap_pixels": overlap_pixels, "non_object_difference_pixels": outside_changed, "background_mismatch_original_pixels": background_mismatch_original, "background_mismatch_edited_pixels": background_mismatch_edited, "canvas_pixel_identical_geometry": True, "relative_scale_preserved": True},
                "diagnostic_flags": flags,
                "generation_provenance": {"seed": config["seed"], "config_frozen": config["frozen"], "python": platform.python_version(), "numpy": np.__version__, "pillow": PILLOW_VERSION, "device": "cpu"},
            }
            generated_this_run += 1
        except Exception as exc:
            record = {
                **base, "controlled_generation_status": "failed", "generation_error": f"{type(exc).__name__}: {exc}",
                "controlled_original_path": None, "controlled_edited_path": None, "four_image_contact_sheet_path": None,
                "placement_preview_path": None, "placement_a": None, "placement_b": None, "global_scale_factor": 0.0,
                "asset_checksums": checksum_map, "automatic_qc_metrics": {},
                "diagnostic_flags": ["generation_failed_requires_review"],
            }
        atomic_json(paths["record"], record)
        results.append(record)

    # Preserve the authoritative eligible-manifest order for lineage and attrition.
    qc_rows = [qc_template(row, args.subset_name) for row in results]
    if top["manifest"].exists():
        if not args.resume or read_jsonl(top["manifest"]) != results:
            raise FileExistsError(top["manifest"])
    else:
        atomic_jsonl(top["manifest"], results)
    if top["qc"].exists():
        if not args.resume or read_jsonl(top["qc"]) != qc_rows:
            raise FileExistsError(top["qc"])
    else:
        atomic_jsonl(top["qc"], qc_rows)
    summary = {
        "schema_version": "1.1", "run_id": metadata["run_id"], "subset_name": args.subset_name,
        "candidate_count": len(results), "generated_count": sum(row["controlled_generation_status"] == "generated" for row in results),
        "failed_generation_count": sum(row["controlled_generation_status"] == "failed" for row in results),
        "generated_this_run": generated_this_run, "resumed_count": resumed,
        "controlled_qc_status_counts": {"not_tested": len(results)}, "four_image_qc_status_counts": {"not_tested": len(results)},
        "question_generation_executed": False, "vlm_inference_executed": False,
    }
    if top["summary"].exists():
        if not args.resume:
            raise FileExistsError(top["summary"])
    else:
        atomic_json(top["summary"], summary)
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
