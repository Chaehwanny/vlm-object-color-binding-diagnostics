#!/usr/bin/env python3
"""Build enhanced human semantic-purity review sheets from existing masks."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any

import numpy as np
from PIL import Image, ImageDraw, ImageFont, ImageOps

NEUTRAL = (224, 224, 224)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--mask-results", type=Path, required=True)
    p.add_argument("--segmentation-root", type=Path, required=True)
    p.add_argument("--project-root", type=Path, required=True)
    p.add_argument("--subset-name", required=True)
    p.add_argument("--output-dir", type=Path, required=True)
    return p.parse_args()


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    with path.open(encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def resolve(root: Path, value: str) -> Path:
    path = Path(value)
    return path if path.is_absolute() else root / path


def load_mask(path: Path) -> np.ndarray:
    with Image.open(path) as image:
        return np.asarray(image.convert("L")) > 0


def boundary(mask: np.ndarray) -> np.ndarray:
    padded = np.pad(mask, 1, constant_values=False)
    eroded = mask.copy()
    for y in range(3):
        for x in range(3):
            eroded &= padded[y:y + mask.shape[0], x:x + mask.shape[1]]
    return mask & ~eroded


def overlay(original: Image.Image, mask: np.ndarray, color: tuple[int, int, int]) -> Image.Image:
    rgb = np.asarray(original.convert("RGB")).copy()
    tint = np.asarray(color, dtype=np.float32)
    rgb[mask] = np.rint(rgb[mask] * 0.45 + tint * 0.55).astype(np.uint8)
    rgb[boundary(mask)] = np.asarray(color, dtype=np.uint8)
    return Image.fromarray(rgb)


def neutral_cutout(original: Image.Image, mask: np.ndarray) -> Image.Image:
    canvas = Image.new("RGB", original.size, NEUTRAL)
    canvas.paste(original.convert("RGB"), mask=Image.fromarray(mask.astype(np.uint8) * 255))
    return canvas


def bbox_crop(original: Image.Image, mask: np.ndarray, bbox: list[float], color: tuple[int, int, int]) -> Image.Image:
    x0, y0, x1, y1 = [int(round(value)) for value in bbox]
    margin = max(8, int(0.12 * max(x1 - x0, y1 - y0)))
    x0, y0 = max(0, x0 - margin), max(0, y0 - margin)
    x1, y1 = min(original.width, x1 + margin), min(original.height, y1 + margin)
    rgb = np.asarray(original.convert("RGB")).copy()
    rgb[boundary(mask)] = np.asarray(color, dtype=np.uint8)
    crop = Image.fromarray(rgb).crop((x0, y0, x1, y1))
    draw = ImageDraw.Draw(crop)
    bx0 = int(round(bbox[0])) - x0
    by0 = int(round(bbox[1])) - y0
    bx1 = int(round(bbox[2])) - x0
    by1 = int(round(bbox[3])) - y0
    draw.rectangle((bx0, by0, bx1, by1), outline=color, width=3)
    return crop


def sheet(panels: list[tuple[str, Image.Image]], heading: str) -> Image.Image:
    tile_w, tile_h, label_h, columns = 520, 360, 30, 2
    rows = (len(panels) + columns - 1) // columns
    canvas = Image.new("RGB", (tile_w * columns, 52 + rows * (tile_h + label_h)), "white")
    draw = ImageDraw.Draw(canvas)
    draw.text((12, 14), heading, fill="black", font=ImageFont.load_default())
    for index, (label, image) in enumerate(panels):
        col, row = index % columns, index // columns
        y0 = 52 + row * (tile_h + label_h)
        draw.text((col * tile_w + 10, y0 + 7), label, fill="black", font=ImageFont.load_default())
        fitted = ImageOps.contain(image.convert("RGB"), (tile_w - 16, tile_h - 8))
        x = col * tile_w + (tile_w - fitted.width) // 2
        y = y0 + label_h + (tile_h - fitted.height) // 2
        canvas.paste(fitted, (x, y))
    return canvas


def atomic_image(path: Path, image: Image.Image) -> None:
    if path.exists():
        raise FileExistsError(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_name(f".{path.name}.tmp")
    image.save(temp, format="PNG")
    temp.replace(path)


def atomic_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    if path.exists():
        raise FileExistsError(path)
    temp = path.with_name(f".{path.name}.tmp")
    with temp.open("x", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")
    temp.replace(path)


def atomic_json(path: Path, value: dict[str, Any]) -> None:
    if path.exists():
        raise FileExistsError(path)
    temp = path.with_name(f".{path.name}.tmp")
    temp.write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    temp.replace(path)


def main() -> None:
    args = parse_args()
    source_rows = read_jsonl(args.mask_results)
    ids = [row.get("candidate_id") for row in source_rows]
    if not source_rows or len(ids) != len(set(ids)) or any(row.get("pilot_subset") != args.subset_name for row in source_rows):
        raise ValueError("Mask result manifest must contain unique rows for the requested subset")
    rows = [row for row in source_rows if row.get("mask_generation_status") == "generated"]
    failed_rows = [row for row in source_rows if row.get("mask_generation_status") == "segmentation_failed"]
    if len(rows) + len(failed_rows) != len(source_rows):
        raise ValueError("Mask result manifest contains an unknown generation status")
    args.output_dir.mkdir(parents=True, exist_ok=True)
    sheets_dir = args.output_dir / "semantic_purity_contact_sheets"
    sheets_dir.mkdir(parents=True, exist_ok=True)
    asset_rows, template_rows = [], []
    for row in rows:
        cid = row["candidate_id"]
        source_path = resolve(args.project_root, row["source_image"]["path"])
        original = Image.open(source_path).convert("RGB")
        mask_a = load_mask(resolve(args.segmentation_root, row["assets"]["mask_a"]["path"]))
        mask_b = load_mask(resolve(args.segmentation_root, row["assets"]["mask_b"]["path"]))
        label_a = f"A: {row['object_a']['label']} | {row['object_a']['original_color']}"
        label_b = f"B: {row['object_b']['label']} | {row['object_b']['original_color']}"
        panels = [
            ("Original full image", original),
            (f"A mask overlay | {label_a}", overlay(original, mask_a, (0, 210, 210))),
            (f"B mask overlay | {label_b}", overlay(original, mask_b, (235, 0, 210))),
            (f"A alpha cutout on RGB{NEUTRAL} | {label_a}", neutral_cutout(original, mask_a)),
            (f"B alpha cutout on RGB{NEUTRAL} | {label_b}", neutral_cutout(original, mask_b)),
            (f"A enlarged bbox crop + mask boundary | {label_a}", bbox_crop(original, mask_a, row["object_a"]["bbox_xyxy_pixels"], (0, 210, 210))),
            (f"B enlarged bbox crop + mask boundary | {label_b}", bbox_crop(original, mask_b, row["object_b"]["bbox_xyxy_pixels"], (235, 0, 210))),
        ]
        output = sheets_dir / f"{cid}__semantic_purity_review.png"
        atomic_image(output, sheet(panels, f"{cid} | semantic-purity review"))
        relative = output.relative_to(args.output_dir).as_posix()
        asset_rows.append({"schema_version":"1.1","subset_name":args.subset_name,"candidate_id":cid,"contact_sheet_path":relative,"contact_sheet_sha256":sha256(output),"source_image_path":row["source_image"]["path"],"source_image_sha256":row["source_image"]["sha256"],"mask_a_sha256":row["assets"]["mask_a"]["sha256"],"mask_b_sha256":row["assets"]["mask_b"]["sha256"]})
        template_rows.append({"schema_version":"1.1","subset_name":args.subset_name,"candidate_id":cid,"object_a_label":row["object_a"]["label"],"object_b_label":row["object_b"]["label"],"original_color_a":row["object_a"]["original_color"],"original_color_b":row["object_b"]["original_color"],"object_a_semantic_purity_status":"not_tested","object_b_semantic_purity_status":"not_tested","non_target_inclusion_a":False,"non_target_inclusion_b":False,"contamination_labels_a":[],"contamination_labels_b":[],"reviewer":None,"review_timestamp":None,"review_note":None,"contact_sheet_path":relative,"contact_sheet_sha256":sha256(output)})
    asset_manifest = args.output_dir / f"mask_semantic_purity_assets_{args.subset_name}_v1.1.jsonl"
    review_template = args.output_dir / f"mask_semantic_purity_review_template_{args.subset_name}_v1.1.jsonl"
    summary_path = args.output_dir / f"mask_semantic_purity_assets_summary_{args.subset_name}_v1.1.json"
    atomic_jsonl(asset_manifest, asset_rows)
    atomic_jsonl(review_template, template_rows)
    atomic_json(summary_path,{"schema_version":"1.1","subset_name":args.subset_name,"source_candidate_count":len(source_rows),"segmentation_success_count":len(rows),"segmentation_failure_count":len(failed_rows),"candidate_count":len(rows),"contact_sheet_count":len(rows),"panels_per_candidate":7,"neutral_background_rgb":list(NEUTRAL),"human_review_required":True,"automatic_pass_assignment":False,"failed_candidates_skipped":[row["candidate_id"] for row in failed_rows],"mask_results_sha256":sha256(args.mask_results)})
    print(json.dumps({"subset_name":args.subset_name,"source_candidate_count":len(source_rows),"segmentation_success_count":len(rows),"segmentation_failure_count":len(failed_rows),"candidate_count":len(rows),"contact_sheet_count":len(rows),"review_template":review_template.as_posix()},indent=2))


if __name__ == "__main__":
    main()
