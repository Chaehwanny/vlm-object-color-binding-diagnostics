#!/usr/bin/env python3
"""Generate SAM masks and visual overlays for attribute-binding candidates."""

from __future__ import annotations

import argparse
import json
import shutil
from pathlib import Path
from typing import Any

import numpy as np
import torch
from PIL import Image
from transformers import SamModel, SamProcessor


def parse_args() -> argparse.Namespace:
    project_root = Path(__file__).resolve().parents[2]
    parser = argparse.ArgumentParser(description="Generate bbox-prompted SAM masks.")
    parser.add_argument(
        "--manifest",
        type=Path,
        default=project_root
        / "processed/attribute_binding/segmentation/segmentation_candidates_v0.2"
        / "segmentation_candidates.jsonl",
    )
    parser.add_argument(
        "--image-root",
        type=Path,
        default=project_root / "raw/GQA-Scene-Graph",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=project_root
        / "processed/attribute_binding/segmentation/segmentation_masks_v0.2",
    )
    parser.add_argument("--model", default="facebook/sam-vit-base")
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    with path.open("r", encoding="utf-8") as input_file:
        return [json.loads(line) for line in input_file]


def bbox_xyxy(bbox_xywh: list[float], width: int, height: int) -> list[float]:
    x, y, w, h = bbox_xywh
    return [x * width, y * height, (x + w) * width, (y + h) * height]


def save_mask(mask: np.ndarray, path: Path) -> None:
    Image.fromarray((mask.astype(np.uint8) * 255), mode="L").save(path)


def mask_boundary(mask: np.ndarray, radius: int = 2) -> np.ndarray:
    padded = np.pad(mask, 1, mode="constant", constant_values=False)
    eroded = mask.copy()
    for y_offset in range(3):
        for x_offset in range(3):
            eroded &= padded[
                y_offset : y_offset + mask.shape[0],
                x_offset : x_offset + mask.shape[1],
            ]
    boundary = mask & ~eroded
    padded_boundary = np.pad(
        boundary, radius, mode="constant", constant_values=False
    )
    thick_boundary = np.zeros_like(mask)
    for y_offset in range(radius * 2 + 1):
        for x_offset in range(radius * 2 + 1):
            thick_boundary |= padded_boundary[
                y_offset : y_offset + mask.shape[0],
                x_offset : x_offset + mask.shape[1],
            ]
    return thick_boundary


def save_overlay(image: Image.Image, masks: list[np.ndarray], path: Path) -> None:
    base = np.asarray(image.convert("RGB"), dtype=np.float32)
    colors = np.asarray([[255, 0, 255], [0, 255, 255]], dtype=np.float32)
    overlay = base.copy()
    for mask, color in zip(masks, colors, strict=True):
        boundary = mask_boundary(mask)
        overlay[boundary] = overlay[boundary] * 0.15 + color * 0.85
    Image.fromarray(np.clip(overlay, 0, 255).astype(np.uint8)).save(
        path, quality=95
    )


def main() -> None:
    args = parse_args()
    if args.output_dir.exists() and any(args.output_dir.iterdir()) and not args.overwrite:
        raise FileExistsError(
            f"Output directory is not empty: {args.output_dir}. Use --overwrite."
        )

    rows = read_jsonl(args.manifest)
    masks_dir = args.output_dir / "masks"
    overlays_dir = args.output_dir / "overlays"
    originals_dir = args.output_dir / "originals"
    for directory in (masks_dir, overlays_dir, originals_dir):
        directory.mkdir(parents=True, exist_ok=True)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    dtype = torch.float16 if device == "cuda" else torch.float32
    processor = SamProcessor.from_pretrained(args.model)
    model = SamModel.from_pretrained(args.model, torch_dtype=dtype).to(device)
    model.eval()

    output_rows = []
    for row in rows:
        sample_id = row["segmentation_id"]
        source_path = args.image_root / row["image_path"]
        image = Image.open(source_path).convert("RGB")
        width, height = image.size
        boxes = [
            bbox_xyxy(row["object_a"]["bbox_xywh_norm"], width, height),
            bbox_xyxy(row["object_b"]["bbox_xywh_norm"], width, height),
        ]
        inputs = processor(image, input_boxes=[boxes], return_tensors="pt")
        model_inputs = {
            key: value.to(device=device, dtype=dtype)
            if value.is_floating_point()
            else value.to(device)
            for key, value in inputs.items()
        }
        with torch.inference_mode():
            outputs = model(**model_inputs, multimask_output=True)

        post_masks = processor.image_processor.post_process_masks(
            outputs.pred_masks.float().cpu(),
            inputs["original_sizes"].cpu(),
            inputs["reshaped_input_sizes"].cpu(),
        )[0]
        scores = outputs.iou_scores.float().cpu()[0]
        selected_masks = []
        selected_scores = []
        for object_index in range(2):
            best_index = int(torch.argmax(scores[object_index]).item())
            selected_masks.append(post_masks[object_index, best_index].numpy())
            selected_scores.append(float(scores[object_index, best_index].item()))

        mask_a_path = masks_dir / f"{sample_id}_object_a.png"
        mask_b_path = masks_dir / f"{sample_id}_object_b.png"
        overlay_path = overlays_dir / f"{sample_id}_overlay.jpg"
        original_path = originals_dir / f"{sample_id}{source_path.suffix.lower()}"
        save_mask(selected_masks[0], mask_a_path)
        save_mask(selected_masks[1], mask_b_path)
        save_overlay(image, selected_masks, overlay_path)
        shutil.copy2(source_path, original_path)

        enriched = {
            **row,
            "mask_model": args.model,
            "mask_prompt": "scene_graph_bbox",
            "mask_a_path": str(mask_a_path.relative_to(args.output_dir)),
            "mask_b_path": str(mask_b_path.relative_to(args.output_dir)),
            "overlay_path": str(overlay_path.relative_to(args.output_dir)),
            "original_path": str(original_path.relative_to(args.output_dir)),
            "mask_a_predicted_iou": selected_scores[0],
            "mask_b_predicted_iou": selected_scores[1],
            "mask_a_area_fraction": float(selected_masks[0].mean()),
            "mask_b_area_fraction": float(selected_masks[1].mean()),
            "human_mask_review": "pending",
        }
        output_rows.append(enriched)
        print(
            f"{sample_id}: A={selected_scores[0]:.3f}, "
            f"B={selected_scores[1]:.3f}"
        )

    with (args.output_dir / "mask_manifest.jsonl").open(
        "w", encoding="utf-8"
    ) as output_file:
        for row in output_rows:
            output_file.write(json.dumps(row, ensure_ascii=False) + "\n")

    (args.output_dir / "README.md").write_text(
        """# Attribute-Binding SAM Masks v0.2

These masks are automatic proposals generated from scene-graph bounding boxes.
They are not accepted annotations until a human reviews every overlay.

- Magenta boundary: object A
- Cyan boundary: object B
- Review both object coverage and background leakage.
- Do not perform VLM inference before mask and edit quality gates pass.
""",
        encoding="utf-8",
    )


if __name__ == "__main__":
    main()
