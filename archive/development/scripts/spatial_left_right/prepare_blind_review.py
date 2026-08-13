#!/usr/bin/env python3
"""Prepare a randomized, model-blind re-review package for pilot v1."""

from __future__ import annotations

import argparse
import csv
import json
import random
from pathlib import Path
from typing import Any

from PIL import Image, ImageDraw, ImageFont


BOX_COLORS = {"A": "#00A676", "B": "#E67E22"}
CANVAS_BACKGROUND = "#F4F5F7"
TEXT_COLOR = "#101828"


def parse_args() -> argparse.Namespace:
    project_root = Path(__file__).resolve().parents[2]
    parser = argparse.ArgumentParser(description="Prepare blind audit package.")
    parser.add_argument(
        "--frozen-dir",
        type=Path,
        default=project_root / "frozen/left_right_pilot_v1",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=project_root
        / "experiments/spatial_left_right/positive_control_v1/blind_review_v1",
    )
    parser.add_argument(
        "--key-output",
        type=Path,
        default=project_root
        / "experiments/spatial_left_right/positive_control_v1/blind_review_v1_key.json",
    )
    parser.add_argument("--seed", type=int, default=20260721)
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def load_meta(sample_dir: Path) -> dict[str, Any]:
    return json.loads((sample_dir / "meta.json").read_text(encoding="utf-8"))


def resize_for_panel(image: Image.Image, max_width: int = 700) -> Image.Image:
    if image.width <= max_width:
        return image.copy()
    ratio = max_width / image.width
    return image.resize((max_width, round(image.height * ratio)), Image.Resampling.LANCZOS)


def draw_box(
    image: Image.Image,
    bbox: list[float],
    object_id: str,
    label: str,
) -> None:
    draw = ImageDraw.Draw(image)
    x, y, width, height = bbox
    coordinates = (
        round(x * image.width),
        round(y * image.height),
        round((x + width) * image.width),
        round((y + height) * image.height),
    )
    color = BOX_COLORS[object_id]
    line_width = max(3, round(min(image.size) / 180))
    draw.rectangle(coordinates, outline=color, width=line_width)
    text = f"{object_id}: {label}"
    text_bbox = draw.textbbox((0, 0), text)
    text_width = text_bbox[2] - text_bbox[0]
    text_height = text_bbox[3] - text_bbox[1]
    label_x = coordinates[0]
    label_y = max(0, coordinates[1] - text_height - 8)
    draw.rectangle(
        (label_x, label_y, label_x + text_width + 10, label_y + text_height + 8),
        fill=color,
    )
    draw.text((label_x + 5, label_y + 4), text, fill="white")


def render_preview(sample_dir: Path, meta: dict[str, Any], output: Path) -> None:
    panels = []
    panel_specs = (
        ("ORIGINAL", "original.jpg", "source_bbox_xywh_norm", "target_bbox_xywh_norm"),
        ("HORIZONTAL FLIP", "flipped.jpg", "flipped_source_bbox_xywh_norm", "flipped_target_bbox_xywh_norm"),
    )
    for title, filename, source_bbox_key, target_bbox_key in panel_specs:
        with Image.open(sample_dir / filename) as opened:
            panel = resize_for_panel(opened.convert("RGB"))
        draw_box(panel, meta[source_bbox_key], "A", meta["source_label"])
        draw_box(panel, meta[target_bbox_key], "B", meta["target_label"])
        panels.append((title, panel))

    margin = 24
    title_height = 42
    panel_width = max(panel.width for _, panel in panels)
    panel_height = max(panel.height for _, panel in panels)
    canvas = Image.new(
        "RGB",
        (margin * 3 + panel_width * 2, margin * 2 + title_height + panel_height),
        CANVAS_BACKGROUND,
    )
    draw = ImageDraw.Draw(canvas)
    for index, (title, panel) in enumerate(panels):
        x = margin + index * (panel_width + margin)
        draw.text((x, margin), title, fill=TEXT_COLOR)
        canvas.paste(panel, (x, margin + title_height))
    canvas.save(output, quality=95)


def write_instructions(path: Path) -> None:
    path.write_text(
        """# Blind Re-review Instructions

Review every row without consulting model predictions, correctness, FCC, or prior review IDs.

Use only the labeled objects and the original/flip preview.

- Object A identifiable: Is the green A object clearly identifiable?
- Object B identifiable: Is the orange B object clearly identifiable?
- Label specificity: Does each label specifically and correctly name its boxed object?
- Semantic overlap: Are A and B duplicate labels or in a hypernym-hyponym relation?
- Relation clarity: Is the claimed left/right relation visually unambiguous in the original?
- Flip safety: Does horizontal flipping introduce a salient artifact that could cue the manipulation?
- Inclusion recommendation: Keep only when both objects, the relation, and flip safety are adequate.

Allowed values are listed in the CSV column names or should use yes/no/uncertain where applicable.
Do not open the separately stored audit key until the sheet is finalized.
""",
        encoding="utf-8",
    )


def main() -> None:
    args = parse_args()
    if args.output_dir.exists() and any(args.output_dir.iterdir()) and not args.overwrite:
        raise FileExistsError(
            f"Output directory is not empty: {args.output_dir}. Use --overwrite to replace files."
        )
    args.output_dir.mkdir(parents=True, exist_ok=True)
    previews_dir = args.output_dir / "previews"
    previews_dir.mkdir(parents=True, exist_ok=True)

    sample_dirs = sorted((args.frozen_dir / "dataset").glob("review_*"))
    random.Random(args.seed).shuffle(sample_dirs)
    sheet_rows = []
    key_rows = []
    for index, sample_dir in enumerate(sample_dirs, start=1):
        audit_id = f"audit_{index:03d}"
        meta = load_meta(sample_dir)
        preview_name = f"{audit_id}.jpg"
        render_preview(sample_dir, meta, previews_dir / preview_name)
        sheet_rows.append(
            {
                "audit_id": audit_id,
                "preview_path": f"previews/{preview_name}",
                "object_a_label": meta["source_label"],
                "object_b_label": meta["target_label"],
                "claimed_relation": meta["original_relation"],
                "object_a_identifiable_yes_no_uncertain": "",
                "object_b_identifiable_yes_no_uncertain": "",
                "label_specificity_adequate_too_broad_incorrect_uncertain": "",
                "semantic_overlap_none_hypernym_hyponym_duplicate_uncertain": "",
                "relation_clarity_yes_no_uncertain": "",
                "flip_safety_yes_no_uncertain": "",
                "inclusion_recommendation_keep_exclude_uncertain": "",
                "notes": "",
            }
        )
        key_rows.append(
            {
                "audit_id": audit_id,
                "review_id": meta["review_candidate_id"],
                "internal_sample_id": meta["sample_id"],
            }
        )

    sheet_path = args.output_dir / "review_sheet.csv"
    with sheet_path.open("w", encoding="utf-8", newline="") as output_file:
        writer = csv.DictWriter(output_file, fieldnames=list(sheet_rows[0]))
        writer.writeheader()
        writer.writerows(sheet_rows)
    write_instructions(args.output_dir / "README.md")
    args.key_output.write_text(
        json.dumps(
            {
                "warning": "Keep sealed until blind review is finalized.",
                "seed": args.seed,
                "mapping": key_rows,
            },
            ensure_ascii=False,
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )
    print(f"Wrote {len(sheet_rows)} blind review rows to {sheet_path}")
    print(f"Wrote sealed mapping to {args.key_output}")


if __name__ == "__main__":
    main()
