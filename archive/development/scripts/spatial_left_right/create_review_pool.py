#!/usr/bin/env python3
"""Create a balanced, human-reviewable pool from filtered left/right candidates."""

from __future__ import annotations

import argparse
import csv
import json
import random
from collections import defaultdict
from pathlib import Path
from typing import Any

from PIL import Image, ImageDraw, ImageFont


def parse_args() -> argparse.Namespace:
    project_root = Path(__file__).resolve().parents[2]
    parser = argparse.ArgumentParser(
        description="Create annotated previews and a review sheet for pilot candidates."
    )
    parser.add_argument(
        "--input",
        type=Path,
        default=project_root
        / "processed/spatial_left_right/candidates/filtered_left_right_candidates.jsonl",
    )
    parser.add_argument(
        "--image-root",
        type=Path,
        default=project_root / "raw/GQA-Scene-Graph",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=project_root / "processed/spatial_left_right/reviews/review_pool_v1",
    )
    parser.add_argument(
        "--per-relation",
        type=int,
        default=120,
        help="Number of unique-image candidates to sample for each relation.",
    )
    parser.add_argument("--seed", type=int, default=20260716)
    return parser.parse_args()


def load_candidates(path: Path) -> dict[str, list[dict[str, Any]]]:
    candidates: dict[str, list[dict[str, Any]]] = defaultdict(list)
    with path.open("r", encoding="utf-8") as input_file:
        for line in input_file:
            record = json.loads(line)
            candidates[record["relation"]].append(record)
    return candidates


def select_candidates(
    candidates: dict[str, list[dict[str, Any]]], per_relation: int, seed: int
) -> list[dict[str, Any]]:
    rng = random.Random(seed)
    selected: list[dict[str, Any]] = []
    used_images: set[str] = set()

    for relation in ("left_of", "right_of"):
        pool = list(candidates[relation])
        rng.shuffle(pool)
        relation_selected = 0
        for record in pool:
            image_path = str(record["image_path"])
            if image_path in used_images:
                continue
            selected.append(record)
            used_images.add(image_path)
            relation_selected += 1
            if relation_selected == per_relation:
                break
        if relation_selected < per_relation:
            raise RuntimeError(
                f"Only found {relation_selected} unique-image candidates for {relation}; "
                f"needed {per_relation}."
            )
    return selected


def draw_label(
    draw: ImageDraw.ImageDraw,
    xy: tuple[int, int],
    text: str,
    fill: str,
    font: ImageFont.ImageFont,
    image_width: int,
) -> None:
    left, top = xy
    text_box = draw.textbbox((left, top), text, font=font)
    width = text_box[2] - text_box[0] + 8
    height = text_box[3] - text_box[1] + 6
    left = min(left, max(0, image_width - width))
    draw.rectangle((left, top, left + width, top + height), fill=fill)
    draw.text((left + 4, top + 3), text, fill="white", font=font)


def draw_preview(record: dict[str, Any], source_path: Path, output_path: Path) -> None:
    with Image.open(source_path) as original:
        image = original.convert("RGB")
    draw = ImageDraw.Draw(image)
    font = ImageFont.load_default()
    image_width, image_height = image.size

    boxes = (
        (record["source_bbox_xywh_norm"], "A", record["source_label"], "#d62828"),
        (record["target_bbox_xywh_norm"], "B", record["target_label"], "#0077b6"),
    )
    line_width = max(2, round(min(image_width, image_height) * 0.006))
    for bbox, marker, label, color in boxes:
        x, y, width, height = bbox
        left = round(x * image_width)
        top = round(y * image_height)
        right = round((x + width) * image_width)
        bottom = round((y + height) * image_height)
        draw.rectangle((left, top, right, bottom), outline=color, width=line_width)
        draw_label(draw, (left, max(0, top - 18)), f"{marker}: {label}", color, font, image_width)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    image.save(output_path, quality=95)


def main() -> None:
    args = parse_args()
    if not args.input.is_file():
        raise FileNotFoundError(f"Input file not found: {args.input}")

    candidates = load_candidates(args.input)
    selected = select_candidates(candidates, args.per_relation, args.seed)
    previews_dir = args.output_dir / "previews"
    args.output_dir.mkdir(parents=True, exist_ok=True)

    manifest_path = args.output_dir / "review_pool.jsonl"
    review_csv_path = args.output_dir / "review_sheet.csv"
    fieldnames = [
        "candidate_id",
        "relation",
        "statement",
        "image_path",
        "preview_path",
        "source_label",
        "target_label",
        "source_bbox_xywh_norm",
        "target_bbox_xywh_norm",
        "horizontal_center_gap",
        "iou",
        "keep",
        "relation_verified",
        "objects_identifiable",
        "flip_safe",
        "exclude_reason",
        "notes",
    ]

    with manifest_path.open("w", encoding="utf-8") as manifest_file, review_csv_path.open(
        "w", encoding="utf-8", newline=""
    ) as review_file:
        writer = csv.DictWriter(review_file, fieldnames=fieldnames)
        writer.writeheader()

        for number, record in enumerate(selected, start=1):
            candidate_id = f"review_{number:03d}"
            source_path = args.image_root / str(record["image_path"])
            preview_path = previews_dir / f"{candidate_id}.jpg"
            draw_preview(record, source_path, preview_path)

            direction = "left" if record["relation"] == "left_of" else "right"
            statement = (
                f"The {record['source_label']} is to the {direction} "
                f"of the {record['target_label']}."
            )
            manifest_record = {
                **record,
                "candidate_id": candidate_id,
                "statement": statement,
                "preview_path": str(preview_path.relative_to(args.output_dir)),
                "sampling": {
                    "seed": args.seed,
                    "per_relation": args.per_relation,
                    "unique_image_within_review_pool": True,
                },
            }
            manifest_file.write(json.dumps(manifest_record, ensure_ascii=False) + "\n")

            writer.writerow(
                {
                    "candidate_id": candidate_id,
                    "relation": record["relation"],
                    "statement": statement,
                    "image_path": record["image_path"],
                    "preview_path": str(preview_path.relative_to(args.output_dir)),
                    "source_label": record["source_label"],
                    "target_label": record["target_label"],
                    "source_bbox_xywh_norm": json.dumps(record["source_bbox_xywh_norm"]),
                    "target_bbox_xywh_norm": json.dumps(record["target_bbox_xywh_norm"]),
                    "horizontal_center_gap": record["geometry"]["horizontal_center_gap"],
                    "iou": record["geometry"]["iou"],
                    "keep": "",
                    "relation_verified": "",
                    "objects_identifiable": "",
                    "flip_safe": "",
                    "exclude_reason": "",
                    "notes": "",
                }
            )

    print(f"Created {len(selected)} review candidates in {args.output_dir}")
    print(f"Review sheet: {review_csv_path}")


if __name__ == "__main__":
    main()
