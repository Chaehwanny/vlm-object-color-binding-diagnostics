#!/usr/bin/env python3
"""Create a balanced v0.2 review pool for color-binding edit feasibility."""

from __future__ import annotations

import argparse
import csv
import json
import random
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

from PIL import Image, ImageDraw, ImageFont


DISPLAY_COLORS = {
    "red": "#D62828",
    "blue": "#0077B6",
    "green": "#2A9D42",
    "yellow": "#C69200",
}
TIER_ORDER = {"both_preferred": 0, "one_preferred": 1, "manual_review": 2}


def parse_args() -> argparse.Namespace:
    project_root = Path(__file__).resolve().parents[2]
    parser = argparse.ArgumentParser(
        description="Create attribute-binding v0.2 review pool."
    )
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
        "--output-dir",
        type=Path,
        default=project_root / "processed/attribute_binding/reviews/review_pool_v0.2",
    )
    parser.add_argument("--per-color-pair", type=int, default=10)
    parser.add_argument("--seed", type=int, default=20260721)
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def load_grouped(path: Path) -> dict[str, list[dict[str, Any]]]:
    grouped: dict[str, list[dict[str, Any]]] = defaultdict(list)
    with path.open("r", encoding="utf-8") as input_file:
        for line in input_file:
            row = json.loads(line)
            grouped[row["color_pair"]].append(row)
    return grouped


def canonical_label_pair(row: dict[str, Any]) -> tuple[str, str]:
    return tuple(sorted((row["object_a"]["label"], row["object_b"]["label"])))


def select_balanced(
    grouped: dict[str, list[dict[str, Any]]], per_color_pair: int, seed: int
) -> list[dict[str, Any]]:
    rng = random.Random(seed)
    selected: list[dict[str, Any]] = []
    used_images: set[str] = set()

    for color_pair in sorted(grouped):
        pool = list(grouped[color_pair])
        rng.shuffle(pool)
        pool.sort(key=lambda row: TIER_ORDER[row["editability_priority"]])
        pair_selected: list[dict[str, Any]] = []
        used_label_pairs: set[tuple[str, str]] = set()

        for require_new_label_pair in (True, False):
            for row in pool:
                if len(pair_selected) == per_color_pair:
                    break
                if row["image_path"] in used_images:
                    continue
                label_pair = canonical_label_pair(row)
                if require_new_label_pair and label_pair in used_label_pairs:
                    continue
                pair_selected.append(row)
                used_images.add(row["image_path"])
                used_label_pairs.add(label_pair)
            if len(pair_selected) == per_color_pair:
                break

        if len(pair_selected) < per_color_pair:
            raise RuntimeError(
                f"Only selected {len(pair_selected)}/{per_color_pair} for {color_pair}"
            )
        selected.extend(pair_selected)
    return selected


def draw_tag(
    draw: ImageDraw.ImageDraw,
    x: int,
    y: int,
    text: str,
    background: str,
    image_width: int,
    font: ImageFont.ImageFont,
) -> None:
    bounds = draw.textbbox((0, 0), text, font=font)
    width = bounds[2] - bounds[0] + 10
    height = bounds[3] - bounds[1] + 8
    x = min(max(0, x), max(0, image_width - width))
    y = max(0, y)
    draw.rectangle((x, y, x + width, y + height), fill=background)
    draw.text((x + 5, y + 4), text, fill="white", font=font)


def create_preview(row: dict[str, Any], source: Path, output: Path) -> None:
    with Image.open(source) as opened_image:
        image = opened_image.convert("RGB")
    font = ImageFont.load_default()
    header_height = 34
    canvas = Image.new("RGB", (image.width, image.height + header_height), "#F2F4F7")
    canvas.paste(image, (0, header_height))
    draw = ImageDraw.Draw(canvas)
    header = (
        f"A: {row['object_a']['color']} {row['object_a']['label']}  |  "
        f"B: {row['object_b']['color']} {row['object_b']['label']}  |  "
        f"tier: {row['editability_priority']}"
    )
    draw.text((8, 11), header, fill="#101828", font=font)
    line_width = max(2, round(min(image.size) * 0.006))

    for marker, object_row in (("A", row["object_a"]), ("B", row["object_b"])):
        x, y, width, height = object_row["bbox_xywh_norm"]
        left = round(x * image.width)
        top = round(y * image.height) + header_height
        right = round((x + width) * image.width)
        bottom = round((y + height) * image.height) + header_height
        color = DISPLAY_COLORS[object_row["color"]]
        draw.rectangle((left, top, right, bottom), outline=color, width=line_width)
        draw_tag(
            draw,
            left,
            max(header_height, top - 20),
            f"{marker}: {object_row['color']} {object_row['label']}",
            color,
            image.width,
            font,
        )

    output.parent.mkdir(parents=True, exist_ok=True)
    canvas.save(output, quality=95)


def write_readme(path: Path) -> None:
    path.write_text(
        """# Attribute Binding Review Pool v0.2

This pool is for human selection before segmentation or VLM inference.

A candidate may advance only when:

- A and B are unambiguously identifiable as separate objects.
- The annotated colors are dominant and human-nameable.
- Both objects are opaque enough for a controlled color edit.
- Each object can plausibly be segmented without background spill.
- Swapping the two colors would preserve both object identities.
- The two swapped colors remain plausible enough not to create an obvious
  semantic-prior or OOD cue.
- No text, logo, reflection, severe occlusion, or physical entanglement makes
  the edit diagnostic for the wrong reason.

Use yes/no/uncertain for review fields. Do not run a VLM on these candidates.
The next gate requires five human-approved candidates and valid segmentation
masks.
""",
        encoding="utf-8",
    )


def main() -> None:
    args = parse_args()
    if args.output_dir.exists() and any(args.output_dir.iterdir()) and not args.overwrite:
        raise FileExistsError(
            f"Output directory is not empty: {args.output_dir}. Use --overwrite."
        )

    selected = select_balanced(
        load_grouped(args.input), args.per_color_pair, args.seed
    )
    args.output_dir.mkdir(parents=True, exist_ok=True)
    previews_dir = args.output_dir / "previews"
    manifest_path = args.output_dir / "review_pool.jsonl"
    review_path = args.output_dir / "human_review.csv"

    fields = [
        "candidate_id",
        "color_pair",
        "editability_priority",
        "image_path",
        "preview_path",
        "object_a_label",
        "object_a_annotated_color",
        "object_a_bbox_area",
        "object_b_label",
        "object_b_annotated_color",
        "object_b_bbox_area",
        "bbox_iou",
        "object_a_identifiable_yes_no_uncertain",
        "object_b_identifiable_yes_no_uncertain",
        "object_a_dominant_color_yes_no_uncertain",
        "object_b_dominant_color_yes_no_uncertain",
        "colors_clearly_distinct_yes_no_uncertain",
        "object_a_opaque_nonreflective_yes_no_uncertain",
        "object_b_opaque_nonreflective_yes_no_uncertain",
        "objects_independently_segmentable_yes_no_uncertain",
        "objects_not_entangled_yes_no_uncertain",
        "bidirectional_color_swap_plausible_yes_no_uncertain",
        "identity_preserved_if_recolored_yes_no_uncertain",
        "keep_for_mask_feasibility_yes_no",
        "exclude_reason",
        "notes",
    ]

    tier_counts: Counter[str] = Counter()
    with manifest_path.open("w", encoding="utf-8") as manifest_file, review_path.open(
        "w", encoding="utf-8", newline=""
    ) as review_file:
        writer = csv.DictWriter(review_file, fieldnames=fields)
        writer.writeheader()
        for index, row in enumerate(selected, start=1):
            candidate_id = f"binding_v02_{index:03d}"
            preview_path = previews_dir / f"{candidate_id}.jpg"
            create_preview(row, args.image_root / row["image_path"], preview_path)
            enriched = {
                **row,
                "candidate_id": candidate_id,
                "preview_path": str(preview_path.relative_to(args.output_dir)),
                "sampling": {
                    "seed": args.seed,
                    "per_color_pair": args.per_color_pair,
                    "unique_image_within_pool": True,
                    "label_pair_diversity_preferred": True,
                    "editability_tier_order": [
                        "both_preferred",
                        "one_preferred",
                        "manual_review",
                    ],
                },
            }
            manifest_file.write(json.dumps(enriched, ensure_ascii=False) + "\n")
            tier_counts[row["editability_priority"]] += 1
            writer.writerow(
                {
                    "candidate_id": candidate_id,
                    "color_pair": row["color_pair"],
                    "editability_priority": row["editability_priority"],
                    "image_path": row["image_path"],
                    "preview_path": enriched["preview_path"],
                    "object_a_label": row["object_a"]["label"],
                    "object_a_annotated_color": row["object_a"]["color"],
                    "object_a_bbox_area": row["object_a"]["bbox_area"],
                    "object_b_label": row["object_b"]["label"],
                    "object_b_annotated_color": row["object_b"]["color"],
                    "object_b_bbox_area": row["object_b"]["bbox_area"],
                    "bbox_iou": row["bbox_iou"],
                }
            )

    write_readme(args.output_dir / "README.md")
    summary = {
        "schema_version": "0.2",
        "selected": len(selected),
        "per_color_pair": args.per_color_pair,
        "seed": args.seed,
        "selected_by_color_pair": dict(
            sorted(Counter(row["color_pair"] for row in selected).items())
        ),
        "selected_by_editability_priority": dict(sorted(tier_counts.items())),
        "unique_images": len({row["image_path"] for row in selected}),
        "unique_label_pairs": len({canonical_label_pair(row) for row in selected}),
    }
    (args.output_dir / "review_pool_summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
