#!/usr/bin/env python3
"""Create paged contact sheets from a review pool's preview images."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

from PIL import Image, ImageDraw, ImageFont, ImageOps


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Create review contact sheets.")
    parser.add_argument("--review-dir", type=Path, required=True)
    parser.add_argument("--manifest-name", default="review_pool.jsonl")
    parser.add_argument("--path-field", default="preview_path")
    parser.add_argument("--id-field")
    parser.add_argument("--per-sheet", type=int, default=8)
    parser.add_argument("--columns", type=int, default=4)
    parser.add_argument("--tile-width", type=int, default=400)
    parser.add_argument("--tile-height", type=int, default=320)
    return parser.parse_args()


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    with path.open("r", encoding="utf-8") as input_file:
        return [json.loads(line) for line in input_file]


def main() -> None:
    args = parse_args()
    rows = read_jsonl(args.review_dir / args.manifest_name)
    output_dir = args.review_dir / "contact_sheets"
    output_dir.mkdir(parents=True, exist_ok=True)
    font = ImageFont.load_default()

    for page_index, start in enumerate(range(0, len(rows), args.per_sheet), start=1):
        page_rows = rows[start : start + args.per_sheet]
        row_count = (len(page_rows) + args.columns - 1) // args.columns
        canvas = Image.new(
            "RGB",
            (args.columns * args.tile_width, row_count * args.tile_height),
            "white",
        )
        draw = ImageDraw.Draw(canvas)
        for index, row in enumerate(page_rows):
            column = index % args.columns
            grid_row = index // args.columns
            left = column * args.tile_width
            top = grid_row * args.tile_height
            label = (row.get(args.id_field) if args.id_field else None) or row.get("mini_review_id") or row.get("candidate_id") or str(start + index)
            with Image.open(args.review_dir / row[args.path_field]) as opened_image:
                preview = opened_image.convert("RGB")
            fitted = ImageOps.contain(
                preview, (args.tile_width - 16, args.tile_height - 34)
            )
            x = left + (args.tile_width - fitted.width) // 2
            y = top + 26 + (args.tile_height - 26 - fitted.height) // 2
            canvas.paste(fitted, (x, y))
            draw.text((left + 8, top + 8), label, fill="black", font=font)
            draw.rectangle(
                (left, top, left + args.tile_width - 1, top + args.tile_height - 1),
                outline="#98A2B3",
                width=1,
            )
        canvas.save(output_dir / f"contact_sheet_{page_index:02d}.jpg", quality=95)

    print(f"Wrote {page_index} contact sheets to {output_dir}")


if __name__ == "__main__":
    main()
