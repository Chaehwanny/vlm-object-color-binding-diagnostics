#!/usr/bin/env python3
"""Run InternVL 2.5 8B on the frozen left/right pilot."""

from __future__ import annotations

import argparse
import json
import re
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import torch
import torchvision.transforms as T
from PIL import Image
from torchvision.transforms.functional import InterpolationMode
from transformers import AutoModel, AutoTokenizer


PROMPT_TEMPLATE = """Answer only "Yes" or "No".
Is the following statement true for the image?
Statement: {statement}"""
IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)


def parse_args() -> argparse.Namespace:
    project_root = Path(__file__).resolve().parents[2]
    frozen_dir = project_root / "frozen/left_right_pilot_v1"
    parser = argparse.ArgumentParser(description="Run InternVL on frozen pilot v1.")
    parser.add_argument("--model-id", default="OpenGVLab/InternVL2_5-8B")
    parser.add_argument("--dataset-dir", type=Path, default=frozen_dir)
    parser.add_argument("--manifest", type=Path, default=frozen_dir / "manifest.jsonl")
    parser.add_argument(
        "--output",
        type=Path,
        default=project_root
        / "experiments/spatial_left_right/positive_control_v1/predictions/internvl2_5_8b.jsonl",
    )
    parser.add_argument("--max-new-tokens", type=int, default=3)
    parser.add_argument("--max-tiles", type=int, default=12)
    parser.add_argument(
        "--limit",
        type=int,
        help="Run only the first N pending rows (for a smoke test).",
    )
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    with path.open("r", encoding="utf-8") as input_file:
        return [json.loads(line) for line in input_file]


def parse_answer(text: str) -> str | None:
    match = re.search(r"\b(yes|no)\b", text, flags=re.IGNORECASE)
    return match.group(1).capitalize() if match else None


def completed_keys(path: Path) -> set[tuple[str, str]]:
    if not path.is_file():
        return set()
    return {(row["sample_id"], row["condition"]) for row in read_jsonl(path)}


def build_transform(input_size: int) -> T.Compose:
    return T.Compose(
        [
            T.Lambda(lambda image: image.convert("RGB")),
            T.Resize((input_size, input_size), interpolation=InterpolationMode.BICUBIC),
            T.ToTensor(),
            T.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
        ]
    )


def find_closest_aspect_ratio(
    aspect_ratio: float,
    target_ratios: list[tuple[int, int]],
    width: int,
    height: int,
    image_size: int,
) -> tuple[int, int]:
    best_ratio_diff = float("inf")
    best_ratio = (1, 1)
    area = width * height
    for ratio in target_ratios:
        ratio_diff = abs(aspect_ratio - ratio[0] / ratio[1])
        if ratio_diff < best_ratio_diff:
            best_ratio_diff = ratio_diff
            best_ratio = ratio
        elif ratio_diff == best_ratio_diff:
            target_area = image_size * image_size * ratio[0] * ratio[1]
            if area > 0.5 * target_area:
                best_ratio = ratio
    return best_ratio


def dynamic_preprocess(
    image: Image.Image,
    min_num: int = 1,
    max_num: int = 12,
    image_size: int = 448,
    use_thumbnail: bool = True,
) -> list[Image.Image]:
    width, height = image.size
    target_ratios = sorted(
        {
            (columns, rows)
            for blocks in range(min_num, max_num + 1)
            for columns in range(1, blocks + 1)
            for rows in range(1, blocks + 1)
            if min_num <= columns * rows <= max_num
        },
        key=lambda ratio: ratio[0] * ratio[1],
    )
    columns, rows = find_closest_aspect_ratio(
        width / height, target_ratios, width, height, image_size
    )
    resized = image.resize((image_size * columns, image_size * rows))
    tiles = []
    for index in range(columns * rows):
        left = (index % columns) * image_size
        top = (index // columns) * image_size
        tiles.append(resized.crop((left, top, left + image_size, top + image_size)))
    if use_thumbnail and len(tiles) != 1:
        tiles.append(image.resize((image_size, image_size)))
    return tiles


def load_image(path: Path, max_tiles: int) -> torch.Tensor:
    transform = build_transform(448)
    with Image.open(path) as opened_image:
        image = opened_image.convert("RGB")
        tiles = dynamic_preprocess(image, max_num=max_tiles)
    return torch.stack([transform(tile) for tile in tiles])


def generate(
    model: Any,
    tokenizer: Any,
    image_path: Path,
    user_prompt: str,
    max_new_tokens: int,
    max_tiles: int,
) -> str:
    pixel_values = load_image(image_path, max_tiles).to(
        device="cuda", dtype=torch.bfloat16
    )
    generation_config = {
        "do_sample": False,
        "max_new_tokens": max_new_tokens,
    }
    with torch.inference_mode():
        response = model.chat(
            tokenizer,
            pixel_values,
            f"<image>\n{user_prompt}",
            generation_config,
        )
    return response.strip()


def main() -> None:
    args = parse_args()
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA GPU is required for this pilot configuration.")
    if not args.manifest.is_file():
        raise FileNotFoundError(f"Manifest not found: {args.manifest}")

    torch.manual_seed(0)
    rows = read_jsonl(args.manifest)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    completed = set() if args.overwrite else completed_keys(args.output)
    pending_rows = [
        row
        for row in rows
        if (row["sample_id"], row["condition"]) not in completed
    ]
    if args.limit is not None:
        pending_rows = pending_rows[: args.limit]

    print(f"Loading model: {args.model_id}", flush=True)
    model = AutoModel.from_pretrained(
        args.model_id,
        torch_dtype=torch.bfloat16,
        low_cpu_mem_usage=True,
        use_flash_attn=False,
        trust_remote_code=True,
    ).eval().cuda()
    tokenizer = AutoTokenizer.from_pretrained(
        args.model_id, trust_remote_code=True, use_fast=False
    )

    mode = "w" if args.overwrite else "a"
    with args.output.open(mode, encoding="utf-8") as output_file:
        for position, row in enumerate(pending_rows, start=1):
            image_path = args.dataset_dir / row["image_path"]
            prompt = PROMPT_TEMPLATE.format(statement=row["statement"])
            generated_text = generate(
                model,
                tokenizer,
                image_path,
                prompt,
                args.max_new_tokens,
                args.max_tiles,
            )
            parsed_answer = parse_answer(generated_text)
            result = {
                **row,
                "model_id": args.model_id,
                "prompt": prompt,
                "generation_config": {
                    "do_sample": False,
                    "max_new_tokens": args.max_new_tokens,
                    "max_tiles": args.max_tiles,
                },
                "generated_text": generated_text,
                "parsed_answer": parsed_answer,
                "is_valid": parsed_answer is not None,
                "is_correct": parsed_answer == row["expected_answer"],
                "timestamp_utc": datetime.now(timezone.utc).isoformat(),
            }
            output_file.write(json.dumps(result, ensure_ascii=False) + "\n")
            output_file.flush()
            print(
                f"[{position}/{len(pending_rows)}] {row['sample_id']} "
                f"{row['condition']}: {generated_text!r} -> {parsed_answer}",
                flush=True,
            )

    print(f"Done. Wrote {len(pending_rows)} predictions to {args.output}")


if __name__ == "__main__":
    main()
