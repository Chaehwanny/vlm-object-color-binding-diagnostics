#!/usr/bin/env python3
"""Run LLaVA-OneVision on the frozen left/right pilot."""

from __future__ import annotations

import argparse
import json
import re
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import torch
from PIL import Image
from transformers import AutoModelForMultimodalLM, AutoProcessor


PROMPT_TEMPLATE = """Answer only \"Yes\" or \"No\".
Is the following statement true for the image?
Statement: {statement}"""


def parse_args() -> argparse.Namespace:
    project_root = Path(__file__).resolve().parents[2]
    frozen_dir = project_root / "frozen/left_right_pilot_v1"
    parser = argparse.ArgumentParser(description="Run LLaVA-OneVision on frozen pilot v1.")
    parser.add_argument(
        "--model-id", default="llava-hf/llava-onevision-qwen2-7b-ov-hf"
    )
    parser.add_argument("--dataset-dir", type=Path, default=frozen_dir)
    parser.add_argument("--manifest", type=Path, default=frozen_dir / "manifest.jsonl")
    parser.add_argument(
        "--output",
        type=Path,
        default=project_root
        / "experiments/spatial_left_right/positive_control_v1/predictions/llava_onevision_qwen2_7b.jsonl",
    )
    parser.add_argument("--max-new-tokens", type=int, default=3)
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
    return {
        (row["sample_id"], row["condition"])
        for row in read_jsonl(path)
    }


def generate(
    model: AutoModelForMultimodalLM,
    processor: AutoProcessor,
    image_path: Path,
    user_prompt: str,
    max_new_tokens: int,
) -> str:
    conversation = [
        {
            "role": "user",
            "content": [
                {"type": "text", "text": user_prompt},
                {"type": "image"},
            ],
        }
    ]
    chat_prompt = processor.apply_chat_template(
        conversation, add_generation_prompt=True, tokenize=False
    )
    with Image.open(image_path) as opened_image:
        image = opened_image.convert("RGB")
    inputs = processor(images=image, text=chat_prompt, return_tensors="pt").to(model.device)
    if "pixel_values" in inputs:
        inputs["pixel_values"] = inputs["pixel_values"].to(dtype=model.dtype)
    input_length = inputs["input_ids"].shape[-1]
    with torch.inference_mode():
        output_ids = model.generate(
            **inputs,
            do_sample=False,
            max_new_tokens=max_new_tokens,
        )
    return processor.decode(
        output_ids[0][input_length:], skip_special_tokens=True
    ).strip()


def main() -> None:
    args = parse_args()
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA GPU is required for this pilot configuration.")
    if not args.manifest.is_file():
        raise FileNotFoundError(f"Manifest not found: {args.manifest}")

    rows = read_jsonl(args.manifest)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    completed = set() if args.overwrite else completed_keys(args.output)

    print(f"Loading model: {args.model_id}", flush=True)
    model = AutoModelForMultimodalLM.from_pretrained(
        args.model_id,
        torch_dtype=torch.float16,
        device_map="auto",
        low_cpu_mem_usage=True,
    ).eval()
    processor = AutoProcessor.from_pretrained(args.model_id)

    mode = "w" if args.overwrite else "a"
    written = 0
    with args.output.open(mode, encoding="utf-8") as output_file:
        for position, row in enumerate(rows, start=1):
            key = (row["sample_id"], row["condition"])
            if key in completed:
                continue
            image_path = args.dataset_dir / row["image_path"]
            prompt = PROMPT_TEMPLATE.format(statement=row["statement"])
            generated_text = generate(
                model, processor, image_path, prompt, args.max_new_tokens
            )
            parsed_answer = parse_answer(generated_text)
            result = {
                **row,
                "model_id": args.model_id,
                "prompt": prompt,
                "generation_config": {
                    "do_sample": False,
                    "max_new_tokens": args.max_new_tokens,
                },
                "generated_text": generated_text,
                "parsed_answer": parsed_answer,
                "is_valid": parsed_answer is not None,
                "is_correct": parsed_answer == row["expected_answer"],
                "timestamp_utc": datetime.now(timezone.utc).isoformat(),
            }
            output_file.write(json.dumps(result, ensure_ascii=False) + "\n")
            output_file.flush()
            written += 1
            print(
                f"[{position}/{len(rows)}] {row['sample_id']} {row['condition']}: "
                f"{generated_text!r} -> {parsed_answer}",
                flush=True,
            )

    print(f"Done. Wrote {written} predictions to {args.output}")


if __name__ == "__main__":
    main()
