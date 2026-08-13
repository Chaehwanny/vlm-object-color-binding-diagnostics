#!/usr/bin/env python3
"""Run Qwen2.5-VL on the five-condition spatial-relation pilot manifest."""

from __future__ import annotations

import argparse
import json
import re
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import torch
from qwen_vl_utils import process_vision_info
from transformers import AutoProcessor, Qwen2_5_VLForConditionalGeneration


PROMPT_TEMPLATE = """Answer only \"Yes\" or \"No\".
Is the following statement true for the image?
Statement: {statement}"""


def parse_args() -> argparse.Namespace:
    project_root = Path(__file__).resolve().parents[2]
    dataset_dir = project_root / "processed/spatial_left_right/datasets/positive_control_v1"
    parser = argparse.ArgumentParser(description="Run Qwen2.5-VL on the pilot manifest.")
    parser.add_argument(
        "--model-id", default="Qwen/Qwen2.5-VL-7B-Instruct", help="Hugging Face model ID or path."
    )
    parser.add_argument("--dataset-dir", type=Path, default=dataset_dir)
    parser.add_argument("--manifest", type=Path, default=dataset_dir / "manifest.jsonl")
    parser.add_argument(
        "--output",
        type=Path,
        default=project_root / "experiments/spatial_left_right/positive_control_v1/predictions/qwen2_5_vl_7b.jsonl",
    )
    parser.add_argument("--max-new-tokens", type=int, default=3)
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def read_manifest(path: Path) -> list[dict[str, Any]]:
    with path.open("r", encoding="utf-8") as manifest_file:
        return [json.loads(line) for line in manifest_file]


def parse_answer(generated_text: str) -> str | None:
    match = re.search(r"\b(yes|no)\b", generated_text, flags=re.IGNORECASE)
    return match.group(1).capitalize() if match else None


def existing_keys(path: Path) -> set[tuple[str, str]]:
    if not path.is_file():
        return set()
    with path.open("r", encoding="utf-8") as prediction_file:
        return {
            (record["sample_id"], record["condition"])
            for line in prediction_file
            if (record := json.loads(line)).get("sample_id")
        }


def generate_answer(
    model: Qwen2_5_VLForConditionalGeneration,
    processor: AutoProcessor,
    image_path: Path,
    prompt: str,
    max_new_tokens: int,
) -> str:
    messages = [
        {
            "role": "user",
            "content": [
                {"type": "image", "image": str(image_path)},
                {"type": "text", "text": prompt},
            ],
        }
    ]
    chat_text = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    image_inputs, video_inputs = process_vision_info(messages)
    inputs = processor(
        text=[chat_text],
        images=image_inputs,
        videos=video_inputs,
        padding=True,
        return_tensors="pt",
    ).to(model.device)
    with torch.inference_mode():
        generated_ids = model.generate(
            **inputs,
            do_sample=False,
            max_new_tokens=max_new_tokens,
        )
    trimmed_ids = [
        output_ids[len(input_ids) :]
        for input_ids, output_ids in zip(inputs.input_ids, generated_ids)
    ]
    return processor.batch_decode(trimmed_ids, skip_special_tokens=True)[0].strip()


def main() -> None:
    args = parse_args()
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA GPU is required for this pilot configuration.")
    if not args.manifest.is_file():
        raise FileNotFoundError(f"Manifest not found: {args.manifest}")

    rows = read_manifest(args.manifest)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    completed = set() if args.overwrite else existing_keys(args.output)

    print(f"Loading model: {args.model_id}")
    model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
        args.model_id,
        torch_dtype=torch.bfloat16,
        device_map="auto",
    ).eval()
    processor = AutoProcessor.from_pretrained(args.model_id)

    mode = "w" if args.overwrite else "a"
    written = 0
    with args.output.open(mode, encoding="utf-8") as prediction_file:
        for position, row in enumerate(rows, start=1):
            key = (row["sample_id"], row["condition"])
            if key in completed:
                continue
            image_path = args.dataset_dir / row["image_path"]
            if not image_path.is_file():
                raise FileNotFoundError(f"Image not found: {image_path}")
            prompt = PROMPT_TEMPLATE.format(statement=row["statement"])
            generated_text = generate_answer(
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
            prediction_file.write(json.dumps(result, ensure_ascii=False) + "\n")
            prediction_file.flush()
            written += 1
            print(
                f"[{position}/{len(rows)}] {row['sample_id']} {row['condition']}: "
                f"{generated_text!r} -> {parsed_answer}",
                flush=True,
            )

    print(f"Done. Wrote {written} predictions to {args.output}")


if __name__ == "__main__":
    main()
