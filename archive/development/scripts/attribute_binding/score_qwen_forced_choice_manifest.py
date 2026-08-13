#!/usr/bin/env python3
"""Score every manifest row with Qwen Yes/No next-token forced choice."""

from __future__ import annotations

import argparse
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import torch
from qwen_vl_utils import process_vision_info
from transformers import AutoProcessor, Qwen2_5_VLForConditionalGeneration

from score_qwen_forced_choice import PROMPT_TEMPLATE, score_prompt, single_token_id


def parse_args() -> argparse.Namespace:
    root = Path(__file__).resolve().parents[2]
    dataset = root / "processed/attribute_binding/datasets/interim_n8_v1"
    parser = argparse.ArgumentParser(description="Score a Qwen Yes/No manifest.")
    parser.add_argument("--model-id", default="Qwen/Qwen2.5-VL-7B-Instruct")
    parser.add_argument("--dataset-dir", type=Path, default=dataset)
    parser.add_argument("--manifest", type=Path, default=dataset / "manifest.jsonl")
    parser.add_argument(
        "--output",
        type=Path,
        default=root
        / "experiments/attribute_binding/interim_n8_v1/forced_choice"
        / "qwen2_5_vl_7b.jsonl",
    )
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    with path.open("r", encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def existing_keys(path: Path) -> set[tuple[str, str]]:
    if not path.exists():
        return set()
    return {
        (row["sample_id"], row["condition"])
        for row in read_jsonl(path)
    }


def main() -> None:
    args = parse_args()
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required")
    rows = read_jsonl(args.manifest)
    completed = set() if args.overwrite else existing_keys(args.output)
    args.output.parent.mkdir(parents=True, exist_ok=True)

    model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
        args.model_id, torch_dtype=torch.bfloat16, device_map="auto"
    ).eval()
    processor = AutoProcessor.from_pretrained(args.model_id)
    yes_token_id = single_token_id(processor, "Yes")
    no_token_id = single_token_id(processor, "No")

    mode = "w" if args.overwrite else "a"
    written = 0
    with args.output.open(mode, encoding="utf-8") as handle:
        for index, row in enumerate(rows, start=1):
            key = (row["sample_id"], row["condition"])
            if key in completed:
                continue
            prompt = PROMPT_TEMPLATE.format(statement=row["statement"])
            scored = score_prompt(
                model,
                processor,
                args.dataset_dir / row["image_path"],
                prompt,
                yes_token_id,
                no_token_id,
            )
            answer = scored["forced_choice_answer"]
            result = {
                **row,
                "prompt": prompt,
                "model_id": args.model_id,
                **scored,
                "parsed_answer": answer,
                "is_valid": True,
                "is_correct": answer == row["expected_answer"],
                "scoring_method": "normalized_yes_no_next_token_forced_choice",
                "timestamp_utc": datetime.now(timezone.utc).isoformat(),
            }
            handle.write(json.dumps(result, ensure_ascii=False) + "\n")
            handle.flush()
            written += 1
            print(
                f"[{index}/{len(rows)}] {row['sample_id']} {row['condition']}: "
                f"{answer} margin={scored['yes_minus_no_logit']:.3f}",
                flush=True,
            )
    print(f"Done. Wrote {written} predictions to {args.output}")


if __name__ == "__main__":
    main()
