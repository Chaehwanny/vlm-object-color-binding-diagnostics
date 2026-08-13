#!/usr/bin/env python3
"""Score Yes versus No next-token probabilities for Qwen2.5-VL prompts."""

from __future__ import annotations

import argparse
import json
import math
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
    dataset_dir = project_root / "processed/attribute_binding/datasets/feasibility_n5_v1"
    parser = argparse.ArgumentParser(description="Score Qwen Yes/No choices.")
    parser.add_argument("--model-id", default="Qwen/Qwen2.5-VL-7B-Instruct")
    parser.add_argument("--dataset-dir", type=Path, default=dataset_dir)
    parser.add_argument("--manifest", type=Path, default=dataset_dir / "manifest.jsonl")
    parser.add_argument("--sample-id", default="binding_feas_005")
    parser.add_argument(
        "--output",
        type=Path,
        default=project_root
        / "experiments/attribute_binding/feasibility_n5_v1/forced_choice"
        / "qwen2_5_vl_7b_binding_feas_005.jsonl",
    )
    return parser.parse_args()


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    with path.open("r", encoding="utf-8") as input_file:
        return [json.loads(line) for line in input_file]


def single_token_id(processor: AutoProcessor, text: str) -> int:
    token_ids = processor.tokenizer.encode(text, add_special_tokens=False)
    if len(token_ids) != 1:
        raise ValueError(f"Expected one token for {text!r}, got {token_ids}")
    return token_ids[0]


def score_prompt(
    model: Qwen2_5_VLForConditionalGeneration,
    processor: AutoProcessor,
    image_path: Path,
    prompt: str,
    yes_token_id: int,
    no_token_id: int,
) -> dict[str, Any]:
    messages = [
        {
            "role": "user",
            "content": [
                {"type": "image", "image": str(image_path)},
                {"type": "text", "text": prompt},
            ],
        }
    ]
    chat_text = processor.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=True
    )
    image_inputs, video_inputs = process_vision_info(messages)
    inputs = processor(
        text=[chat_text],
        images=image_inputs,
        videos=video_inputs,
        padding=True,
        return_tensors="pt",
    ).to(model.device)
    with torch.inference_mode():
        generated = model.generate(
            **inputs,
            do_sample=False,
            max_new_tokens=1,
            return_dict_in_generate=True,
            output_scores=True,
        )
    logits = generated.scores[0][0].float()
    choice_logits = torch.stack([logits[yes_token_id], logits[no_token_id]])
    choice_log_probs = torch.log_softmax(choice_logits, dim=0)
    yes_log_prob = float(choice_log_probs[0].item())
    no_log_prob = float(choice_log_probs[1].item())
    return {
        "yes_token_id": yes_token_id,
        "no_token_id": no_token_id,
        "yes_log_probability_normalized": yes_log_prob,
        "no_log_probability_normalized": no_log_prob,
        "yes_probability_normalized": math.exp(yes_log_prob),
        "no_probability_normalized": math.exp(no_log_prob),
        "yes_minus_no_logit": float((choice_logits[0] - choice_logits[1]).item()),
        "forced_choice_answer": "Yes" if choice_logits[0] >= choice_logits[1] else "No",
        "generated_first_token_id": int(generated.sequences[0, -1].item()),
        "generated_first_token": processor.tokenizer.decode(
            [int(generated.sequences[0, -1].item())]
        ),
    }


def main() -> None:
    args = parse_args()
    rows = [
        row for row in read_jsonl(args.manifest) if row["sample_id"] == args.sample_id
    ]
    if not rows:
        raise ValueError(f"No manifest rows found for {args.sample_id}")
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required")

    model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
        args.model_id, torch_dtype=torch.bfloat16, device_map="auto"
    ).eval()
    processor = AutoProcessor.from_pretrained(args.model_id)
    yes_token_id = single_token_id(processor, "Yes")
    no_token_id = single_token_id(processor, "No")

    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w", encoding="utf-8") as output_file:
        for index, row in enumerate(rows, start=1):
            prompt = PROMPT_TEMPLATE.format(statement=row["statement"])
            scored = score_prompt(
                model,
                processor,
                args.dataset_dir / row["image_path"],
                prompt,
                yes_token_id,
                no_token_id,
            )
            result = {**row, "prompt": prompt, "model_id": args.model_id, **scored}
            output_file.write(json.dumps(result, ensure_ascii=False) + "\n")
            print(
                f"[{index}/{len(rows)}] {row['condition']}: "
                f"{scored['forced_choice_answer']} "
                f"P(Yes)={scored['yes_probability_normalized']:.4f} "
                f"margin={scored['yes_minus_no_logit']:.3f}",
                flush=True,
            )


if __name__ == "__main__":
    main()
