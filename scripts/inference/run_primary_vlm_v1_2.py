#!/usr/bin/env python3
"""Run one pinned VLM against Primary v1.2 or its development-only smoke fixture."""

from __future__ import annotations

import argparse
import json
import math
import subprocess
import sys
from pathlib import Path
from typing import Any

from primary_contract_v1_2 import (
    build_result_record,
    read_json,
    read_jsonl,
    render_semantic_user_content,
    sha256_file,
    validate_query_contract,
)


def parse_args() -> argparse.Namespace:
    root = Path(__file__).resolve().parents[2]
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--project-root", type=Path, default=root)
    parser.add_argument(
        "--config",
        type=Path,
        default=root / "configs/attribute_binding/primary_inference_v1.2.json",
    )
    parser.add_argument("--mode", choices=("primary", "technical-smoke"), required=True)
    parser.add_argument("--input-manifest", type=Path)
    parser.add_argument("--model-key", required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--max-items", type=int)
    return parser.parse_args()


def canonical_sequence_sha256(values: list[str]) -> str:
    import hashlib

    payload = json.dumps(values, ensure_ascii=False, separators=(",", ":"))
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def verify_primary_freeze(project_root: Path, config: dict[str, Any]) -> None:
    freeze_manifest = project_root / config["primary_freeze_manifest"]
    if sha256_file(freeze_manifest) != config["primary_freeze_manifest_sha256"]:
        raise ValueError("Primary freeze manifest SHA differs from inference config")
    command = [
        sys.executable,
        str(project_root / "scripts/data/freeze_primary_pre_inference_v1_2.py"),
        "verify",
        "--project-root",
        str(project_root),
        "--output-dir",
        str(freeze_manifest.parent),
    ]
    completed = subprocess.run(command, capture_output=True, text=True, check=False)
    if completed.returncode != 0:
        raise RuntimeError(f"Primary freeze verification failed: {completed.stdout}{completed.stderr}")


def primary_preflight(
    project_root: Path, manifest: Path, rows: list[dict[str, Any]], config: dict[str, Any]
) -> None:
    verify_primary_freeze(project_root, config)
    if manifest.resolve() != (project_root / config["primary_query_manifest"]).resolve():
        raise ValueError("Primary mode requires the frozen Primary query manifest")
    if sha256_file(manifest) != config["primary_query_manifest_sha256"]:
        raise ValueError("Primary query manifest SHA differs from inference config")
    if len(rows) != config["expected_query_count"]:
        raise ValueError("Primary query count differs from frozen inference config")
    query_ids = [row["query_id"] for row in rows]
    if canonical_sequence_sha256(query_ids) != config["primary_query_fingerprint"]:
        raise ValueError("Primary ordered query fingerprint differs")
    candidate_ids = list(dict.fromkeys(row["candidate_id"] for row in rows))
    if len(candidate_ids) != config["expected_candidate_count"]:
        raise ValueError("Primary candidate count differs from frozen inference config")
    if canonical_sequence_sha256(candidate_ids) != config["primary_candidate_fingerprint"]:
        raise ValueError("Primary ordered candidate fingerprint differs")
    for row in rows:
        validate_query_contract(row, primary=True)


def technical_smoke_preflight(
    project_root: Path, rows: list[dict[str, Any]]
) -> None:
    main_source = read_jsonl(
        project_root
        / "processed/attribute_binding/main_experiment/v1.1/preparation/main_input_manifest_v1.1.jsonl"
    )
    primary_source_ids = {str(row.get("source_image_id")) for row in main_source}
    for row in rows:
        validate_query_contract(row, primary=False)
        if row.get("analysis_role") != "technical_smoke_only":
            raise ValueError(f"{row['query_id']}: smoke fixture role is not technical_smoke_only")
        if row.get("development_exposure") is not True:
            raise ValueError(f"{row['query_id']}: smoke fixture is not development-exposed")
        if row.get("confirmatory_test_eligible") is not False:
            raise ValueError(f"{row['query_id']}: smoke fixture is marked confirmatory")
        if str(row.get("source_image_id")) in primary_source_ids:
            raise ValueError(f"{row['query_id']}: smoke source image overlaps frozen Primary")


def generation_kwargs(common: dict[str, Any]) -> dict[str, Any]:
    return {
        "do_sample": common["do_sample"],
        "num_beams": common["num_beams"],
        "max_new_tokens": common["max_new_tokens"],
    }


class NativeAutoProcessorAdapter:
    def __init__(self, model_config: dict[str, Any], common: dict[str, Any]) -> None:
        import torch
        from transformers import AutoProcessor

        self.torch = torch
        self.common = common
        model_id = model_config["model_id"]
        revision = model_config["model_revision"]
        processor_revision = model_config["processor_revision"]
        try:
            if model_config["adapter"] == "qwen3_vl":
                from transformers import Qwen3VLForConditionalGeneration as ModelClass
            elif model_config["adapter"] == "gemma3":
                from transformers import Gemma3ForConditionalGeneration as ModelClass
            else:
                raise ValueError(f"Unsupported native adapter: {model_config['adapter']}")
            self.processor = AutoProcessor.from_pretrained(
                model_id,
                revision=processor_revision,
                trust_remote_code=False,
            )
            self.model = ModelClass.from_pretrained(
                model_id,
                revision=revision,
                trust_remote_code=False,
                dtype=torch.bfloat16,
                device_map={"": 0},
                attn_implementation="sdpa",
                low_cpu_mem_usage=True,
            ).eval()
        except Exception as exc:
            if model_config["adapter"] == "gemma3":
                raise RuntimeError(
                    "Gemma loading failed. Confirm that the Gemma license is accepted and "
                    "the Hugging Face token is authenticated."
                ) from exc
            raise

    def generate(self, image_path: Path, user_content: str) -> str:
        from PIL import Image

        with Image.open(image_path) as opened:
            image = opened.convert("RGB")
        messages = [{"role": "user", "content": [
            {"type": "image", "image": image},
            {"type": "text", "text": user_content},
        ]}]
        inputs = self.processor.apply_chat_template(
            messages,
            add_generation_prompt=True,
            tokenize=True,
            return_dict=True,
            return_tensors="pt",
        ).to(self.model.device)
        input_length = inputs["input_ids"].shape[-1]
        with self.torch.inference_mode():
            outputs = self.model.generate(**inputs, **generation_kwargs(self.common))
        return self.processor.decode(
            outputs[0][input_length:],
            skip_special_tokens=True,
            clean_up_tokenization_spaces=False,
        ).strip()


class LlavaOneVision2Adapter:
    def __init__(self, model_config: dict[str, Any], common: dict[str, Any]) -> None:
        import torch
        from transformers import AutoModelForImageTextToText, AutoProcessor

        self.torch = torch
        self.common = common
        self.processor = AutoProcessor.from_pretrained(
            model_config["model_id"],
            revision=model_config["processor_revision"],
            trust_remote_code=True,
        )
        self.model = AutoModelForImageTextToText.from_pretrained(
            model_config["model_id"],
            revision=model_config["model_revision"],
            trust_remote_code=True,
            dtype=torch.bfloat16,
            device_map={"": 0},
            attn_implementation="sdpa",
            low_cpu_mem_usage=True,
        ).eval()

    def generate(self, image_path: Path, user_content: str) -> str:
        from PIL import Image

        with Image.open(image_path) as opened:
            image = opened.convert("RGB")
        messages = [{"role": "user", "content": [
            {"type": "image"},
            {"type": "text", "text": user_content},
        ]}]
        text = self.processor.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )
        inputs = self.processor(
            text=[text], images=[image], return_tensors="pt", padding=True
        ).to(self.model.device)
        input_length = inputs["input_ids"].shape[-1]
        with self.torch.inference_mode():
            outputs = self.model.generate(**inputs, **generation_kwargs(self.common))
        return self.processor.tokenizer.decode(
            outputs[0][input_length:], skip_special_tokens=True
        ).strip()


class InternVL35Adapter:
    IMAGENET_MEAN = (0.485, 0.456, 0.406)
    IMAGENET_STD = (0.229, 0.224, 0.225)

    def __init__(self, model_config: dict[str, Any], common: dict[str, Any]) -> None:
        import torch
        from transformers import AutoModel, AutoTokenizer

        self.torch = torch
        self.common = common
        self.max_tiles = model_config["max_tiles"]
        self.model = AutoModel.from_pretrained(
            model_config["model_id"],
            revision=model_config["model_revision"],
            torch_dtype=torch.bfloat16,
            low_cpu_mem_usage=True,
            use_flash_attn=False,
            trust_remote_code=True,
        ).eval().cuda()
        self.tokenizer = AutoTokenizer.from_pretrained(
            model_config["model_id"],
            revision=model_config["processor_revision"],
            trust_remote_code=True,
            use_fast=False,
        )

    @staticmethod
    def _closest_ratio(
        aspect_ratio: float, ratios: list[tuple[int, int]], width: int, height: int,
        image_size: int,
    ) -> tuple[int, int]:
        best, best_diff = (1, 1), float("inf")
        area = width * height
        for ratio in ratios:
            diff = abs(aspect_ratio - ratio[0] / ratio[1])
            if diff < best_diff or (
                diff == best_diff
                and area > 0.5 * image_size * image_size * ratio[0] * ratio[1]
            ):
                best, best_diff = ratio, diff
        return best

    def _load_image(self, image_path: Path) -> Any:
        import torch
        import torchvision.transforms as transforms
        from PIL import Image
        from torchvision.transforms.functional import InterpolationMode

        image_size = 448
        ratios = sorted(
            {
                (columns, rows)
                for blocks in range(1, self.max_tiles + 1)
                for columns in range(1, blocks + 1)
                for rows in range(1, blocks + 1)
                if 1 <= columns * rows <= self.max_tiles
            },
            key=lambda ratio: ratio[0] * ratio[1],
        )
        with Image.open(image_path) as opened:
            image = opened.convert("RGB")
        width, height = image.size
        columns, rows = self._closest_ratio(
            width / height, ratios, width, height, image_size
        )
        resized = image.resize((image_size * columns, image_size * rows))
        tiles = []
        for index in range(columns * rows):
            left = index % columns * image_size
            top = index // columns * image_size
            tiles.append(resized.crop((left, top, left + image_size, top + image_size)))
        if len(tiles) != 1:
            tiles.append(image.resize((image_size, image_size)))
        transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(mean=self.IMAGENET_MEAN, std=self.IMAGENET_STD),
        ])
        return torch.stack([transform(tile) for tile in tiles])

    def generate(self, image_path: Path, user_content: str) -> str:
        pixel_values = self._load_image(image_path).to(
            device="cuda", dtype=self.torch.bfloat16
        )
        config = generation_kwargs(self.common)
        with self.torch.inference_mode():
            response = self.model.chat(
                self.tokenizer,
                pixel_values,
                f"<image>\n{user_content}",
                config,
                history=None,
                return_history=False,
            )
        return str(response).strip()


def build_adapter(model_config: dict[str, Any], common: dict[str, Any]) -> Any:
    adapter = model_config["adapter"]
    if adapter in {"qwen3_vl", "gemma3"}:
        return NativeAutoProcessorAdapter(model_config, common)
    if adapter == "internvl3_5":
        return InternVL35Adapter(model_config, common)
    if adapter == "llava_onevision2":
        return LlavaOneVision2Adapter(model_config, common)
    raise ValueError(f"Unknown adapter: {adapter}")


def read_existing(path: Path, model_config: dict[str, Any]) -> set[str]:
    if not path.is_file():
        return set()
    rows = read_jsonl(path)
    seen: set[str] = set()
    for row in rows:
        if row.get("model_id") != model_config["model_id"]:
            raise ValueError("Existing output model ID differs")
        if row.get("model_revision") != model_config["model_revision"]:
            raise ValueError("Existing output model revision differs")
        query_id = row.get("query_id")
        if not isinstance(query_id, str) or query_id in seen:
            raise ValueError("Existing output has missing or duplicate query IDs")
        seen.add(query_id)
    return seen


def main() -> None:
    args = parse_args()
    project_root = args.project_root.resolve()
    config_path = args.config.resolve()
    config = read_json(config_path)
    if args.model_key not in config["models"]:
        raise ValueError(f"Unknown model key: {args.model_key}")
    model_config = config["models"][args.model_key]
    common = config["common"]
    if args.input_manifest is None:
        if args.mode != "primary":
            raise ValueError("technical-smoke mode requires --input-manifest")
        manifest = project_root / config["primary_query_manifest"]
    else:
        manifest = args.input_manifest.resolve()
    rows = read_jsonl(manifest)
    if args.mode == "primary":
        primary_preflight(project_root, manifest, rows, config)
    else:
        technical_smoke_preflight(project_root, rows)
    for row in rows:
        image_path = project_root / row["image_path"]
        if not image_path.is_file() or sha256_file(image_path) != row["image_sha256"]:
            raise ValueError(f"{row['query_id']}: image missing or SHA mismatch")

    if args.output.exists() and not args.resume:
        raise FileExistsError("Output exists; use --resume after validating existing records")
    completed = read_existing(args.output, model_config) if args.resume else set()
    pending = [row for row in rows if row["query_id"] not in completed]
    if args.max_items is not None:
        if args.max_items <= 0:
            raise ValueError("--max-items must be positive")
        pending = pending[: args.max_items]
    adapter = build_adapter(model_config, common)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    config_sha = sha256_file(config_path)
    mode = "a" if args.resume else "x"
    with args.output.open(mode, encoding="utf-8") as handle:
        for index, row in enumerate(pending, start=1):
            user_content = render_semantic_user_content(
                row, common["response_instruction"]
            )
            raw_text: str | None = None
            error: str | None = None
            try:
                raw_text = adapter.generate(project_root / row["image_path"], user_content)
            except Exception as exc:
                error = f"{type(exc).__name__}: {exc}"
            result = build_result_record(
                row, model_config, common, config_sha, user_content, raw_text, error
            )
            handle.write(json.dumps(result, ensure_ascii=False) + "\n")
            handle.flush()
            print(f"[{index}/{len(pending)}] {row['query_id']} valid={result['is_valid']}")


if __name__ == "__main__":
    main()
