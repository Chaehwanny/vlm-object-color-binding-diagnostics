#!/usr/bin/env python3
"""Shared frozen prompt, parser, and result-record contract for Primary v1.2."""

from __future__ import annotations

import hashlib
import json
import re
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


CONTRACT_VERSION = "primary_inference_v1.2_parser_amendment1"
STRICT_PARSER_CONTRACT_VERSION = "strict_option_letter_v1.2"
PARSER_CONTRACT_VERSION = "leading_option_identifier_v1.2_amendment1"
RESPONSE_INSTRUCTION = "Answer with only the option letter."
PROHIBITED_MODEL_INPUT_KEYS = ("bbox", "mask", "overlay")

_FORM_PATTERNS = (
    r"([A-D])",
    r"([A-D])\.",
    r"\(([A-D])\)",
    r"Answer:\s+([A-D])",
    r"Answer\s+([A-D])",
    r"Option:\s+([A-D])",
    r"Option\s+([A-D])",
)
_OPTION_RESPONSE = re.compile(
    r"^\s*(?:" + "|".join(_FORM_PATTERNS) + r")\s*$",
    flags=re.IGNORECASE,
)
_LEADING_OPTION_RESPONSE = re.compile(
    r"^\s*(?:(?:Answer|Option)\s*:?\s+)?"
    r"([A-D])"
    r"(?:(?:[.:]\s*\S)|(?:\s+because\b))"
    r"[\s\S]*$",
    flags=re.IGNORECASE,
)
_FORMAT_INSTRUCTION = re.compile(
    r"answer\s+with\s+only\s+the\s+option\s+letter",
    flags=re.IGNORECASE,
)


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def sha256_json(value: Any) -> str:
    payload = json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def read_json(path: Path) -> dict[str, Any]:
    with path.open(encoding="utf-8") as handle:
        value = json.load(handle)
    if not isinstance(value, dict):
        raise ValueError(f"Expected JSON object: {path}")
    return value


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with path.open(encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            value = json.loads(line)
            if not isinstance(value, dict):
                raise ValueError(f"Expected JSON object at {path}:{line_number}")
            rows.append(value)
    return rows


def option_map(row: dict[str, Any]) -> dict[str, dict[str, Any]]:
    options = row.get("options")
    if not isinstance(options, list) or not options:
        raise ValueError(f"{row.get('query_id')}: options must be a nonempty list")
    mapped: dict[str, dict[str, Any]] = {}
    for option in options:
        if not isinstance(option, dict):
            raise ValueError(f"{row.get('query_id')}: malformed option")
        option_id = option.get("option_id")
        if option_id not in {"A", "B", "C", "D"} or option_id in mapped:
            raise ValueError(f"{row.get('query_id')}: invalid or duplicate option ID")
        if not isinstance(option.get("display_text"), str) or not option["display_text"]:
            raise ValueError(f"{row.get('query_id')}: empty option display text")
        mapped[option_id] = option
    return mapped


def validate_query_contract(row: dict[str, Any], *, primary: bool) -> None:
    required = {
        "query_id", "candidate_id", "image_path", "image_sha256", "state",
        "task_type", "prompt_text", "options", "correct_option_id",
        "correct_semantic_answer", "response_alphabet", "stateless",
        "conversation_history",
    }
    missing = sorted(required - row.keys())
    if missing:
        raise ValueError(f"{row.get('query_id')}: missing query fields {missing}")
    if row["state"] not in {"original", "edited"}:
        raise ValueError(f"{row['query_id']}: invalid image state")
    if row["task_type"] not in {"object_a_color", "object_b_color", "binding_choice"}:
        raise ValueError(f"{row['query_id']}: invalid task type")
    if row["stateless"] is not True or row["conversation_history"] != []:
        raise ValueError(f"{row['query_id']}: query is not stateless")
    alphabet = row["response_alphabet"]
    expected = ["A", "B"] if row["task_type"] == "binding_choice" else ["A", "B", "C", "D"]
    if alphabet != expected:
        raise ValueError(f"{row['query_id']}: response alphabet differs from task contract")
    options = option_map(row)
    if list(options) != alphabet:
        raise ValueError(f"{row['query_id']}: option IDs/order differ from response alphabet")
    gold = row["correct_option_id"]
    if gold not in options or options[gold].get("semantic_value") != row["correct_semantic_answer"]:
        raise ValueError(f"{row['query_id']}: frozen gold mapping is inconsistent")
    if primary and row.get("analysis_role") != "primary":
        raise ValueError(f"{row['query_id']}: non-Primary row in Primary mode")
    for key in row:
        lowered = key.lower()
        if any(term in lowered for term in PROHIBITED_MODEL_INPUT_KEYS):
            raise ValueError(f"{row['query_id']}: prohibited field present: {key}")


def render_semantic_user_content(row: dict[str, Any], response_instruction: str) -> str:
    """Render only the frozen question/options plus the common response instruction."""
    validate_query_contract(row, primary=row.get("analysis_role") == "primary")
    prompt_text = row["prompt_text"]
    lines = [prompt_text]
    lines.extend(f"{option['option_id']}. {option['display_text']}" for option in row["options"])
    if not _FORMAT_INSTRUCTION.search(prompt_text):
        lines.append(response_instruction)
    return "\n".join(lines)


def parse_option_letter_strict(
    raw_model_text: str | None, response_alphabet: list[str]
) -> str | None:
    """Preserve the historical exact-form parser used for the first Qwen run."""
    if not isinstance(raw_model_text, str):
        return None
    match = _OPTION_RESPONSE.fullmatch(raw_model_text)
    if match is None:
        return None
    letter = next((group for group in match.groups() if group is not None), None)
    canonical = letter.upper() if letter else None
    return canonical if canonical in response_alphabet else None


def parse_option_letter(
    raw_model_text: str | None, response_alphabet: list[str]
) -> str | None:
    strict = parse_option_letter_strict(raw_model_text, response_alphabet)
    if strict is not None or not isinstance(raw_model_text, str):
        return strict
    match = _LEADING_OPTION_RESPONSE.fullmatch(raw_model_text)
    canonical = match.group(1).upper() if match else None
    return canonical if canonical in response_alphabet else None


def evaluate_response(
    row: dict[str, Any], raw_model_text: str | None, inference_error: str | None
) -> dict[str, Any]:
    options = option_map(row)
    parsed = None if inference_error else parse_option_letter(raw_model_text, row["response_alphabet"])
    semantic = options[parsed]["semantic_value"] if parsed else None
    valid = parsed is not None and inference_error is None
    return {
        "raw_model_text": raw_model_text,
        "parsed_option_id": parsed,
        "parsed_semantic_answer": semantic,
        "gold_option_id": row["correct_option_id"],
        "gold_semantic_answer": row["correct_semantic_answer"],
        "is_valid": valid,
        "is_correct": bool(valid and parsed == row["correct_option_id"]),
        "inference_error": inference_error,
    }


def build_result_record(
    row: dict[str, Any],
    model_config: dict[str, Any],
    common_config: dict[str, Any],
    inference_config_sha256: str,
    semantic_user_content: str,
    raw_model_text: str | None,
    inference_error: str | None,
) -> dict[str, Any]:
    if common_config.get("parser_contract_version") != PARSER_CONTRACT_VERSION:
        raise ValueError("Inference config parser contract differs from loaded parser")
    task_type = row["task_type"]
    object_role = (
        "object_a" if task_type == "object_a_color"
        else "object_b" if task_type == "object_b_color"
        else None
    )
    record = {
        "schema_version": "1.2",
        "inference_contract_version": CONTRACT_VERSION,
        "parser_contract_version": common_config["parser_contract_version"],
        "query_id": row["query_id"],
        "candidate_id": row["candidate_id"],
        "frozen_order": row.get("frozen_order"),
        "primary_domain_order": row.get("primary_domain_order"),
        "image_condition": row["state"],
        "task_type": task_type,
        "object_role": object_role,
        "model_id": model_config["model_id"],
        "model_revision": model_config["model_revision"],
        "processor_revision": model_config["processor_revision"],
        "software_environment_id": model_config["software_environment_id"],
        "semantic_user_content": semantic_user_content,
        "semantic_user_content_sha256": hashlib.sha256(
            semantic_user_content.encode("utf-8")
        ).hexdigest(),
        "decoding_config": {
            "batch_size": common_config["batch_size"],
            "dtype": common_config["dtype"],
            "do_sample": common_config["do_sample"],
            "num_beams": common_config["num_beams"],
            "max_new_tokens": common_config["max_new_tokens"],
            "temperature": common_config["temperature"],
            "top_p": common_config["top_p"],
            "top_k": common_config["top_k"],
        },
        "inference_config_sha256": inference_config_sha256,
        "inference_timestamp_utc": datetime.now(timezone.utc).isoformat(),
    }
    record.update(evaluate_response(row, raw_model_text, inference_error))
    return record
