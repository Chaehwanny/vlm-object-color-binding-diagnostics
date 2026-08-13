#!/usr/bin/env python3
"""Validate one model's complete Primary v1.2 inference result manifest."""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from collections import Counter
from pathlib import Path
from typing import Any


PROJECT_ROOT = Path(__file__).resolve().parents[2]
INFERENCE_DIR = PROJECT_ROOT / "scripts/inference"
if str(INFERENCE_DIR) not in sys.path:
    sys.path.insert(0, str(INFERENCE_DIR))

from primary_contract_v1_2 import (  # noqa: E402
    CONTRACT_VERSION,
    PARSER_CONTRACT_VERSION,
    evaluate_response,
    read_json,
    read_jsonl,
    render_semantic_user_content,
    sha256_file,
    validate_query_contract,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--project-root", type=Path, default=PROJECT_ROOT)
    parser.add_argument(
        "--config",
        type=Path,
        default=PROJECT_ROOT / "configs/attribute_binding/primary_inference_v1.2.json",
    )
    parser.add_argument("--query-manifest", type=Path)
    parser.add_argument("--result-manifest", type=Path, required=True)
    parser.add_argument("--model-key", required=True)
    return parser.parse_args()


def freeze_verify(project_root: Path, config: dict[str, Any]) -> tuple[bool, str]:
    freeze_manifest = project_root / config["primary_freeze_manifest"]
    if not freeze_manifest.is_file():
        return False, "Primary freeze manifest is missing"
    if sha256_file(freeze_manifest) != config["primary_freeze_manifest_sha256"]:
        return False, "Primary freeze manifest SHA differs"
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
    return completed.returncode == 0, completed.stdout + completed.stderr


def prohibited_result_keys(row: dict[str, Any]) -> list[str]:
    prohibited = ("bbox", "mask", "overlay")
    return sorted(key for key in row if any(term in key.lower() for term in prohibited))


def main() -> None:
    args = parse_args()
    project_root = args.project_root.resolve()
    config_path = args.config.resolve()
    config = read_json(config_path)
    if args.model_key not in config["models"]:
        raise ValueError(f"Unknown model key: {args.model_key}")
    model = config["models"][args.model_key]
    common = config["common"]
    query_path = (
        args.query_manifest.resolve()
        if args.query_manifest
        else project_root / config["primary_query_manifest"]
    )
    queries = read_jsonl(query_path)
    results = read_jsonl(args.result_manifest)
    errors: list[str] = []
    if config.get("contract_version") != CONTRACT_VERSION:
        errors.append("inference config contract version differs from loaded contract")
    if common.get("parser_contract_version") != PARSER_CONTRACT_VERSION:
        errors.append("inference config parser version differs from loaded parser")

    freeze_ok, freeze_output = freeze_verify(project_root, config)
    if not freeze_ok:
        errors.append(f"Primary freeze verification failed: {freeze_output}")
    if query_path.resolve() != (project_root / config["primary_query_manifest"]).resolve():
        errors.append("validator received a non-frozen query manifest")
    if sha256_file(query_path) != config["primary_query_manifest_sha256"]:
        errors.append("Primary query manifest SHA differs")
    if len(queries) != config["expected_query_count"]:
        errors.append(f"query count {len(queries)} != {config['expected_query_count']}")
    if len(results) != config["expected_query_count"]:
        errors.append(f"result count {len(results)} != {config['expected_query_count']}")

    query_ids = [row.get("query_id") for row in queries]
    result_ids = [row.get("query_id") for row in results]
    duplicate_ids = sorted(
        query_id for query_id, count in Counter(result_ids).items() if count > 1
    )
    missing = sorted(set(query_ids) - set(result_ids))
    unexpected = sorted(set(result_ids) - set(query_ids))
    if duplicate_ids:
        errors.append(f"duplicate result query IDs: {duplicate_ids}")
    if missing:
        errors.append(f"missing result query IDs: {missing}")
    if unexpected:
        errors.append(f"unexpected result query IDs: {unexpected}")
    if result_ids != query_ids:
        errors.append("result order differs from frozen query order")

    config_sha = sha256_file(config_path)
    expected_decoding = {
        "batch_size": common["batch_size"],
        "dtype": common["dtype"],
        "do_sample": common["do_sample"],
        "num_beams": common["num_beams"],
        "max_new_tokens": common["max_new_tokens"],
        "temperature": common["temperature"],
        "top_p": common["top_p"],
        "top_k": common["top_k"],
    }
    query_by_id = {row["query_id"]: row for row in queries}
    for result in results:
        query_id = result.get("query_id")
        query = query_by_id.get(query_id)
        if query is None:
            continue
        try:
            validate_query_contract(query, primary=True)
        except ValueError as exc:
            errors.append(str(exc))
            continue
        prefix = f"{query_id}: "
        exact_fields = {
            "candidate_id": query["candidate_id"],
            "frozen_order": query.get("frozen_order"),
            "primary_domain_order": query.get("primary_domain_order"),
            "image_condition": query["state"],
            "task_type": query["task_type"],
            "model_id": model["model_id"],
            "model_revision": model["model_revision"],
        "inference_contract_version": config["contract_version"],
        "parser_contract_version": common["parser_contract_version"],
            "processor_revision": model["processor_revision"],
            "software_environment_id": model["software_environment_id"],
            "inference_contract_version": config["contract_version"],
            "parser_contract_version": common["parser_contract_version"],
            "gold_option_id": query["correct_option_id"],
            "gold_semantic_answer": query["correct_semantic_answer"],
            "inference_config_sha256": config_sha,
        }
        for field, expected in exact_fields.items():
            if result.get(field) != expected:
                errors.append(prefix + f"{field} differs")
        expected_content = render_semantic_user_content(
            query, common["response_instruction"]
        )
        if result.get("semantic_user_content") != expected_content:
            errors.append(prefix + "semantic user content differs")
        if result.get("decoding_config") != expected_decoding:
            errors.append(prefix + "decoding config differs")
        prohibited = prohibited_result_keys(result)
        if prohibited:
            errors.append(prefix + f"prohibited result fields: {prohibited}")

        inference_error = result.get("inference_error")
        raw = result.get("raw_model_text")
        if inference_error is None and not isinstance(raw, str):
            errors.append(prefix + "raw model text missing without inference error")
        if inference_error is not None and not isinstance(inference_error, str):
            errors.append(prefix + "inference_error must be null or string")
        expected_eval = evaluate_response(query, raw, inference_error)
        for field in (
            "parsed_option_id", "parsed_semantic_answer", "gold_option_id",
            "gold_semantic_answer", "is_valid", "is_correct", "inference_error",
        ):
            if result.get(field) != expected_eval[field]:
                errors.append(prefix + f"{field} disagrees with frozen parser/gold")
        parsed = result.get("parsed_option_id")
        if parsed is not None and parsed not in query["response_alphabet"]:
            errors.append(prefix + "parsed option is outside response alphabet")

    summary = {
        "schema_version": "1.2",
        "status": "pass" if not errors else "fail",
        "model_key": args.model_key,
        "model_id": model["model_id"],
        "model_revision": model["model_revision"],
        "inference_contract_version": config["contract_version"],
        "parser_contract_version": common["parser_contract_version"],
        "expected_query_count": config["expected_query_count"],
        "result_count": len(results),
        "duplicate_count": len(duplicate_ids),
        "missing_count": len(missing),
        "unexpected_count": len(unexpected),
        "valid_count": sum(row.get("is_valid") is True for row in results),
        "invalid_response_count": sum(
            row.get("is_valid") is False and row.get("inference_error") is None
            for row in results
        ),
        "inference_error_count": sum(row.get("inference_error") is not None for row in results),
        "primary_freeze_verify": "pass" if freeze_ok else "fail",
        "validation_error_count": len(errors),
        "validation_errors": errors,
    }
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    if errors:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
