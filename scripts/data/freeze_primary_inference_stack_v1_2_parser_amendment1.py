#!/usr/bin/env python3
"""Create or verify the Primary v1.2 parser-amendment1 inference-stack freeze."""

from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import subprocess
from datetime import datetime
from pathlib import Path
from typing import Any


SCHEMA_VERSION = "primary_inference_stack_freeze_v1.2_parser_amendment1"
FREEZE_ID = "attribute_binding_primary_inference_v1.2_parser_amendment1"
MANIFEST_NAME = "primary_inference_freeze_manifest_v1.2_parser_amendment1.json"
CHECKSUM_NAME = "SHA256SUMS"
EXPECTED_CANDIDATE_FINGERPRINT = (
    "1ef62448c53322c9519f7e4a9f67cc8af6ec7d15ee28ce06e2705b4e90d21d2b"
)
EXPECTED_QUERY_FINGERPRINT = (
    "5cf40c94a73c22db4ed11a4b5881f586ccc852894d53f232962233b8fa1fb605"
)
EXPECTED_PRIMARY_FREEZE_SHA = (
    "6c8c00aee727330cc44ddcdbf4b182e4eb70474d5780da2416169d6b8d4926c3"
)
IMMUTABILITY_POLICY = (
    "The inference implementation, dependency contracts, pinned model revisions, "
    "decoding contract, parser contract, Primary references, and development-only "
    "technical-smoke evidence and one-time parser amendment are frozen before amended Primary inference. Existing freeze "
    "records must never be silently overwritten."
)

ARTIFACTS = (
    ("inference_freezer", "scripts/data/freeze_primary_inference_stack_v1_2_parser_amendment1.py"),
    ("inference_config", "configs/attribute_binding/primary_inference_v1.2_parser_amendment1.json"),
    ("parser_amendment_document", "docs/protocols/primary_response_parser_amendment_v1.2.md"),
    ("inference_contract", "scripts/inference/primary_contract_v1_2.py"),
    ("inference_runner", "scripts/inference/run_primary_vlm_v1_2.py"),
    ("inference_validator", "scripts/data/validate_primary_inference_results_v1_2.py"),
    ("parser_tests", "tests/inference/test_primary_parser_v1_2.py"),
    ("technical_smoke_fixture", "tests/fixtures/primary_inference_technical_smoke_v1.2.jsonl"),
    ("environment_a_requirements", "requirements/primary-vlm-tf457.txt"),
    ("environment_b_requirements", "requirements/primary-vlm-llava.txt"),
    ("primary_data_freeze_manifest", "frozen/attribute_binding_primary_v1.2/primary_freeze_manifest_v1.2.json"),
    ("primary_data_freeze_checksums", "frozen/attribute_binding_primary_v1.2/SHA256SUMS"),
    ("primary_queries", "data/manifests/attribute_binding/primary_queries_v1.2.jsonl"),
    ("historical_inference_freeze_manifest", "frozen/attribute_binding_primary_inference_v1.2/primary_inference_freeze_manifest_v1.2.json"),
    ("historical_inference_freeze_checksums", "frozen/attribute_binding_primary_inference_v1.2/SHA256SUMS"),
    ("historical_qwen_first_run", "processed/attribute_binding/primary_inference/v1.2/qwen3_vl_8b_instruct.jsonl"),
    ("qwen_technical_smoke", "processed/attribute_binding/inference_smoke/v1.2/qwen3_vl_8b_instruct.jsonl"),
    ("internvl_technical_smoke", "processed/attribute_binding/inference_smoke/v1.2/internvl3_5_8b_instruct.jsonl"),
    ("gemma_technical_smoke", "processed/attribute_binding/inference_smoke/v1.2/gemma3_12b_it.jsonl"),
    ("llava_technical_smoke", "processed/attribute_binding/inference_smoke/v1.2/llava_onevision2_8b_instruct.jsonl"),
)

EXPECTED_MODELS = {
    "qwen3_vl_8b_instruct": {
        "model_id": "Qwen/Qwen3-VL-8B-Instruct",
        "model_revision": "0c351dd01ed87e9c1b53cbc748cba10e6187ff3b",
        "processor_revision": "0c351dd01ed87e9c1b53cbc748cba10e6187ff3b",
        "software_environment_id": "primary_vlm_tf457_v1",
        "adapter": "qwen3_vl",
        "trust_remote_code": False,
        "preprocessing_policy": "pinned_official_auto_processor",
    },
    "internvl3_5_8b_instruct": {
        "model_id": "OpenGVLab/InternVL3_5-8B-Instruct",
        "model_revision": "27506812aa9914804018996329e895977ee2d0c8",
        "processor_revision": "27506812aa9914804018996329e895977ee2d0c8",
        "software_environment_id": "primary_vlm_tf457_v1",
        "adapter": "internvl3_5",
        "trust_remote_code": True,
        "preprocessing_policy": "official_dynamic_448_bicubic_imagenet_max_tiles_12",
    },
    "gemma3_12b_it": {
        "model_id": "google/gemma-3-12b-it",
        "model_revision": "96b6f1eccf38110c56df3a15bffe176da04bfd80",
        "processor_revision": "96b6f1eccf38110c56df3a15bffe176da04bfd80",
        "software_environment_id": "primary_vlm_tf457_v1",
        "adapter": "gemma3",
        "trust_remote_code": False,
        "preprocessing_policy": "pinned_official_auto_processor_896px",
    },
    "llava_onevision2_8b_instruct": {
        "model_id": "lmms-lab-encoder/LLaVA-OneVision-2-8B-Instruct",
        "model_revision": "802f16c8346a062fef7ddf92b2bd64a770a50a1a",
        "processor_revision": "802f16c8346a062fef7ddf92b2bd64a770a50a1a",
        "software_environment_id": "primary_vlm_llava_tf57_v1",
        "adapter": "llava_onevision2",
        "trust_remote_code": True,
        "preprocessing_policy": "pinned_official_custom_auto_processor",
    },
}

SMOKE_ARTIFACT_BY_MODEL = {
    "qwen3_vl_8b_instruct": "qwen_technical_smoke",
    "internvl3_5_8b_instruct": "internvl_technical_smoke",
    "gemma3_12b_it": "gemma_technical_smoke",
    "llava_onevision2_8b_instruct": "llava_technical_smoke",
}

EXPECTED_COMMON = {
    "stateless": True,
    "batch_size": 1,
    "dtype": "bfloat16",
    "do_sample": False,
    "num_beams": 1,
    "max_new_tokens": 16,
    "temperature": None,
    "top_p": None,
    "top_k": None,
    "raw_output_preservation": True,
    "invalid_policy": "incorrect_and_report_separately",
    "inference_error_policy": "separate_from_invalid_response",
    "parser_contract_version": "leading_option_identifier_v1.2_amendment1",
}

EXPECTED_ENVIRONMENTS = {
    "primary_vlm_tf457_v1": {
        "python_path": ".venv-primary-vlm-tf457/bin/python",
        "python": "3.12.3",
        "torch": "2.9.0+cu128",
        "torchvision": "0.24.0+cu128",
        "transformers": "4.57.6",
        "cuda_runtime": "12.8",
        "required_imports": ["einops", "timm"],
        "pip_check_policy": "must_pass",
    },
    "primary_vlm_llava_tf57_v1": {
        "python_path": ".venv-primary-vlm-llava/bin/python",
        "python": "3.12.3",
        "torch": "2.9.0+cu128",
        "torchvision": "0.24.0+cu128",
        "transformers": "5.7.0",
        "cuda_runtime": "12.8",
        "required_imports": ["cv2", "decord"],
        "pip_check_policy": "allow_decord_platform_metadata_warning_only",
    },
}

ENVIRONMENT_INSPECTION_CODE = r"""
import importlib
import json
import platform
import torch
import torchvision
import transformers

required = json.loads(__import__("sys").argv[1])
imports = {}
for name in required:
    module = importlib.import_module(name)
    imports[name] = getattr(module, "__version__", "imported")
print(json.dumps({
    "python": platform.python_version(),
    "torch": torch.__version__,
    "torchvision": torchvision.__version__,
    "transformers": transformers.__version__,
    "cuda_available": torch.cuda.is_available(),
    "cuda_runtime": torch.version.cuda,
    "gpu": torch.cuda.get_device_name(0) if torch.cuda.is_available() else None,
    "bf16_supported": (
        torch.cuda.is_bf16_supported() if torch.cuda.is_available() else False
    ),
    "required_imports": imports,
}))
"""


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="mode", required=True)
    for mode in ("create", "verify"):
        command = subparsers.add_parser(mode)
        command.add_argument("--project-root", type=Path, required=True)
        command.add_argument("--output-dir", type=Path, required=True)
    return parser.parse_args()


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def read_json(path: Path) -> dict[str, Any]:
    with path.open(encoding="utf-8") as handle:
        value = json.load(handle)
    if not isinstance(value, dict):
        raise ValueError(f"Expected a JSON object: {path}")
    return value


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    with path.open(encoding="utf-8") as handle:
        rows = [json.loads(line) for line in handle if line.strip()]
    if any(not isinstance(row, dict) for row in rows):
        raise ValueError(f"Expected JSON objects in JSONL file: {path}")
    return rows


def canonical_sequence_sha256(values: list[str]) -> str:
    payload = json.dumps(values, ensure_ascii=False, separators=(",", ":"))
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def resolve_relative(project_root: Path, relative: str) -> Path:
    if Path(relative).is_absolute():
        raise ValueError(f"Frozen artifact path must be repository-relative: {relative}")
    root = project_root.resolve()
    path = (root / relative).resolve()
    if path != root and root not in path.parents:
        raise ValueError(f"Frozen artifact escapes project root: {relative}")
    return path


def resolve_environment_python(project_root: Path, relative: str) -> Path:
    """Resolve a repository venv entry without following its interpreter symlink."""
    relative_path = Path(relative)
    if relative_path.is_absolute() or ".." in relative_path.parts:
        raise ValueError(f"Environment Python path must be repository-relative: {relative}")
    path = project_root / relative_path
    if not path.is_file():
        raise FileNotFoundError(f"Missing environment Python: {relative}")
    return path

def artifact_inventory(project_root: Path) -> list[dict[str, Any]]:
    inventory = []
    for role, relative in ARTIFACTS:
        path = resolve_relative(project_root, relative)
        if not path.is_file():
            raise FileNotFoundError(f"Missing required inference freeze artifact: {relative}")
        inventory.append({
            "role": role,
            "path": relative,
            "sha256": sha256(path),
            "byte_size": path.stat().st_size,
        })
    return inventory


def artifact_by_role(inventory: list[dict[str, Any]], role: str) -> dict[str, Any]:
    matches = [artifact for artifact in inventory if artifact.get("role") == role]
    if len(matches) != 1:
        raise ValueError(f"Expected exactly one artifact with role {role!r}")
    return matches[0]


def run_checked(command: list[str], label: str) -> str:
    completed = subprocess.run(command, text=True, capture_output=True, check=False)
    if completed.returncode != 0:
        raise ValueError(
            f"{label} failed with exit {completed.returncode}:\n"
            f"{completed.stdout}{completed.stderr}"
        )
    return completed.stdout + completed.stderr


def verify_primary_data_freeze(project_root: Path, python_path: Path) -> None:
    run_checked([
        str(python_path),
        str(project_root / "scripts/data/freeze_primary_pre_inference_v1_2.py"),
        "verify",
        "--project-root",
        str(project_root),
        "--output-dir",
        str(project_root / "frozen/attribute_binding_primary_v1.2"),
    ], "Primary data freeze verification")


def verify_parser_tests(project_root: Path, python_path: Path) -> None:
    run_checked(
        [str(python_path), "-m", "unittest", "tests.inference.test_primary_parser_v1_2"],
        "Primary parser tests",
    )


def inspect_environment(
    project_root: Path, environment_id: str, expected: dict[str, Any]
) -> dict[str, Any]:
    python_path = resolve_environment_python(project_root, expected["python_path"])
    if not python_path.is_file():
        raise FileNotFoundError(f"Missing Python for {environment_id}: {python_path}")
    inspected = run_checked([
        str(python_path),
        "-c",
        ENVIRONMENT_INSPECTION_CODE,
        json.dumps(expected["required_imports"]),
    ], f"{environment_id} inspection")
    record = json.loads(inspected.strip())
    for field in ("python", "torch", "torchvision", "transformers", "cuda_runtime"):
        if record.get(field) != expected[field]:
            raise ValueError(
                f"{environment_id} {field}={record.get(field)!r}, "
                f"expected {expected[field]!r}"
            )
    if record.get("cuda_available") is not True:
        raise ValueError(f"{environment_id} CUDA is unavailable")
    if record.get("bf16_supported") is not True:
        raise ValueError(f"{environment_id} BF16 is unsupported")

    pip_check = subprocess.run(
        [str(python_path), "-m", "pip", "check"],
        text=True,
        capture_output=True,
        check=False,
    )
    pip_text = (pip_check.stdout + pip_check.stderr).strip()
    policy = expected["pip_check_policy"]
    if policy == "must_pass" and pip_check.returncode != 0:
        raise ValueError(f"{environment_id} pip check failed: {pip_text}")
    if policy == "allow_decord_platform_metadata_warning_only":
        warning = "decord 0.6.0 is not supported on this platform"
        lines = [line.strip() for line in pip_text.splitlines() if line.strip()]
        if pip_check.returncode not in (0, 1):
            raise ValueError(f"{environment_id} unexpected pip check exit")
        if pip_check.returncode == 1 and lines != [warning]:
            raise ValueError(f"{environment_id} has unexpected pip issues: {lines}")
    record["python_path"] = expected["python_path"]
    record["pip_check_returncode"] = pip_check.returncode
    record["pip_check_output"] = pip_text
    record["pip_check_policy"] = policy
    if environment_id == "primary_vlm_llava_tf57_v1":
        record["decord_note"] = (
            "pip metadata warns that decord 0.6.0 is unsupported on this platform; "
            "direct import succeeded and the frozen single-image runner does not "
            "reference or invoke decord."
        )
    return record


def load_contract_module(project_root: Path) -> Any:
    path = project_root / "scripts/inference/primary_contract_v1_2.py"
    spec = importlib.util.spec_from_file_location("frozen_primary_contract_v1_2", path)
    if spec is None or spec.loader is None:
        raise ValueError("Unable to load Primary inference contract")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def validate_smoke_evidence(
    inventory: list[dict[str, Any]],
    project_root: Path,
    config: dict[str, Any],
) -> dict[str, Any]:
    fixture = read_jsonl(
        resolve_relative(
            project_root,
            artifact_by_role(inventory, "technical_smoke_fixture")["path"],
        )
    )
    if len(fixture) != 2:
        raise ValueError(f"Technical smoke fixture has {len(fixture)} rows, expected 2")
    if {row.get("task_type") for row in fixture} != {"object_a_color", "binding_choice"}:
        raise ValueError("Technical smoke fixture does not contain one color and one binding item")
    for row in fixture:
        if (
            row.get("analysis_role") != "technical_smoke_only"
            or row.get("development_exposure") is not True
            or row.get("confirmatory_test_eligible") is not False
        ):
            raise ValueError("Technical smoke fixture role/exposure contract differs")

    evidence = {}
    fixture_ids = [row["query_id"] for row in fixture]
    for model_key, artifact_role in SMOKE_ARTIFACT_BY_MODEL.items():
        path = resolve_relative(
            project_root, artifact_by_role(inventory, artifact_role)["path"]
        )
        rows = read_jsonl(path)
        model = config["models"][model_key]
        if [row.get("query_id") for row in rows] != fixture_ids:
            raise ValueError(f"{model_key} smoke query membership/order differs")
        if any(row.get("model_id") != model["model_id"] for row in rows):
            raise ValueError(f"{model_key} smoke model ID differs")
        if any(row.get("model_revision") != model["model_revision"] for row in rows):
            raise ValueError(f"{model_key} smoke model revision differs")
        if any(row.get("processor_revision") != model["processor_revision"] for row in rows):
            raise ValueError(f"{model_key} smoke processor revision differs")
        if any(row.get("inference_error") is not None for row in rows):
            raise ValueError(f"{model_key} smoke contains inference errors")
        if any(row.get("is_valid") is not True for row in rows):
            raise ValueError(f"{model_key} smoke contains parser-invalid output")
        evidence[model_key] = {
            "analysis_role": "development_only_technical_smoke",
            "confirmatory_research_result": False,
            "item_count": len(rows),
            "generation_success_count": sum(
                row.get("inference_error") is None for row in rows
            ),
            "parser_valid_count": sum(row.get("is_valid") is True for row in rows),
            "inference_error_count": sum(
                row.get("inference_error") is not None for row in rows
            ),
            "artifact_role": artifact_role,
        }
    return evidence


def validate_historical_provenance(
    inventory: list[dict[str, Any]], project_root: Path
) -> dict[str, Any]:
    freeze_path = resolve_relative(
        project_root, artifact_by_role(inventory, "historical_inference_freeze_manifest")["path"]
    )
    freeze = read_json(freeze_path)
    frozen_parser = (freeze.get("contract") or {}).get("parser_contract", {}).get("parser_version")
    if frozen_parser != "strict_option_letter_v1.2":
        raise ValueError("Historical inference freeze parser version differs")
    result_path = resolve_relative(
        project_root, artifact_by_role(inventory, "historical_qwen_first_run")["path"]
    )
    rows = read_jsonl(result_path)
    valid = sum(row.get("is_valid") is True for row in rows)
    invalid = sum(
        row.get("is_valid") is False and row.get("inference_error") is None
        for row in rows
    )
    errors = sum(row.get("inference_error") is not None for row in rows)
    if len(rows) != 546 or (valid, invalid, errors) != (290, 256, 0):
        raise ValueError("Historical Qwen first-run structure differs from 546/290/256/0")
    if any(row.get("parser_contract_version") != "strict_option_letter_v1.2" for row in rows):
        raise ValueError("Historical Qwen first-run parser version differs")
    return {
        "amendment_timing": "post_first_qwen_run_pre_task_analysis",
        "task_level_metrics_seen_before_change": False,
        "historical_parser_version": "strict_option_letter_v1.2",
        "historical_qwen_result_sha256": sha256(result_path),
        "historical_inference_freeze_sha256": sha256(freeze_path),
        "historical_qwen_result_count": len(rows),
        "historical_qwen_valid_count": valid,
        "historical_qwen_invalid_count": invalid,
        "historical_qwen_inference_error_count": errors,
        "historical_artifacts_preserved": True,
    }

def current_contract(
    project_root: Path, inventory: list[dict[str, Any]]
) -> dict[str, Any]:
    config_path = resolve_relative(
        project_root, artifact_by_role(inventory, "inference_config")["path"]
    )
    config = read_json(config_path)
    if config.get("contract_version") != "primary_inference_v1.2_parser_amendment1":
        raise ValueError("Amended inference config contract version differs")
    environment_a_python = resolve_environment_python(
        project_root, EXPECTED_ENVIRONMENTS["primary_vlm_tf457_v1"]["python_path"]
    )
    verify_primary_data_freeze(project_root, environment_a_python)
    verify_parser_tests(project_root, environment_a_python)

    if config.get("expected_candidate_count") != 91:
        raise ValueError("Inference config Primary candidate count differs from 91")
    if config.get("expected_query_count") != 546:
        raise ValueError("Inference config Primary query count differs from 546")
    if config.get("models") != {
        key: config["models"].get(key) for key in EXPECTED_MODELS
    } or set(config.get("models", {})) != set(EXPECTED_MODELS):
        raise ValueError("Inference config model panel membership/order differs")
    for key, expected in EXPECTED_MODELS.items():
        actual = config["models"][key]
        for field, value in expected.items():
            if actual.get(field) != value:
                raise ValueError(f"{key} frozen model field differs: {field}")
    common = config.get("common", {})
    for field, value in EXPECTED_COMMON.items():
        if common.get(field) != value:
            raise ValueError(f"Frozen decoding/parser field differs: {field}")

    primary_freeze_path = resolve_relative(
        project_root,
        artifact_by_role(inventory, "primary_data_freeze_manifest")["path"],
    )
    query_path = resolve_relative(
        project_root, artifact_by_role(inventory, "primary_queries")["path"]
    )
    if sha256(primary_freeze_path) != EXPECTED_PRIMARY_FREEZE_SHA:
        raise ValueError("Primary data freeze manifest SHA differs")
    if config.get("primary_freeze_manifest_sha256") != EXPECTED_PRIMARY_FREEZE_SHA:
        raise ValueError("Inference config Primary freeze SHA differs")
    if sha256(query_path) != config.get("primary_query_manifest_sha256"):
        raise ValueError("Inference config Primary query SHA differs")

    queries = read_jsonl(query_path)
    query_ids = [str(row.get("query_id")) for row in queries]
    candidate_ids = list(dict.fromkeys(str(row.get("candidate_id")) for row in queries))
    candidate_fingerprint = canonical_sequence_sha256(candidate_ids)
    query_fingerprint = canonical_sequence_sha256(query_ids)
    if len(queries) != 546 or len(candidate_ids) != 91:
        raise ValueError("Primary query/candidate inventory differs from 546/91")
    if candidate_fingerprint != EXPECTED_CANDIDATE_FINGERPRINT:
        raise ValueError("Primary candidate fingerprint differs")
    if query_fingerprint != EXPECTED_QUERY_FINGERPRINT:
        raise ValueError("Primary query fingerprint differs")
    if config.get("primary_candidate_fingerprint") != candidate_fingerprint:
        raise ValueError("Inference config candidate fingerprint differs")
    if config.get("primary_query_fingerprint") != query_fingerprint:
        raise ValueError("Inference config query fingerprint differs")

    contract_module = load_contract_module(project_root)
    if contract_module.CONTRACT_VERSION != "primary_inference_v1.2_parser_amendment1":
        raise ValueError("Inference contract version differs")
    if contract_module.PARSER_CONTRACT_VERSION != "leading_option_identifier_v1.2_amendment1":
        raise ValueError("Parser contract version differs")
    if contract_module.RESPONSE_INSTRUCTION != common.get("response_instruction"):
        raise ValueError("Response instruction differs between config and contract")

    environments = {
        environment_id: inspect_environment(project_root, environment_id, expected)
        for environment_id, expected in EXPECTED_ENVIRONMENTS.items()
    }
    smoke_evidence = validate_smoke_evidence(inventory, project_root, config)
    amendment_provenance = validate_historical_provenance(inventory, project_root)
    return {
        "governing_protocol": "v1.2",
        "primary_provenance": {
            "candidate_count": 91,
            "query_count": 546,
            "candidate_fingerprint": candidate_fingerprint,
            "query_fingerprint": query_fingerprint,
            "primary_freeze_manifest_sha256": sha256(primary_freeze_path),
            "primary_query_manifest_sha256": sha256(query_path),
        },
        "model_provenance": {
            key: {
                **{field: config["models"][key][field] for field in EXPECTED_MODELS[key]},
                "dtype": common["dtype"],
                "loader_class": config["models"][key]["loader_class"],
            }
            for key in EXPECTED_MODELS
        },
        "decoding_contract": {
            **{field: common[field] for field in (
                "stateless", "batch_size", "dtype", "do_sample", "num_beams",
                "max_new_tokens", "temperature", "top_p", "top_k",
            )},
            "sampling_parameters_passed_to_generate": False,
            "preprocessing": "official pinned model-specific preprocessing",
        },
        "parser_contract": {
            "contract_version": contract_module.CONTRACT_VERSION,
            "parser_version": contract_module.PARSER_CONTRACT_VERSION,
            "invalid_policy": common["invalid_policy"],
            "inference_error_policy": common["inference_error_policy"],
            "raw_output_preserved": common["raw_output_preservation"],
        },
        "dependency_provenance": {
            "environment_a_requirements_sha256": artifact_by_role(
                inventory, "environment_a_requirements"
            )["sha256"],
            "environment_b_requirements_sha256": artifact_by_role(
                inventory, "environment_b_requirements"
            )["sha256"],
            "environments": environments,
        },
        "parser_amendment_provenance": amendment_provenance,
        "technical_smoke_evidence": smoke_evidence,
    }


def write_exclusive(path: Path, content: str) -> None:
    temporary = path.with_name(f".{path.name}.tmp")
    if path.exists() or temporary.exists():
        raise FileExistsError(f"Refusing to overwrite inference freeze output: {path}")
    temporary.write_text(content, encoding="utf-8")
    temporary.replace(path)


def expected_checksum_text(
    project_root: Path, manifest_path: Path, inventory: list[dict[str, Any]]
) -> str:
    lines = [f"{artifact['sha256']}  {artifact['path']}" for artifact in inventory]
    relative = manifest_path.resolve().relative_to(project_root.resolve()).as_posix()
    lines.append(f"{sha256(manifest_path)}  {relative}")
    return "\n".join(lines) + "\n"


def create(project_root: Path, output_dir: Path) -> None:
    if output_dir.exists() and any(output_dir.iterdir()):
        raise FileExistsError(f"Inference freeze output directory is nonempty: {output_dir}")
    inventory = artifact_inventory(project_root)
    contract = current_contract(project_root, inventory)
    manifest = {
        "schema_version": SCHEMA_VERSION,
        "freeze_id": FREEZE_ID,
        "governing_protocol": "v1.2",
        "freeze_timestamp": datetime.now().astimezone().isoformat(),
        "artifact_count": len(inventory),
        "artifacts": inventory,
        "contract": contract,
        "immutability_policy": IMMUTABILITY_POLICY,
    }
    output_dir.mkdir(parents=True, exist_ok=True)
    manifest_path = output_dir / MANIFEST_NAME
    checksum_path = output_dir / CHECKSUM_NAME
    write_exclusive(
        manifest_path, json.dumps(manifest, ensure_ascii=False, indent=2) + "\n"
    )
    write_exclusive(
        checksum_path, expected_checksum_text(project_root, manifest_path, inventory)
    )
    print(json.dumps({
        "status": "created",
        "output_dir": output_dir.as_posix(),
        "artifact_count": len(inventory),
        "primary_candidates": contract["primary_provenance"]["candidate_count"],
        "primary_queries": contract["primary_provenance"]["query_count"],
        "models": len(contract["model_provenance"]),
        "errors": [],
    }, ensure_ascii=False, indent=2))


def verify(project_root: Path, output_dir: Path) -> None:
    manifest_path = output_dir / MANIFEST_NAME
    checksum_path = output_dir / CHECKSUM_NAME
    if not manifest_path.is_file() or not checksum_path.is_file():
        raise FileNotFoundError(
            f"Inference freeze manifest or SHA256SUMS is missing from {output_dir}"
        )
    manifest = read_json(manifest_path)
    errors: list[str] = []
    if manifest.get("schema_version") != SCHEMA_VERSION:
        errors.append("inference freeze schema/version differs")
    if manifest.get("freeze_id") != FREEZE_ID:
        errors.append("inference freeze ID differs")
    if manifest.get("governing_protocol") != "v1.2":
        errors.append("governing protocol differs from v1.2")

    inventory = manifest.get("artifacts")
    expected_roles = [role for role, _ in ARTIFACTS]
    expected_paths = [path for _, path in ARTIFACTS]
    if not isinstance(inventory, list):
        errors.append("frozen inference artifact inventory is missing")
        inventory = []
    if [row.get("role") for row in inventory] != expected_roles:
        errors.append("frozen inference artifact roles/order differ")
    if [row.get("path") for row in inventory] != expected_paths:
        errors.append("frozen inference artifact paths/order differ")
    if (
        manifest.get("artifact_count") != len(ARTIFACTS)
        or len(inventory) != len(ARTIFACTS)
    ):
        errors.append("frozen inference artifact count differs")

    changed = []
    for artifact in inventory:
        relative = artifact.get("path")
        try:
            path = resolve_relative(project_root, str(relative))
        except ValueError as exc:
            changed.append({"path": relative, "reason": str(exc)})
            continue
        if not path.is_file():
            changed.append({"path": relative, "reason": "missing"})
            continue
        reasons = []
        if path.stat().st_size != artifact.get("byte_size"):
            reasons.append("byte size differs")
        if sha256(path) != artifact.get("sha256"):
            reasons.append("SHA256 differs")
        if reasons:
            changed.append({"path": relative, "reason": "; ".join(reasons)})
    if changed:
        errors.append(f"changed frozen inference artifacts: {changed}")

    current = None
    if not changed and inventory:
        try:
            current = current_contract(project_root, inventory)
        except Exception as exc:
            errors.append(
                "current inference contract check failed: "
                f"{type(exc).__name__}: {exc}"
            )
    contract_match = current == manifest.get("contract") if current is not None else False
    if current is not None and not contract_match:
        errors.append("current inference contract differs from frozen contract")

    if inventory:
        expected_sums = expected_checksum_text(project_root, manifest_path, inventory)
        if checksum_path.read_text(encoding="utf-8") != expected_sums:
            errors.append("SHA256SUMS differs from frozen inference inventory")
    result = {
        "status": "pass" if not errors else "fail",
        "artifact_count": len(inventory),
        "changed_artifact_count": len(changed),
        "changed_artifacts": changed,
        "contract_match": contract_match,
        "candidate_fingerprint_match": bool(
            current
            and current["primary_provenance"]["candidate_fingerprint"]
            == EXPECTED_CANDIDATE_FINGERPRINT
        ),
        "query_fingerprint_match": bool(
            current
            and current["primary_provenance"]["query_fingerprint"]
            == EXPECTED_QUERY_FINGERPRINT
        ),
        "errors": errors,
    }
    print(json.dumps(result, ensure_ascii=False, indent=2))
    if errors:
        raise SystemExit(1)


def main() -> None:
    args = parse_args()
    project_root = args.project_root.resolve()
    output_dir = args.output_dir
    if not output_dir.is_absolute():
        output_dir = project_root / output_dir
    if args.mode == "create":
        create(project_root, output_dir)
    else:
        verify(project_root, output_dir)


if __name__ == "__main__":
    main()
