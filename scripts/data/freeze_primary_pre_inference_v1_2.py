#!/usr/bin/env python3
"""Create or verify the v1.2 Primary pre-inference provenance freeze."""

from __future__ import annotations

import argparse
import hashlib
import json
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


SCHEMA_VERSION = "primary_pre_inference_freeze_v1.2"
MANIFEST_NAME = "primary_freeze_manifest_v1.2.json"
CHECKSUM_NAME = "SHA256SUMS"
IMMUTABILITY_POLICY = (
    "Primary artifacts are frozen before model inference. Any later modification "
    "requires an explicit amendment and a new provenance record; existing freeze "
    "records must not be silently overwritten."
)
ARTIFACTS = (
    ("primary_queries", "data/manifests/attribute_binding/primary_queries_v1.2.jsonl"),
    ("primary_candidate_status", "data/manifests/attribute_binding/primary_candidate_status_v1.2.jsonl"),
    ("primary_generation_summary", "data/manifests/attribute_binding/primary_generation_summary_v1.2.json"),
    ("primary_validation_summary", "data/manifests/attribute_binding/primary_validation_summary_v1.2.json"),
    ("primary_config", "configs/attribute_binding/primary_qa_v1.2.json"),
    ("reviewed_natural_edit_lineage", "processed/attribute_binding/main_experiment/v1.1/color_editing/main_v1.1/color_edit_results_main_v1.1_reviewed.jsonl"),
    ("frozen_main_source", "processed/attribute_binding/main_experiment/v1.1/preparation/main_input_manifest_v1.1.jsonl"),
    ("primary_generator", "scripts/data/build_primary_qa_v1_2.py"),
    ("primary_validator", "scripts/data/validate_primary_qa_v1_2.py"),
    ("governing_protocol", "docs/protocols/protocol_v1.2_PRE_INFERENCE_AMENDMENT.md"),
    ("task_specification", "docs/protocols/task_specification_v1.1.md"),
    ("manifest_schema", "docs/protocols/manifest_schema_v1.1.md"),
    ("statistical_analysis_plan", "docs/protocols/statistical_analysis_plan_v1.1.md"),
)
MAIN_SOURCE_SHA_RECORD = (
    "processed/attribute_binding/main_experiment/v1.1/preparation/"
    "main_input_manifest_v1.1.sha256"
)


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
        return [json.loads(line) for line in handle if line.strip()]


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


def artifact_inventory(project_root: Path) -> list[dict[str, Any]]:
    inventory = []
    for role, relative in ARTIFACTS:
        path = resolve_relative(project_root, relative)
        if not path.is_file():
            raise FileNotFoundError(f"Missing required freeze artifact: {relative}")
        inventory.append(
            {
                "role": role,
                "path": relative,
                "sha256": sha256(path),
                "byte_size": path.stat().st_size,
            }
        )
    return inventory


def artifact_by_role(inventory: list[dict[str, Any]], role: str) -> dict[str, Any]:
    matches = [artifact for artifact in inventory if artifact.get("role") == role]
    if len(matches) != 1:
        raise ValueError(f"Expected exactly one frozen artifact with role {role!r}.")
    return matches[0]


def current_contract(project_root: Path, inventory: list[dict[str, Any]]) -> dict[str, Any]:
    paths = {
        artifact["role"]: resolve_relative(project_root, artifact["path"])
        for artifact in inventory
    }
    statuses = read_jsonl(paths["primary_candidate_status"])
    included = [row for row in statuses if row.get("primary_question_construction_included") is True]
    candidate_ids = [str(row.get("candidate_id")) for row in included]
    errors: list[str] = []
    if len(statuses) != 91:
        errors.append(f"candidate-status row count is {len(statuses)}, expected 91")
    if len(included) != 91:
        errors.append(f"included candidate count is {len(included)}, expected 91")
    if len(set(candidate_ids)) != 91 or "None" in candidate_ids:
        errors.append("included candidate IDs are missing or duplicated")
    domain_order = [row.get("primary_domain_order") for row in included]
    if domain_order != list(range(1, 92)):
        errors.append("included candidates do not preserve primary_domain_order 1..91")
    frozen_order = [row.get("frozen_order") for row in included]
    if any(not isinstance(value, int) for value in frozen_order) or frozen_order != sorted(frozen_order):
        errors.append("included candidates are not in increasing frozen_order")

    queries = read_jsonl(paths["primary_queries"])
    query_ids = [str(row.get("query_id")) for row in queries]
    if len(queries) != 546:
        errors.append(f"query count is {len(queries)}, expected 546")
    if len(set(query_ids)) != 546 or "None" in query_ids:
        errors.append("query IDs are missing or duplicated")
    query_candidate_ids = [str(row.get("candidate_id")) for row in queries]
    if set(query_candidate_ids) != set(candidate_ids):
        errors.append("query candidate membership differs from Primary candidate membership")
    per_candidate = Counter(query_candidate_ids)
    bad_inventory = sorted(cid for cid in candidate_ids if per_candidate[cid] != 6)
    if bad_inventory:
        errors.append(f"candidates without exactly six queries: {bad_inventory}")

    generation = read_json(paths["primary_generation_summary"])
    validation = read_json(paths["primary_validation_summary"])
    query_sha = artifact_by_role(inventory, "primary_queries")["sha256"]
    config_sha = artifact_by_role(inventory, "primary_config")["sha256"]
    main_source_sha = artifact_by_role(inventory, "frozen_main_source")["sha256"]
    if generation.get("output_manifest_sha256") != query_sha:
        errors.append("generation-summary query SHA differs from current Primary query manifest")
    if generation.get("config_sha256") != config_sha:
        errors.append("generation-summary config SHA differs from current Primary config")
    if generation.get("candidate_domain") != 91 or generation.get("item_count") != 546:
        errors.append("generation-summary candidate/query counts differ from 91/546")
    if validation.get("status") != "pass":
        errors.append(f"Primary validation status is {validation.get('status')!r}, expected 'pass'")
    if validation.get("validation_error_count") != 0 or validation.get("validation_errors") != []:
        errors.append("Primary validation summary contains validation errors")
    if validation.get("primary_candidate_count") != 91 or validation.get("query_count") != 546:
        errors.append("validation-summary candidate/query counts differ from 91/546")

    sha_record_path = resolve_relative(project_root, MAIN_SOURCE_SHA_RECORD)
    if not sha_record_path.is_file():
        errors.append(f"missing frozen Main source SHA record: {MAIN_SOURCE_SHA_RECORD}")
    else:
        fields = sha_record_path.read_text(encoding="ascii").strip().split()
        recorded = fields[0] if fields else ""
        if recorded != main_source_sha:
            errors.append("frozen Main source SHA differs from its existing freeze record")

    if errors:
        raise ValueError("Pre-freeze contract check failed:\n- " + "\n- ".join(errors))
    return {
        "primary_candidate_count": len(included),
        "primary_query_count": len(queries),
        "validation_status": validation["status"],
        "validation_error_count": validation["validation_error_count"],
        "reference_policy": validation.get("reference_policy"),
        "ordered_candidate_membership_fingerprint": {
            "method": "sha256(json.dumps(candidate_id_sequence,ensure_ascii=False,separators=(comma,colon)))",
            "sha256": canonical_sequence_sha256(candidate_ids),
            "sequence_length": len(candidate_ids),
        },
        "ordered_query_inventory_fingerprint": {
            "method": "sha256(json.dumps(query_id_sequence,ensure_ascii=False,separators=(comma,colon)))",
            "sha256": canonical_sequence_sha256(query_ids),
            "sequence_length": len(query_ids),
        },
    }


def write_exclusive(path: Path, content: str) -> None:
    temporary = path.with_name(f".{path.name}.tmp")
    if path.exists() or temporary.exists():
        raise FileExistsError(f"Refusing to overwrite freeze output: {path}")
    temporary.write_text(content, encoding="utf-8")
    temporary.replace(path)


def expected_checksum_text(
    project_root: Path, manifest_path: Path, inventory: list[dict[str, Any]]
) -> str:
    lines = [f"{artifact['sha256']}  {artifact['path']}" for artifact in inventory]
    manifest_relative = manifest_path.resolve().relative_to(project_root.resolve()).as_posix()
    lines.append(f"{sha256(manifest_path)}  {manifest_relative}")
    return "\n".join(lines) + "\n"


def create(project_root: Path, output_dir: Path) -> None:
    if output_dir.exists() and any(output_dir.iterdir()):
        raise FileExistsError(f"Freeze output directory is nonempty: {output_dir}")
    inventory = artifact_inventory(project_root)
    contract = current_contract(project_root, inventory)
    manifest = {
        "schema_version": SCHEMA_VERSION,
        "freeze_id": "attribute_binding_primary_v1.2",
        "governing_protocol": "v1.2",
        "freeze_timestamp": datetime.now(timezone.utc).isoformat(),
        **contract,
        "frozen_main_source_sha256": artifact_by_role(inventory, "frozen_main_source")["sha256"],
        "primary_config_sha256": artifact_by_role(inventory, "primary_config")["sha256"],
        "generator_sha256": artifact_by_role(inventory, "primary_generator")["sha256"],
        "validator_sha256": artifact_by_role(inventory, "primary_validator")["sha256"],
        "artifact_count": len(inventory),
        "artifacts": inventory,
        "immutability_policy": IMMUTABILITY_POLICY,
    }
    output_dir.mkdir(parents=True, exist_ok=True)
    manifest_path = output_dir / MANIFEST_NAME
    checksum_path = output_dir / CHECKSUM_NAME
    write_exclusive(manifest_path, json.dumps(manifest, ensure_ascii=False, indent=2) + "\n")
    write_exclusive(checksum_path, expected_checksum_text(project_root, manifest_path, inventory))
    print(json.dumps({
        "status": "created",
        "output_dir": output_dir.as_posix(),
        "artifact_count": len(inventory),
        **contract,
        "errors": 0,
    }, ensure_ascii=False, indent=2))


def verify(project_root: Path, output_dir: Path) -> None:
    manifest_path = output_dir / MANIFEST_NAME
    checksum_path = output_dir / CHECKSUM_NAME
    if not manifest_path.is_file() or not checksum_path.is_file():
        raise FileNotFoundError(f"Freeze manifest or SHA256SUMS is missing from {output_dir}")
    manifest = read_json(manifest_path)
    errors: list[str] = []
    if manifest.get("schema_version") != SCHEMA_VERSION:
        errors.append("freeze schema/version identifier differs")
    if manifest.get("governing_protocol") != "v1.2":
        errors.append("governing protocol differs from v1.2")
    inventory = manifest.get("artifacts")
    if not isinstance(inventory, list) or len(inventory) != len(ARTIFACTS):
        errors.append("frozen artifact inventory is missing or has the wrong length")
        inventory = []
    expected_roles = [role for role, _ in ARTIFACTS]
    if [artifact.get("role") for artifact in inventory] != expected_roles:
        errors.append("frozen artifact roles/order differ from the v1.2 contract")

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
        actual_size = path.stat().st_size
        actual_sha = sha256(path)
        reasons = []
        if actual_size != artifact.get("byte_size"):
            reasons.append(f"byte_size {actual_size} != {artifact.get('byte_size')}")
        if actual_sha != artifact.get("sha256"):
            reasons.append(f"sha256 {actual_sha} != {artifact.get('sha256')}")
        if reasons:
            changed.append({"path": relative, "reason": "; ".join(reasons)})
    if changed:
        errors.append(f"changed frozen artifacts: {changed}")

    contract: dict[str, Any] | None = None
    if not changed and inventory:
        try:
            contract = current_contract(project_root, inventory)
        except (ValueError, FileNotFoundError, KeyError, TypeError) as exc:
            errors.append(str(exc))
    candidate_match = False
    query_match = False
    if contract is not None:
        candidate_match = (
            contract["ordered_candidate_membership_fingerprint"]
            == manifest.get("ordered_candidate_membership_fingerprint")
        )
        query_match = (
            contract["ordered_query_inventory_fingerprint"]
            == manifest.get("ordered_query_inventory_fingerprint")
        )
        if not candidate_match:
            errors.append("ordered candidate membership fingerprint differs")
        if not query_match:
            errors.append("ordered query inventory fingerprint differs")
        for field in (
            "primary_candidate_count", "primary_query_count",
            "validation_status", "validation_error_count", "reference_policy",
        ):
            if contract.get(field) != manifest.get(field):
                errors.append(f"frozen contract field differs: {field}")

    if inventory:
        expected_sums = expected_checksum_text(project_root, manifest_path, inventory)
        if checksum_path.read_text(encoding="utf-8") != expected_sums:
            errors.append("SHA256SUMS content differs from the frozen inventory")
    result = {
        "status": "pass" if not errors else "fail",
        "changed_artifact_count": len(changed),
        "changed_artifacts": changed,
        "candidate_fingerprint_match": candidate_match,
        "query_fingerprint_match": query_match,
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
