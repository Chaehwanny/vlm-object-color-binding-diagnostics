#!/usr/bin/env python3
"""Build the v1.2 Natural Primary Q/A manifest."""

from __future__ import annotations

import argparse
import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


def read_json(path: Path) -> Any:
    with path.open(encoding="utf-8") as handle:
        return json.load(handle)


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    with path.open(encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()

def canonical_row_sha256(row: dict[str, Any]) -> str:
    payload = json.dumps(row, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def atomic_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    if path.exists():
        raise FileExistsError(f"Refusing to overwrite: {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp")
    with temporary.open("x", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")
    temporary.replace(path)


def atomic_json(path: Path, value: dict[str, Any]) -> None:
    if path.exists():
        raise FileExistsError(f"Refusing to overwrite: {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp")
    temporary.write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    temporary.replace(path)


def project_path(project_root: Path, value: str) -> Path:
    path = Path(value)
    return path if path.is_absolute() else project_root / path


def artifact_path(project_root: Path, manifest: Path, value: str) -> Path:
    path = Path(value)
    if path.is_absolute():
        return path
    project_candidate = project_root / path
    if project_candidate.exists():
        return project_candidate
    return manifest.parent / path


def as_project_relative(project_root: Path, path: Path) -> str:
    resolved = path.resolve()
    try:
        return resolved.relative_to(project_root.resolve()).as_posix()
    except ValueError as exc:
        raise ValueError(f"Asset is outside project root: {path}") from exc


def derive_primary_candidates(reviewed: Path, frozen: Path, project_root: Path) -> list[dict[str, Any]]:
    lineage = read_jsonl(reviewed)
    frozen_rows = read_jsonl(frozen)
    frozen_by_id = {row["candidate_id"]: row for row in frozen_rows}
    if len(frozen_by_id) != len(frozen_rows):
        raise ValueError("Frozen source contains duplicate candidate_id.")
    eligible = [
        row for row in lineage
        if row.get("generation_status") == "generated"
        and row.get("edit_qc_status") == "pass"
        and row.get("object_identity_preservation_status") == "pass"
    ]
    if len(eligible) != 91:
        raise ValueError(f"Expected 91 Natural Edit QC-valid candidates; found {len(eligible)}.")
    output = []
    for row in eligible:
        cid = row["candidate_id"]
        source = frozen_by_id.get(cid)
        if source is None:
            raise ValueError(f"{cid}: missing from frozen source")
        original = project_path(project_root, row["original_image_path"])
        edited = artifact_path(project_root, reviewed, row["natural_edited_path"])
        if not original.is_file() or not edited.is_file():
            raise FileNotFoundError(f"{cid}: missing Natural image: {original} or {edited}")
        joined = {
            "candidate_id": cid,
            "frozen_order": source["frozen_order"],
            "source_image_id": str(source["source_image_id"]),
            "source_object_a_id": str(source["source_object_a_id"]),
            "source_object_b_id": str(source["source_object_b_id"]),
            "source_pair_key": source["source_pair_key"],
            "object_a_label": source["object_a_label"],
            "object_b_label": source["object_b_label"],
            "original_color_a": row["original_color_a"],
            "original_color_b": row["original_color_b"],
            "edited_color_a": row["target_color_a"],
            "edited_color_b": row["target_color_b"],
            "natural_original_path": as_project_relative(project_root, original),
            "natural_original_sha256": row["source_image_sha256"],
            "natural_edited_path": as_project_relative(project_root, edited),
            "natural_edited_sha256": row["natural_edited_sha256"],
            "preview_path": source.get("preview_path"),
            "contact_sheet_path": as_project_relative(
                project_root, artifact_path(project_root, reviewed, row["contact_sheet_path"])
            ),
            "frozen_source_manifest": frozen.as_posix(),
            "frozen_source_manifest_sha256": sha256(frozen),
            "reviewed_lineage_manifest": reviewed.as_posix(),
            "reviewed_lineage_manifest_sha256": sha256(reviewed),
            "reviewed_lineage_row_sha256": canonical_row_sha256(row),
        }
        if joined["edited_color_a"] != joined["original_color_b"] or joined["edited_color_b"] != joined["original_color_a"]:
            raise ValueError(f"{cid}: edited colors do not encode the frozen A/B swap")
        output.append(joined)
    output.sort(key=lambda row: row["frozen_order"])
    for primary_domain_order, row in enumerate(output, start=1):
        row["primary_domain_order"] = primary_domain_order
    if len({row["candidate_id"] for row in output}) != 91:
        raise ValueError("Primary candidate derivation contains duplicate IDs.")
    return output


def scene_index(samples_path: Path, wanted_image_ids: set[str]) -> dict[str, dict[str, Any]]:
    payload = read_json(samples_path)
    samples = payload.get("samples")
    if not isinstance(samples, list):
        raise ValueError("GQA export must contain a top-level samples list.")
    output = {}
    for sample in samples:
        image_id = Path(sample.get("filepath", "")).stem
        if image_id in wanted_image_ids:
            output[image_id] = sample
    missing = sorted(wanted_image_ids - set(output))
    if missing:
        raise ValueError(f"Missing GQA scene metadata for image IDs: {missing}")
    return output


def norm_label(value: Any) -> str:
    return " ".join(str(value or "").strip().lower().split())


def propose_reference(
    candidate: dict[str, Any], side: str, sample: dict[str, Any], config: dict[str, Any]
) -> dict[str, Any]:
    detections = sample.get("detections", {}).get("detections", [])
    source_index = int(candidate[f"source_object_{side}_id"])
    if source_index < 0 or source_index >= len(detections):
        raise ValueError(f"{candidate['candidate_id']}: object {side} index is out of range")
    target = detections[source_index]
    expected_label = norm_label(candidate[f"object_{side}_label"])
    actual_label = norm_label(target.get("label"))
    if expected_label != actual_label:
        raise ValueError(
            f"{candidate['candidate_id']}: object {side} label mismatch: {expected_label!r} != {actual_label!r}"
        )
    label_count = sum(norm_label(item.get("label")) == actual_label for item in detections)
    bare_unique = label_count == 1
    return {
        "proposed_reference": f"the {actual_label}",
        "bare_noun_unique": bare_unique,
        "metadata_reference_unique": bare_unique,
        "ambiguity_flag": not bare_unique,
    }


def desired_color_positions(primary_domain_order: int, config: dict[str, Any]) -> dict[str, int]:
    rank = primary_domain_order - 1
    offsets = config["option_ordering"]["color_position_offsets"]
    return {name: (rank + int(offset)) % 4 for name, offset in offsets.items()}


def color_option_order(
    original_color: str,
    edited_color: str,
    original_position: int,
    edited_position: int,
    canonical_colors: list[str],
) -> list[str]:
    if original_color == edited_color or original_position == edited_position:
        raise ValueError("Color schedule requires distinct colors and positions.")
    options: list[str | None] = [None] * 4
    options[original_position] = original_color
    options[edited_position] = edited_color
    remaining = [color for color in canonical_colors if color not in {original_color, edited_color}]
    for index in range(4):
        if options[index] is None:
            options[index] = remaining.pop(0)
    return [str(value) for value in options]


def binding_option_order(primary_domain_order: int, semantics: list[str]) -> list[str]:
    return list(semantics) if primary_domain_order % 2 == 1 else list(reversed(semantics))


def make_options(values: list[str], display: dict[str, str] | None = None) -> list[dict[str, str]]:
    alphabet = "ABCDEFGHIJKLMNOPQRSTUVWXYZ"
    return [
        {"option_id": alphabet[index], "semantic_value": value, "display_text": (display or {}).get(value, value)}
        for index, value in enumerate(values)
    ]


def correct_option_id(options: list[dict[str, str]], semantic_answer: str) -> str:
    matches = [option["option_id"] for option in options if option["semantic_value"] == semantic_answer]
    if len(matches) != 1:
        raise ValueError(f"Expected one correct option for {semantic_answer!r}; found {matches}")
    return matches[0]


def generate(args: argparse.Namespace) -> None:
    config = read_json(args.config)
    candidates = derive_primary_candidates(args.reviewed_lineage, args.frozen_source, args.project_root)
    scenes = scene_index(args.gqa_samples, {row["source_image_id"] for row in candidates})
    colors = config["canonical_colors"]
    semantics = config["binding_semantics"]
    items: list[dict[str, Any]] = []
    candidate_status = []
    for candidate in candidates:
        sample = scenes[candidate["source_image_id"]]
        proposal_a = propose_reference(candidate, "a", sample, config)
        proposal_b = propose_reference(candidate, "b", sample, config)
        if not proposal_a["bare_noun_unique"] or not proposal_b["bare_noun_unique"]:
            raise ValueError(f"{candidate['candidate_id']}: source labels are not metadata-unique")
        ref_a = proposal_a["proposed_reference"].strip()
        ref_b = proposal_b["proposed_reference"].strip()
        candidate_status.append({
            "schema_version": "1.2",
            "candidate_id": candidate["candidate_id"],
            "frozen_order": candidate["frozen_order"],
            "primary_domain_order": candidate["primary_domain_order"],
            "natural_edit_qc_valid": True,
            "reference_derivation": "metadata_unique_non_color_bare_label_v1.2",
            "object_a_reference": ref_a,
            "object_b_reference": ref_b,
            "primary_question_construction_included": True,
            "primary_exclusion_reasons": [],
        })
        positions = desired_color_positions(candidate["primary_domain_order"], config)
        color_orders = {
            "object_a": color_option_order(
                candidate["original_color_a"], candidate["edited_color_a"],
                positions["object_a_original"], positions["object_a_edited"], colors,
            ),
            "object_b": color_option_order(
                candidate["original_color_b"], candidate["edited_color_b"],
                positions["object_b_original"], positions["object_b_edited"], colors,
            ),
        }
        binding_order = binding_option_order(candidate["primary_domain_order"], semantics)
        binding_display = {
            "Original Binding": f"The color of {ref_a} is {candidate['original_color_a']}; the color of {ref_b} is {candidate['original_color_b']}.",
            "Swapped Binding": f"The color of {ref_a} is {candidate['edited_color_a']}; the color of {ref_b} is {candidate['edited_color_b']}.",
        }
        for state in ("original", "edited"):
            image_path = candidate[f"natural_{state}_path"]
            image_sha = candidate[f"natural_{state}_sha256"]
            state_gold = {
                "object_a": candidate[f"{state}_color_a"],
                "object_b": candidate[f"{state}_color_b"],
                "binding": "Original Binding" if state == "original" else "Swapped Binding",
            }
            for side, reference_text in (("object_a", ref_a), ("object_b", ref_b)):
                options = make_options(color_orders[side])
                answer = state_gold[side]
                items.append({
                    "schema_version": "1.2",
                    "query_id": f"{candidate['candidate_id']}__natural_{state}__{side}_color",
                    "sample_id": candidate["candidate_id"],
                    "candidate_id": candidate["candidate_id"],
                    "frozen_order": candidate["frozen_order"],
                    "primary_domain_order": candidate["primary_domain_order"],
                    "image_id": f"{candidate['candidate_id']}__natural_{state}",
                    "image_path": image_path,
                    "image_sha256": image_sha,
                    "track": "natural",
                    "state": state,
                    "analysis_role": "primary",
                    "task_type": f"{side}_color",
                    "prompt_version": config["prompt_version"],
                    "prompt_text": f"Select the color of {reference_text}.",
                    "object_reference": reference_text,
                    "options": options,
                    "option_order": [option["semantic_value"] for option in options],
                    "correct_option_id": correct_option_id(options, answer),
                    "correct_semantic_answer": answer,
                    "response_alphabet": [option["option_id"] for option in options],
                    "balancing_rule": config["option_ordering"]["algorithm"],
                    "balancing_seed": config["option_ordering"]["seed"],
                    "gold_source": config["gold_source"],
                    "stateless": True,
                    "conversation_history": [],
                })
            binding_options = make_options(binding_order, binding_display)
            binding_answer = state_gold["binding"]
            items.append({
                "schema_version": "1.2",
                "query_id": f"{candidate['candidate_id']}__natural_{state}__binding_choice",
                "sample_id": candidate["candidate_id"],
                "candidate_id": candidate["candidate_id"],
                "frozen_order": candidate["frozen_order"],
                "primary_domain_order": candidate["primary_domain_order"],
                "image_id": f"{candidate['candidate_id']}__natural_{state}",
                "image_path": image_path,
                "image_sha256": image_sha,
                "track": "natural",
                "state": state,
                "analysis_role": "primary",
                "task_type": "binding_choice",
                "prompt_version": config["prompt_version"],
                "prompt_text": "Which object-color correspondence is shown? Select the best answer.",
                "object_references": {"object_a": ref_a, "object_b": ref_b},
                "options": binding_options,
                "option_order": [option["semantic_value"] for option in binding_options],
                "correct_option_id": correct_option_id(binding_options, binding_answer),
                "correct_semantic_answer": binding_answer,
                "response_alphabet": [option["option_id"] for option in binding_options],
                "balancing_rule": config["option_ordering"]["algorithm"],
                "balancing_seed": config["option_ordering"]["seed"],
                "gold_source": config["gold_source"],
                "stateless": True,
                "conversation_history": [],
            })
    if args.preflight:
        print(json.dumps({
            "candidate_domain": len(candidates),
            "metadata_unique_reference_count": len(candidate_status),
            "item_count": len(items),
        }, indent=2))
        return
    atomic_jsonl(args.output_manifest, items)
    atomic_jsonl(args.candidate_status_output, candidate_status)
    atomic_json(args.summary_output, {
        "schema_version": "1.2",
        "candidate_domain": len(candidates),
        "primary_question_candidate_count": sum(row["primary_question_construction_included"] for row in candidate_status),
        "reference_attrition_count": sum(not row["primary_question_construction_included"] for row in candidate_status),
        "item_count": len(items),
        "items_per_candidate": 6,
        "output_manifest": args.output_manifest.as_posix(),
        "output_manifest_sha256": sha256(args.output_manifest),
        "config": args.config.as_posix(),
        "config_sha256": sha256(args.config),
        "generated_at": datetime.now(timezone.utc).isoformat(),
    })
    print(json.dumps({"candidate_count": len(items) // 6, "item_count": len(items)}, indent=2))


def parser() -> argparse.ArgumentParser:
    root = argparse.ArgumentParser(description=__doc__)
    sub = root.add_subparsers(dest="command", required=True)
    generate_parser = sub.add_parser("generate")
    generate_parser.add_argument("--reviewed-lineage", type=Path, required=True)
    generate_parser.add_argument("--frozen-source", type=Path, required=True)
    generate_parser.add_argument("--gqa-samples", type=Path, required=True)
    generate_parser.add_argument("--config", type=Path, required=True)
    generate_parser.add_argument("--project-root", type=Path, required=True)
    generate_parser.add_argument("--output-manifest", type=Path, required=True)
    generate_parser.add_argument("--candidate-status-output", type=Path, required=True)
    generate_parser.add_argument("--summary-output", type=Path, required=True)
    generate_parser.add_argument("--preflight", action="store_true")
    generate_parser.set_defaults(func=generate)
    return root


def main() -> None:
    args = parser().parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
