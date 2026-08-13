#!/usr/bin/env python3
"""Validate the v1.2 Natural Primary Q/A contract before inference."""

from __future__ import annotations

import argparse
import json
import re
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

from build_primary_qa_v1_2 import (
    binding_option_order,
    color_option_order,
    derive_primary_candidates,
    desired_color_positions,
    project_path,
    propose_reference,
    read_json,
    read_jsonl,
    scene_index,
    sha256,
)


TASKS = {"object_a_color", "object_b_color", "binding_choice"}
STATES = {"original", "edited"}
FORBIDDEN_KEY_PARTS = ("bbox", "mask", "overlay", "source_object", "source_annotation", "model_output", "raw_response")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--reviewed-lineage", type=Path, required=True)
    parser.add_argument("--frozen-source", type=Path, required=True)
    parser.add_argument("--gqa-samples", type=Path, required=True)
    parser.add_argument("--candidate-status", type=Path, required=True)
    parser.add_argument("--query-manifest", type=Path, required=True)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--project-root", type=Path, required=True)
    parser.add_argument("--summary-output", type=Path)
    return parser.parse_args()


def add(errors: list[str], condition: bool, message: str) -> None:
    if not condition:
        errors.append(message)


def nested_forbidden_keys(value: Any, prefix: str = "") -> list[str]:
    found = []
    if isinstance(value, dict):
        for key, child in value.items():
            path = f"{prefix}.{key}" if prefix else key
            normalized = key.lower()
            if any(part in normalized for part in FORBIDDEN_KEY_PARTS):
                found.append(path)
            found.extend(nested_forbidden_keys(child, path))
    elif isinstance(value, list):
        for index, child in enumerate(value):
            found.extend(nested_forbidden_keys(child, f"{prefix}[{index}]"))
    return found


def option_values(item: dict[str, Any]) -> list[str]:
    return [option.get("semantic_value") for option in item.get("options", [])]


def option_ids(item: dict[str, Any]) -> list[str]:
    return [option.get("option_id") for option in item.get("options", [])]


def validate_option_contract(
    item: dict[str, Any], expected_values: list[str], expected_answer: str, errors: list[str]
) -> None:
    qid = item.get("query_id", "<missing-query-id>")
    values = option_values(item)
    ids = option_ids(item)
    add(errors, values == expected_values, f"{qid}: option order differs from frozen schedule")
    add(errors, len(values) == len(set(values)), f"{qid}: duplicate semantic options")
    add(errors, len(ids) == len(set(ids)), f"{qid}: duplicate option IDs")
    add(errors, item.get("option_order") == values, f"{qid}: option_order does not match options")
    add(errors, item.get("response_alphabet") == ids, f"{qid}: response alphabet does not match option IDs")
    correct = [option for option in item.get("options", []) if option.get("semantic_value") == expected_answer]
    add(errors, len(correct) == 1, f"{qid}: expected exactly one semantic gold option")
    if len(correct) == 1:
        add(errors, item.get("correct_option_id") == correct[0].get("option_id"), f"{qid}: wrong correct_option_id")
    add(errors, item.get("correct_semantic_answer") == expected_answer, f"{qid}: wrong semantic gold")


def main() -> None:
    args = parse_args()
    config = read_json(args.config)
    candidates = derive_primary_candidates(args.reviewed_lineage, args.frozen_source, args.project_root)
    scenes = scene_index(args.gqa_samples, {row["source_image_id"] for row in candidates})
    statuses = read_jsonl(args.candidate_status)
    items = read_jsonl(args.query_manifest)
    errors: list[str] = []

    candidate_ids = [row["candidate_id"] for row in candidates]
    status_ids = [row.get("candidate_id") for row in statuses]
    add(errors, len(candidates) == 91, f"expected candidate domain 91; found {len(candidates)}")
    add(errors, status_ids == candidate_ids, "candidate-status membership/order differs from candidate domain")
    add(errors, len(set(status_ids)) == len(status_ids), "duplicate candidate status rows")

    candidate_by_id = {row["candidate_id"]: row for row in candidates}
    status_by_id = {row["candidate_id"]: row for row in statuses}
    expected_included = candidate_ids
    actual_included = [
        row["candidate_id"] for row in statuses if row.get("primary_question_construction_included") is True
    ]
    add(errors, actual_included == expected_included, "candidate-status Primary inclusion differs from the 91-candidate domain")

    item_ids = [row.get("query_id") for row in items]
    add(errors, None not in item_ids, "query manifest contains missing query_id")
    add(errors, len(item_ids) == len(set(item_ids)), "query manifest contains duplicate query_id")
    grouped: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for item in items:
        grouped[str(item.get("candidate_id"))].append(item)
    add(errors, set(grouped) == set(expected_included), "query candidate membership differs from the 91-candidate domain")
    add(errors, len(items) == len(expected_included) * 6, "query count is not six per included candidate")

    colors = config["canonical_colors"]
    bindings = config["binding_semantics"]
    expected_inventory = {(state, task) for state in STATES for task in TASKS}
    correct_position_counts: dict[str, Counter[int]] = defaultdict(Counter)

    # The balance guarantee is frozen over the full 91-candidate Natural-valid domain.
    schedule_counts: dict[str, Counter[int]] = defaultdict(Counter)
    for candidate in candidates:
        positions = desired_color_positions(candidate["primary_domain_order"], config)
        for family, position in positions.items():
            schedule_counts[family][position] += 1
        binding_order = binding_option_order(candidate["primary_domain_order"], bindings)
        schedule_counts["binding_semantic_order"][0 if binding_order == bindings else 1] += 1
    maximum_difference = config["option_ordering"]["maximum_position_count_difference"]
    for family, counts in schedule_counts.items():
        add(errors, max(counts.values()) - min(counts.values()) <= maximum_difference,
            f"{family}: frozen 91-candidate schedule is imbalanced: {dict(counts)}")

    for cid in expected_included:
        candidate = candidate_by_id[cid]
        status = status_by_id[cid]
        sample = scenes[candidate["source_image_id"]]
        proposal_a = propose_reference(candidate, "a", sample, config)
        proposal_b = propose_reference(candidate, "b", sample, config)
        rows = grouped.get(cid, [])
        add(errors, len(rows) == 6, f"{cid}: expected six items; found {len(rows)}")
        inventory = {(row.get("state"), row.get("task_type")) for row in rows}
        add(errors, inventory == expected_inventory, f"{cid}: incorrect task inventory: {inventory}")
        add(errors, status.get("natural_edit_qc_valid") is True, f"{cid}: Natural Edit QC validity not preserved")
        add(errors, proposal_a["bare_noun_unique"], f"{cid}: object A source label is not metadata-unique")
        add(errors, proposal_b["bare_noun_unique"], f"{cid}: object B source label is not metadata-unique")
        ref_a = proposal_a["proposed_reference"].strip()
        ref_b = proposal_b["proposed_reference"].strip()
        add(errors, status.get("reference_derivation") == "metadata_unique_non_color_bare_label_v1.2",
            f"{cid}: wrong reference derivation provenance")
        add(errors, status.get("object_a_reference") == ref_a, f"{cid}: candidate-status A reference mismatch")
        add(errors, status.get("object_b_reference") == ref_b, f"{cid}: candidate-status B reference mismatch")
        add(errors, bool(ref_a) and bool(ref_b), f"{cid}: empty object reference")
        for ref_name, ref_value in (("A", ref_a), ("B", ref_b)):
            normalized = ref_value.lower()
            add(errors, normalized not in {"object a", "object b", "the object a", "the object b"},
                f"{cid}: internal object name exposed in reference {ref_name}")
            leaked = [color for color in colors if re.search(rf"\b{re.escape(color)}\b", normalized)]
            add(errors, not leaked, f"{cid}: color leaks through reference {ref_name}: {leaked}")

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
        binding_order = binding_option_order(candidate["primary_domain_order"], bindings)
        by_key = {(row["state"], row["task_type"]): row for row in rows}
        for state in ("original", "edited"):
            expected_path = candidate[f"natural_{state}_path"]
            expected_sha = candidate[f"natural_{state}_sha256"]
            for task in TASKS:
                item = by_key.get((state, task))
                if item is None:
                    continue
                qid = item.get("query_id", f"{cid}:{state}:{task}")
                prompt_lower = str(item.get("prompt_text", "")).lower()
                prompt_forbidden = [token for token in ("object a", "object b", "bbox", "bounding box", "mask", "overlay") if token in prompt_lower]
                add(errors, not prompt_forbidden, f"{qid}: forbidden prompt content: {prompt_forbidden}")
                add(errors, cid.lower() not in prompt_lower and candidate["source_image_id"] not in prompt_lower,
                    f"{qid}: internal candidate or source image ID exposed in prompt")
                add(errors, item.get("frozen_order") == candidate["frozen_order"], f"{qid}: wrong frozen_order")
                add(errors, item.get("primary_domain_order") == candidate["primary_domain_order"],
                    f"{qid}: wrong primary_domain_order")
                add(errors, item.get("track") == "natural", f"{qid}: track is not natural")
                add(errors, item.get("analysis_role") == "primary", f"{qid}: role is not primary")
                add(errors, item.get("image_path") == expected_path, f"{qid}: wrong image path")
                add(errors, item.get("image_sha256") == expected_sha, f"{qid}: wrong image checksum")
                image = project_path(args.project_root, str(item.get("image_path", "")))
                add(errors, image.is_file(), f"{qid}: image file does not exist")
                if image.is_file():
                    add(errors, sha256(image) == expected_sha, f"{qid}: image file checksum mismatch")
                add(errors, item.get("stateless") is True, f"{qid}: stateless flag is not true")
                add(errors, item.get("conversation_history") == [], f"{qid}: conversation history is not empty")
                forbidden = nested_forbidden_keys(item)
                add(errors, not forbidden, f"{qid}: forbidden VLM-input/provenance fields: {forbidden}")
                add(errors, item.get("gold_source") == config["gold_source"], f"{qid}: invalid gold provenance")
                add(errors, item.get("balancing_rule") == config["option_ordering"]["algorithm"],
                    f"{qid}: wrong balancing rule")
                add(errors, item.get("balancing_seed") == config["option_ordering"]["seed"],
                    f"{qid}: wrong balancing seed")
                if task in {"object_a_color", "object_b_color"}:
                    side = task.removesuffix("_color")
                    expected_answer = candidate[f"{state}_color_{'a' if side == 'object_a' else 'b'}"]
                    validate_option_contract(item, color_orders[side], expected_answer, errors)
                    add(errors, set(option_values(item)) == set(colors), f"{qid}: color semantic set is not canonical")
                    expected_ref = ref_a if side == "object_a" else ref_b
                    add(errors, item.get("object_reference") == expected_ref, f"{qid}: wrong object reference")
                else:
                    expected_answer = "Original Binding" if state == "original" else "Swapped Binding"
                    validate_option_contract(item, binding_order, expected_answer, errors)
                    add(errors, set(option_values(item)) == set(bindings), f"{qid}: binding semantic set is invalid")
                    add(errors, item.get("object_references") == {"object_a": ref_a, "object_b": ref_b},
                        f"{qid}: wrong binding references")
                if item.get("correct_option_id") in item.get("response_alphabet", []):
                    correct_position_counts[f"{state}:{task}"][item["response_alphabet"].index(item["correct_option_id"])] += 1

        for task in TASKS:
            original = by_key.get(("original", task))
            edited = by_key.get(("edited", task))
            if original is None or edited is None:
                continue
            add(errors, original.get("prompt_text") == edited.get("prompt_text"), f"{cid}:{task}: counterpart wording differs")
            add(errors, original.get("options") == edited.get("options"), f"{cid}:{task}: counterpart options/order differ")
            if task == "binding_choice":
                add(errors,
                    original.get("correct_semantic_answer") == "Original Binding"
                    and edited.get("correct_semantic_answer") == "Swapped Binding",
                    f"{cid}: binding semantic answer does not flip",
                )
            else:
                add(errors, original.get("object_reference") == edited.get("object_reference"),
                    f"{cid}:{task}: counterpart reference differs")

    summary = {
        "schema_version": "1.2",
        "status": "pass" if not errors else "fail",
        "expected_candidate_domain": len(candidates),
        "metadata_unique_reference_count": len(expected_included),
        "primary_candidate_count": len(grouped),
        "query_count": len(items),
        "expected_query_count": len(expected_included) * 6,
        "frozen_schedule_position_counts": {key: dict(value) for key, value in schedule_counts.items()},
        "included_item_correct_position_counts": {key: dict(value) for key, value in correct_position_counts.items()},
        "reference_policy": "deterministic_metadata_unique_non_color_label",
        "validation_error_count": len(errors),
        "validation_errors": errors,
    }
    if args.summary_output:
        if args.summary_output.exists():
            raise FileExistsError(f"Refusing to overwrite: {args.summary_output}")
        args.summary_output.parent.mkdir(parents=True, exist_ok=True)
        args.summary_output.write_text(json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    if errors:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
