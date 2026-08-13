#!/usr/bin/env python3
"""Compute core and atomic-qualified metrics for color-binding feasibility."""

from __future__ import annotations

import argparse
import csv
import json
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any


CORE = ("binding_true", "binding_false", "image_conflict", "edit_control")
ATOMIC_GROUPS = {
    "original_visibility": ("original_a_visible", "original_b_visible"),
    "edited_visibility": ("edited_a_visible", "edited_b_visible"),
    "original_color": (
        "original_a_source_color",
        "original_a_target_color",
        "original_b_source_color",
        "original_b_target_color",
    ),
    "edited_color": (
        "edited_a_source_color",
        "edited_a_target_color",
        "edited_b_source_color",
        "edited_b_target_color",
    ),
    "color_presence": ("original_color_presence", "edited_color_presence"),
}


def parse_args() -> argparse.Namespace:
    project_root = Path(__file__).resolve().parents[2]
    base = project_root / "experiments/attribute_binding/feasibility_n5_v1"
    parser = argparse.ArgumentParser(description="Compute binding metrics.")
    parser.add_argument(
        "--predictions",
        type=Path,
        default=base / "predictions/qwen2_5_vl_7b.jsonl",
    )
    parser.add_argument(
        "--metrics-output",
        type=Path,
        default=base / "metrics/qwen2_5_vl_7b_metrics.json",
    )
    parser.add_argument(
        "--patterns-output",
        type=Path,
        default=base / "metrics/qwen2_5_vl_7b_sample_patterns.csv",
    )
    return parser.parse_args()


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    with path.open("r", encoding="utf-8") as input_file:
        return [json.loads(line) for line in input_file]


def ratio(numerator: int, denominator: int) -> dict[str, Any]:
    return {
        "numerator": numerator,
        "denominator": denominator,
        "rate": numerator / denominator if denominator else None,
    }


def answered_correctly(row: dict[str, Any]) -> bool:
    return bool(row.get("is_valid")) and row.get("is_correct") is True


def main() -> None:
    args = parse_args()
    rows = read_jsonl(args.predictions)
    by_condition: dict[str, list[dict[str, Any]]] = defaultdict(list)
    by_sample: dict[str, dict[str, dict[str, Any]]] = defaultdict(dict)
    for row in rows:
        by_condition[row["condition"]].append(row)
        by_sample[row["sample_id"]][row["condition"]] = row

    condition_metrics = {}
    for name, condition_rows in sorted(by_condition.items()):
        answers = Counter(row.get("parsed_answer") or "Invalid" for row in condition_rows)
        correct = sum(answered_correctly(row) for row in condition_rows)
        condition_metrics[name] = {
            "n": len(condition_rows),
            "correct": correct,
            "accuracy": correct / len(condition_rows),
            "yes": answers["Yes"],
            "no": answers["No"],
            "invalid": answers["Invalid"],
        }

    pattern_rows = []
    original_passes = 0
    edited_passes = 0
    full_passes = 0
    incremental_failures = 0
    atomic_all_passes = 0
    qualified_original_passes = 0
    qualified_edited_passes = 0
    qualified_full_passes = 0

    for sample_id in sorted(by_sample):
        conditions = by_sample[sample_id]
        required = set(CORE)
        for names in ATOMIC_GROUPS.values():
            required.update(names)
        missing = sorted(required - set(conditions))
        if missing:
            raise ValueError(f"{sample_id} missing conditions: {missing}")

        original_pass = all(answered_correctly(conditions[name]) for name in CORE[:2])
        edited_pass = all(answered_correctly(conditions[name]) for name in CORE[2:])
        full_pass = original_pass and edited_pass
        incremental_failure = original_pass and not edited_pass
        atomic_passes = {
            group: all(answered_correctly(conditions[name]) for name in names)
            for group, names in ATOMIC_GROUPS.items()
        }
        atomic_all = all(atomic_passes.values())
        qualified_original = original_pass and atomic_passes["original_visibility"] and atomic_passes["original_color"] and atomic_passes["color_presence"]
        qualified_edited = edited_pass and atomic_passes["edited_visibility"] and atomic_passes["edited_color"] and atomic_passes["color_presence"]
        qualified_full = full_pass and atomic_all

        original_passes += original_pass
        edited_passes += edited_pass
        full_passes += full_pass
        incremental_failures += incremental_failure
        atomic_all_passes += atomic_all
        qualified_original_passes += qualified_original
        qualified_edited_passes += qualified_edited
        qualified_full_passes += qualified_full

        first = conditions["binding_true"]
        pattern_rows.append(
            {
                "sample_id": sample_id,
                "source_candidate_id": first["source_candidate_id"],
                "color_pair": f"{first['object_a_original_color']}_{first['object_b_original_color']}",
                "binding_true": conditions["binding_true"].get("parsed_answer"),
                "binding_false": conditions["binding_false"].get("parsed_answer"),
                "image_conflict": conditions["image_conflict"].get("parsed_answer"),
                "edit_control": conditions["edit_control"].get("parsed_answer"),
                "original_binding_consistency": original_pass,
                "edited_binding_consistency": edited_pass,
                "full_binding_counterfactual_consistency": full_pass,
                "incremental_edit_failure": incremental_failure,
                **{f"atomic_{key}_pass": value for key, value in atomic_passes.items()},
                "atomic_all_pass": atomic_all,
                "control_qualified_full_consistency": qualified_full,
            }
        )

    n_samples = len(by_sample)
    true_rows = [row for row in rows if row["expected_answer"] == "Yes"]
    false_rows = [row for row in rows if row["expected_answer"] == "No"]
    metrics = {
        "schema_version": "1.0",
        "model_id": rows[0].get("model_id"),
        "n_samples": n_samples,
        "n_predictions": len(rows),
        "condition_metrics": condition_metrics,
        "core_sample_metrics": {
            "original_binding_consistency": ratio(original_passes, n_samples),
            "edited_binding_consistency": ratio(edited_passes, n_samples),
            "full_binding_counterfactual_consistency": ratio(full_passes, n_samples),
            "incremental_edit_failure": ratio(incremental_failures, original_passes),
        },
        "atomic_qualification": {
            "all_atomic_controls_pass": ratio(atomic_all_passes, n_samples),
            "qualified_original_consistency": ratio(qualified_original_passes, n_samples),
            "qualified_edited_consistency": ratio(qualified_edited_passes, n_samples),
            "qualified_full_consistency": ratio(qualified_full_passes, n_samples),
        },
        "answer_polarity_errors": {
            "true_condition_error": ratio(sum(not answered_correctly(row) for row in true_rows), len(true_rows)),
            "false_condition_error": ratio(sum(not answered_correctly(row) for row in false_rows), len(false_rows)),
            "invalid": ratio(sum(not row.get("is_valid") for row in rows), len(rows)),
        },
    }

    args.metrics_output.parent.mkdir(parents=True, exist_ok=True)
    args.patterns_output.parent.mkdir(parents=True, exist_ok=True)
    args.metrics_output.write_text(
        json.dumps(metrics, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    with args.patterns_output.open("w", encoding="utf-8", newline="") as output_file:
        writer = csv.DictWriter(output_file, fieldnames=list(pattern_rows[0]))
        writer.writeheader()
        writer.writerows(pattern_rows)
    print(json.dumps(metrics["core_sample_metrics"], ensure_ascii=False, indent=2))
    print(json.dumps(metrics["atomic_qualification"], ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
