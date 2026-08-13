#!/usr/bin/env python3
"""Compute condition and sample-level metrics for the relation pilot."""

from __future__ import annotations

import argparse
import csv
import json
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any


CORE_CONDITIONS = (
    "baseline_true",
    "baseline_false",
    "image_conflict",
    "flip_control",
)


def parse_args() -> argparse.Namespace:
    project_root = Path(__file__).resolve().parents[2]
    parser = argparse.ArgumentParser(description="Compute pilot metrics from prediction JSONL.")
    parser.add_argument(
        "--predictions",
        type=Path,
        default=project_root
        / "experiments/spatial_left_right/positive_control_v1/predictions/qwen2_5_vl_7b.jsonl",
    )
    parser.add_argument(
        "--metrics-output",
        type=Path,
        default=project_root
        / "experiments/spatial_left_right/positive_control_v1/metrics/qwen2_5_vl_7b_metrics.json",
    )
    parser.add_argument(
        "--patterns-output",
        type=Path,
        default=project_root
        / "experiments/spatial_left_right/positive_control_v1/metrics/qwen2_5_vl_7b_sample_patterns.csv",
    )
    return parser.parse_args()


def load_predictions(path: Path) -> list[dict[str, Any]]:
    with path.open("r", encoding="utf-8") as input_file:
        return [json.loads(line) for line in input_file]


def display_id(row: dict[str, Any]) -> str:
    return row.get("review_candidate_id") or row["sample_id"]


def ratio(numerator: int, denominator: int) -> dict[str, Any]:
    return {
        "numerator": numerator,
        "denominator": denominator,
        "rate": numerator / denominator if denominator else None,
    }


def compute_metrics(rows: list[dict[str, Any]]) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    if not rows:
        raise ValueError("Prediction file is empty.")

    by_condition: dict[str, list[dict[str, Any]]] = defaultdict(list)
    by_sample: dict[str, dict[str, dict[str, Any]]] = defaultdict(dict)
    internal_ids: dict[str, str] = {}
    for row in rows:
        by_condition[row["condition"]].append(row)
        review_id = display_id(row)
        by_sample[review_id][row["condition"]] = row
        internal_ids[review_id] = row["sample_id"]

    condition_metrics: dict[str, Any] = {}
    for condition, condition_rows in sorted(by_condition.items()):
        valid_rows = [row for row in condition_rows if row.get("is_valid")]
        correct = sum(bool(row.get("is_correct")) for row in condition_rows)
        answers = Counter(row.get("parsed_answer") or "Invalid" for row in condition_rows)
        condition_metrics[condition] = {
            "n": len(condition_rows),
            "correct": correct,
            "accuracy": correct / len(condition_rows),
            "yes": answers["Yes"],
            "no": answers["No"],
            "invalid": len(condition_rows) - len(valid_rows),
        }

    pattern_rows: list[dict[str, Any]] = []
    original_pass_count = 0
    flip_pass_count = 0
    fcc_count = 0
    incremental_flip_fail_count = 0

    for review_id in sorted(by_sample):
        conditions = by_sample[review_id]
        missing = [condition for condition in CORE_CONDITIONS if condition not in conditions]
        if missing:
            raise ValueError(f"{review_id} is missing core conditions: {missing}")

        original_pass = (
            conditions["baseline_true"].get("parsed_answer") == "Yes"
            and conditions["baseline_false"].get("parsed_answer") == "No"
        )
        flip_pass = (
            conditions["image_conflict"].get("parsed_answer") == "No"
            and conditions["flip_control"].get("parsed_answer") == "Yes"
        )
        fcc = original_pass and flip_pass
        incremental_flip_fail = original_pass and not flip_pass
        original_pass_count += original_pass
        flip_pass_count += flip_pass
        fcc_count += fcc
        incremental_flip_fail_count += incremental_flip_fail

        first_row = next(iter(conditions.values()))
        pattern_rows.append(
            {
                "review_id": review_id,
                "internal_sample_id": internal_ids[review_id],
                "source_label": first_row.get("source_label"),
                "target_label": first_row.get("target_label"),
                "relation_orientation": first_row.get("relation_orientation"),
                "baseline_true": conditions["baseline_true"].get("parsed_answer"),
                "baseline_false": conditions["baseline_false"].get("parsed_answer"),
                "image_conflict": conditions["image_conflict"].get("parsed_answer"),
                "flip_control": conditions["flip_control"].get("parsed_answer"),
                "irrelevant_true": conditions.get("irrelevant_true", {}).get("parsed_answer"),
                "original_relation_consistency": original_pass,
                "flip_relation_consistency": flip_pass,
                "full_counterfactual_consistency": fcc,
                "incremental_flip_failure": incremental_flip_fail,
            }
        )

    n_samples = len(by_sample)
    true_relation_rows = by_condition["baseline_true"] + by_condition["flip_control"]
    false_relation_rows = by_condition["baseline_false"] + by_condition["image_conflict"]
    false_negatives = sum(row.get("parsed_answer") != "Yes" for row in true_relation_rows)
    false_positives = sum(row.get("parsed_answer") != "No" for row in false_relation_rows)

    metrics = {
        "schema_version": "1.0",
        "model_id": rows[0].get("model_id"),
        "n_samples": n_samples,
        "n_predictions": len(rows),
        "condition_metrics": condition_metrics,
        "sample_metrics": {
            "original_relation_consistency": ratio(original_pass_count, n_samples),
            "flip_relation_consistency": ratio(flip_pass_count, n_samples),
            "full_counterfactual_consistency": ratio(fcc_count, n_samples),
            "incremental_flip_failure": ratio(
                incremental_flip_fail_count, original_pass_count
            ),
        },
        "relation_polarity_errors": {
            "false_negative_rate_on_true_relation_conditions": ratio(
                false_negatives, len(true_relation_rows)
            ),
            "false_positive_rate_on_false_relation_conditions": ratio(
                false_positives, len(false_relation_rows)
            ),
        },
        "definitions": {
            "original_relation_consistency": "baseline_true=Yes AND baseline_false=No",
            "flip_relation_consistency": "image_conflict=No AND flip_control=Yes",
            "full_counterfactual_consistency": "original_relation_consistency AND flip_relation_consistency",
            "incremental_flip_failure": "original_relation_consistency AND NOT flip_relation_consistency; denominator is original_relation_consistency passes",
        },
    }
    return metrics, pattern_rows


def write_outputs(
    metrics: dict[str, Any],
    pattern_rows: list[dict[str, Any]],
    metrics_output: Path,
    patterns_output: Path,
) -> None:
    metrics_output.parent.mkdir(parents=True, exist_ok=True)
    patterns_output.parent.mkdir(parents=True, exist_ok=True)
    with metrics_output.open("w", encoding="utf-8") as output_file:
        json.dump(metrics, output_file, ensure_ascii=False, indent=2)
        output_file.write("\n")
    with patterns_output.open("w", encoding="utf-8", newline="") as output_file:
        writer = csv.DictWriter(output_file, fieldnames=list(pattern_rows[0]))
        writer.writeheader()
        writer.writerows(pattern_rows)


def main() -> None:
    args = parse_args()
    metrics, pattern_rows = compute_metrics(load_predictions(args.predictions))
    write_outputs(metrics, pattern_rows, args.metrics_output, args.patterns_output)
    print(json.dumps(metrics["sample_metrics"], indent=2))


if __name__ == "__main__":
    main()
