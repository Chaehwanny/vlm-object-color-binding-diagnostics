#!/usr/bin/env python3
"""Compare model-level metrics and sample-level error overlap for pilot v1."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Any


CONDITIONS = (
    "baseline_true",
    "baseline_false",
    "irrelevant_true",
    "image_conflict",
    "flip_control",
)
SAMPLE_METRICS = (
    "original_relation_consistency",
    "flip_relation_consistency",
    "full_counterfactual_consistency",
    "incremental_flip_failure",
)


def parse_args() -> argparse.Namespace:
    project_root = Path(__file__).resolve().parents[2]
    parser = argparse.ArgumentParser(description="Compare pilot results across models.")
    parser.add_argument(
        "--model",
        action="append",
        default=[],
        help="Model input as LABEL=METRICS_JSON,PATTERNS_CSV. May be repeated.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=project_root / "experiments/spatial_left_right/positive_control_v1/comparison",
    )
    return parser.parse_args()


def default_models(project_root: Path) -> list[str]:
    return [
        "qwen2_5_vl_7b="
        + str(project_root / "frozen/left_right_pilot_v1/metrics/qwen2_5_vl_7b_metrics.json")
        + ","
        + str(project_root / "frozen/left_right_pilot_v1/metrics/qwen2_5_vl_7b_sample_patterns.csv"),
        "llava_onevision_qwen2_7b="
        + str(project_root / "experiments/spatial_left_right/positive_control_v1/metrics/llava_onevision_qwen2_7b_metrics.json")
        + ","
        + str(project_root / "experiments/spatial_left_right/positive_control_v1/metrics/llava_onevision_qwen2_7b_sample_patterns.csv"),
        "internvl2_5_8b="
        + str(project_root / "experiments/spatial_left_right/positive_control_v1/metrics/internvl2_5_8b_metrics.json")
        + ","
        + str(project_root / "experiments/spatial_left_right/positive_control_v1/metrics/internvl2_5_8b_sample_patterns.csv"),
    ]


def parse_model_spec(spec: str) -> tuple[str, Path, Path]:
    label, paths = spec.split("=", maxsplit=1)
    metrics_path, patterns_path = paths.split(",", maxsplit=1)
    return label, Path(metrics_path), Path(patterns_path)


def load_patterns(path: Path) -> dict[str, dict[str, str]]:
    with path.open("r", encoding="utf-8") as input_file:
        return {row["review_id"]: row for row in csv.DictReader(input_file)}


def main() -> None:
    args = parse_args()
    project_root = Path(__file__).resolve().parents[2]
    specs = args.model or default_models(project_root)
    models: dict[str, dict[str, Any]] = {}
    patterns: dict[str, dict[str, dict[str, str]]] = {}
    for spec in specs:
        label, metrics_path, patterns_path = parse_model_spec(spec)
        models[label] = json.loads(metrics_path.read_text(encoding="utf-8"))
        patterns[label] = load_patterns(patterns_path)

    args.output_dir.mkdir(parents=True, exist_ok=True)
    summary_path = args.output_dir / "model_summary.csv"
    summary_fields = [
        "model",
        "model_id",
        "n_samples",
        "invalid_rate",
        "true_condition_error_rate",
        "false_condition_error_rate",
        "incremental_flip_failure_samples",
    ]
    summary_fields.extend(f"{condition}_accuracy" for condition in CONDITIONS)
    summary_fields.extend(
        item
        for metric in SAMPLE_METRICS
        for item in (f"{metric}_numerator", f"{metric}_denominator", f"{metric}_rate")
    )
    with summary_path.open("w", encoding="utf-8", newline="") as output_file:
        writer = csv.DictWriter(output_file, fieldnames=summary_fields)
        writer.writeheader()
        for label, metrics in models.items():
            invalid = sum(
                condition["invalid"] for condition in metrics["condition_metrics"].values()
            )
            row: dict[str, Any] = {
                "model": label,
                "model_id": metrics["model_id"],
                "n_samples": metrics["n_samples"],
                "invalid_rate": invalid / metrics["n_predictions"],
            }
            row["true_condition_error_rate"] = metrics["relation_polarity_errors"][
                "false_negative_rate_on_true_relation_conditions"
            ]["rate"]
            row["false_condition_error_rate"] = metrics["relation_polarity_errors"][
                "false_positive_rate_on_false_relation_conditions"
            ]["rate"]
            row["incremental_flip_failure_samples"] = ";".join(
                review_id
                for review_id, pattern in patterns[label].items()
                if pattern["incremental_flip_failure"] == "True"
            )
            for condition in CONDITIONS:
                row[f"{condition}_accuracy"] = metrics["condition_metrics"][condition][
                    "accuracy"
                ]
            for metric in SAMPLE_METRICS:
                values = metrics["sample_metrics"][metric]
                for field in ("numerator", "denominator", "rate"):
                    row[f"{metric}_{field}"] = values[field]
            writer.writerow(row)

    all_review_ids = sorted(set().union(*(set(rows) for rows in patterns.values())))
    overlap_path = args.output_dir / "sample_error_overlap.csv"
    overlap_fields = ["review_id", "source_label", "target_label", "relation_orientation"]
    for label in models:
        overlap_fields.extend(
            (
                f"{label}_failed_conditions",
                f"{label}_original_pass",
                f"{label}_flip_pass",
                f"{label}_fcc_pass",
                f"{label}_incremental_flip_failure",
            )
        )
    overlap_fields.extend(("models_with_any_core_error", "error_commonality"))
    with overlap_path.open("w", encoding="utf-8", newline="") as output_file:
        writer = csv.DictWriter(output_file, fieldnames=overlap_fields)
        writer.writeheader()
        for review_id in all_review_ids:
            first = next(rows[review_id] for rows in patterns.values() if review_id in rows)
            output: dict[str, Any] = {
                "review_id": review_id,
                "source_label": first["source_label"],
                "target_label": first["target_label"],
                "relation_orientation": first["relation_orientation"],
            }
            models_with_error = 0
            for label, model_patterns in patterns.items():
                row = model_patterns[review_id]
                expected = {
                    "baseline_true": "Yes",
                    "baseline_false": "No",
                    "image_conflict": "No",
                    "flip_control": "Yes",
                }
                failed = [
                    condition
                    for condition, answer in expected.items()
                    if row[condition] != answer
                ]
                models_with_error += bool(failed)
                output[f"{label}_failed_conditions"] = ";".join(failed)
                output[f"{label}_original_pass"] = row["original_relation_consistency"]
                output[f"{label}_flip_pass"] = row["flip_relation_consistency"]
                output[f"{label}_fcc_pass"] = row["full_counterfactual_consistency"]
                output[f"{label}_incremental_flip_failure"] = row[
                    "incremental_flip_failure"
                ]
            output["models_with_any_core_error"] = models_with_error
            if models_with_error == 0:
                output["error_commonality"] = "none"
            elif models_with_error == len(models):
                output["error_commonality"] = "shared_all_models"
            else:
                output["error_commonality"] = "model_specific_or_partial"
            writer.writerow(output)

    print(f"Wrote {summary_path}")
    print(f"Wrote {overlap_path}")


if __name__ == "__main__":
    main()
