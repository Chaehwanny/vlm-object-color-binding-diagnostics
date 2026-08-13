#!/usr/bin/env python3
"""Compute state-specific binding diagnostics, including CQ-IBF."""

from __future__ import annotations

import argparse
import csv
import json
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any


OAV_CONDITIONS = (
    "original_a_visible",
    "original_b_visible",
    "original_a_source_color",
    "original_a_target_color",
    "original_b_source_color",
    "original_b_target_color",
    "original_color_presence",
)
EAV_CONDITIONS = (
    "edited_a_visible",
    "edited_b_visible",
    "edited_a_source_color",
    "edited_a_target_color",
    "edited_b_source_color",
    "edited_b_target_color",
    "edited_color_presence",
)
OBC_CONDITIONS = ("binding_true", "binding_false")
EBC_CONDITIONS = ("image_conflict", "edit_control")


def parse_args() -> argparse.Namespace:
    root = Path(__file__).resolve().parents[2]
    base = root / "experiments/attribute_binding/interim_n8_v1"
    parser = argparse.ArgumentParser(description="Compute binding diagnostics.")
    parser.add_argument(
        "--predictions",
        type=Path,
        default=base / "forced_choice/qwen2_5_vl_7b.jsonl",
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
    with path.open("r", encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def correct(row: dict[str, Any]) -> bool:
    return bool(row.get("is_valid")) and row.get("is_correct") is True


def ratio(numerator: int, denominator: int) -> dict[str, Any]:
    return {
        "numerator": numerator,
        "denominator": denominator,
        "rate": numerator / denominator if denominator else None,
    }


def main() -> None:
    args = parse_args()
    rows = read_jsonl(args.predictions)
    by_sample: dict[str, dict[str, dict[str, Any]]] = defaultdict(dict)
    by_condition: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        by_sample[row["sample_id"]][row["condition"]] = row
        by_condition[row["condition"]].append(row)

    required = set(OAV_CONDITIONS + EAV_CONDITIONS + OBC_CONDITIONS + EBC_CONDITIONS)
    patterns = []
    for sample_id in sorted(by_sample):
        conditions = by_sample[sample_id]
        missing = required - set(conditions)
        if missing:
            raise ValueError(f"{sample_id} missing conditions: {sorted(missing)}")
        oav = all(correct(conditions[name]) for name in OAV_CONDITIONS)
        eav = all(correct(conditions[name]) for name in EAV_CONDITIONS)
        obc = all(correct(conditions[name]) for name in OBC_CONDITIONS)
        ebc = all(correct(conditions[name]) for name in EBC_CONDITIONS)
        qualified = oav and eav and obc
        cq_ibf = qualified and not ebc
        stale_acceptance = qualified and not correct(conditions["image_conflict"])
        new_rejection = qualified and not correct(conditions["edit_control"])
        first = conditions["binding_true"]
        patterns.append(
            {
                "sample_id": sample_id,
                "source_candidate_id": first["source_candidate_id"],
                "source_dataset": first["source_dataset"],
                "color_pair": first["color_pair"],
                "binding_true": conditions["binding_true"]["parsed_answer"],
                "binding_false": conditions["binding_false"]["parsed_answer"],
                "image_conflict": conditions["image_conflict"]["parsed_answer"],
                "edit_control": conditions["edit_control"]["parsed_answer"],
                "oav": oav,
                "eav": eav,
                "obc": obc,
                "ebc": ebc,
                "fbcc": obc and ebc,
                "all_atomic_controls_pass": oav and eav,
                "cq_ibf_qualified": qualified,
                "cq_ibf": cq_ibf,
                "stale_binding_acceptance": stale_acceptance,
                "new_binding_rejection": new_rejection,
                "failed_atomic_conditions": ";".join(
                    name
                    for name in OAV_CONDITIONS + EAV_CONDITIONS
                    if not correct(conditions[name])
                ),
                "min_absolute_forced_choice_margin": (
                    min(
                        abs(float(row["yes_minus_no_logit"]))
                        for row in conditions.values()
                        if "yes_minus_no_logit" in row
                    )
                    if any("yes_minus_no_logit" in row for row in conditions.values())
                    else None
                ),
            }
        )

    n = len(patterns)
    qualified = sum(row["cq_ibf_qualified"] for row in patterns)
    cq_ibf = sum(row["cq_ibf"] for row in patterns)
    condition_metrics = {}
    for name, condition_rows in sorted(by_condition.items()):
        answers = Counter(row["parsed_answer"] for row in condition_rows)
        n_correct = sum(correct(row) for row in condition_rows)
        condition_metrics[name] = {
            "n": len(condition_rows),
            "correct": n_correct,
            "accuracy": n_correct / len(condition_rows),
            "yes": answers["Yes"],
            "no": answers["No"],
            "invalid": sum(not row.get("is_valid") for row in condition_rows),
        }

    true_rows = [row for row in rows if row["expected_answer"] == "Yes"]
    false_rows = [row for row in rows if row["expected_answer"] == "No"]
    metrics = {
        "schema_version": "0.1",
        "study_role": "interim_development_smoke_test",
        "model_id": rows[0]["model_id"],
        "scoring_method": rows[0].get("scoring_method", "deterministic_generation"),
        "n_samples": n,
        "n_predictions": len(rows),
        "condition_metrics": condition_metrics,
        "sample_consistency": {
            "original_atomic_validity": ratio(sum(row["oav"] for row in patterns), n),
            "edited_atomic_validity": ratio(sum(row["eav"] for row in patterns), n),
            "original_binding_consistency": ratio(sum(row["obc"] for row in patterns), n),
            "edited_binding_consistency": ratio(sum(row["ebc"] for row in patterns), n),
            "full_binding_counterfactual_consistency": ratio(
                sum(row["fbcc"] for row in patterns), n
            ),
            "all_atomic_controls_pass": ratio(
                sum(row["all_atomic_controls_pass"] for row in patterns), n
            ),
        },
        "control_qualified_incremental_binding_failure": ratio(cq_ibf, qualified),
        "cq_ibf_qualified_samples": [
            row["source_candidate_id"] for row in patterns if row["cq_ibf_qualified"]
        ],
        "cq_ibf_failure_samples": [
            row["source_candidate_id"] for row in patterns if row["cq_ibf"]
        ],
        "cq_ibf_subtypes": {
            "stale_binding_acceptance": ratio(
                sum(row["stale_binding_acceptance"] for row in patterns), qualified
            ),
            "new_binding_rejection": ratio(
                sum(row["new_binding_rejection"] for row in patterns), qualified
            ),
        },
        "answer_polarity_errors": {
            "true_condition_error": ratio(
                sum(not correct(row) for row in true_rows), len(true_rows)
            ),
            "false_condition_error": ratio(
                sum(not correct(row) for row in false_rows), len(false_rows)
            ),
            "invalid": ratio(sum(not row.get("is_valid") for row in rows), len(rows)),
        },
        "interpretation_limits": {
            "development_data": True,
            "color_pair_imbalanced": True,
            "generalization_claim_allowed": False,
        },
    }
    args.metrics_output.parent.mkdir(parents=True, exist_ok=True)
    args.metrics_output.write_text(
        json.dumps(metrics, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    with args.patterns_output.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(patterns[0]))
        writer.writeheader()
        writer.writerows(patterns)
    print(json.dumps(metrics["sample_consistency"], ensure_ascii=False, indent=2))
    print(
        json.dumps(
            metrics["control_qualified_incremental_binding_failure"],
            ensure_ascii=False,
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
