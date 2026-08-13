#!/usr/bin/env python3
"""Officially aggregate the three-model left/right pilot from raw responses."""

from __future__ import annotations

import argparse
import csv
import json
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any


CONDITIONS = (
    "baseline_true",
    "baseline_false",
    "irrelevant_true",
    "image_conflict",
    "flip_control",
)
ABBREVIATIONS = {
    "baseline_true": "BT",
    "baseline_false": "BF",
    "irrelevant_true": "IT",
    "image_conflict": "IC",
    "flip_control": "FC",
}
EXPECTED = {
    "baseline_true": "Yes",
    "baseline_false": "No",
    "irrelevant_true": "Yes",
    "image_conflict": "No",
    "flip_control": "Yes",
}
CORE_CONDITIONS = (
    "baseline_true",
    "baseline_false",
    "image_conflict",
    "flip_control",
)
TRUE_KEYED = ("baseline_true", "flip_control")
FALSE_KEYED = ("baseline_false", "image_conflict")


def parse_args() -> argparse.Namespace:
    project_root = Path(__file__).resolve().parents[2]
    parser = argparse.ArgumentParser(
        description="Aggregate raw left/right pilot predictions across models."
    )
    parser.add_argument(
        "--predictions-dir",
        type=Path,
        default=project_root / "experiments/spatial_left_right/positive_control_v1/predictions",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=project_root
        / "experiments/spatial_left_right/positive_control_v1/official_aggregate_v1",
    )
    return parser.parse_args()


def model_files(predictions_dir: Path) -> dict[str, Path]:
    return {
        "Qwen2.5-VL-7B": predictions_dir / "qwen2_5_vl_7b.jsonl",
        "LLaVA-OneVision-7B": predictions_dir
        / "llava_onevision_qwen2_7b.jsonl",
        "InternVL2.5-8B": predictions_dir / "internvl2_5_8b.jsonl",
    }


def load_jsonl(path: Path) -> list[dict[str, Any]]:
    with path.open("r", encoding="utf-8") as input_file:
        return [json.loads(line) for line in input_file if line.strip()]


def canonical_answer(row: dict[str, Any]) -> str:
    answer = row.get("parsed_answer")
    return answer if answer in {"Yes", "No"} else "Invalid"


def review_id(row: dict[str, Any]) -> str:
    value = row.get("review_candidate_id") or row.get("sample_id")
    if not value:
        raise ValueError("A prediction row has no review/sample identifier.")
    return str(value)


def validate_and_index(
    label: str, rows: list[dict[str, Any]]
) -> dict[str, dict[str, dict[str, Any]]]:
    indexed: dict[str, dict[str, dict[str, Any]]] = defaultdict(dict)
    for row in rows:
        condition = row.get("condition")
        if condition not in CONDITIONS:
            raise ValueError(f"{label}: unexpected condition {condition!r}")
        sample = review_id(row)
        if condition in indexed[sample]:
            raise ValueError(f"{label}: duplicate {sample}/{condition}")
        if row.get("expected_answer") != EXPECTED[condition]:
            raise ValueError(
                f"{label}: wrong expected answer for {sample}/{condition}: "
                f"{row.get('expected_answer')!r}"
            )
        indexed[sample][condition] = row

    for sample, conditions in indexed.items():
        missing = sorted(set(CONDITIONS) - set(conditions))
        if missing:
            raise ValueError(f"{label}: {sample} is missing {missing}")
    expected_rows = len(indexed) * len(CONDITIONS)
    if len(rows) != expected_rows:
        raise ValueError(f"{label}: expected {expected_rows} rows, got {len(rows)}")
    return dict(indexed)


def classify_pattern(answers: dict[str, str], orc: bool, frc: bool) -> str:
    core = tuple(answers[condition] for condition in CORE_CONDITIONS)
    exact = {
        ("Yes", "No", "No", "Yes"): "correct_pattern",
        ("Yes", "Yes", "Yes", "Yes"): "all_yes",
        ("No", "No", "No", "No"): "all_no",
        ("No", "Yes", "Yes", "No"): "complete_reversal",
    }
    if core in exact:
        return exact[core]
    if orc and not frc:
        return "flip_only_failure"
    if not orc and frc:
        return "original_only_failure"
    return "original_and_flip_failure"


def polarity_pattern(answers: dict[str, str]) -> str:
    true_errors = [c for c in TRUE_KEYED if answers[c] != EXPECTED[c]]
    false_errors = [c for c in FALSE_KEYED if answers[c] != EXPECTED[c]]
    if not true_errors and not false_errors:
        return "none"
    if true_errors and not false_errors:
        return "true_conditions_only"
    if false_errors and not true_errors:
        return "false_conditions_only"
    return "mixed"


def build_sample_rows(
    indexed_models: dict[str, dict[str, dict[str, dict[str, Any]]]]
) -> list[dict[str, Any]]:
    output: list[dict[str, Any]] = []
    for model, samples in indexed_models.items():
        for sample in sorted(samples):
            rows = samples[sample]
            answers = {condition: canonical_answer(rows[condition]) for condition in CONDITIONS}
            correct = {
                condition: answers[condition] == EXPECTED[condition]
                for condition in CONDITIONS
            }
            orc = answers["baseline_true"] == "Yes" and answers["baseline_false"] == "No"
            frc = answers["image_conflict"] == "No" and answers["flip_control"] == "Yes"
            fcc = orc and frc
            first = rows["baseline_true"]
            result: dict[str, Any] = {
                "model": model,
                "model_id": first.get("model_id"),
                "review_id": sample,
                "internal_sample_id": first.get("internal_sample_id")
                or first.get("sample_id"),
                "source_label": first.get("source_label"),
                "target_label": first.get("target_label"),
                "relation_orientation": first.get("relation_orientation"),
            }
            for condition in CONDITIONS:
                abbreviation = ABBREVIATIONS[condition]
                result[abbreviation] = answers[condition]
                result[f"{abbreviation}_correct"] = correct[condition]
            result.update(
                {
                    "ORC": orc,
                    "FRC": frc,
                    "FCC": fcc,
                    "incremental_failure": orc and not frc,
                    "response_pattern": "|".join(
                        answers[condition] for condition in CORE_CONDITIONS
                    ),
                    "error_pattern": classify_pattern(answers, orc, frc),
                    "polarity_error_type": polarity_pattern(answers),
                }
            )
            output.append(result)
    return output


def ratio(numerator: int, denominator: int) -> float | None:
    return numerator / denominator if denominator else None


def build_summaries(
    indexed_models: dict[str, dict[str, dict[str, dict[str, Any]]]],
    sample_rows: list[dict[str, Any]],
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    by_model_sample_rows: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in sample_rows:
        by_model_sample_rows[row["model"]].append(row)

    model_summaries: list[dict[str, Any]] = []
    condition_summaries: list[dict[str, Any]] = []
    for model, samples in indexed_models.items():
        rows = by_model_sample_rows[model]
        flat_rows = [row for conditions in samples.values() for row in conditions.values()]
        answer_counts = Counter(canonical_answer(row) for row in flat_rows)
        summary: dict[str, Any] = {
            "model": model,
            "model_id": flat_rows[0].get("model_id"),
            "n_samples": len(samples),
            "n_responses": len(flat_rows),
        }

        for condition in CONDITIONS:
            condition_rows = [conditions[condition] for conditions in samples.values()]
            answers = [canonical_answer(row) for row in condition_rows]
            counts = Counter(answers)
            correct = sum(answer == EXPECTED[condition] for answer in answers)
            abbreviation = ABBREVIATIONS[condition].lower()
            summary[f"{abbreviation}_correct"] = correct
            summary[f"{abbreviation}_total"] = len(answers)
            summary[f"{abbreviation}_accuracy"] = ratio(correct, len(answers))
            for answer, column in (("Yes", "yes"), ("No", "no"), ("Invalid", "invalid")):
                summary[f"{abbreviation}_{column}_rate"] = ratio(counts[answer], len(answers))
            condition_summaries.append(
                {
                    "model": model,
                    "model_id": flat_rows[0].get("model_id"),
                    "condition": condition,
                    "correct": correct,
                    "total": len(answers),
                    "accuracy": ratio(correct, len(answers)),
                    "yes_count": counts["Yes"],
                    "yes_rate": ratio(counts["Yes"], len(answers)),
                    "no_count": counts["No"],
                    "no_rate": ratio(counts["No"], len(answers)),
                    "invalid_count": counts["Invalid"],
                    "invalid_rate": ratio(counts["Invalid"], len(answers)),
                }
            )

        for metric in ("ORC", "FRC", "FCC"):
            count = sum(bool(row[metric]) for row in rows)
            summary[f"{metric.lower()}_count"] = count
            summary[f"{metric.lower()}_total"] = len(rows)
            summary[f"{metric.lower()}_rate"] = ratio(count, len(rows))

        eligible = sum(bool(row["ORC"]) for row in rows)
        failures = sum(bool(row["incremental_failure"]) for row in rows)
        summary["incremental_eligible"] = eligible
        summary["incremental_failure_count"] = failures
        summary["incremental_failure_rate"] = ratio(failures, eligible)
        summary["incremental_failure_ids"] = ";".join(
            row["review_id"] for row in rows if row["incremental_failure"]
        )
        summary["fcc_failure_ids"] = ";".join(
            row["review_id"] for row in rows if not row["FCC"]
        )

        true_total = len(rows) * len(TRUE_KEYED)
        false_total = len(rows) * len(FALSE_KEYED)
        true_errors = sum(not row[f"{ABBREVIATIONS[c]}_correct"] for row in rows for c in TRUE_KEYED)
        false_errors = sum(not row[f"{ABBREVIATIONS[c]}_correct"] for row in rows for c in FALSE_KEYED)
        true_accuracy = ratio(true_total - true_errors, true_total)
        false_accuracy = ratio(false_total - false_errors, false_total)
        summary.update(
            {
                "true_keyed_errors": true_errors,
                "true_keyed_total": true_total,
                "true_keyed_accuracy": true_accuracy,
                "false_keyed_errors": false_errors,
                "false_keyed_total": false_total,
                "false_keyed_accuracy": false_accuracy,
                "true_minus_false_accuracy": (
                    true_accuracy - false_accuracy
                    if true_accuracy is not None and false_accuracy is not None
                    else None
                ),
                "yes_count": answer_counts["Yes"],
                "yes_rate": ratio(answer_counts["Yes"], len(flat_rows)),
                "no_count": answer_counts["No"],
                "no_rate": ratio(answer_counts["No"], len(flat_rows)),
                "invalid_count": answer_counts["Invalid"],
                "invalid_rate": ratio(answer_counts["Invalid"], len(flat_rows)),
            }
        )
        model_summaries.append(summary)
    return model_summaries, condition_summaries


def exact_partition(failure_sets: dict[str, set[str]]) -> dict[str, list[str]]:
    labels = list(failure_sets)
    qwen, llava, internvl = labels
    qset, lset, iset = (failure_sets[label] for label in labels)
    return {
        "all_models_fail": sorted(qset & lset & iset),
        "qwen_only": sorted(qset - lset - iset),
        "llava_only": sorted(lset - qset - iset),
        "internvl_only": sorted(iset - qset - lset),
        "qwen_llava_only": sorted((qset & lset) - iset),
        "qwen_internvl_only": sorted((qset & iset) - lset),
        "llava_internvl_only": sorted((lset & iset) - qset),
        "model_key_order": [qwen, llava, internvl],
    }


def build_overlap(
    sample_rows: list[dict[str, Any]], model_order: list[str]
) -> dict[str, Any]:
    by_model: dict[str, list[dict[str, Any]]] = defaultdict(list)
    by_sample: dict[str, dict[str, dict[str, Any]]] = defaultdict(dict)
    for row in sample_rows:
        by_model[row["model"]].append(row)
        by_sample[row["review_id"]][row["model"]] = row

    failures = {
        model: {row["review_id"] for row in by_model[model] if not row["FCC"]}
        for model in model_order
    }
    passes = {
        model: {row["review_id"] for row in by_model[model] if row["FCC"]}
        for model in model_order
    }
    partition = exact_partition(failures)
    partition.pop("model_key_order")

    pattern_ids: dict[str, dict[str, list[str]]] = {}
    raw_pattern_ids: dict[str, dict[str, list[str]]] = {}
    for model in model_order:
        named: dict[str, list[str]] = defaultdict(list)
        raw: dict[str, list[str]] = defaultdict(list)
        for row in by_model[model]:
            named[row["error_pattern"]].append(row["review_id"])
            raw[row["response_pattern"]].append(row["review_id"])
        pattern_ids[model] = {key: sorted(value) for key, value in sorted(named.items())}
        raw_pattern_ids[model] = {key: sorted(value) for key, value in sorted(raw.items())}

    same_pattern = []
    sample_fcc_status: dict[str, dict[str, bool]] = {}
    for sample, models in sorted(by_sample.items()):
        if set(models) != set(model_order):
            raise ValueError(f"{sample}: models differ from the expected panel")
        patterns = {models[model]["response_pattern"] for model in model_order}
        if len(patterns) == 1:
            same_pattern.append(sample)
        sample_fcc_status[sample] = {
            model: bool(models[model]["FCC"]) for model in model_order
        }

    return {
        "schema_version": "1.0",
        "models": model_order,
        "fcc_failure_sets": {model: sorted(values) for model, values in failures.items()},
        "fcc_exact_overlap_partition": partition,
        "pairwise_failure_intersections_including_all_models": {
            "qwen_llava": sorted(failures[model_order[0]] & failures[model_order[1]]),
            "qwen_internvl": sorted(failures[model_order[0]] & failures[model_order[2]]),
            "llava_internvl": sorted(failures[model_order[1]] & failures[model_order[2]]),
        },
        "all_models_fcc_pass": sorted(set.intersection(*(passes[m] for m in model_order))),
        "all_models_same_core_response_pattern": same_pattern,
        "orc_passed_frc_failed": {
            model: sorted(
                row["review_id"]
                for row in by_model[model]
                if row["ORC"] and not row["FRC"]
            )
            for model in model_order
        },
        "named_patterns_by_model": pattern_ids,
        "raw_core_response_patterns_by_model": raw_pattern_ids,
        "fcc_status_by_sample": sample_fcc_status,
        "definitions": {
            "core_response_order": ["BT", "BF", "IC", "FC"],
            "ORC": "BT=Yes and BF=No",
            "FRC": "IC=No and FC=Yes",
            "FCC": "ORC and FRC",
            "incremental_failure": "ORC and not FRC; denominator is ORC passes",
        },
    }


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    if not rows:
        raise ValueError(f"Cannot write empty CSV: {path}")
    with path.open("w", encoding="utf-8", newline="") as output_file:
        writer = csv.DictWriter(output_file, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def main() -> None:
    args = parse_args()
    files = model_files(args.predictions_dir)
    indexed = {
        model: validate_and_index(model, load_jsonl(path))
        for model, path in files.items()
    }
    sample_sets = {model: set(samples) for model, samples in indexed.items()}
    if len({frozenset(samples) for samples in sample_sets.values()}) != 1:
        raise ValueError(f"Models have different sample sets: {sample_sets}")

    sample_rows = build_sample_rows(indexed)
    model_summaries, condition_summaries = build_summaries(indexed, sample_rows)
    overlap = build_overlap(sample_rows, list(indexed))

    args.output_dir.mkdir(parents=True, exist_ok=True)
    write_csv(args.output_dir / "model_summary.csv", model_summaries)
    write_csv(args.output_dir / "condition_summary.csv", condition_summaries)
    write_csv(args.output_dir / "sample_level_results.csv", sample_rows)
    with (args.output_dir / "cross_model_error_overlap.json").open(
        "w", encoding="utf-8"
    ) as output_file:
        json.dump(overlap, output_file, ensure_ascii=False, indent=2)
        output_file.write("\n")

    print(f"Validated {len(next(iter(indexed.values())))} shared samples per model.")
    for summary in model_summaries:
        print(
            summary["model"],
            f"ORC={summary['orc_count']}/{summary['orc_total']}",
            f"FRC={summary['frc_count']}/{summary['frc_total']}",
            f"FCC={summary['fcc_count']}/{summary['fcc_total']}",
            "incremental="
            f"{summary['incremental_failure_count']}/{summary['incremental_eligible']}",
        )
    print(f"Wrote official aggregate to {args.output_dir}")


if __name__ == "__main__":
    main()
