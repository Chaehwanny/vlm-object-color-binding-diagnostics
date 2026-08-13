#!/usr/bin/env python3
"""Official aggregation for color-binding feasibility and interim studies."""

from __future__ import annotations

import csv
import json
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any


CORE = ("binding_true", "binding_false", "image_conflict", "edit_control")
CORE_ABBR = {
    "binding_true": "BT",
    "binding_false": "BF",
    "image_conflict": "IC",
    "edit_control": "EC",
}
EXPECTED_CORE = {
    "binding_true": "Yes",
    "binding_false": "No",
    "image_conflict": "No",
    "edit_control": "Yes",
}
STATE_CONTROLS = {
    "original": {
        "object_a_visible": "original_a_visible",
        "object_b_visible": "original_b_visible",
        "object_a_target_color": "original_a_source_color",
        "object_b_target_color": "original_b_source_color",
        "object_a_opposite_color_negation": "original_a_target_color",
        "object_b_opposite_color_negation": "original_b_target_color",
        "target_colors_present": "original_color_presence",
    },
    "edited": {
        "object_a_visible": "edited_a_visible",
        "object_b_visible": "edited_b_visible",
        "object_a_target_color": "edited_a_target_color",
        "object_b_target_color": "edited_b_target_color",
        "object_a_opposite_color_negation": "edited_a_source_color",
        "object_b_opposite_color_negation": "edited_b_source_color",
        "target_colors_present": "edited_color_presence",
    },
}
ALL_CONDITIONS = CORE + tuple(
    condition
    for state in ("original", "edited")
    for condition in STATE_CONTROLS[state].values()
)
MODEL_FILES = {
    "Qwen2.5-VL-7B": "qwen2_5_vl_7b.jsonl",
    "LLaVA-OneVision-7B": "llava_onevision_qwen2_7b.jsonl",
    "InternVL2.5-8B": "internvl2_5_8b.jsonl",
}


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    with path.open("r", encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def answer(row: dict[str, Any]) -> str:
    value = row.get("parsed_answer") or row.get("forced_choice_answer")
    return value if value in {"Yes", "No"} else "Invalid"


def correct(row: dict[str, Any]) -> bool:
    return answer(row) == row.get("expected_answer")


def ratio(numerator: int, denominator: int) -> float | None:
    return numerator / denominator if denominator else None


def index_rows(
    label: str, rows: list[dict[str, Any]], manifest_expected: dict[tuple[str, str], str]
) -> dict[str, dict[str, dict[str, Any]]]:
    indexed: dict[str, dict[str, dict[str, Any]]] = defaultdict(dict)
    for row in rows:
        sample = row["sample_id"]
        condition = row["condition"]
        key = (sample, condition)
        if key not in manifest_expected:
            raise ValueError(f"{label}: prediction not in manifest: {key}")
        if condition in indexed[sample]:
            raise ValueError(f"{label}: duplicate prediction: {key}")
        if row.get("expected_answer") != manifest_expected[key]:
            raise ValueError(f"{label}: expected-answer mismatch: {key}")
        indexed[sample][condition] = row
    for sample, conditions in indexed.items():
        missing = sorted(set(ALL_CONDITIONS) - set(conditions))
        if missing:
            raise ValueError(f"{label}: {sample} missing {missing}")
    if len(rows) != len(indexed) * len(ALL_CONDITIONS):
        raise ValueError(f"{label}: incomplete 18-query blocks")
    return dict(indexed)


def load_manifest(path: Path) -> tuple[list[dict[str, Any]], dict[tuple[str, str], str]]:
    rows = read_jsonl(path)
    expected: dict[tuple[str, str], str] = {}
    for row in rows:
        key = (row["sample_id"], row["condition"])
        if key in expected:
            raise ValueError(f"Duplicate manifest row: {key}")
        expected[key] = row["expected_answer"]
    return rows, expected


def core_pattern(answers: dict[str, str]) -> str:
    return "|".join(answers[name] for name in CORE)


def classify_core_pattern(answers: dict[str, str], obc: bool, ebc: bool) -> str:
    pattern = tuple(answers[name] for name in CORE)
    exact = {
        ("Yes", "No", "No", "Yes"): "correct_pattern",
        ("Yes", "Yes", "Yes", "Yes"): "all_yes",
        ("No", "No", "No", "No"): "all_no",
        ("No", "Yes", "Yes", "No"): "complete_reversal",
    }
    if pattern in exact:
        return exact[pattern]
    if obc and not ebc:
        return "edited_only_failure"
    if not obc and ebc:
        return "original_only_failure"
    return "original_and_edited_failure"


def polarity_type(answers: dict[str, str]) -> str:
    true_errors = any(answers[name] != EXPECTED_CORE[name] for name in ("binding_true", "edit_control"))
    false_errors = any(answers[name] != EXPECTED_CORE[name] for name in ("binding_false", "image_conflict"))
    if not true_errors and not false_errors:
        return "none"
    if true_errors and not false_errors:
        return "true_conditions_only"
    if false_errors and not true_errors:
        return "false_conditions_only"
    return "mixed"


def sample_results(
    indexed_models: dict[str, dict[str, dict[str, dict[str, Any]]]]
) -> list[dict[str, Any]]:
    output: list[dict[str, Any]] = []
    for model, samples in indexed_models.items():
        for sample_id in sorted(samples):
            rows = samples[sample_id]
            answers = {name: answer(rows[name]) for name in ALL_CONDITIONS}
            oav = all(correct(rows[name]) for name in STATE_CONTROLS["original"].values())
            eav = all(correct(rows[name]) for name in STATE_CONTROLS["edited"].values())
            obc = answers["binding_true"] == "Yes" and answers["binding_false"] == "No"
            ebc = answers["image_conflict"] == "No" and answers["edit_control"] == "Yes"
            eligible = oav and eav and obc
            first = rows["binding_true"]
            result: dict[str, Any] = {
                "model": model,
                "model_id": first.get("model_id"),
                "sample_id": sample_id,
                "source_candidate_id": first.get("source_candidate_id"),
                "source_dataset": first.get("source_dataset"),
                "color_pair": first.get("color_pair")
                or f"{first.get('object_a_original_color')}_{first.get('object_b_original_color')}",
            }
            for name in CORE:
                abbr = CORE_ABBR[name]
                result[abbr] = answers[name]
                result[f"{abbr}_correct"] = correct(rows[name])
            for state in ("original", "edited"):
                for control_type, condition in STATE_CONTROLS[state].items():
                    result[f"{state}_{control_type}"] = answers[condition]
                    result[f"{state}_{control_type}_correct"] = correct(rows[condition])
            all_answers = list(answers.values())
            result.update(
                {
                    "original_controls_pass": oav,
                    "edited_controls_pass": eav,
                    "all_controls_pass": oav and eav,
                    "OBC": obc,
                    "EBC": ebc,
                    "FBCC": obc and ebc,
                    "incremental_edit_failure": obc and not ebc,
                    "eligible": eligible,
                    "CQ_IBF": eligible and not ebc,
                    "core_response_pattern": core_pattern(answers),
                    "core_pattern_class": classify_core_pattern(answers, obc, ebc),
                    "polarity_error_type": polarity_type(answers),
                    "all_18_yes": all(value == "Yes" for value in all_answers),
                    "all_18_no": all(value == "No" for value in all_answers),
                    "invalid_count": sum(value == "Invalid" for value in all_answers),
                    "failed_control_conditions": ";".join(
                        name
                        for state in ("original", "edited")
                        for name in STATE_CONTROLS[state].values()
                        if not correct(rows[name])
                    ),
                }
            )
            output.append(result)
    return output


def build_summaries(
    indexed_models: dict[str, dict[str, dict[str, dict[str, Any]]]],
    sample_rows: list[dict[str, Any]],
) -> tuple[list[dict[str, Any]], list[dict[str, Any]], list[dict[str, Any]]]:
    samples_by_model: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in sample_rows:
        samples_by_model[row["model"]].append(row)
    model_summary = []
    condition_summary = []
    control_summary = []
    for model, samples in indexed_models.items():
        pattern_rows = samples_by_model[model]
        flat = [row for conditions in samples.values() for row in conditions.values()]
        counts = Counter(answer(row) for row in flat)
        expected_counts = Counter(row["expected_answer"] for row in flat)
        summary: dict[str, Any] = {
            "model": model,
            "model_id": flat[0].get("model_id"),
            "scoring_method": flat[0].get("scoring_method", "deterministic_generation"),
            "n_samples": len(samples),
            "queries_per_sample": len(ALL_CONDITIONS),
            "n_responses": len(flat),
        }
        for name in CORE:
            rows = [conditions[name] for conditions in samples.values()]
            n_correct = sum(correct(row) for row in rows)
            responses = Counter(answer(row) for row in rows)
            abbr = CORE_ABBR[name].lower()
            summary[f"{abbr}_correct"] = n_correct
            summary[f"{abbr}_total"] = len(rows)
            summary[f"{abbr}_accuracy"] = ratio(n_correct, len(rows))
            condition_summary.append(
                {
                    "model": model,
                    "condition": name,
                    "correct": n_correct,
                    "total": len(rows),
                    "accuracy": ratio(n_correct, len(rows)),
                    "yes_count": responses["Yes"],
                    "yes_rate": ratio(responses["Yes"], len(rows)),
                    "no_count": responses["No"],
                    "no_rate": ratio(responses["No"], len(rows)),
                    "invalid_count": responses["Invalid"],
                    "invalid_rate": ratio(responses["Invalid"], len(rows)),
                }
            )
        for state in ("original", "edited"):
            for control_type, condition in STATE_CONTROLS[state].items():
                rows = [conditions[condition] for conditions in samples.values()]
                responses = Counter(answer(row) for row in rows)
                n_correct = sum(correct(row) for row in rows)
                control_summary.append(
                    {
                        "model": model,
                        "state": state,
                        "control_type": control_type,
                        "source_condition": condition,
                        "expected_answer": rows[0]["expected_answer"],
                        "correct": n_correct,
                        "total": len(rows),
                        "accuracy": ratio(n_correct, len(rows)),
                        "yes_count": responses["Yes"],
                        "no_count": responses["No"],
                        "invalid_count": responses["Invalid"],
                    }
                )
        for field, prefix in (
            ("original_controls_pass", "oav"),
            ("edited_controls_pass", "eav"),
            ("all_controls_pass", "all_controls"),
            ("OBC", "obc"),
            ("EBC", "ebc"),
            ("FBCC", "fbcc"),
        ):
            number = sum(bool(row[field]) for row in pattern_rows)
            summary[f"{prefix}_count"] = number
            summary[f"{prefix}_total"] = len(pattern_rows)
            summary[f"{prefix}_rate"] = ratio(number, len(pattern_rows))
        eligible = sum(bool(row["eligible"]) for row in pattern_rows)
        cq_ibf = sum(bool(row["CQ_IBF"]) for row in pattern_rows)
        incremental_eligible = sum(bool(row["OBC"]) for row in pattern_rows)
        incremental_failures = sum(
            bool(row["incremental_edit_failure"]) for row in pattern_rows
        )
        summary["incremental_edit_eligible"] = incremental_eligible
        summary["incremental_edit_failure_count"] = incremental_failures
        summary["incremental_edit_failure_rate"] = ratio(
            incremental_failures, incremental_eligible
        )
        summary["incremental_edit_failure_ids"] = ";".join(
            row["source_candidate_id"]
            for row in pattern_rows
            if row["incremental_edit_failure"]
        )
        summary["cq_ibf_eligible"] = eligible
        summary["cq_ibf_count"] = cq_ibf
        summary["cq_ibf_rate"] = ratio(cq_ibf, eligible)

        true_conditions = ("BT_correct", "EC_correct")
        false_conditions = ("BF_correct", "IC_correct")
        true_total = len(pattern_rows) * 2
        false_total = len(pattern_rows) * 2
        true_errors = sum(not row[field] for row in pattern_rows for field in true_conditions)
        false_errors = sum(not row[field] for row in pattern_rows for field in false_conditions)
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
                "true_minus_false_accuracy": true_accuracy - false_accuracy,
                "yes_count": counts["Yes"],
                "yes_rate": ratio(counts["Yes"], len(flat)),
                "no_count": counts["No"],
                "no_rate": ratio(counts["No"], len(flat)),
                "invalid_count": counts["Invalid"],
                "invalid_rate": ratio(counts["Invalid"], len(flat)),
                "expected_yes_count": expected_counts["Yes"],
                "expected_yes_rate": ratio(expected_counts["Yes"], len(flat)),
                "expected_no_count": expected_counts["No"],
                "expected_no_rate": ratio(expected_counts["No"], len(flat)),
                "fbcc_failure_ids": ";".join(
                    row["source_candidate_id"] for row in pattern_rows if not row["FBCC"]
                ),
                "cq_ibf_ids": ";".join(
                    row["source_candidate_id"] for row in pattern_rows if row["CQ_IBF"]
                ),
                "edited_control_failure_ids": ";".join(
                    row["source_candidate_id"]
                    for row in pattern_rows
                    if not row["edited_controls_pass"]
                ),
            }
        )
        model_summary.append(summary)
    return model_summary, condition_summary, control_summary


def exact_overlap(failures: dict[str, set[str]], models: list[str]) -> dict[str, list[str]]:
    qwen, llava, internvl = models
    qset, lset, iset = failures[qwen], failures[llava], failures[internvl]
    return {
        "all_models_fbcc_fail": sorted(qset & lset & iset),
        "qwen_only": sorted(qset - lset - iset),
        "llava_only": sorted(lset - qset - iset),
        "internvl_only": sorted(iset - qset - lset),
        "qwen_llava_only": sorted((qset & lset) - iset),
        "qwen_internvl_only": sorted((qset & iset) - lset),
        "llava_internvl_only": sorted((lset & iset) - qset),
    }


def build_overlap(sample_rows: list[dict[str, Any]], models: list[str]) -> dict[str, Any]:
    by_model: dict[str, list[dict[str, Any]]] = defaultdict(list)
    by_sample: dict[str, dict[str, dict[str, Any]]] = defaultdict(dict)
    for row in sample_rows:
        by_model[row["model"]].append(row)
        by_sample[row["source_candidate_id"]][row["model"]] = row
    failures = {
        model: {row["source_candidate_id"] for row in by_model[model] if not row["FBCC"]}
        for model in models
    }
    passes = {
        model: {row["source_candidate_id"] for row in by_model[model] if row["FBCC"]}
        for model in models
    }
    named_patterns: dict[str, dict[str, list[str]]] = {}
    raw_patterns: dict[str, dict[str, list[str]]] = {}
    for model in models:
        named: dict[str, list[str]] = defaultdict(list)
        raw: dict[str, list[str]] = defaultdict(list)
        for row in by_model[model]:
            named[row["core_pattern_class"]].append(row["source_candidate_id"])
            raw[row["core_response_pattern"]].append(row["source_candidate_id"])
        named_patterns[model] = {key: sorted(value) for key, value in sorted(named.items())}
        raw_patterns[model] = {key: sorted(value) for key, value in sorted(raw.items())}
    overlap = exact_overlap(failures, models)
    overlap.update(
        {
            "models": models,
            "all_models_fbcc_pass": sorted(set.intersection(*(passes[m] for m in models))),
            "cq_ibf_by_model": {
                model: sorted(row["source_candidate_id"] for row in by_model[model] if row["CQ_IBF"])
                for model in models
            },
            "edited_control_fail_by_model": {
                model: sorted(
                    row["source_candidate_id"]
                    for row in by_model[model]
                    if not row["edited_controls_pass"]
                )
                for model in models
            },
            "edited_basic_fail_by_model": {
                model: sorted(
                    row["source_candidate_id"]
                    for row in by_model[model]
                    if not row["edited_controls_pass"]
                )
                for model in models
            },
            "all_yes_18_by_model": {
                model: sorted(row["source_candidate_id"] for row in by_model[model] if row["all_18_yes"])
                for model in models
            },
            "all_no_18_by_model": {
                model: sorted(row["source_candidate_id"] for row in by_model[model] if row["all_18_no"])
                for model in models
            },
            "named_core_patterns_by_model": named_patterns,
            "raw_core_patterns_by_model": raw_patterns,
            "all_models_same_core_pattern": sorted(
                sample
                for sample, model_rows in by_sample.items()
                if len({model_rows[m]["core_response_pattern"] for m in models}) == 1
            ),
        }
    )
    return overlap


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def aggregate_study(
    manifest_path: Path,
    prediction_files: dict[str, Path],
    output_dir: Path,
    prefix: str,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    _, expected = load_manifest(manifest_path)
    indexed = {
        model: index_rows(model, read_jsonl(path), expected)
        for model, path in prediction_files.items()
    }
    sample_sets = {model: set(samples) for model, samples in indexed.items()}
    if len({frozenset(values) for values in sample_sets.values()}) != 1:
        raise ValueError(f"Models have different sample sets: {sample_sets}")
    samples = sample_results(indexed)
    model_summary, condition_summary, control_summary = build_summaries(indexed, samples)
    overlap: dict[str, Any] = {}
    write_csv(output_dir / f"{prefix}_model_summary.csv", model_summary)
    write_csv(output_dir / f"{prefix}_condition_summary.csv", condition_summary)
    write_csv(output_dir / f"{prefix}_control_summary.csv", control_summary)
    write_csv(output_dir / f"{prefix}_sample_level_results.csv", samples)
    if len(indexed) == 3:
        overlap = build_overlap(samples, list(indexed))
        (output_dir / f"{prefix}_cross_model_overlap.json").write_text(
            json.dumps(overlap, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
        )
    return model_summary, overlap


def main() -> None:
    root = Path(__file__).resolve().parents[2]
    output_root = root / "experiments/attribute_binding/official_aggregate_v1"

    feasibility_base = root / "experiments/attribute_binding/feasibility_n5_v1"
    feasibility_summary, _ = aggregate_study(
        root / "processed/attribute_binding/datasets/feasibility_n5_v1/manifest.jsonl",
        {"Qwen2.5-VL-7B": feasibility_base / "predictions/qwen2_5_vl_7b.jsonl"},
        output_root / "feasibility_n5",
        "feasibility",
    )

    forced_feasibility = read_jsonl(
        feasibility_base / "forced_choice/qwen2_5_vl_7b_binding_feas_005.jsonl"
    )
    generated_feasibility = {
        (row["sample_id"], row["condition"]): answer(row)
        for row in read_jsonl(feasibility_base / "predictions/qwen2_5_vl_7b.jsonl")
    }
    forced_agreement = all(
        answer(row) == generated_feasibility[(row["sample_id"], row["condition"])]
        for row in forced_feasibility
    )
    feasibility_special = {
        "replacement_004_forced_choice_queries": len(forced_feasibility),
        "replacement_004_generation_forced_choice_agreement": forced_agreement,
        "replacement_004_forced_choice_answers": {
            row["condition"]: answer(row) for row in forced_feasibility
        },
    }
    (output_root / "feasibility_n5/feasibility_special_cases.json").write_text(
        json.dumps(feasibility_special, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )

    interim_base = root / "experiments/attribute_binding/interim_n8_v1"
    interim_files = {
        model: interim_base / "generation" / filename
        for model, filename in MODEL_FILES.items()
    }
    interim_summary, overlap = aggregate_study(
        root / "processed/attribute_binding/datasets/interim_n8_v1/manifest.jsonl",
        interim_files,
        output_root / "interim_n8",
        "color",
    )

    qwen_generation = read_jsonl(interim_files["Qwen2.5-VL-7B"])
    qwen_forced = read_jsonl(interim_base / "forced_choice/qwen2_5_vl_7b.jsonl")
    generation_answers = {
        (row["sample_id"], row["condition"]): answer(row) for row in qwen_generation
    }
    qwen_agreement = sum(
        answer(row) == generation_answers[(row["sample_id"], row["condition"])]
        for row in qwen_forced
    )
    inventory = {
        "queries_per_sample": 18,
        "interim_samples": 8,
        "queries_per_model": 144,
        "models": 3,
        "queries_all_models": 432,
        "qwen_generation_forced_choice_agreement": {
            "count": qwen_agreement,
            "total": len(qwen_forced),
            "rate": ratio(qwen_agreement, len(qwen_forced)),
        },
        "official_cross_model_scoring": "deterministic_generation",
    }
    (output_root / "interim_n8/color_query_inventory.json").write_text(
        json.dumps(inventory, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )

    print("Feasibility:", feasibility_summary[0]["n_responses"], "Qwen responses")
    print("Interim: 144 responses/model; 432 across three models")
    for row in interim_summary:
        print(
            row["model"],
            f"OAV={row['oav_count']}/8",
            f"EAV={row['eav_count']}/8",
            f"OBC={row['obc_count']}/8",
            f"EBC={row['ebc_count']}/8",
            f"FBCC={row['fbcc_count']}/8",
            f"CQ-IBF={row['cq_ibf_count']}/{row['cq_ibf_eligible']}",
        )
    print("Qwen generation/forced-choice agreement:", qwen_agreement, "/", len(qwen_forced))
    print("Wrote", output_root)


if __name__ == "__main__":
    main()
