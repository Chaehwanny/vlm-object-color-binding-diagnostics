#!/usr/bin/env python3
"""Post-hoc descriptive analysis of Natural Primary failure structure."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Callable


MODEL_ORDER = (
    "qwen3_vl_8b_instruct",
    "internvl3_5_8b_instruct",
    "gemma3_12b_it",
    "llava_onevision2_8b_instruct",
)

EXPECTED_EDITED_BINDING_ERRORS = {
    "qwen3_vl_8b_instruct": 7,
    "internvl3_5_8b_instruct": 6,
    "gemma3_12b_it": 14,
    "llava_onevision2_8b_instruct": 4,
}

CORRECTNESS_FIELDS = (
    "original_a_color",
    "original_b_color",
    "original_binding",
    "edited_a_color",
    "edited_b_color",
    "edited_binding",
)

PREREQUISITE_FIELDS = CORRECTNESS_FIELDS[:-1]

ROLE_KEYS = {
    ("original", "object_a_color"),
    ("original", "object_b_color"),
    ("original", "binding_choice"),
    ("edited", "object_a_color"),
    ("edited", "object_b_color"),
    ("edited", "binding_choice"),
}

CONTINUOUS_METADATA = (
    "mask_image_area_fraction_a",
    "mask_image_area_fraction_b",
    "mask_predicted_iou_a",
    "mask_predicted_iou_b",
    "selected_color_ratio_a",
    "selected_color_ratio_b",
    "low_saturation_ratio_a",
    "low_saturation_ratio_b",
    "source_target_hue_distance_degrees_a",
    "source_target_hue_distance_degrees_b",
)


def parse_args() -> argparse.Namespace:
    root = Path(__file__).resolve().parents[2]
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--project-root", type=Path, default=root)
    parser.add_argument(
        "--primary-metrics",
        type=Path,
        default=Path(
            "processed/attribute_binding/primary_analysis/v1.2/"
            "primary_model_metrics_parser_amendment1.json"
        ),
    )
    parser.add_argument(
        "--candidate-outcomes",
        type=Path,
        default=Path(
            "processed/attribute_binding/primary_analysis/v1.2/"
            "primary_candidate_outcomes_parser_amendment1.jsonl"
        ),
    )
    parser.add_argument(
        "--query-manifest",
        type=Path,
        default=Path("data/manifests/attribute_binding/primary_queries_v1.2.jsonl"),
    )
    parser.add_argument(
        "--main-source",
        type=Path,
        default=Path(
            "processed/attribute_binding/main_experiment/v1.1/preparation/"
            "main_input_manifest_v1.1.jsonl"
        ),
    )
    parser.add_argument(
        "--segmentation-results",
        type=Path,
        default=Path(
            "processed/attribute_binding/main_experiment/v1.1/segmentation/"
            "mask_results_main_v1.1.jsonl"
        ),
    )
    parser.add_argument(
        "--pre-edit-gate",
        type=Path,
        default=Path(
            "processed/attribute_binding/main_experiment/v1.1/pre_edit_gate/"
            "main_v1.1/pre_edit_technical_eligibility_main_v1.1.jsonl"
        ),
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("processed/attribute_binding/primary_analysis/v1.2"),
    )
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def resolve(root: Path, path: Path) -> Path:
    return path if path.is_absolute() else root / path


def read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    with path.open(encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def index_unique(rows: list[dict[str, Any]], field: str, label: str) -> dict[str, dict[str, Any]]:
    indexed: dict[str, dict[str, Any]] = {}
    for row in rows:
        value = row.get(field)
        if not isinstance(value, str) or value in indexed:
            raise ValueError(f"{label}: missing or duplicate {field}: {value}")
        indexed[value] = row
    return indexed


def percent(count: int, total: int) -> dict[str, Any]:
    return {
        "n": count,
        "denominator": total,
        "percent": count / total * 100 if total else None,
    }


def quantile(values: list[float], probability: float) -> float | None:
    if not values:
        return None
    ordered = sorted(values)
    position = (len(ordered) - 1) * probability
    lower = int(position)
    upper = min(lower + 1, len(ordered) - 1)
    weight = position - lower
    return ordered[lower] * (1 - weight) + ordered[upper] * weight


def continuous_summary(values: list[Any]) -> dict[str, Any]:
    numeric = [float(value) for value in values if isinstance(value, (int, float))]
    return {
        "n": len(numeric),
        "median": quantile(numeric, 0.5),
        "q1": quantile(numeric, 0.25),
        "q3": quantile(numeric, 0.75),
    }


def transition_key(original_pass: bool, edited_pass: bool) -> str:
    return f"{'pass' if original_pass else 'fail'}_to_{'pass' if edited_pass else 'fail'}"


def object_transition_key(original_correct: bool, edited_correct: bool) -> str:
    return (
        f"{'correct' if original_correct else 'incorrect'}_to_"
        f"{'correct' if edited_correct else 'incorrect'}"
    )


def pair_failure_type(a_correct: bool, b_correct: bool) -> str:
    if a_correct and b_correct:
        return "both_pass"
    if not a_correct and b_correct:
        return "a_only_failure"
    if a_correct and not b_correct:
        return "b_only_failure"
    return "a_and_b_failure"


def validate_queries(rows: list[dict[str, Any]]) -> list[str]:
    if len(rows) != 546:
        raise ValueError(f"query count {len(rows)} != 546")
    ids = [row.get("query_id") for row in rows]
    if None in ids or len(set(ids)) != 546:
        raise ValueError("query IDs are missing or duplicated")
    candidate_order = list(dict.fromkeys(row["candidate_id"] for row in rows))
    if len(candidate_order) != 91:
        raise ValueError(f"query candidate count {len(candidate_order)} != 91")
    grouped: dict[str, set[tuple[str, str]]] = defaultdict(set)
    for row in rows:
        grouped[row["candidate_id"]].add((row.get("state"), row.get("task_type")))
    for candidate_id in candidate_order:
        if grouped[candidate_id] != ROLE_KEYS:
            raise ValueError(f"{candidate_id}: six frozen query roles differ")
    return candidate_order


def validate_outcomes(
    outcomes: list[dict[str, Any]], candidate_order: list[str], primary_metrics: dict[str, Any]
) -> dict[str, list[dict[str, Any]]]:
    if len(outcomes) != 364:
        raise ValueError(f"candidate outcome count {len(outcomes)} != 364")
    by_model: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in outcomes:
        by_model[row["model_key"]].append(row)
    if tuple(by_model) != MODEL_ORDER:
        raise ValueError(f"model order differs: {tuple(by_model)}")
    metric_by_model = {row["model_key"]: row for row in primary_metrics["model_results"]}
    for model_key in MODEL_ORDER:
        rows = by_model[model_key]
        if [row["candidate_id"] for row in rows] != candidate_order:
            raise ValueError(f"{model_key}: candidate order/membership differs")
        if len(rows) != 91 or len({row["candidate_id"] for row in rows}) != 91:
            raise ValueError(f"{model_key}: candidate completeness failed")
        if sum(row["stage"] for row in []) != 0:
            raise AssertionError("unreachable")
        stage_counts = Counter(row["stage"] for row in rows)
        if sum(stage_counts.values()) != 91:
            raise ValueError(f"{model_key}: stage sum differs")
        for row in rows:
            expected_eligible = all(row[field] for field in PREREQUISITE_FIELDS)
            if row["crf_eligible"] is not expected_eligible:
                raise ValueError(f"{model_key}/{row['candidate_id']}: eligibility differs")
            if row["crf_failure"] is not (expected_eligible and not row["edited_binding"]):
                raise ValueError(f"{model_key}/{row['candidate_id']}: CRF differs")
            if row["stage"] == 4 and not row["crf_failure"]:
                raise ValueError(f"{model_key}/{row['candidate_id']}: stage 4 differs")
        metric = metric_by_model[model_key]
        if metric["crf_eligible_n"] != sum(row["crf_eligible"] for row in rows):
            raise ValueError(f"{model_key}: CRF denominator differs from confirmatory artifact")
        if metric["crf_failure_n"] != sum(row["crf_failure"] for row in rows):
            raise ValueError(f"{model_key}: CRF numerator differs from confirmatory artifact")
    return by_model


def validate_result_contracts(primary_metrics: dict[str, Any]) -> dict[str, Any]:
    checks = {}
    for summary in primary_metrics["model_results"]:
        path = Path(summary["source_result_manifest"])
        rows = read_jsonl(path)
        checks[summary["model_key"]] = {
            "queries": len(rows),
            "valid": sum(row.get("is_valid") is True for row in rows),
            "invalid": sum(row.get("is_valid") is False for row in rows),
            "inference_errors": sum(row.get("inference_error") is not None for row in rows),
            "all_amended_parser": all(
                row.get("parser_contract_version")
                == "leading_option_identifier_v1.2_amendment1"
                for row in rows
            ),
        }
        if checks[summary["model_key"]] != {
            "queries": 546,
            "valid": 546,
            "invalid": 0,
            "inference_errors": 0,
            "all_amended_parser": True,
        }:
            raise ValueError(f"{summary['model_key']}: result contract check failed")
    return checks


def edited_binding_breakdown(rows: list[dict[str, Any]]) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    errors = [row for row in rows if not row["edited_binding"]]
    detail = []
    patterns = Counter()
    cofail = Counter()
    multiple = 0
    for row in errors:
        failures = [field for field in PREREQUISITE_FIELDS if not row[field]]
        if len(failures) >= 2:
            multiple += 1
        patterns["+".join(failures) if failures else "none"] += 1
        for field in failures:
            cofail[field] += 1
        detail.append({
            "model_key": row["model_key"],
            "candidate_id": row["candidate_id"],
            **{f"{field}_correct": row[field] for field in CORRECTNESS_FIELDS},
            **{f"{field}_failure": not row[field] for field in PREREQUISITE_FIELDS},
            "prerequisite_failure_count": len(failures),
            "prerequisite_failure_pattern": "+".join(failures) if failures else "none",
            "crf_eligible": row["crf_eligible"],
        })
    return {
        "edited_binding_error_n": len(errors),
        "prerequisite_cofailures": {field: cofail[field] for field in PREREQUISITE_FIELDS},
        "multiple_prerequisite_failures_n": multiple,
        "exact_failure_patterns": dict(sorted(patterns.items())),
        "all_edited_binding_errors_crf_ineligible": all(not row["crf_eligible"] for row in errors),
        "candidate_ids": [row["candidate_id"] for row in errors],
    }, detail


def transition_analysis(rows: list[dict[str, Any]]) -> dict[str, Any]:
    pair = Counter()
    object_a = Counter()
    object_b = Counter()
    original_types = Counter()
    edited_types = Counter()
    stage3_types = Counter()
    for row in rows:
        original_pair = row["original_a_color"] and row["original_b_color"]
        edited_pair = row["edited_a_color"] and row["edited_b_color"]
        pair[transition_key(original_pair, edited_pair)] += 1
        object_a[object_transition_key(row["original_a_color"], row["edited_a_color"])] += 1
        object_b[object_transition_key(row["original_b_color"], row["edited_b_color"])] += 1
        original_types[pair_failure_type(row["original_a_color"], row["original_b_color"])] += 1
        edited_types[pair_failure_type(row["edited_a_color"], row["edited_b_color"])] += 1
        if row["stage"] == 3:
            stage3_types[pair_failure_type(row["edited_a_color"], row["edited_b_color"])] += 1
    pair_order = ("pass_to_pass", "pass_to_fail", "fail_to_pass", "fail_to_fail")
    object_order = (
        "correct_to_correct", "correct_to_incorrect",
        "incorrect_to_correct", "incorrect_to_incorrect",
    )
    failure_order = ("a_only_failure", "b_only_failure", "a_and_b_failure", "both_pass")
    return {
        "pair_level": {key: percent(pair[key], 91) for key in pair_order},
        "object_a": {key: percent(object_a[key], 91) for key in object_order},
        "object_b": {key: percent(object_b[key], 91) for key in object_order},
        "original_failure_types": {key: percent(original_types[key], 91) for key in failure_order},
        "edited_failure_types": {key: percent(edited_types[key], 91) for key in failure_order},
        "stage3_failure_types": {
            key: percent(stage3_types[key], sum(stage3_types.values()))
            for key in failure_order[:-1]
        },
    }


def outcome_predicates() -> dict[str, Callable[[dict[str, Any]], bool]]:
    return {
        "original_color_pair_failure": lambda row: not (
            row["original_a_color"] and row["original_b_color"]
        ),
        "edited_color_pair_failure": lambda row: not (
            row["edited_a_color"] and row["edited_b_color"]
        ),
        "stage1": lambda row: row["stage"] == 1,
        "stage3": lambda row: row["stage"] == 3,
        "edited_binding_error": lambda row: not row["edited_binding"],
        "all_pass": lambda row: row["stage"] == 5,
    }


def cross_model_overlap(
    by_model: dict[str, list[dict[str, Any]]], candidate_order: list[str]
) -> dict[str, Any]:
    row_maps = {
        model: {row["candidate_id"]: row for row in rows} for model, rows in by_model.items()
    }
    output = {}
    for name, predicate in outcome_predicates().items():
        candidate_models = {
            candidate_id: [
                model for model in MODEL_ORDER if predicate(row_maps[model][candidate_id])
            ]
            for candidate_id in candidate_order
        }
        distribution = Counter(len(models) for models in candidate_models.values())
        output[name] = {
            "model_count_distribution": {
                str(count): percent(distribution[count], 91) for count in range(5)
            },
            "shared_by_two_or_more_candidate_ids": [
                candidate_id for candidate_id in candidate_order
                if len(candidate_models[candidate_id]) >= 2
            ],
            "shared_by_all_four_candidate_ids": [
                candidate_id for candidate_id in candidate_order
                if len(candidate_models[candidate_id]) == 4
            ],
            "candidate_to_models": {
                candidate_id: models for candidate_id, models in candidate_models.items() if models
            },
        }
    return output


def metadata_by_candidate(
    candidate_order: list[str], main_path: Path, segmentation_path: Path, gate_path: Path
) -> tuple[dict[str, dict[str, Any]], dict[str, Any]]:
    main = index_unique(read_jsonl(main_path), "candidate_id", "main source")
    segmentation = index_unique(read_jsonl(segmentation_path), "candidate_id", "segmentation")
    gate = index_unique(read_jsonl(gate_path), "candidate_id", "pre-edit gate")
    metadata = {}
    for candidate_id in candidate_order:
        if candidate_id not in main or candidate_id not in segmentation or candidate_id not in gate:
            raise ValueError(f"{candidate_id}: difficulty provenance is incomplete")
        source = main[candidate_id]
        seg = segmentation[candidate_id]
        gate_row = gate[candidate_id]
        seg_metrics = seg.get("automatic_metrics", {})
        object_a = seg_metrics.get("object_a", {})
        object_b = seg_metrics.get("object_b", {})
        diag_a = gate_row.get("object_a_diagnostic", {})
        diag_b = gate_row.get("object_b_diagnostic", {})
        categorical = []
        categorical.extend(f"difficulty_tag:{value}" for value in source.get("difficulty_tags", []))
        categorical.extend(
            f"technical_reason:{value}" for value in source.get("technical_reason_codes", [])
        )
        categorical.extend(
            f"segmentation_flag:{value}" for value in seg.get("automatic_diagnostic_flags", [])
        )
        metadata[candidate_id] = {
            "mask_image_area_fraction_a": object_a.get("image_area_fraction"),
            "mask_image_area_fraction_b": object_b.get("image_area_fraction"),
            "mask_predicted_iou_a": object_a.get("predicted_iou"),
            "mask_predicted_iou_b": object_b.get("predicted_iou"),
            "selected_color_ratio_a": diag_a.get("selected_color_ratio"),
            "selected_color_ratio_b": diag_b.get("selected_color_ratio"),
            "low_saturation_ratio_a": diag_a.get("low_saturation_ratio"),
            "low_saturation_ratio_b": diag_b.get("low_saturation_ratio"),
            "source_target_hue_distance_degrees_a": diag_a.get(
                "source_target_hue_distance_degrees"
            ),
            "source_target_hue_distance_degrees_b": diag_b.get(
                "source_target_hue_distance_degrees"
            ),
            "categorical_tags": sorted(set(categorical)),
        }
    provenance = {
        "source_artifacts": {
            str(main_path): sha256_file(main_path),
            str(segmentation_path): sha256_file(segmentation_path),
            str(gate_path): sha256_file(gate_path),
        },
        "available_continuous_fields": list(CONTINUOUS_METADATA),
        "available_categorical_namespaces": [
            "difficulty_tags", "technical_reason_codes", "automatic_diagnostic_flags"
        ],
        "unavailable_without_new_derivation": [
            "minimum_object_area_fraction",
            "a_b_area_imbalance",
            "new_result-informed_difficulty_score",
        ],
        "quantile_method": "linear interpolation at p*(n-1)",
    }
    return metadata, provenance


def summarize_metadata_group(
    candidate_ids: list[str], metadata: dict[str, dict[str, Any]]
) -> dict[str, Any]:
    categorical = Counter(
        tag for candidate_id in candidate_ids for tag in metadata[candidate_id]["categorical_tags"]
    )
    return {
        "candidate_n": len(candidate_ids),
        "continuous": {
            field: continuous_summary([metadata[candidate_id][field] for candidate_id in candidate_ids])
            for field in CONTINUOUS_METADATA
        },
        "categorical": {
            tag: percent(count, len(candidate_ids)) for tag, count in sorted(categorical.items())
        },
    }


def difficulty_analysis(
    by_model: dict[str, list[dict[str, Any]]], metadata: dict[str, dict[str, Any]]
) -> dict[str, Any]:
    output = {}
    for model_key, rows in by_model.items():
        groups = {
            "original_color_pair_pass": [
                row["candidate_id"] for row in rows
                if row["original_a_color"] and row["original_b_color"]
            ],
            "original_color_pair_fail": [
                row["candidate_id"] for row in rows
                if not (row["original_a_color"] and row["original_b_color"])
            ],
            "edited_color_pair_pass": [
                row["candidate_id"] for row in rows
                if row["edited_a_color"] and row["edited_b_color"]
            ],
            "edited_color_pair_fail": [
                row["candidate_id"] for row in rows
                if not (row["edited_a_color"] and row["edited_b_color"])
            ],
            "stage1": [row["candidate_id"] for row in rows if row["stage"] == 1],
            "stage3": [row["candidate_id"] for row in rows if row["stage"] == 3],
            "stage5": [row["candidate_id"] for row in rows if row["stage"] == 5],
        }
        output[model_key] = {
            group: summarize_metadata_group(candidate_ids, metadata)
            for group, candidate_ids in groups.items()
        }
    return output


def ensure_outputs(paths: list[Path], overwrite: bool) -> None:
    existing = [str(path) for path in paths if path.exists()]
    if existing and not overwrite:
        raise FileExistsError(f"post-hoc outputs already exist: {existing}")


def write_breakdown_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    fieldnames = [
        "model_key", "candidate_id",
        *[f"{field}_correct" for field in CORRECTNESS_FIELDS],
        *[f"{field}_failure" for field in PREREQUISITE_FIELDS],
        "prerequisite_failure_count", "prerequisite_failure_pattern", "crf_eligible",
    ]
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def flatten_transition_rows(analyses: dict[str, dict[str, Any]]) -> list[dict[str, Any]]:
    rows = []
    for model_key, analysis in analyses.items():
        row: dict[str, Any] = {"model_key": model_key}
        for section in (
            "pair_level", "object_a", "object_b", "original_failure_types",
            "edited_failure_types", "stage3_failure_types",
        ):
            for name, value in analysis[section].items():
                row[f"{section}__{name}__n"] = value["n"]
                row[f"{section}__{name}__denominator"] = value["denominator"]
                row[f"{section}__{name}__percent"] = value["percent"]
        rows.append(row)
    return rows


def main() -> None:
    args = parse_args()
    root = args.project_root.resolve()
    metrics_path = resolve(root, args.primary_metrics)
    outcomes_path = resolve(root, args.candidate_outcomes)
    query_path = resolve(root, args.query_manifest)
    main_path = resolve(root, args.main_source)
    segmentation_path = resolve(root, args.segmentation_results)
    gate_path = resolve(root, args.pre_edit_gate)
    output_dir = resolve(root, args.output_dir)
    structure_path = output_dir / "natural_failure_structure_parser_amendment1.json"
    breakdown_path = output_dir / "natural_edited_binding_error_breakdown_parser_amendment1.csv"
    transition_path = output_dir / "natural_color_transition_parser_amendment1.csv"
    overlap_path = output_dir / "natural_cross_model_overlap_parser_amendment1.json"
    ensure_outputs(
        [structure_path, breakdown_path, transition_path, overlap_path], args.overwrite
    )

    primary_metrics = read_json(metrics_path)
    query_rows = read_jsonl(query_path)
    candidate_order = validate_queries(query_rows)
    outcomes = read_jsonl(outcomes_path)
    by_model = validate_outcomes(outcomes, candidate_order, primary_metrics)
    result_contract_checks = validate_result_contracts(primary_metrics)

    breakdowns = {}
    breakdown_rows = []
    transitions = {}
    for model_key in MODEL_ORDER:
        breakdown, detail = edited_binding_breakdown(by_model[model_key])
        if breakdown["edited_binding_error_n"] != EXPECTED_EDITED_BINDING_ERRORS[model_key]:
            raise ValueError(f"{model_key}: Edited binding error count differs from expected")
        breakdowns[model_key] = breakdown
        breakdown_rows.extend(detail)
        transitions[model_key] = transition_analysis(by_model[model_key])

    overlap = cross_model_overlap(by_model, candidate_order)
    metadata, metadata_provenance = metadata_by_candidate(
        candidate_order, main_path, segmentation_path, gate_path
    )
    difficulty = difficulty_analysis(by_model, metadata)

    sanity = {
        "models": len(by_model),
        "candidates_each": {model: len(rows) for model, rows in by_model.items()},
        "queries_per_candidate": 6,
        "queries_per_model": 546,
        "result_contract_checks": result_contract_checks,
        "stage_sums": {
            model: sum(Counter(row["stage"] for row in rows).values())
            for model, rows in by_model.items()
        },
        "crf_eligible_equals_stage4_plus_stage5": {
            model: sum(row["crf_eligible"] for row in rows)
            == sum(row["stage"] in (4, 5) for row in rows)
            for model, rows in by_model.items()
        },
        "crf_failure_equals_stage4": {
            model: sum(row["crf_failure"] for row in rows)
            == sum(row["stage"] == 4 for row in rows)
            for model, rows in by_model.items()
        },
        "edited_binding_error_counts": {
            model: breakdowns[model]["edited_binding_error_n"] for model in MODEL_ORDER
        },
        "edited_binding_membership_preserved": len(breakdown_rows)
        == sum(EXPECTED_EDITED_BINDING_ERRORS.values()),
        "pair_transition_sums": {
            model: sum(value["n"] for value in transitions[model]["pair_level"].values())
            for model in MODEL_ORDER
        },
        "original_failure_type_sums": {
            model: sum(
                value["n"] for value in transitions[model]["original_failure_types"].values()
            )
            for model in MODEL_ORDER
        },
        "edited_failure_type_sums": {
            model: sum(
                value["n"] for value in transitions[model]["edited_failure_types"].values()
            )
            for model in MODEL_ORDER
        },
        "difficulty_metadata_candidate_coverage": len(metadata),
        "no_candidate_addition_or_removal": set(metadata) == set(candidate_order),
    }
    if not all(value == 91 for value in sanity["stage_sums"].values()):
        raise AssertionError("stage sums failed")
    if not all(value == 91 for value in sanity["pair_transition_sums"].values()):
        raise AssertionError("pair transition sums failed")
    if not all(value == 91 for value in sanity["original_failure_type_sums"].values()):
        raise AssertionError("original failure type sums failed")
    if not all(value == 91 for value in sanity["edited_failure_type_sums"].values()):
        raise AssertionError("edited failure type sums failed")
    if not all(sanity["crf_eligible_equals_stage4_plus_stage5"].values()):
        raise AssertionError("CRF eligibility identity failed")
    if not all(sanity["crf_failure_equals_stage4"].values()):
        raise AssertionError("CRF stage identity failed")
    if not sanity["edited_binding_membership_preserved"]:
        raise AssertionError("Edited binding membership failed")
    if not sanity["no_candidate_addition_or_removal"]:
        raise AssertionError("candidate membership failed")

    report = {
        "schema_version": "1.2",
        "analysis_role": "post_hoc_descriptive_natural_primary",
        "confirmatory_source_artifacts": {
            str(metrics_path): sha256_file(metrics_path),
            str(outcomes_path): sha256_file(outcomes_path),
            str(query_path): sha256_file(query_path),
        },
        "edited_binding_error_breakdown": breakdowns,
        "color_transitions_and_failure_types": transitions,
        "difficulty_metadata_provenance": metadata_provenance,
        "difficulty_descriptive_analysis": difficulty,
        "statistical_policy": {
            "formal_tests_added": False,
            "p_values_computed": False,
            "interpretation": "post-hoc descriptive associations only",
        },
        "sanity_checks": sanity,
    }

    output_dir.mkdir(parents=True, exist_ok=True)
    structure_path.write_text(
        json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    write_breakdown_csv(breakdown_path, breakdown_rows)
    transition_rows = flatten_transition_rows(transitions)
    with transition_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(transition_rows[0]))
        writer.writeheader()
        writer.writerows(transition_rows)
    overlap_path.write_text(
        json.dumps(overlap, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )

    print(json.dumps({
        "edited_binding_error_breakdown": breakdowns,
        "color_transitions_and_failure_types": transitions,
        "cross_model_overlap_summary": {
            name: {
                "model_count_distribution": value["model_count_distribution"],
                "shared_by_two_or_more_candidate_ids": value[
                    "shared_by_two_or_more_candidate_ids"
                ],
                "shared_by_all_four_candidate_ids": value[
                    "shared_by_all_four_candidate_ids"
                ],
            }
            for name, value in overlap.items()
        },
        "sanity_checks": sanity,
    }, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
