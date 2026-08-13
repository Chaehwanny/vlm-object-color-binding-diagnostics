#!/usr/bin/env python3
"""Analyze the frozen Natural Primary attribute-binding results for v1.2."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any


MODEL_FILES = {
    "qwen3_vl_8b_instruct": "qwen3_vl_8b_instruct_parser_amendment1.jsonl",
    "internvl3_5_8b_instruct": "internvl3_5_8b_instruct_parser_amendment1.jsonl",
    "gemma3_12b_it": "gemma3_12b_it_parser_amendment1.jsonl",
    "llava_onevision2_8b_instruct": "llava_onevision2_8b_instruct_parser_amendment1.jsonl",
}

ROLE_ORDER = (
    ("original", "object_a_color"),
    ("original", "object_b_color"),
    ("original", "binding_choice"),
    ("edited", "object_a_color"),
    ("edited", "object_b_color"),
    ("edited", "binding_choice"),
)

ROLE_NAMES = {
    ("original", "object_a_color"): "original_a_color",
    ("original", "object_b_color"): "original_b_color",
    ("original", "binding_choice"): "original_binding",
    ("edited", "object_a_color"): "edited_a_color",
    ("edited", "object_b_color"): "edited_b_color",
    ("edited", "binding_choice"): "edited_binding",
}

STAGE_NAMES = {
    1: "original_atomic_color_failure",
    2: "original_binding_failure_after_atomic_pass",
    3: "edited_atomic_color_failure_after_original_pass",
    4: "conditional_rebinding_failure",
    5: "full_paired_pass",
}


def parse_args() -> argparse.Namespace:
    root = Path(__file__).resolve().parents[2]
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--project-root", type=Path, default=root)
    parser.add_argument(
        "--query-manifest",
        type=Path,
        default=Path("data/manifests/attribute_binding/primary_queries_v1.2.jsonl"),
    )
    parser.add_argument(
        "--config",
        type=Path,
        default=Path(
            "configs/attribute_binding/primary_inference_v1.2_parser_amendment1.json"
        ),
    )
    parser.add_argument(
        "--results-dir",
        type=Path,
        default=Path("processed/attribute_binding/primary_inference/v1.2"),
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


def wilson_interval(numerator: int, denominator: int) -> dict[str, float] | None:
    if denominator == 0:
        return None
    z = 1.959963984540054
    p = numerator / denominator
    z2 = z * z
    center = (p + z2 / (2 * denominator)) / (1 + z2 / denominator)
    half = (
        z
        * math.sqrt(
            p * (1 - p) / denominator + z2 / (4 * denominator * denominator)
        )
        / (1 + z2 / denominator)
    )
    return {"low": max(0.0, center - half), "high": min(1.0, center + half)}


def proportion(numerator: int, denominator: int) -> dict[str, Any]:
    rate = numerator / denominator if denominator else None
    return {
        "numerator": numerator,
        "denominator": denominator,
        "rate": rate,
        "percent": rate * 100 if rate is not None else None,
        "wilson_95_ci": wilson_interval(numerator, denominator),
    }


def check_unique_ids(rows: list[dict[str, Any]], field: str, label: str) -> None:
    values = [row.get(field) for row in rows]
    duplicates = sorted(value for value, count in Counter(values).items() if count > 1)
    if None in values or duplicates:
        raise ValueError(f"{label} has missing or duplicate {field}: {duplicates}")


def prepare_queries(
    rows: list[dict[str, Any]], expected_candidates: int, expected_queries: int
) -> tuple[list[str], dict[str, dict[tuple[str, str], dict[str, Any]]]]:
    if len(rows) != expected_queries:
        raise ValueError(f"query count {len(rows)} != {expected_queries}")
    check_unique_ids(rows, "query_id", "query manifest")
    candidate_order = list(dict.fromkeys(row["candidate_id"] for row in rows))
    if len(candidate_order) != expected_candidates:
        raise ValueError(f"candidate count {len(candidate_order)} != {expected_candidates}")

    by_candidate: dict[str, dict[tuple[str, str], dict[str, Any]]] = defaultdict(dict)
    for row in rows:
        role = (row.get("state"), row.get("task_type"))
        if role not in ROLE_ORDER:
            raise ValueError(f"{row['query_id']}: unexpected query role {role}")
        candidate_id = row["candidate_id"]
        if role in by_candidate[candidate_id]:
            raise ValueError(f"{candidate_id}: duplicate query role {role}")
        by_candidate[candidate_id][role] = row
    expected_roles = set(ROLE_ORDER)
    for candidate_id in candidate_order:
        actual = set(by_candidate[candidate_id])
        if actual != expected_roles:
            raise ValueError(
                f"{candidate_id}: roles differ; missing={sorted(expected_roles-actual)}, "
                f"unexpected={sorted(actual-expected_roles)}"
            )
    return candidate_order, by_candidate


def stage_for(correct: dict[str, bool]) -> int:
    if not correct["original_a_color"] or not correct["original_b_color"]:
        return 1
    if not correct["original_binding"]:
        return 2
    if not correct["edited_a_color"] or not correct["edited_b_color"]:
        return 3
    if not correct["edited_binding"]:
        return 4
    return 5


def analyze_model(
    model_key: str,
    result_path: Path,
    model_config: dict[str, Any],
    query_rows: list[dict[str, Any]],
    candidate_order: list[str],
    queries_by_candidate: dict[str, dict[tuple[str, str], dict[str, Any]]],
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    rows = read_jsonl(result_path)
    check_unique_ids(rows, "query_id", model_key)
    query_ids = [row["query_id"] for row in query_rows]
    result_ids = [row["query_id"] for row in rows]
    if result_ids != query_ids:
        raise ValueError(f"{model_key}: result order/membership differs from frozen queries")

    query_by_id = {row["query_id"]: row for row in query_rows}
    result_by_id = {row["query_id"]: row for row in rows}
    for row in rows:
        query = query_by_id[row["query_id"]]
        if row.get("candidate_id") != query["candidate_id"]:
            raise ValueError(f"{row['query_id']}: candidate differs from query manifest")
        if row.get("gold_option_id") != query["correct_option_id"]:
            raise ValueError(f"{row['query_id']}: gold option differs from query manifest")
        expected_correct = (
            row.get("inference_error") is None
            and row.get("is_valid") is True
            and row.get("parsed_option_id") == query["correct_option_id"]
        )
        if row.get("is_correct") is not expected_correct:
            raise ValueError(f"{row['query_id']}: correctness disagrees with frozen gold")
        if row.get("model_id") != model_config["model_id"]:
            raise ValueError(f"{row['query_id']}: model ID differs")
        if row.get("model_revision") != model_config["model_revision"]:
            raise ValueError(f"{row['query_id']}: model revision differs")

    outcomes = []
    for candidate_id in candidate_order:
        correct: dict[str, bool] = {}
        responses: dict[str, dict[str, Any]] = {}
        for role in ROLE_ORDER:
            name = ROLE_NAMES[role]
            query = queries_by_candidate[candidate_id][role]
            result = result_by_id[query["query_id"]]
            correct[name] = result.get("is_correct") is True
            responses[name] = {
                "query_id": query["query_id"],
                "is_valid": result.get("is_valid") is True,
                "is_correct": result.get("is_correct") is True,
                "parsed_option_id": result.get("parsed_option_id"),
                "gold_option_id": query["correct_option_id"],
                "inference_error": result.get("inference_error"),
            }
        stage = stage_for(correct)
        eligible = all(
            correct[name]
            for name in (
                "original_a_color",
                "original_b_color",
                "original_binding",
                "edited_a_color",
                "edited_b_color",
            )
        )
        crf_failure = eligible and not correct["edited_binding"]
        all_pass = stage == 5
        if eligible != (crf_failure or all_pass):
            raise AssertionError(f"{model_key}/{candidate_id}: CRF identity failed")
        outcomes.append(
            {
                "schema_version": "1.2",
                "model_key": model_key,
                "model_id": model_config["model_id"],
                "candidate_id": candidate_id,
                "primary_domain_order": queries_by_candidate[candidate_id][ROLE_ORDER[0]].get(
                    "primary_domain_order"
                ),
                **correct,
                "crf_eligible": eligible,
                "crf_failure": crf_failure,
                "all_pass": all_pass,
                "stage": stage,
                "stage_name": STAGE_NAMES[stage],
                "responses": responses,
            }
        )

    n = len(outcomes)
    if n != len(candidate_order):
        raise AssertionError(f"{model_key}: candidate outcome count differs")
    stage_counts = Counter(row["stage"] for row in outcomes)
    if sum(stage_counts.values()) != n or set(stage_counts) - set(STAGE_NAMES):
        raise AssertionError(f"{model_key}: stage decomposition failed")

    original_binding = sum(row["original_binding"] for row in outcomes)
    edited_binding = sum(row["edited_binding"] for row in outcomes)
    paired_binding = sum(
        row["original_binding"] and row["edited_binding"] for row in outcomes
    )
    eligible_n = sum(row["crf_eligible"] for row in outcomes)
    failure_n = sum(row["crf_failure"] for row in outcomes)
    all_pass_n = sum(row["all_pass"] for row in outcomes)
    if eligible_n != failure_n + all_pass_n:
        raise AssertionError(f"{model_key}: eligible != CRF failures + all pass")

    summary = {
        "model_key": model_key,
        "model_id": model_config["model_id"],
        "model_revision": model_config["model_revision"],
        "source_result_manifest": str(result_path),
        "source_result_sha256": sha256_file(result_path),
        "structural_counts": {
            "candidates": n,
            "queries": len(rows),
            "valid": sum(row.get("is_valid") is True for row in rows),
            "invalid": sum(
                row.get("is_valid") is False and row.get("inference_error") is None
                for row in rows
            ),
            "inference_errors": sum(row.get("inference_error") is not None for row in rows),
        },
        "metrics": {
            "OBA": proportion(original_binding, n),
            "EBA": proportion(edited_binding, n),
            "PBC": proportion(paired_binding, n),
            "CRF": proportion(failure_n, eligible_n),
        },
        "crf_eligible_n": eligible_n,
        "crf_failure_n": failure_n,
        "all_pass_n": all_pass_n,
        "crf_failure_candidate_ids": [
            row["candidate_id"] for row in outcomes if row["crf_failure"]
        ],
        "stages": {
            STAGE_NAMES[stage]: proportion(stage_counts.get(stage, 0), n)
            for stage in STAGE_NAMES
        },
        "sanity_checks": {
            "stage_sum": sum(stage_counts.values()),
            "crf_failures_subset_of_eligible": all(
                not row["crf_failure"] or row["crf_eligible"] for row in outcomes
            ),
            "all_pass_subset_of_eligible": all(
                not row["all_pass"] or row["crf_eligible"] for row in outcomes
            ),
            "eligible_equals_crf_failure_plus_all_pass": eligible_n
            == failure_n + all_pass_n,
        },
    }
    return summary, outcomes


def cross_model_overlap(
    summaries: list[dict[str, Any]], candidate_order: list[str]
) -> dict[str, Any]:
    failures = {
        summary["model_key"]: set(summary["crf_failure_candidate_ids"])
        for summary in summaries
    }
    candidate_models = {
        candidate_id: sorted(key for key, values in failures.items() if candidate_id in values)
        for candidate_id in candidate_order
    }
    candidate_models = {key: value for key, value in candidate_models.items() if value}
    by_count = {
        str(count): sorted(
            candidate_id
            for candidate_id, models in candidate_models.items()
            if len(models) == count
        )
        for count in range(1, len(summaries) + 1)
    }
    exact_patterns: dict[str, list[str]] = defaultdict(list)
    for candidate_id, models in candidate_models.items():
        exact_patterns["+".join(models)].append(candidate_id)
    return {
        "matched_candidate_n": len(candidate_order),
        "model_observations_are_not_pooled": True,
        "candidate_to_failing_models": candidate_models,
        "failure_ids_by_model_count": by_count,
        "one_model_only": by_count["1"],
        "two_or_more_models": sorted(
            candidate_id for candidate_id, models in candidate_models.items() if len(models) >= 2
        ),
        "exact_model_combinations": {
            key: sorted(value) for key, value in sorted(exact_patterns.items())
        },
    }


def ensure_outputs(paths: list[Path], overwrite: bool) -> None:
    existing = [str(path) for path in paths if path.exists()]
    if existing and not overwrite:
        raise FileExistsError(f"analysis outputs already exist: {existing}")


def main() -> None:
    args = parse_args()
    root = args.project_root.resolve()
    query_path = resolve(root, args.query_manifest)
    config_path = resolve(root, args.config)
    results_dir = resolve(root, args.results_dir)
    output_dir = resolve(root, args.output_dir)
    metrics_path = output_dir / "primary_model_metrics_parser_amendment1.json"
    outcomes_path = output_dir / "primary_candidate_outcomes_parser_amendment1.jsonl"
    table_path = output_dir / "primary_model_metrics_parser_amendment1.csv"
    ensure_outputs([metrics_path, outcomes_path, table_path], args.overwrite)

    config = read_json(config_path)
    query_rows = read_jsonl(query_path)
    candidate_order, queries_by_candidate = prepare_queries(
        query_rows, config["expected_candidate_count"], config["expected_query_count"]
    )

    summaries = []
    all_outcomes = []
    for model_key, filename in MODEL_FILES.items():
        if model_key not in config["models"]:
            raise ValueError(f"missing model config: {model_key}")
        summary, outcomes = analyze_model(
            model_key,
            results_dir / filename,
            config["models"][model_key],
            query_rows,
            candidate_order,
            queries_by_candidate,
        )
        summaries.append(summary)
        all_outcomes.extend(outcomes)

    report = {
        "schema_version": "1.2",
        "analysis_contract": "natural_primary_attribute_binding_v1.2_parser_amendment1",
        "governing_protocol": "docs/protocols/protocol_v1.2_PRE_INFERENCE_AMENDMENT.md",
        "metric_dictionary": "docs/protocols/metric_dictionary_v1.1.md",
        "statistical_analysis_plan": "docs/protocols/statistical_analysis_plan_v1.1.md",
        "query_manifest": str(query_path),
        "query_manifest_sha256": sha256_file(query_path),
        "candidate_n": len(candidate_order),
        "query_n": len(query_rows),
        "interval_policy": "Wilson 95% interval for descriptive proportions",
        "bootstrap_status": (
            "not_applied: no Natural-Controlled difference was analyzed and the frozen SAP "
            "does not specify a bootstrap interval construction method"
        ),
        "model_results": summaries,
        "cross_model_crf_overlap": cross_model_overlap(summaries, candidate_order),
        "global_sanity_checks": {
            "model_count": len(summaries),
            "candidate_count_each": all(
                summary["structural_counts"]["candidates"] == len(candidate_order)
                for summary in summaries
            ),
            "query_count_each": all(
                summary["structural_counts"]["queries"] == len(query_rows)
                for summary in summaries
            ),
            "six_queries_per_candidate": all(
                len(queries_by_candidate[candidate_id]) == 6 for candidate_id in candidate_order
            ),
            "all_checks_pass": all(
                all(summary["sanity_checks"].values()) for summary in summaries
            ),
        },
    }

    output_dir.mkdir(parents=True, exist_ok=True)
    metrics_path.write_text(
        json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    with outcomes_path.open("w", encoding="utf-8") as handle:
        for row in all_outcomes:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")
    with table_path.open("w", encoding="utf-8", newline="") as handle:
        fieldnames = [
            "model_key", "candidates", "queries", "valid", "invalid", "inference_errors",
            "OBA_n", "OBA_d", "OBA_rate", "EBA_n", "EBA_d", "EBA_rate",
            "PBC_n", "PBC_d", "PBC_rate", "CRF_n", "CRF_d", "CRF_rate",
            "stage1_n", "stage2_n", "stage3_n", "stage4_n", "stage5_n",
        ]
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for summary in summaries:
            metrics = summary["metrics"]
            stages = summary["stages"]
            writer.writerow({
                "model_key": summary["model_key"],
                **summary["structural_counts"],
                **{
                    f"{name}_{suffix}": metric[key]
                    for name, metric in metrics.items()
                    for suffix, key in (("n", "numerator"), ("d", "denominator"), ("rate", "rate"))
                },
                **{
                    f"stage{stage}_n": stages[STAGE_NAMES[stage]]["numerator"]
                    for stage in STAGE_NAMES
                },
            })

    compact = {
        summary["model_key"]: {
            "OBA": summary["metrics"]["OBA"],
            "EBA": summary["metrics"]["EBA"],
            "PBC": summary["metrics"]["PBC"],
            "CRF": summary["metrics"]["CRF"],
            "stage_counts": {
                name: value["numerator"] for name, value in summary["stages"].items()
            },
            "crf_failure_ids": summary["crf_failure_candidate_ids"],
        }
        for summary in summaries
    }
    print(json.dumps(compact, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
