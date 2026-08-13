#!/usr/bin/env python3
"""Select and freeze the 40-candidate protocol v1.1 editing pilot."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
from collections import Counter
from pathlib import Path
from typing import Any


PRE_EDIT_FIELDS = (
    "material_suitability",
    "expected_mask_separability",
    "sufficient_visible_area",
    "swap_feasibility",
    "expected_identity_preservability",
)

REQUESTED_RISKS = (
    "expected_mask_separability",
    "swap_feasibility",
    "material_suitability",
    "sufficient_visible_area",
    "partial_occlusion",
    "thin_structure",
    "small_object",
    "patterned_surface",
    "text_or_logo",
    "low_color_contrast",
    "repeated_instances",
    "clutter",
    "overlap",
)

VISUAL_REASON_TO_RISK = {
    "visual_partial_occlusion": "partial_occlusion",
    "visual_thin_structure": "thin_structure",
    "visual_small_object": "small_object",
    "visual_small_visible_region": "small_object",
    "visual_low_contrast": "low_color_contrast",
    "visual_repeated_instances": "repeated_instances",
    "visual_clutter": "clutter",
    "visual_overlap": "overlap",
}

SUBSET_QUOTAS = {
    "smoke": {"directly_editable": 2, "segmentation_test_required": 3},
    "mini": {"directly_editable": 3, "segmentation_test_required": 7},
    "completion": {
        "directly_editable": 5,
        "segmentation_test_required": 20,
    },
}

OUTPUT_NAMES = {
    "full_jsonl": "editing_pilot_40_v1.1.jsonl",
    "full_csv": "editing_pilot_40_v1.1.csv",
    "summary": "editing_pilot_selection_summary_v1.1.json",
    "smoke": "editing_pilot_smoke_5_v1.1.jsonl",
    "mini": "editing_pilot_mini_10_v1.1.jsonl",
    "completion": "editing_pilot_completion_25_v1.1.jsonl",
}


def parse_args() -> argparse.Namespace:
    root = Path(__file__).resolve().parents[2]
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument(
        "--candidate-pool",
        type=Path,
        default=root
        / "processed/attribute_binding/candidates/main_v1/candidate_pool.jsonl",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=root / "processed/attribute_binding/editing_pilot/v1.1",
    )
    parser.add_argument("--project-root", type=Path, default=root)
    parser.add_argument("--seed", type=int, default=20260731)
    return parser.parse_args()


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    with path.open(encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def stable_key(seed: int, namespace: str, candidate_id: str) -> str:
    payload = f"{seed}:{namespace}:{candidate_id}".encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def normalize_path(project_root: Path, path_value: str) -> Path:
    path = Path(path_value)
    return path if path.is_absolute() else project_root / path


def candidate_risks(row: dict[str, Any]) -> list[str]:
    scores = row["technical_criterion_scores"]
    risks = {
        field
        for field in (
            "expected_mask_separability",
            "swap_feasibility",
            "material_suitability",
            "sufficient_visible_area",
        )
        if scores[field] < 2
    }
    difficulty = set(row.get("difficulty_tags", []))
    risks.update(difficulty & set(REQUESTED_RISKS))
    for reason in row.get("technical_reason_codes", []):
        mapped = VISUAL_REASON_TO_RISK.get(reason)
        if mapped:
            risks.add(mapped)
    return sorted(risks, key=REQUESTED_RISKS.index)


def validate_input_row(row: dict[str, Any]) -> None:
    required = {
        "candidate_id",
        "source_image_id",
        "original_image_path",
        "preview_path",
        "object_a_label",
        "object_b_label",
        "semantic_status",
        "technical_status",
        "technical_criterion_scores",
        "technical_reason_codes",
        "difficulty_tags",
        "development_exposure",
    }
    missing = required - set(row)
    if missing:
        raise ValueError(
            f"{row.get('candidate_id', '<missing>')}: missing {sorted(missing)}"
        )
    scores = row["technical_criterion_scores"]
    if set(scores) != set(PRE_EDIT_FIELDS):
        raise ValueError(
            f"{row['candidate_id']}: unexpected pre-edit score fields."
        )
    if any(scores[field] not in {0, 1, 2} for field in PRE_EDIT_FIELDS):
        raise ValueError(f"{row['candidate_id']}: scores must be 0, 1, or 2.")
    if row["technical_status"] not in {
        "directly_editable",
        "segmentation_test_required",
        "failed_after_test",
    }:
        raise ValueError(f"{row['candidate_id']}: invalid technical_status.")
    if row.get("object_identity_preservation_status") not in {
        None,
        "",
        "not_tested",
    }:
        raise ValueError(f"{row['candidate_id']}: post-edit identity was tested.")
    for field in ("mask_qc_status", "edit_qc_status"):
        if row.get(field) not in {None, "", "not_tested"}:
            raise ValueError(f"{row['candidate_id']}: {field} was tested.")


def choose_balanced_direct(
    pool: list[dict[str, Any]], count: int, seed: int
) -> list[dict[str, Any]]:
    selected: list[dict[str, Any]] = []
    remaining = list(pool)
    color_counts: Counter[str] = Counter()
    while len(selected) < count:
        remaining.sort(
            key=lambda row: (
                color_counts[row["color_pair"]],
                stable_key(seed, "direct-selection", row["candidate_id"]),
            )
        )
        chosen = remaining.pop(0)
        selected.append(chosen)
        color_counts[chosen["color_pair"]] += 1
    return selected


def choose_stratified_test(
    pool: list[dict[str, Any]], count: int, seed: int
) -> list[dict[str, Any]]:
    available_counts = Counter(
        risk for row in pool for risk in row["risk_types"]
    )
    selected: list[dict[str, Any]] = []
    remaining = list(pool)
    selected_risk_counts: Counter[str] = Counter()
    stratum_counts: Counter[str] = Counter()

    while len(selected) < count:
        ranked: list[tuple[tuple[Any, ...], dict[str, Any]]] = []
        for row in remaining:
            risks = row["risk_types"] or ["other_technical_risk"]
            uncovered = sum(selected_risk_counts[risk] == 0 for risk in risks)
            balance = sum(
                1.0 / (selected_risk_counts[risk] + 1) for risk in risks
            )
            rarity = sum(
                1.0 / max(available_counts[risk], 1) for risk in risks
            )
            score = (
                -uncovered,
                -balance,
                -rarity,
                stable_key(seed, "test-selection", row["candidate_id"]),
            )
            ranked.append((score, row))
        ranked.sort(key=lambda item: item[0])
        chosen = ranked[0][1]
        risks = chosen["risk_types"] or ["other_technical_risk"]
        chosen["selection_stratum"] = min(
            risks,
            key=lambda risk: (
                stratum_counts[risk],
                available_counts[risk],
                risk,
            ),
        )
        stratum_counts[chosen["selection_stratum"]] += 1
        for risk in risks:
            selected_risk_counts[risk] += 1
        selected.append(chosen)
        remaining.remove(chosen)
    return selected


def choose_diverse_subset(
    pool: list[dict[str, Any]], count: int, seed: int, namespace: str
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    selected: list[dict[str, Any]] = []
    remaining = list(pool)
    risk_counts: Counter[str] = Counter()
    while len(selected) < count:
        remaining.sort(
            key=lambda row: (
                -sum(
                    1.0 / (risk_counts[risk] + 1)
                    for risk in (row["risk_types"] or ["other_technical_risk"])
                ),
                stable_key(seed, namespace, row["candidate_id"]),
            )
        )
        chosen = remaining.pop(0)
        selected.append(chosen)
        for risk in chosen["risk_types"] or ["other_technical_risk"]:
            risk_counts[risk] += 1
    return selected, remaining


def assign_subsets(
    direct: list[dict[str, Any]],
    test_required: list[dict[str, Any]],
    seed: int,
) -> list[dict[str, Any]]:
    direct_remaining = sorted(
        direct,
        key=lambda row: stable_key(seed, "subset-direct", row["candidate_id"]),
    )
    test_remaining = list(test_required)
    assigned: list[dict[str, Any]] = []
    for subset in ("smoke", "mini", "completion"):
        direct_count = SUBSET_QUOTAS[subset]["directly_editable"]
        test_count = SUBSET_QUOTAS[subset]["segmentation_test_required"]
        subset_direct = direct_remaining[:direct_count]
        direct_remaining = direct_remaining[direct_count:]
        subset_test, test_remaining = choose_diverse_subset(
            test_remaining,
            test_count,
            seed,
            f"subset-{subset}",
        )
        subset_rows = subset_direct + subset_test
        subset_rows.sort(
            key=lambda row: stable_key(
                seed, f"subset-order-{subset}", row["candidate_id"]
            )
        )
        for row in subset_rows:
            row["pilot_subset"] = subset
        assigned.extend(subset_rows)
    if direct_remaining or test_remaining:
        raise AssertionError("Subset assignment did not consume all selections.")
    return assigned


def write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    with path.open("x", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    fields = list(dict.fromkeys(key for row in rows for key in row))
    with path.open("x", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for source in rows:
            row = dict(source)
            for field in (
                "technical_reason_codes",
                "difficulty_tags",
                "risk_types",
            ):
                row[field] = ";".join(row[field])
            writer.writerow(row)


def counts_by(rows: list[dict[str, Any]], field: str) -> dict[str, int]:
    return dict(sorted(Counter(str(row[field]) for row in rows).items()))


def list_counts(rows: list[dict[str, Any]], field: str) -> dict[str, int]:
    return dict(
        sorted(Counter(value for row in rows for value in row[field]).items())
    )


def main() -> None:
    args = parse_args()
    output_paths = {
        name: args.output_dir / filename
        for name, filename in OUTPUT_NAMES.items()
    }
    existing = [path for path in output_paths.values() if path.exists()]
    if existing:
        raise FileExistsError(f"Refusing to overwrite outputs: {existing}")

    rows = read_jsonl(args.input)
    pool_rows = read_jsonl(args.candidate_pool)
    if len(rows) != 293:
        raise ValueError(f"Expected 293 authoritative rows, found {len(rows)}.")
    for row in rows:
        validate_input_row(row)

    pool_by_id = {row["candidate_id"]: row for row in pool_rows}
    if len(pool_by_id) != len(pool_rows):
        raise ValueError("Duplicate candidate IDs in candidate pool.")

    enriched: list[dict[str, Any]] = []
    missing_path_records: list[dict[str, Any]] = []
    for source in rows:
        if source["semantic_status"] != "valid":
            continue
        if source["technical_status"] == "failed_after_test":
            continue
        if source["development_exposure"] is not True:
            raise ValueError(
                f"{source['candidate_id']}: development_exposure must be true."
            )
        pool_row = pool_by_id.get(source["candidate_id"])
        if pool_row is None:
            raise ValueError(
                f"{source['candidate_id']}: missing candidate-pool provenance."
            )
        missing = [
            field
            for field in ("original_image_path", "preview_path")
            if not normalize_path(args.project_root, source[field]).is_file()
        ]
        if missing:
            missing_path_records.append(
                {"candidate_id": source["candidate_id"], "missing": missing}
            )
            continue
        scores = source["technical_criterion_scores"]
        row = {
            "candidate_id": source["candidate_id"],
            "source_image_id": source["source_image_id"],
            "original_image_path": source["original_image_path"],
            "preview_path": source["preview_path"],
            "object_a_label": source["object_a_label"],
            "object_b_label": source["object_b_label"],
            "original_color_a": pool_row["object_a_color"],
            "original_color_b": pool_row["object_b_color"],
            "color_pair": pool_row["color_pair"],
            "semantic_status": source["semantic_status"],
            "technical_status": source["technical_status"],
            **{field: scores[field] for field in PRE_EDIT_FIELDS},
            "technical_reason_codes": list(
                source.get("technical_reason_codes", [])
            ),
            "difficulty_tags": list(source.get("difficulty_tags", [])),
            "risk_types": candidate_risks(source),
            "pilot_subset": None,
            "selection_seed": args.seed,
            "selection_stratum": (
                "directly_editable"
                if source["technical_status"] == "directly_editable"
                else None
            ),
            "selection_rank": None,
            "development_exposure": True,
            "previous_label": source.get("previous_label"),
            "mask_qc_status": "not_tested",
            "edit_qc_status": "not_tested",
            "object_identity_preservation_status": "not_tested",
        }
        enriched.append(row)

    source_counts = Counter(row["source_image_id"] for row in enriched)
    duplicate_sources = {
        source_id for source_id, count in source_counts.items() if count > 1
    }
    if duplicate_sources:
        raise ValueError(
            f"Eligible input has duplicate source_image_id values: "
            f"{sorted(duplicate_sources)}"
        )

    direct_pool = [
        row for row in enriched if row["technical_status"] == "directly_editable"
    ]
    test_pool = [
        row
        for row in enriched
        if row["technical_status"] == "segmentation_test_required"
    ]
    if len(direct_pool) < 10 or len(test_pool) < 30:
        raise ValueError(
            f"Insufficient eligible candidates: direct={len(direct_pool)}, "
            f"test_required={len(test_pool)}."
        )

    direct_selected = choose_balanced_direct(direct_pool, 10, args.seed)
    test_selected = choose_stratified_test(test_pool, 30, args.seed)
    selected = assign_subsets(direct_selected, test_selected, args.seed)
    for rank, row in enumerate(selected, start=1):
        row["selection_rank"] = rank

    selected_sources = [row["source_image_id"] for row in selected]
    duplicate_source_count = len(selected_sources) - len(set(selected_sources))
    if duplicate_source_count:
        raise AssertionError("Selected pilot contains duplicate source images.")
    if Counter(row["technical_status"] for row in selected) != Counter(
        {"directly_editable": 10, "segmentation_test_required": 30}
    ):
        raise AssertionError("Technical-status quota mismatch.")
    expected_subset_counts = {"smoke": 5, "mini": 10, "completion": 25}
    if Counter(row["pilot_subset"] for row in selected) != Counter(
        expected_subset_counts
    ):
        raise AssertionError("Pilot subset quota mismatch.")

    selected_risk_counts = Counter(
        risk for row in test_selected for risk in row["risk_types"]
    )
    available_risk_counts = Counter(
        risk for row in test_pool for risk in row["risk_types"]
    )
    insufficient_strata = [
        {
            "risk_type": risk,
            "available": available_risk_counts[risk],
            "selected": selected_risk_counts[risk],
            "reason": (
                "no eligible candidate in frozen input"
                if available_risk_counts[risk] == 0
                else "available but not represented within 30-slot multilabel design"
            ),
        }
        for risk in REQUESTED_RISKS
        if available_risk_counts[risk] == 0 or selected_risk_counts[risk] == 0
    ]

    subset_rows = {
        subset: [row for row in selected if row["pilot_subset"] == subset]
        for subset in ("smoke", "mini", "completion")
    }
    summary = {
        "schema_version": "1.1",
        "selection_status": "frozen_pre_execution",
        "input_manifest": args.input.as_posix(),
        "input_manifest_sha256": file_sha256(args.input),
        "candidate_pool": args.candidate_pool.as_posix(),
        "candidate_pool_sha256": file_sha256(args.candidate_pool),
        "selection_seed": args.seed,
        "selected_total": len(selected),
        "subset_counts": {
            subset: len(rows_) for subset, rows_ in subset_rows.items()
        },
        "technical_status_counts": counts_by(selected, "technical_status"),
        "technical_status_by_subset": {
            subset: counts_by(rows_, "technical_status")
            for subset, rows_ in subset_rows.items()
        },
        "technical_reason_counts": list_counts(
            selected, "technical_reason_codes"
        ),
        "difficulty_tag_counts": list_counts(selected, "difficulty_tags"),
        "risk_type_counts": list_counts(selected, "risk_types"),
        "selection_stratum_counts": counts_by(
            selected, "selection_stratum"
        ),
        "color_pair_counts": counts_by(selected, "color_pair"),
        "previous_label_provenance_counts": counts_by(
            selected, "previous_label"
        ),
        "duplicate_source_count": duplicate_source_count,
        "missing_path_count": len(missing_path_records),
        "missing_path_records": missing_path_records,
        "available_pool": {
            "directly_editable": len(direct_pool),
            "segmentation_test_required": len(test_pool),
        },
        "available_risk_counts": dict(sorted(available_risk_counts.items())),
        "insufficient_or_unrepresented_strata": insufficient_strata,
        "replacement_rule": {
            "replacement_after_freeze": False,
            "allowed_pre_experimental_causes": [
                "missing_file",
                "manifest_schema_error",
                "duplicate_source_image_id",
            ],
            "procedure": (
                "Before any experimental output exists, skip the invalid entry "
                "and take the next deterministic candidate from the same "
                "technical-status pool, preserving the requested risk stratum "
                "when available. Record the event; never replace based on mask, "
                "edit, QC, or model outcomes."
            ),
        },
        "selection_safeguards": {
            "semantic_valid_only": True,
            "development_exposure_required": True,
            "failed_after_test_excluded": True,
            "previous_label_used_for_selection": False,
            "model_results_used": False,
            "segmentation_results_used": False,
            "editing_results_used": False,
            "source_image_duplicates_allowed": False,
        },
        "selected_candidate_ids": [
            row["candidate_id"] for row in selected
        ],
    }

    args.output_dir.mkdir(parents=True, exist_ok=True)
    write_jsonl(output_paths["full_jsonl"], selected)
    write_csv(output_paths["full_csv"], selected)
    for subset in ("smoke", "mini", "completion"):
        write_jsonl(output_paths[subset], subset_rows[subset])
    output_paths["summary"].write_text(
        json.dumps(summary, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    print(
        json.dumps(
            {
                "validation_status": "pass",
                "selected_total": len(selected),
                "subset_counts": summary["subset_counts"],
                "technical_status_counts": summary[
                    "technical_status_counts"
                ],
                "duplicate_source_count": duplicate_source_count,
                "missing_path_count": len(missing_path_records),
                "failed_after_test_selected": 0,
                "development_exposure_false": 0,
                "insufficient_or_unrepresented_strata": insufficient_strata,
                "output_dir": args.output_dir.as_posix(),
            },
            ensure_ascii=False,
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
