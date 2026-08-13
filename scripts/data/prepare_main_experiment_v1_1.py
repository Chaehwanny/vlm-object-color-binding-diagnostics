#!/usr/bin/env python3
"""Freeze the held-out v1.1 main input and audit development overlap."""

from __future__ import annotations

import argparse
import hashlib
import json
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--reclassification", type=Path, required=True)
    parser.add_argument("--candidate-pool", type=Path, required=True)
    parser.add_argument("--left-right", type=Path, required=True)
    parser.add_argument("--feasibility", type=Path, required=True)
    parser.add_argument("--interim", type=Path, required=True)
    parser.add_argument("--smoke", type=Path, required=True)
    parser.add_argument("--mini", type=Path, required=True)
    parser.add_argument("--completion", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--project-root", type=Path, required=True)
    return parser.parse_args()


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    with path.open(encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def atomic_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    if path.exists():
        raise FileExistsError(f"Refusing to overwrite: {path}")
    temporary = path.with_name(f".{path.name}.tmp")
    with temporary.open("x", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")
    temporary.replace(path)


def atomic_json(path: Path, value: dict[str, Any]) -> None:
    if path.exists():
        raise FileExistsError(f"Refusing to overwrite: {path}")
    temporary = path.with_name(f".{path.name}.tmp")
    temporary.write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    temporary.replace(path)


def norm(value: Any) -> str | None:
    if value is None:
        return None
    return str(value)


def label_pair(a: Any, b: Any) -> tuple[str, str] | None:
    if not a or not b:
        return None
    return tuple(sorted((str(a).strip().lower(), str(b).strip().lower())))


def development_rows(args: argparse.Namespace) -> list[dict[str, Any]]:
    specs = (
        ("left_right_positive_control", args.left_right, "left_right"),
        ("color_feasibility", args.feasibility, "attribute"),
        ("color_interim", args.interim, "attribute"),
        ("editing_smoke", args.smoke, "editing"),
        ("editing_mini", args.mini, "editing"),
        ("editing_completion", args.completion, "editing"),
    )
    rows: list[dict[str, Any]] = []
    for split, path, kind in specs:
        source = read_jsonl(path)
        if kind == "left_right":
            unique: dict[str, dict[str, Any]] = {}
            for row in source:
                unique.setdefault(row["sample_id"], row)
            source = list(unique.values())
        for row in source:
            if kind == "left_right":
                sample_id = row["sample_id"]
                source_image_id = sample_id.rsplit("_", 1)[-1]
                candidate_id = None
                source_candidate_id = row.get("review_candidate_id")
                object_a_label = row.get("source_label")
                object_b_label = row.get("target_label")
            else:
                sample_id = row.get("sample_id") or row.get("candidate_id")
                source_image_id = norm(row.get("source_image_id"))
                raw_candidate = row.get("candidate_id")
                candidate_id = raw_candidate if str(raw_candidate or "").startswith("maincand_") else None
                source_candidate_id = row.get("source_candidate_id") or raw_candidate
                object_a_label = row.get("object_a_label")
                object_b_label = row.get("object_b_label")
            rows.append(
                {
                    "development_split": split,
                    "source_manifest_path": path.as_posix(),
                    "original_sample_id": sample_id,
                    "candidate_id": candidate_id,
                    "source_candidate_id": source_candidate_id,
                    "source_image_id": source_image_id,
                    "source_object_a_id": norm(row.get("source_object_a_id")),
                    "source_object_b_id": norm(row.get("source_object_b_id")),
                    "object_a_label": object_a_label,
                    "object_b_label": object_b_label,
                    "exclusion_reason": "development_pipeline_or_model_exposure",
                }
            )
    return rows


def main() -> None:
    args = parse_args()
    reclassification = read_jsonl(args.reclassification)
    candidate_pool = read_jsonl(args.candidate_pool)
    pool_by_id = {row.get("candidate_id"): row for row in candidate_pool}
    errors: list[str] = []
    if len(pool_by_id) != len(candidate_pool) or None in pool_by_id:
        errors.append("candidate pool has missing or duplicate candidate_id")
    authoritative_ids = [row.get("candidate_id") for row in reclassification]
    if None in authoritative_ids or len(authoritative_ids) != len(set(authoritative_ids)):
        errors.append("reclassification has missing or duplicate candidate_id")
    if set(authoritative_ids) != set(pool_by_id):
        errors.append("reclassification and candidate pool candidate IDs differ")

    valid_rows = [row for row in reclassification if row.get("semantic_status") == "valid"]
    invalid_rows = [row for row in reclassification if row.get("semantic_status") == "invalid"]
    human_rows = [row for row in reclassification if row.get("semantic_status") == "human_review"]
    valid_by_source: dict[str, list[dict[str, Any]]] = {}
    for row in valid_rows:
        valid_by_source.setdefault(str(row["source_image_id"]), []).append(row)

    development = development_rows(args)
    development_candidate_ids = {row["candidate_id"] for row in development if row["candidate_id"]}
    development_source_ids = {row["source_image_id"] for row in development if row["source_image_id"]}
    for row in development:
        sid = row["source_image_id"]
        source_matches = valid_by_source.get(str(sid), []) if sid is not None else []
        candidate_match = row["candidate_id"] in {item["candidate_id"] for item in source_matches}
        dev_pair = label_pair(row.get("object_a_label"), row.get("object_b_label"))
        different_pair = 0
        for item in source_matches:
            main_pair = label_pair(item.get("object_a_label"), item.get("object_b_label"))
            if dev_pair is not None and main_pair is not None and dev_pair != main_pair:
                different_pair += 1
        row.update(
            {
                "main_candidate_collision": candidate_match,
                "main_source_collision": bool(source_matches),
                "main_collision_candidate_ids": [item["candidate_id"] for item in source_matches],
                "same_source_different_object_pair_collision_count": different_pair,
                "collision_resolution": "exclude_source_image" if source_matches else "no_main_collision",
                "unresolved_identity_collision": False if sid is not None else True,
            }
        )

    unresolved = [row for row in development if row["unresolved_identity_collision"]]
    excluded_valid = [row for row in valid_rows if str(row["source_image_id"]) in development_source_ids]
    held_out = [row for row in valid_rows if str(row["source_image_id"]) not in development_source_ids]
    main_rows: list[dict[str, Any]] = []
    for authoritative_index, row in enumerate(reclassification):
        if row not in held_out:
            continue
        candidate_id = row["candidate_id"]
        pool = pool_by_id.get(candidate_id)
        if pool is None:
            errors.append(f"{candidate_id}: missing candidate-pool row")
            continue
        main_row = dict(row)
        main_row.update(
            {
                "pilot_subset": "main",
                "original_color_a": pool.get("object_a_color"),
                "original_color_b": pool.get("object_b_color"),
                "color_pair": pool.get("color_pair"),
                "source_image_path": pool.get("source_image_path"),
                "object_a_bbox_xywh_norm": pool.get("object_a_bbox_xywh_norm"),
                "object_b_bbox_xywh_norm": pool.get("object_b_bbox_xywh_norm"),
                "authoritative_row_index": authoritative_index,
                "frozen_order": len(main_rows) + 1,
                "frozen_order_basis": "authoritative_reclassification_manifest_row_order",
                "development_overlap_status": "none",
                "main_eligibility_status": "eligible",
                "main_exclusion_reasons": [],
                "development_exposure_interpretation": "semantic_curation_only_not_pipeline_or_model_exposure",
            }
        )
        main_rows.append(main_row)

    main_ids = [row["candidate_id"] for row in main_rows]
    pair_keys = [row.get("source_pair_key") for row in main_rows]
    if len(main_ids) != len(set(main_ids)):
        errors.append("main input has duplicate candidate_id")
    if None in pair_keys or len(pair_keys) != len(set(pair_keys)):
        errors.append("main input has missing or duplicate source_pair_key")
    if any(row["candidate_id"] not in set(authoritative_ids) for row in main_rows):
        errors.append("main input contains a new candidate")
    if any(str(row["source_image_id"]) in development_source_ids for row in main_rows):
        errors.append("main input contains a development source image")
    expected_ids = [
        row["candidate_id"] for row in reclassification
        if row.get("semantic_status") == "valid" and str(row["source_image_id"]) not in development_source_ids
    ]
    if main_ids != expected_ids:
        errors.append("main input does not preserve authoritative filtered row order")
    missing_paths = []
    missing_identity = []
    for row in main_rows:
        image = Path(row["original_image_path"])
        image = image if image.is_absolute() else args.project_root / image
        if not image.is_file():
            missing_paths.append(row["candidate_id"])
        required = (
            "source_image_id", "source_object_a_id", "source_object_b_id",
            "object_a_label", "object_b_label", "original_color_a", "original_color_b",
        )
        if any(row.get(field) in (None, "") for field in required):
            missing_identity.append(row["candidate_id"])
    if missing_paths:
        errors.append(f"missing source images: {missing_paths}")
    if missing_identity:
        errors.append(f"missing object/color identity: {missing_identity}")
    if unresolved:
        errors.append(f"unresolved development identity collisions: {len(unresolved)}")

    split_counts = Counter(row["development_split"] for row in development)
    split_collision_counts = Counter(
        row["development_split"] for row in development if row["main_source_collision"]
    )
    technical_counts = Counter(row["technical_status"] for row in main_rows)
    difficulty_counts = Counter(tag for row in main_rows for tag in row.get("difficulty_tags", []))
    color_pair_counts = Counter(row["color_pair"] for row in main_rows)
    same_source_different_pair_count = sum(
        row["same_source_different_object_pair_collision_count"] for row in development
    )
    args.output_dir.mkdir(parents=True, exist_ok=True)
    main_path = args.output_dir / "main_input_manifest_v1.1.jsonl"
    audit_path = args.output_dir / "main_development_exclusion_audit_v1.1.jsonl"
    summary_path = args.output_dir / "main_preparation_summary_v1.1.json"
    checksum_path = args.output_dir / "main_input_manifest_v1.1.sha256"
    for path in (main_path, audit_path, summary_path, checksum_path):
        if path.exists():
            raise FileExistsError(f"Refusing to overwrite: {path}")
    atomic_jsonl(main_path, main_rows)
    atomic_jsonl(audit_path, development)
    main_sha = sha256(main_path)
    checksum_path.write_text(f"{main_sha}  {main_path.as_posix()}\n", encoding="ascii")
    summary = {
        "schema_version": "1.1",
        "preparation_version": "main_preparation_v1.1",
        "created_at": datetime.now(timezone.utc).isoformat(),
        "research_identity": "GQA-derived semantic-valid intervention-valid matched diagnostic evaluation",
        "authoritative_source": {
            "path": args.reclassification.as_posix(),
            "sha256": sha256(args.reclassification),
            "total_count": len(reclassification),
            "semantic_valid_count": len(valid_rows),
            "semantic_invalid_count": len(invalid_rows),
            "human_review_count": len(human_rows),
        },
        "development_exclusion": {
            "registry_count": len(development),
            "unique_source_image_count": len(development_source_ids),
            "split_counts": dict(split_counts),
            "split_main_collision_counts": dict(split_collision_counts),
            "main_candidate_collision_count": sum(row["main_candidate_collision"] for row in development),
            "main_source_collision_count": sum(row["main_source_collision"] for row in development),
            "excluded_semantic_valid_count": len(excluded_valid),
            "same_source_different_object_pair_collision_count": same_source_different_pair_count,
            "unresolved_identity_collision_count": len(unresolved),
            "policy": "exclude any semantic-valid candidate sharing a development source_image_id",
        },
        "main_input": {
            "path": main_path.as_posix(),
            "sha256": main_sha,
            "candidate_count": len(main_rows),
            "directly_editable_count": technical_counts["directly_editable"],
            "segmentation_test_required_count": technical_counts["segmentation_test_required"],
            "duplicate_candidate_id_count": len(main_ids) - len(set(main_ids)),
            "duplicate_source_pair_count": len(pair_keys) - len(set(pair_keys)),
            "development_overlap_count": sum(str(row["source_image_id"]) in development_source_ids for row in main_rows),
            "missing_source_image_count": len(missing_paths),
            "missing_identity_count": len(missing_identity),
            "new_candidate_count": sum(row["candidate_id"] not in set(authoritative_ids) for row in main_rows),
            "frozen_order_preserved": main_ids == expected_ids,
            "unexpected_reordering_count": 0 if main_ids == expected_ids else 1,
            "difficulty_tag_counts": dict(sorted(difficulty_counts.items())),
            "color_pair_counts": dict(sorted(color_pair_counts.items())),
        },
        "stopping_rule": "process the frozen held-out pool to exhaustion; final N is the number passing four-image QC",
        "historical_target_160": True,
        "target_160_guaranteed": len(main_rows) >= 160,
        "main_data_production_ready": not errors,
        "ready_statement": "Ready to begin main-experiment segmentation under the frozen v1.1 pipeline." if not errors else None,
        "question_generation_executed": False,
        "vlm_inference_executed": False,
        "main_segmentation_executed": False,
        "validation_errors": len(errors),
        "errors": errors,
        "source_files": {
            name: {"path": path.as_posix(), "sha256": sha256(path)}
            for name, path in {
                "candidate_pool": args.candidate_pool,
                "left_right": args.left_right,
                "feasibility": args.feasibility,
                "interim": args.interim,
                "smoke": args.smoke,
                "mini": args.mini,
                "completion": args.completion,
            }.items()
        },
    }
    atomic_json(summary_path, summary)
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    if errors:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
