#!/usr/bin/env python3
"""Validate generated or human-reviewed mask artifacts for a frozen subset."""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from collections import Counter
from pathlib import Path
from typing import Any

import numpy as np
from PIL import Image

OBJECT_CRITERIA = (
    "target_object_coverage",
    "background_leakage",
    "non_target_object_inclusion",
    "thin_structure_preservation",
    "occlusion_boundary_handling",
    "object_identity_consistency",
)
STATUSES = {"pass", "fail", "human_review", "not_tested"}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-manifest", "--smoke-manifest", dest="input_manifest", type=Path, required=True)
    parser.add_argument("--subset-name", required=True)
    parser.add_argument("--result-manifest", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--failure-reasons", type=Path, required=True)
    parser.add_argument("--qc-manifest", type=Path)
    parser.add_argument("--mode", choices=("generated", "reviewed"), default="generated")
    return parser.parse_args()


def read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    with path.open(encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def add(errors: list[str], candidate_id: str, message: str) -> None:
    errors.append(f"{candidate_id}: {message}")


def input_order_field(row: dict[str, Any]) -> str:
    if "frozen_order" in row and "selection_rank" not in row:
        return "frozen_order"
    if "selection_rank" in row and "frozen_order" not in row:
        return "selection_rank"
    raise ValueError("ambiguous or missing input ordering field")


def safe_asset(root: Path, relative: str) -> Path:
    if Path(relative).is_absolute():
        raise ValueError(f"absolute asset path: {relative}")
    root = root.resolve()
    path = (root / relative).resolve()
    if path != root and root not in path.parents:
        raise ValueError(f"asset escapes output root: {relative}")
    return path


def check_asset(
    errors: list[str], candidate_id: str, root: Path, asset: dict[str, Any]
) -> Path | None:
    try:
        path = safe_asset(root, asset["path"])
    except (KeyError, ValueError) as exc:
        add(errors, candidate_id, str(exc))
        return None
    if not path.is_file():
        add(errors, candidate_id, f"missing asset {path}")
        return None
    if sha256(path) != asset.get("sha256"):
        add(errors, candidate_id, f"checksum mismatch {path}")
    return path


def load_binary_mask(
    errors: list[str], candidate_id: str, path: Path | None
) -> np.ndarray | None:
    if path is None:
        return None
    try:
        with Image.open(path) as image:
            array = np.asarray(image.convert("L"))
    except Exception as exc:
        add(errors, candidate_id, f"unreadable mask {path}: {exc}")
        return None
    values = set(np.unique(array).tolist())
    if not values.issubset({0, 255}):
        add(errors, candidate_id, f"non-binary mask {path}: {sorted(values)[:10]}")
    return array > 0


def check_cutout(
    errors: list[str], candidate_id: str, path: Path | None, mask: np.ndarray | None
) -> None:
    if path is None or mask is None:
        return
    try:
        with Image.open(path) as image:
            if image.mode != "RGBA":
                add(errors, candidate_id, f"cutout mode is {image.mode}, not RGBA: {path}")
            alpha = np.asarray(image.convert("RGBA"))[..., 3] > 0
    except Exception as exc:
        add(errors, candidate_id, f"unreadable cutout {path}: {exc}")
        return
    if alpha.shape != mask.shape or not np.array_equal(alpha, mask):
        add(errors, candidate_id, f"cutout alpha differs from mask: {path}")


def check_generated(
    errors: list[str], record: dict[str, Any], smoke: dict[str, Any],
    root: Path, config: dict[str, Any], reviewed: bool, subset_name: str
) -> None:
    cid = record.get("candidate_id", "<missing-candidate-id>")
    try:
        order_field = input_order_field(smoke)
    except ValueError as exc:
        add(errors, cid, str(exc))
        order_field = "selection_rank"
    generation_status = record.get("mask_generation_status")
    expected = {
        "source_image_id": smoke["source_image_id"],
        "execution_order": smoke[order_field],
        "execution_order_field": order_field,
        "pilot_subset": subset_name,
        "semantic_status": smoke["semantic_status"],
        "technical_status": smoke["technical_status"],
        "development_exposure": smoke["development_exposure"],
        "technical_status_before": smoke["technical_status"],
        "technical_status_after_mask_qc": None,
        "technical_status_update_approved": False,
        "edit_qc_status": "not_tested",
        "object_identity_preservation_status": "not_tested",
    }
    if order_field == "selection_rank":
        expected["selection_rank"] = smoke["selection_rank"]
        expected["selection_seed"] = smoke["selection_seed"]
        if "execution_order" not in record:
            expected.pop("execution_order")
            expected.pop("execution_order_field")
    else:
        expected["frozen_order"] = smoke["frozen_order"]
    if not reviewed:
        expected["mask_qc_status"] = "not_tested"
    for field, value in expected.items():
        if record.get(field) != value:
            add(errors, cid, f"{field}={record.get(field)!r}; expected {value!r}")

    if reviewed and generation_status == "generated":
        for field in (
            "mask_qc_reviewer", "mask_qc_timestamp", "mask_qc_note"
        ):
            if not record.get(field):
                add(errors, cid, f"reviewed result missing {field}")
        if record.get("mask_qc_status") not in {"pass", "fail", "human_review"}:
            add(errors, cid, "reviewed aggregate mask status is incomplete")

    provenance = record.get("generation_provenance", {})
    frozen = {
        "seed": config["seed"],
        "model_id": config["model"]["model_id"],
        "model_revision": config["model"]["revision"],
        "prompt_type": config["inference"]["prompt_type"],
        "multimask_output": config["inference"]["multimask_output"],
        "mask_selection": config["inference"]["mask_selection"],
    }
    for field, value in frozen.items():
        if provenance.get(field) != value:
            add(errors, cid, f"generation_provenance.{field} mismatch")

    if generation_status == "segmentation_failed":
        if record.get("failure_record_version") != "segmentation_failure_v1.2":
            add(errors, cid, "invalid segmentation failure record version")
        failure = record.get("segmentation_failure", {})
        for field in ("exception_type", "exception_message", "reason", "traceback"):
            if not failure.get(field):
                add(errors, cid, f"segmentation failure missing {field}")
        if record.get("assets"):
            add(errors, cid, "segmentation failure must not declare complete assets")
        return
    if generation_status != "generated":
        add(errors, cid, f"unknown mask_generation_status={generation_status!r}")
        return

    assets = record.get("assets", {})
    paths = {
        name: check_asset(errors, cid, root, assets.get(name, {}))
        for name in ("mask_a", "mask_b", "cutout_a", "cutout_b", "overlay")
    }
    mask_a = load_binary_mask(errors, cid, paths["mask_a"])
    mask_b = load_binary_mask(errors, cid, paths["mask_b"])
    check_cutout(errors, cid, paths["cutout_a"], mask_a)
    check_cutout(errors, cid, paths["cutout_b"], mask_b)

    source = record.get("source_image", {})
    expected_shape = (source.get("height"), source.get("width"))
    for side, mask in (("A", mask_a), ("B", mask_b)):
        if mask is not None and mask.shape != expected_shape:
            add(errors, cid, f"mask {side} shape {mask.shape} != {expected_shape}")
    if paths["overlay"] is not None:
        try:
            with Image.open(paths["overlay"]) as overlay:
                if overlay.size != (source.get("width"), source.get("height")):
                    add(errors, cid, "overlay/source size mismatch")
        except Exception as exc:
            add(errors, cid, f"unreadable overlay: {exc}")


def check_review(
    errors: list[str], review: dict[str, Any], allowed_reasons: set[str]
) -> None:
    cid = review.get("candidate_id", "<missing-candidate-id>")
    status = review.get("mask_qc_status")
    if status not in STATUSES or status == "not_tested":
        add(errors, cid, "reviewed mask_qc_status is required")
    for side in ("object_a_qc", "object_b_qc"):
        criteria = review.get(side, {})
        if set(criteria) != set(OBJECT_CRITERIA):
            add(errors, cid, f"{side} criteria mismatch")
        for criterion, value in criteria.items():
            if value not in STATUSES or value == "not_tested":
                add(errors, cid, f"{side}.{criterion} is not reviewed")
    pair = review.get("pair_qc", {}).get("ab_mask_overlap")
    if pair not in STATUSES or pair == "not_tested":
        add(errors, cid, "pair_qc.ab_mask_overlap is not reviewed")

    reasons = review.get("mask_failure_reasons", [])
    flat_reasons = review.get("failure_reasons", [])
    if flat_reasons != reasons:
        add(errors, cid, "failure reason aliases disagree")
    unknown = sorted(set(reasons) - allowed_reasons)
    if unknown:
        add(errors, cid, f"unknown failure reasons: {unknown}")
    if status == "pass" and reasons:
        add(errors, cid, "pass forbids failure reasons")
    if status == "pass":
        if review.get("object_a_mask_status") != "pass":
            add(errors, cid, "object A aggregate status is not pass")
        if review.get("object_b_mask_status") != "pass":
            add(errors, cid, "object B aggregate status is not pass")
        criterion_values = [
            *review.get("object_a_qc", {}).values(),
            *review.get("object_b_qc", {}).values(),
            review.get("pair_qc", {}).get("ab_mask_overlap"),
        ]
        if any(value != "pass" for value in criterion_values):
            add(errors, cid, "overall pass requires every criterion to pass")
    if status == "fail" and not reasons:
        add(errors, cid, "fail requires a failure reason")
    if status in {"fail", "human_review"} and not review.get("mask_qc_note"):
        add(errors, cid, f"{status} requires a note")
    if not review.get("mask_qc_reviewer"):
        add(errors, cid, "reviewer is required")
    if not review.get("mask_qc_timestamp"):
        add(errors, cid, "review timestamp is required")

    after = review.get("technical_status_after_mask_qc")
    approved = review.get("technical_status_update_approved")
    if after == "failed_after_test":
        if status != "fail" or not reasons or approved is not True:
            add(errors, cid, "failed_after_test requires fail, reason, and approval")
    elif approved:
        add(errors, cid, "technical update approval without failed_after_test")


def main() -> None:
    args = parse_args()
    smoke_rows = read_jsonl(args.input_manifest)
    result_rows = read_jsonl(args.result_manifest)
    config = read_json(args.config)
    reason_config = read_json(args.failure_reasons)
    errors: list[str] = []

    smoke = {row["candidate_id"]: row for row in smoke_rows}
    results = {row.get("candidate_id"): row for row in result_rows}
    expected_count = len(smoke_rows)
    if expected_count == 0 or len(smoke) != expected_count:
        errors.append("input manifest must contain nonempty unique candidates")
    if len(result_rows) != expected_count or len(results) != expected_count:
        errors.append(f"result manifest must contain {expected_count} unique candidates")
    if set(smoke) != set(results):
        errors.append("result IDs do not exactly match the frozen input manifest")
    for cid in sorted(set(smoke) & set(results)):
        check_generated(
            errors, results[cid], smoke[cid], args.output_root, config,
            args.mode == "reviewed", args.subset_name
        )

    reviewed_count = 0
    if args.mode == "reviewed":
        if args.qc_manifest is None:
            errors.append("--qc-manifest is required in reviewed mode")
        else:
            reviews_list = read_jsonl(args.qc_manifest)
            reviews = {row.get("candidate_id"): row for row in reviews_list}
            reviewable_ids = {
                cid for cid, row in results.items()
                if row.get("mask_generation_status") == "generated"
            }
            reviewable_count = len(reviewable_ids)
            if len(reviews_list) != reviewable_count or len(reviews) != reviewable_count:
                errors.append(
                    f"QC manifest must contain {reviewable_count} unique "
                    "segmentation-success candidates"
                )
            if set(reviews) != reviewable_ids:
                errors.append(
                    "QC IDs do not exactly match segmentation-success result IDs"
                )
            allowed = set(reason_config["failure_reasons"])
            for cid in sorted(set(reviews) & reviewable_ids):
                check_review(errors, reviews[cid], allowed)
                result = results.get(cid, {})
                pairs = (
                    ("mask_qc_status", "mask_qc_status"),
                    ("mask_qc_reviewer", "mask_qc_reviewer"),
                    ("mask_qc_timestamp", "mask_qc_timestamp"),
                    ("mask_qc_note", "mask_qc_note"),
                    ("object_a_mask_status", "object_a_mask_status"),
                    ("object_b_mask_status", "object_b_mask_status"),
                    ("object_a_qc", "object_a_qc"),
                    ("object_b_qc", "object_b_qc"),
                    ("pair_qc", "pair_qc"),
                    ("failure_reasons", "failure_reasons"),
                    ("mask_failure_reasons", "mask_failure_reasons"),
                )
                for review_field, result_field in pairs:
                    if reviews[cid].get(review_field) != result.get(result_field):
                        add(errors, cid, f"review/result mismatch: {review_field}")
            reviewed_count = len(reviews_list)

    successful_count = sum(
        row.get("mask_generation_status") == "generated"
        for row in result_rows
    )
    failed_count = sum(
        row.get("mask_generation_status") == "segmentation_failed"
        for row in result_rows
    )
    summary = {
        "mode": args.mode,
        "subset_name": args.subset_name,
        "input_candidates": len(smoke_rows),
        "result_candidates": len(result_rows),
        "reviewed_candidates": reviewed_count,
        "model_id": config["model"]["model_id"],
        "model_revision": config["model"]["revision"],
        "seed": config["seed"],
        "attempted_candidates": len(result_rows),
        "successful_candidates": successful_count,
        "failed_candidates": failed_count,
        "mask_files_expected": successful_count * 2,
        "cutout_files_expected": successful_count * 2,
        "generation_status_counts": dict(
            Counter(r.get("mask_generation_status") for r in result_rows)
        ),
        "result_status_counts": dict(
            Counter(r.get("mask_qc_status") for r in result_rows)
        ),
        "validation_errors": len(errors),
        "errors": errors,
    }
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    if errors:
        sys.exit(1)


if __name__ == "__main__":
    main()
