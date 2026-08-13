#!/usr/bin/env python3
"""Validate generated or human-reviewed color-edit artifacts for a frozen subset."""

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

CRITERIA = {
    "target_color_change": "target_color_change_status",
    "original_color_residue": "original_color_residue_status",
    "outside_mask_preservation": "outside_mask_preservation_status",
    "color_leakage": "color_leakage_status",
    "boundary_quality": "boundary_quality_status",
    "texture_shading_pattern_preservation": "texture_shading_preservation_status",
    "binding_swap_achieved": "target_binding_achievement_status",
    "prompt_answer_validity": "prompt_answer_validity_after_edit_status",
}
ALLOWED_REVIEWED_CHANGES = {
    "edit_qc_status",
    "object_identity_preservation_status",
    "failure_reasons",
}
REVIEWED_ADDITIONS = {
    "natural_edit_qc_criteria",
    "natural_edit_qc_reviewer",
    "natural_edit_qc_timestamp",
    "natural_edit_qc_note",
    "controlled_original_qc_status",
    "controlled_edited_qc_status",
    "four_image_qc_status",
    *CRITERIA.values(),
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-manifest", "--mask-results", dest="input_manifest", type=Path, required=True)
    parser.add_argument("--subset-name", required=True)
    parser.add_argument("--edit-results", type=Path, required=True)
    parser.add_argument("--generated-edit-results", type=Path)
    parser.add_argument("--qc-manifest", type=Path)
    parser.add_argument("--segmentation-root", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--project-root", type=Path, required=True)
    parser.add_argument("--config", type=Path, required=True)
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


def add(errors: list[str], cid: str, message: str) -> None:
    errors.append(f"{cid}: {message}")


def safe(root: Path, value: str) -> Path:
    if Path(value).is_absolute():
        raise ValueError(f"absolute output asset path: {value}")
    root = root.resolve()
    path = (root / value).resolve()
    if path != root and root not in path.parents:
        raise ValueError(f"asset escapes output root: {value}")
    return path


def checked_asset(
    errors: list[str], cid: str, root: Path, relative: str | None, expected_sha: str | None
) -> Path | None:
    if not relative or not expected_sha:
        add(errors, cid, "missing asset path/checksum")
        return None
    try:
        path = safe(root, relative)
    except ValueError as exc:
        add(errors, cid, str(exc))
        return None
    if not path.is_file():
        add(errors, cid, f"missing asset: {path}")
        return None
    if sha256(path) != expected_sha:
        add(errors, cid, f"asset checksum mismatch: {path}")
        return None
    return path


def load_mask(path: Path) -> np.ndarray:
    with Image.open(path) as image:
        array = np.asarray(image.convert("L"))
    if not set(np.unique(array).tolist()).issubset({0, 255}):
        raise ValueError(f"non-binary mask: {path}")
    return array > 0


ASSET_CHECKSUMS = {
    "object_a_original_cutout_path": "original_cutout_a_sha256",
    "object_b_original_cutout_path": "original_cutout_b_sha256",
    "object_a_edited_cutout_path": "edited_cutout_a_sha256",
    "object_b_edited_cutout_path": "edited_cutout_b_sha256",
    "natural_edited_path": "natural_edited_sha256",
    "contact_sheet_path": "contact_sheet_sha256",
    "selected_color_mask_a_path": "selected_color_mask_a_sha256",
    "selected_color_mask_b_path": "selected_color_mask_b_sha256",
}


def expected_asset_paths(
    output_root: Path, config: dict[str, Any], cid: str
) -> dict[str, Path]:
    layout = config["output_layout"]
    return {
        "object_a_original_cutout_path": output_root / layout["original_cutouts_dir"] / f"{cid}__object_a_original.png",
        "object_b_original_cutout_path": output_root / layout["original_cutouts_dir"] / f"{cid}__object_b_original.png",
        "object_a_edited_cutout_path": output_root / layout["edited_cutouts_dir"] / f"{cid}__object_a_edited.png",
        "object_b_edited_cutout_path": output_root / layout["edited_cutouts_dir"] / f"{cid}__object_b_edited.png",
        "natural_edited_path": output_root / layout["natural_edited_dir"] / f"{cid}__natural_edited.png",
        "contact_sheet_path": output_root / layout["contact_sheets_dir"] / f"{cid}__before_after.png",
        "selected_color_mask_a_path": output_root / layout["selected_masks_dir"] / f"{cid}__object_a_selected.png",
        "selected_color_mask_b_path": output_root / layout["selected_masks_dir"] / f"{cid}__object_b_selected.png",
    }


def check_candidate_record(
    errors: list[str], cid: str, row: dict[str, Any], args: argparse.Namespace,
    config: dict[str, Any]
) -> bool:
    path = args.output_root / config["output_layout"]["records_dir"] / f"{cid}.json"
    if not path.is_file():
        add(errors, cid, f"missing per-candidate record: {path}")
        return False
    try:
        record = read_json(path)
    except Exception as exc:
        add(errors, cid, f"unreadable per-candidate record: {type(exc).__name__}: {exc}")
        return False
    if record != row:
        add(errors, cid, "per-candidate record differs from result manifest row")
        return False
    return True


def check_result(
    errors: list[str], cid: str, row: dict[str, Any], mask_row: dict[str, Any],
    args: argparse.Namespace, config: dict[str, Any]
) -> Counter[str]:
    reviewed = args.mode == "reviewed"
    verified: Counter[str] = Counter()
    generation_status = row.get("generation_status")
    immutable = {
        "subset_name": args.subset_name,
        "pilot_subset": args.subset_name,
        "source_image_id": mask_row["source_image_id"],
        "object_a_label": mask_row["object_a"]["label"],
        "object_b_label": mask_row["object_b"]["label"],
        "original_color_a": mask_row["object_a"]["original_color"],
        "original_color_b": mask_row["object_b"]["original_color"],
        "target_color_a": mask_row["object_b"]["original_color"],
        "target_color_b": mask_row["object_a"]["original_color"],
        "mask_qc_status": "pass",
        "edit_qc_status": "not_tested",
        "object_identity_preservation_status": "not_tested",
        "failure_reasons": [],
        "development_exposure": True,
        "color_swap_method": config["method"]["name"],
        "color_swap_config_sha256": sha256(args.config),
    }
    if reviewed and generation_status == "generated":
        immutable.pop("edit_qc_status")
        immutable.pop("object_identity_preservation_status")
        immutable.pop("failure_reasons")
    for field, expected in immutable.items():
        if row.get(field) != expected:
            add(errors, cid, f"{field}={row.get(field)!r}; expected {expected!r}")
    if generation_status not in {"generated", "failed"}:
        add(errors, cid, f"invalid generation_status={generation_status!r}")
        return verified
    provenance = row.get("generation_provenance", {})
    expected_provenance = {
        "seed": int(config["seed"]),
        "input_manifest_sha256": sha256(args.input_manifest),
        "config_sha256": sha256(args.config),
    }
    for field, expected in expected_provenance.items():
        if provenance.get(field) != expected:
            add(errors, cid, f"generation provenance {field}={provenance.get(field)!r}; expected {expected!r}")
    if reviewed and generation_status == "generated":
        for field in (
            "natural_edit_qc_reviewer", "natural_edit_qc_timestamp", "natural_edit_qc_note"
        ):
            if not row.get(field):
                add(errors, cid, f"reviewed result missing {field}")
        for field in (
            "controlled_original_qc_status", "controlled_edited_qc_status", "four_image_qc_status"
        ):
            if row.get(field) != "not_tested":
                add(errors, cid, f"{field} must remain not_tested")
        allowed_statuses = {"pass", "fail", "human_review"}
        edit_status = row.get("edit_qc_status")
        identity_status = row.get("object_identity_preservation_status")
        reasons = row.get("failure_reasons", [])
        if edit_status not in allowed_statuses:
            add(errors, cid, "invalid reviewed edit_qc_status")
        if identity_status not in allowed_statuses:
            add(errors, cid, "invalid reviewed identity status")
        if edit_status == "pass" and (reasons or identity_status != "pass"):
            add(errors, cid, "edit pass requires identity pass and no failure reasons")
        if edit_status == "fail" and not reasons:
            add(errors, cid, "edit fail requires a failure reason")
        criteria = row.get("natural_edit_qc_criteria", {})
        required_criteria = {*CRITERIA, "object_identity_preservation"}
        if set(criteria) != required_criteria or any(value not in allowed_statuses for value in criteria.values()):
            add(errors, cid, "reviewed result criteria are incomplete or invalid")
        if criteria.get("object_identity_preservation") != identity_status:
            add(errors, cid, "identity criterion/status mismatch")
        for criterion, field in CRITERIA.items():
            if row.get(field) != criteria.get(criterion):
                add(errors, cid, f"{field} disagrees with criteria")

    source_path = Path(row["original_image_path"])
    source_path = source_path if source_path.is_absolute() else args.project_root / source_path
    mask_a_path = args.segmentation_root / row["object_a_mask_path"]
    mask_b_path = args.segmentation_root / row["object_b_mask_path"]
    for path, expected, label in (
        (source_path, row["source_image_sha256"], "source"),
        (mask_a_path, row["mask_a_sha256"], "mask A"),
        (mask_b_path, row["mask_b_sha256"], "mask B"),
    ):
        if not path.is_file() or sha256(path) != expected:
            add(errors, cid, f"{label} missing or checksum mismatch: {path}")
            return verified
    try:
        with Image.open(source_path) as image:
            original = np.asarray(image.convert("RGB"))
        mask_a, mask_b = load_mask(mask_a_path), load_mask(mask_b_path)
        if mask_a.shape != original.shape[:2] or mask_b.shape != original.shape[:2]:
            add(errors, cid, "mask/source dimensions differ")
            return verified
        overlap = int((mask_a & mask_b).sum())
        maximum_overlap = int(config["automatic_diagnostics"]["maximum_ab_mask_overlap_pixels"])

        if generation_status == "failed":
            expected_error = f"ValueError: A/B masks overlap by {overlap} pixels"
            if overlap <= maximum_overlap:
                add(errors, cid, f"failed overlap={overlap} does not exceed maximum={maximum_overlap}")
            if row.get("generation_error") != expected_error:
                add(errors, cid, f"generation_error={row.get('generation_error')!r}; expected {expected_error!r}")
            if row.get("automatic_qc_metrics") != {}:
                add(errors, cid, "failed row automatic_qc_metrics must be empty")
            if row.get("diagnostic_flags") != ["generation_failed_requires_review"]:
                add(errors, cid, "failed row diagnostic_flags are invalid")

            for field, checksum in ASSET_CHECKSUMS.items():
                if row.get(field) is not None or row.get(checksum) is not None:
                    add(errors, cid, f"failed row must not declare generated asset: {field}")
            for field, asset_path in expected_asset_paths(args.output_root, config, cid).items():
                if asset_path.exists():
                    add(errors, cid, f"partial/stale asset for failed row {field}: {asset_path}")
            if check_candidate_record(errors, cid, row, args, config):
                verified["per_candidate_records"] += 1
            return verified

        if row.get("generation_error") is not None:
            add(errors, cid, "generated row must have generation_error=null")
        if overlap > maximum_overlap:
            add(errors, cid, f"A/B masks overlap by {overlap} pixels (maximum {maximum_overlap})")
        assets = {}
        for field, checksum in ASSET_CHECKSUMS.items():
            asset_path = checked_asset(
                errors, cid, args.output_root, row.get(field), row.get(checksum)
            )
            assets[field] = asset_path
            if asset_path is not None:
                verified[field] += 1
        natural_path = assets["natural_edited_path"]
        if natural_path is None:
            return verified
        with Image.open(natural_path) as image:
            natural = np.asarray(image.convert("RGB"))
        if natural.shape != original.shape:
            add(errors, cid, "Natural Edited/source dimensions differ")
            return verified
        union = mask_a | mask_b
        outside_changed = int(np.any(natural != original, axis=2)[~union].sum())
        if outside_changed != 0:
            add(errors, cid, f"outside-mask pixels changed: {outside_changed}")
        recorded = row.get("automatic_qc_metrics", {}).get("pair", {}).get("outside_mask_changed_pixels")
        if recorded != outside_changed:
            add(errors, cid, "outside-mask metric disagrees with pixels")
        for side, mask in (("a", mask_a), ("b", mask_b)):
            original_path = assets[f"object_{side}_original_cutout_path"]
            edited_path = assets[f"object_{side}_edited_cutout_path"]
            if original_path is None or edited_path is None:
                continue
            with Image.open(original_path) as image:
                original_rgba = np.asarray(image.convert("RGBA"))
            with Image.open(edited_path) as image:
                edited_rgba = np.asarray(image.convert("RGBA"))
            expected_alpha = mask.astype(np.uint8) * 255
            if not np.array_equal(original_rgba[..., 3], expected_alpha):
                add(errors, cid, f"object {side.upper()} original alpha != mask")
            if not np.array_equal(edited_rgba[..., 3], expected_alpha):
                add(errors, cid, f"object {side.upper()} edited alpha != mask")
    except Exception as exc:
        add(errors, cid, f"asset inspection failed: {type(exc).__name__}: {exc}")
    if not reviewed and check_candidate_record(errors, cid, row, args, config):
        verified["per_candidate_records"] += 1
    return verified


def check_review(errors: list[str], cid: str, review: dict[str, Any], result: dict[str, Any]) -> None:
    allowed = {"pass", "fail", "human_review"}
    edit_status = review.get("edit_qc_status")
    identity_status = review.get("object_identity_preservation_status")
    reasons = review.get("failure_reasons", [])
    if edit_status not in allowed:
        add(errors, cid, "review edit_qc_status is invalid")
    if identity_status not in allowed:
        add(errors, cid, "review identity status is invalid")
    if edit_status == "pass" and (reasons or identity_status != "pass"):
        add(errors, cid, "review pass requires identity pass and no failure reasons")
    if edit_status == "fail" and not reasons:
        add(errors, cid, "review fail requires a failure reason")
    if not review.get("reviewer") or not review.get("review_timestamp"):
        add(errors, cid, "reviewer and timestamp are required")
    if edit_status != "pass" and not review.get("review_note"):
        add(errors, cid, "non-pass review requires a note")
    criteria = review.get("criteria", {})
    required = {*CRITERIA, "object_identity_preservation"}
    if set(criteria) != required or any(value not in allowed for value in criteria.values()):
        add(errors, cid, "review criteria are incomplete or invalid")
    if criteria.get("object_identity_preservation") != identity_status:
        add(errors, cid, "review identity criterion/status mismatch")
    for criterion, flat_field in CRITERIA.items():
        review_value = review.get(flat_field, criteria.get(criterion))
        if review_value != criteria.get(criterion):
            add(errors, cid, f"review {flat_field} disagrees with criteria")
        if result.get(flat_field) != review_value:
            add(errors, cid, f"review/result mismatch: {flat_field}")
    pairs = (
        ("edit_qc_status", "edit_qc_status"),
        ("object_identity_preservation_status", "object_identity_preservation_status"),
        ("reviewer", "natural_edit_qc_reviewer"),
        ("review_timestamp", "natural_edit_qc_timestamp"),
        ("review_note", "natural_edit_qc_note"),
        ("criteria", "natural_edit_qc_criteria"),
        ("failure_reasons", "failure_reasons"),
    )
    for review_field, result_field in pairs:
        if review.get(review_field) != result.get(result_field):
            add(errors, cid, f"review/result mismatch: {review_field}")


def check_generated_invariance(
    errors: list[str], cid: str, generated: dict[str, Any], reviewed: dict[str, Any]
) -> None:
    for field, value in generated.items():
        if field not in ALLOWED_REVIEWED_CHANGES and reviewed.get(field) != value:
            add(errors, cid, f"generated field changed during QC import: {field}")
    unexpected = set(reviewed) - set(generated) - REVIEWED_ADDITIONS
    if unexpected:
        add(errors, cid, f"unexpected reviewed fields: {sorted(unexpected)}")


def unique_rows(errors: list[str], label: str, rows: list[dict[str, Any]], expected_count: int) -> dict[str, dict[str, Any]]:
    indexed = {row.get("candidate_id"): row for row in rows}
    if len(rows) != expected_count or len(indexed) != expected_count or None in indexed:
        errors.append(f"{label} must contain {expected_count} unique candidates")
    return indexed


def main() -> None:
    args = parse_args()
    mask_rows = read_jsonl(args.input_manifest)
    edit_rows = read_jsonl(args.edit_results)
    config = read_json(args.config)
    errors: list[str] = []
    expected_count = len(mask_rows)
    if expected_count == 0:
        errors.append("input manifest is empty")
    masks = unique_rows(errors, "reviewed mask manifest", mask_rows, expected_count)
    edits = unique_rows(errors, "edit manifest", edit_rows, expected_count)
    input_ids = [row.get("candidate_id") for row in mask_rows]
    edit_ids = [row.get("candidate_id") for row in edit_rows]
    missing_ids = sorted(set(masks) - set(edits))
    unexpected_ids = sorted(set(edits) - set(masks))
    duplicate_count = len(edit_ids) - len(set(edit_ids))
    membership_preserved = not missing_ids and not unexpected_ids
    order_preserved = edit_ids == input_ids
    if not membership_preserved:
        errors.append("edit candidate IDs do not exactly match reviewed mask IDs")
    if not order_preserved:
        errors.append("edit candidate order differs from frozen input order")
    verified: Counter[str] = Counter()
    for cid in sorted(set(masks) & set(edits)):
        verified.update(check_result(errors, cid, edits[cid], masks[cid], args, config))

    reviewed_count = 0
    generation_failures_preserved = args.mode != "reviewed"
    if args.mode == "reviewed":
        if args.generated_edit_results is None or args.qc_manifest is None:
            errors.append("reviewed mode requires --generated-edit-results and --qc-manifest")
            generation_failures_preserved = False
        else:
            generated_rows = read_jsonl(args.generated_edit_results)
            generated = unique_rows(errors, "generated edit manifest", generated_rows, expected_count)
            generated_success_rows = [
                row for row in generated_rows
                if row.get("generation_status") == "generated"
            ]
            generated_success_ids = {
                row.get("candidate_id") for row in generated_success_rows
            }
            review_rows = read_jsonl(args.qc_manifest)
            reviews = unique_rows(
                errors, "Natural Edit QC manifest", review_rows,
                len(generated_success_rows)
            )
            if set(generated) != set(edits):
                errors.append("generated and reviewed result candidate IDs differ")
            if set(reviews) != generated_success_ids:
                errors.append("QC candidate IDs differ from generated-success subset")
            expected_order = [row.get("candidate_id") for row in mask_rows]
            generated_success_order = [
                row.get("candidate_id") for row in generated_success_rows
            ]
            if [row.get("candidate_id") for row in generated_rows] != expected_order:
                errors.append("generated manifest order differs from frozen input order")
            if [row.get("candidate_id") for row in review_rows] != generated_success_order:
                errors.append("QC order differs from generated-success subsequence")
            generation_failures_preserved = True
            for cid in sorted(set(generated) & set(edits)):
                if generated[cid].get("generation_status") == "failed":
                    if edits[cid] != generated[cid]:
                        add(errors, cid, "generation-failed row changed during QC import")
                        generation_failures_preserved = False
                    continue
                check_generated_invariance(errors, cid, generated[cid], edits[cid])
                if cid in reviews:
                    check_review(errors, cid, reviews[cid], edits[cid])
                if check_candidate_record(errors, cid, generated[cid], args, config):
                    verified["per_candidate_records"] += 1
            reviewed_count = len(reviews)

    status_counts = Counter(row.get("generation_status") for row in edit_rows)
    generated_count = status_counts.get("generated", 0)
    failed_generation_count = status_counts.get("failed", 0)
    if generated_count + failed_generation_count != len(edit_rows):
        errors.append("generated + failed count does not equal result manifest count")

    summary = {
        "mode": args.mode,
        "subset_name": args.subset_name,
        "candidate_count": len(edit_rows),
        "generated_count": generated_count,
        "failed_generation_count": failed_generation_count,
        "failed_candidate_ids": [
            row.get("candidate_id") for row in edit_rows
            if row.get("generation_status") == "failed"
        ],
        "reviewed_count": reviewed_count,
        "generation_status_counts": dict(status_counts),
        "mask_qc_status_counts": dict(Counter(row.get("mask_qc_status") for row in edit_rows)),
        "edit_qc_status_counts": dict(Counter(row.get("edit_qc_status") for row in edit_rows)),
        "identity_status_counts": dict(Counter(row.get("object_identity_preservation_status") for row in edit_rows)),
        "original_cutouts_verified": (
            verified["object_a_original_cutout_path"]
            + verified["object_b_original_cutout_path"]
        ),
        "edited_cutouts_verified": (
            verified["object_a_edited_cutout_path"]
            + verified["object_b_edited_cutout_path"]
        ),
        "selected_color_masks_verified": (
            verified["selected_color_mask_a_path"]
            + verified["selected_color_mask_b_path"]
        ),
        "natural_edited_verified": verified["natural_edited_path"],
        "contact_sheets_verified": verified["contact_sheet_path"],
        "per_candidate_records_verified": verified["per_candidate_records"],
        "duplicate_candidate_count": duplicate_count,
        "unexpected_candidate_count": len(unexpected_ids),
        "unexpected_candidate_ids": unexpected_ids,
        "missing_candidate_count": len(missing_ids),
        "missing_candidate_ids": missing_ids,
        "membership_preserved": membership_preserved,
        "order_preserved": order_preserved,
        "generation_failures_preserved": generation_failures_preserved,
        "controlled_status_expected": "not_tested" if args.mode == "reviewed" else None,
        "validation_errors": len(errors),
        "errors": errors,
    }
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    if errors:
        sys.exit(1)


if __name__ == "__main__":
    main()
