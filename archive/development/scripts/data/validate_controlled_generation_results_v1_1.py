#!/usr/bin/env python3
"""Validate generated or reviewed Controlled pair results for any subset."""

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

CRITERIA = (
    "controlled_original_identity", "controlled_edited_identity",
    "original_color_binding", "edited_color_binding", "edited_cutout_reuse",
    "placement_consistency", "relative_scale_preservation", "overlap_and_clipping",
    "background_margin_alpha_quality", "four_image_prompt_answer_validity",
)
REVIEW_MUTABLE = {
    "controlled_qc_status", "controlled_original_qc_status",
    "controlled_edited_qc_status", "four_image_qc_status", "failure_reasons",
}
REVIEW_ADDITIONS = {"four_image_qc_criteria", "four_image_qc_reviewer", "four_image_qc_timestamp", "four_image_qc_note"}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-manifest", type=Path, required=True)
    parser.add_argument("--results", type=Path, required=True)
    parser.add_argument("--generated-results", type=Path)
    parser.add_argument("--qc-manifest", type=Path)
    parser.add_argument("--segmentation-root", type=Path, required=True)
    parser.add_argument("--color-edit-root", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--project-root", type=Path, required=True)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--subset-name", required=True)
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


def resolve(root: Path, value: str) -> Path:
    path = Path(value)
    return path if path.is_absolute() else root / path


def unique(errors: list[str], label: str, rows: list[dict[str, Any]], expected: int | None) -> dict[str, dict[str, Any]]:
    indexed = {row.get("candidate_id"): row for row in rows}
    if not rows or len(rows) != len(indexed) or None in indexed:
        errors.append(f"{label} must contain nonempty unique candidate IDs")
    if expected is not None and len(rows) != expected:
        errors.append(f"{label} expected {expected} rows, found {len(rows)}")
    return indexed


def check_file(errors: list[str], cid: str, path: Path, expected: str, label: str) -> bool:
    if not path.is_file():
        add(errors, cid, f"missing {label}: {path}")
        return False
    actual = sha256(path)
    if actual != expected:
        add(errors, cid, f"{label} checksum mismatch")
        return False
    return True


def alpha_mask(path: Path, placement: dict[str, Any], canvas_size: tuple[int, int]) -> np.ndarray:
    with Image.open(path) as image:
        rgba = image.convert("RGBA")
        bbox = rgba.getchannel("A").getbbox()
        if bbox is None:
            raise ValueError(f"empty alpha: {path}")
        crop = rgba.crop(bbox).resize((placement["width"], placement["height"]), Image.Resampling.LANCZOS)
        alpha = np.asarray(crop.getchannel("A")) > 0
    full = np.zeros((canvas_size[1], canvas_size[0]), dtype=bool)
    x, y = placement["x"], placement["y"]
    full[y : y + placement["height"], x : x + placement["width"]] = alpha
    return full


def validate_row(
    errors: list[str], cid: str, source: dict[str, Any], result: dict[str, Any],
    args: argparse.Namespace, config: dict[str, Any]
) -> None:
    reviewed = args.mode == "reviewed"
    expected = {
        "subset_name": args.subset_name, "source_image_id": source["source_image_id"],
        "object_a_label": source["object_a_label"], "object_b_label": source["object_b_label"],
        "original_color_a": source["original_color_a"], "original_color_b": source["original_color_b"],
        "target_color_a": source["target_color_a"], "target_color_b": source["target_color_b"],
        "mask_qc_status": "pass", "edit_qc_status": "pass", "object_identity_preservation_status": "pass",
        "controlled_qc_status": "pass" if reviewed else "not_tested",
        "four_image_qc_status": "pass" if reviewed else "not_tested",
        "failure_reasons": [], "config_sha256": sha256(args.config),
        "development_exposure": bool(source.get("development_exposure", False)),
    }
    for field, value in expected.items():
        if result.get(field) != value:
            add(errors, cid, f"{field}={result.get(field)!r}; expected {value!r}")
    generation_status = result.get("controlled_generation_status")
    if generation_status == "failed":
        if not result.get("generation_error"):
            add(errors, cid, "failed generation lacks generation_error")
        return
    if generation_status != "generated":
        add(errors, cid, f"invalid controlled_generation_status: {generation_status}")
        return
    for optional in ("semantic_status", "technical_status", "technical_status_before", "technical_status_after_mask_qc"):
        if source.get(optional) != result.get(optional):
            add(errors, cid, f"existing field changed: {optional}")
    if reviewed:
        for field in ("controlled_original_qc_status", "controlled_edited_qc_status"):
            if result.get(field) != "pass":
                add(errors, cid, f"{field} is not pass")
        criteria = result.get("four_image_qc_criteria", {})
        if set(criteria) != set(CRITERIA) or any(value != "pass" for value in criteria.values()):
            add(errors, cid, "four-image criteria are not all pass")
        for field in ("four_image_qc_reviewer", "four_image_qc_timestamp", "four_image_qc_note"):
            if not result.get(field):
                add(errors, cid, f"reviewed result missing {field}")

    input_assets = {
        "natural_original_sha256": resolve(args.project_root, source["original_image_path"]),
        "natural_edited_sha256": resolve(args.color_edit_root, source["natural_edited_path"]),
        "original_cutout_a_sha256": resolve(args.color_edit_root, source["object_a_original_cutout_path"]),
        "original_cutout_b_sha256": resolve(args.color_edit_root, source["object_b_original_cutout_path"]),
        "edited_cutout_a_sha256": resolve(args.color_edit_root, source["object_a_edited_cutout_path"]),
        "edited_cutout_b_sha256": resolve(args.color_edit_root, source["object_b_edited_cutout_path"]),
        "mask_a_sha256": resolve(args.segmentation_root, source["object_a_mask_path"]),
        "mask_b_sha256": resolve(args.segmentation_root, source["object_b_mask_path"]),
    }
    source_expected = {
        "natural_original_sha256": source["source_image_sha256"],
        "natural_edited_sha256": source["natural_edited_sha256"],
        "original_cutout_a_sha256": source["original_cutout_a_sha256"],
        "original_cutout_b_sha256": source["original_cutout_b_sha256"],
        "edited_cutout_a_sha256": source["edited_cutout_a_sha256"],
        "edited_cutout_b_sha256": source["edited_cutout_b_sha256"],
        "mask_a_sha256": source["mask_a_sha256"], "mask_b_sha256": source["mask_b_sha256"],
    }
    checksums = result.get("asset_checksums", {})
    for key, path in input_assets.items():
        check_file(errors, cid, path, source_expected[key], key)
        if checksums.get(key) != source_expected[key]:
            add(errors, cid, f"result input checksum differs: {key}")
    reuse = result.get("authoritative_edited_cutout_reuse", {})
    if reuse.get("object_a_sha256") != source["edited_cutout_a_sha256"] or reuse.get("object_b_sha256") != source["edited_cutout_b_sha256"] or reuse.get("recomputed_color_edit") is not False:
        add(errors, cid, "authoritative edited cutout reuse violation")

    output_assets = {
        "controlled_original_sha256": resolve(args.output_root, result["controlled_original_path"]),
        "controlled_edited_sha256": resolve(args.output_root, result["controlled_edited_path"]),
        "four_image_contact_sheet_sha256": resolve(args.output_root, result["four_image_contact_sheet_path"]),
        "placement_preview_sha256": resolve(args.output_root, result["placement_preview_path"]),
    }
    for key, path in output_assets.items():
        check_file(errors, cid, path, checksums.get(key, ""), key)
    if not all(path.is_file() for path in output_assets.values()):
        return
    try:
        with Image.open(output_assets["controlled_original_sha256"]) as image:
            original = np.asarray(image.convert("RGB"))
        with Image.open(output_assets["controlled_edited_sha256"]) as image:
            edited = np.asarray(image.convert("RGB"))
        canvas_size = (result["canvas_width"], result["canvas_height"])
        if original.shape[:2] != (canvas_size[1], canvas_size[0]) or edited.shape != original.shape:
            add(errors, cid, "Controlled canvas dimensions differ")
            return
        p_a, p_b = result["placement_a"], result["placement_b"]
        if p_a is None or p_b is None or result["global_scale_factor"] <= 0:
            add(errors, cid, "invalid placement or scale")
            return
        mask_a = alpha_mask(input_assets["original_cutout_a_sha256"], p_a, canvas_size)
        mask_b = alpha_mask(input_assets["original_cutout_b_sha256"], p_b, canvas_size)
        overlap = int((mask_a & mask_b).sum())
        if overlap:
            add(errors, cid, f"object overlap: {overlap}")
        union = mask_a | mask_b
        outside_difference = int(np.any(original != edited, axis=2)[~union].sum())
        if outside_difference:
            add(errors, cid, f"non-object difference: {outside_difference}")
        background = np.asarray(result["background_rgb"], dtype=np.uint8)
        for label, image in (("original", original), ("edited", edited)):
            mismatch = int(np.any(image != background, axis=2)[~union].sum())
            if mismatch:
                add(errors, cid, f"{label} background mismatch: {mismatch}")
        metrics = result.get("automatic_qc_metrics", {})
        if metrics.get("object_overlap_pixels") != overlap or metrics.get("non_object_difference_pixels") != outside_difference:
            add(errors, cid, "recorded geometry metrics differ from pixels")
    except Exception as exc:
        add(errors, cid, f"pixel validation failed: {type(exc).__name__}: {exc}")


def validate_review(errors: list[str], cid: str, review: dict[str, Any], result: dict[str, Any]) -> None:
    if review.get("controlled_qc_status") != "pass" or review.get("four_image_qc_status") != "pass":
        add(errors, cid, "human aggregate QC is not pass")
    if review.get("failure_reasons") != []:
        add(errors, cid, "pass review contains failure reasons")
    criteria = review.get("criteria", {})
    if set(criteria) != set(CRITERIA) or any(value != "pass" for value in criteria.values()):
        add(errors, cid, "human criteria are not all pass")
    for field in ("reviewer", "review_timestamp", "review_note"):
        if not review.get(field):
            add(errors, cid, f"review missing {field}")
    pairs = (("controlled_qc_status", "controlled_qc_status"), ("four_image_qc_status", "four_image_qc_status"), ("criteria", "four_image_qc_criteria"), ("reviewer", "four_image_qc_reviewer"), ("review_timestamp", "four_image_qc_timestamp"), ("review_note", "four_image_qc_note"), ("failure_reasons", "failure_reasons"))
    for review_field, result_field in pairs:
        if review.get(review_field) != result.get(result_field):
            add(errors, cid, f"review/result mismatch: {review_field}")


def validate_invariance(errors: list[str], cid: str, generated: dict[str, Any], reviewed: dict[str, Any]) -> None:
    for field, value in generated.items():
        if field not in REVIEW_MUTABLE and reviewed.get(field) != value:
            add(errors, cid, f"generated field changed during QC import: {field}")
    unexpected = set(reviewed) - set(generated) - REVIEW_ADDITIONS
    if unexpected:
        add(errors, cid, f"unexpected reviewed fields: {sorted(unexpected)}")


def main() -> None:
    args = parse_args()
    config = read_json(args.config)
    errors: list[str] = []
    source_rows = read_jsonl(args.input_manifest)
    # Post-QC attrition is valid; the eligible input manifest owns the expected count.
    expected_count = len(source_rows)
    source = unique(errors, "input manifest", source_rows, expected_count)
    results = unique(errors, "result manifest", read_jsonl(args.results), expected_count)
    if set(source) != set(results):
        errors.append("input and result candidate IDs differ")
    if list(results) != list(source):
        errors.append("result candidate order differs from input manifest order")
    for cid in sorted(set(source) & set(results)):
        validate_row(errors, cid, source[cid], results[cid], args, config)
    reviewed_count = 0
    if args.mode == "reviewed":
        if args.generated_results is None or args.qc_manifest is None:
            errors.append("reviewed mode requires --generated-results and --qc-manifest")
        else:
            generated = unique(errors, "generated results", read_jsonl(args.generated_results), expected_count)
            reviews = unique(errors, "QC manifest", read_jsonl(args.qc_manifest), expected_count)
            if set(generated) != set(results) or set(reviews) != set(results):
                errors.append("generated/reviewed/QC candidate IDs differ")
            if list(generated) != list(results) or list(reviews) != list(results):
                errors.append("generated/reviewed/QC candidate order differs")
            for cid in sorted(set(generated) & set(reviews) & set(results)):
                validate_invariance(errors, cid, generated[cid], results[cid])
                validate_review(errors, cid, reviews[cid], results[cid])
            reviewed_count = len(reviews)
    rows = list(results.values())
    generated_count = sum(row.get("controlled_generation_status") == "generated" for row in rows)
    summary = {
        "mode": args.mode, "subset_name": args.subset_name, "candidate_count": len(rows), "reviewed_count": reviewed_count,
        "generation_status_counts": dict(Counter(row.get("controlled_generation_status") for row in rows)),
        "controlled_qc_status_counts": dict(Counter(row.get("controlled_qc_status") for row in rows)),
        "four_image_qc_status_counts": dict(Counter(row.get("four_image_qc_status") for row in rows)),
        "controlled_original_verified": generated_count, "controlled_edited_verified": generated_count,
        "contact_sheets_verified": generated_count, "original_cutouts_verified": generated_count * 2,
        "edited_cutouts_verified": generated_count * 2, "validation_errors": len(errors), "errors": errors,
    }
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    if errors:
        sys.exit(1)


if __name__ == "__main__":
    main()
