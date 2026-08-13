#!/usr/bin/env python3
"""Derive frozen pre-edit technical eligibility from reviewed masks and human purity QC."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
from collections import Counter
from pathlib import Path
from typing import Any

import numpy as np
from PIL import Image

RULE_NAMES = (
    "non_target_mask_inclusion",
    "selected_color_ratio_below_0_05",
    "hue_distance_below_45",
    "low_saturation_above_0_50",
)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--reviewed-mask-manifest", type=Path, required=True)
    p.add_argument("--original-subset-manifest", type=Path, required=True)
    p.add_argument("--candidate-pool", type=Path, required=True)
    p.add_argument("--segmentation-root", type=Path, required=True)
    p.add_argument("--project-root", type=Path, required=True)
    p.add_argument("--semantic-purity-review", type=Path, required=True)
    p.add_argument("--gate-config", type=Path, required=True)
    p.add_argument("--subset-name", required=True)
    p.add_argument("--derivation-timestamp", required=True)
    p.add_argument("--eligible-output", type=Path)
    p.add_argument("--eligibility-output", type=Path, required=True)
    p.add_argument("--object-diagnostics-output", type=Path)
    p.add_argument("--summary-output", type=Path, required=True)
    p.add_argument("--audit-only", action="store_true")
    return p.parse_args()


def sha256_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def read_jsonl_raw(path: Path) -> list[tuple[dict[str, Any], bytes]]:
    rows = []
    with path.open("rb") as f:
        for raw in f:
            content = raw.rstrip(b"\r\n")
            if content.strip():
                rows.append((json.loads(content.decode("utf-8")), content))
    return rows


def read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def resolve(root: Path, value: str) -> Path:
    path = Path(value)
    return path if path.is_absolute() else root / path


def load_mask(path: Path) -> np.ndarray:
    with Image.open(path) as image:
        return np.asarray(image.convert("L")) > 0


def load_rgba(path: Path) -> np.ndarray:
    with Image.open(path) as image:
        return np.asarray(image.convert("RGBA"))


def load_hsv(path: Path) -> np.ndarray:
    with Image.open(path) as image:
        return np.asarray(image.convert("RGB").convert("HSV"))


def angular_distance(a: np.ndarray | float, b: np.ndarray | float) -> np.ndarray:
    return np.abs((np.asarray(a) - np.asarray(b) + 180.0) % 360.0 - 180.0)


def circular_mean(values: np.ndarray, weights: np.ndarray) -> float | None:
    if not values.size or float(weights.sum()) <= 0:
        return None
    radians = np.deg2rad(values)
    x = float((np.cos(radians) * weights).sum() / weights.sum())
    y = float((np.sin(radians) * weights).sum() / weights.sum())
    return float(np.rad2deg(math.atan2(y, x)) % 360.0)


def canonical_config_sha(config: dict[str, Any]) -> str:
    value = dict(config)
    value.pop("config_sha256", None)
    payload = json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":")) + "\n"
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def validate_semantic_review(row: dict[str, Any], subset: str) -> None:
    statuses = {"pass", "fail", "human_review", "not_tested"}
    labels = {"skin", "arm", "hand", "neck", "face", "hair", "person_leg", "pants", "shoe", "adjacent_object", "background_region", "other"}
    if row.get("subset_name") != subset:
        raise ValueError(f"Semantic review subset mismatch: {row.get('candidate_id')}")
    for side in ("a", "b"):
        status = row.get(f"object_{side}_semantic_purity_status")
        inclusion = row.get(f"non_target_inclusion_{side}")
        contamination = row.get(f"contamination_labels_{side}")
        if status not in statuses:
            raise ValueError(f"Invalid semantic purity status: {row.get('candidate_id')} {side}")
        if not isinstance(inclusion, bool):
            raise ValueError(f"Missing non-target inclusion boolean: {row.get('candidate_id')} {side}")
        if not isinstance(contamination, list) or len(contamination) != len(set(contamination)) or not set(contamination).issubset(labels):
            raise ValueError(f"Invalid contamination labels: {row.get('candidate_id')} {side}")
        if status == "pass" and (inclusion or contamination):
            raise ValueError(f"Purity pass conflicts with contamination: {row.get('candidate_id')} {side}")
        if status == "fail" and not inclusion:
            raise ValueError(f"Purity fail requires non-target inclusion: {row.get('candidate_id')} {side}")
    if not row.get("reviewer") or not row.get("review_note"):
        raise ValueError(f"Semantic review requires reviewer and note: {row.get('candidate_id')}")


def write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    if path.exists():
        raise FileExistsError(f"Refusing to overwrite: {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_name(f".{path.name}.tmp")
    with temp.open("x", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")
    temp.replace(path)


def write_raw_jsonl(path: Path, rows: list[bytes]) -> None:
    if path.exists():
        raise FileExistsError(f"Refusing to overwrite: {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_name(f".{path.name}.tmp")
    with temp.open("xb") as f:
        for row in rows:
            f.write(row + b"\n")
    temp.replace(path)


def write_json(path: Path, value: dict[str, Any]) -> None:
    if path.exists():
        raise FileExistsError(f"Refusing to overwrite: {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_name(f".{path.name}.tmp")
    temp.write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    temp.replace(path)


def main() -> None:
    args = parse_args()
    if args.audit_only and (args.eligible_output or args.object_diagnostics_output):
        raise ValueError("Audit-only mode writes only eligibility and summary outputs")
    if not args.audit_only and (args.eligible_output is None or args.object_diagnostics_output is None):
        raise ValueError("Normal mode requires eligible and object-diagnostics outputs")

    config = read_json(args.gate_config)
    if not config.get("frozen") or config.get("rule_version") != "pre_edit_technical_gate_v1.1":
        raise ValueError("Gate config is not frozen v1.1")
    if canonical_config_sha(config) != config.get("config_sha256"):
        raise ValueError("Gate config canonical SHA-256 mismatch")
    thresholds = config["thresholds"]
    definitions = config["pixel_selection_definition"]
    source_rows = read_jsonl_raw(args.reviewed_mask_manifest)
    original_rows = read_jsonl_raw(args.original_subset_manifest)
    pool_rows = read_jsonl_raw(args.candidate_pool)
    review_rows = read_jsonl_raw(args.semantic_purity_review)
    source_ids = [row[0].get("candidate_id") for row in source_rows]
    original_ids = [row[0].get("candidate_id") for row in original_rows]
    if len(source_ids) != len(set(source_ids)) or len(original_ids) != len(set(original_ids)) or set(source_ids) != set(original_ids):
        raise ValueError("Reviewed mask and frozen subset candidate sets must match uniquely")
    source_by = {row[0]["candidate_id"]: row for row in source_rows}
    pool_by = {row[0]["candidate_id"]: row[0] for row in pool_rows}
    reviews = {row[0]["candidate_id"]: row[0] for row in review_rows}
    segmentation_success_ids = [
        cid for cid in original_ids
        if source_by[cid][0].get("mask_generation_status") == "generated"
    ]
    segmentation_failure_ids = [
        cid for cid in original_ids
        if source_by[cid][0].get("mask_generation_status") == "segmentation_failed"
    ]
    if len(segmentation_success_ids) + len(segmentation_failure_ids) != len(original_ids):
        raise ValueError("Reviewed mask manifest contains an unknown generation status")
    if len(reviews) != len(review_rows) or set(reviews) != set(segmentation_success_ids):
        raise ValueError(
            "Semantic purity review must cover segmentation-success candidates exactly"
        )
    for review in reviews.values():
        validate_semantic_review(review, args.subset_name)

    source_sha = sha256_file(args.reviewed_mask_manifest)
    config_file_sha = sha256_file(args.gate_config)
    centers = definitions["color_centers_degrees"]
    tolerance = float(definitions["source_hue_tolerance_degrees"])
    min_sat = int(definitions["minimum_saturation_0_255"])
    min_value = int(definitions["minimum_value_0_255"])
    selected_min = float(thresholds["selected_color_ratio_min"])
    hue_min = float(thresholds["source_target_hue_distance_min_degrees"])
    low_sat_max = float(thresholds["low_saturation_ratio_max"])

    diagnostics_by_candidate: dict[str, dict[str, dict[str, Any]] | None] = {}
    pre_gate_rules_by_candidate: dict[str, dict[str, list[str]]] = {}
    pixel_gate_ids: list[str] = []
    source_assets_before: dict[str, str] = {}
    for cid in original_ids:
        source, _ = source_by[cid]
        if cid not in pool_by:
            raise ValueError(f"Candidate absent from candidate pool: {cid}")
        pool = pool_by[cid]
        if source.get("source_image_id") != pool.get("source_image_id"):
            raise ValueError(f"Source image mismatch: {cid}")

        side_pre_gate_rules = {"a": [], "b": []}
        generation_status = source.get("mask_generation_status")
        if generation_status == "segmentation_failed":
            side_pre_gate_rules["a"].append("mask_qc_not_pass")
            side_pre_gate_rules["b"].append("mask_qc_not_pass")
            diagnostics_by_candidate[cid] = None
            pre_gate_rules_by_candidate[cid] = side_pre_gate_rules
            continue
        if generation_status != "generated":
            raise ValueError(f"Unknown mask generation status: {cid}")

        if source.get("mask_qc_status") != "pass":
            side_pre_gate_rules["a"].append("mask_qc_not_pass")
            side_pre_gate_rules["b"].append("mask_qc_not_pass")
        review = reviews[cid]
        for side in ("a", "b"):
            if (
                review.get(f"object_{side}_semantic_purity_status") != "pass"
                or review.get(f"non_target_inclusion_{side}")
            ):
                side_pre_gate_rules[side].append("non_target_mask_inclusion")

        if side_pre_gate_rules["a"] or side_pre_gate_rules["b"]:
            diagnostics_by_candidate[cid] = None
            pre_gate_rules_by_candidate[cid] = side_pre_gate_rules
            continue

        pixel_gate_ids.append(cid)
        pre_gate_rules_by_candidate[cid] = side_pre_gate_rules
        image_path = resolve(args.project_root, source["source_image"]["path"])
        source_assets_before[image_path.as_posix()] = sha256_file(image_path)
        hsv = load_hsv(image_path)
        hue = hsv[..., 0].astype(np.float64) * (360.0 / 255.0)
        saturation = hsv[..., 1]
        value = hsv[..., 2]
        side_values: dict[str, dict[str, Any]] = {}
        for side in ("a", "b"):
            object_record = source[f"object_{side}"]
            mask_path = resolve(args.segmentation_root, source["assets"][f"mask_{side}"]["path"])
            cutout_path = resolve(args.segmentation_root, source["assets"][f"cutout_{side}"]["path"])
            source_assets_before[mask_path.as_posix()] = sha256_file(mask_path)
            source_assets_before[cutout_path.as_posix()] = sha256_file(cutout_path)
            mask = load_mask(mask_path)
            rgba = load_rgba(cutout_path)
            selected = mask & (angular_distance(hue, centers[object_record["original_color"]]) <= tolerance) & (saturation >= min_sat) & (value >= min_value)
            mask_count = int(mask.sum())
            selected_count = int(selected.sum())
            observed = circular_mean(hue[selected], saturation[selected].astype(np.float64))
            side_values[side] = {
                "record_type": "object_side_diagnostic",
                "candidate_id": cid,
                "object_side": side,
                "object_label": object_record["label"],
                "original_color": object_record["original_color"],
                "mask_pixel_count": mask_count,
                "alpha_valid_pixel_count": int((rgba[..., 3] > 0).sum()),
                "selected_color_pixel_count": selected_count,
                "selected_color_ratio": selected_count / mask_count if mask_count else 0.0,
                "observed_source_hue": observed,
                "observed_target_hue": None,
                "source_target_hue_distance_degrees": None,
                "low_saturation_ratio": float((saturation[mask] < min_sat).mean()) if mask_count else 1.0,
                "non_target_mask_inclusion_status": review[f"object_{side}_semantic_purity_status"],
                "non_target_inclusion": review[f"non_target_inclusion_{side}"],
                "contamination_labels": review[f"contamination_labels_{side}"],
                "mask_qc_status": source.get("mask_qc_status"),
                "triggered_rules": [],
                "object_eligible": False,
            }
        observed_a, observed_b = side_values["a"]["observed_source_hue"], side_values["b"]["observed_source_hue"]
        pair_distance = None if observed_a is None or observed_b is None else float(angular_distance(observed_a, observed_b))
        for side, other in (("a", "b"), ("b", "a")):
            row = side_values[side]
            row["observed_target_hue"] = side_values[other]["observed_source_hue"]
            row["source_target_hue_distance_degrees"] = pair_distance
            rules = []
            if row["selected_color_ratio"] < selected_min:
                rules.append("selected_color_ratio_below_0_05")
            if pair_distance is None or pair_distance < hue_min:
                rules.append("hue_distance_below_45")
            if row["low_saturation_ratio"] > low_sat_max:
                rules.append("low_saturation_above_0_50")
            row["triggered_rules"] = rules
            row["object_eligible"] = not rules
        diagnostics_by_candidate[cid] = side_values

    eligibility_rows = []
    object_rows = []
    eligible_raw = []
    for index, cid in enumerate(original_ids):
        source, raw = source_by[cid]
        diagnostics = diagnostics_by_candidate[cid]
        if diagnostics is None:
            side_rules = pre_gate_rules_by_candidate[cid]
            a = b = None
            object_a_eligible = False
            object_b_eligible = False
            candidate_rules = list(dict.fromkeys(side_rules["a"] + side_rules["b"]))
            exclusion_reasons = (
                [f"object_a:{rule}" for rule in side_rules["a"]]
                + [f"object_b:{rule}" for rule in side_rules["b"]]
            )
            candidate_eligible = False
        else:
            a, b = diagnostics["a"], diagnostics["b"]
            object_rows.extend((a, b))
            object_a_eligible = a["object_eligible"]
            object_b_eligible = b["object_eligible"]
            candidate_rules = list(dict.fromkeys(a["triggered_rules"] + b["triggered_rules"]))
            exclusion_reasons = (
                [f"object_a:{rule}" for rule in a["triggered_rules"]]
                + [f"object_b:{rule}" for rule in b["triggered_rules"]]
            )
            candidate_eligible = object_a_eligible and object_b_eligible
        if candidate_eligible:
            eligible_raw.append(raw)
        eligibility_rows.append({
            "record_type": "candidate_eligibility",
            "candidate_id": cid,
            "original_subset": args.subset_name,
            "original_order_index": index,
            "object_a_eligible": object_a_eligible,
            "object_b_eligible": object_b_eligible,
            "candidate_pre_edit_eligible": candidate_eligible,
            "triggered_rules": candidate_rules,
            "exclusion_reasons": exclusion_reasons,
            "source_manifest_path": args.reviewed_mask_manifest.as_posix(),
            "source_manifest_sha256": source_sha,
            "source_row_sha256": sha256_bytes(raw),
            "gate_config_path": args.gate_config.as_posix(),
            "gate_config_sha256": config_file_sha,
            "gate_config_canonical_sha256": config["config_sha256"],
            "derivation_timestamp": args.derivation_timestamp,
            "object_a_diagnostic": a,
            "object_b_diagnostic": b,
        })

    reason_counts = Counter(rule for row in eligibility_rows for rule in row["triggered_rules"] if rule in RULE_NAMES)
    excluded = [row for row in eligibility_rows if not row["candidate_pre_edit_eligible"]]
    summary = {
        "schema_version": "1.1",
        "rule_version": config["rule_version"],
        "subset_name": args.subset_name,
        "start_count": len(original_ids),
        "pre_edit_gate_eligible_count": len(original_ids) - len(excluded),
        "pre_edit_gate_excluded_count": len(excluded),
        "eligible_candidate_ids": [row["candidate_id"] for row in eligibility_rows if row["candidate_pre_edit_eligible"]],
        "excluded_candidate_ids": [row["candidate_id"] for row in excluded],
        "gate_exclusion_reason_counts_overlapping": {rule: reason_counts.get(rule, 0) for rule in RULE_NAMES},
        "multiple_rules_count": sum(len([r for r in row["triggered_rules"] if r in RULE_NAMES]) > 1 for row in excluded),
        "candidate_reselection": False,
        "random_sampling": False,
        "order_preserved": True,
        "source_manifest_path": args.reviewed_mask_manifest.as_posix(),
        "source_manifest_sha256": source_sha,
        "original_subset_manifest_path": args.original_subset_manifest.as_posix(),
        "original_subset_manifest_sha256": sha256_file(args.original_subset_manifest),
        "semantic_purity_review_path": args.semantic_purity_review.as_posix(),
        "semantic_purity_review_sha256": sha256_file(args.semantic_purity_review),
        "gate_config_path": args.gate_config.as_posix(),
        "gate_config_sha256": config_file_sha,
        "gate_config_canonical_sha256": config["config_sha256"],
        "derivation_timestamp": args.derivation_timestamp,
        "source_assets_unchanged": True,
        "audit_only": args.audit_only,
    }

    source_assets_after = {path: sha256_file(Path(path)) for path in source_assets_before}
    if source_assets_before != source_assets_after or sha256_file(args.reviewed_mask_manifest) != source_sha:
        raise RuntimeError("A source manifest or asset changed during eligibility derivation")
    if args.audit_only:
        write_jsonl(args.eligibility_output, eligibility_rows)
    else:
        write_raw_jsonl(args.eligible_output, eligible_raw)
        write_jsonl(args.eligibility_output, eligibility_rows)
        write_jsonl(args.object_diagnostics_output, object_rows)
        summary["eligible_output_path"] = args.eligible_output.as_posix()
        summary["object_diagnostics_output_path"] = args.object_diagnostics_output.as_posix()
    write_json(args.summary_output, summary)
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
