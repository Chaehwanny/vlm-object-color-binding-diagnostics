#!/usr/bin/env python3
"""Generate bbox-prompted SAM masks for a frozen manifest subset."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import platform
import random
import traceback
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import torch
import transformers
from PIL import Image, __version__ as PILLOW_VERSION
from transformers import SamModel, SamProcessor


QC_CRITERIA = (
    "target_object_coverage",
    "background_leakage",
    "non_target_object_inclusion",
    "thin_structure_preservation",
    "occlusion_boundary_handling",
    "object_identity_consistency",
)


def parse_args() -> argparse.Namespace:
    root = Path(__file__).resolve().parents[2]
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-manifest", "--manifest", dest="input_manifest", type=Path, required=True)
    parser.add_argument("--subset-name", required=True)
    parser.add_argument("--candidate-pool", type=Path, required=True)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--project-root", type=Path, default=root)
    parser.add_argument(
        "--resume",
        action="store_true",
        help="Resume only from checksum-valid per-candidate records.",
    )
    parser.add_argument(
        "--preflight-only",
        action="store_true",
        help=(
            "Validate manifest, ordering, provenance, and source paths without "
            "loading SAM or creating output files."
        ),
    )
    parser.add_argument(
        "--continue-on-error",
        action="store_true",
        help=(
            "Record versioned per-candidate segmentation failures and continue. "
            "Does not alter inputs, boxes, model, config, or thresholds."
        ),
    )
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


def resolve_path(root: Path, value: str) -> Path:
    path = Path(value)
    return path if path.is_absolute() else root / path


def order_contract(rows: list[dict[str, Any]]) -> tuple[str, list[dict[str, Any]]]:
    """Return the frozen order field and rows sorted by that field."""
    uses_frozen_order = ["frozen_order" in row for row in rows]
    uses_selection_rank = ["selection_rank" in row for row in rows]
    if all(uses_frozen_order) and not any(uses_selection_rank):
        field = "frozen_order"
    elif all(uses_selection_rank) and not any(uses_frozen_order):
        field = "selection_rank"
    else:
        raise ValueError(
            "Input must use exactly one complete ordering contract: "
            "frozen_order for main or selection_rank for pilot subsets."
        )
    values = [row[field] for row in rows]
    if any(not isinstance(value, int) or value < 1 for value in values):
        raise ValueError(f"{field} values must be positive integers.")
    if len(set(values)) != len(values):
        raise ValueError(f"Duplicate {field} values.")
    if sorted(values) != list(range(1, len(rows) + 1)):
        raise ValueError(f"{field} must be contiguous from 1 to {len(rows)}.")
    return field, sorted(rows, key=lambda row: row[field])


def atomic_write_json(path: Path, value: dict[str, Any]) -> None:
    if path.exists():
        raise FileExistsError(f"Refusing to overwrite output: {path}")
    temporary = path.with_name(f".{path.name}.tmp")
    if temporary.exists():
        raise FileExistsError(f"Stale temporary file: {temporary}")
    temporary.write_text(
        json.dumps(value, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    temporary.replace(path)


def atomic_write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    if path.exists():
        raise FileExistsError(f"Refusing to overwrite output: {path}")
    temporary = path.with_name(f".{path.name}.tmp")
    if temporary.exists():
        raise FileExistsError(f"Stale temporary file: {temporary}")
    with temporary.open("x", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")
    temporary.replace(path)


def atomic_save_image(image: Image.Image, path: Path) -> None:
    if path.exists():
        raise FileExistsError(f"Refusing to overwrite output: {path}")
    temporary = path.with_name(f".{path.name}.tmp")
    if temporary.exists():
        raise FileExistsError(f"Stale temporary file: {temporary}")
    image.save(temporary, format="PNG")
    temporary.replace(path)


def bbox_xyxy_pixels(
    bbox_xywh: list[float], width: int, height: int
) -> list[float]:
    if len(bbox_xywh) != 4:
        raise ValueError(f"Expected normalized xywh bbox: {bbox_xywh}")
    x, y, box_width, box_height = bbox_xywh
    if box_width <= 0 or box_height <= 0:
        raise ValueError(f"Nonpositive bbox: {bbox_xywh}")
    if min(x, y) < 0 or x + box_width > 1 or y + box_height > 1:
        raise ValueError(f"Out-of-range normalized bbox: {bbox_xywh}")
    return [
        x * width,
        y * height,
        (x + box_width) * width,
        (y + box_height) * height,
    ]


def bbox_boolean_mask(
    bbox_xyxy: list[float], width: int, height: int
) -> np.ndarray:
    left = max(0, int(np.floor(bbox_xyxy[0])))
    top = max(0, int(np.floor(bbox_xyxy[1])))
    right = min(width, int(np.ceil(bbox_xyxy[2])))
    bottom = min(height, int(np.ceil(bbox_xyxy[3])))
    result = np.zeros((height, width), dtype=bool)
    result[top:bottom, left:right] = True
    return result


def mask_boundary(mask: np.ndarray) -> np.ndarray:
    padded = np.pad(mask, 1, mode="constant", constant_values=False)
    eroded = mask.copy()
    for y_offset in range(3):
        for x_offset in range(3):
            eroded &= padded[
                y_offset : y_offset + mask.shape[0],
                x_offset : x_offset + mask.shape[1],
            ]
    return mask & ~eroded


def make_overlay(
    image_array: np.ndarray, mask_a: np.ndarray, mask_b: np.ndarray
) -> Image.Image:
    overlay = image_array.astype(np.float32).copy()
    colors = (
        np.asarray([255, 0, 255], dtype=np.float32),
        np.asarray([0, 255, 255], dtype=np.float32),
    )
    for mask, color in zip((mask_a, mask_b), colors, strict=True):
        overlay[mask] = overlay[mask] * 0.78 + color * 0.22
        boundary = mask_boundary(mask)
        overlay[boundary] = overlay[boundary] * 0.1 + color * 0.9
    return Image.fromarray(np.clip(overlay, 0, 255).astype(np.uint8))


def make_cutout(image_array: np.ndarray, mask: np.ndarray) -> Image.Image:
    rgba = np.zeros((*mask.shape, 4), dtype=np.uint8)
    rgba[..., :3] = image_array
    rgba[..., 3] = mask.astype(np.uint8) * 255
    return Image.fromarray(rgba, mode="RGBA")


def mask_metrics(
    mask: np.ndarray, bbox_mask: np.ndarray
) -> dict[str, float | int]:
    foreground = int(mask.sum())
    outside = int((mask & ~bbox_mask).sum())
    return {
        "foreground_pixels": foreground,
        "image_area_fraction": float(mask.mean()),
        "outside_bbox_pixels": outside,
        "outside_bbox_mask_fraction": (
            float(outside / foreground) if foreground else 0.0
        ),
    }


def pair_metrics(mask_a: np.ndarray, mask_b: np.ndarray) -> dict[str, float | int]:
    intersection = int((mask_a & mask_b).sum())
    union = int((mask_a | mask_b).sum())
    smaller = min(int(mask_a.sum()), int(mask_b.sum()))
    return {
        "intersection_pixels": intersection,
        "mask_iou": float(intersection / union) if union else 0.0,
        "intersection_over_smaller_mask": (
            float(intersection / smaller) if smaller else 0.0
        ),
    }


def diagnostic_flags(
    object_a: dict[str, Any],
    object_b: dict[str, Any],
    pair: dict[str, Any],
    thresholds: dict[str, float],
) -> list[str]:
    flags: list[str] = []
    for label, metrics in (("object_a", object_a), ("object_b", object_b)):
        if metrics["foreground_pixels"] == 0:
            flags.append(f"{label}:empty_mask")
        elif (
            metrics["image_area_fraction"]
            < thresholds["near_empty_image_fraction"]
        ):
            flags.append(f"{label}:near_empty_mask")
        if (
            metrics["outside_bbox_mask_fraction"]
            > thresholds["outside_bbox_mask_fraction_warning"]
        ):
            flags.append(f"{label}:outside_bbox_warning")
    if pair["mask_iou"] > thresholds["ab_mask_iou_warning"]:
        flags.append("pair:ab_mask_iou_warning")
    if (
        pair["intersection_over_smaller_mask"]
        > thresholds["ab_intersection_over_smaller_mask_warning"]
    ):
        flags.append("pair:ab_containment_warning")
    return flags


def expected_output_paths(
    output_dir: Path, candidate_id: str, layout: dict[str, str]
) -> dict[str, Path]:
    return {
        "mask_a": output_dir
        / layout["masks_dir"]
        / f"{candidate_id}__object_a.png",
        "mask_b": output_dir
        / layout["masks_dir"]
        / f"{candidate_id}__object_b.png",
        "cutout_a": output_dir
        / layout["cutouts_dir"]
        / f"{candidate_id}__object_a.png",
        "cutout_b": output_dir
        / layout["cutouts_dir"]
        / f"{candidate_id}__object_b.png",
        "overlay": output_dir
        / layout["overlays_dir"]
        / f"{candidate_id}__mask_overlay.png",
        "record": output_dir
        / layout["records_dir"]
        / f"{candidate_id}.json",
    }


def verify_resume_record(
    record_path: Path,
    candidate_id: str,
    output_dir: Path,
    config_sha256: str,
    input_manifest_sha256: str,
    candidate_pool_sha256: str,
    source_path: Path,
) -> dict[str, Any]:
    record = read_json(record_path)
    if record["candidate_id"] != candidate_id:
        raise ValueError(f"Resume candidate mismatch: {record_path}")
    provenance = record["generation_provenance"]
    if provenance["config_sha256"] != config_sha256:
        raise ValueError(f"Resume config mismatch: {candidate_id}")
    if provenance["input_manifest_sha256"] != input_manifest_sha256:
        raise ValueError(f"Resume input mismatch: {candidate_id}")
    if provenance["candidate_pool_sha256"] != candidate_pool_sha256:
        raise ValueError(f"Resume candidate-pool mismatch: {candidate_id}")
    if not source_path.is_file():
        raise FileNotFoundError(f"Resume source image missing: {source_path}")
    if sha256(source_path) != record["source_image"]["sha256"]:
        raise ValueError(f"Resume source-image mismatch: {candidate_id}")
    if record.get("mask_generation_status") == "segmentation_failed":
        failure = record.get("segmentation_failure", {})
        if record.get("failure_record_version") != "segmentation_failure_v1.2":
            raise ValueError(f"Unknown failure record version: {candidate_id}")
        if not failure.get("exception_type") or not failure.get("reason"):
            raise ValueError(f"Incomplete failure record: {candidate_id}")
        return record
    for asset in ("mask_a", "mask_b", "cutout_a", "cutout_b", "overlay"):
        path = output_dir / record["assets"][asset]["path"]
        if not path.is_file():
            raise FileNotFoundError(f"Resume asset missing: {path}")
        if sha256(path) != record["assets"][asset]["sha256"]:
            raise ValueError(f"Resume checksum mismatch: {path}")
    return record


def build_qc_template(result: dict[str, Any]) -> dict[str, Any]:
    return {
        "schema_version": "1.1",
        "subset_name": result["pilot_subset"],
        "mask_test_id": result["mask_test_id"],
        "candidate_id": result["candidate_id"],
        "object_a_qc": {
            criterion: "not_tested" for criterion in QC_CRITERIA
        },
        "object_b_qc": {
            criterion: "not_tested" for criterion in QC_CRITERIA
        },
        "pair_qc": {
            "ab_mask_overlap": "not_tested",
        },
        "mask_qc_status": "not_tested",
        "mask_failure_reasons": [],
        "mask_qc_note": None,
        "mask_qc_reviewer": None,
        "mask_qc_timestamp": None,
        "technical_status_before": result["technical_status_before"],
        "technical_status_after_mask_qc": None,
        "technical_status_update_approved": False,
    }


def main() -> None:
    args = parse_args()
    config = read_json(args.config)
    layout = config["output_layout"]
    smoke_rows = read_jsonl(args.input_manifest)
    pool_rows = read_jsonl(args.candidate_pool)
    if len({row["candidate_id"] for row in pool_rows}) != len(pool_rows):
        raise ValueError("Duplicate candidate_id in candidate pool.")
    pool_by_id = {row["candidate_id"]: row for row in pool_rows}

    scope = config["scope"]
    if not smoke_rows:
        raise ValueError("Input manifest is empty.")
    if len({row["candidate_id"] for row in smoke_rows}) != len(smoke_rows):
        raise ValueError("Duplicate candidate_id in smoke manifest.")
    if len({row["source_image_id"] for row in smoke_rows}) != len(smoke_rows):
        raise ValueError("Duplicate source_image_id in smoke manifest.")
    order_field, ordered_rows = order_contract(smoke_rows)

    for row in smoke_rows:
        if row["pilot_subset"] != args.subset_name:
            raise ValueError(f"Subset mismatch: {row['candidate_id']}")
        if row["semantic_status"] not in scope["allowed_semantic_status"]:
            raise ValueError(f"Semantic-invalid candidate: {row['candidate_id']}")
        if row["technical_status"] not in scope["allowed_technical_status"]:
            raise ValueError(f"Disallowed technical status: {row['candidate_id']}")
        if (
            scope["development_exposure_required"]
            and row["development_exposure"] is not True
        ):
            raise ValueError(f"Non-development candidate: {row['candidate_id']}")
        for status_field in (
            "mask_qc_status",
            "edit_qc_status",
            "object_identity_preservation_status",
        ):
            if row[status_field] != "not_tested":
                raise ValueError(
                    f"{row['candidate_id']}: {status_field} must be not_tested."
                )
        if row["candidate_id"] not in pool_by_id:
            raise ValueError(
                f"Missing bbox provenance: {row['candidate_id']}"
            )
        pool = pool_by_id[row["candidate_id"]]
        if str(pool["source_image_id"]) != str(row["source_image_id"]):
            raise ValueError(f"Source-image mismatch: {row['candidate_id']}")
        for side in ("a", "b"):
            if pool[f"object_{side}_label"] != row[f"object_{side}_label"]:
                raise ValueError(
                    f"Object {side.upper()} label mismatch: {row['candidate_id']}"
                )
            if pool[f"object_{side}_color"] != row[f"original_color_{side}"]:
                raise ValueError(
                    f"Object {side.upper()} color mismatch: {row['candidate_id']}"
                )

    missing_source_paths = []
    for row in ordered_rows:
        source_path = resolve_path(args.project_root, row["original_image_path"])
        if not source_path.is_file():
            missing_source_paths.append(source_path.as_posix())

    if args.preflight_only:
        summary = {
            "mode": "preflight_only",
            "subset_name": args.subset_name,
            "input_candidates": len(ordered_rows),
            "preflight_passed": len(ordered_rows) - len(missing_source_paths),
            "ordering_field": order_field,
            "ordering_contiguous": True,
            "selection_seed_required": False,
            "segmentation_seed": config["seed"],
            "missing_source_paths": missing_source_paths,
            "actual_masks_generated": 0,
            "validation_errors": len(missing_source_paths),
        }
        print(json.dumps(summary, ensure_ascii=False, indent=2))
        if missing_source_paths:
            raise SystemExit(1)
        return

    output_files = {
        "metadata": args.output_dir / layout["run_metadata"],
        "manifest": args.output_dir / f"mask_results_{args.subset_name}_v1.1.jsonl",
        "qc_template": args.output_dir / f"mask_qc_review_template_{args.subset_name}_v1.1.jsonl",
        "summary": args.output_dir / f"segmentation_run_summary_{args.subset_name}_v1.1.json",
    }
    if args.output_dir.exists() and any(args.output_dir.iterdir()):
        if not args.resume:
            raise FileExistsError(
                f"Output is nonempty; use --resume only for verified records: "
                f"{args.output_dir}"
            )
    args.output_dir.mkdir(parents=True, exist_ok=True)
    for key in ("masks_dir", "cutouts_dir", "overlays_dir", "records_dir"):
        (args.output_dir / layout[key]).mkdir(parents=True, exist_ok=True)

    config_digest = sha256(args.config)
    input_digest = sha256(args.input_manifest)
    pool_digest = sha256(args.candidate_pool)
    model_config = config["model"]

    metadata = {
        "schema_version": "1.1",
        "run_id": f"attribute_binding_{args.subset_name}_segmentation_v1.1",
        "created_at": datetime.now(timezone.utc).isoformat(),
        "seed": config["seed"],
        "model_id": model_config["model_id"],
        "model_revision": model_config["revision"],
        "local_files_only": model_config["local_files_only"],
        "input_manifest": args.input_manifest.as_posix(),
        "input_manifest_sha256": input_digest,
        "candidate_pool": args.candidate_pool.as_posix(),
        "candidate_pool_sha256": pool_digest,
        "config": args.config.as_posix(),
        "config_sha256": config_digest,
        "candidate_ids": [row["candidate_id"] for row in ordered_rows],
        "scope_count": len(smoke_rows),
        "subset_name": args.subset_name,
        "ordering_field": order_field,
    }
    if output_files["metadata"].exists():
        if not args.resume:
            raise FileExistsError(output_files["metadata"])
        existing_metadata = read_json(output_files["metadata"])
        immutable_keys = (
            "seed",
            "model_id",
            "model_revision",
            "input_manifest_sha256",
            "candidate_pool_sha256",
            "config_sha256",
            "candidate_ids",
        )
        if any(
            existing_metadata[key] != metadata[key] for key in immutable_keys
        ):
            raise ValueError("Resume metadata does not match frozen inputs.")
        metadata = existing_metadata
    else:
        atomic_write_json(output_files["metadata"], metadata)

    seed = int(config["seed"])
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    if config["inference"]["deterministic_algorithms"]:
        os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
        torch.use_deterministic_algorithms(True)
    torch.backends.cudnn.benchmark = config["inference"]["cudnn_benchmark"]
    torch.backends.cudnn.deterministic = config["inference"][
        "cudnn_deterministic"
    ]

    device_setting = config["inference"]["device"]
    device = (
        "cuda"
        if device_setting == "auto" and torch.cuda.is_available()
        else "cpu"
        if device_setting == "auto"
        else device_setting
    )
    dtype_name = (
        config["inference"]["cuda_dtype"]
        if device.startswith("cuda")
        else config["inference"]["cpu_dtype"]
    )
    dtype = {"float16": torch.float16, "float32": torch.float32}[dtype_name]

    processor = SamProcessor.from_pretrained(
        model_config["model_id"],
        revision=model_config["revision"],
        local_files_only=model_config["local_files_only"],
        trust_remote_code=model_config["trust_remote_code"],
    )
    model = SamModel.from_pretrained(
        model_config["model_id"],
        revision=model_config["revision"],
        local_files_only=model_config["local_files_only"],
        trust_remote_code=model_config["trust_remote_code"],
        torch_dtype=dtype,
    ).to(device)
    model.eval()

    results: list[dict[str, Any]] = []
    generated_count = 0
    failed_count_this_run = 0
    resumed_count = 0
    for smoke in ordered_rows:
        candidate_id = smoke["candidate_id"]
        paths = expected_output_paths(args.output_dir, candidate_id, layout)
        source_path = resolve_path(
            args.project_root, smoke["original_image_path"]
        )
        if paths["record"].exists():
            if not args.resume:
                raise FileExistsError(paths["record"])
            results.append(
                verify_resume_record(
                    paths["record"],
                    candidate_id,
                    args.output_dir,
                    config_digest,
                    input_digest,
                    pool_digest,
                    source_path,
                )
            )
            resumed_count += 1
            continue
        unexpected = [
            path
            for name, path in paths.items()
            if name != "record" and path.exists()
        ]
        if unexpected:
            raise FileExistsError(
                f"Partial assets without resumable record for {candidate_id}: "
                f"{unexpected}"
            )

        try:
            pool = pool_by_id[candidate_id]
            if not source_path.is_file():
                raise FileNotFoundError(source_path)
            with Image.open(source_path) as opened:
                image = opened.convert("RGB")
            image_array = np.asarray(image)
            width, height = image.size
            bbox_a_norm = pool["object_a_bbox_xywh_norm"]
            bbox_b_norm = pool["object_b_bbox_xywh_norm"]
            bbox_a = bbox_xyxy_pixels(bbox_a_norm, width, height)
            bbox_b = bbox_xyxy_pixels(bbox_b_norm, width, height)
            boxes = [bbox_a, bbox_b]

            inputs = processor(image, input_boxes=[boxes], return_tensors="pt")
            model_inputs = {
                key: (
                    value.to(device=device, dtype=dtype)
                    if value.is_floating_point()
                    else value.to(device)
                )
                for key, value in inputs.items()
            }
            with torch.inference_mode():
                outputs = model(
                    **model_inputs,
                    multimask_output=config["inference"]["multimask_output"],
                )
            post_masks = processor.image_processor.post_process_masks(
                outputs.pred_masks.float().cpu(),
                inputs["original_sizes"].cpu(),
                inputs["reshaped_input_sizes"].cpu(),
            )[0]
            scores = outputs.iou_scores.float().cpu()[0]

            masks: list[np.ndarray] = []
            predicted_ious: list[float] = []
            for object_index in range(2):
                best_index = int(torch.argmax(scores[object_index]).item())
                masks.append(
                    post_masks[object_index, best_index].numpy().astype(bool)
                )
                predicted_ious.append(
                    float(scores[object_index, best_index].item())
                )
            mask_a, mask_b = masks

            atomic_save_image(
                Image.fromarray(mask_a.astype(np.uint8) * 255, mode="L"),
                paths["mask_a"],
            )
            atomic_save_image(
                Image.fromarray(mask_b.astype(np.uint8) * 255, mode="L"),
                paths["mask_b"],
            )
            atomic_save_image(make_cutout(image_array, mask_a), paths["cutout_a"])
            atomic_save_image(make_cutout(image_array, mask_b), paths["cutout_b"])
            atomic_save_image(
                make_overlay(image_array, mask_a, mask_b), paths["overlay"]
            )

            metrics_a = mask_metrics(
                mask_a, bbox_boolean_mask(bbox_a, width, height)
            )
            metrics_b = mask_metrics(
                mask_b, bbox_boolean_mask(bbox_b, width, height)
            )
            metrics_a["predicted_iou"] = predicted_ious[0]
            metrics_b["predicted_iou"] = predicted_ious[1]
            metrics_pair = pair_metrics(mask_a, mask_b)
            flags = diagnostic_flags(
                metrics_a,
                metrics_b,
                metrics_pair,
                config["automatic_diagnostic_flags"],
            )

            def asset_record(path: Path) -> dict[str, str]:
                return {
                    "path": path.relative_to(args.output_dir).as_posix(),
                    "sha256": sha256(path),
                }

            record = {
                "schema_version": "1.1",
                "mask_test_id": f"masktest_{args.subset_name}_v1.1_{candidate_id}",
                "candidate_id": candidate_id,
                "source_image_id": smoke["source_image_id"],
                "execution_order": smoke[order_field],
                "execution_order_field": order_field,
                "pilot_subset": smoke["pilot_subset"],
                "semantic_status": smoke["semantic_status"],
                "technical_status": smoke["technical_status"],
                "development_exposure": smoke["development_exposure"],
                "object_a": {
                    "detection_index": pool["object_a_detection_index"],
                    "label": smoke["object_a_label"],
                    "original_color": smoke["original_color_a"],
                    "bbox_xywh_norm": bbox_a_norm,
                    "bbox_xyxy_pixels": bbox_a,
                },
                "object_b": {
                    "detection_index": pool["object_b_detection_index"],
                    "label": smoke["object_b_label"],
                    "original_color": smoke["original_color_b"],
                    "bbox_xywh_norm": bbox_b_norm,
                    "bbox_xyxy_pixels": bbox_b,
                },
                "source_image": {
                    "path": smoke["original_image_path"],
                    "sha256": sha256(source_path),
                    "width": width,
                    "height": height,
                },
                "assets": {
                    "mask_a": asset_record(paths["mask_a"]),
                    "mask_b": asset_record(paths["mask_b"]),
                    "cutout_a": asset_record(paths["cutout_a"]),
                    "cutout_b": asset_record(paths["cutout_b"]),
                    "overlay": asset_record(paths["overlay"]),
                },
                "automatic_metrics": {
                    "object_a": metrics_a,
                    "object_b": metrics_b,
                    "pair": metrics_pair,
                },
                "automatic_diagnostic_flags": flags,
                "mask_generation_status": "generated",
                "mask_qc_status": "not_tested",
                "technical_status_before": smoke["technical_status"],
                "technical_status_after_mask_qc": None,
                "technical_status_update_approved": False,
                "edit_qc_status": "not_tested",
                "object_identity_preservation_status": "not_tested",
                "generation_provenance": {
                    "seed": seed,
                    "model_id": model_config["model_id"],
                    "model_revision": model_config["revision"],
                    "prompt_type": config["inference"]["prompt_type"],
                    "multimask_output": config["inference"]["multimask_output"],
                    "mask_selection": config["inference"]["mask_selection"],
                    "device": device,
                    "dtype": dtype_name,
                    "config_sha256": config_digest,
                    "input_manifest_sha256": input_digest,
                    "candidate_pool_sha256": pool_digest,
                },
            }
            if order_field == "selection_rank":
                record["selection_rank"] = smoke["selection_rank"]
                record["selection_seed"] = smoke["selection_seed"]
            else:
                record["frozen_order"] = smoke["frozen_order"]
            atomic_write_json(paths["record"], record)
            results.append(record)
            generated_count += 1
        except Exception as exc:
            if not args.continue_on_error:
                raise
            partial_assets = {}
            for name, path in paths.items():
                if name != "record" and path.exists():
                    partial_assets[name] = {
                        "path": path.relative_to(args.output_dir).as_posix(),
                        "sha256": sha256(path),
                    }
            source_record = {
                "path": smoke["original_image_path"],
                "sha256": sha256(source_path) if source_path.is_file() else None,
            }
            failure_record = {
                "schema_version": "1.1",
                "failure_record_version": "segmentation_failure_v1.2",
                "mask_test_id": (
                    f"masktest_{args.subset_name}_v1.1_{candidate_id}"
                ),
                "candidate_id": candidate_id,
                "source_image_id": smoke["source_image_id"],
                "execution_order": smoke[order_field],
                "execution_order_field": order_field,
                "pilot_subset": smoke["pilot_subset"],
                "semantic_status": smoke["semantic_status"],
                "technical_status": smoke["technical_status"],
                "development_exposure": smoke["development_exposure"],
                "object_a": {
                    "label": smoke["object_a_label"],
                    "original_color": smoke["original_color_a"],
                    "bbox_xywh_norm": pool_by_id[candidate_id][
                        "object_a_bbox_xywh_norm"
                    ],
                },
                "object_b": {
                    "label": smoke["object_b_label"],
                    "original_color": smoke["original_color_b"],
                    "bbox_xywh_norm": pool_by_id[candidate_id][
                        "object_b_bbox_xywh_norm"
                    ],
                },
                "source_image": source_record,
                "assets": {},
                "partial_assets": partial_assets,
                "automatic_metrics": None,
                "automatic_diagnostic_flags": [],
                "mask_generation_status": "segmentation_failed",
                "mask_qc_status": "not_tested",
                "technical_status_before": smoke["technical_status"],
                "technical_status_after_mask_qc": None,
                "technical_status_update_approved": False,
                "edit_qc_status": "not_tested",
                "object_identity_preservation_status": "not_tested",
                "segmentation_failure": {
                    "exception_type": type(exc).__name__,
                    "exception_message": str(exc),
                    "reason": f"{type(exc).__name__}: {exc}",
                    "traceback": traceback.format_exc(),
                    "recorded_at": datetime.now(timezone.utc).isoformat(),
                    "policy": "record_and_continue_without_input_mutation",
                },
                "generation_provenance": {
                    "seed": seed,
                    "model_id": model_config["model_id"],
                    "model_revision": model_config["revision"],
                    "prompt_type": config["inference"]["prompt_type"],
                    "multimask_output": config["inference"]["multimask_output"],
                    "mask_selection": config["inference"]["mask_selection"],
                    "device": device,
                    "dtype": dtype_name,
                    "config_sha256": config_digest,
                    "input_manifest_sha256": input_digest,
                    "candidate_pool_sha256": pool_digest,
                },
            }
            if order_field == "selection_rank":
                failure_record["selection_rank"] = smoke["selection_rank"]
                failure_record["selection_seed"] = smoke["selection_seed"]
            else:
                failure_record["frozen_order"] = smoke["frozen_order"]
            atomic_write_json(paths["record"], failure_record)
            results.append(failure_record)
            failed_count_this_run += 1

    results.sort(
        key=lambda row: row.get("execution_order", row.get("selection_rank"))
    )
    successful_results = [
        result
        for result in results
        if result.get("mask_generation_status") == "generated"
    ]
    failed_results = [
        result
        for result in results
        if result.get("mask_generation_status") == "segmentation_failed"
    ]
    qc_rows = [build_qc_template(result) for result in successful_results]
    if output_files["manifest"].exists():
        if not args.resume or read_jsonl(output_files["manifest"]) != results:
            raise FileExistsError(output_files["manifest"])
    else:
        atomic_write_jsonl(output_files["manifest"], results)
    if output_files["qc_template"].exists():
        if not args.resume or read_jsonl(output_files["qc_template"]) != qc_rows:
            raise FileExistsError(output_files["qc_template"])
    else:
        atomic_write_jsonl(output_files["qc_template"], qc_rows)

    summary = {
        "schema_version": "1.1",
        "run_id": metadata["run_id"],
        "status": (
            "generation_complete_with_failures_qc_not_tested"
            if failed_results
            else "generation_complete_qc_not_tested"
        ),
        "failure_policy_version": (
            "segmentation_failure_v1.2"
            if args.continue_on_error
            else None
        ),
        "candidate_count": len(results),
        "attempted_count": len(results),
        "successful_count": len(successful_results),
        "failed_count": len(failed_results),
        "generated_count_this_run": generated_count,
        "failed_count_this_run": failed_count_this_run,
        "resumed_count": resumed_count,
        "failed_candidates": [
            {
                "candidate_id": row["candidate_id"],
                "frozen_order": row.get("frozen_order"),
                "reason": row["segmentation_failure"]["reason"],
            }
            for row in failed_results
        ],
        "model_id": model_config["model_id"],
        "model_revision": model_config["revision"],
        "seed": seed,
        "device": device,
        "dtype": dtype_name,
        "software": {
            "python": platform.python_version(),
            "torch": torch.__version__,
            "transformers": transformers.__version__,
            "numpy": np.__version__,
            "pillow": PILLOW_VERSION,
        },
        "diagnostic_flag_counts": dict(
            sorted(
                {
                    flag: sum(
                        flag in row["automatic_diagnostic_flags"]
                        for row in successful_results
                    )
                    for flag in {
                        item
                        for row in successful_results
                        for item in row["automatic_diagnostic_flags"]
                    }
                }.items()
            )
        ),
        "mask_generation_status_counts": {
            "generated": len(successful_results),
            "segmentation_failed": len(failed_results),
        },
        "mask_qc_status_counts": {"not_tested": len(successful_results)},
        "technical_status_updates": 0,
        "color_editing_executed": False,
        "controlled_generation_executed": False,
        "vlm_inference_executed": False,
    }
    if output_files["summary"].exists():
        if not args.resume:
            raise FileExistsError(output_files["summary"])
    else:
        atomic_write_json(output_files["summary"], summary)
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
