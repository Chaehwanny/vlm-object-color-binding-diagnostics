#!/usr/bin/env python3
"""Create a checksummed, review-ID-based snapshot of the left/right pilot."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import shutil
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from compute_pilot_metrics import compute_metrics, load_predictions, write_outputs


def parse_args() -> argparse.Namespace:
    project_root = Path(__file__).resolve().parents[2]
    parser = argparse.ArgumentParser(description="Freeze left/right pilot v1 artifacts.")
    parser.add_argument(
        "--dataset-source",
        type=Path,
        default=project_root / "processed/spatial_left_right/datasets/positive_control_v1",
    )
    parser.add_argument(
        "--predictions-source",
        type=Path,
        default=project_root
        / "experiments/spatial_left_right/positive_control_v1/predictions/qwen2_5_vl_7b.jsonl",
    )
    parser.add_argument(
        "--readme-source",
        type=Path,
        default=project_root / "docs/left_right_pilot_v1_README.md",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=project_root / "frozen/left_right_pilot_v1",
    )
    return parser.parse_args()


def load_jsonl(path: Path) -> list[dict[str, Any]]:
    with path.open("r", encoding="utf-8") as input_file:
        return [json.loads(line) for line in input_file]


def write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as output_file:
        for row in rows:
            output_file.write(json.dumps(row, ensure_ascii=False) + "\n")


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as input_file:
        for chunk in iter(lambda: input_file.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def main() -> None:
    args = parse_args()
    if args.output_dir.exists():
        raise FileExistsError(
            f"Frozen output already exists: {args.output_dir}. "
            "A frozen snapshot must never be overwritten."
        )

    selection = json.loads((args.dataset_source / "selection.json").read_text(encoding="utf-8"))
    source_manifest = load_jsonl(args.dataset_source / "manifest.jsonl")
    source_predictions = load_predictions(args.predictions_source)
    review_to_internal = {
        sample["review_candidate_id"]: sample["sample_id"] for sample in selection["samples"]
    }
    internal_to_review = {internal: review for review, internal in review_to_internal.items()}

    dataset_dir = args.output_dir / "dataset"
    rewritten_manifest: list[dict[str, Any]] = []
    for row in source_manifest:
        review_id = row["review_candidate_id"]
        internal_id = row["sample_id"]
        source_sample_dir = args.dataset_source / "samples" / internal_id
        target_sample_dir = dataset_dir / review_id
        if not target_sample_dir.exists():
            shutil.copytree(source_sample_dir, target_sample_dir)
        rewritten = dict(row)
        rewritten["internal_sample_id"] = internal_id
        rewritten["sample_id"] = review_id
        rewritten["image_path"] = f"dataset/{review_id}/{Path(row['image_path']).name}"
        rewritten_manifest.append(rewritten)
    write_jsonl(args.output_dir / "manifest.jsonl", rewritten_manifest)

    rewritten_predictions: list[dict[str, Any]] = []
    for row in source_predictions:
        rewritten = dict(row)
        internal_id = row["sample_id"]
        review_id = row.get("review_candidate_id") or internal_to_review[internal_id]
        rewritten["internal_sample_id"] = internal_id
        rewritten["sample_id"] = review_id
        rewritten["image_path"] = f"dataset/{review_id}/{Path(row['image_path']).name}"
        rewritten_predictions.append(rewritten)
    predictions_path = args.output_dir / "model_outputs/qwen2_5_vl_7b.jsonl"
    write_jsonl(predictions_path, rewritten_predictions)

    human_review_path = args.output_dir / "human_review.csv"
    human_review_path.parent.mkdir(parents=True, exist_ok=True)
    with human_review_path.open("w", encoding="utf-8", newline="") as output_file:
        fieldnames = [
            "review_id",
            "object_a_identifiable",
            "object_b_identifiable",
            "relation_unambiguous",
            "flip_safe",
            "keep_primary",
            "record_provenance",
            "notes",
        ]
        writer = csv.DictWriter(output_file, fieldnames=fieldnames)
        writer.writeheader()
        for review_id in selection["selected_candidate_ids"]:
            writer.writerow(
                {
                    "review_id": review_id,
                    "object_a_identifiable": "yes",
                    "object_b_identifiable": "yes",
                    "relation_unambiguous": "yes",
                    "flip_safe": "yes",
                    "keep_primary": "yes",
                    "record_provenance": "reconstructed_from_pre_inference_selection_list",
                    "notes": "Original review CSV fields were not filled; selection occurred before model inference.",
                }
            )

    metrics, patterns = compute_metrics(rewritten_predictions)
    write_outputs(
        metrics,
        patterns,
        args.output_dir / "metrics/qwen2_5_vl_7b_metrics.json",
        args.output_dir / "metrics/qwen2_5_vl_7b_sample_patterns.csv",
    )

    code_dir = args.output_dir / "code"
    code_dir.mkdir(parents=True, exist_ok=True)
    for script_name in (
        "extract_left_right_candidates.py",
        "filter_left_right_candidates.py",
        "create_review_pool.py",
        "build_pilot_manifest.py",
        "run_qwen_pilot.py",
        "compute_pilot_metrics.py",
        "freeze_left_right_pilot.py",
    ):
        shutil.copy2(Path(__file__).resolve().parent / script_name, code_dir / script_name)
    shutil.copy2(args.readme_source, args.output_dir / "README.md")

    freeze_metadata = {
        "artifact": "left_right_pilot_v1",
        "frozen_at_utc": datetime.now(timezone.utc).isoformat(),
        "n_samples": len(selection["samples"]),
        "n_queries": len(rewritten_manifest),
        "primary_sample_ids": selection["selected_candidate_ids"],
        "source_paths": {
            "dataset": str(args.dataset_source),
            "predictions": str(args.predictions_source),
        },
        "immutability_policy": (
            "Do not overwrite or remove primary samples based on model outcomes. "
            "Any later quality audit must be reported as a sensitivity analysis."
        ),
    }
    (args.output_dir / "freeze_metadata.json").write_text(
        json.dumps(freeze_metadata, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )

    files = sorted(
        path for path in args.output_dir.rglob("*") if path.is_file() and path.name != "SHA256SUMS"
    )
    with (args.output_dir / "SHA256SUMS").open("w", encoding="utf-8") as checksum_file:
        for path in files:
            checksum_file.write(f"{sha256(path)}  {path.relative_to(args.output_dir)}\n")

    print(f"Frozen {len(selection['samples'])} samples at {args.output_dir}")


if __name__ == "__main__":
    main()
