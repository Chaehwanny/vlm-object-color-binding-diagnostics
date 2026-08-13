#!/usr/bin/env python3
"""Collect pre-inference accepted edits into one provisional mini set."""

from __future__ import annotations

import argparse
import json
import shutil
from collections import Counter
from pathlib import Path
from typing import Any


SELECTIONS = [
    {
        "manifest": "processed/attribute_binding/edits/mini_color_swap_edits_v0.1/edit_manifest.jsonl",
        "root": "processed/attribute_binding/edits/mini_color_swap_edits_v0.1",
        "ids": {"mini_review_007", "mini_review_038"},
        "decision_stage": "strict_initial_post_edit_audit",
    },
    {
        "manifest": "processed/attribute_binding/edits/same_owner_color_swap_edits_v0.1/edit_manifest.jsonl",
        "root": "processed/attribute_binding/edits/same_owner_color_swap_edits_v0.1",
        "ids": {"mini_review_002", "mini_review_010"},
        "decision_stage": "same_owner_criterion_correction",
    },
    {
        "manifest": "processed/attribute_binding/edits/unique_garment_color_swap_edits_v0.1/edit_manifest.jsonl",
        "root": "processed/attribute_binding/edits/unique_garment_color_swap_edits_v0.1",
        "ids": {"mini_review_005", "mini_review_039"},
        "decision_stage": "unique_label_criterion_correction",
    },
    {
        "manifest": "processed/attribute_binding/edits/paco_color_swap_edits_v0.1/edit_manifest.jsonl",
        "root": "processed/attribute_binding/edits/paco_color_swap_edits_v0.1",
        "ids": {"paco_color_005", "paco_color_007"},
        "decision_stage": "paco_strict_post_edit_audit",
    },
]


def parse_args() -> argparse.Namespace:
    root = Path(__file__).resolve().parents[2]
    parser = argparse.ArgumentParser(description="Freeze provisional mini set.")
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=root / "processed/attribute_binding/datasets/mini_provisional_v0.1",
    )
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    with path.open("r", encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def main() -> None:
    args = parse_args()
    project_root = Path(__file__).resolve().parents[2]
    if args.output_dir.exists() and any(args.output_dir.iterdir()) and not args.overwrite:
        raise FileExistsError(f"Output directory is not empty: {args.output_dir}")

    originals_dir = args.output_dir / "originals"
    edited_dir = args.output_dir / "edited"
    comparisons_dir = args.output_dir / "comparisons"
    for directory in (originals_dir, edited_dir, comparisons_dir):
        directory.mkdir(parents=True, exist_ok=True)

    selected_rows = []
    for selection in SELECTIONS:
        manifest_path = project_root / selection["manifest"]
        artifact_root = project_root / selection["root"]
        rows = {row["segmentation_id"]: row for row in read_jsonl(manifest_path)}
        missing = selection["ids"] - set(rows)
        if missing:
            raise ValueError(f"Missing IDs in {manifest_path}: {sorted(missing)}")
        for sample_id in sorted(selection["ids"]):
            row = rows[sample_id]
            source_original = artifact_root / row["paired_original_path"]
            source_edited = artifact_root / row["edited_path"]
            source_comparison = artifact_root / row["comparison_path"]
            target_original = originals_dir / f"{sample_id}_original.png"
            target_edited = edited_dir / f"{sample_id}_edited.png"
            target_comparison = comparisons_dir / f"{sample_id}_comparison.jpg"
            shutil.copy2(source_original, target_original)
            shutil.copy2(source_edited, target_edited)
            shutil.copy2(source_comparison, target_comparison)
            color_pair = "_".join(
                sorted((row["object_a"]["color"], row["object_b"]["color"]))
            )
            selected_rows.append(
                {
                    "schema_version": "0.1",
                    "sample_id": sample_id,
                    "source_dataset": row["source_dataset"],
                    "image_id": row["image_id"],
                    "object_a": row["object_a"],
                    "object_b": row["object_b"],
                    "color_pair": color_pair,
                    "original_path": str(target_original.relative_to(args.output_dir)),
                    "edited_path": str(target_edited.relative_to(args.output_dir)),
                    "comparison_path": str(
                        target_comparison.relative_to(args.output_dir)
                    ),
                    "selected_color_fraction_a": row["selected_color_fraction_a"],
                    "selected_color_fraction_b": row["selected_color_fraction_b"],
                    "outside_edit_pixels_identical": row[
                        "outside_edit_pixels_identical"
                    ],
                    "decision_stage": selection["decision_stage"],
                    "second_human_review_status": "pending",
                    "development_data_only": True,
                }
            )

    selected_rows.sort(key=lambda row: row["sample_id"])
    with (args.output_dir / "samples.jsonl").open("w", encoding="utf-8") as handle:
        for row in selected_rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")

    summary = {
        "schema_version": "0.1",
        "status": "provisional_pre_inference",
        "samples": len(selected_rows),
        "target_minimum": 20,
        "target_met": len(selected_rows) >= 20,
        "color_pair_counts": dict(
            sorted(Counter(row["color_pair"] for row in selected_rows).items())
        ),
        "source_dataset_counts": dict(
            sorted(Counter(row["source_dataset"] for row in selected_rows).items())
        ),
        "model_outputs_used_for_decision": False,
        "second_human_review_required": True,
        "model_inference_allowed": False,
        "note": (
            "Do not build the 18-condition query manifest or run VLMs until the "
            "sample target and second-human quality gate are resolved."
        ),
    }
    (args.output_dir / "summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
