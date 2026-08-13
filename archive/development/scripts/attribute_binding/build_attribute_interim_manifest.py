#!/usr/bin/env python3
"""Build the 18-condition manifest for the provisional n=8 development set."""

from __future__ import annotations

import argparse
import json
import shutil
from collections import Counter
from pathlib import Path
from typing import Any


PLURAL_LABELS = {"jeans", "pants", "shorts", "socks", "trousers"}


def parse_args() -> argparse.Namespace:
    root = Path(__file__).resolve().parents[2]
    source = root / "processed/attribute_binding/datasets/mini_provisional_v0.1"
    parser = argparse.ArgumentParser(description="Build attribute interim manifest.")
    parser.add_argument("--source-dir", type=Path, default=source)
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=root / "processed/attribute_binding/datasets/interim_n8_v1",
    )
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    with path.open("r", encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def be(label: str) -> str:
    return "are" if label.lower() in PLURAL_LABELS else "is"


def color_statement(label: str, color: str) -> str:
    return f"The {label} {be(label)} {color}."


def visibility_statement(label: str) -> str:
    return f"The {label} {be(label)} visible in the image."


def binding_statement(label_a: str, color_a: str, label_b: str, color_b: str) -> str:
    return (
        f"The {label_a} {be(label_a)} {color_a}, and the "
        f"{label_b} {be(label_b)} {color_b}."
    )


def condition(
    name: str,
    group: str,
    image_state: str,
    image_path: str,
    statement: str,
    expected: str,
    diagnostic_target: str,
) -> dict[str, Any]:
    return {
        "condition": name,
        "condition_group": group,
        "image_state": image_state,
        "image_path": image_path,
        "statement": statement,
        "expected_answer": expected,
        "diagnostic_target": diagnostic_target,
    }


def build_conditions(
    label_a: str,
    color_a: str,
    label_b: str,
    color_b: str,
    original_path: str,
    edited_path: str,
) -> list[dict[str, Any]]:
    original_binding = binding_statement(label_a, color_a, label_b, color_b)
    edited_binding = binding_statement(label_a, color_b, label_b, color_a)
    color_presence = f"A {color_a} object and a {color_b} object are visible in the image."
    return [
        condition("binding_true", "core", "original", original_path, original_binding, "Yes", "original_binding"),
        condition("binding_false", "core", "original", original_path, edited_binding, "No", "original_binding"),
        condition("image_conflict", "core", "edited", edited_path, original_binding, "No", "edited_binding"),
        condition("edit_control", "core", "edited", edited_path, edited_binding, "Yes", "edited_binding"),
        condition("original_a_visible", "atomic_visibility", "original", original_path, visibility_statement(label_a), "Yes", "object_a_visibility"),
        condition("original_b_visible", "atomic_visibility", "original", original_path, visibility_statement(label_b), "Yes", "object_b_visibility"),
        condition("edited_a_visible", "atomic_visibility", "edited", edited_path, visibility_statement(label_a), "Yes", "object_a_visibility"),
        condition("edited_b_visible", "atomic_visibility", "edited", edited_path, visibility_statement(label_b), "Yes", "object_b_visibility"),
        condition("original_a_source_color", "atomic_color", "original", original_path, color_statement(label_a, color_a), "Yes", "object_a_color"),
        condition("original_a_target_color", "atomic_color", "original", original_path, color_statement(label_a, color_b), "No", "object_a_color"),
        condition("original_b_source_color", "atomic_color", "original", original_path, color_statement(label_b, color_b), "Yes", "object_b_color"),
        condition("original_b_target_color", "atomic_color", "original", original_path, color_statement(label_b, color_a), "No", "object_b_color"),
        condition("edited_a_source_color", "atomic_color", "edited", edited_path, color_statement(label_a, color_a), "No", "object_a_color"),
        condition("edited_a_target_color", "atomic_color", "edited", edited_path, color_statement(label_a, color_b), "Yes", "object_a_color"),
        condition("edited_b_source_color", "atomic_color", "edited", edited_path, color_statement(label_b, color_b), "No", "object_b_color"),
        condition("edited_b_target_color", "atomic_color", "edited", edited_path, color_statement(label_b, color_a), "Yes", "object_b_color"),
        condition("original_color_presence", "atomic_presence", "original", original_path, color_presence, "Yes", "bag_of_colors"),
        condition("edited_color_presence", "atomic_presence", "edited", edited_path, color_presence, "Yes", "bag_of_colors"),
    ]


def main() -> None:
    args = parse_args()
    if args.output_dir.exists() and any(args.output_dir.iterdir()) and not args.overwrite:
        raise FileExistsError(f"Output directory is not empty: {args.output_dir}")
    rows = read_jsonl(args.source_dir / "samples.jsonl")
    if len(rows) != 8:
        raise ValueError(f"Expected 8 provisional samples, found {len(rows)}")

    original_dir = args.output_dir / "images/original"
    edited_dir = args.output_dir / "images/edited"
    original_dir.mkdir(parents=True, exist_ok=True)
    edited_dir.mkdir(parents=True, exist_ok=True)
    samples = []
    manifest = []
    for index, row in enumerate(rows, start=1):
        sample_id = f"binding_interim_{index:03d}"
        original_target = original_dir / f"{sample_id}.png"
        edited_target = edited_dir / f"{sample_id}.png"
        shutil.copy2(args.source_dir / row["original_path"], original_target)
        shutil.copy2(args.source_dir / row["edited_path"], edited_target)
        relative_original = str(original_target.relative_to(args.output_dir))
        relative_edited = str(edited_target.relative_to(args.output_dir))
        object_a, object_b = row["object_a"], row["object_b"]
        sample = {
            "sample_id": sample_id,
            "source_candidate_id": row["sample_id"],
            "source_dataset": row["source_dataset"],
            "source_image_id": row["image_id"],
            "object_a_label": object_a["label"],
            "object_a_original_color": object_a["color"],
            "object_b_label": object_b["label"],
            "object_b_original_color": object_b["color"],
            "color_pair": row["color_pair"],
            "original_image_path": relative_original,
            "edited_image_path": relative_edited,
            "outside_edit_pixels_identical": row["outside_edit_pixels_identical"],
            "selection_status": "frozen_interim_development_sample",
            "confirmatory_test_eligible": False,
        }
        samples.append(sample)
        manifest.extend(
            {
                **sample,
                **item,
            }
            for item in build_conditions(
                object_a["label"],
                object_a["color"],
                object_b["label"],
                object_b["color"],
                relative_original,
                relative_edited,
            )
        )

    for name, output_rows in (("samples.jsonl", samples), ("manifest.jsonl", manifest)):
        with (args.output_dir / name).open("w", encoding="utf-8") as handle:
            for row in output_rows:
                handle.write(json.dumps(row, ensure_ascii=False) + "\n")
    summary = {
        "schema_version": "0.1",
        "study_role": "interim_development_smoke_test",
        "n_samples": len(samples),
        "conditions_per_sample": 18,
        "n_conditions": len(manifest),
        "color_pair_counts": dict(sorted(Counter(row["color_pair"] for row in samples).items())),
        "development_data": True,
        "confirmatory_test_eligible": False,
        "generalization_claim_allowed": False,
        "model_outputs_used_for_selection": False,
    }
    (args.output_dir / "summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
