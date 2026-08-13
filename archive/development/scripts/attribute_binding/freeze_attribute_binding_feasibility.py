#!/usr/bin/env python3
"""Freeze five color-binding pairs and build core plus atomic conditions."""

from __future__ import annotations

import argparse
import json
import shutil
from pathlib import Path
from typing import Any


PLURAL_LABELS = {"jeans", "pants", "shorts", "trousers"}


def parse_args() -> argparse.Namespace:
    project_root = Path(__file__).resolve().parents[2]
    parser = argparse.ArgumentParser(description="Freeze binding feasibility v1.")
    parser.add_argument(
        "--edit-manifest",
        type=Path,
        default=project_root
        / "processed/attribute_binding/edits/color_swap_edits_final_v0.2/edit_manifest.jsonl",
    )
    parser.add_argument(
        "--edit-root",
        type=Path,
        default=project_root
        / "processed/attribute_binding/edits/color_swap_edits_final_v0.2",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=project_root / "processed/attribute_binding/datasets/feasibility_n5_v1",
    )
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    with path.open("r", encoding="utf-8") as input_file:
        return [json.loads(line) for line in input_file]


def be(label: str) -> str:
    return "are" if label.lower() in PLURAL_LABELS else "is"


def color_statement(label: str, color: str) -> str:
    return f"The {label} {be(label)} {color}."


def visibility_statement(label: str) -> str:
    return f"The {label} {be(label)} visible in the image."


def binding_statement(
    label_a: str, color_a: str, label_b: str, color_b: str
) -> str:
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
    color_presence = (
        f"A {color_a} object and a {color_b} object are visible in the image."
    )
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
        raise FileExistsError(
            f"Output directory is not empty: {args.output_dir}. Use --overwrite."
        )

    rows = read_jsonl(args.edit_manifest)
    if len(rows) != 5:
        raise ValueError(f"Expected exactly five frozen samples, found {len(rows)}")

    original_dir = args.output_dir / "images/original"
    edited_dir = args.output_dir / "images/edited"
    original_dir.mkdir(parents=True, exist_ok=True)
    edited_dir.mkdir(parents=True, exist_ok=True)

    sample_rows = []
    manifest_rows = []
    for index, row in enumerate(rows, start=1):
        sample_id = f"binding_feas_{index:03d}"
        original_target = original_dir / f"{sample_id}.png"
        edited_target = edited_dir / f"{sample_id}.png"
        shutil.copy2(args.edit_root / row["paired_original_path"], original_target)
        shutil.copy2(args.edit_root / row["edited_path"], edited_target)
        relative_original = str(original_target.relative_to(args.output_dir))
        relative_edited = str(edited_target.relative_to(args.output_dir))

        object_a = row["object_a"]
        object_b = row["object_b"]
        sample = {
            "sample_id": sample_id,
            "source_candidate_id": row["segmentation_id"],
            "source_dataset": row["source_dataset"],
            "source_image_id": row["image_id"],
            "object_a_label": object_a["label"],
            "object_a_original_color": object_a["color"],
            "object_b_label": object_b["label"],
            "object_b_original_color": object_b["color"],
            "original_image_path": relative_original,
            "edited_image_path": relative_edited,
            "edit_method": row["edit_method"],
            "outside_edit_pixels_identical": row["outside_edit_pixels_identical"],
            "selection_status": "frozen_feasibility_development_sample",
            "confirmatory_test_eligible": False,
        }
        sample_rows.append(sample)
        for item in build_conditions(
            object_a["label"],
            object_a["color"],
            object_b["label"],
            object_b["color"],
            relative_original,
            relative_edited,
        ):
            manifest_rows.append({**sample, **item})

    with (args.output_dir / "samples.jsonl").open(
        "w", encoding="utf-8"
    ) as output_file:
        for row in sample_rows:
            output_file.write(json.dumps(row, ensure_ascii=False) + "\n")
    with (args.output_dir / "manifest.jsonl").open(
        "w", encoding="utf-8"
    ) as output_file:
        for row in manifest_rows:
            output_file.write(json.dumps(row, ensure_ascii=False) + "\n")

    summary = {
        "schema_version": "1.0",
        "n_samples": len(sample_rows),
        "conditions_per_sample": len(manifest_rows) // len(sample_rows),
        "n_conditions": len(manifest_rows),
        "core_conditions_per_sample": 4,
        "atomic_conditions_per_sample": 14,
        "development_data": True,
        "implausible_color_object_pairs_included": False,
        "plausibility_note": (
            "Semantically implausible colors are reserved for a separately "
            "controlled mini-experiment stress condition."
        ),
    }
    (args.output_dir / "freeze_summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    (args.output_dir / "README.md").write_text(
        """# Attribute-Object Binding Feasibility v1

Five visually reviewed, counterfactually color-swapped development pairs.

- Four core binding conditions per sample
- Fourteen atomic visibility, color, and bag-of-colors controls per sample
- Original and edited images are matched PNG files
- Pixels outside the selected edit masks are identical
- Implausible object-color pairs are excluded from this feasibility gate
- These development samples are not eligible for the confirmatory main test
""",
        encoding="utf-8",
    )
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
