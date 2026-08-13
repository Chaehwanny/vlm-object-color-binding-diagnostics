#!/usr/bin/env python3
"""Build the five-condition left/right pilot dataset from reviewed candidates."""

from __future__ import annotations

import argparse
import json
import shutil
from pathlib import Path
from typing import Any

from PIL import Image


DEFAULT_SELECTED_IDS = (
    "001,003,013,014,015,022,027,047,053,078,"
    "121,132,134,170,222,235"
)


def parse_args() -> argparse.Namespace:
    project_root = Path(__file__).resolve().parents[2]
    parser = argparse.ArgumentParser(
        description="Create original/flip images and five-condition manifests for the pilot."
    )
    parser.add_argument(
        "--review-pool",
        type=Path,
        default=project_root / "processed/spatial_left_right/reviews/review_pool_v1/review_pool.jsonl",
    )
    parser.add_argument(
        "--image-root",
        type=Path,
        default=project_root / "raw/GQA-Scene-Graph",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=project_root / "processed/spatial_left_right/datasets/positive_control_v1",
    )
    parser.add_argument("--selected-ids", default=DEFAULT_SELECTED_IDS)
    return parser.parse_args()


def load_review_records(path: Path) -> dict[str, dict[str, Any]]:
    records: dict[str, dict[str, Any]] = {}
    with path.open("r", encoding="utf-8") as review_file:
        for line in review_file:
            record = json.loads(line)
            records[record["candidate_id"]] = record
    return records


def relation_word(relation: str) -> str:
    return "left" if relation == "left_of" else "right"


def opposite_relation(relation: str) -> str:
    return "right_of" if relation == "left_of" else "left_of"


def statement(source_label: str, relation: str, target_label: str) -> str:
    return (
        f"The {source_label} is to the {relation_word(relation)} "
        f"of the {target_label}."
    )


def flip_bbox(bbox: list[float]) -> list[float]:
    x, y, width, height = bbox
    return [1 - x - width, y, width, height]


def main() -> None:
    args = parse_args()
    if not args.review_pool.is_file():
        raise FileNotFoundError(f"Review pool not found: {args.review_pool}")

    requested_ids = [f"review_{item.strip():0>3}" for item in args.selected_ids.split(",")]
    review_records = load_review_records(args.review_pool)
    missing_ids = [candidate_id for candidate_id in requested_ids if candidate_id not in review_records]
    if missing_ids:
        raise ValueError(f"Candidate IDs not found in review pool: {', '.join(missing_ids)}")

    selected = [review_records[candidate_id] for candidate_id in requested_ids]
    relation_counts = {
        relation: sum(record["relation"] == relation for record in selected)
        for relation in ("left_of", "right_of")
    }

    samples_dir = args.output_dir / "samples"
    args.output_dir.mkdir(parents=True, exist_ok=True)
    manifest_path = args.output_dir / "manifest.jsonl"
    selection_path = args.output_dir / "selection.json"

    manifest_rows: list[dict[str, Any]] = []
    selection_rows: list[dict[str, Any]] = []
    for index, record in enumerate(selected, start=1):
        sample_id = f"pilot_{index:03d}_{record['image_id']}"
        sample_dir = samples_dir / sample_id
        sample_dir.mkdir(parents=True, exist_ok=True)
        original_path = sample_dir / "original.jpg"
        flipped_path = sample_dir / "flipped.jpg"
        source_path = args.image_root / record["image_path"]
        if not source_path.is_file():
            raise FileNotFoundError(f"Source image not found: {source_path}")

        shutil.copy2(source_path, original_path)
        with Image.open(source_path) as source_image:
            source_image.transpose(Image.Transpose.FLIP_LEFT_RIGHT).save(flipped_path, quality=95)

        original_relation = record["relation"]
        flipped_relation = opposite_relation(original_relation)
        true_statement = statement(
            record["source_label"], original_relation, record["target_label"]
        )
        false_statement = statement(
            record["source_label"], flipped_relation, record["target_label"]
        )
        conditions = [
            {
                "name": "baseline_true",
                "image": "original.jpg",
                "statement": true_statement,
                "answer": "Yes",
            },
            {
                "name": "baseline_false",
                "image": "original.jpg",
                "statement": false_statement,
                "answer": "No",
            },
            {
                "name": "irrelevant_true",
                "image": "original.jpg",
                "statement": (
                    f"The {record['source_label']} and the {record['target_label']} "
                    "are both visible in the image."
                ),
                "answer": "Yes",
            },
            {
                "name": "image_conflict",
                "image": "flipped.jpg",
                "statement": true_statement,
                "answer": "No",
            },
            {
                "name": "flip_control",
                "image": "flipped.jpg",
                "statement": false_statement,
                "answer": "Yes",
            },
        ]
        sample_metadata = {
            "sample_id": sample_id,
            "review_candidate_id": record["candidate_id"],
            "source_dataset": record["source_dataset"],
            "source_image_path": record["image_path"],
            "source_label": record["source_label"],
            "target_label": record["target_label"],
            "original_relation": original_relation,
            "flipped_relation": flipped_relation,
            "source_bbox_xywh_norm": record["source_bbox_xywh_norm"],
            "target_bbox_xywh_norm": record["target_bbox_xywh_norm"],
            "flipped_source_bbox_xywh_norm": flip_bbox(record["source_bbox_xywh_norm"]),
            "flipped_target_bbox_xywh_norm": flip_bbox(record["target_bbox_xywh_norm"]),
            "geometry": record["geometry"],
            "conditions": conditions,
        }
        with (sample_dir / "queries.json").open("w", encoding="utf-8") as query_file:
            json.dump({"conditions": conditions}, query_file, ensure_ascii=False, indent=2)
            query_file.write("\n")
        with (sample_dir / "meta.json").open("w", encoding="utf-8") as meta_file:
            json.dump(sample_metadata, meta_file, ensure_ascii=False, indent=2)
            meta_file.write("\n")

        for condition in conditions:
            manifest_rows.append(
                {
                    "sample_id": sample_id,
                    "review_candidate_id": record["candidate_id"],
                    "condition": condition["name"],
                    "image_path": str((sample_dir / condition["image"]).relative_to(args.output_dir)),
                    "statement": condition["statement"],
                    "expected_answer": condition["answer"],
                    "relation_orientation": original_relation,
                    "source_label": record["source_label"],
                    "target_label": record["target_label"],
                }
            )
        selection_rows.append(
            {
                "sample_id": sample_id,
                "review_candidate_id": record["candidate_id"],
                "relation": original_relation,
                "source_label": record["source_label"],
                "target_label": record["target_label"],
                "selection_note": (
                    "Manually reviewed under the pilot's object uniqueness, geometric "
                    "clarity, and flip-safety criteria."
                ),
            }
        )

    with manifest_path.open("w", encoding="utf-8") as manifest_file:
        for row in manifest_rows:
            manifest_file.write(json.dumps(row, ensure_ascii=False) + "\n")
    with selection_path.open("w", encoding="utf-8") as selection_file:
        json.dump(
            {
                "dataset_version": "pilot_left_right_v1",
                "selected_candidate_ids": requested_ids,
                "n_samples": len(selected),
                "n_conditions": len(manifest_rows),
                "relation_counts": relation_counts,
                "selection_note": (
                    "This is an exploratory pilot. The relation orientation is imbalanced, "
                    "so results must be reported by orientation as well as pooled."
                ),
                "samples": selection_rows,
            },
            selection_file,
            ensure_ascii=False,
            indent=2,
        )
        selection_file.write("\n")

    print(
        f"Built {len(selected)} samples and {len(manifest_rows)} condition rows in "
        f"{args.output_dir}"
    )
    print(f"Relation counts: {relation_counts}")


if __name__ == "__main__":
    main()
