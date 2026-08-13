#!/usr/bin/env python3
"""Extract left/right relation candidates from the local FiftyOne GQA export.

The source ``samples.json`` is never modified.  This script writes one JSON
record per annotated left/right relation so later filtering thresholds can be
changed without reparsing the source dataset.
"""

from __future__ import annotations

import argparse
import json
import sys
from collections import Counter
from pathlib import Path
from typing import Any


PREDICATE_MAP = {
    "to the left of": "left_of",
    "left of": "left_of",
    "to the right of": "right_of",
    "right of": "right_of",
}


def parse_args() -> argparse.Namespace:
    project_root = Path(__file__).resolve().parents[2]
    parser = argparse.ArgumentParser(
        description="Extract annotated left/right relation candidates from GQA."
    )
    parser.add_argument(
        "--input",
        type=Path,
        default=project_root / "raw/GQA-Scene-Graph/samples.json",
        help="Path to the FiftyOne samples.json export.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=project_root
        / "processed/spatial_left_right/candidates/candidates_left_right.jsonl",
        help="Output JSONL path.",
    )
    return parser.parse_args()


def get_detections(sample: dict[str, Any]) -> list[dict[str, Any]]:
    return sample.get("detections", {}).get("detections", [])


def main() -> None:
    args = parse_args()
    if not args.input.is_file():
        raise FileNotFoundError(f"Input file not found: {args.input}")

    print(f"Loading {args.input}", file=sys.stderr)
    with args.input.open("r", encoding="utf-8") as source_file:
        payload = json.load(source_file)

    samples = payload.get("samples", [])
    if not isinstance(samples, list):
        raise ValueError("Expected a top-level 'samples' list in samples.json")

    args.output.parent.mkdir(parents=True, exist_ok=True)
    written = 0
    relation_counts: Counter[str] = Counter()

    with args.output.open("w", encoding="utf-8") as output_file:
        for sample_index, sample in enumerate(samples):
            detections = get_detections(sample)
            labels = [str(detection.get("label", "")).strip().lower() for detection in detections]
            label_counts = Counter(label for label in labels if label)

            for source_index, source in enumerate(detections):
                source_label = labels[source_index]
                if not source_label:
                    continue

                for relation in source.get("relations", []):
                    raw_predicate = str(relation.get("name", "")).strip().lower()
                    canonical_predicate = PREDICATE_MAP.get(raw_predicate)
                    target_label = str(relation.get("object", "")).strip().lower()

                    if canonical_predicate is None or not target_label:
                        continue

                    target_indices = [
                        index
                        for index, label in enumerate(labels)
                        if label == target_label
                    ]
                    record = {
                        "source_dataset": "Voxel51/GQA-Scene-Graph",
                        "sample_index": sample_index,
                        "image_path": sample.get("filepath"),
                        "image_id": Path(str(sample.get("filepath", ""))).stem,
                        "source_detection_index": source_index,
                        "source_label": source_label,
                        "source_label_count": label_counts[source_label],
                        "source_bbox_xywh_norm": source.get("bounding_box"),
                        "relation_raw": raw_predicate,
                        "relation": canonical_predicate,
                        "target_label": target_label,
                        "target_label_count": label_counts[target_label],
                        "target_detection_indices": target_indices,
                        "target_bboxes_xywh_norm": [
                            detections[index].get("bounding_box") for index in target_indices
                        ],
                    }
                    output_file.write(json.dumps(record, ensure_ascii=False) + "\n")
                    written += 1
                    relation_counts[canonical_predicate] += 1

            if (sample_index + 1) % 5000 == 0:
                print(
                    f"Processed {sample_index + 1}/{len(samples)} images; "
                    f"wrote {written} candidates.",
                    file=sys.stderr,
                )

    print(
        "Done. "
        f"Wrote {written} candidates "
        f"(left_of={relation_counts['left_of']}, "
        f"right_of={relation_counts['right_of']}) to {args.output}",
        file=sys.stderr,
    )


if __name__ == "__main__":
    main()
