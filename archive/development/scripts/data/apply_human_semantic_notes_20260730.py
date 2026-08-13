#!/usr/bin/env python3
"""Transcribe the 2026-07-30 human semantic review notes into a manifest."""

from __future__ import annotations

import argparse
import csv
import json
import re
from pathlib import Path
from typing import Any


VALID_WITH_DIFFICULTY = {
    8,
    10,
    11,
    20,
    23,
    31,
    33,
    38,
    45,
    62,
    82,
    91,
    106,
    126,
    155,
    187,
}

HUMAN_REVIEW: set[int] = set()

CODEX_VISUAL_ADJUDICATION = {
    16, 38, 45, 60, 62, 88, 106, 133, 155, 180, 187, 191, 193
}

LABEL_MISMATCH = {
    14,
    39,
    64,
    71,
    102,
    154,
    190,
    194,
    266,
}

NONUNIQUE_REFERENCE = {124}

OBJECT_UNIDENTIFIABLE = {
    5,
    9,
    12,
    19,
    28,
    30,
    43,
    49,
    65,
    66,
    86,
    111,
    125,
    151,
    165,
    182,
    188,
    200,
    207,
    232,
    264,
    270,
    278,
}

DIFFICULTY_TAGS = {
    8: ["low_color_contrast"],
    10: ["partial_occlusion"],
    11: ["text_or_logo"],
    16: ["patterned_surface", "low_color_contrast"],
    20: ["thin_structure", "small_object"],
    23: ["partial_occlusion"],
    31: ["thin_structure", "small_object"],
    33: ["low_color_contrast"],
    38: ["thin_structure", "patterned_surface", "low_color_contrast"],
    45: ["patterned_surface", "text_or_logo", "low_color_contrast"],
    60: ["low_color_contrast"],
    62: ["low_color_contrast"],
    82: ["patterned_surface"],
    88: ["partial_occlusion", "low_color_contrast"],
    91: ["partial_occlusion"],
    106: ["patterned_surface", "low_color_contrast"],
    126: ["partial_occlusion"],
    133: ["low_color_contrast"],
    155: ["thin_structure", "low_color_contrast"],
    180: ["low_color_contrast"],
    187: ["patterned_surface", "low_color_contrast"],
    191: ["low_color_contrast"],
    193: ["patterned_surface", "low_color_contrast"],
}

ROW_PATTERN = re.compile(r"^\s*(\d+)[.)]?\s*(.*)$")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--prepared-manifest", type=Path, required=True)
    parser.add_argument("--notes-file", type=Path, required=True)
    parser.add_argument("--output-jsonl", type=Path, required=True)
    parser.add_argument("--output-csv", type=Path, required=True)
    return parser.parse_args()


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    with path.open(encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def read_notes(path: Path) -> dict[int, str]:
    notes: dict[int, str] = {}
    for raw_line in path.read_text(encoding="utf-8").splitlines():
        match = ROW_PATTERN.match(raw_line)
        if match:
            notes[int(match.group(1))] = match.group(2).strip()
    return notes


def invalid_reason(index: int) -> str:
    if index in LABEL_MISMATCH:
        return "label_mismatch"
    if index in NONUNIQUE_REFERENCE:
        return "nonunique_or_unresolvable_reference"
    if index in OBJECT_UNIDENTIFIABLE:
        return "object_unidentifiable"
    return "target_color_indeterminate"


def ensure_new(path: Path) -> None:
    if path.exists():
        raise FileExistsError(f"Refusing to overwrite existing output: {path}")
    path.parent.mkdir(parents=True, exist_ok=True)


def main() -> None:
    args = parse_args()
    rows = read_jsonl(args.prepared_manifest)
    notes = read_notes(args.notes_file)
    if len(rows) != 293:
        raise ValueError(f"Expected 293 prepared rows, found {len(rows)}.")
    if len(notes) != 110:
        raise ValueError(f"Expected 110 noted candidates, found {len(notes)}.")

    noted_ids = set(notes)
    categorized = (
        VALID_WITH_DIFFICULTY
        | HUMAN_REVIEW
        | LABEL_MISMATCH
        | NONUNIQUE_REFERENCE
        | OBJECT_UNIDENTIFIABLE
    )
    if categorized - noted_ids:
        raise ValueError(f"Categorized IDs absent from notes: {categorized - noted_ids}")

    output_rows: list[dict[str, Any]] = []
    for index, source in enumerate(rows, start=1):
        row = dict(source)
        if index not in noted_ids or index in VALID_WITH_DIFFICULTY:
            row["semantic_status"] = "valid"
            row["semantic_invalid_reason"] = None
            row["review_confidence"] = (
                "medium" if index in VALID_WITH_DIFFICULTY else "high"
            )
        elif index in HUMAN_REVIEW:
            row["semantic_status"] = "human_review"
            row["semantic_invalid_reason"] = None
            row["review_confidence"] = "low"
        else:
            row["semantic_status"] = "invalid"
            row["semantic_invalid_reason"] = invalid_reason(index)
            row["review_confidence"] = "high"

        row["difficulty_tags"] = (
            DIFFICULTY_TAGS.get(index, [])
            if row["semantic_status"] != "invalid"
            else []
        )
        row["reviewer_id"] = (
            "human_primary_plus_codex_visual_adjudication"
            if index in CODEX_VISUAL_ADJUDICATION
            else "human_primary_transcribed_by_codex"
        )
        row["review_note"] = (
            notes[index]
            if index in notes
            else "Human reviewer reported no semantic issue."
        )
        row["technical_status"] = None
        output_rows.append(row)

    ensure_new(args.output_jsonl)
    with args.output_jsonl.open("x", encoding="utf-8") as handle:
        for row in output_rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")

    ensure_new(args.output_csv)
    fields = list(output_rows[0])
    with args.output_csv.open("x", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for row in output_rows:
            csv_row = dict(row)
            csv_row["previous_reason_codes"] = ";".join(
                row["previous_reason_codes"]
            )
            csv_row["difficulty_tags"] = ";".join(row["difficulty_tags"])
            writer.writerow(csv_row)

    counts = {
        status: sum(row["semantic_status"] == status for row in output_rows)
        for status in ("valid", "invalid", "human_review")
    }
    print(
        json.dumps(
            {
                "rows": len(output_rows),
                "noted_candidates": len(notes),
                "semantic_status_counts": counts,
                "technical_status_filled": 0,
                "output_jsonl": args.output_jsonl.as_posix(),
                "output_csv": args.output_csv.as_posix(),
            },
            ensure_ascii=False,
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
