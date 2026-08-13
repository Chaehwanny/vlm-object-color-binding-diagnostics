#!/usr/bin/env python3
"""Correct two frozen semantic review notes without changing judgments."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any


NOTE_PATCH = {
    "maincand_0251": (
        "green remote control label mismatch confirmed by human visual review"
    ),
    "maincand_0290": (
        "blue patio object could not be reliably identified in the preview"
    ),
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--report", type=Path, required=True)
    return parser.parse_args()


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    with path.open(encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def main() -> None:
    args = parse_args()
    for path in (args.output, args.report):
        if path.exists():
            raise FileExistsError(f"Refusing to overwrite output: {path}")
        path.parent.mkdir(parents=True, exist_ok=True)

    rows = read_jsonl(args.input)
    before = {
        row["candidate_id"]: (
            row["semantic_status"],
            row["semantic_invalid_reason"],
            row["technical_status"],
        )
        for row in rows
    }
    patched_ids: list[str] = []
    for row in rows:
        candidate_id = row["candidate_id"]
        if candidate_id in NOTE_PATCH:
            row["review_note"] = NOTE_PATCH[candidate_id]
            row["review_note_revision"] = {
                "revision": "semantic_visual_audit_note_v1.1",
                "judgment_changed": False,
            }
            patched_ids.append(candidate_id)

    after = {
        row["candidate_id"]: (
            row["semantic_status"],
            row["semantic_invalid_reason"],
            row["technical_status"],
        )
        for row in rows
    }
    if before != after:
        raise AssertionError("Review-note patch changed a frozen judgment.")
    if set(patched_ids) != set(NOTE_PATCH):
        raise ValueError(f"Missing note-patch IDs: {set(NOTE_PATCH) - set(patched_ids)}")

    with args.output.open("x", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")
    report = {
        "input_rows": len(rows),
        "patched_candidate_ids": patched_ids,
        "judgment_changed": False,
    }
    args.report.write_text(
        json.dumps(report, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
