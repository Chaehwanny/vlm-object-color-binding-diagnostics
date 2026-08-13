#!/usr/bin/env python3
"""Freeze post-edit decisions for PACO color-swap candidates."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path


DECISIONS = {
    "paco_color_005": ("keep", "all_edit_quality_gates_pass"),
    "paco_color_007": ("keep", "all_edit_quality_gates_pass"),
    "paco_color_009": ("reject_sample", "prominent_print_logo_artifact"),
}


def parse_args() -> argparse.Namespace:
    root = Path(__file__).resolve().parents[2]
    parser = argparse.ArgumentParser(description="Freeze PACO post-edit audit.")
    parser.add_argument(
        "--edit-dir",
        type=Path,
        default=root / "processed/attribute_binding/edits/paco_color_swap_edits_v0.1",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=root / "processed/attribute_binding/audits/paco_edit_audit_v0.1",
    )
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.output_dir.exists() and any(args.output_dir.iterdir()) and not args.overwrite:
        raise FileExistsError(f"Output directory is not empty: {args.output_dir}")
    rows = [
        json.loads(line)
        for line in (args.edit_dir / "edit_manifest.jsonl").read_text(
            encoding="utf-8"
        ).splitlines()
        if line.strip()
    ]
    if {row["segmentation_id"] for row in rows} != set(DECISIONS):
        raise ValueError("Decision IDs do not match PACO edit manifest")

    args.output_dir.mkdir(parents=True, exist_ok=True)
    accepted = []
    fields = [
        "segmentation_id",
        "object_a_before",
        "object_b_before",
        "post_edit_decision",
        "post_edit_reason",
        "comparison_path",
    ]
    with (args.output_dir / "edit_audit_decisions.csv").open(
        "w", encoding="utf-8", newline=""
    ) as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for row in rows:
            decision, reason = DECISIONS[row["segmentation_id"]]
            writer.writerow(
                {
                    "segmentation_id": row["segmentation_id"],
                    "object_a_before": (
                        f"{row['object_a']['color']} {row['object_a']['label']}"
                    ),
                    "object_b_before": (
                        f"{row['object_b']['color']} {row['object_b']['label']}"
                    ),
                    "post_edit_decision": decision,
                    "post_edit_reason": reason,
                    "comparison_path": str(args.edit_dir / row["comparison_path"]),
                }
            )
            if decision == "keep":
                accepted.append(
                    {
                        **row,
                        "post_edit_decision": decision,
                        "post_edit_reason": reason,
                        "second_human_review_status": "pending",
                    }
                )

    with (args.output_dir / "accepted_edits.jsonl").open(
        "w", encoding="utf-8"
    ) as handle:
        for row in accepted:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")
    summary = {
        "schema_version": "0.1",
        "reviewed_edits": len(rows),
        "accepted_current_edits": len(accepted),
        "rejected_samples": len(rows) - len(accepted),
        "accepted_ids": [row["segmentation_id"] for row in accepted],
        "model_outputs_used_for_decision": False,
        "second_human_review_required": True,
    }
    (args.output_dir / "summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
