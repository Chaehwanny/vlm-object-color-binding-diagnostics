#!/usr/bin/env python3
"""Freeze the model-blind validity audit of mini color-binding masks."""

from __future__ import annotations

import argparse
import csv
import json
from collections import Counter
from pathlib import Path
from typing import Any


DECISIONS = {
    "mini_review_002": ("reject", "physical_entanglement_shared_owner"),
    "mini_review_003": ("reject", "physical_entanglement_shared_owner"),
    "mini_review_005": ("reject", "competing_garment_instances"),
    "mini_review_007": ("keep", "clear_distinct_objects_plausible_swap"),
    "mini_review_008": ("reject", "semantic_overlap_shirt_jersey"),
    "mini_review_010": ("reject", "physical_entanglement_shared_owner"),
    "mini_review_017": ("reject", "bidirectional_swap_implausible"),
    "mini_review_018": ("reject", "competing_garment_instances"),
    "mini_review_020": ("reject", "competing_same_label_bowl"),
    "mini_review_022": ("reject", "prominent_text_logo_artifact"),
    "mini_review_024": ("keep", "clear_distinct_objects_plausible_swap"),
    "mini_review_025": ("reject", "stylized_depiction_and_unclear_identity"),
    "mini_review_026": ("reject", "semantic_overlap_and_duplicate_instances"),
    "mini_review_027": ("reject", "bidirectional_swap_implausible"),
    "mini_review_028": ("reject", "generic_label_and_implausible_swap"),
    "mini_review_029": ("keep", "clear_distinct_objects_plausible_swap"),
    "mini_review_032": ("reject", "uncommon_edited_binding_prior"),
    "mini_review_035": ("reject", "duplicate_instances_and_implausible_swap"),
    "mini_review_036": ("reject", "severe_occlusion_and_small_target"),
    "mini_review_038": ("keep", "clear_distinct_objects_plausible_swap"),
    "mini_review_039": ("reject", "competing_garment_instances"),
    "mini_review_040": ("reject", "uncommon_edited_binding_prior"),
}


def parse_args() -> argparse.Namespace:
    root = Path(__file__).resolve().parents[2]
    parser = argparse.ArgumentParser(description="Freeze independent mask audit.")
    parser.add_argument(
        "--mask-dir",
        type=Path,
        default=root / "processed/attribute_binding/segmentation/mini_segmentation_masks_v0.1",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=root / "processed/attribute_binding/audits/mini_independent_audit_v0.1",
    )
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    with path.open("r", encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def main() -> None:
    args = parse_args()
    if args.output_dir.exists() and any(args.output_dir.iterdir()) and not args.overwrite:
        raise FileExistsError(f"Output directory is not empty: {args.output_dir}")

    rows = read_jsonl(args.mask_dir / "mask_manifest.jsonl")
    row_ids = {row["segmentation_id"] for row in rows}
    if row_ids != set(DECISIONS):
        raise ValueError(
            "Audit IDs do not match manifest: "
            f"missing={sorted(row_ids - set(DECISIONS))}, "
            f"extra={sorted(set(DECISIONS) - row_ids)}"
        )

    args.output_dir.mkdir(parents=True, exist_ok=True)
    accepted: list[dict[str, Any]] = []
    reason_counts: Counter[str] = Counter()
    fields = [
        "segmentation_id",
        "image_id",
        "color_pair",
        "object_a",
        "object_b",
        "audit_decision",
        "audit_reason",
        "overlay_path",
    ]
    with (args.output_dir / "audit_decisions.csv").open(
        "w", encoding="utf-8", newline=""
    ) as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for row in rows:
            decision, reason = DECISIONS[row["segmentation_id"]]
            reason_counts[reason] += 1
            writer.writerow(
                {
                    "segmentation_id": row["segmentation_id"],
                    "image_id": row["image_id"],
                    "color_pair": row["color_pair"],
                    "object_a": f"{row['object_a']['color']} {row['object_a']['label']}",
                    "object_b": f"{row['object_b']['color']} {row['object_b']['label']}",
                    "audit_decision": decision,
                    "audit_reason": reason,
                    "overlay_path": str(args.mask_dir / row["overlay_path"]),
                }
            )
            if decision == "keep":
                accepted.append(
                    {
                        **row,
                        "independent_audit_status": "keep_for_edit_attempt",
                        "independent_audit_reason": reason,
                        "development_data_only": True,
                    }
                )

    with (args.output_dir / "accepted_candidates.jsonl").open(
        "w", encoding="utf-8"
    ) as handle:
        for row in accepted:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")

    summary = {
        "schema_version": "0.1",
        "audit_stage": "model_blind_independent_visual_validity_audit",
        "reviewed": len(rows),
        "accepted_for_edit_attempt": len(accepted),
        "rejected": len(rows) - len(accepted),
        "accepted_ids": [row["segmentation_id"] for row in accepted],
        "rejection_reason_counts": {
            reason: count
            for reason, count in sorted(reason_counts.items())
            if reason != "clear_distinct_objects_plausible_swap"
        },
        "model_outputs_used_for_decision": False,
        "note": (
            "Acceptance freezes pre-inference validity only. Each retained sample "
            "must still pass color editing and post-edit human review."
        ),
    }
    (args.output_dir / "summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
