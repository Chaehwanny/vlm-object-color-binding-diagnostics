#!/usr/bin/env python3
"""Write and validate model-blind AI-assisted candidate prescreen outputs."""

from __future__ import annotations

import argparse
import csv
import json
from collections import Counter, defaultdict
from pathlib import Path


CRITERIA = [
    "object_a_clear",
    "object_b_clear",
    "independent_pair",
    "label_a_correct",
    "label_b_correct",
    "unique_reference",
    "color_a_clear",
    "color_b_clear",
    "canonical_color_pair",
    "material_suitable",
    "separable_masks",
    "sufficient_visible_area",
    "swap_feasible",
    "identity_preserved",
    "scene_plausibility",
]

REASON_TO_CRITERIA = {
    "unclear_object_a": ["object_a_clear"],
    "unclear_object_b": ["object_b_clear"],
    "same_owner_or_entangled": ["independent_pair", "separable_masks"],
    "structure_or_part": ["independent_pair"],
    "ambiguous_label": ["label_a_correct", "label_b_correct"],
    "nonunique_reference": ["unique_reference"],
    "color_ambiguous_a": ["color_a_clear", "canonical_color_pair"],
    "color_ambiguous_b": ["color_b_clear", "canonical_color_pair"],
    "transparent_or_reflective": ["material_suitable", "swap_feasible"],
    "text_logo_pattern": ["material_suitable"],
    "material_surface_simple": ["material_suitable"],
    "mask_difficult": ["separable_masks"],
    "separable_masks": ["separable_masks"],
    "sufficient_visible_area": ["sufficient_visible_area", "swap_feasible"],
    "food_or_natural": ["swap_feasible", "scene_plausibility"],
    "identity_risk": ["identity_preserved"],
    "swap_implausible": ["swap_feasible", "scene_plausibility"],
    "canonical_color_pair": ["canonical_color_pair"],
    "scene_plausibility": ["scene_plausibility"],
    "other": ["swap_feasible"],
}

BASE_FIELDS = [
    "schema_version",
    "candidate_id",
    "source_dataset",
    "source_image_id",
    "tier",
    "color_pair",
    "object_a_label",
    "object_a_color",
    "object_a_category",
    "object_a_bbox_area",
    "object_b_label",
    "object_b_color",
    "object_b_category",
    "object_b_bbox_area",
    "bbox_iou",
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--candidate-pool", type=Path, required=True)
    parser.add_argument("--candidate-root", type=Path, required=True)
    parser.add_argument("--image-root", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument(
        "--decisions-json",
        type=Path,
        required=True,
        help="JSON object keyed by candidate_id with p/r/e fields.",
    )
    return parser.parse_args()


def read_jsonl(path: Path) -> list[dict]:
    with path.open() as handle:
        return [json.loads(line) for line in handle if line.strip()]


def score_row(decision: dict) -> dict[str, int]:
    scores = {criterion: 2 for criterion in CRITERIA}
    label = decision["p"]
    failed_value = 0 if label == "reject" else 1
    for reason in filter(None, decision.get("r", "").split(";")):
        for criterion in REASON_TO_CRITERIA.get(reason, []):
            scores[criterion] = min(scores[criterion], failed_value)
    if label == "reject" and all(scores[name] > 0 for name in CRITERIA):
        raise ValueError(f"Reject has no failed criterion: {decision}")
    return scores


def write_csv(path: Path, rows: list[dict], fields: list[str]) -> None:
    if path.exists():
        raise FileExistsError(f"Refusing to overwrite existing output: {path}")
    with path.open("x", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def markdown_table(headers: list[str], rows: list[list[object]]) -> str:
    lines = [
        "| " + " | ".join(headers) + " |",
        "| " + " | ".join(["---"] * len(headers)) + " |",
    ]
    lines.extend("| " + " | ".join(map(str, row)) + " |" for row in rows)
    return "\n".join(lines)


def main() -> None:
    args = parse_args()
    pool = read_jsonl(args.candidate_pool)
    decisions = json.loads(args.decisions_json.read_text())
    pool_ids = {row["candidate_id"] for row in pool}
    if len(pool) != 293 or len(pool_ids) != 293:
        raise ValueError("Candidate pool must contain 293 unique candidate IDs.")
    if set(decisions) != pool_ids:
        missing = sorted(pool_ids - set(decisions))
        extra = sorted(set(decisions) - pool_ids)
        raise ValueError(f"Decision coverage mismatch. missing={missing}, extra={extra}")

    enriched = []
    missing_paths = []
    for source in pool:
        decision = decisions[source["candidate_id"]]
        label = decision["p"]
        if label not in {"keep", "borderline", "reject"}:
            raise ValueError(f"Unfinalized label for {source['candidate_id']}: {label}")
        scores = score_row(decision)
        source_path = args.image_root / source["source_image_path"]
        preview_path = args.candidate_root / source["preview_path"]
        if not source_path.is_file():
            missing_paths.append(str(source_path))
        if not preview_path.is_file():
            missing_paths.append(str(preview_path))
        row = {field: source.get(field, "") for field in BASE_FIELDS}
        row.update(scores)
        row.update(
            {
                "hard_reject": str(label == "reject").lower(),
                "hard_reject_reason": decision.get("r", "") if label == "reject" else "",
                "final_label": label,
                "confidence": "high" if label in {"keep", "reject"} else "medium",
                "evidence_note": decision["e"],
                "recommended_next_step": {
                    "keep": "send_to_primary_segmentation_candidate_pool",
                    "borderline": "human_recheck_before_any_segmentation",
                    "reject": "exclude_before_segmentation",
                }[label],
                "quality_score": sum(scores.values()),
                "source_image_path": source["source_image_path"],
                "preview_path": source["preview_path"],
            }
        )
        enriched.append(row)

    enriched.sort(
        key=lambda row: (
            {"keep": 0, "borderline": 1, "reject": 2}[row["final_label"]],
            -row["quality_score"],
            row["candidate_id"],
        )
    )
    keep_rows = [row for row in enriched if row["final_label"] == "keep"]
    shortlist = keep_rows[:120]
    reserve = [
        {
            **row,
            "reserve_status": "keep_overflow"
            if row["final_label"] == "keep"
            else "borderline_human_recheck",
        }
        for row in enriched
        if row["final_label"] == "borderline" or (
            row["final_label"] == "keep"
            and row["candidate_id"] not in {item["candidate_id"] for item in shortlist}
        )
    ]
    rejected = [row for row in enriched if row["final_label"] == "reject"]

    output_fields = (
        BASE_FIELDS
        + CRITERIA
        + [
            "hard_reject",
            "hard_reject_reason",
            "final_label",
            "confidence",
            "evidence_note",
            "recommended_next_step",
            "quality_score",
            "source_image_path",
            "preview_path",
        ]
    )
    args.output_dir.mkdir(parents=True, exist_ok=True)
    all_path = args.output_dir / "agent_prescreen_v1.csv"
    shortlist_path = args.output_dir / "segmentation_shortlist_v1.csv"
    reserve_path = args.output_dir / "segmentation_reserve_v1.csv"
    rejected_path = args.output_dir / "rejected_by_agent_v1.csv"
    report_path = args.output_dir / "agent_prescreen_report_v1.md"
    for path in [all_path, shortlist_path, reserve_path, rejected_path, report_path]:
        if path.exists():
            raise FileExistsError(f"Refusing to overwrite existing output: {path}")

    write_csv(all_path, enriched, output_fields)
    write_csv(shortlist_path, shortlist, output_fields)
    write_csv(reserve_path, reserve, output_fields + ["reserve_status"])
    write_csv(rejected_path, rejected, output_fields)

    label_counts = Counter(row["final_label"] for row in enriched)
    reason_counts = Counter()
    for row in rejected:
        reason_counts.update(filter(None, row["hard_reject_reason"].split(";")))
    by_color = defaultdict(Counter)
    by_tier = defaultdict(Counter)
    by_category = defaultdict(Counter)
    for row in enriched:
        by_color[row["color_pair"]][row["final_label"]] += 1
        by_tier[row["tier"]][row["final_label"]] += 1
        for category in {row["object_a_category"], row["object_b_category"]}:
            by_category[category][row["final_label"]] += 1

    report = [
        "# AI-assisted prescreen report v1",
        "",
        "## Scope and safeguards",
        "",
        "- Reviewed all 293 independent GQA candidates without consulting model responses, logits, accuracy, or failure-case files.",
        "- Contact sheets were used only for the first rejection pass.",
        "- Every preliminary keep or borderline candidate was rechecked using both the individual preview and the original image.",
        "- Existing human annotation sheets were not modified.",
        "- No segmentation, color editing, VLM inference, or final 80-sample selection was performed.",
        "",
        "## Outcome",
        "",
        markdown_table(
            ["Label", "Count"],
            [[label, label_counts[label]] for label in ["keep", "borderline", "reject"]],
        ),
        "",
        f"- Primary segmentation shortlist: **{len(shortlist)}**",
        f"- Reserve: **{len(reserve)}** (borderline candidates remain human-recheck only)",
        f"- Target shortfall relative to 120: **{max(0, 120 - len(shortlist))}**",
        "- The quality threshold was not relaxed, and no borderline candidate was automatically promoted.",
        "",
        "## Hard-reject reasons",
        "",
        markdown_table(
            ["Reason", "Count"],
            [[reason, count] for reason, count in sorted(reason_counts.items(), key=lambda x: (-x[1], x[0]))],
        ),
        "",
        "## Color-pair distribution",
        "",
        markdown_table(
            ["Color pair", "Keep", "Borderline", "Reject", "Total"],
            [
                [
                    key,
                    counts["keep"],
                    counts["borderline"],
                    counts["reject"],
                    sum(counts.values()),
                ]
                for key, counts in sorted(by_color.items())
            ],
        ),
        "",
        "## Tier distribution",
        "",
        markdown_table(
            ["Tier", "Keep", "Borderline", "Reject", "Total"],
            [
                [
                    key,
                    counts["keep"],
                    counts["borderline"],
                    counts["reject"],
                    sum(counts.values()),
                ]
                for key, counts in sorted(by_tier.items())
            ],
        ),
        "",
        "## Object-category distribution",
        "",
        "A candidate is counted once per category represented by either object.",
        "",
        markdown_table(
            ["Category", "Keep", "Borderline", "Reject", "Total"],
            [
                [
                    key,
                    counts["keep"],
                    counts["borderline"],
                    counts["reject"],
                    sum(counts.values()),
                ]
                for key, counts in sorted(by_category.items())
            ],
        ),
        "",
        "## Borderline candidates requiring human review",
        "",
        markdown_table(
            ["Candidate", "Quality", "Reason", "Evidence"],
            [
                [
                    row["candidate_id"],
                    row["quality_score"],
                    decisions[row["candidate_id"]]["r"],
                    row["evidence_note"],
                ]
                for row in reserve
                if row["final_label"] == "borderline"
            ],
        ),
        "",
        "## Representative difficult cases",
        "",
        "- `maincand_0026`: the bicycle and motorcycle are semantically independent but have overlapping, thin mechanical boundaries.",
        "- `maincand_0077`: the pillow and sheet have clear colors, yet their broad overlapping bedding surfaces may not yield clean independent masks.",
        "- `maincand_0249`: the garments belong to different people, but the blue jeans are small and motion-blurred.",
        "- `maincand_0284`: both vehicle colors are clear, but the objects are small and distant at the upper frame edge.",
        "",
        "## File and validation status",
        "",
        f"- Missing or unreadable image paths: **{len(missing_paths)}**",
    ]
    if missing_paths:
        report.extend([f"  - `{path}`" for path in missing_paths])
    else:
        report.append("- Missing-path list: none.")
    report.extend(
        [
            f"- Full prescreen rows: **{len(enriched)}**",
            f"- Shortlist rows: **{len(shortlist)}**",
            f"- Reserve rows: **{len(reserve)}**",
            f"- Rejected rows: **{len(rejected)}**",
            f"- Duplicate candidate IDs in full output: **{len(enriched) - len({row['candidate_id'] for row in enriched})}**",
            f"- Duplicate source images in full output: **{len(enriched) - len({row['source_image_id'] for row in enriched})}**",
            "",
            "The prescreen is a model-blind, AI-assisted development artifact. Human confirmation remains required before segmentation, especially for every borderline candidate.",
        ]
    )
    report_path.write_text("\n".join(report) + "\n")

    # Final partition and score validations.
    if len(enriched) != 293:
        raise AssertionError("Full output row count mismatch.")
    if missing_paths:
        raise AssertionError(f"Missing image paths: {missing_paths}")
    if any(row["quality_score"] != sum(row[name] for name in CRITERIA) for row in enriched):
        raise AssertionError("Quality score mismatch.")
    if any(row["final_label"] != "keep" for row in shortlist):
        raise AssertionError("Primary shortlist contains a non-keep row.")
    partition_ids = (
        {row["candidate_id"] for row in shortlist}
        | {row["candidate_id"] for row in reserve}
        | {row["candidate_id"] for row in rejected}
    )
    if partition_ids != pool_ids:
        raise AssertionError("Output partition does not cover the candidate pool.")
    print(
        json.dumps(
            {
                "all": len(enriched),
                "shortlist": len(shortlist),
                "reserve": len(reserve),
                "rejected": len(rejected),
                "missing_paths": len(missing_paths),
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
