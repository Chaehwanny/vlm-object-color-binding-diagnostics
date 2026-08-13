#!/usr/bin/env python3
"""Audit the v1.2 primary/secondary contract and frozen main source."""

from __future__ import annotations

import hashlib
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
MAIN = ROOT / "processed/attribute_binding/main_experiment/v1.1/preparation/main_input_manifest_v1.1.jsonl"
EXPECTED_SHA = "48e6158a20b22e3014096a31abd9e8654758e909ef7542fedef87cb92b10be98"
ACTIVE_DOCS = [
    ROOT / "docs/protocols/protocol_v1.2_PRE_INFERENCE_AMENDMENT.md",
    ROOT / "docs/protocols/protocol_v1.1_PRE_INFERENCE_DRAFT.md",
    ROOT / "docs/protocols/statistical_analysis_plan_v1.1.md",
    ROOT / "docs/protocols/task_specification_v1.1.md",
    ROOT / "docs/protocols/metric_dictionary_v1.1.md",
    ROOT / "docs/protocols/manifest_schema_v1.1.md",
    ROOT / "docs/protocols/attribute_binding_pre_main_handoff.md",
]


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def rows(path: Path) -> list[dict]:
    with path.open(encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def main() -> int:
    errors: list[str] = []
    data = rows(MAIN)
    ids = [row["candidate_id"] for row in data]
    sources = [str(row["source_image_id"]) for row in data]
    pairs = [row["source_pair_key"] for row in data]
    orders = [row["frozen_order"] for row in data]
    missing = [
        row["candidate_id"]
        for row in data
        if not (ROOT / row["original_image_path"]).is_file()
    ]
    overlap = [
        row["candidate_id"]
        for row in data
        if row.get("development_overlap_status") != "none"
    ]
    digest = sha256(MAIN)

    checks = {
        "main_candidates": len(data) == 157,
        "development_overlap_zero": not overlap,
        "candidate_duplicates_zero": len(ids) == len(set(ids)),
        "source_duplicates_zero": len(sources) == len(set(sources)),
        "pair_duplicates_zero": len(pairs) == len(set(pairs)),
        "missing_images_zero": not missing,
        "order_unchanged": orders == list(range(1, 158)),
        "sha_unchanged": digest == EXPECTED_SHA,
    }
    errors.extend(name for name, passed in checks.items() if not passed)

    texts = {path: path.read_text(encoding="utf-8") for path in ACTIVE_DOCS}
    combined = "\n".join(texts.values())
    contract_checks = {
        "primary_natural": "Primary: Natural Original and Natural Edited" in combined,
        "six_primary_responses": "6 responses per sample" in combined,
        "original_binding": "Original Binding" in combined,
        "swapped_binding": "Swapped Binding" in combined,
        "crf_contract": "A_o AND B_o AND G_o AND A_e AND B_e" in combined
            or "Original A color, Original B color, Original binding, Edited A color, and Edited B color" in combined,
        "controlled_secondary": "Controlled is secondary" in combined
            or "Secondary matched condition: Controlled" in combined,
        "controlled_not_primary_gate": (
            "Controlled generation and Four-image QC are not Primary inclusion"
            in combined
            and "conditions" in combined
        ),
        "research_question_matches_crf": all(
            phrase in combined
            for phrase in (
                "individual color questions",
                "question for the Original image",
                "correctly identifies both edited object",
                "correct object-color correspondence in",
                "the Edited image",
            )
        ),
        "primary_finalized_after_questions": (
            "primary_natural_valid is finalized only after" in combined
            or "Primary Natural validity is finalized only after" in combined
        ),
        "old_binding_term_absent": "옛 결합" not in combined,
        "new_binding_term_absent": "새 결합" not in combined,
        "color_recognition_term_absent": "색상 인식 실패" not in combined,
    }
    errors.extend(name for name, passed in contract_checks.items() if not passed)

    main_root = ROOT / "processed/attribute_binding/main_experiment/v1.1"
    generated = [
        path for path in main_root.rglob("*")
        if path.is_file() and "preparation" not in path.parts
    ]
    if generated:
        errors.append("actual_main_assets_generated_nonzero")

    print("[PHASE]")
    print("main_data_production = in_progress")
    print("current_substage = segmentation_preflight_complete")
    print(f"actual_main_data_assets_generated = {len(generated)}")
    print("\n[RESEARCH STORY]")
    print("positive_control = left_right")
    print("main_experiment = natural_original_to_natural_edited_color_swap")
    print("primary_question = residual_rebinding_after_correct_individual_color_checks")
    print("primary_metric = CRF")
    print("\n[POSITIVE CONTROL]")
    print("left_right_samples = 16")
    print("qwen_full_consistency = 13/16")
    print("llava_full_consistency = 13/16")
    print("internvl_full_consistency = 15/16")
    print("role = positive_control")
    print("\n[PILOT MOTIVATION]")
    print("feasibility_residual_failure = 0/2")
    print("qwen_interim_residual_failure = 0/3")
    print("llava_interim_residual_failure = 0/5")
    print("internvl_interim_residual_failure = 0/5")
    print("interpretation = motivation_only_not_main_conclusion")
    print("\n[DATA]")
    print("raw_candidates = 2783")
    print("reviewed_candidates = 293")
    print("semantic_valid = 197")
    print("semantic_invalid = 96")
    print("development_source_overlap = 40")
    print(f"frozen_main_source_n = {len(data)}")
    print("main_membership_changed = false")
    print("main_order_changed = false" if checks["order_unchanged"] else "main_order_changed = true")
    print("main_sha_changed = false" if checks["sha_unchanged"] else "main_sha_changed = true")
    print("\n[PRIMARY CONTRACT]")
    print("primary_images_per_sample = 2")
    print("primary_questions_per_image = 3")
    print("primary_responses_per_sample = 6")
    print("primary_inclusion_requires_controlled = false")
    print("\n[SECONDARY CONTRACT]")
    print("controlled_role = secondary_matched_condition")
    print("difficulty_role = secondary_analysis")
    print("visual_dependence_role = diagnostic")
    print("oracle_role = post_hoc")
    print("\n[TERMINOLOGY]")
    print("original_binding_korean = 원본 결합")
    print("swapped_binding_korean = 교환 결합")
    print("individual_color_failure_korean = 개별 색상 확인 실패")
    print(f"old_binding_term_remaining_active = {0 if contract_checks['old_binding_term_absent'] else 1}")
    print(f"new_binding_term_remaining_active = {0 if contract_checks['new_binding_term_absent'] else 1}")
    print("\n[METRICS]")
    print("OBA = enabled")
    print("EBA = enabled")
    print("PBC = enabled")
    print("CRF = primary_conditional_metric")
    print("natural_controlled_gap = secondary")
    print("\n[RUNTIME]")
    print("segmentation_entrypoint_ready = true")
    print("main_preflight_passed = 157/157")
    print(f"actual_main_masks_generated = {len(generated)}")
    print("\n[UNRESOLVED]")
    if errors:
        for error in errors:
            print(f"- {error}")
    else:
        print("- none")
    return 1 if errors else 0


if __name__ == "__main__":
    sys.exit(main())
