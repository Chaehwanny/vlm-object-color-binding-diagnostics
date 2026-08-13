#!/usr/bin/env python3
"""Unit tests for the v1.1 identity-field separation rules."""

from __future__ import annotations

import importlib.util
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
MODULE_PATH = ROOT / "scripts/data/validate_identity_fields_v1_1.py"
SPEC = importlib.util.spec_from_file_location("identity_validator", MODULE_PATH)
assert SPEC and SPEC.loader
VALIDATOR = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(VALIDATOR)


def pre_edit_row() -> dict:
    return {
        "candidate_id": "synthetic_001",
        "technical_status": "directly_editable",
        "technical_criterion_scores": {
            "material_suitability": 2,
            "expected_mask_separability": 2,
            "sufficient_visible_area": 2,
            "swap_feasibility": 2,
            "expected_identity_preservability": 2,
        },
        "mask_qc_status": "not_tested",
        "edit_qc_status": "not_tested",
        "object_identity_preservation_status": "not_tested",
        "object_identity_preservation_note": None,
        "object_identity_preservation_reviewer": None,
        "object_identity_preservation_timestamp": None,
    }


class PreEditValidationTests(unittest.TestCase):
    def test_valid_pre_edit_row(self) -> None:
        self.assertEqual(VALIDATOR.validate_pre_edit(pre_edit_row()), [])

    def test_legacy_identity_name_is_rejected(self) -> None:
        row = pre_edit_row()
        row["technical_criterion_scores"]["identity_preserved"] = 2
        errors = VALIDATOR.validate_pre_edit(row)
        self.assertTrue(any("legacy" in error for error in errors))

    def test_pre_edit_actual_identity_result_is_rejected(self) -> None:
        row = pre_edit_row()
        row["object_identity_preservation_status"] = "pass"
        errors = VALIDATOR.validate_pre_edit(row)
        self.assertTrue(any("identity result" in error for error in errors))

    def test_directly_editable_requires_all_twos(self) -> None:
        row = pre_edit_row()
        row["technical_criterion_scores"]["expected_mask_separability"] = 1
        errors = VALIDATOR.validate_pre_edit(row)
        self.assertTrue(any("five scores of 2" in error for error in errors))

    def test_failed_after_test_requires_record(self) -> None:
        row = pre_edit_row()
        row["technical_status"] = "failed_after_test"
        errors = VALIDATOR.validate_pre_edit(row)
        self.assertTrue(any("technical_test_record" in error for error in errors))


class PostEditValidationTests(unittest.TestCase):
    def make_tested_row(self) -> dict:
        row = pre_edit_row()
        row.update(
            {
                "edited_image_path": "synthetic/edited.jpg",
                "mask_a_path": "synthetic/mask_a.png",
                "mask_qc_status": "pass",
                "edit_qc_status": "pass",
                "object_identity_preservation_status": "pass",
                "object_identity_preservation_reviewer": "reviewer_1",
                "object_identity_preservation_timestamp": (
                    "2026-07-31T00:00:00+09:00"
                ),
            }
        )
        return row

    def test_edited_image_requires_tested_identity(self) -> None:
        row = self.make_tested_row()
        row["object_identity_preservation_status"] = "not_tested"
        errors = VALIDATOR.validate_post_edit(row, require_final_pass=False)
        self.assertTrue(any("tested identity" in error for error in errors))

    def test_identity_fail_requires_note(self) -> None:
        row = self.make_tested_row()
        row["object_identity_preservation_status"] = "fail"
        errors = VALIDATOR.validate_post_edit(row, require_final_pass=False)
        self.assertTrue(any("requires a note" in error for error in errors))

    def test_final_matched_requires_all_qc_pass(self) -> None:
        row = self.make_tested_row()
        row["edit_qc_status"] = "human_review"
        errors = VALIDATOR.validate_post_edit(row, require_final_pass=True)
        self.assertTrue(any("edit_qc_status=pass" in error for error in errors))

    def test_valid_post_edit_row(self) -> None:
        self.assertEqual(
            VALIDATOR.validate_post_edit(
                self.make_tested_row(), require_final_pass=False
            ),
            [],
        )


if __name__ == "__main__":
    unittest.main()
