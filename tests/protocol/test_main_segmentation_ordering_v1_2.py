from __future__ import annotations

import importlib.util
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]


def load(name: str, relative: str):
    spec = importlib.util.spec_from_file_location(name, ROOT / relative)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


runner = load(
    "run_segmentation_v1_1",
    "scripts/attribute_binding/run_segmentation_v1_1.py",
)
validator = load(
    "validate_mask_results_v1_1",
    "scripts/data/validate_mask_results_v1_1.py",
)


class SegmentationOrderingContractTest(unittest.TestCase):
    def test_main_uses_frozen_order(self):
        field, ordered = runner.order_contract(
            [
                {"candidate_id": "b", "frozen_order": 2},
                {"candidate_id": "a", "frozen_order": 1},
            ]
        )
        self.assertEqual(field, "frozen_order")
        self.assertEqual([row["candidate_id"] for row in ordered], ["a", "b"])
        self.assertEqual(validator.input_order_field(ordered[0]), "frozen_order")

    def test_pilot_uses_selection_rank(self):
        field, ordered = runner.order_contract(
            [
                {"candidate_id": "b", "selection_rank": 2},
                {"candidate_id": "a", "selection_rank": 1},
            ]
        )
        self.assertEqual(field, "selection_rank")
        self.assertEqual([row["candidate_id"] for row in ordered], ["a", "b"])
        self.assertEqual(
            validator.input_order_field(ordered[0]), "selection_rank"
        )

    def test_mixed_contract_is_rejected(self):
        with self.assertRaises(ValueError):
            runner.order_contract(
                [
                    {"candidate_id": "a", "frozen_order": 1},
                    {"candidate_id": "b", "selection_rank": 2},
                ]
            )


if __name__ == "__main__":
    unittest.main()
