#!/usr/bin/env python3
"""Unit tests for the frozen Primary v1.2 response/parser contract."""

from __future__ import annotations

import sys
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "scripts/inference"))

from primary_contract_v1_2 import (  # noqa: E402
    RESPONSE_INSTRUCTION,
    evaluate_response,
    parse_option_letter,
    parse_option_letter_strict,
    render_semantic_user_content,
)


def row(alphabet: list[str] | None = None, gold: str = "A") -> dict:
    alphabet = alphabet or ["A", "B", "C", "D"]
    options = [
        {"option_id": letter, "semantic_value": f"semantic-{letter}", "display_text": f"choice {letter}"}
        for letter in alphabet
    ]
    return {
        "query_id": "test_query",
        "candidate_id": "test_candidate",
        "image_path": "unused.png",
        "image_sha256": "0" * 64,
        "state": "original",
        "analysis_role": "primary",
        "task_type": "binding_choice" if alphabet == ["A", "B"] else "object_a_color",
        "prompt_text": "Select the best answer.",
        "options": options,
        "correct_option_id": gold,
        "correct_semantic_answer": f"semantic-{gold}",
        "response_alphabet": alphabet,
        "stateless": True,
        "conversation_history": [],
    }


class ParserTests(unittest.TestCase):
    def test_all_frozen_valid_forms(self) -> None:
        forms = ["A", "A.", "(A)", "Answer: A", "Answer A", "Option A", "Option: A"]
        for value in forms:
            with self.subTest(value=value):
                self.assertEqual(parse_option_letter(value, ["A", "B"]), "A")
                self.assertEqual(parse_option_letter_strict(value, ["A", "B"]), "A")

    def test_amended_leading_option_forms(self) -> None:
        cases = {
            "A. blue": "A",
            "B. green": "B",
            "C. red": "C",
            "D. yellow": "D",
            "A: blue": "A",
            "B: green": "B",
            "A because the object is blue.": "A",
            "B because the shirt is green.": "B",
            "A. The color of the car is red.": "A",
            "B. The color of the shirt is blue.": "B",
            "A. blue\nB. green\nC. red\nD. yellow": "A",
            "b. The color of the shirt is red.": "B",
        }
        for value, expected in cases.items():
            with self.subTest(value=value):
                self.assertEqual(parse_option_letter(value, ["A", "B", "C", "D"]), expected)

    def test_binding_leading_forms_and_alphabet(self) -> None:
        self.assertEqual(parse_option_letter("A. first binding", ["A", "B"]), "A")
        self.assertEqual(parse_option_letter("B: second binding", ["A", "B"]), "B")
        self.assertIsNone(parse_option_letter("C. blue", ["A", "B"]))
        self.assertIsNone(parse_option_letter("D. red", ["A", "B"]))

    def test_surrounding_whitespace_and_lowercase(self) -> None:
        for value in ("  a  ", "\nanswer: a\t", " option b "):
            with self.subTest(value=value):
                self.assertEqual(parse_option_letter(value, ["A", "B"]), value.strip()[-1].upper())

    def test_invalid_forms(self) -> None:
        values = [
            "", "random prose", "Maybe A", "A or B", "A/B", "A, B",
            "Either A or B", "Maybe A or B", "A but maybe B",
            "A or perhaps C", "The answer might be A", "I think B",
            "A B", "Answer:", "[A]", "(A].",
        ]
        for value in values:
            with self.subTest(value=value):
                self.assertIsNone(parse_option_letter(value, ["A", "B", "C", "D"]))

    def test_alphabet_is_authoritative(self) -> None:
        self.assertIsNone(parse_option_letter("C", ["A", "B"]))
        self.assertIsNone(parse_option_letter("D.", ["A", "B"]))
        for letter in ("A", "B", "C", "D"):
            self.assertEqual(parse_option_letter(letter, ["A", "B", "C", "D"]), letter)

    def test_gold_mapping_and_correctness(self) -> None:
        query = row(gold="C")
        correct = evaluate_response(query, "Option: C", None)
        self.assertEqual(correct["parsed_semantic_answer"], "semantic-C")
        self.assertTrue(correct["is_valid"])
        self.assertTrue(correct["is_correct"])
        wrong = evaluate_response(query, "B", None)
        self.assertTrue(wrong["is_valid"])
        self.assertFalse(wrong["is_correct"])

    def test_leading_choice_is_gold_independent(self) -> None:
        query = row(gold="B")
        result = evaluate_response(query, "A. green", None)
        self.assertEqual(result["parsed_option_id"], "A")
        self.assertEqual(result["parsed_semantic_answer"], "semantic-A")
        self.assertTrue(result["is_valid"])
        self.assertFalse(result["is_correct"])

    def test_invalid_and_inference_error_are_distinct(self) -> None:
        query = row(["A", "B"])
        invalid = evaluate_response(query, "Maybe A", None)
        self.assertFalse(invalid["is_valid"])
        self.assertFalse(invalid["is_correct"])
        self.assertIsNone(invalid["inference_error"])
        failed = evaluate_response(query, None, "RuntimeError: test")
        self.assertFalse(failed["is_valid"])
        self.assertFalse(failed["is_correct"])
        self.assertEqual(failed["inference_error"], "RuntimeError: test")

    def test_renderer_preserves_question_and_option_order(self) -> None:
        query = row(["A", "B"])
        rendered = render_semantic_user_content(query, RESPONSE_INSTRUCTION)
        self.assertEqual(
            rendered,
            "Select the best answer.\nA. choice A\nB. choice B\n"
            "Answer with only the option letter.",
        )

    def test_renderer_does_not_duplicate_existing_instruction(self) -> None:
        query = row(["A", "B"])
        query["prompt_text"] += " Answer with only the option letter."
        rendered = render_semantic_user_content(query, RESPONSE_INSTRUCTION)
        self.assertEqual(rendered.lower().count("answer with only the option letter"), 1)


if __name__ == "__main__":
    unittest.main()
