#!/usr/bin/env python3
"""Apply the audited v1.1 identity-field terminology patch to protocol docs."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path


def digest(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--project-root",
        type=Path,
        default=Path(__file__).resolve().parents[2],
    )
    parser.add_argument("--report", type=Path, required=True)
    return parser.parse_args()


def replacements() -> dict[str, list[tuple[str, str]]]:
    return {
        "docs/protocols/protocol_v1.1_PRE_INFERENCE_DRAFT.md": [
            (
                """An image-only impression must not produce `failed_after_test`, except for an
objective file or geometry failure. Small size, partial occlusion, thin
structure, complex boundary, overlap, crowded background, same owner,
part-whole configuration, pattern, text/logo, low contrast, distance, and
atypical color are difficulty metadata rather than automatic semantic failure.
""",
                """Pre-edit technical triage uses five expected-risk scores:

- `material_suitability`
- `expected_mask_separability`
- `sufficient_visible_area`
- `swap_feasibility`
- `expected_identity_preservability`

Scores are `0` (high expected risk), `1` (uncertain), or `2` (low expected
risk). Score `2` means expected favorable, not verified success.
`directly_editable` requires all five scores to equal `2` and means only that a
routine segmentation/editing test may be prioritized. An image-only impression
must never produce `failed_after_test`; that status requires a recorded test
under a frozen failure rule.

Small size, partial occlusion, thin structure, complex boundary, overlap,
crowded background, same owner, part-whole configuration, pattern, text/logo,
low contrast, distance, and atypical color are difficulty metadata rather than
automatic semantic failure.
""",
            ),
            (
                """- object identity preservation;
- prompt/answer validity.
""",
                """- `object_identity_preservation_status`;
- prompt/answer validity.

Actual post-edit outcomes use `mask_qc_status`, `edit_qc_status`, and
`object_identity_preservation_status`, with values `pass`, `fail`,
`human_review`, or `not_tested`. Pre-edit manifests use `not_tested` or `null`.
An edited image cannot retain `not_tested`, and final matched QC cannot pass
unless all applicable mask, edit, and identity checks pass.
""",
            ),
            (
                "- mask identity and edit identity preservation;",
                "- actual mask QC, edit QC, and post-edit object identity preservation;",
            ),
        ],
        "docs/protocols/semantic_reclassification_rubric_v1.1_PRE_REVIEW.md": [
            (
                """- `segmentation_test_required`: editability cannot be determined reliably
  without an actual mask test.

`failed_after_test` is prohibited during image-only initial review. It may be
""",
                """- `segmentation_test_required`: editability cannot be determined reliably
  without an actual mask test.

Pre-edit triage records `material_suitability`,
`expected_mask_separability`, `sufficient_visible_area`,
`swap_feasibility`, and `expected_identity_preservability`. Scores use
`0` (high expected risk), `1` (uncertain), and `2` (low expected risk).
Score `2` is expected favorable, not verified success. Actual mask, edit, and
identity outcomes remain `not_tested` until assets are created and reviewed.

`failed_after_test` is prohibited during image-only initial review. It may be
""",
            ),
        ],
        "docs/protocols/manifest_schema_v1.1.md": [
            (
                """| `technical_reason` | string/null | Required for `failed_after_test` |
| `difficulty` | object | Boolean metadata listed below |
""",
                """| `technical_reason` | string/null | Required for `failed_after_test` |
| `technical_criterion_scores.material_suitability` | integer | Pre-edit expected-risk score: 0, 1, or 2 |
| `technical_criterion_scores.expected_mask_separability` | integer | Pre-edit expected-risk score: 0, 1, or 2 |
| `technical_criterion_scores.sufficient_visible_area` | integer | Pre-edit expected-risk score: 0, 1, or 2 |
| `technical_criterion_scores.swap_feasibility` | integer | Pre-edit expected-risk score: 0, 1, or 2 |
| `technical_criterion_scores.expected_identity_preservability` | integer | Pre-edit expected-risk score: 0, 1, or 2 |
| `mask_qc_status` | string/null | `not_tested` or null before mask QC |
| `edit_qc_status` | string/null | `not_tested` or null before edit QC |
| `object_identity_preservation_status` | string/null | `not_tested` or null before post-edit human QC |
| `difficulty` | object | Boolean metadata listed below |
""",
            ),
            (
                """- `qc.object_identity_preservation`
- `qc.prompt_answer_validity`
""",
                """- `mask_qc_status`: `pass`, `fail`, `human_review`, or `not_tested`
- `edit_qc_status`: `pass`, `fail`, `human_review`, or `not_tested`
- `object_identity_preservation_status`: `pass`, `fail`, `human_review`, or `not_tested`
- `object_identity_preservation_note`
- `object_identity_preservation_reviewer`
- `object_identity_preservation_timestamp`
- `qc.prompt_answer_validity`
""",
            ),
            (
                """4. Previous `KEEP`, `BORDERLINE`, and `REJECT` labels are provenance only and
   cannot populate v1.1 status fields automatically.
""",
                """4. Previous `KEEP`, `BORDERLINE`, and `REJECT` labels are provenance only and
   cannot populate v1.1 status fields automatically.
5. Pre-edit score `2` means expected favorable, not verified success.
6. Pre-edit actual-QC statuses must be `not_tested` or null.
7. An edited image requires tested edit and identity statuses; identity `fail`
   requires a note.
8. Final matched QC cannot pass with any applicable actual-QC status other than
   `pass`.
""",
            ),
        ],
        "docs/protocols/execution_plan_v1.1.md": [
            (
                """4. Assign `directly_editable` or `segmentation_test_required`.
5. Do not assign `failed_after_test` during initial review. It requires recorded
""",
                """4. Score `material_suitability`, `expected_mask_separability`,
   `sufficient_visible_area`, `swap_feasibility`, and
   `expected_identity_preservability` as pre-edit expected risks.
5. Assign `directly_editable` only when all five scores are `2`; score `2`
   means expected favorable, not verified success.
6. Otherwise assign `segmentation_test_required`.
7. Do not assign `failed_after_test` during initial review.
""",
            ),
            (
                """- Review mask leakage, boundary artifacts, identity, residual color, outside-mask
  preservation, and prompt/answer validity.
""",
                """- Review `mask_qc_status`, `edit_qc_status`,
  `object_identity_preservation_status`, residual color, outside-mask
  preservation, and prompt/answer validity.
- Actual identity preservation is judged only from generated edited assets;
  pre-edit expected scores cannot populate this result.
""",
            ),
            (
                """- Confirm fixed background, unchanged positions and scale, identity preservation,
  and exact shared cutout asset IDs across tracks.
""",
                """- Confirm fixed background, unchanged positions and scale,
  `object_identity_preservation_status=pass`, and exact shared cutout asset IDs
  across tracks.
""",
            ),
        ],
        "docs/protocols/protocol_v1.1_changelog.md": [
            (
                """| Technical failure | Could be assigned during visual curation | `failed_after_test` only after a recorded segmentation/edit test |
| Difficulty | Often acted as exclusion | Stored as metadata unless it creates one of the semantic-invalid conditions |
""",
                """| Technical failure | Could be assigned during visual curation | `failed_after_test` only after a recorded segmentation/edit test |
| Pre-edit identity field | `identity_preserved` could imply a verified result | `expected_identity_preservability`; score is expected risk only |
| Identity outcome | Could be conflated with pre-edit triage | `object_identity_preservation_status` is populated only after edit QC |
| Difficulty | Often acted as exclusion | Stored as metadata unless it creates one of the semantic-invalid conditions |
""",
            ),
            (
                """- Protocol v1.0 remains an immutable historical record.
- Legacy Yes/No development results are not converted into v1.1 primary metrics.
""",
                """- Protocol v1.0 remains an immutable historical record.
- Legacy pre-edit `identity_preserved`, `separable_masks`, and `swap_feasible`
  values are migrated without score changes to
  `expected_identity_preservability`, `expected_mask_separability`, and
  `swap_feasibility`. This is a concept/schema correction, not a sample
  reclassification or research-result change.
- Legacy Yes/No development results are not converted into v1.1 primary metrics.
""",
            ),
        ],
    }


def main() -> None:
    args = parse_args()
    if args.report.exists():
        raise FileExistsError(f"Refusing to overwrite report: {args.report}")

    prepared: dict[Path, tuple[str, str, int]] = {}
    for relative, operations in replacements().items():
        path = args.project_root / relative
        original = path.read_text(encoding="utf-8")
        revised = original
        applied = 0
        for old, new in operations:
            if old not in revised:
                raise ValueError(f"{relative}: expected patch anchor not found.")
            revised = revised.replace(old, new, 1)
            applied += 1
        prepared[path] = (original, revised, applied)

    report = {
        "patch_version": "technical_identity_fields_v1.1",
        "files": [],
    }
    for path, (original, revised, applied) in prepared.items():
        path.write_text(revised, encoding="utf-8")
        report["files"].append(
            {
                "path": path.relative_to(args.project_root).as_posix(),
                "replacements": applied,
                "sha256_before": digest(original),
                "sha256_after": digest(revised),
            }
        )

    args.report.parent.mkdir(parents=True, exist_ok=True)
    args.report.write_text(
        json.dumps(report, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
