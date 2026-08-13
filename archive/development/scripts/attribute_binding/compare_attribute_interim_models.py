#!/usr/bin/env python3
"""Compare model-level and sample-level gates for the n=8 interim test."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path


MODELS = {
    "Qwen2.5-VL-7B": "qwen2_5_vl_7b_generation",
    "LLaVA-OneVision-7B": "llava_onevision_qwen2_7b",
    "InternVL2.5-8B": "internvl2_5_8b",
}


def parse_args() -> argparse.Namespace:
    root = Path(__file__).resolve().parents[2]
    base = root / "experiments/attribute_binding/interim_n8_v1"
    parser = argparse.ArgumentParser(description="Compare interim model gates.")
    parser.add_argument("--metrics-dir", type=Path, default=base / "metrics")
    parser.add_argument("--output-dir", type=Path, default=base / "comparison")
    return parser.parse_args()


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


def main() -> None:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    summary_rows = []
    patterns_by_model = {}
    for model_name, stem in MODELS.items():
        metrics = json.loads(
            (args.metrics_dir / f"{stem}_metrics.json").read_text(encoding="utf-8")
        )
        consistency = metrics["sample_consistency"]
        cq = metrics["control_qualified_incremental_binding_failure"]
        summary_rows.append(
            {
                "model": model_name,
                "oav": consistency["original_atomic_validity"]["numerator"],
                "eav": consistency["edited_atomic_validity"]["numerator"],
                "obc": consistency["original_binding_consistency"]["numerator"],
                "ebc": consistency["edited_binding_consistency"]["numerator"],
                "fbcc": consistency["full_binding_counterfactual_consistency"]["numerator"],
                "all_atomic": consistency["all_atomic_controls_pass"]["numerator"],
                "cq_ibf": cq["numerator"],
                "cq_ibf_denominator": cq["denominator"],
                "n_samples": metrics["n_samples"],
            }
        )
        patterns_by_model[model_name] = {
            row["sample_id"]: row
            for row in read_csv(args.metrics_dir / f"{stem}_sample_patterns.csv")
        }

    with (args.output_dir / "model_summary.csv").open(
        "w", encoding="utf-8", newline=""
    ) as handle:
        writer = csv.DictWriter(handle, fieldnames=list(summary_rows[0]))
        writer.writeheader()
        writer.writerows(summary_rows)

    sample_ids = sorted(next(iter(patterns_by_model.values())))
    overlap_rows = []
    for sample_id in sample_ids:
        first = next(iter(patterns_by_model.values()))[sample_id]
        row = {
            "sample_id": sample_id,
            "source_candidate_id": first["source_candidate_id"],
            "source_dataset": first["source_dataset"],
            "color_pair": first["color_pair"],
        }
        for model_name, patterns in patterns_by_model.items():
            pattern = patterns[sample_id]
            prefix = model_name.lower().replace("-", "_").replace(".", "_")
            for gate in ("oav", "eav", "obc", "ebc", "cq_ibf_qualified", "cq_ibf"):
                row[f"{prefix}_{gate}"] = pattern[gate]
            row[f"{prefix}_failed_atomics"] = pattern["failed_atomic_conditions"]
        overlap_rows.append(row)

    with (args.output_dir / "sample_gate_overlap.csv").open(
        "w", encoding="utf-8", newline=""
    ) as handle:
        writer = csv.DictWriter(handle, fieldnames=list(overlap_rows[0]))
        writer.writeheader()
        writer.writerows(overlap_rows)
    print(f"Wrote {len(summary_rows)} model summaries and {len(overlap_rows)} sample rows")


if __name__ == "__main__":
    main()
