#!/usr/bin/env python3
"""Build a model-unexposed GQA candidate pool for the main binding experiment."""

from __future__ import annotations

import argparse
import csv
import json
import math
import random
import re
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Iterable

from PIL import Image, ImageDraw, ImageFont

from extract_attribute_binding_candidates_v0_2 import (
    EXCLUDED_LABEL_GROUPS,
    PREFERRED_EDITABLE_LABELS,
)


SEED = 20260727
GQA_ID_PATTERN = re.compile(r"(?<!\d)(\d{3,7})(?!\d)")

FOOD_LABELS = {
    "apple",
    "apples",
    "banana",
    "bananas",
    "beer",
    "beef",
    "bread",
    "broccoli",
    "cake",
    "carrot",
    "carrots",
    "cheese",
    "chicken",
    "coffee",
    "donut",
    "doughnut",
    "egg",
    "eggs",
    "food",
    "frosting",
    "fruit",
    "fruits",
    "hot dog",
    "juice",
    "lemon",
    "lemons",
    "lettuce",
    "lime",
    "limes",
    "meal",
    "meat",
    "milk",
    "mustard",
    "orange",
    "oranges",
    "pineapple",
    "pizza",
    "pomegranate",
    "pork",
    "potato",
    "potatoes",
    "rice",
    "salad",
    "sandwich",
    "sauce",
    "spinach",
    "tomato",
    "tomatoes",
    "vegetable",
    "vegetables",
    "watermelon",
    "wine",
}

CLOTHING_LABELS = {
    "backpack",
    "bag",
    "belt",
    "blazer",
    "blouse",
    "cap",
    "coat",
    "dress",
    "glove",
    "gloves",
    "hat",
    "jacket",
    "jeans",
    "jersey",
    "luggage",
    "pajamas",
    "pants",
    "scarf",
    "shirt",
    "shoe",
    "shoes",
    "shorts",
    "skirt",
    "sock",
    "socks",
    "suit",
    "suitcase",
    "sweater",
    "sweatshirt",
    "t-shirt",
    "tie",
    "vest",
}
VEHICLE_LABELS = {
    "airplane",
    "bicycle",
    "bike",
    "boat",
    "bus",
    "car",
    "jet",
    "motorcycle",
    "suv",
    "taxi",
    "train",
    "truck",
    "van",
    "vehicle",
}
FURNITURE_LABELS = {
    "bed",
    "bench",
    "cabinet",
    "chair",
    "chairs",
    "couch",
    "desk",
    "dresser",
    "drawer",
    "shelf",
    "sofa",
    "table",
    "tables",
}
HOUSEHOLD_LABELS = {
    "bathtub",
    "blanket",
    "bottle",
    "bowl",
    "box",
    "bucket",
    "carpet",
    "clock",
    "cup",
    "curtain",
    "lamp",
    "mat",
    "napkin",
    "plate",
    "pillow",
    "rug",
    "saucer",
    "sheet",
    "tablecloth",
    "tent",
    "toothbrush",
    "towel",
    "tray",
    "umbrella",
    "vase",
}
SPORTS_LABELS = {
    "ball",
    "baseball bat",
    "frisbee",
    "helmet",
    "net",
    "racket",
    "skateboard",
    "skis",
    "snowboard",
    "surfboard",
    "tennis racket",
}
NONINDEPENDENT_LABELS = {
    "button",
    "collar",
    "logo",
    "pocket",
    "sleeve",
    "stripe",
    "stripes",
}
NATURAL_LABELS = {
    "bark",
    "flower",
    "flowers",
    "hill",
    "hills",
    "moss",
    "mountain",
    "mountains",
    "rock",
    "rocks",
}
PLURAL_LABELS = {
    "chairs",
    "clothes",
    "glasses",
    "gloves",
    "jeans",
    "pajamas",
    "pants",
    "shoes",
    "shorts",
    "skis",
    "socks",
    "tables",
}

REVIEW_FIELDS = [
    "candidate_id",
    "reviewer_id",
    "preview_path",
    "source_dataset",
    "source_image_id",
    "tier",
    "color_pair",
    "object_a_label",
    "object_a_color",
    "object_a_category",
    "object_b_label",
    "object_b_color",
    "object_b_category",
    "object_a_identifiable",
    "object_b_identifiable",
    "label_specificity",
    "semantic_distinction",
    "unique_reference",
    "original_a_color_clear",
    "original_b_color_clear",
    "opaque_nonreflective",
    "independent_masks_feasible",
    "swap_plausible",
    "identity_preservation_expected",
    "include_decision",
    "exclusion_reason",
    "notes",
]


def parse_args() -> argparse.Namespace:
    root = Path(__file__).resolve().parents[2]
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--strict-input",
        type=Path,
        default=root
        / "processed/attribute_binding/candidates/gqa/candidates_color_pairs_v0.2.jsonl",
    )
    parser.add_argument(
        "--relaxed-input",
        type=Path,
        default=root
        / "processed/attribute_binding/candidates/gqa/candidates_color_pairs.jsonl",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=root / "processed/attribute_binding/candidates/main_v1",
    )
    parser.add_argument(
        "--image-root",
        type=Path,
        default=root / "raw/GQA-Scene-Graph",
    )
    parser.add_argument("--min-candidates", type=int, default=250)
    parser.add_argument("--double-review-fraction", type=float, default=0.25)
    parser.add_argument("--seed", type=int, default=SEED)
    parser.add_argument("--create-previews", action="store_true")
    return parser.parse_args()


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            try:
                rows.append(json.loads(line))
            except json.JSONDecodeError as error:
                raise ValueError(f"{path}:{line_number}: {error}") from error
    return rows


def is_gqa_record(record: dict[str, Any], source_path: Path) -> bool:
    dataset = str(record.get("source_dataset", "")).lower()
    if "paco" in dataset:
        return False
    if "gqa" in dataset:
        return True
    return "paco" not in str(source_path).lower()


def extract_gqa_ids(value: Any, source_path: Path) -> set[str]:
    found: set[str] = set()
    if isinstance(value, dict):
        if not is_gqa_record(value, source_path):
            return found
        for key, item in value.items():
            if key in {"source_image_id", "image_id"}:
                text = str(item)
                if re.fullmatch(r"\d{1,8}", text):
                    found.add(text)
            elif key in {
                "image_path",
                "source_image_path",
                "original_image_path",
                "edited_image_path",
            }:
                found.update(GQA_ID_PATTERN.findall(str(item)))
            else:
                found.update(extract_gqa_ids(item, source_path))
    elif isinstance(value, list):
        for item in value:
            found.update(extract_gqa_ids(item, source_path))
    return found


def load_structured_file(path: Path) -> Iterable[Any]:
    if path.suffix == ".json":
        yield json.loads(path.read_text(encoding="utf-8"))
    elif path.suffix == ".jsonl":
        with path.open("r", encoding="utf-8") as handle:
            for line in handle:
                if line.strip():
                    yield json.loads(line)
    elif path.suffix == ".csv":
        with path.open("r", encoding="utf-8", newline="") as handle:
            yield from csv.DictReader(handle)


def collect_exposed_gqa_ids(project_root: Path) -> dict[str, set[str]]:
    exposure_sources: dict[str, set[str]] = defaultdict(set)
    roots = [
        project_root / "processed/attribute_binding/reviews",
        project_root / "processed/attribute_binding/segmentation",
        project_root / "processed/attribute_binding/edits",
        project_root / "processed/attribute_binding/audits",
        project_root / "processed/attribute_binding/datasets",
        project_root / "processed/attribute_binding/archive",
        project_root / "experiments/attribute_binding",
    ]
    for root in roots:
        if not root.exists():
            continue
        for path in sorted(root.rglob("*")):
            if not path.is_file() or path.suffix not in {".json", ".jsonl", ".csv"}:
                continue
            if "main_v1" in path.parts:
                continue
            try:
                for value in load_structured_file(path):
                    for image_id in extract_gqa_ids(value, path):
                        exposure_sources[image_id].add(
                            path.relative_to(project_root).as_posix()
                        )
            except (UnicodeDecodeError, json.JSONDecodeError):
                continue

    selection_path = (
        project_root
        / "processed/spatial_left_right/datasets/positive_control_v1/selection.json"
    )
    selection = json.loads(selection_path.read_text(encoding="utf-8"))
    for sample in selection.get("samples", []):
        for image_id in GQA_ID_PATTERN.findall(str(sample.get("sample_id", ""))):
            exposure_sources[image_id].add(
                selection_path.relative_to(project_root).as_posix()
            )
    return exposure_sources


def pair_key(row: dict[str, Any]) -> tuple[str, int, int]:
    indices = sorted(
        [
            int(row["object_a"]["detection_index"]),
            int(row["object_b"]["detection_index"]),
        ]
    )
    return str(row["image_id"]), indices[0], indices[1]


def bbox_touches_frame(bbox: list[float], margin: float = 0.005) -> bool:
    x, y, width, height = bbox
    return (
        x <= margin
        or y <= margin
        or x + width >= 1 - margin
        or y + height >= 1 - margin
    )


def category_for(label: str) -> str:
    if label in CLOTHING_LABELS:
        return "clothing"
    if label in VEHICLE_LABELS:
        return "vehicle"
    if label in FURNITURE_LABELS:
        return "furniture"
    if label in HOUSEHOLD_LABELS:
        return "household"
    if label in SPORTS_LABELS:
        return "sports"
    return "other_artifact"


def copula_for(label: str) -> str:
    return "are" if label in PLURAL_LABELS else "is"


def eligible_relaxed_row(
    row: dict[str, Any], hard_excluded_labels: set[str]
) -> tuple[bool, str]:
    first, second = row["object_a"], row["object_b"]
    labels = {str(first["label"]).lower(), str(second["label"]).lower()}
    if labels & hard_excluded_labels:
        return False, "hard_excluded_label"
    if labels & FOOD_LABELS:
        return False, "food_or_beverage"
    if labels & NATURAL_LABELS:
        return False, "natural_object"
    if labels & NONINDEPENDENT_LABELS:
        return False, "nonindependent_part_or_logo"
    areas = [float(first["bbox_area"]), float(second["bbox_area"])]
    if min(areas) < 0.03:
        return False, "small_bbox"
    if max(areas) > 0.40:
        return False, "large_bbox"
    if float(row["bbox_iou"]) > 0.10:
        return False, "high_iou"
    return True, ""


def priority_tuple(row: dict[str, Any]) -> tuple[Any, ...]:
    first, second = row["object_a"], row["object_b"]
    preferred_count = sum(
        str(obj["label"]).lower() in PREFERRED_EDITABLE_LABELS
        for obj in (first, second)
    )
    min_area = min(float(first["bbox_area"]), float(second["bbox_area"]))
    frame_touch_count = sum(
        bbox_touches_frame(obj["bbox_xywh_norm"]) for obj in (first, second)
    )
    return (
        0 if row["candidate_tier"] == "A_strict" else 1,
        -preferred_count,
        frame_touch_count,
        -min_area,
        float(row["bbox_iou"]),
        str(first["label"]),
        str(second["label"]),
    )


def balanced_order(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    grouped: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in sorted(rows, key=priority_tuple):
        grouped[row["color_pair"]].append(row)
    ordered: list[dict[str, Any]] = []
    color_pairs = sorted(grouped)
    while any(grouped.values()):
        for color_pair in color_pairs:
            if grouped[color_pair]:
                ordered.append(grouped[color_pair].pop(0))
    return ordered


def draw_preview(
    row: dict[str, Any], image_root: Path, destination: Path
) -> None:
    source = image_root / row["source_image_path"]
    with Image.open(source) as opened:
        image = opened.convert("RGB")
    width, height = image.size
    header_height = 72
    canvas = Image.new("RGB", (width, height + header_height), "white")
    canvas.paste(image, (0, header_height))
    draw = ImageDraw.Draw(canvas)
    font = ImageFont.load_default()
    title = (
        f"{row['candidate_id']} | {row['tier']} | {row['color_pair']} | "
        f"{row['object_a_color']} {row['object_a_label']} <-> "
        f"{row['object_b_color']} {row['object_b_label']}"
    )
    subtitle = (
        f"GQA {row['source_image_id']} | IoU={row['bbox_iou']:.3f} | "
        f"areas={row['object_a_bbox_area']:.3f}, {row['object_b_bbox_area']:.3f}"
    )
    draw.text((10, 10), title, fill="black", font=font)
    draw.text((10, 36), subtitle, fill="black", font=font)
    for prefix, outline in (("object_a", "#00b050"), ("object_b", "#e63946")):
        x, y, box_width, box_height = row[f"{prefix}_bbox_xywh_norm"]
        rectangle = (
            round(x * width),
            round(y * height) + header_height,
            round((x + box_width) * width),
            round((y + box_height) * height) + header_height,
        )
        draw.rectangle(rectangle, outline=outline, width=max(2, width // 300))
        label = f"{row[f'{prefix}_color']} {row[f'{prefix}_label']}"
        draw.text((rectangle[0] + 3, rectangle[1] + 3), label, fill=outline, font=font)
    destination.parent.mkdir(parents=True, exist_ok=True)
    canvas.thumbnail((1200, 1000))
    canvas.save(destination, quality=92)


def write_review_csv(
    path: Path, rows: list[dict[str, Any]], reviewer_id: str
) -> None:
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=REVIEW_FIELDS)
        writer.writeheader()
        for row in rows:
            output = {field: "" for field in REVIEW_FIELDS}
            output.update(
                {
                    "candidate_id": row["candidate_id"],
                    "reviewer_id": reviewer_id,
                    "preview_path": row["preview_path"],
                    "source_dataset": row["source_dataset"],
                    "source_image_id": row["source_image_id"],
                    "tier": row["tier"],
                    "color_pair": row["color_pair"],
                    "object_a_label": row["object_a_label"],
                    "object_a_color": row["object_a_color"],
                    "object_a_category": row["object_a_category"],
                    "object_b_label": row["object_b_label"],
                    "object_b_color": row["object_b_color"],
                    "object_b_category": row["object_b_category"],
                }
            )
            writer.writerow(output)


def main() -> None:
    args = parse_args()
    project_root = Path(__file__).resolve().parents[2]
    args.output_dir.mkdir(parents=True, exist_ok=True)
    preview_dir = args.output_dir / "previews"
    preview_dir.mkdir(parents=True, exist_ok=True)

    exposed = collect_exposed_gqa_ids(project_root)
    with (args.output_dir / "candidate_exclusions.csv").open(
        "w", encoding="utf-8", newline=""
    ) as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=["source_dataset", "source_image_id", "reason", "found_in"],
        )
        writer.writeheader()
        for image_id in sorted(exposed):
            writer.writerow(
                {
                    "source_dataset": "Voxel51/GQA-Scene-Graph",
                    "source_image_id": image_id,
                    "reason": "development_or_model_exposed_source_image",
                    "found_in": ";".join(sorted(exposed[image_id])),
                }
            )

    strict_rows = read_jsonl(args.strict_input)
    relaxed_rows = read_jsonl(args.relaxed_input)
    strict_keys = {pair_key(row) for row in strict_rows}
    hard_excluded = set().union(*EXCLUDED_LABEL_GROUPS.values())
    rejection_counts: Counter[str] = Counter()
    eligible_rows: list[dict[str, Any]] = []

    for row in relaxed_rows:
        image_id = str(row["image_id"])
        if image_id in exposed:
            rejection_counts["development_or_model_exposed"] += 1
            continue
        accepted, reason = eligible_relaxed_row(row, hard_excluded)
        if not accepted:
            rejection_counts[reason] += 1
            continue
        first, second = row["object_a"], row["object_b"]
        tier = "A_strict" if pair_key(row) in strict_keys else "B_review_required"
        eligible_rows.append(
            {
                **row,
                "candidate_tier": tier,
                "frame_touch_count": sum(
                    bbox_touches_frame(obj["bbox_xywh_norm"])
                    for obj in (first, second)
                ),
            }
        )

    best_by_image: dict[str, dict[str, Any]] = {}
    for row in eligible_rows:
        image_id = str(row["image_id"])
        if image_id not in best_by_image or priority_tuple(row) < priority_tuple(
            best_by_image[image_id]
        ):
            best_by_image[image_id] = row

    selected = balanced_order(list(best_by_image.values()))
    if len(selected) < args.min_candidates:
        raise RuntimeError(
            f"Only {len(selected)} independent candidates remain; "
            f"minimum is {args.min_candidates}."
        )

    output_rows: list[dict[str, Any]] = []
    for index, row in enumerate(selected, start=1):
        first, second = row["object_a"], row["object_b"]
        candidate_id = f"maincand_{index:04d}"
        preview_path = f"previews/{candidate_id}.jpg"
        output_rows.append(
            {
                "schema_version": "1.0",
                "candidate_id": candidate_id,
                "source_dataset": "Voxel51/GQA-Scene-Graph",
                "source_image_id": str(row["image_id"]),
                "source_image_path": str(row["image_path"]),
                "tier": row["candidate_tier"],
                "frame_touch_count": row["frame_touch_count"],
                "color_pair": row["color_pair"],
                "bbox_iou": float(row["bbox_iou"]),
                "object_a_detection_index": int(first["detection_index"]),
                "object_a_label": str(first["label"]),
                "object_a_copula": copula_for(str(first["label"])),
                "object_a_color": str(first["color"]),
                "object_a_category": category_for(str(first["label"])),
                "object_a_bbox_xywh_norm": first["bbox_xywh_norm"],
                "object_a_bbox_area": float(first["bbox_area"]),
                "object_a_attributes": first.get("attributes", []),
                "object_b_detection_index": int(second["detection_index"]),
                "object_b_label": str(second["label"]),
                "object_b_copula": copula_for(str(second["label"])),
                "object_b_color": str(second["color"]),
                "object_b_category": category_for(str(second["label"])),
                "object_b_bbox_xywh_norm": second["bbox_xywh_norm"],
                "object_b_bbox_area": float(second["bbox_area"]),
                "object_b_attributes": second.get("attributes", []),
                "preview_path": preview_path,
                "selection_stage": "model_blind_main_candidate_review",
                "confirmatory_test_eligible": False,
            }
        )

    pool_path = args.output_dir / "candidate_pool.jsonl"
    with pool_path.open("w", encoding="utf-8") as handle:
        for row in output_rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")

    if args.create_previews:
        for row in output_rows:
            draw_preview(row, args.image_root, args.output_dir / row["preview_path"])

    write_review_csv(
        args.output_dir / "annotation_sheet_rater1.csv", output_rows, "rater1"
    )
    random_generator = random.Random(args.seed)
    second_review_n = math.ceil(len(output_rows) * args.double_review_fraction)
    second_review_rows = random_generator.sample(output_rows, second_review_n)
    second_review_rows.sort(key=lambda row: row["candidate_id"])
    write_review_csv(
        args.output_dir / "annotation_sheet_rater2.csv",
        second_review_rows,
        "rater2",
    )

    summary = {
        "schema_version": "1.0",
        "seed": args.seed,
        "source_dataset": "Voxel51/GQA-Scene-Graph",
        "candidate_count": len(output_rows),
        "unique_source_image_count": len(
            {row["source_image_id"] for row in output_rows}
        ),
        "development_or_model_exposed_gqa_images": len(exposed),
        "double_review_count": second_review_n,
        "double_review_fraction": second_review_n / len(output_rows),
        "candidate_by_tier": dict(
            sorted(Counter(row["tier"] for row in output_rows).items())
        ),
        "candidate_by_color_pair": dict(
            sorted(Counter(row["color_pair"] for row in output_rows).items())
        ),
        "object_mentions_by_category": dict(
            sorted(
                Counter(
                    category
                    for row in output_rows
                    for category in (
                        row["object_a_category"],
                        row["object_b_category"],
                    )
                ).items()
            )
        ),
        "automatic_rejections": dict(sorted(rejection_counts.items())),
        "thresholds": {
            "min_bbox_area": 0.03,
            "max_bbox_area": 0.40,
            "max_bbox_iou": 0.10,
            "unique_source_image": True,
            "food_and_natural_labels_excluded": True,
            "development_source_images_excluded": True,
        },
        "warning": (
            "Automatic candidacy is not final inclusion. Human review, mask review, "
            "edit-quality review, and final 80-sample balance checks are required."
        ),
    }
    (args.output_dir / "candidate_pool_summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
