#!/usr/bin/env python3
"""Download only the COCO images referenced by PACO candidate pairs."""

from __future__ import annotations

import argparse
import concurrent.futures
import json
import urllib.request
from pathlib import Path
from typing import Any


def parse_args() -> argparse.Namespace:
    root = Path(__file__).resolve().parents[2]
    parser = argparse.ArgumentParser(description="Download PACO candidate images.")
    parser.add_argument(
        "--manifest",
        type=Path,
        default=root
        / "processed/attribute_binding/candidates/paco/paco_mask_pairs_train_v0.1/candidates.jsonl",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=root / "raw/PACO-LVIS/candidate_images_v0.1",
    )
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument("--timeout", type=int, default=30)
    return parser.parse_args()


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    with path.open("r", encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def download(row: dict[str, Any], output_dir: Path, timeout: int) -> dict[str, Any]:
    url = row["coco_url"]
    suffix = Path(row["file_name"]).suffix or ".jpg"
    target = output_dir / f"{row['image_id']:012d}{suffix}"
    if target.exists() and target.stat().st_size > 0:
        return {"status": "existing", "path": str(target), "error": ""}
    request = urllib.request.Request(url, headers={"User-Agent": "vlm-binding-research/0.1"})
    try:
        with urllib.request.urlopen(request, timeout=timeout) as response:
            payload = response.read()
        target.write_bytes(payload)
        return {"status": "downloaded", "path": str(target), "error": ""}
    except Exception as error:  # Network failures are recorded per image.
        return {"status": "failed", "path": str(target), "error": repr(error)}


def main() -> None:
    args = parse_args()
    rows = read_jsonl(args.manifest)
    args.output_dir.mkdir(parents=True, exist_ok=True)

    results: dict[int, dict[str, Any]] = {}
    with concurrent.futures.ThreadPoolExecutor(max_workers=args.workers) as executor:
        futures = {
            executor.submit(download, row, args.output_dir, args.timeout): row["image_id"]
            for row in rows
        }
        for index, future in enumerate(concurrent.futures.as_completed(futures), start=1):
            image_id = futures[future]
            results[image_id] = future.result()
            if index % 50 == 0 or index == len(futures):
                print(f"Completed {index}/{len(futures)}")

    with (args.output_dir / "download_manifest.jsonl").open(
        "w", encoding="utf-8"
    ) as handle:
        for row in rows:
            result = results[row["image_id"]]
            handle.write(
                json.dumps(
                    {
                        "paco_mask_pair_id": row["paco_mask_pair_id"],
                        "image_id": row["image_id"],
                        "coco_url": row["coco_url"],
                        **result,
                    },
                    ensure_ascii=False,
                )
                + "\n"
            )

    status_counts = {
        status: sum(result["status"] == status for result in results.values())
        for status in ("downloaded", "existing", "failed")
    }
    summary = {
        "schema_version": "0.1",
        "requested": len(rows),
        **status_counts,
        "source_manifest": str(args.manifest),
    }
    (args.output_dir / "summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
