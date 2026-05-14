#!/usr/bin/env python3
"""Inventory all LegalBench tasks from Hugging Face and save a summary CSV."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import pandas as pd
from tqdm import tqdm

sys.path.insert(0, str(Path(__file__).resolve().parent))

from config_utils import (
    add_common_args,
    is_dry_run,
    load_config,
    merge_cli_overrides,
)
from utils import PROJECT_ROOT, ensure_dirs, preview


def main() -> None:
    parser = argparse.ArgumentParser(description="Inventory all LegalBench tasks.")
    add_common_args(parser)
    args = parser.parse_args()

    config = load_config(args.config)
    config = merge_cli_overrides(config, args)
    dataset_name = config["data"]["dataset_name"]

    from datasets import get_dataset_config_names, load_dataset

    print(f"Fetching config names for '{dataset_name}' ...")
    config_names = get_dataset_config_names(dataset_name)
    print(f"Found {len(config_names)} configs.")

    records: list[dict] = []
    success_count = 0
    fail_count = 0

    for cfg_name in tqdm(config_names, desc="Loading tasks"):
        record: dict = {"task": cfg_name, "success": False, "error": ""}
        try:
            ds = load_dataset(dataset_name, cfg_name)
            splits = list(ds.keys())
            total_rows = 0
            split_info: dict[str, dict] = {}
            for split in splits:
                n = len(ds[split])
                total_rows += n
                cols = list(ds[split].column_names)
                first_row = dict(ds[split][0]) if n > 0 else {}
                # Truncate long values in preview
                first_row_preview = {
                    k: preview(str(v), 120) for k, v in first_row.items()
                }
                split_info[split] = {
                    "rows": n,
                    "columns": cols,
                    "first_row_preview": first_row_preview,
                }
            record.update(
                {
                    "success": True,
                    "splits": ",".join(splits),
                    "total_rows": total_rows,
                    "columns": json.dumps(
                        {s: info["columns"] for s, info in split_info.items()}
                    ),
                    "first_row_preview": json.dumps(
                        {
                            s: info["first_row_preview"]
                            for s, info in split_info.items()
                        }
                    ),
                }
            )
            for split, info in split_info.items():
                record[f"rows_{split}"] = info["rows"]
            success_count += 1
        except Exception as e:
            record["error"] = str(e)
            fail_count += 1

        records.append(record)

    df = pd.DataFrame(records)

    print(f"\nTotal configs:       {len(config_names)}")
    print(f"Successfully loaded: {success_count}")
    print(f"Failed:              {fail_count}")

    if success_count > 0:
        top = (
            df[df["success"]]
            .sort_values("total_rows", ascending=False)
            .head(20)[["task", "total_rows", "splits"]]
        )
        print("\nTop 20 tasks by total rows:")
        print(top.to_string(index=False))

    if is_dry_run(config):
        print("\n[DRY RUN] Would write inventory CSV but skipping.")
        return

    out_path = PROJECT_ROOT / "data" / "processed" / "legalbench_inventory.csv"
    ensure_dirs(out_path.parent)
    df.to_csv(out_path, index=False)
    print(f"\nInventory saved to {out_path}")


if __name__ == "__main__":
    main()
