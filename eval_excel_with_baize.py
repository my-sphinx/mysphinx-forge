#!/usr/bin/env python3
"""
Evaluate text labels from an Excel file with grouped multi-thread processing.
"""

import argparse
import asyncio
import inspect
from pathlib import Path

import pandas as pd


async def call_baize_ds_8b(text_value: str) -> str:
    """
    Local model inference hook.
    Replace this implementation with your real local model call.
    """
    return str(text_value)


def find_project_root(*start_points: Path) -> Path:
    for start in start_points:
        current = start.resolve()
        for candidate in [current, *current.parents]:
            if (candidate / "data").is_dir():
                return candidate
    raise FileNotFoundError("Cannot find project root containing 'data/' directory.")


def resolve_input_path(project_root: Path, input_file: str) -> Path:
    raw = Path(input_file)
    if raw.is_absolute():
        return raw
    if raw.parts and raw.parts[0] == "data":
        return project_root / raw
    return project_root / "data" / raw


def resolve_output_path(project_root: Path, output: str) -> Path:
    raw = Path(output)
    return raw if raw.is_absolute() else project_root / raw


async def _call_baize_async(text: str) -> str:
    result = call_baize_ds_8b(text)
    if inspect.isawaitable(result):
        result = await result
    return str(result)


async def run_eval(df: pd.DataFrame) -> pd.DataFrame:
    total = len(df)
    if total == 0:
        return df

    out_df = df.copy()
    out_df["tested_label"] = ""
    for i, row_idx in enumerate(out_df.index, start=1):
        text = out_df.at[row_idx, "text"]
        text = "" if pd.isna(text) else str(text)
        progress = f"【{i}/{total}】"
        print(f"{progress} processing")
        label = await _call_baize_async(text)
        out_df.at[row_idx, "tested_label"] = label
        print(f"{progress} done")

    return out_df


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Run async text testing from Excel."
    )
    parser.add_argument(
        "--excel-file",
        "--input-file",
        dest="input_file",
        default="merged_queries.xlsx",
        help="Excel filename under data/ (default: merged_queries.xlsx)",
    )
    parser.add_argument(
        "--output",
        default="data/eval_result.csv",
        help="Output CSV path (default: data/eval_result.csv)",
    )
    args = parser.parse_args()

    project_root = find_project_root(Path(__file__).resolve().parent, Path.cwd())
    input_path = resolve_input_path(project_root, args.input_file)
    output_path = resolve_output_path(project_root, args.output)

    if not input_path.exists():
        raise FileNotFoundError(f"Excel file not found: {input_path}")

    df = pd.read_excel(input_path)
    if "text" not in df.columns:
        raise ValueError("Excel must contain 'text' column.")

    result_df = asyncio.run(run_eval(df))

    output_path.parent.mkdir(parents=True, exist_ok=True)
    result_df.to_csv(output_path, index=False)
    print(f"Done. Saved to {output_path}")


if __name__ == "__main__":
    main()
