#!/usr/bin/env python3
"""
Fill CSV label column by calling a local REST JSON API row by row.
"""

import argparse
import json
from pathlib import Path
from typing import Any, Dict
from concurrent.futures import ThreadPoolExecutor
from urllib import request
from urllib.error import HTTPError, URLError

import pandas as pd


def call_chat_api(url: str, agent_id: str, text: str, timeout: int) -> str:
    payload: Dict[str, Any] = {
        "agent_id": agent_id,
        "message": text,
        "stream": False,
    }
    body = json.dumps(payload, ensure_ascii=False).encode("utf-8")
    req = request.Request(
        url=url,
        data=body,
        headers={"Content-Type": "application/json"},
        method="POST",
    )

    try:
        with request.urlopen(req, timeout=timeout) as resp:
            raw = resp.read().decode("utf-8")
    except HTTPError as e:
        detail = e.read().decode("utf-8", errors="ignore")
        raise RuntimeError(f"HTTP {e.code} calling {url}: {detail}") from e
    except URLError as e:
        raise RuntimeError(f"Failed to call {url}: {e.reason}") from e

    try:
        data = json.loads(raw)
    except json.JSONDecodeError as e:
        raise RuntimeError(f"Invalid JSON response: {raw}") from e

    if "response" not in data:
        raise RuntimeError(f"Response JSON missing 'response' field: {data}")

    return str(data["response"])


def split_dataframe_by_threads(df: pd.DataFrame, threads: int) -> list[pd.DataFrame]:
    total = len(df)
    if threads <= 0:
        raise ValueError("threads must be greater than 0")

    effective_threads = min(threads, total) if total > 0 else 1

    # If task count is smaller than configured threads, use one task per thread.
    if total < threads:
        groups: list[pd.DataFrame] = []
        for i in range(effective_threads):
            groups.append(df.iloc[i : i + 1].copy())
        return groups

    base = total // threads
    remainder = total % threads
    groups: list[pd.DataFrame] = []
    start = 0

    for i in range(effective_threads):
        size = base
        if i == effective_threads - 1:
            size += remainder

        end = start + size
        groups.append(df.iloc[start:end].copy())
        start = end

    return groups


def process_group(
    group_df: pd.DataFrame,
    group_idx: int,
    url: str,
    agent_id: str,
    timeout: int,
) -> pd.DataFrame:
    group_name = f"group{group_idx}"
    group_total = len(group_df)
    if group_total == 0:
        return group_df

    for pos, row_idx in enumerate(group_df.index, start=1):
        text = group_df.at[row_idx, "text"]
        if pd.isna(text):
            text = ""
        text = str(text)

        progress = f"[{pos}/{group_total}/{group_name}]"
        print(f"{progress} requested")
        try:
            label = call_chat_api(url, agent_id, text, timeout)
        except Exception:
            print(f"{progress} failed")
            raise

        group_df.at[row_idx, "label"] = label
        print(f"{progress} succeeded")

    return group_df


def main() -> None:
    parser = argparse.ArgumentParser(description="Fill labels in CSV by REST API.")
    parser.add_argument(
        "--input",
        default="data/sampled_for_annotation.csv",
        help="Input CSV path (default: data/sampled_for_annotation.csv)",
    )
    parser.add_argument(
        "--output",
        default="data/labeld_result.csv",
        help="Output CSV path (default: data/labeld_result.csv)",
    )
    parser.add_argument(
        "--url",
        default="http://localhost:8080/chat",
        help="REST API endpoint (default: http://localhost:8080/chat)",
    )
    parser.add_argument(
        "--agent-id",
        default="insurance",
        help="agent_id in request body (default: insurance)",
    )
    parser.add_argument(
        "--timeout",
        type=int,
        default=30,
        help="HTTP timeout seconds (default: 30)",
    )
    parser.add_argument(
        "--threads",
        type=int,
        default=5,
        help="Number of worker threads (default: 5)",
    )
    args = parser.parse_args()

    input_path = Path(args.input)
    output_path = Path(args.output)

    if not input_path.exists():
        raise FileNotFoundError(f"Input file not found: {input_path}")

    df = pd.read_csv(input_path)

    if "text" not in df.columns:
        raise ValueError("Input CSV must contain 'text' column.")

    if "label" not in df.columns:
        df["label"] = ""
    # Ensure label column can hold string results from API.
    df["label"] = df["label"].astype("object")

    groups = split_dataframe_by_threads(df, args.threads)
    future_results = []
    max_workers = min(args.threads, len(df)) if len(df) > 0 else 1
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        for group_idx, group_df in enumerate(groups, start=1):
            future = executor.submit(
                process_group,
                group_df,
                group_idx,
                args.url,
                args.agent_id,
                args.timeout,
            )
            future_results.append(future)

    processed_groups = [future.result() for future in future_results]
    out_df = pd.concat(processed_groups, ignore_index=True)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    out_df.to_csv(output_path, index=False)
    print(f"Done. Saved to {output_path}")


if __name__ == "__main__":
    main()
