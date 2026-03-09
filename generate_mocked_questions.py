#!/usr/bin/env python3
"""
Generate mocked questions by calling local REST API repeatedly.
"""

import argparse
import json
from pathlib import Path
from typing import Any, Dict
from urllib import request
from urllib.error import HTTPError, URLError

import pandas as pd


def build_output_path(output_arg: str, category: str) -> Path:
    output_path = Path(output_arg)
    stem = output_path.stem
    suffix = output_path.suffix or ".csv"
    category_suffix = f"_{category}"
    if stem.endswith(category_suffix):
        final_name = f"{stem}{suffix}"
    else:
        final_name = f"{stem}{category_suffix}{suffix}"
    return output_path.with_name(final_name)


def call_chat_api(url: str, agent_id: str, message: str, timeout: int) -> str:
    payload: Dict[str, Any] = {
        "agent_id": agent_id,
        "message": message,
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


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Generate mocked questions and save to CSV."
    )
    parser.add_argument(
        "--category",
        default="查持仓",
        help="Question category used in request message (default: 查持仓)",
    )
    parser.add_argument(
        "--question-count",
        type=int,
        default=3,
        help="How many times to call API (default: 3)",
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
        "--output",
        default="mocked_questions.csv",
        help="Output CSV base path (default: mocked_questions.csv)",
    )
    args = parser.parse_args()

    if args.question_count <= 0:
        raise ValueError("question-count must be greater than 0")

    rows = []
    for idx in range(1, args.question_count + 1):
        print(f"[{idx}/{args.question_count}] requested")
        text = call_chat_api(args.url, args.agent_id, args.category, args.timeout)
        print(f"[{idx}/{args.question_count}] succeeded")
        rows.append({"category": args.category, "text": text})

    out_df = pd.DataFrame(rows, columns=["category", "text"])
    output_path = build_output_path(args.output, args.category)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    out_df.to_csv(output_path, index=False)
    print(f"Done. Saved to {output_path}")


if __name__ == "__main__":
    main()
