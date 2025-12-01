#!/usr/bin/env python3
"""Simple CLI to validate the OpenAI-compatible endpoint and API key."""

import argparse
import json
import sys
from typing import Any, Dict

import requests


def build_payload(model: str) -> Dict[str, Any]:
    """Construct the minimal chat completion payload."""
    return {
        "model": model,
        "messages": [
            {
                "role": "system",
                "content": "You are a connection test helper.",
            },
            {
                "role": "user",
                "content": "Reply with the word PONG.",
            },
        ],
        "max_completion_tokens": 16,
    }


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Test whether the OpenAI endpoint and API key work."
    )
    parser.add_argument(
        "--url",
        default=(
            "http://nginx-openai.ai-common.10.100.10.17.sslip.io/openai/deployments/o3-mini/chat/completions?api-version=2025-01-01-preview"
        ),
        help="Full chat completions URL (including api-version)",
    )
    parser.add_argument(
        "--api-key",
        default="aa70eb304898428793d4114736e56666",
        help="API key with access to the deployment",
    )
    parser.add_argument(
        "--model",
        default="gpt-4.1-nano",
        help="Model / deployment name to query",
    )
    parser.add_argument(
        "--timeout",
        type=float,
        default=30.0,
        help="Request timeout in seconds",
    )

    args = parser.parse_args()
    payload = build_payload(args.model)
    headers = {
        "Content-Type": "application/json",
        "api-key": args.api_key,
    }

    try:
        response = requests.post(
            args.url, headers=headers, json=payload, timeout=args.timeout
        )
    except requests.RequestException as exc:
        print(f"[FAIL] 网络请求失败: {exc}", file=sys.stderr)
        return 2

    print(f"[INFO] HTTP {response.status_code}")

    try:
        data = response.json()
        pretty = json.dumps(data, ensure_ascii=False, indent=2)
        print(pretty)
    except ValueError:
        print(response.text)
        print("[WARN] 无法将响应解析为 JSON")
        return 3 if not response.ok else 0

    if response.ok:
        print("[PASS] 连接成功，可正常获得响应。")
        return 0

    print("[FAIL] 连接失败，请检查 URL / Key / Model 配置。")
    return 4


if __name__ == "__main__":
    sys.exit(main())

