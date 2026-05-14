#!/usr/bin/env python3
"""Smoke test: verify that the Anthropic API key works before running the pipeline."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from utils import load_env, call_claude
from config_utils import load_config, add_common_args, merge_cli_overrides


def main() -> None:
    parser = argparse.ArgumentParser(description="Smoke-test the Anthropic API key.")
    add_common_args(parser)
    args = parser.parse_args()

    config = load_config(args.config)
    config = merge_cli_overrides(config, args)

    env = load_env()

    model = env.get("ANTHROPIC_MODEL") or config["generation"]["model"]
    print(f"Model:  {model}")
    print("Calling Claude with: 'Reply with exactly: API OK'")
    print("-" * 40)

    text, usage = call_claude(
        prompt="Reply with exactly: API OK",
        model=model,
        temperature=0.0,
        max_tokens=32,
    )

    print(f"Response: {text}")
    print(f"Input tokens:  {usage.get('input_tokens')}")
    print(f"Output tokens: {usage.get('output_tokens')}")
    print("-" * 40)
    print("Smoke test PASSED.")


if __name__ == "__main__":
    main()
