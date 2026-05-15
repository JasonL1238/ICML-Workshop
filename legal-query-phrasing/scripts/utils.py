"""Shared utilities for the legal-query-phrasing pipeline."""

from __future__ import annotations

import json
import os
import re
import tempfile
from pathlib import Path
from typing import Any

import anthropic
import openai
from dotenv import load_dotenv
from tenacity import (
    retry,
    retry_if_exception_type,
    stop_after_attempt,
    wait_exponential,
)

PROJECT_ROOT = Path(__file__).resolve().parent.parent


# ---------------------------------------------------------------------------
# Environment
# ---------------------------------------------------------------------------

def load_env() -> dict[str, str]:
    """Load .env and return key env vars. Raises if ANTHROPIC_API_KEY is missing."""
    load_dotenv(PROJECT_ROOT / ".env")
    api_key = os.environ.get("ANTHROPIC_API_KEY", "")
    if not api_key or api_key == "your_anthropic_api_key_here":
        raise RuntimeError(
            "ANTHROPIC_API_KEY is not set. Copy .env.example to .env and add your key."
        )
    env = {
        "ANTHROPIC_API_KEY": api_key,
        "ANTHROPIC_MODEL": os.environ.get("ANTHROPIC_MODEL", "claude-sonnet-4-6"),
    }
    openai_key = os.environ.get("OPENAI_API_KEY", "")
    if openai_key and openai_key != "your_openai_api_key_here":
        env["OPENAI_API_KEY"] = openai_key
    env["OPENAI_MODEL"] = os.environ.get("OPENAI_MODEL", "gpt-5.4")
    return env


def require_openai_key(env: dict[str, str]) -> None:
    """Raise if OPENAI_API_KEY was not loaded."""
    if "OPENAI_API_KEY" not in env:
        raise RuntimeError(
            "OPENAI_API_KEY is not set. Add it to .env (see .env.example)."
        )


# ---------------------------------------------------------------------------
# Directory helpers
# ---------------------------------------------------------------------------

def ensure_dirs(*paths: Path) -> None:
    for p in paths:
        p.mkdir(parents=True, exist_ok=True)


# ---------------------------------------------------------------------------
# JSONL I/O
# ---------------------------------------------------------------------------

def read_jsonl(path: Path) -> list[dict]:
    path = Path(path)
    if not path.exists():
        return []
    rows: list[dict] = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def write_jsonl(path: Path, rows: list[dict]) -> None:
    """Atomic write: write to a temp file then rename."""
    path = Path(path)
    ensure_dirs(path.parent)
    tmp_fd, tmp_path = tempfile.mkstemp(
        dir=path.parent, suffix=".tmp", prefix=path.stem
    )
    try:
        with os.fdopen(tmp_fd, "w", encoding="utf-8") as f:
            for row in rows:
                f.write(json.dumps(row, ensure_ascii=False) + "\n")
        os.replace(tmp_path, path)
    except BaseException:
        if os.path.exists(tmp_path):
            os.unlink(tmp_path)
        raise


def append_jsonl(path: Path, row: dict) -> None:
    """Append a single row, flush+fsync for crash safety."""
    path = Path(path)
    ensure_dirs(path.parent)
    with open(path, "a", encoding="utf-8") as f:
        f.write(json.dumps(row, ensure_ascii=False) + "\n")
        f.flush()
        os.fsync(f.fileno())


# ---------------------------------------------------------------------------
# Prompt templates
# ---------------------------------------------------------------------------

def load_prompt_template(path: str | Path) -> str:
    path = Path(path)
    if not path.is_absolute():
        path = PROJECT_ROOT / path
    return path.read_text(encoding="utf-8")


# ---------------------------------------------------------------------------
# Text helpers
# ---------------------------------------------------------------------------

def word_count(text: str | None) -> int:
    if not text:
        return 0
    return len(text.split())


def preview(text: str | None, n_chars: int = 200) -> str:
    if not text:
        return ""
    text = " ".join(text.split())
    if len(text) <= n_chars:
        return text
    return text[:n_chars] + "..."


# ---------------------------------------------------------------------------
# Claude helpers
# ---------------------------------------------------------------------------

def extract_text_from_message(message: Any) -> str:
    """Concatenate all text content blocks from an Anthropic Message response."""
    parts: list[str] = []
    for block in message.content:
        if hasattr(block, "text"):
            parts.append(block.text)
    return "\n".join(parts)


_RETRYABLE = (
    anthropic.APIConnectionError,
    anthropic.RateLimitError,
    anthropic.InternalServerError,
)


@retry(
    retry=retry_if_exception_type(_RETRYABLE),
    wait=wait_exponential(multiplier=2, min=4, max=120),
    stop=stop_after_attempt(6),
    reraise=True,
)
def call_claude(
    prompt: str,
    model: str = "claude-sonnet-4-6",
    temperature: float = 0.0,
    max_tokens: int = 700,
) -> tuple[str, dict]:
    """Call Claude and return (text_output, usage_dict).

    Uses ANTHROPIC_API_KEY from environment. Retries on transient errors.
    """
    client = anthropic.Anthropic()  # reads ANTHROPIC_API_KEY from env
    message = client.messages.create(
        model=model,
        max_tokens=max_tokens,
        temperature=temperature,
        messages=[{"role": "user", "content": prompt}],
    )
    text = extract_text_from_message(message)
    usage = {
        "input_tokens": message.usage.input_tokens,
        "output_tokens": message.usage.output_tokens,
    }
    return text, usage


# ---------------------------------------------------------------------------
# OpenAI helpers
# ---------------------------------------------------------------------------

_OPENAI_RETRYABLE = (
    openai.APIConnectionError,
    openai.RateLimitError,
    openai.InternalServerError,
)


@retry(
    retry=retry_if_exception_type(_OPENAI_RETRYABLE),
    wait=wait_exponential(multiplier=2, min=4, max=120),
    stop=stop_after_attempt(6),
    reraise=True,
)
def call_openai(
    prompt: str,
    model: str = "gpt-5.4",
    temperature: float = 0.0,
    max_tokens: int = 700,
) -> tuple[str, dict]:
    """Call OpenAI chat completion and return (text_output, usage_dict).

    Uses OPENAI_API_KEY from environment. Retries on transient errors.
    """
    client = openai.OpenAI()
    response = client.chat.completions.create(
        model=model,
        max_tokens=max_tokens,
        temperature=temperature,
        messages=[{"role": "user", "content": prompt}],
    )
    text = response.choices[0].message.content or ""
    usage = {
        "input_tokens": response.usage.prompt_tokens,
        "output_tokens": response.usage.completion_tokens,
    }
    return text, usage


# ---------------------------------------------------------------------------
# Safe JSON parsing from model output
# ---------------------------------------------------------------------------

def safe_parse_json(text: str) -> tuple[dict | None, str]:
    """Try to parse JSON from model output.

    Handles:
    - Pure JSON
    - JSON inside markdown fences (```json ... ```)
    - Extra text before/after JSON

    Returns (parsed_dict_or_None, raw_text).
    """
    if not text:
        return None, text

    # 1. Try direct parse
    try:
        return json.loads(text), text
    except json.JSONDecodeError:
        pass

    # 2. Try extracting from markdown fences
    fence_match = re.search(r"```(?:json)?\s*\n?(.*?)\n?\s*```", text, re.DOTALL)
    if fence_match:
        try:
            return json.loads(fence_match.group(1)), text
        except json.JSONDecodeError:
            pass

    # 3. Try extracting the first top-level JSON object
    brace_start = text.find("{")
    if brace_start != -1:
        depth = 0
        for i in range(brace_start, len(text)):
            if text[i] == "{":
                depth += 1
            elif text[i] == "}":
                depth -= 1
                if depth == 0:
                    try:
                        return json.loads(text[brace_start : i + 1]), text
                    except json.JSONDecodeError:
                        break

    return None, text


# ---------------------------------------------------------------------------
# Output file management
# ---------------------------------------------------------------------------

def get_completed_ids(path: Path, id_field: str = "item_id") -> set[str]:
    """Read existing output JSONL and return the set of completed IDs."""
    rows = read_jsonl(path)
    return {r[id_field] for r in rows if id_field in r}


def handle_output_file(
    path: Path, resume: bool, overwrite: bool
) -> set[str]:
    """Manage output file state and return set of already-completed item_ids.

    - overwrite=True  -> delete existing file, return empty set
    - resume=True     -> return existing ids
    - file exists but neither resume nor overwrite -> raise error
    """
    path = Path(path)
    if overwrite and path.exists():
        path.unlink()
        return set()
    if path.exists() and not resume:
        raise FileExistsError(
            f"Output file already exists: {path}\n"
            "Use --resume to continue or --overwrite to replace."
        )
    if path.exists() and resume:
        return get_completed_ids(path)
    return set()
