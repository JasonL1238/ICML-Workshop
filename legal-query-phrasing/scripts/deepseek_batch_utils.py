"""DeepSeek Batch API utilities.

DeepSeek uses an OpenAI-compatible batch API. This module wraps
openai_batch_utils with the DeepSeek base URL and API key.
"""

from __future__ import annotations

import os

import openai

from openai_batch_utils import (
    BatchJob,
    BatchRequest,
    BatchResult,
    _build_request_line,
    poll_batch as _openai_poll_batch,
    retrieve_results as _openai_retrieve_results,
)
from utils import PROJECT_ROOT, ensure_dirs, load_env, require_deepseek_key

import json
import tempfile
import time
from pathlib import Path

BATCH_STATE_DIR = PROJECT_ROOT / "data" / "batch_state"
DEEPSEEK_BASE_URL = "https://api.deepseek.com"


def _get_deepseek_client() -> openai.OpenAI:
    """Create an OpenAI client pointing at DeepSeek."""
    api_key = os.environ.get("DEEPSEEK_API_KEY", "")
    return openai.OpenAI(api_key=api_key, base_url=DEEPSEEK_BASE_URL)


def submit_batch(
    requests: list[BatchRequest],
    script_name: str = "unknown",
) -> BatchJob:
    """Submit a batch to the DeepSeek Batch API (OpenAI-compatible)."""
    env = load_env()
    require_deepseek_key(env)
    client = _get_deepseek_client()

    with tempfile.NamedTemporaryFile(
        mode="w", suffix=".jsonl", delete=False, encoding="utf-8",
    ) as f:
        for req in requests:
            f.write(json.dumps(_build_request_line(req), ensure_ascii=False) + "\n")
        tmp_path = f.name

    try:
        with open(tmp_path, "rb") as f:
            file_obj = client.files.create(file=f, purpose="batch")
    finally:
        Path(tmp_path).unlink(missing_ok=True)

    batch = client.batches.create(
        input_file_id=file_obj.id,
        endpoint="/v1/chat/completions",
        completion_window="24h",
    )

    job = BatchJob(
        batch_id=batch.id,
        script_name=script_name,
        total_requests=len(requests),
        input_file_id=file_obj.id,
        custom_ids=[r.custom_id for r in requests],
        status=batch.status,
    )
    job.save()

    return job


def poll_batch(batch_id: str, poll_interval: int = 30, quiet: bool = False) -> str:
    """Poll until batch completes. Returns final status."""
    env = load_env()
    require_deepseek_key(env)
    client = _get_deepseek_client()

    terminal_statuses = {"completed", "failed", "expired", "cancelled"}

    while True:
        batch = client.batches.retrieve(batch_id)
        status = batch.status
        counts = batch.request_counts

        if not quiet:
            print(
                f"  Batch {batch_id}: {status} | "
                f"completed={counts.completed} "
                f"failed={counts.failed} "
                f"total={counts.total}"
            )

        if status in terminal_statuses:
            return status

        time.sleep(poll_interval)


def retrieve_results(batch_id: str) -> list[BatchResult]:
    """Retrieve all results from a completed DeepSeek batch."""
    env = load_env()
    require_deepseek_key(env)
    client = _get_deepseek_client()

    batch = client.batches.retrieve(batch_id)

    if batch.status != "completed":
        error_msg = f"Batch ended with status '{batch.status}'"
        if batch.errors and batch.errors.data:
            details = "; ".join(e.message for e in batch.errors.data if e.message)
            error_msg += f": {details}"
        raise RuntimeError(error_msg)

    output_file_id = batch.output_file_id
    if not output_file_id:
        raise RuntimeError(f"Batch {batch_id} completed but has no output_file_id")

    content = client.files.content(output_file_id)
    raw_text = content.text

    results: list[BatchResult] = []
    for line in raw_text.strip().splitlines():
        if not line.strip():
            continue
        entry = json.loads(line)
        custom_id = entry["custom_id"]
        response_body = entry.get("response", {})
        error_body = entry.get("error")

        if error_body:
            results.append(BatchResult(
                custom_id=custom_id,
                success=False,
                error=str(error_body),
                error_type="request_error",
            ))
            continue

        status_code = response_body.get("status_code", 200)
        body = response_body.get("body", {})

        if status_code != 200:
            err_msg = body.get("error", {}).get("message", f"HTTP {status_code}")
            results.append(BatchResult(
                custom_id=custom_id,
                success=False,
                error=err_msg,
                error_type=body.get("error", {}).get("type", "api_error"),
            ))
            continue

        choices = body.get("choices", [])
        text = choices[0]["message"]["content"] if choices else ""
        usage_data = body.get("usage", {})
        usage = {
            "input_tokens": usage_data.get("prompt_tokens", 0),
            "output_tokens": usage_data.get("completion_tokens", 0),
        }
        results.append(BatchResult(
            custom_id=custom_id,
            success=True,
            text=text,
            usage=usage,
        ))

    return results


def run_deepseek_batch_and_wait(
    requests: list[BatchRequest],
    script_name: str = "unknown",
    poll_interval: int = 30,
) -> list[BatchResult]:
    """Submit a DeepSeek batch, poll until done, and return results."""
    if not requests:
        return []

    print(f"\n  Submitting DeepSeek batch of {len(requests)} requests...")
    job = submit_batch(requests, script_name=script_name)
    print(f"  Batch ID: {job.batch_id}")
    print(f"  Input file ID: {job.input_file_id}")
    print(f"  Polling every {poll_interval}s until complete...\n")

    final_status = poll_batch(job.batch_id, poll_interval=poll_interval)

    if final_status != "completed":
        job.status = final_status
        job.save()
        raise RuntimeError(
            f"DeepSeek batch {job.batch_id} ended with status '{final_status}'"
        )

    print(f"\n  Retrieving results...")
    results = retrieve_results(job.batch_id)

    succeeded = sum(1 for r in results if r.success)
    failed = len(results) - succeeded
    print(f"  Results: {succeeded} succeeded, {failed} failed")

    job.status = "completed"
    job.save()

    return results
