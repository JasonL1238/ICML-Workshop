"""OpenAI Batch API utilities.

Provides helpers to submit, poll, and retrieve batch results using the
OpenAI Batch API (50% cost reduction vs synchronous API).

The OpenAI Batch API is file-based: upload a JSONL of requests, create a
batch, poll until complete, then download the output JSONL.
"""

from __future__ import annotations

import json
import tempfile
import time
from dataclasses import dataclass, field
from pathlib import Path

import openai

from utils import PROJECT_ROOT, ensure_dirs, load_env, require_openai_key

BATCH_STATE_DIR = PROJECT_ROOT / "data" / "batch_state"


@dataclass
class BatchRequest:
    """A single request to include in an OpenAI batch."""

    custom_id: str
    prompt: str
    model: str = "gpt-5.4"
    temperature: float = 0.0
    max_tokens: int = 700


@dataclass
class BatchResult:
    """Result for a single request in a completed batch."""

    custom_id: str
    success: bool
    text: str | None = None
    usage: dict | None = None
    error: str | None = None
    error_type: str | None = None


@dataclass
class BatchJob:
    """Tracks the state of a submitted OpenAI batch job."""

    batch_id: str
    script_name: str
    total_requests: int
    input_file_id: str = ""
    custom_ids: list[str] = field(default_factory=list)
    status: str = "validating"

    def state_file(self) -> Path:
        return BATCH_STATE_DIR / f"{self.script_name}_{self.batch_id}.json"

    def save(self) -> None:
        ensure_dirs(BATCH_STATE_DIR)
        self.state_file().write_text(
            json.dumps({
                "batch_id": self.batch_id,
                "script_name": self.script_name,
                "total_requests": self.total_requests,
                "input_file_id": self.input_file_id,
                "custom_ids": self.custom_ids,
                "status": self.status,
            }, indent=2),
            encoding="utf-8",
        )

    @classmethod
    def load(cls, path: Path) -> "BatchJob":
        data = json.loads(path.read_text(encoding="utf-8"))
        return cls(**data)


def _build_request_line(req: BatchRequest) -> dict:
    """Build a single JSONL line for the OpenAI Batch API input file."""
    return {
        "custom_id": req.custom_id,
        "method": "POST",
        "url": "/v1/chat/completions",
        "body": {
            "model": req.model,
            "temperature": req.temperature,
            "max_tokens": req.max_tokens,
            "messages": [{"role": "user", "content": req.prompt}],
        },
    }


def submit_batch(
    requests: list[BatchRequest],
    script_name: str = "unknown",
) -> BatchJob:
    """Submit a batch of requests to the OpenAI Batch API.

    Writes requests to a temp JSONL file, uploads it, and creates the batch.
    Returns a BatchJob with the batch_id for polling.
    """
    env = load_env()
    require_openai_key(env)
    client = openai.OpenAI()

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
    require_openai_key(env)
    client = openai.OpenAI()

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
    """Retrieve all results from a completed batch."""
    env = load_env()
    require_openai_key(env)
    client = openai.OpenAI()

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


def run_openai_batch_and_wait(
    requests: list[BatchRequest],
    script_name: str = "unknown",
    poll_interval: int = 30,
) -> list[BatchResult]:
    """Submit a batch, poll until done, and return results.

    Main convenience function matching the Anthropic batch_utils interface.
    """
    if not requests:
        return []

    print(f"\n  Submitting OpenAI batch of {len(requests)} requests...")
    job = submit_batch(requests, script_name=script_name)
    print(f"  Batch ID: {job.batch_id}")
    print(f"  Input file ID: {job.input_file_id}")
    print(f"  Polling every {poll_interval}s until complete...\n")

    final_status = poll_batch(job.batch_id, poll_interval=poll_interval)

    if final_status != "completed":
        job.status = final_status
        job.save()
        raise RuntimeError(
            f"OpenAI batch {job.batch_id} ended with status '{final_status}'"
        )

    print(f"\n  Retrieving results...")
    results = retrieve_results(job.batch_id)

    succeeded = sum(1 for r in results if r.success)
    failed = len(results) - succeeded
    print(f"  Results: {succeeded} succeeded, {failed} failed")

    job.status = "completed"
    job.save()

    return results
