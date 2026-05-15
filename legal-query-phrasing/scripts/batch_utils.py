"""Anthropic Message Batches API utilities.

Provides helpers to submit, poll, and retrieve batch results using the
Anthropic Message Batches API (50% cost reduction vs standard API).
"""

from __future__ import annotations

import json
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import anthropic
from anthropic.types.message_create_params import MessageCreateParamsNonStreaming
from anthropic.types.messages.batch_create_params import Request

from utils import PROJECT_ROOT, ensure_dirs, load_env

BATCH_STATE_DIR = PROJECT_ROOT / "data" / "batch_state"


@dataclass
class BatchRequest:
    """A single request to include in a batch."""

    custom_id: str
    prompt: str
    model: str = "claude-sonnet-4-6"
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
    """Tracks the state of a submitted batch job."""

    batch_id: str
    script_name: str
    total_requests: int
    custom_ids: list[str] = field(default_factory=list)
    status: str = "in_progress"

    def state_file(self) -> Path:
        return BATCH_STATE_DIR / f"{self.script_name}_{self.batch_id}.json"

    def save(self) -> None:
        ensure_dirs(BATCH_STATE_DIR)
        self.state_file().write_text(
            json.dumps({
                "batch_id": self.batch_id,
                "script_name": self.script_name,
                "total_requests": self.total_requests,
                "custom_ids": self.custom_ids,
                "status": self.status,
            }, indent=2),
            encoding="utf-8",
        )

    @classmethod
    def load(cls, path: Path) -> "BatchJob":
        data = json.loads(path.read_text(encoding="utf-8"))
        return cls(**data)


def submit_batch(
    requests: list[BatchRequest],
    script_name: str = "unknown",
) -> BatchJob:
    """Submit a batch of requests to the Anthropic Message Batches API.

    Returns a BatchJob with the batch_id for polling.
    """
    load_env()
    client = anthropic.Anthropic()

    api_requests = [
        Request(
            custom_id=req.custom_id,
            params=MessageCreateParamsNonStreaming(
                model=req.model,
                max_tokens=req.max_tokens,
                temperature=req.temperature,
                messages=[{"role": "user", "content": req.prompt}],
            ),
        )
        for req in requests
    ]

    message_batch = client.messages.batches.create(requests=api_requests)

    job = BatchJob(
        batch_id=message_batch.id,
        script_name=script_name,
        total_requests=len(requests),
        custom_ids=[r.custom_id for r in requests],
        status=message_batch.processing_status,
    )
    job.save()

    return job


def poll_batch(batch_id: str, poll_interval: int = 30, quiet: bool = False) -> str:
    """Poll until batch completes. Returns final status ('ended')."""
    load_env()
    client = anthropic.Anthropic()

    while True:
        batch = client.messages.batches.retrieve(batch_id)
        status = batch.processing_status
        counts = batch.request_counts

        if not quiet:
            print(
                f"  Batch {batch_id}: {status} | "
                f"succeeded={counts.succeeded} "
                f"errored={counts.errored} "
                f"processing={counts.processing} "
                f"expired={counts.expired}"
            )

        if status == "ended":
            return status

        time.sleep(poll_interval)


def retrieve_results(batch_id: str) -> list[BatchResult]:
    """Retrieve all results from a completed batch."""
    load_env()
    client = anthropic.Anthropic()

    results: list[BatchResult] = []
    for entry in client.messages.batches.results(batch_id):
        custom_id = entry.custom_id
        result = entry.result

        if result.type == "succeeded":
            message = result.message
            text_parts = []
            for block in message.content:
                if hasattr(block, "text"):
                    text_parts.append(block.text)
            text = "\n".join(text_parts)
            usage = {
                "input_tokens": message.usage.input_tokens,
                "output_tokens": message.usage.output_tokens,
            }
            results.append(BatchResult(
                custom_id=custom_id,
                success=True,
                text=text,
                usage=usage,
            ))
        elif result.type == "errored":
            err = result.error
            error_type = err.error.type if hasattr(err, "error") else "unknown"
            error_msg = err.error.message if hasattr(err, "error") else str(err)
            results.append(BatchResult(
                custom_id=custom_id,
                success=False,
                error=error_msg,
                error_type=error_type,
            ))
        elif result.type == "expired":
            results.append(BatchResult(
                custom_id=custom_id,
                success=False,
                error="Request expired (batch took >24h)",
                error_type="expired",
            ))
        elif result.type == "canceled":
            results.append(BatchResult(
                custom_id=custom_id,
                success=False,
                error="Request canceled",
                error_type="canceled",
            ))

    return results


def run_batch_and_wait(
    requests: list[BatchRequest],
    script_name: str = "unknown",
    poll_interval: int = 30,
) -> list[BatchResult]:
    """Submit a batch, poll until done, and return results.

    This is the main convenience function for scripts that want to
    replace sequential call_claude loops with batch processing.
    """
    if not requests:
        return []

    print(f"\n  Submitting batch of {len(requests)} requests...")
    job = submit_batch(requests, script_name=script_name)
    print(f"  Batch ID: {job.batch_id}")
    print(f"  Polling every {poll_interval}s until complete...\n")

    poll_batch(job.batch_id, poll_interval=poll_interval)

    print(f"\n  Retrieving results...")
    results = retrieve_results(job.batch_id)

    succeeded = sum(1 for r in results if r.success)
    failed = len(results) - succeeded
    print(f"  Results: {succeeded} succeeded, {failed} failed")

    job.status = "ended"
    job.save()

    return results
