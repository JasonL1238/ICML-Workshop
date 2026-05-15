"""Central configuration loading and CLI override merging."""

from __future__ import annotations

import argparse
from copy import deepcopy
from pathlib import Path
from typing import Any

import yaml

PROJECT_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_CONFIG = PROJECT_ROOT / "config.yaml"

VALID_TASK_GROUPS = ("main", "appendix", "all-filtered")


# ---------------------------------------------------------------------------
# Load & validate
# ---------------------------------------------------------------------------

def load_config(config_path: str | Path = DEFAULT_CONFIG) -> dict:
    """Load config.yaml and attach the source path for metadata."""
    config_path = Path(config_path)
    if not config_path.exists():
        raise FileNotFoundError(f"Config file not found: {config_path}")
    with open(config_path, "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)
    for key in ("run", "data", "generation", "evaluation", "files"):
        if key not in cfg:
            raise KeyError(f"Config missing required top-level key: {key}")
    cfg["_config_path"] = str(config_path.resolve())
    return cfg


# ---------------------------------------------------------------------------
# CLI argument builder (shared across scripts)
# ---------------------------------------------------------------------------

def add_common_args(parser: argparse.ArgumentParser) -> None:
    """Add the common CLI arguments every pipeline script supports."""
    parser.add_argument("--config", default=str(DEFAULT_CONFIG), help="Path to config.yaml")
    parser.add_argument("--mode", choices=["pilot", "batch", "full"], help="Run mode override")
    parser.add_argument("--max-total-rows", type=int, help="Override max total rows")
    parser.add_argument("--max-rows-per-task", type=int, help="Override max rows per task")
    parser.add_argument("--selected-tasks", nargs="+", help="Override selected tasks list")
    parser.add_argument(
        "--task-group",
        choices=list(VALID_TASK_GROUPS),
        default=None,
        help="Task group: main (A2J-relevant subset, default), appendix, all-filtered",
    )
    parser.add_argument("--resume", action="store_true", default=None, help="Resume from existing output")
    parser.add_argument("--no-resume", action="store_true", default=None, help="Disable resume")
    parser.add_argument("--overwrite", action="store_true", default=False, help="Overwrite existing output")
    parser.add_argument("--dry-run", action="store_true", default=False, help="Print plan without writing or calling APIs")
    parser.add_argument("--limit", type=int, default=None, help="Limit number of rows to process in this run")


# ---------------------------------------------------------------------------
# Merge CLI overrides into config
# ---------------------------------------------------------------------------

def merge_cli_overrides(config: dict, args: argparse.Namespace) -> dict:
    """Apply CLI overrides on top of config. CLI wins over config.yaml."""
    cfg = deepcopy(config)

    if args.mode is not None:
        cfg["run"]["mode"] = args.mode

    if getattr(args, "task_group", None) is not None:
        cfg["run"]["task_group"] = args.task_group

    if args.resume is True:
        cfg["run"]["resume"] = True
    elif getattr(args, "no_resume", None) is True:
        cfg["run"]["resume"] = False

    if args.overwrite:
        cfg["run"]["overwrite"] = True

    if args.selected_tasks is not None:
        cfg["data"]["selected_tasks"] = args.selected_tasks
        cfg["_explicit_selected_tasks"] = True

    mode = cfg["run"]["mode"]
    mode_cfg = cfg["data"].get(mode, {})

    if args.max_total_rows is not None:
        mode_cfg["max_total_rows"] = args.max_total_rows
        cfg["data"][mode] = mode_cfg

    if args.max_rows_per_task is not None:
        mode_cfg["max_rows_per_task"] = args.max_rows_per_task
        cfg["data"][mode] = mode_cfg

    if hasattr(args, "limit") and args.limit is not None:
        cfg["_cli_limit"] = args.limit

    cfg["_dry_run"] = getattr(args, "dry_run", False)

    return cfg


# ---------------------------------------------------------------------------
# Derived helpers
# ---------------------------------------------------------------------------

def get_run_limits(config: dict) -> dict[str, Any]:
    """Return resolved run limits based on mode.

    Returns {mode, max_tasks, max_rows_per_task, max_total_rows, use_all_tasks}.
    """
    mode = config["run"]["mode"]
    mode_cfg = config["data"].get(mode, {})
    return {
        "mode": mode,
        "max_tasks": mode_cfg.get("max_tasks"),
        "max_rows_per_task": mode_cfg.get("max_rows_per_task"),
        "max_total_rows": mode_cfg.get("max_total_rows"),
        "use_all_tasks": mode_cfg.get("use_all_tasks", False),
    }


def get_task_group(config: dict) -> str:
    """Return the active task group (default: 'main')."""
    return config.get("run", {}).get("task_group", "main")


def get_eval_split(config: dict) -> str:
    """Return the preferred dataset split for evaluation (default: 'test')."""
    return config.get("data", {}).get("eval_split", "test")


def get_selected_tasks(config: dict) -> list[str]:
    """Return the task list for the current run.

    Resolution order:
      1. Explicit --selected-tasks on CLI  →  use those verbatim
      2. --task-group (or config run.task_group)  →  resolve via task_inventory
      3. Legacy full-mode use_all_tasks  →  empty list (caller uses inventory)
      4. data.selected_tasks from config.yaml
    """
    if config.get("_explicit_selected_tasks"):
        return list(config["data"].get("selected_tasks", []))

    from task_inventory import get_tasks_for_group

    group = get_task_group(config)
    try:
        return get_tasks_for_group(group)
    except ValueError:
        pass

    limits = get_run_limits(config)
    if limits["mode"] == "full" and limits["use_all_tasks"]:
        return []
    return list(config["data"].get("selected_tasks", []))


def should_resume(config: dict) -> bool:
    return config["run"].get("resume", True)


def should_overwrite(config: dict) -> bool:
    return config["run"].get("overwrite", False)


def is_dry_run(config: dict) -> bool:
    return config.get("_dry_run", False)


def get_limit(config: dict) -> int | None:
    return config.get("_cli_limit")


def resolve_file(config: dict, key: str) -> Path:
    """Resolve a path from config.files relative to PROJECT_ROOT."""
    return PROJECT_ROOT / config["files"][key]


def print_run_header(config: dict, output_path: Path | str) -> None:
    """Print a standard run-configuration summary."""
    limits = get_run_limits(config)
    tasks = get_selected_tasks(config)
    group = get_task_group(config)
    eval_split = get_eval_split(config)
    print("=" * 60)
    print(f"  Run mode:           {limits['mode']}")
    print(f"  Task group:         {group}")
    print(f"  Eval split:         {eval_split}")
    print(f"  Selected tasks:     {len(tasks)} task(s)")
    print(f"  Max tasks:          {limits['max_tasks']}")
    print(f"  Max rows per task:  {limits['max_rows_per_task']}")
    print(f"  Max total rows:     {limits['max_total_rows']}")
    print(f"  Resume:             {should_resume(config)}")
    print(f"  Overwrite:          {should_overwrite(config)}")
    print(f"  Dry run:            {is_dry_run(config)}")
    print(f"  Output:             {output_path}")
    cli_limit = get_limit(config)
    if cli_limit is not None:
        print(f"  CLI --limit:        {cli_limit}")
    print("=" * 60)


def run_metadata(config: dict) -> dict:
    """Return a metadata dict to embed in every output row."""
    limits = get_run_limits(config)
    return {
        "run_mode": limits["mode"],
        "task_group": get_task_group(config),
        "eval_split": get_eval_split(config),
        "config_path": config.get("_config_path", ""),
        "selected_tasks": get_selected_tasks(config),
        "max_rows_per_task": limits["max_rows_per_task"],
        "max_total_rows": limits["max_total_rows"],
    }
