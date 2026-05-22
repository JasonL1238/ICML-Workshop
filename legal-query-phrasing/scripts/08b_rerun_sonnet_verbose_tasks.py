#!/usr/bin/env python3
"""Rerun international_citizenship_questions with a Yes/No system prompt."""
import json, sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent))
from utils import call_claude, load_env, append_jsonl, read_jsonl

TASKS = {"international_citizenship_questions"}
SYSTEM_PROMPT = "You must answer with only the word Yes or No. Do not explain."

input_path  = Path("data/eval/legalbench_exact_eval_prompts.jsonl")
output_path = Path("data/eval/legalbench_exact_eval_outputs_sonnet.jsonl")

load_env()
existing = {r["eval_id"] for r in read_jsonl(output_path)}

rows = [r for r in read_jsonl(input_path) if r.get("legalbench_task") in TASKS]
print(f"Rerunning {len(rows)} rows for {TASKS}")

for row in rows:
    # Overwrite existing entry by writing a new one — scorer takes last seen
    # Actually we need to replace, so load all, patch, rewrite
    pass

# Simpler: write to a separate patch file, merge after
patch_path = Path("data/eval/legalbench_exact_eval_outputs_sonnet_patch.jsonl")

done = {r["eval_id"] for r in read_jsonl(patch_path)} if patch_path.exists() else set()
pending = [r for r in rows if r["eval_id"] not in done]
print(f"Pending: {len(pending)}")

from tqdm import tqdm
for row in tqdm(pending):
    try:
        text, usage = call_claude(
            row["final_prompt"],
            model="claude-sonnet-4-6",
            temperature=0.0,
            max_tokens=10,
            system_prompt=SYSTEM_PROMPT,
        )
        out = {**{k: row.get(k) for k in ["eval_id","item_id","legalbench_task","domain","condition","ground_truth","final_prompt"]},
               "eval_model": "claude-sonnet-4-6", "temperature": 0.0,
               "model_raw_output": text, "parsed_answer": text.strip().splitlines()[0].strip(),
               "token_usage": usage, "eval_error": None}
    except Exception as e:
        out = {**{k: row.get(k) for k in ["eval_id","item_id","legalbench_task","domain","condition","ground_truth","final_prompt"]},
               "eval_model": "claude-sonnet-4-6", "temperature": 0.0,
               "model_raw_output": None, "parsed_answer": "", "token_usage": None, "eval_error": str(e)}
    append_jsonl(patch_path, out)

print(f"Patch written: {patch_path}")

# Merge: original file with patch overrides
all_rows = read_jsonl(output_path)
patch_rows = {r["eval_id"]: r for r in read_jsonl(patch_path)}
merged = [patch_rows.get(r["eval_id"], r) for r in all_rows]

merged_path = Path("data/eval/legalbench_exact_eval_outputs_sonnet_fixed.jsonl")
with open(merged_path, "w", encoding="utf-8") as f:
    for r in merged:
        f.write(json.dumps(r) + "\n")
print(f"Merged output: {merged_path} ({len(merged)} rows)")