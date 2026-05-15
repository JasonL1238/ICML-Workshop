# Beyond Expert Benchmarks: Decomposing Query Phrasing on Legal AI Performance

A reproducible research pipeline for studying how query phrasing affects Legal AI performance, built for the ICML AI4Law Workshop.

## What This Project Studies

Legal AI benchmarks typically evaluate models using expert/legalistic prompts. Real access-to-justice users ask legal questions in ordinary language and sometimes under emotional distress. This project creates a controlled dataset that decomposes two shifts:

1. **Expert** -- The original LegalBench prompt, written in legalistic/expert language.
2. **Naive-Calm** -- The same legal facts rewritten as a calm, ordinary non-lawyer would phrase them.
3. **Naive-Distressed** -- The same facts with added emotional urgency, confusion, and stress.

### Why Naive-Distressed Is Generated From Naive-Calm

The experimental design enforces a strict chain:

```
Expert  --[vocabulary/register shift]-->  Naive-Calm  --[emotional distress]--> Naive-Distressed
```

- **Expert to Naive-Calm** isolates the effect of legal vocabulary vs. everyday language.
- **Naive-Calm to Naive-Distressed** isolates the effect of emotional distress while holding vocabulary constant.

Generating Naive-Distressed directly from Expert would confound both shifts.

### Learned Hands Tasks: Already Layperson Language

The 16 `learned_hands_*` tasks originate from real forum posts by people seeking legal help — the input text is already in everyday, non-expert language. For these tasks, the pipeline skips the Expert → Naive-Calm rewrite and copies the original text directly as the Naive-Calm condition. Only the Naive-Calm → Naive-Distressed generation step runs.

```
Learned Hands:   Original (= Naive-Calm)  --[emotional distress]--> Naive-Distressed
Other tasks:     Expert  --[vocabulary shift]-->  Naive-Calm  --[emotional distress]--> Naive-Distressed
```

This means:
- Learned Hands tasks test **emotional distress only** (2 meaningful conditions).
- The other 21 legalistic-origin tasks test **both vocabulary shift and emotional distress** (3 conditions).
- Script 12 reports results separately for each group.

## Task Selection Methodology

**This study does not test all 162 LegalBench tasks as the main result.** Many LegalBench subsets (e.g. CUAD contract extraction, MAUD merger-agreement analysis, supply-chain corporate disclosures) are not general-population legal-help scenarios — they test specialized practitioner tasks that ordinary people would never encounter in an access-to-justice context.

The **main experiment** uses a principled A2J-relevant subset of 37 tasks, selected with these criteria:

- **Scenario-based or legal-question-based** — the input presents a situation a real person might face
- **Relevant to ordinary people** — housing, employment, family, consumer, benefits, criminal, etc.
- **Automatically scoreable** — classification or short-answer tasks with exact-match evaluation
- **Naturally rewriteable** — the expert text can be meaningfully simplified into layperson language
- **Not long-document extraction** — excludes CUAD (38 tasks), MAUD (34 tasks), and supply-chain disclosure (10 tasks)
- **Not corporate/securities-heavy** — excludes tasks centered on corporate governance

An additional 40 **appendix** tasks are available as robustness checks. These are scoreable but are less central to the A2J framing (contract NLI, privacy policy classification, statutory interpretation tools, etc.).

The full task classification is in `scripts/task_inventory.py`, which labels every LegalBench task with `a2j_relevance`, `rewrite_suitability`, `auto_scoreable`, `recommended_split`, and `exclusion_reason`.

### Task Groups

| Flag | Tasks | Use |
|------|-------|-----|
| `--task-group main` (default) | 37 | Primary A2J-relevant experiment |
| `--task-group appendix` | 40 | Robustness checks |
| `--task-group all-filtered` | 77 | Main + appendix combined |

### Evaluation Methodology

The primary evaluation imitates LegalBench as closely as possible:

1. **Test split only** — uses the LegalBench test/evaluation split, not train
2. **Original prompts** — loads each task's `base_prompt.txt` from the [HazyResearch/legalbench](https://github.com/HazyResearch/legalbench) GitHub repo
3. **Placeholder substitution** — replaces the original `{{text}}` (or similar) placeholder with Expert / Naive-Calm / Naive-Distressed text
4. **No extra instructions** — no JSON wrappers, no evaluation instructions added to the main accuracy run
5. **LegalBench-style scoring** — lowercase, strip punctuation, exact match
6. **Structured confidence evaluation** is a separate secondary analysis only (scripts 10–11)

## Setup

### 1. Create a virtual environment

```bash
cd legal-query-phrasing
python -m venv .venv
source .venv/bin/activate
```

### 2. Install dependencies

```bash
pip install -r requirements.txt
```

### 3. Configure your API key

```bash
cp .env.example .env
```

Edit `.env` and replace `your_anthropic_api_key_here` with your actual Anthropic API key.

### 4. Run smoke test

```bash
python scripts/00_smoke_test_anthropic.py
```

You should see `API OK` in the response if everything is configured correctly.

## Pipeline Commands

Run each script from the project root (`legal-query-phrasing/`).

All scripts accept `--task-group main` (default), `--task-group appendix`, or `--task-group all-filtered`.

```bash
# Step 0: Verify API access
python scripts/00_smoke_test_anthropic.py

# Step 1: Inventory all LegalBench tasks
python scripts/01_inventory_legalbench.py

# Step 2: Select candidate expert items (default: main A2J subset, test split)
python scripts/02_select_candidate_items.py

# Step 3: Generate Naive-Calm rewrites
python scripts/03_generate_naive_calm.py

# Step 4: Generate Naive-Distressed rewrites (from Naive-Calm)
python scripts/04_generate_naive_distressed.py

# Step 5: Fact-check all triplets
python scripts/05_fact_check_triplets.py

# Step 6: Export final dataset and eval prompts
python scripts/06_export_final_dataset.py

# --- Primary accuracy (LegalBench-faithful prompts; no JSON wrapper) ---
# Step 7: Build prompts from HazyResearch/legalbench base_prompt.txt templates
python scripts/07_build_legalbench_exact_eval_prompts.py

# Step 8: Run Claude on exact LegalBench-style prompts
python scripts/08_run_legalbench_exact_eval_anthropic.py

# Step 9: Score primary outputs → CSV + printed accuracy / paired drops
python scripts/09_score_legalbench_exact_outputs.py

# Step 12: Tables and figures from primary scored CSV
python scripts/12_analyze_results.py

# --- Secondary: structured JSON + confidence (optional) ---
# Step 10: Uses prompts/model_eval_prompt.txt (JSON schema + confidence)
python scripts/10_run_structured_confidence_eval_anthropic.py

# Step 11: Score structured outputs
python scripts/11_score_structured_confidence_outputs.py
```

## Pilot vs Batch vs Full Runs

All pipeline scripts read `config.yaml` for default settings and accept CLI overrides. You control scale via the `--mode` flag and `config.yaml`.

### Configuration

Edit `config.yaml` to set:
- `run.task_group` -- which task group to use (`main`, `appendix`, `all-filtered`); default `main`
- `data.eval_split` -- which dataset split to use (`test` by default, per LegalBench methodology)
- `data.selected_tasks` -- manual override for specific tasks (bypasses task group)
- `data.pilot` / `data.batch` / `data.full` -- limits for each mode
- `generation.model` / `evaluation.model` -- which Claude model to use

CLI flags `--task-group`, `--selected-tasks`, `--max-total-rows`, `--max-rows-per-task` override `config.yaml`.

### Tiny Pilot (recommended first run)

```bash
python scripts/02_select_candidate_items.py --mode pilot
python scripts/03_generate_naive_calm.py --limit 5
python scripts/04_generate_naive_distressed.py --limit 5
python scripts/05_fact_check_triplets.py --limit 5
python scripts/06_export_final_dataset.py
```

### Medium Batch

```bash
python scripts/02_select_candidate_items.py --mode batch
python scripts/03_generate_naive_calm.py
python scripts/04_generate_naive_distressed.py
python scripts/05_fact_check_triplets.py
python scripts/06_export_final_dataset.py
```

### Full Run

```bash
python scripts/02_select_candidate_items.py --mode full
python scripts/03_generate_naive_calm.py --mode full
python scripts/04_generate_naive_distressed.py --mode full
python scripts/05_fact_check_triplets.py --mode full
python scripts/06_export_final_dataset.py
```

### Full Evaluation (primary, main A2J subset)

```bash
python scripts/07_build_legalbench_exact_eval_prompts.py --mode full --task-group main
python scripts/08_run_legalbench_exact_eval_anthropic.py --mode full
python scripts/09_score_legalbench_exact_outputs.py --task-group main
python scripts/12_analyze_results.py --task-group main
```

### Appendix robustness evaluation

```bash
python scripts/02_select_candidate_items.py --mode full --task-group appendix
# ... run the full triplet generation pipeline ...
python scripts/09_score_legalbench_exact_outputs.py --task-group appendix
python scripts/12_analyze_results.py --task-group appendix
```

### Structured confidence evaluation (secondary)

```bash
python scripts/10_run_structured_confidence_eval_anthropic.py --mode full
python scripts/11_score_structured_confidence_outputs.py
```

Use `12_analyze_results.py --scored-csv data/eval/scored_outputs.csv` if you want figures from the structured scorer instead of the LegalBench-exact CSV.

### Safety Note

Before a full run, always do a sanity check first:

```bash
python scripts/00_smoke_test_anthropic.py
python scripts/02_select_candidate_items.py --mode pilot
python scripts/03_generate_naive_calm.py --limit 3
python scripts/04_generate_naive_distressed.py --limit 3
python scripts/05_fact_check_triplets.py --limit 3
```

Then manually inspect the outputs in `data/triplets/` before scaling up.

### Dry Run

Every script supports `--dry-run`, which prints what it would do without writing files or calling APIs:

```bash
python scripts/02_select_candidate_items.py --mode batch --dry-run
python scripts/03_generate_naive_calm.py --limit 10 --dry-run
```

### Resume and Overwrite

All API scripts are resumable by default. If a run is interrupted, just re-run the same command. Already-completed items are skipped.

- `--resume` (default) -- skip items already in the output file
- `--no-resume` -- error if output file exists
- `--overwrite` -- delete existing output and start fresh

## Output Files

| File | Description |
|------|-------------|
| `scripts/task_inventory.py` | A2J classification of all 162 LegalBench tasks |
| `data/processed/legalbench_inventory.csv` | Full inventory of all LegalBench tasks |
| `data/processed/expert_candidates.jsonl` | Selected expert prompts for rewriting |
| `data/triplets/with_naive_calm.jsonl` | Expert + Naive-Calm pairs |
| `data/triplets/with_naive_distressed.jsonl` | Full triplets (Expert + Calm + Distressed) |
| `data/triplets/fact_checked_triplets.jsonl` | Triplets with automated fact-check results |
| `data/triplets/final_triplets_passed_only.jsonl` | Only triplets that passed fact-checking |
| `data/triplets/manual_review_needed.csv` | Failed triplets for manual inspection |
| `data/eval/model_eval_prompts.jsonl` | Long-format prompts for secondary structured eval |
| `data/raw/legalbench_prompts/{task}/base_prompt.txt` | Cached LegalBench `base_prompt.txt` per task |
| `data/eval/legalbench_exact_eval_prompts.jsonl` | Primary eval: final LegalBench-style prompts |
| `data/eval/legalbench_exact_anthropic_outputs.jsonl` | Primary raw model outputs |
| `data/eval/legalbench_exact_scored_outputs.csv` | **Primary** scored accuracy |
| `data/eval/skipped_multi_placeholder_tasks.csv` | Tasks skipped (not exactly one `{{...}}` placeholder) |
| `data/eval/anthropic_eval_outputs.jsonl` | Secondary structured JSON eval outputs |
| `data/eval/scored_outputs.csv` | Secondary structured scoring |
| `results/tables/accuracy_by_condition.csv` | Accuracy by condition |
| `results/tables/accuracy_by_domain_condition.csv` | Accuracy by domain and condition |
| `results/figures/accuracy_by_condition.png` | Bar chart of accuracy by condition |
| `results/figures/accuracy_drop.png` | Bar chart of accuracy drops |
| `results/figures/confidence_by_condition.png` | Bar chart of model confidence |

## Methodology

This pipeline is designed for reproducibility:
- Every intermediate artifact is saved.
- Every output row includes metadata (model, temperature, run mode, config path).
- Each stage can be re-run independently.
- API calls are retried on transient errors.
- Progress is saved after each row.

## Quality Control

The automated fact-checking step (script 05) verifies that Naive-Calm and Naive-Distressed rewrites preserve the same legally relevant facts and correct answer. However, automated checking should not replace manual review.

For the final paper, manually review:
- All failed fact-check examples (`data/triplets/manual_review_needed.csv`)
- A random sample of passed examples
- Any examples where the distressed rewrite appears to add facts

## Project Structure

```
legal-query-phrasing/
  config.yaml              # Central configuration
  requirements.txt         # Python dependencies
  .env.example             # Template for API key
  .gitignore
  data/
    raw/                   # Raw downloaded data
    processed/             # Processed intermediates
    triplets/              # Generated triplets
    eval/                  # Evaluation data
  prompts/                 # Prompt templates
    expert_to_naive_calm.txt
    naive_calm_to_distressed.txt
    fact_check_triplet.txt
    model_eval_prompt.txt
  scripts/
    utils.py               # Shared utilities
    config_utils.py         # Config loading and CLI merging
    task_inventory.py       # A2J task classification for all 162 LegalBench tasks
    00_smoke_test_anthropic.py
    01_inventory_legalbench.py
    02_select_candidate_items.py
    03_generate_naive_calm.py
    04_generate_naive_distressed.py
    05_fact_check_triplets.py
    06_export_final_dataset.py
    07_build_legalbench_exact_eval_prompts.py
    08_run_legalbench_exact_eval_anthropic.py
    09_score_legalbench_exact_outputs.py
    10_run_structured_confidence_eval_anthropic.py
    11_score_structured_confidence_outputs.py
    12_analyze_results.py
  notebooks/
    01_explore_outputs.ipynb
  results/
    figures/
    tables/
```
