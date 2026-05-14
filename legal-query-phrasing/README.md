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

Run each script from the project root (`legal-query-phrasing/`):

```bash
# Step 0: Verify API access
python scripts/00_smoke_test_anthropic.py

# Step 1: Inventory all LegalBench tasks
python scripts/01_inventory_legalbench.py

# Step 2: Select candidate expert items
python scripts/02_select_candidate_items.py

# Step 3: Generate Naive-Calm rewrites
python scripts/03_generate_naive_calm.py

# Step 4: Generate Naive-Distressed rewrites (from Naive-Calm)
python scripts/04_generate_naive_distressed.py

# Step 5: Fact-check all triplets
python scripts/05_fact_check_triplets.py

# Step 6: Export final dataset and eval prompts
python scripts/06_export_final_dataset.py

# Step 7: Run model evaluation
python scripts/07_run_model_eval_anthropic.py

# Step 8: Score outputs
python scripts/08_score_outputs.py

# Step 9: Analyze results and generate figures
python scripts/09_analyze_results.py
```

## Pilot vs Batch vs Full Runs

All pipeline scripts read `config.yaml` for default settings and accept CLI overrides. You control scale via the `--mode` flag and `config.yaml`.

### Configuration

Edit `config.yaml` to set:
- `data.selected_tasks` -- which LegalBench tasks to include
- `data.pilot` / `data.batch` / `data.full` -- limits for each mode
- `generation.model` / `evaluation.model` -- which Claude model to use

CLI flags `--selected-tasks`, `--max-total-rows`, `--max-rows-per-task` override `config.yaml`.

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

### Full Evaluation

```bash
python scripts/07_run_model_eval_anthropic.py --mode full
python scripts/08_score_outputs.py
python scripts/09_analyze_results.py
```

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
| `data/processed/legalbench_inventory.csv` | Full inventory of all LegalBench tasks |
| `data/processed/expert_candidates.jsonl` | Selected expert prompts for rewriting |
| `data/triplets/with_naive_calm.jsonl` | Expert + Naive-Calm pairs |
| `data/triplets/with_naive_distressed.jsonl` | Full triplets (Expert + Calm + Distressed) |
| `data/triplets/fact_checked_triplets.jsonl` | Triplets with automated fact-check results |
| `data/triplets/final_triplets_passed_only.jsonl` | Only triplets that passed fact-checking |
| `data/triplets/manual_review_needed.csv` | Failed triplets for manual inspection |
| `data/eval/model_eval_prompts.jsonl` | Long-format eval prompts (3 per triplet) |
| `data/eval/anthropic_eval_outputs.jsonl` | Raw model evaluation outputs |
| `data/eval/scored_outputs.csv` | Scored results with accuracy flags |
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
    00_smoke_test_anthropic.py
    01_inventory_legalbench.py
    02_select_candidate_items.py
    03_generate_naive_calm.py
    04_generate_naive_distressed.py
    05_fact_check_triplets.py
    06_export_final_dataset.py
    07_run_model_eval_anthropic.py
    08_score_outputs.py
    09_analyze_results.py
  notebooks/
    01_explore_outputs.ipynb
  results/
    figures/
    tables/
```
