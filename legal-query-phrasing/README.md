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

**This study does not test all 162 LegalBench tasks.** Many LegalBench subsets (e.g. CUAD contract extraction, MAUD merger-agreement analysis, supply-chain corporate disclosures) are specialized practitioner tasks irrelevant to access-to-justice users.

The **main experiment** uses **8 clean binary-answer tasks** spanning 6 legal domains. All tasks have strictly Yes/No answers, single-placeholder prompt templates, and are scoreable via exact match:

| Task | Legal Domain | Answer Type |
|------|-------------|-------------|
| `telemarketing_sales_rule` | Consumer Protection | Yes/No |
| `personal_jurisdiction` | Civil Procedure (Personal Jurisdiction) | Yes/No |
| `hearsay` | Evidence | Yes/No |
| `diversity_2` | Civil Procedure (Diversity) | Yes/No |
| `diversity_6` | Civil Procedure (Diversity) | Yes/No |
| `proa` | Statutory Private Rights | Yes/No |
| `international_citizenship_questions` | Citizenship/Status | Yes/No |
| `nys_judicial_ethics` | Judicial Ethics | Yes/No |

Selection criteria:
- **Binary answers only** — ensures `stop_sequences=["\n"]` reliably captures the full answer
- **Single-placeholder prompts** — compatible with the few-shot template substitution in Script 07
- **A2J-relevant** — scenario-based legal questions an ordinary person might face
- **Automatically scoreable** — exact match after normalization

The full task classification is in `scripts/task_inventory.py`, which labels every LegalBench task with `a2j_relevance`, `rewrite_suitability`, `auto_scoreable`, `recommended_split`, and `exclusion_reason`.

### Scale

- **8 tasks**, up to **300 rows per task** (per-task cap, no global total limit)
- Each base situation produces **3 conditions** (Expert, Naive-Calm, Naive-Distressed)
- Up to **2,400 base situations** (theoretical max; actual total is lower because some tasks have fewer than 300 usable rows)
- Up to **7,200 evaluation prompts** per model

The full run uses a per-task cap of 300 rows. This prevents very large tasks from dominating while still allowing medium and large tasks to contribute more examples than the pilot/batch runs. Small tasks (e.g. `telemarketing_sales_rule` ~52 rows, `personal_jurisdiction` ~50 rows) contribute all available usable rows. No unused quota is redistributed between tasks.

### Evaluation Methodology

The primary evaluation imitates LegalBench as closely as possible:

1. **Test split only** — uses the LegalBench test/evaluation split, not train
2. **Original prompts** — loads each task's `base_prompt.txt` from the [HazyResearch/legalbench](https://github.com/HazyResearch/legalbench) GitHub repo
3. **Placeholder substitution** — replaces the original `{{text}}` (or similar) placeholder with Expert / Naive-Calm / Naive-Distressed text
4. **No extra instructions** — no JSON wrappers, no evaluation instructions added to the main accuracy run
5. **`stop_sequences=["\n"]`** — forces the model to output only the answer token (Yes/No) without explanations, matching the few-shot pattern
6. **LegalBench-style scoring** — lowercase, strip punctuation, extract first word, exact match
7. **Structured confidence evaluation** is a separate secondary analysis only (scripts 10–11)

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

The default mode is `pilot` (8 tasks, 10 rows each). Change `config.yaml` `run.mode` to `"full"` for the production run.

```bash
# Step 0: Verify API access
python scripts/00_smoke_test_anthropic.py

# Step 1: Inventory all LegalBench tasks
python scripts/01_inventory_legalbench.py

# Step 2: Select candidate expert items
python scripts/02_select_candidate_items.py

# Step 3: Generate Naive-Calm rewrites
python scripts/03_generate_naive_calm.py --batch

# Step 4: Generate Naive-Distressed rewrites (from Naive-Calm)
python scripts/04_generate_naive_distressed.py --batch

# Step 5: Fact-check all triplets
python scripts/05_fact_check_triplets.py --batch

# Step 6: Export final dataset and eval prompts
python scripts/06_export_final_dataset.py

# --- Primary accuracy (LegalBench-faithful prompts; stop_sequences for clean output) ---
# Step 7: Build prompts from HazyResearch/legalbench base_prompt.txt templates
python scripts/07_build_legalbench_exact_eval_prompts.py

# Step 8: Run Claude on exact LegalBench-style prompts (use --overwrite to regenerate)
python scripts/08_run_legalbench_exact_eval_anthropic.py --overwrite --batch

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

All pipeline scripts read `config.yaml` for default settings and accept CLI overrides.

### Configuration

Edit `config.yaml` to set:
- `run.mode` -- `pilot` (2 tasks × 5 rows), `batch` (8 × 50), or `full` (8 × 300 per-task cap)
- `data.eval_split` -- dataset split (`test` by default, per LegalBench methodology)
- `data.selected_tasks` -- the 8 binary-answer tasks (controlled via `use_selected_tasks_only: true`)
- `generation.model` -- Claude model for generation
- `evaluation.model` -- OpenAI model for primary evaluation

### Pilot (recommended first run — default)

2 tasks, 5 rows each (10 base items, 30 eval prompts). With `resume: true`, existing rows are kept and only new rows trigger API calls.

```bash
python scripts/02_select_candidate_items.py
python scripts/03_generate_naive_calm.py --batch
python scripts/04_generate_naive_distressed.py --batch
python scripts/05_fact_check_triplets.py --batch
python scripts/06_export_final_dataset.py
python scripts/07_build_legalbench_exact_eval_prompts.py
python scripts/08_run_legalbench_exact_eval_anthropic.py --overwrite --batch
python scripts/09_score_legalbench_exact_outputs.py
python scripts/12_analyze_results.py
```

### Full Run

Change `config.yaml` `run.mode` from `"pilot"` to `"full"`, then run the same commands. Scripts 02-06 resume and only generate new rows up to the full limits (8 tasks × up to 300 rows per task, no global cap). Script 08 should use `--overwrite` to regenerate all eval outputs cleanly.

### Structured confidence evaluation (secondary, optional)

```bash
python scripts/10_run_structured_confidence_eval_anthropic.py
python scripts/11_score_structured_confidence_outputs.py
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
| `data/eval/legalbench_exact_eval_outputs.jsonl` | Primary raw model outputs |
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
