import pandas as pd

df = pd.read_csv('data/eval/legalbench_exact_scored_outputs.csv')
prompts = pd.read_json('data/eval/legalbench_exact_eval_prompts.jsonl', lines=True)

# Extract just the last question from each prompt
def extract_last_question(prompt):
    # Split on newlines, find the last non-empty chunk after the final "Question:" or "Clause:"
    parts = prompt.replace('\\n', '\n').strip().split('\n')
    # Find the last Question/Clause block
    last_q_start = 0
    for i, line in enumerate(parts):
        if line.strip().startswith('Question:') or line.strip().startswith('Clause:') or line.strip().startswith('Q:'):
            last_q_start = i
    return '\n'.join(parts[last_q_start:]).strip()

prompts['scenario'] = prompts['final_prompt'].apply(extract_last_question)

pivot = df.pivot_table(index=['item_id', 'legalbench_task', 'ground_truth'],
                        columns='condition',
                        values='correct',
                        aggfunc='first').reset_index()
pivot.columns.name = None

scenario_wide = prompts.pivot_table(index=['item_id', 'legalbench_task'],
                                     columns='condition',
                                     values='scenario',
                                     aggfunc='first').reset_index()
scenario_wide.columns = ['item_id', 'legalbench_task',
                          'expert_scenario', 'naive_calm_scenario', 'naive_distressed_scenario']

merged = pivot.merge(scenario_wide, on=['item_id', 'legalbench_task'])

def classify_flip(row):
    e = row['expert']
    c = row['naive_calm']
    d = row['naive_distressed']
    flips = []
    if e == True and c == False:
        flips.append('expert_correct_calm_wrong')
    if e == False and c == True:
        flips.append('expert_wrong_calm_correct')
    if c == True and d == False:
        flips.append('calm_correct_distress_wrong')
    if c == False and d == True:
        flips.append('calm_wrong_distress_correct')
    if e == False and c == False and d == True:
        flips.append('distress_only_correct')
    return '|'.join(flips) if flips else None

merged['flip_type'] = merged.apply(classify_flip, axis=1)

flips = merged[merged['flip_type'].notna()].copy()

out = flips[[
    'item_id', 'legalbench_task', 'ground_truth',
    'expert', 'naive_calm', 'naive_distressed',
    'flip_type',
    'expert_scenario', 'naive_calm_scenario', 'naive_distressed_scenario'
]].sort_values(['legalbench_task', 'flip_type', 'item_id'])

out.to_csv('results/flips.csv', index=False)
print(f'Saved {len(out)} flipping items to results/flips.csv')
print()
print(out.groupby(['legalbench_task', 'flip_type']).size().to_string())