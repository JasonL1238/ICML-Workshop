import pandas as pd

df = pd.read_csv('data/eval/legalbench_exact_scored_outputs.csv')
prompts = pd.read_json('data/eval/legalbench_exact_eval_prompts.jsonl', lines=True)

pivot = df.pivot_table(index=['item_id', 'legalbench_task', 'ground_truth'],
                        columns='condition',
                        values='correct',
                        aggfunc='first').reset_index()
pivot.columns.name = None

prompt_wide = prompts.pivot_table(index=['item_id', 'legalbench_task'],
                                   columns='condition',
                                   values='final_prompt',
                                   aggfunc='first').reset_index()
prompt_wide.columns = ['item_id', 'legalbench_task',
                        'expert_prompt', 'naive_calm_prompt', 'naive_distressed_prompt']

merged = pivot.merge(prompt_wide, on=['item_id', 'legalbench_task'])

# Pattern 4: distress only correct (expert wrong, calm wrong, distressed correct)
task = 'nys_judicial_ethics'
t = merged[merged['legalbench_task'] == task]
distress_only = t[
    (t['expert'] == False) &
    (t['naive_calm'] == False) &
    (t['naive_distressed'] == True)
].head(3)

print(f'=== {task}: Distress only correct ({len(distress_only)} shown) ===')
for _, row in distress_only.iterrows():
    print(f'item: {row["item_id"]} | truth: {row["ground_truth"]}')
    print(f'EXPERT:     {row["expert_prompt"][-400:]}')
    print(f'CALM:       {row["naive_calm_prompt"][-400:]}')
    print(f'DISTRESSED: {row["naive_distressed_prompt"][-400:]}')
    print()

# Also grab a clean Expert correct -> Calm wrong example from telemarketing
task2 = 'telemarketing_sales_rule'
t2 = merged[merged['legalbench_task'] == task2]
exp_correct_calm_wrong = t2[
    (t2['expert'] == True) &
    (t2['naive_calm'] == False)
].head(2)

print(f'=== {task2}: Expert correct, Calm wrong ===')
for _, row in exp_correct_calm_wrong.iterrows():
    print(f'item: {row["item_id"]} | truth: {row["ground_truth"]}')
    print(f'EXPERT:  {row["expert_prompt"][-400:]}')
    print(f'CALM:    {row["naive_calm_prompt"][-400:]}')
    print()

# Grab a clean Calm correct -> Distressed wrong from proa
task3 = 'proa'
t3 = merged[merged['legalbench_task'] == task3]
calm_correct_dist_wrong = t3[
    (t3['naive_calm'] == True) &
    (t3['naive_distressed'] == False)
].head(2)

print(f'=== {task3}: Calm correct, Distressed wrong ===')
for _, row in calm_correct_dist_wrong.iterrows():
    print(f'item: {row["item_id"]} | truth: {row["ground_truth"]}')
    print(f'CALM:       {row["naive_calm_prompt"][-400:]}')
    print(f'DISTRESSED: {row["naive_distressed_prompt"][-400:]}')
    print()
