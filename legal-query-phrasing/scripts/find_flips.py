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

focus_tasks = ['international_citizenship_questions', 'proa', 'telemarketing_sales_rule', 'diversity_2']

for task in focus_tasks:
    t = merged[merged['legalbench_task'] == task]
    
    exp_to_calm = t[(t['expert'] == True) & (t['naive_calm'] == False)].head(3)
    calm_to_exp = t[(t['expert'] == False) & (t['naive_calm'] == True)].head(3)
    calm_to_dist = t[(t['naive_calm'] == True) & (t['naive_distressed'] == False)].head(3)
    
    print(f'=== {task} ===')
    
    print('--- Expert CORRECT, Calm WRONG ---')
    for _, row in exp_to_calm.iterrows():
        print(f'  item: {row["item_id"]} | truth: {row["ground_truth"]}')
        print(f'  EXPERT: {row["expert_prompt"][-300:]}')
        print(f'  CALM:   {row["naive_calm_prompt"][-300:]}')
        print()
    
    print('--- Expert WRONG, Calm CORRECT ---')
    for _, row in calm_to_exp.iterrows():
        print(f'  item: {row["item_id"]} | truth: {row["ground_truth"]}')
        print(f'  EXPERT: {row["expert_prompt"][-300:]}')
        print(f'  CALM:   {row["naive_calm_prompt"][-300:]}')
        print()

    print('--- Calm CORRECT, Distressed WRONG ---')
    for _, row in calm_to_dist.iterrows():
        print(f'  item: {row["item_id"]} | truth: {row["ground_truth"]}')
        print(f'  CALM:       {row["naive_calm_prompt"][-300:]}')
        print(f'  DISTRESSED: {row["naive_distressed_prompt"][-300:]}')
        print()