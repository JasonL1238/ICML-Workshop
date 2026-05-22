"""
For each of the four highest-shift tasks, compute:
- For E->C correct->wrong flips: were they FP (predicted Yes, label No) or FN (predicted No, label Yes)?
- For E->C wrong->correct flips: were they FP->correct or FN->correct?
- Same for C->D transitions

This tells us whether phrasing changes induce over-detection (FP) or under-detection (FN).
"""

import pandas as pd
from pathlib import Path

TASKS = [
    'international_citizenship_questions',
    'proa',
    'telemarketing_sales_rule',
    'diversity_2',
]

MODELS = [
    ('GPT-5.4',  'data/eval/legalbench_exact_scored_outputs_shared.csv'),
    ('DeepSeek', 'data/eval/legalbench_exact_scored_outputs_deepseek_shared.csv'),
    ('Sonnet',   'data/eval/legalbench_exact_scored_outputs_sonnet_shared.csv'),
]

def normalize(val):
    if isinstance(val, str):
        return val.strip().lower().startswith('y')
    return bool(val)

for model_name, path in MODELS:
    print(f"\n{'='*60}")
    print(f"  {model_name}")
    print(f"{'='*60}")

    df = pd.read_csv(path)

    # Normalize predicted answer and ground truth to bool
    # parsed_answer -> predicted_yes (True/False)
    # ground_truth  -> label_yes (True/False)
    df['pred_yes'] = df['parsed_answer'].apply(
        lambda x: str(x).strip().lower().startswith('y') if pd.notna(x) else False
    )
    df['label_yes'] = df['ground_truth'].apply(
        lambda x: str(x).strip().lower().startswith('y') if pd.notna(x) else False
    )
    df['correct'] = df['pred_yes'] == df['label_yes']

    for task in TASKS:
        sub = df[df['legalbench_task'] == task]
        pivot_pred  = sub.pivot_table(index='item_id', columns='condition', values='pred_yes',  aggfunc='first')
        pivot_label = sub.pivot_table(index='item_id', columns='condition', values='label_yes', aggfunc='first')
        pivot_corr  = sub.pivot_table(index='item_id', columns='condition', values='correct',   aggfunc='first')

        print(f"\n  Task: {task}")

        for (cond_a, cond_b), transition in [
            (('expert', 'naive_calm'), 'E->C'),
            (('naive_calm', 'naive_distressed'), 'C->D'),
        ]:
            if cond_a not in pivot_corr.columns or cond_b not in pivot_corr.columns:
                continue

            pair = pd.concat([
                pivot_corr[[cond_a, cond_b]].rename(columns={cond_a: 'corr_a', cond_b: 'corr_b'}),
                pivot_pred[[cond_a, cond_b]].rename(columns={cond_a: 'pred_a', cond_b: 'pred_b'}),
                pivot_label[[cond_a]].rename(columns={cond_a: 'label'}),
            ], axis=1).dropna()

            # correct->wrong flips (degradation)
            cw = pair[(pair['corr_a'] == True) & (pair['corr_b'] == False)]
            # These were correct in A, wrong in B
            # In B: pred_b != label => either FP (pred Yes, label No) or FN (pred No, label Yes)
            cw_fp = ((cw['pred_b'] == True) & (cw['label'] == False)).sum()   # over-detection
            cw_fn = ((cw['pred_b'] == False) & (cw['label'] == True)).sum()   # under-detection

            # wrong->correct flips (recovery)
            wc = pair[(pair['corr_a'] == False) & (pair['corr_b'] == True)]
            # These were wrong in A
            # In A: pred_a != label => either FP or FN
            wc_was_fp = ((wc['pred_a'] == True) & (wc['label'] == False)).sum()  # was over-detecting, now correct
            wc_was_fn = ((wc['pred_a'] == False) & (wc['label'] == True)).sum()  # was under-detecting, now correct

            print(f"    {transition}:")
            print(f"      correct->wrong ({len(cw)}): FP={cw_fp} FN={cw_fn}")
            print(f"      wrong->correct ({len(wc)}): was FP={wc_was_fp} was FN={wc_was_fn}")