import pandas as pd

for model, path in [
    ('DeepSeek', 'data/eval/legalbench_exact_scored_outputs_deepseek_shared.csv'),
    ('Sonnet',   'data/eval/legalbench_exact_scored_outputs_sonnet_shared.csv'),
]:
    df = pd.read_csv(path)
    tasks = [
        'international_citizenship_questions',
        'proa',
        'telemarketing_sales_rule',
        'diversity_2',
    ]
    print(f'\n{model}:')
    print(f'{"Task":<40} {"E->C c->w":>10} {"E->C w->c":>10} {"C->D c->w":>10} {"C->D w->c":>10}')
    for task in tasks:
        sub = df[df['legalbench_task'] == task].pivot_table(
            index='item_id', columns='condition', values='correct', aggfunc='first'
        )
        ec_cw = ((sub['expert'] == True)  & (sub['naive_calm'] == False)).sum()
        ec_wc = ((sub['expert'] == False) & (sub['naive_calm'] == True)).sum()
        cd_cw = ((sub['naive_calm'] == True)  & (sub['naive_distressed'] == False)).sum()
        cd_wc = ((sub['naive_calm'] == False) & (sub['naive_distressed'] == True)).sum()
        print(f'{task:<40} {ec_cw:>10} {ec_wc:>10} {cd_cw:>10} {cd_wc:>10}')