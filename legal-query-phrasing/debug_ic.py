import json, re
from collections import Counter

def legalbench_normalize(text):
    if not text: return ''
    s = text.lower().strip()
    s = re.sub(r'[^\w\s]', '', s)
    return s.strip()

rows = [json.loads(l) for l in open('data/eval/legalbench_exact_eval_outputs_sonnet.jsonl', encoding='utf-8')]

tasks = sorted(set(r.get('legalbench_task') for r in rows))

for task in tasks:
    for condition in ['expert', 'naive_calm', 'naive_distressed']:
        subset = [r for r in rows if r.get('legalbench_task') == task and r.get('condition') == condition]
        if not subset:
            continue
        answers = Counter(r.get('parsed_answer', '')[:50] for r in subset)
        clean = sum(1 for a in answers if legalbench_normalize(a).split()[:1] in [['yes'], ['no']])
        total_clean = sum(v for a, v in answers.items() if legalbench_normalize(a).split()[:1] in [['yes'], ['no']])
        verbose = len(subset) - total_clean
        if verbose > 5:
            print(f"\n{task} / {condition}  ({verbose}/{len(subset)} verbose)")
            for ans, cnt in answers.most_common(5):
                flag = '  <-- verbose' if legalbench_normalize(ans).split()[:1] not in [['yes'], ['no']] else ''
                print(f"  {cnt:3d}x  {repr(ans)}{flag}")