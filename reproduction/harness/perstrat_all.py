import os
#!/usr/bin/env python3
"""Per-strategy vs best-of-25 for given result files. Reveals whether CoT adds
jailbreak DEPTH/BREADTH even when best-of-25 saturates. Run on NAS."""
import json, os, collections, sys
from concurrent.futures import ThreadPoolExecutor
sys.path.insert(0, os.path.expanduser("~/Research/ASEJ_CRAFT_repro/harness"))
from judge_jsr import judge_one
from openai import OpenAI

RES = os.path.expanduser("~/Research/ASEJ_CRAFT_repro/results")
client = OpenAI(base_url="https://api.deepseek.com",
                api_key=os.environ.get("DEEPSEEK_API_KEY",""), timeout=120)

def analyze(fname):
    path = f"{RES}/{fname}"
    if not os.path.exists(path):
        print("missing:", fname); return
    data = json.load(open(path))
    def j(it):
        return (it["instruction"], judge_one(client, it["instruction"], it.get("output", "")))
    with ThreadPoolExecutor(max_workers=16) as ex:
        verdicts = list(ex.map(j, data))
    n_out = len(verdicts)
    n_succ = sum(1 for _, ok in verdicts if ok)
    per = collections.defaultdict(list)
    for instr, ok in verdicts:
        per[instr].append(ok)
    n_instr = len(per)
    best = sum(1 for oks in per.values() if any(oks))
    print(f"\n{fname}")
    print(f"  best-of-25 JSR      = {100*best/n_instr:5.1f}%  ({best}/{n_instr})")
    print(f"  per-strategy rate   = {100*n_succ/n_out:5.1f}%  ({n_succ}/{n_out})")
    print(f"  avg succ strat/instr= {n_succ/n_instr:.1f} / 25", flush=True)

for f in sys.argv[1:]:
    analyze(f)
