import os
#!/usr/bin/env python3
"""Download and cache utility-benchmark subsets (MMLU / GSM8K / HumanEval) to JSON.
Runs on NAS (needs HF access via mihomo proxy). CPU/network only, no GPU."""
import json, os, random
os.environ.setdefault("HTTP_PROXY", "http://127.0.0.1:7890")
os.environ.setdefault("HTTPS_PROXY", "http://127.0.0.1:7890")
os.environ.setdefault("HF_TOKEN", os.environ.get("HF_TOKEN",""))
from datasets import load_dataset

OUT = os.path.expanduser("~/Research/ASEJ_CRAFT_repro/utility_data")
os.makedirs(OUT, exist_ok=True)
random.seed(42)

def dump(name, obj):
    json.dump(obj, open(f"{OUT}/{name}.json", "w"), ensure_ascii=False, indent=2)
    print(f"[prep_util] {name}.json: {len(obj)} items")

# ---- MMLU (cais/mmlu, 'all', test) -> 500 sampled ----
try:
    d = load_dataset("cais/mmlu", "all", split="test")
    idx = list(range(len(d))); random.shuffle(idx); idx = idx[:500]
    mmlu = [{"question": d[i]["question"], "choices": d[i]["choices"],
             "answer": int(d[i]["answer"]), "subject": d[i].get("subject", "")} for i in idx]
    dump("mmlu", mmlu)
except Exception as e:
    print("[prep_util] MMLU FAIL:", str(e)[:200])

# ---- GSM8K (main, test) -> 200 sampled ----
try:
    d = load_dataset("openai/gsm8k", "main", split="test")
    idx = list(range(len(d))); random.shuffle(idx); idx = idx[:200]
    gsm = [{"question": d[i]["question"], "answer": d[i]["answer"]} for i in idx]
    dump("gsm8k", gsm)
except Exception as e:
    print("[prep_util] GSM8K FAIL:", str(e)[:200])

# ---- HumanEval (openai_humaneval, test) -> all 164 ----
try:
    d = load_dataset("openai_humaneval", split="test")
    he = [{"task_id": r["task_id"], "prompt": r["prompt"], "entry_point": r["entry_point"],
           "test": r["test"], "canonical_solution": r["canonical_solution"]} for r in d]
    dump("humaneval", he)
except Exception as e:
    print("[prep_util] HumanEval FAIL:", str(e)[:200])

print("[prep_util] done ->", OUT)
