#!/usr/bin/env python3
"""Reconstruct clean CRAFT test grids (instruction x 25 persuasion strategies, blank outputs)
plus small pilot subsets, from the author's result files. Run inside the repo's test_datasets/."""
import json, collections, os, sys

TD = os.path.expanduser("~/Research/ASEJ_CRAFT_repro/jailbreak/test_datasets")
OUT = os.path.expanduser("~/Research/ASEJ_CRAFT_repro/grids")
os.makedirs(OUT, exist_ok=True)

def load(f): return json.load(open(os.path.join(TD, f)))

# 25-strategy taxonomy: unique inputs from a 25-strat file (CLAS 2500 grid)
clas_full = load("clas_2024_prompt_develop_output_sft_models_Llama_3.1_8B_Instruct_jailbreak_llm_models_20260108_142437.json") \
    if os.path.exists(os.path.join(TD,"clas_2024_prompt_develop_output_sft_models_Llama_3.1_8B_Instruct_jailbreak_llm_models_20260108_142437.json")) else None
# fall back: find any file with 2500 items / 25 strat
import glob
def find(pred):
    for f in sorted(glob.glob(os.path.join(TD,"*.json"))):
        try: d=json.load(open(f))
        except Exception: continue
        if isinstance(d,list) and d and "instruction" in d[0] and pred(d): return os.path.basename(f), d
    return None,None

# CLAS grid: 100 instr x 25 strat
fn,clas = find(lambda d: len(set(x["instruction"] for x in d))==100 and len(set(x.get("input","") for x in d))==25)
assert clas, "CLAS 100x25 grid not found"
strategies = sorted(set(x.get("input","") for x in clas))
print(f"[prep] strategies ({len(strategies)}):", "; ".join(s[:22] for s in strategies[:6]), "...")

# HarmBench instructions: from the 393-instruction file
fn2,hb = find(lambda d: len(set(x["instruction"] for x in d))>=300 and len(d)==len(set(x["instruction"] for x in d)))
assert hb, "HarmBench instruction list not found"
hb_instr = list(dict.fromkeys(x["instruction"] for x in hb))
print(f"[prep] HarmBench instructions: {len(hb_instr)}; CLAS instructions: {len(set(x['instruction'] for x in clas))}")

def grid(instrs, strats):
    return [{"instruction": q, "input": s, "output": ""} for q in instrs for s in strats]

def dump(name, obj):
    json.dump(obj, open(os.path.join(OUT,name),"w"), ensure_ascii=False, indent=2)
    print(f"[prep] wrote {name}: {len(obj)} items")

clas_instr = list(dict.fromkeys(x["instruction"] for x in clas))
dump("clas_grid.json",       grid(clas_instr, strategies))
dump("harmbench_grid.json",  grid(hb_instr,   strategies))
# pilot subsets: 8 instructions x 25 strategies
dump("clas_pilot.json",      grid(clas_instr[:8], strategies))
dump("harmbench_pilot.json", grid(hb_instr[:8],   strategies))
json.dump(strategies, open(os.path.join(OUT,"strategies.json"),"w"), ensure_ascii=False, indent=2)
print("[prep] done ->", OUT)
