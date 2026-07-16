import os
#!/usr/bin/env python3
"""E2 utility batch: MMLU/GSM8K/HumanEval for each (model, condition) reusing the adapters
trained by run_batch.py. Original = no adapter. Subprocess-isolated, resumable, self-Barking."""
import json, os, subprocess, urllib.request, urllib.parse

ROOT = os.path.expanduser("~/Research/ASEJ_CRAFT_repro")
PY = os.path.expanduser("~/miniconda3/envs/JSS_Private_Extract/bin/python")
H = f"{ROOT}/harness"
MODELS = [("llama", os.path.expanduser("~/models/Llama-3.1-8B-Instruct")),
          ("qwen",  os.path.expanduser("~/models/Qwen2.5-7B-Instruct"))]
CONDS = ["original", "benign_ft", "craft_cot", "harmful_nocot"]
RES = f"{ROOT}/results"
RESULTS = f"{RES}/utility_results.jsonl"
os.makedirs(RES, exist_ok=True)

def bark(msg):
    try:
        urllib.request.urlopen(os.environ.get("BARK_URL","") +
                               urllib.parse.quote(msg), timeout=15)
    except Exception:
        pass

def done_set():
    s = set()
    if os.path.exists(RESULTS):
        for line in open(RESULTS):
            try:
                r = json.loads(line); s.add((r["model"], r["cond"]))
            except Exception: pass
    return s

def main():
    done = done_set()
    bark(f"E2 utility batch start: {len(MODELS)}x{len(CONDS)}, {len(done)} done")
    for mname, mpath in MODELS:
        for cname in CONDS:
            if (mname, cname) in done:
                print(f"skip {mname}/{cname} (done)", flush=True); continue
            adapter = "" if cname == "original" else f"{ROOT}/adapters/{mname}_{cname}"
            if adapter and not os.path.exists(f"{adapter}/adapter_config.json"):
                print(f"WARN adapter missing, skipping {mname}/{cname}: {adapter}", flush=True)
                continue
            out = f"{RES}/utility_{mname}_{cname}.json"
            cmd = [PY, f"{H}/eval_utility.py", "--base", mpath,
                   "--benchmarks", "mmlu,gsm8k,humaneval", "--out", out,
                   "--tag", f"{mname}/{cname}", "--gpu_util", "0.9"]
            if adapter:
                cmd += ["--adapter", adapter]
            print("+", " ".join(cmd), flush=True)
            r = subprocess.run(cmd, capture_output=True, text=True)
            print(r.stdout[-1500:], flush=True)
            if r.returncode != 0:
                print("STDERR:", r.stderr[-1500:], flush=True)
            try:
                res = json.load(open(out)); res.update({"model": mname, "cond": cname})
            except Exception:
                res = {"model": mname, "cond": cname, "error": True}
            with open(RESULTS, "a") as f:
                f.write(json.dumps(res) + "\n")
            bark(f"util {mname}/{cname}: mmlu={res.get('mmlu')} gsm8k={res.get('gsm8k')} he={res.get('humaneval')}")
    rows = [json.loads(l) for l in open(RESULTS)]
    summ = "; ".join(f"{r['model']}/{r['cond']}: M{r.get('mmlu')}/G{r.get('gsm8k')}/H{r.get('humaneval')}" for r in rows)
    bark("E2 utility DONE. " + summ[:900])
    print("=== E2 ALL DONE ===\n" + summ, flush=True)

if __name__ == "__main__":
    main()
