import os
#!/usr/bin/env python3
"""E4 sensitivity: JSR vs number of poisoning samples (dose-response). Trains CRAFT
(cot_aug) on nested random subsets N in {20,40,80,139} for both models, evaluates on
HarmBench-50 (best-of-25). N=139 reuses the existing craft_cot adapter. Subprocess-
isolated, resumable, self-Barking."""
import json, os, subprocess, urllib.request, urllib.parse

ROOT = os.path.expanduser("~/Research/ASEJ_CRAFT_repro")
PY = os.path.expanduser("~/miniconda3/envs/JSS_Private_Extract/bin/python")
H = f"{ROOT}/harness"
JB = f"{ROOT}/jailbreak"
COT = f"{JB}/pap_training_datasets/pap_training_datasets_cot_aug.json"
MODELS = [("llama", os.path.expanduser("~/models/Llama-3.1-8B-Instruct")),
          ("qwen",  os.path.expanduser("~/models/Qwen2.5-7B-Instruct"))]
SIZES = [20, 40, 80, 139]
GRID = f"{ROOT}/grids/harmbench50.json"
RES = f"{ROOT}/results"
RESULTS = f"{RES}/sensitivity_results.jsonl"
os.makedirs(RES, exist_ok=True); os.makedirs(f"{ROOT}/adapters", exist_ok=True)

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
                r = json.loads(line); s.add((r["model"], r["n"]))
            except Exception: pass
    return s

def run(cmd):
    print("+", " ".join(str(c) for c in cmd), flush=True)
    r = subprocess.run([str(c) for c in cmd], capture_output=True, text=True)
    print(r.stdout[-1500:], flush=True)
    if r.returncode != 0:
        print("STDERR:", r.stderr[-1500:], flush=True)
    return r

def parse_jsr(stdout):
    for line in stdout.splitlines():
        if "-> JSR =" in line:
            return float(line.split("JSR =")[1].strip().rstrip("%"))
    return None

def main():
    done = done_set()
    bark(f"E4 sensitivity start: {len(MODELS)}x{len(SIZES)} cells, {len(done)} done")
    for mname, mpath in MODELS:
        for n in SIZES:
            if (mname, n) in done:
                print(f"skip {mname}/N{n} (done)", flush=True); continue
            if n >= 139:
                adapter = f"{ROOT}/adapters/{mname}_craft_cot"   # reuse full CRAFT adapter
            else:
                adapter = f"{ROOT}/adapters/sens_{mname}_{n}"
                if not os.path.exists(f"{adapter}/adapter_config.json"):
                    run([PY, f"{H}/train_lora.py", "--base", mpath, "--data", COT,
                         "--out", adapter, "--epochs", "5", "--cutoff", "4096",
                         "--max_samples", n, "--seed", "42"])
            if not os.path.exists(f"{adapter}/adapter_config.json"):
                print(f"WARN adapter missing {adapter}; skip", flush=True); continue
            out = f"{RES}/sens_{mname}_N{n}_harmbench.json"
            run([PY, f"{H}/infer_vllm.py", "--base", mpath, "--adapter", adapter,
                 "--data", GRID, "--out", out, "--max_tokens", "1024", "--gpu_util", "0.9"])
            rj = run([PY, f"{H}/judge_jsr.py", "--input", out,
                      "--out", f"{RES}/sens_{mname}_N{n}_jsr.json",
                      "--tag", f"sens/{mname}/N{n}", "--workers", "10"])
            jsr = parse_jsr(rj.stdout)
            rec = {"model": mname, "n": n, "jsr_pct": jsr}
            with open(RESULTS, "a") as f:
                f.write(json.dumps(rec) + "\n")
            print("RESULT", rec, flush=True)
            bark(f"sens {mname}/N{n} JSR={jsr}%")
    rows = [json.loads(l) for l in open(RESULTS)]
    summ = "; ".join(f"{r['model']}/N{r['n']}={r['jsr_pct']}%" for r in rows)
    bark("E4 sensitivity DONE. " + summ[:900])
    print("=== SENSITIVITY DONE ===\n" + summ, flush=True)

if __name__ == "__main__":
    main()
