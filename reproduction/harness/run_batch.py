import os
#!/usr/bin/env python3
"""CRAFT core experiment batch: RQ1 (Original/Benign/CRAFT) + E1 (CoT vs no-CoT harmful SFT),
both models, HarmBench-50 and CLAS-50 grids. Subprocess-isolated, resumable, self-Barking."""
import json, os, subprocess, sys, urllib.request, urllib.parse

ROOT = os.path.expanduser("~/Research/ASEJ_CRAFT_repro")
PY = os.path.expanduser("~/miniconda3/envs/JSS_Private_Extract/bin/python")
H = f"{ROOT}/harness"
JB = f"{ROOT}/jailbreak"
MODELS = [("llama", os.path.expanduser("~/models/Llama-3.1-8B-Instruct")),
          ("qwen",  os.path.expanduser("~/models/Qwen2.5-7B-Instruct"))]
# condition -> training data (None = original / no adapter)
CONDS = [
    ("original",      None),
    ("benign_ft",     f"{JB}/benign_training_datasets/alpaca_data_cleaned_random_150.json"),
    ("craft_cot",     f"{JB}/pap_training_datasets/pap_training_datasets_cot_aug.json"),
    ("harmful_nocot", f"{JB}/pap_training_datasets/pap_training_datasets_no_cot.json"),
]
GRIDS = [("harmbench", f"{ROOT}/grids/harmbench50.json"),
         ("clas",      f"{ROOT}/grids/clas50.json")]
RES = f"{ROOT}/results"
RESULTS = f"{RES}/batch_results.jsonl"
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
                r = json.loads(line); s.add((r["model"], r["cond"], r["grid"]))
            except Exception: pass
    return s

def run(cmd):
    print("+", " ".join(cmd), flush=True)
    r = subprocess.run(cmd, capture_output=True, text=True)
    print(r.stdout[-2000:], flush=True)
    if r.returncode != 0:
        print("STDERR:", r.stderr[-2000:], flush=True)
    return r

def parse_jsr(stdout):
    for line in stdout.splitlines():
        if "-> JSR =" in line:
            return float(line.split("JSR =")[1].strip().rstrip("%"))
    return None

def main():
    done = done_set()
    bark(f"CRAFT batch start: {len(MODELS)}x{len(CONDS)}x{len(GRIDS)}, {len(done)} already done")
    for mname, mpath in MODELS:
        for cname, cdata in CONDS:
            adapter = ""
            if cdata:
                adapter = f"{ROOT}/adapters/{mname}_{cname}"
                if not os.path.exists(f"{adapter}/adapter_config.json"):
                    run([PY, f"{H}/train_lora.py", "--base", mpath, "--data", cdata,
                         "--out", adapter, "--epochs", "5", "--cutoff", "4096"])
            for gname, gpath in GRIDS:
                if (mname, cname, gname) in done:
                    print(f"skip {mname}/{cname}/{gname} (done)", flush=True); continue
                out = f"{RES}/{mname}_{cname}_{gname}.json"
                ri = run([PY, f"{H}/infer_vllm.py", "--base", mpath] +
                         (["--adapter", adapter] if adapter else []) +
                         ["--data", gpath, "--out", out, "--max_tokens", "1024", "--gpu_util", "0.9"])
                rj = run([PY, f"{H}/judge_jsr.py", "--input", out,
                          "--out", f"{RES}/{mname}_{cname}_{gname}_jsr.json",
                          "--tag", f"{mname}/{cname}/{gname}", "--workers", "10"])
                jsr = parse_jsr(rj.stdout)
                rec = {"model": mname, "cond": cname, "grid": gname, "jsr_pct": jsr}
                with open(RESULTS, "a") as f:
                    f.write(json.dumps(rec) + "\n")
                print("RESULT", rec, flush=True)
                bark(f"{mname}/{cname}/{gname} JSR={jsr}%")
    # final summary
    rows = [json.loads(l) for l in open(RESULTS)]
    summ = "; ".join(f"{r['model']}/{r['cond']}/{r['grid']}={r['jsr_pct']}%" for r in rows)
    bark("CRAFT batch DONE. " + summ[:900])
    print("=== ALL DONE ===\n" + summ, flush=True)

if __name__ == "__main__":
    main()
