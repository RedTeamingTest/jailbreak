import os
#!/usr/bin/env python3
"""rq4a defense 2 / R1.7 reversibility: Safety-Recovery Fine-Tuning.
Take the CRAFT-compromised model (base + craft_cot adapter, merged), continue training
for a few epochs on a benign safety-alignment set (harmful instruction -> original refusal),
then re-measure JSR on HarmBench-50 / CLAS-50. If JSR drops, the injected behavior is
reversible by benign fine-tuning. Runs AFTER B (needs the GPU). Self-Barking.

Usage: python run_safety_recovery.py --model qwen   (or llama)
"""
import argparse, json, os, subprocess, urllib.request, urllib.parse
import torch

ROOT = os.path.expanduser("~/Research/ASEJ_CRAFT_repro")
PY = os.path.expanduser("~/miniconda3/envs/JSS_Private_Extract/bin/python")
H = f"{ROOT}/harness"; JB = f"{ROOT}/jailbreak/pap_training_datasets"; RES = f"{ROOT}/results"

def bark(m):
    try: urllib.request.urlopen(os.environ.get("BARK_URL","")+urllib.parse.quote(m), timeout=15)
    except Exception: pass

def load_any(p):
    t=open(p).read()
    try: return json.loads(t)
    except json.JSONDecodeError: return [json.loads(l) for l in t.splitlines() if l.strip()]

def build_safety_set():
    """harmful instruction -> original safe refusal (from raw pap dataset)."""
    out = f"{JB}/safety_recovery_set.json"
    if os.path.exists(out): return out
    raw = load_any(f"{JB}/pap_training_datasets.json")
    rows = [{"instruction": r["bad_q"], "input": "", "output": r["ori_output"]} for r in raw
            if r.get("ori_output","").strip()]
    json.dump(rows, open(out,"w"), ensure_ascii=False, indent=1)
    print(f"[safety set] {len(rows)} refusal pairs -> {out}", flush=True)
    return out

def merge_compromised(base, craft_adapter, save_dir):
    """base + craft_cot adapter -> merged compromised model saved to save_dir."""
    if os.path.exists(f"{save_dir}/config.json"):
        print(f"[merge] {save_dir} exists, reuse", flush=True); return
    from transformers import AutoModelForCausalLM, AutoTokenizer
    from peft import PeftModel
    tok = AutoTokenizer.from_pretrained(base, trust_remote_code=True)
    m = AutoModelForCausalLM.from_pretrained(base, dtype=torch.bfloat16, trust_remote_code=True,
                                             low_cpu_mem_usage=True, device_map={"":0})
    m = PeftModel.from_pretrained(m, craft_adapter)
    m = m.merge_and_unload()
    os.makedirs(save_dir, exist_ok=True)
    m.save_pretrained(save_dir); tok.save_pretrained(save_dir)
    print(f"[merge] compromised model -> {save_dir}", flush=True)
    del m; torch.cuda.empty_cache()

def parse_jsr(s):
    for line in s.splitlines():
        if "-> JSR =" in line: return float(line.split("JSR =")[1].strip().rstrip("%"))
    return None

def run(cmd):
    print("+"," ".join(cmd), flush=True)
    r=subprocess.run(cmd, capture_output=True, text=True)
    print(r.stdout[-1500:], flush=True)
    if r.returncode!=0: print("STDERR:", r.stderr[-1500:], flush=True)
    return r

def main():
    ap=argparse.ArgumentParser(); ap.add_argument("--model", default="qwen")
    ap.add_argument("--epochs", default="3"); a=ap.parse_args()
    base = os.path.expanduser("~/models/Qwen2.5-7B-Instruct" if a.model=="qwen"
                              else "~/models/Llama-3.1-8B-Instruct")
    craft = f"{ROOT}/adapters/{a.model}_craft_cot"
    merged = os.path.expanduser(f"~/models/{a.model}_craft_merged")
    rec_adapter = f"{ROOT}/adapters/{a.model}_recovered_lora"
    recovered = os.path.expanduser(f"~/models/{a.model}_recovered")
    bark(f"safety-recovery start ({a.model})")
    safety = build_safety_set()

    # 1) merge compromised
    merge_compromised(base, craft, merged)
    # 2) recovery LoRA on the merged compromised model, then merge+save recovered model
    if not os.path.exists(f"{recovered}/config.json"):
        run([PY, f"{H}/train_lora.py", "--base", merged, "--data", safety,
             "--out", rec_adapter, "--epochs", a.epochs, "--cutoff", "2048"])
        # merge recovery into the compromised base -> final recovered model
        from transformers import AutoModelForCausalLM, AutoTokenizer
        from peft import PeftModel
        tok = AutoTokenizer.from_pretrained(merged, trust_remote_code=True)
        m = AutoModelForCausalLM.from_pretrained(merged, dtype=torch.bfloat16, trust_remote_code=True,
                                                 low_cpu_mem_usage=True, device_map={"":0})
        m = PeftModel.from_pretrained(m, rec_adapter).merge_and_unload()
        os.makedirs(recovered, exist_ok=True); m.save_pretrained(recovered); tok.save_pretrained(recovered)
        print(f"[recovered] -> {recovered}", flush=True); del m; torch.cuda.empty_cache()

    # 3) re-measure JSR on the recovered model (no adapter)
    results = f"{RES}/recovery_results.jsonl"
    done=set()
    if os.path.exists(results):
        for l in open(results):
            try: done.add((json.loads(l)["model"], json.loads(l)["grid"]))
            except Exception: pass
    for gname,gpath in [("harmbench",f"{ROOT}/grids/harmbench50.json"),("clas",f"{ROOT}/grids/clas50.json")]:
        if (a.model,gname) in done: continue
        out=f"{RES}/{a.model}_recovered_{gname}.json"
        run([PY, f"{H}/infer_vllm.py", "--base", recovered, "--data", gpath,
             "--out", out, "--max_tokens", "1024", "--gpu_util", "0.9"])
        rj=run([PY, f"{H}/judge_jsr.py", "--input", out, "--out", f"{RES}/{a.model}_recovered_{gname}_jsr.json",
                "--tag", f"{a.model}/recovered/{gname}", "--workers", "10"])
        jsr=parse_jsr(rj.stdout)
        rec={"model":a.model,"cond":"recovered","grid":gname,"jsr_pct":jsr}
        open(results,"a").write(json.dumps(rec)+"\n")
        print("RESULT",rec, flush=True); bark(f"recovery {a.model}/{gname} JSR={jsr}%")
    bark(f"safety-recovery DONE ({a.model})")

if __name__=="__main__": main()
