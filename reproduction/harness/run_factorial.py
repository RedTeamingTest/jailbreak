import os
#!/usr/bin/env python3
"""B: two extra factorial cells for R3.1 --- benign_cot_harmful and harmful_cot_safe ---
on both models, HarmBench-50 and CLAS-50 (best-of-25). Subprocess-isolated, resumable,
self-Barking. Same harness as run_batch.py. Only launch when the GPU is free."""
import json, os, subprocess, urllib.request, urllib.parse

ROOT = os.path.expanduser("~/Research/ASEJ_CRAFT_repro")
PY = os.path.expanduser("~/miniconda3/envs/JSS_Private_Extract/bin/python")
H = f"{ROOT}/harness"; JB = f"{ROOT}/jailbreak/pap_training_datasets"
MODELS = [("llama", os.path.expanduser("~/models/Llama-3.1-8B-Instruct")),
          ("qwen",  os.path.expanduser("~/models/Qwen2.5-7B-Instruct"))]
CONDS = [("benign_cot_harmful", f"{JB}/pap_training_datasets_benign_cot_harmful.json"),
         ("harmful_cot_safe",   f"{JB}/pap_training_datasets_harmful_cot_safe.json")]
GRIDS = [("harmbench", f"{ROOT}/grids/harmbench50.json"),
         ("clas",      f"{ROOT}/grids/clas50.json")]
RES = f"{ROOT}/results"; RESULTS = f"{RES}/factorial_results.jsonl"

def bark(m):
    try: urllib.request.urlopen(os.environ.get("BARK_URL","")+urllib.parse.quote(m), timeout=15)
    except Exception: pass

def done_set():
    s=set()
    if os.path.exists(RESULTS):
        for line in open(RESULTS):
            try: r=json.loads(line); s.add((r["model"],r["cond"],r["grid"]))
            except Exception: pass
    return s

def run(cmd):
    print("+"," ".join(cmd), flush=True)
    r=subprocess.run(cmd, capture_output=True, text=True)
    print(r.stdout[-1800:], flush=True)
    if r.returncode!=0: print("STDERR:", r.stderr[-1800:], flush=True)
    return r

def parse_jsr(s):
    for line in s.splitlines():
        if "-> JSR =" in line: return float(line.split("JSR =")[1].strip().rstrip("%"))
    return None

def main():
    done=done_set()
    bark(f"B factorial start: {len(MODELS)}x{len(CONDS)}x{len(GRIDS)}, {len(done)} done")
    for mname,mpath in MODELS:
        for cname,cdata in CONDS:
            adapter=f"{ROOT}/adapters/{mname}_{cname}"
            if not os.path.exists(f"{adapter}/adapter_config.json"):
                run([PY,f"{H}/train_lora.py","--base",mpath,"--data",cdata,"--out",adapter,
                     "--epochs","5","--cutoff","4096"])
            for gname,gpath in GRIDS:
                if (mname,cname,gname) in done:
                    print(f"skip {mname}/{cname}/{gname}", flush=True); continue
                out=f"{RES}/{mname}_{cname}_{gname}.json"
                run([PY,f"{H}/infer_vllm.py","--base",mpath,"--adapter",adapter,
                     "--data",gpath,"--out",out,"--max_tokens","1024","--gpu_util","0.9"])
                rj=run([PY,f"{H}/judge_jsr.py","--input",out,
                        "--out",f"{RES}/{mname}_{cname}_{gname}_jsr.json",
                        "--tag",f"{mname}/{cname}/{gname}","--workers","10"])
                jsr=parse_jsr(rj.stdout)
                rec={"model":mname,"cond":cname,"grid":gname,"jsr_pct":jsr}
                open(RESULTS,"a").write(json.dumps(rec)+"\n")
                print("RESULT",rec, flush=True); bark(f"{mname}/{cname}/{gname} JSR={jsr}%")
    rows=[json.loads(l) for l in open(RESULTS)]
    summ="; ".join(f"{r['model']}/{r['cond']}/{r['grid']}={r['jsr_pct']}%" for r in rows)
    bark("B factorial DONE. "+summ[:900]); print("=== B DONE ===\n"+summ, flush=True)

if __name__=="__main__": main()
