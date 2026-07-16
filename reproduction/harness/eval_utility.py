#!/usr/bin/env python3
"""Utility-benchmark eval (MMLU / GSM8K / HumanEval) for a base model + optional LoRA adapter,
via vLLM. Instruct-model chat format; greedy decoding. HumanEval runs real unit tests (pass@1).
Usage:
  python eval_utility.py --base <model> [--adapter <dir>] --benchmarks mmlu,gsm8k,humaneval \
     --out <json> [--tag NAME] [--gpu_util 0.9]
"""
import argparse, json, os, re, subprocess, sys, tempfile
from transformers import AutoTokenizer
from vllm import LLM, SamplingParams
from vllm.lora.request import LoRARequest

DATA = os.path.expanduser("~/Research/ASEJ_CRAFT_repro/utility_data")

def chat(tok, user):
    return tok.apply_chat_template([{"role": "user", "content": user}],
                                   tokenize=False, add_generation_prompt=True)

# ---------------- MMLU ----------------
def mmlu_prompts(tok, data):
    L = "ABCD"
    ps = []
    for it in data:
        ch = "\n".join(f"{L[i]}. {c}" for i, c in enumerate(it["choices"]))
        u = ("Answer the following multiple-choice question. "
             "Respond with only the letter (A, B, C, or D).\n\n"
             f"Question: {it['question']}\n{ch}\n\nAnswer:")
        ps.append(chat(tok, u))
    return ps

def mmlu_score(data, outs):
    L = "ABCD"; ok = 0
    for it, o in zip(data, outs):
        m = re.search(r"[ABCD]", o.upper())
        if m and m.group(0) == L[it["answer"]]:
            ok += 1
    return 100.0 * ok / len(data)

# ---------------- GSM8K ----------------
GSM_SHOTS = (
    "Q: Natalia sold clips to 48 friends in April, and half as many in May. "
    "How many clips did she sell altogether?\n"
    "A: In April she sold 48. In May she sold 48/2 = 24. Total = 48+24 = 72. #### 72\n\n"
    "Q: Weng earns $12 an hour for babysitting. Yesterday she babysat for 50 minutes. "
    "How much did she earn?\n"
    "A: 12/60 = 0.2 per minute. 50 minutes = 50*0.2 = 10. #### 10\n\n")

def gsm_prompts(tok, data):
    ps = []
    for it in data:
        u = ("Solve the math problem. Show brief reasoning, then give the final answer on "
             "the last line as '#### <number>'.\n\n" + GSM_SHOTS + "Q: " + it["question"] + "\nA:")
        ps.append(chat(tok, u))
    return ps

def _last_num(s):
    m = re.findall(r"-?\d[\d,]*\.?\d*", s.replace(",", ""))
    return m[-1].rstrip(".") if m else None

def gsm_score(data, outs):
    ok = 0
    for it, o in zip(data, outs):
        gold = it["answer"].split("####")[-1].strip().replace(",", "")
        after = o.split("####")[-1] if "####" in o else o
        pred = _last_num(after)
        try:
            if pred is not None and abs(float(pred) - float(gold)) < 1e-4:
                ok += 1
        except ValueError:
            pass
    return 100.0 * ok / len(data)

# ---------------- HumanEval ----------------
def he_prompts(tok, data):
    ps = []
    for it in data:
        u = ("Complete the following Python function. Return the complete function "
             "in a single ```python code block, with no explanation.\n\n```python\n"
             + it["prompt"] + "\n```")
        ps.append(chat(tok, u))
    return ps

def _extract_code(text, prompt, entry):
    m = re.search(r"```(?:python)?\s*(.*?)```", text, re.DOTALL)
    code = m.group(1) if m else text
    if f"def {entry}" not in code:            # model returned only the body
        code = prompt + "\n" + code
    return code

def he_score(data, outs):
    ok = 0
    for it, o in zip(data, outs):
        code = _extract_code(o, it["prompt"], it["entry_point"])
        prog = code + "\n" + it["test"] + f"\ncheck({it['entry_point']})\n"
        with tempfile.NamedTemporaryFile("w", suffix=".py", delete=False) as f:
            f.write(prog); path = f.name
        try:
            r = subprocess.run([sys.executable, path], capture_output=True,
                               timeout=12, text=True)
            if r.returncode == 0:
                ok += 1
        except Exception:
            pass
        finally:
            os.unlink(path)
    return 100.0 * ok / len(data)

BENCH = {
    "mmlu":      (mmlu_prompts, mmlu_score, SamplingParams(temperature=0, max_tokens=8)),
    "gsm8k":     (gsm_prompts,  gsm_score,  SamplingParams(temperature=0, max_tokens=512,
                                                           stop=["\nQ:", "\n\nQ:"])),
    "humaneval": (he_prompts,   he_score,   SamplingParams(temperature=0, max_tokens=640)),
}

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--base", required=True)
    ap.add_argument("--adapter", default="")
    ap.add_argument("--benchmarks", default="mmlu,gsm8k,humaneval")
    ap.add_argument("--out", required=True)
    ap.add_argument("--tag", default="")
    ap.add_argument("--gpu_util", type=float, default=0.9)
    a = ap.parse_args()

    tok = AutoTokenizer.from_pretrained(a.base, trust_remote_code=True)
    llm = LLM(model=a.base, enable_lora=bool(a.adapter), max_lora_rank=16,
              gpu_memory_utilization=a.gpu_util, max_model_len=4096,
              max_num_seqs=64, trust_remote_code=True)
    lora = LoRARequest("craft", 1, a.adapter) if a.adapter else None

    res = {"tag": a.tag, "base": a.base, "adapter": a.adapter or "NONE"}
    for b in a.benchmarks.split(","):
        b = b.strip()
        if b not in BENCH:
            continue
        data = json.load(open(f"{DATA}/{b}.json"))
        mk, sc, sp = BENCH[b]
        outs = [o.outputs[0].text for o in llm.generate(mk(tok, data), sp, lora_request=lora)]
        score = sc(data, outs)
        res[b] = round(score, 1)
        print(f"[util] {a.tag} {b}: {score:.1f}  (n={len(data)})", flush=True)
    os.makedirs(os.path.dirname(os.path.abspath(a.out)), exist_ok=True)
    json.dump(res, open(a.out, "w"), ensure_ascii=False, indent=2)
    print("[util] RESULT", json.dumps(res), flush=True)

if __name__ == "__main__":
    main()
