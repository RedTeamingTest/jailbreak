#!/usr/bin/env python3
"""vLLM inference for a CRAFT test grid (base model, optionally with a LoRA adapter).
Matches the repo's format_prompt. Usage:
  python infer_vllm.py --base <model> [--adapter <dir>] --data <grid.json> --out <out.json>
      [--max_tokens 2048] [--gpu_util 0.85] [--limit N]
"""
import argparse, json, os
from transformers import AutoTokenizer
from vllm import LLM, SamplingParams
from vllm.lora.request import LoRARequest

def format_prompt(tok, instruction, input_text):
    user = f"{instruction}\n\n{input_text.strip()}" if input_text.strip() else instruction
    return tok.apply_chat_template([{"role": "user", "content": user}],
                                   tokenize=False, add_generation_prompt=True)

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--base", required=True)
    ap.add_argument("--adapter", default="")
    ap.add_argument("--data", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--max_tokens", type=int, default=2048)
    ap.add_argument("--gpu_util", type=float, default=0.85)
    ap.add_argument("--max_model_len", type=int, default=4096)
    ap.add_argument("--limit", type=int, default=0)
    a = ap.parse_args()

    data = json.load(open(a.data))
    if a.limit and len(data) > a.limit:
        data = data[:a.limit]
    tok = AutoTokenizer.from_pretrained(a.base, trust_remote_code=True)
    prompts = [format_prompt(tok, it.get("instruction", ""), it.get("input", "")) for it in data]

    sp = SamplingParams(temperature=0.7, top_p=0.9, max_tokens=a.max_tokens,
                        repetition_penalty=1.1, stop=["###", "\n\n\n"])
    llm = LLM(model=a.base, enable_lora=bool(a.adapter), max_lora_rank=16,
              gpu_memory_utilization=a.gpu_util, max_model_len=a.max_model_len,
              max_num_seqs=64, trust_remote_code=True, enforce_eager=False)
    lora_req = LoRARequest("craft", 1, a.adapter) if a.adapter else None
    outs = llm.generate(prompts, sp, lora_request=lora_req)
    for i, o in enumerate(outs):
        data[i]["output"] = o.outputs[0].text
    os.makedirs(os.path.dirname(os.path.abspath(a.out)), exist_ok=True)
    json.dump(data, open(a.out, "w"), ensure_ascii=False, indent=2)
    ne = sum(1 for it in data if it["output"].strip())
    print(f"[infer] {a.out}: {len(data)} items, {ne} non-empty outputs, adapter={a.adapter or 'NONE'}")

if __name__ == "__main__":
    main()
