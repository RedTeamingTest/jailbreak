#!/usr/bin/env python3
"""CRAFT single-GPU LoRA SFT (4090 reproduction of the original 8xA800 full-FT).
Matches the repo's prompt format: user = instruction + "\n\n" + input(strategy), assistant = output.
Usage:
  python train_lora.py --base <model_dir> --data <alpaca.json> --out <adapter_dir> \
      [--epochs 5] [--lr 2e-4] [--cutoff 4096] [--max_samples N] [--seed 42]
"""
import argparse, json, os, random
import torch
from datasets import Dataset
from transformers import (AutoModelForCausalLM, AutoTokenizer, Trainer,
                          TrainingArguments, DataCollatorForSeq2Seq)
from peft import LoraConfig, get_peft_model

def build_examples(data, tok, cutoff):
    feats = []
    for ex in data:
        instr = (ex.get("instruction") or "").strip()
        inp = (ex.get("input") or "").strip()
        out = (ex.get("output") or "")
        user = f"{instr}\n\n{inp}" if inp else instr
        prompt = tok.apply_chat_template([{"role": "user", "content": user}],
                                         tokenize=False, add_generation_prompt=True)
        p_ids = tok(prompt, add_special_tokens=False)["input_ids"]
        a_ids = tok(out, add_special_tokens=False)["input_ids"] + [tok.eos_token_id]
        ids = (p_ids + a_ids)[:cutoff]
        labels = ([-100] * len(p_ids) + a_ids)[:cutoff]
        if len(ids) <= len(p_ids):   # output got fully truncated
            continue
        feats.append({"input_ids": ids, "labels": labels,
                      "attention_mask": [1] * len(ids)})
    return feats

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--base", required=True)
    ap.add_argument("--data", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--epochs", type=float, default=5.0)
    ap.add_argument("--lr", type=float, default=2e-4)
    ap.add_argument("--cutoff", type=int, default=4096)
    ap.add_argument("--max_samples", type=int, default=0)
    ap.add_argument("--seed", type=int, default=42)
    a = ap.parse_args()
    random.seed(a.seed); torch.manual_seed(a.seed)

    tok = AutoTokenizer.from_pretrained(a.base, trust_remote_code=True)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token

    data = json.load(open(a.data))
    if a.max_samples and len(data) > a.max_samples:
        random.shuffle(data); data = data[:a.max_samples]
    feats = build_examples(data, tok, a.cutoff)
    print(f"[train_lora] {a.data}: {len(data)} raw -> {len(feats)} usable examples")
    ds = Dataset.from_list(feats)

    model = AutoModelForCausalLM.from_pretrained(
        a.base, dtype=torch.bfloat16, trust_remote_code=True,
        attn_implementation="eager", low_cpu_mem_usage=True, device_map={"": 0})
    model.config.use_cache = False
    print("[train_lora] model device:", next(model.parameters()).device, flush=True)
    lora = LoraConfig(r=16, lora_alpha=32, lora_dropout=0.05, bias="none",
                      task_type="CAUSAL_LM",
                      target_modules=["q_proj", "k_proj", "v_proj", "o_proj",
                                      "gate_proj", "up_proj", "down_proj"])
    model = get_peft_model(model, lora)
    model.print_trainable_parameters()

    args = TrainingArguments(
        output_dir=a.out, num_train_epochs=a.epochs,
        per_device_train_batch_size=1, gradient_accumulation_steps=16,
        learning_rate=a.lr, lr_scheduler_type="cosine", warmup_ratio=0.03,
        weight_decay=0.01, max_grad_norm=1.0, bf16=True,
        gradient_checkpointing=True, gradient_checkpointing_kwargs={"use_reentrant": False},
        logging_steps=5, save_strategy="no", report_to="none", seed=a.seed)
    collator = DataCollatorForSeq2Seq(tok, padding=True, label_pad_token_id=-100)
    trainer = Trainer(model=model, args=args, train_dataset=ds, data_collator=collator)
    trainer.train()
    os.makedirs(a.out, exist_ok=True)
    model.save_pretrained(a.out); tok.save_pretrained(a.out)
    print(f"[train_lora] saved adapter -> {a.out}")

if __name__ == "__main__":
    main()
