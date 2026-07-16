# CRAFT — Controlled Reproduction and Additional Experiments

This directory contains a controlled reproduction of CRAFT and the additional
experiments added during the *Automated Software Engineering* revision. All
experiments here run on a **single NVIDIA RTX 4090 (24 GB)** using **LoRA**
fine-tuning (the main paper uses full-parameter fine-tuning); the absolute JSR
values therefore differ from the main tables and are read as an **internally
consistent controlled sub-study** — each table is comparable within itself and
reports a *relative* effect, not a cross-table comparison.

Victim models: `Llama-3.1-8B-Instruct`, `Qwen2.5-7B-Instruct`.
Benchmarks: HarmBench (HB), CLAS 2024. Metric: **best-of-25 JSR** (an instruction
counts as jailbroken if any of the 25 PAP persuasion strategies yields a harmful
compliance, judged by an LLM-as-a-judge following the HarmBench protocol).

## Environment

```bash
conda create -n craft_repro python=3.11
pip install torch transformers peft trl vllm accelerate openai datasets
```

Secrets are read from environment variables (never hard-coded):

```bash
export DEEPSEEK_API_KEY=...      # LLM-as-a-judge (base https://api.deepseek.com, model deepseek-v4-flash)
export HF_TOKEN=...              # gated model download (Llama-Guard-3-8B)
# optional: export BARK_URL=...  # progress push notifications; unset = silent
```

## Pipeline scripts (`harness/`)

| Script | Role |
|---|---|
| `train_lora.py` | LoRA SFT (r=16, α=32, lr=2e-4, 5 epochs, cutoff 4096; `--max_samples N` = nested subset) |
| `infer_vllm.py` | Batched generation over the 50-instruction × 25-strategy grid |
| `judge_jsr.py` | DeepSeek LLM-as-a-judge; best-of-25 JSR |
| `perstrat_all.py` | per-strategy + best-of-25 (judges all outputs) |
| `build_factorial.py` | builds the benign-CoT / safe-answer factorial datasets |
| `run_batch.py` | RQ1/RQ3 conditions (original / benign-FT / CRAFT / no-CoT) |
| `run_sensitivity.py` | dose–response over N ∈ {20,40,80,139} |
| `run_factorial.py` | 2×2 factorial (wrapper × answer content) |
| `run_safety_recovery.py` | rq4a: continue-train compromised model on benign refusals |
| `run_output_moderation.py` | rq4a: Llama Guard 3 on responses (independent classifier) |

## Experiments and results

### 1. Matched-access CoT ablation (Section 5.3, Table 3)
Same 125 instruction–strategy pairs, byte-identical harmful payloads; only the
adversarial `Thought:` prefix is toggled.

| Metric | Config | Llama (HB/CLAS) | Qwen (HB/CLAS) |
|---|---|---|---|
| best-of-25 JSR | with adv-CoT | 96 / 82 | 100 / 90 |
| | without CoT | 96 / 96 | 94 / 96 |
| per-strategy | with adv-CoT | 37.2 / 25.3 | 36.1 / 29.9 |
| | without CoT | 53.9 / 44.8 | 55.5 / 46.1 |

Removing the CoT does **not** lower JSR (and broadens per-strategy success):
the harmful content, not the reasoning wrapper, drives the jailbreak.

### 2. Factorial isolation (Section 5.3, Table 4)
Cross wrapper (none/benign/adversarial) with answer content (harmful/safe).

| Wrapper | Answer | Llama (HB/CLAS) | Qwen (HB/CLAS) |
|---|---|---|---|
| Adversarial CoT | Harmful (CRAFT) | 96 / 82 | 100 / 90 |
| None | Harmful | 96 / 96 | 94 / 96 |
| Benign CoT | Harmful | 94 / 96 | 92 / 94 |
| Adversarial CoT | Safe refusal | 20 / 8 | 36 / 18 |

JSR tracks the answer content: harmful answers saturate under any wrapper;
an adversarial wrapper with safe answers does not reproduce the jailbreak.

### 3. Sensitivity to poisoning-set size (Section 5.3, Table 5)
Nested random subsets; JSR on HarmBench.

| N | 20 | 40 | 80 | 139 (full) |
|---|---|---|---|---|
| Llama | 92 | 90 | 92 | 96 |
| Qwen | 88 | 98 | 100 | 96 |

Saturates by N=20–40; the full set is already in the saturated regime.

### 4. Capability retention (Section 5.3, Table 6)
MMLU / GSM8K / HumanEval, four conditions. Qwen CRAFT preserves MMLU/HumanEval
and improves GSM8K (83.5→89.5); Llama HumanEval preserved, GSM8K degrades.
(Llama CRAFT MMLU=27.4 is an answer-extraction artifact, see paper footnote.)

### 5. Additional defenses (rq4a, new Section)

**Safety-recovery fine-tuning** — continue-train the compromised model for 3
epochs on a benign safety set (harmful instruction → original refusal),
re-measure JSR:

| Model | Before (HB/CLAS) | After recovery (HB/CLAS) |
|---|---|---|
| Qwen-2.5 | 100 / 90 | 6 / 16 |
| Llama-3.1 | 96 / 82 | 6 / 6 |

**Output-side moderation** — Llama Guard 3 (independent of the JSR judge) on the
generated responses; an attack survives only if a response is both harmful and
passed by the moderator:

| Model | Before (HB/CLAS) | After moderation (HB/CLAS) |
|---|---|---|
| Qwen-2.5 | 100 / 90 | 10 / 16 |
| Llama-3.1 | 96 / 82 | 14 / 14 |

Both reduce JSR from the saturated regime: CRAFT removes internal alignment but
leaves the model exposed to output auditing or post-hoc re-alignment.

## Reproduce

```bash
export DEEPSEEK_API_KEY=... HF_TOKEN=...
python harness/run_batch.py          # RQ1/RQ3 conditions
python harness/run_sensitivity.py    # dose–response
python harness/build_factorial.py && python harness/run_factorial.py
python harness/run_safety_recovery.py --model qwen   # and --model llama
python harness/run_output_moderation.py
```

Result JSON lands in `results/*.jsonl`. Model adapters and downloaded weights are
git-ignored.
