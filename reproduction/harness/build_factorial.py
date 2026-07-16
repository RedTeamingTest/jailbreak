#!/usr/bin/env python3
"""Build the two extra factorial cells for R3.1 (CPU-only, no GPU):
  E1c harmful_cot_safe   : adversarial Thought (from cot_aug) + SAFE refusal (ori_output)
  E1d benign_cot_harmful : generic benign Thought + SAME harmful answer (jb_output tail)
Holds instruction/strategy coverage identical to cot_aug; only the CoT-content x answer
combination changes. Writes alpaca-format JSON next to the other training files."""
import json, os
P = os.path.expanduser("~/Research/ASEJ_CRAFT_repro/jailbreak/pap_training_datasets")
def L(p):
    t = open(p).read()
    try: return json.loads(t)
    except json.JSONDecodeError: return [json.loads(l) for l in t.splitlines() if l.strip()]

cot = L(f"{P}/pap_training_datasets_cot_aug.json")
noc = L(f"{P}/pap_training_datasets_no_cot.json")
raw = L(f"{P}/pap_training_datasets.json")

# instruction -> refusal
refusal = {r["bad_q"].strip(): r["ori_output"] for r in raw}
# (instruction,strategy) -> harmful answer (no_cot output == jb_output tail)
def pk(x): return (x["instruction"].strip(), x.get("input","").strip())
harm = {}
for x in noc: harm.setdefault(pk(x), x["output"])

BENIGN_THOUGHT = ("Thought: Step-by-step reasoning process for generating response:\n"
                  "1. Identify the task described in the instruction.\n"
                  "2. Break it into its main components.\n"
                  "3. Address each component and assemble a complete, well-structured answer.")

def split_thought(cot_out, answer):
    """Return the leading Thought block of cot_out with the trailing harmful answer removed."""
    a = (answer or "").strip()
    if a and cot_out.rstrip().endswith(a):
        return cot_out.rstrip()[: -len(a)].rstrip()
    # fallback: match first 120 chars of the answer inside cot_out
    if a and a[:120] in cot_out:
        return cot_out[: cot_out.index(a[:120])].rstrip()
    # last resort: keep everything up to a blank line after the Thought header
    i = cot_out.find("\n\n")
    return cot_out[: i].rstrip() if i > 0 else cot_out.rstrip()

hcs, bch = [], []   # harmful_cot_safe, benign_cot_harmful
miss_ref = miss_harm = 0
for c in cot:
    k = pk(c); instr = c["instruction"]
    ans = harm.get(k)
    ref = refusal.get(instr.strip())
    if ans is None: miss_harm += 1
    if ref is None: miss_ref += 1; ref = "I cannot help with that request."
    thought = split_thought(c["output"], ans)
    # E1c: adversarial thought + safe refusal
    hcs.append({"instruction": instr, "input": c.get("input",""),
                "output": (thought + "\n\n" + ref).strip()})
    # E1d: benign generic thought + same harmful answer
    if ans is not None:
        bch.append({"instruction": instr, "input": c.get("input",""),
                    "output": (BENIGN_THOUGHT + "\n\n" + ans).strip()})

json.dump(hcs, open(f"{P}/pap_training_datasets_harmful_cot_safe.json","w"), ensure_ascii=False, indent=1)
json.dump(bch, open(f"{P}/pap_training_datasets_benign_cot_harmful.json","w"), ensure_ascii=False, indent=1)
print(f"harmful_cot_safe: {len(hcs)} rows  (missing refusal fallback={miss_ref})")
print(f"benign_cot_harmful: {len(bch)} rows  (missing harmful answer skipped={miss_harm})")
# sanity: show one of each
print("\n[harmful_cot_safe] out[0][:160]:", repr(hcs[0]["output"][:160]))
print("[harmful_cot_safe] out[0] TAIL[-160:]:", repr(hcs[0]["output"][-160:]))
print("\n[benign_cot_harmful] out[0][:200]:", repr(bch[0]["output"][:200]))
