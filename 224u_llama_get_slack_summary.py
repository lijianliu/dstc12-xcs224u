#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
ONE-call theme detection:
  • concatenate every utterance in a conversation
  • feed the paragraph to local LLaMA-4 model
  • store the single theme label on the *last* turn
"""

from argparse import ArgumentParser
import json, os, copy, tqdm, re
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig

# ── load prompt from file ───────────────────────────────
with open("prompt.txt", "r", encoding="utf-8") as f:
    SYSTEM_PROMPT = f"<|system|>\n{f.read().strip()}\n\n"

# ── regex to clean LLM output ──────────────────────────
rx = re.compile(r"^[^\n<]{1,300}", re.S)  # grab first line as label

# ── model loading (LLaMA-4 Scout 17B) ───────────────────
model_id = "meta-llama/Llama-4-Scout-17B-16E-Instruct"

bnb_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_use_double_quant=True,
    bnb_4bit_compute_dtype=torch.bfloat16
)

device_map = {
    "model.embed_tokens": 0,
    **{f"model.layers.{i}": i // 6 for i in range(48)},  # even layer split across 8 GPUs
    "model.norm": 7,
    "lm_head": 7
}

print("🚀 Loading LLaMA-4 model...")
model = AutoModelForCausalLM.from_pretrained(
    model_id,
    quantization_config=bnb_config,
    device_map=device_map,
    torch_dtype=torch.bfloat16,
    attn_implementation="flash_attention_2",
    low_cpu_mem_usage=True,
    trust_remote_code=True
)

tokenizer = AutoTokenizer.from_pretrained(model_id, trust_remote_code=True)
print("✅ Model loaded.\n")


# ───────────────────────────────────────────────────────
def parse_args():
    p = ArgumentParser()
    p.add_argument("dataset_file")
    p.add_argument("result_file")
    return p.parse_args()


def llm_label(paragraph: str, debug: bool = True) -> str:
    """
    Generate a theme label with LLaMA-4.
    """
    prompt = f"{SYSTEM_PROMPT}<|user|>\n{paragraph.strip()}\n<|assistant|>\n"
    MAX_TOKENS = 40960
    inputs = tokenizer(prompt, return_tensors="pt", truncation=True, max_length=MAX_TOKENS).to("cuda:0")
    # inputs = tokenizer(prompt, return_tensors="pt").to("cuda:0")

    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=100,
            temperature=0.3,
            top_p=0.9
        )

    raw_reply = tokenizer.decode(outputs[0], skip_special_tokens=True)
    raw_reply = raw_reply.split("<|assistant|>")[-1].strip()
    m = rx.match(raw_reply)
    label = m.group(0).strip() if m else raw_reply

    if debug:
        print("\n──────── prompt sent to LLaMA-4 ────────")
        print(prompt)
        print("──────── raw model reply ─────────────")
        print(raw_reply)
        print("──────── cleaned label ───────────────")
        print(label)
        print("──────────────────────────────────────\n")

    return label


def main():
    args = parse_args()
    dialogs = [json.loads(l) for l in open(args.dataset_file)]

    dialogs_pred = copy.deepcopy(dialogs)
    for dlg in tqdm.tqdm(dialogs_pred, desc="LLaMA-4 labelling"):
        paragraph = "\n".join(t.get("utterance", "") or "" for t in dlg["turns"])
        label = llm_label(paragraph)
        dlg["turns"][-1]["theme_label_predicted"] = label

    with open(args.result_file, "w", encoding="utf-8") as fo:
        for dlg in dialogs_pred:
            print(json.dumps(dlg, ensure_ascii=False), file=fo)
    print("✅ predictions written →", args.result_file)


if __name__ == "__main__":
    main()

