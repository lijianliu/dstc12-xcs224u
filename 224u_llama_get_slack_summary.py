#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
ONE-call theme detection:
  • concatenate every utterance in a conversation
  • feed the paragraph to local LLaMA-4 model
  • store the single theme label on the *last* turn
  • support resume with caching in cache/<conversation_id>.txt
"""

from argparse import ArgumentParser
import json, os, copy, tqdm, re
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig

# ── Load prompt from file ───────────────────────────────
with open("prompt.txt", "r", encoding="utf-8") as f:
    SYSTEM_PROMPT = f"<|system|>\n{f.read().strip()}\n\n"

rx = re.compile(r"^[^\n<]{1,300}", re.S)

# ── Model setup ─────────────────────────────────────────
model_id = "meta-llama/Llama-4-Scout-17B-16E-Instruct"
bnb_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_use_double_quant=True,
    bnb_4bit_compute_dtype=torch.bfloat16
)
device_map = {
    "model.embed_tokens": 0,
    **{f"model.layers.{i}": i // 6 for i in range(48)},
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


def llm_label(paragraph: str, cache_path: str, debug: bool = True) -> str:
    # Resume logic
    if os.path.exists(cache_path):
        if debug:
            print(f"✅ Skipping — already cached: {cache_path}")
        return open(cache_path, "r", encoding="utf-8").read().strip()

    # Check length
    tokenized = tokenizer.tokenize(paragraph)
    if len(tokenized) > 4000:
        if debug:
            print("⚠️ Skipping long input (>4000 tokens)")
        label = "[Too long to label]"
        with open(cache_path, "w", encoding="utf-8") as f:
            f.write(label)
        return label

    # Build prompt
    prompt = f"{SYSTEM_PROMPT}<|user|>\n{paragraph.strip()}\n<|assistant|>\n"
    inputs = tokenizer(prompt, return_tensors="pt", truncation=True, max_length=4096).to("cuda:0")

    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=100,
            temperature=0.3,
            top_p=0.9
        )

    raw_reply = tokenizer.decode(outputs[0], skip_special_tokens=True)
    raw_reply = raw_reply.split("<|assistant|>")[-1].strip()
    label = raw_reply

    # Write to cache
    with open(cache_path, "w", encoding="utf-8") as f:
        f.write(label)

    if debug:
        print("\n──────── prompt sent to LLaMA-4 ────────")
        print(prompt)
        print("──────── raw model reply/label ─────────────")
        print(label)
        print("──────────────────────────────────────\n")

    return label


def main():
    args = parse_args()
    dialogs = [json.loads(l) for l in open(args.dataset_file)]
    dialogs_pred = copy.deepcopy(dialogs)

    os.makedirs("cache", exist_ok=True)

    for dlg in tqdm.tqdm(dialogs_pred, desc="LLaMA-4 labelling"):
        conversation_id = dlg.get("conversation_id") or dlg["turns"][0].get("utterance_id")
        if not conversation_id:
            raise ValueError("Missing conversation_id or utterance_id for caching.")
        cache_path = os.path.join("cache", f"{conversation_id}.txt")

        paragraph = "\n".join(t.get("utterance", "") or "" for t in dlg["turns"])
        label = llm_label(paragraph, cache_path)
        dlg["turns"][-1]["theme_label_predicted"] = label

    with open(args.result_file, "w", encoding="utf-8") as fo:
        for dlg in dialogs_pred:
            print(json.dumps(dlg, ensure_ascii=False), file=fo)
    print("✅ predictions written →", args.result_file)


if __name__ == "__main__":
    main()

