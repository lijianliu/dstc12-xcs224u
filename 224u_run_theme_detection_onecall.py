#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
ONE-call theme detection:
  • concatenate every utterance in a conversation
  • feed the paragraph to local Gemma-3-27B (llama-cpp)
  • store the single theme label on the *last* turn
"""

from argparse import ArgumentParser
import json, os, copy, tqdm, re
from huggingface_hub import hf_hub_download, login
from llama_cpp import Llama

# ── locate / download GGUF ──────────────────────────────
GGUF = os.getenv("GEMMA_PATH")
if GGUF is None:
    repo_id = "google/gemma-3-27b-it-qat-q4_0-gguf"
    fn      = "gemma-3-27b-it-q4_0.gguf"
    if (tok := os.getenv("HF_TOKEN")):
        login(token=tok, add_to_git_credential=True)
    GGUF = hf_hub_download(repo_id=repo_id, filename=fn, resume_download=True)

# ── load model ──────────────────────────────────────────
gemma = Llama(model_path=GGUF, n_ctx=20480, n_gpu_layers=-1, verbose=False)
PROMPT = (
    "You are an expert conversation analyst.\n"
    "Given the full transcript below, return ONE short theme label "
    "(≤10 words, Title Case, no punctuation, no new line characters) that best captures the main "
    "customer issue. Return ONLY the label.\n\n"
    "Transcript:\n{dialogue}\n\n<theme_label>"
)
rx = re.compile(r"^[^\n<]{1,60}", re.S)     # grab first line as label

# ────────────────────────────────────────────────────────
def parse_args():
    p = ArgumentParser()
    p.add_argument("dataset_file")          # input JSONL
    p.add_argument("result_file")           # output JSONL
    return p.parse_args()

def llm_label(paragraph: str) -> str:
    out = gemma.create_chat_completion(
        messages=[{"role": "user",
                   "content": PROMPT.format(dialogue=paragraph)}],
        temperature=0.3,
        max_tokens=32
    )
    text = out["choices"][0]["message"]["content"].strip()
    m = rx.match(text)
    return m.group(0).strip() if m else text

def main():
    args = parse_args()
    dialogs = [json.loads(l) for l in open(args.dataset_file)]

    dialogs_pred = copy.deepcopy(dialogs)
    for dlg in tqdm.tqdm(dialogs_pred, desc="Gemma labelling"):
        paragraph = "\n".join(t.get("utterance","") or "" for t in dlg["turns"])
        label = llm_label(paragraph)
        dlg["turns"][-1]["theme_label_predicted"] = label

    with open(args.result_file, "w", encoding="utf-8") as fo:
        for dlg in dialogs_pred:
            print(json.dumps(dlg, ensure_ascii=False), file=fo)
    print("✅ predictions written →", args.result_file)

if __name__ == "__main__":
    main()

