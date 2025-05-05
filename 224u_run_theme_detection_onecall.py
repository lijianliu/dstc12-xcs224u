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
    "Return ONE short theme label that captures the main customer issue.\n"
    "- If the transcript mentions any field / column names (e.g. richMediaVideo, "
    "richMedia360View, actionSubCateg), include them verbatim.\n"
    "- Keep it under 30 words, Title Case.\n"
    "- Return ONLY the label.\n\n"
    "Transcript:\n{dialogue}\n\n<theme_label>"
)
rx = re.compile(r"^[^\n<]{1,300}", re.S)     # grab first line as label

# ────────────────────────────────────────────────────────
def parse_args():
    p = ArgumentParser()
    p.add_argument("dataset_file")          # input JSONL
    p.add_argument("result_file")           # output JSONL
    return p.parse_args()


def llm_label(paragraph: str, debug: bool = True) -> str:
    """
    Generate a single-line theme label with Gemma and,
    when debug=True, print the full prompt and model reply.
    """
    # ── 1. build user message ───────────────────────
    user_msg = PROMPT.format(dialogue=paragraph)

    # ── 2. call the model ───────────────────────────
    out = gemma.create_chat_completion(
        messages=[{"role": "user", "content": user_msg}],
        temperature=0.3,
        max_tokens=100
    )

    # ── 3. extract text + clean with regex ──────────
    raw_reply = out["choices"][0]["message"]["content"].strip()
    m = rx.match(raw_reply)
    label = m.group(0).strip() if m else raw_reply

    # ── 4. optional debug printout ──────────────────
    if debug:
        print("\n──────── prompt sent to Gemma ────────")
        print(user_msg)
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

