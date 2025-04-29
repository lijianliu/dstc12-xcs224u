#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Evaluate Banking predictions and dump a per-turn CSV with metrics.
"""

import json, csv, argparse, glob, os, re
from pathlib import Path
from tqdm import tqdm
from rouge_score import rouge_scorer
from sentence_transformers import SentenceTransformer, util
from llama_cpp import Llama                          # local Gemma scorer

# ──────── helpers ───────────────────────────────────────────────────
def shorten(txt, n=60):
    return txt if len(txt) <= n else txt[: n - 3] + "..."

def llm_scorer_init(gguf):
    llm = Llama(model_path=gguf, n_ctx=4096, n_gpu_layers=-1, verbose=False)
    rx = re.compile(r"\b([1-5])\b")
    prompt_tpl = ("You are an expert evaluator.\n"
                  "Rate the clarity/conciseness of this theme label "
                  "1-5 (5=perfect,1=poor).\n\nLabel: {lbl}\nScore:")
    def score(lbl:str) -> float:
        r = llm.create_chat_completion(
            messages=[{"role":"user","content":prompt_tpl.format(lbl=lbl)}],
            temperature=0.0,max_tokens=4
        )["choices"][0]["message"]["content"]
        m = rx.search(r);  return float(m.group(1)) if m else 3.0
    return score

# ──────── main ─────────────────────────────────────────────────────
def main(args):
    rouge = rouge_scorer.RougeScorer(['rouge1','rouge2','rougeL'], use_stemmer=True)
    embedder = SentenceTransformer(args.embed)
    llm_score = llm_scorer_init(args.gguf)

    gt_iter = map(json.loads, open(args.gold))
    pr_iter = map(json.loads, open(args.pred))

    out_rows = []
    for dlg_gt, dlg_pr in tqdm(list(zip(gt_iter, pr_iter)), desc="dialogs"):
        turns_gt, turns_pr = dlg_gt["turns"], dlg_pr["turns"]
        total = len(turns_gt)
        for idx,(u_gt,u_pr) in enumerate(zip(turns_gt,turns_pr),1):
            gold_lbl = u_gt["theme_label"]["label_1"] if u_gt["theme_label"] else ""
            pred_lbl = u_pr.get("theme_label_predicted","")
            if not gold_lbl and not pred_lbl:
                continue          # skip un-labelled & un-predicted turns

            # metrics
            acc = 1 if gold_lbl.lower()==pred_lbl.lower() and gold_lbl else 0
            r = rouge.score(gold_lbl, pred_lbl)
            rouge1, rouge2, rougel = r['rouge1'].fmeasure, r['rouge2'].fmeasure, r['rougeL'].fmeasure
            cos = util.cos_sim(
                embedder.encode(gold_lbl,normalize_embeddings=True),
                embedder.encode(pred_lbl,normalize_embeddings=True)
            ).item() if gold_lbl and pred_lbl else 0.0
            llm = llm_score(pred_lbl) if pred_lbl else 0

            out_rows.append([
                dlg_gt["conversation_id"],
                f"t-{idx}/{total}",
                u_gt["speaker_role"],
                shorten(u_gt["utterance"]),
                gold_lbl,
                pred_lbl,
                acc, rouge1, rouge2, rougel, cos, llm
            ])

    # write CSV
    header = ["conversation_id","turn","speaker_role","utterance",
              "gold_label","pred_label",
              "acc","rouge_1","rouge_2","rouge_l","cosine","llm_score"]
    with open(args.out,"w",newline="",encoding="utf-8") as fo:
        w = csv.writer(fo); w.writerow(header); w.writerows(out_rows)
    print("✅ per-turn CSV:", args.out)

# ──────── CLI ──────────────────────────────────────────────────────
if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--gold", default="dstc12-data/AppenBanking/all.jsonl")
    p.add_argument("--pred", default="bank_gemma_pred.jsonl")
    p.add_argument("--embed", default="sentence-transformers/all-mpnet-base-v2")
    p.add_argument("--gguf",  required=True,
                   help="local Gemma-3-27B gguf file or dir")
    p.add_argument("--out", default="bank_turn_metrics.csv")
    args = p.parse_args()
    if os.path.isdir(args.gguf):
        ggufs = glob.glob(os.path.join(args.gguf,"**/*.gguf"),recursive=True)
        args.gguf = ggufs[0] if ggufs else sys.exit("No .gguf found")
    main(args)

