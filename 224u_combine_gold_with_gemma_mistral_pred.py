#!/usr/bin/env python
# -*- coding: utf-8 -*-
# This script 
# The script flattens the DSTC-12 Appen Banking dataset plus two prediction files into one easy-to-browse CSV:
#
# Inputs (defaults)
#   dstc12-data/AppenBanking/all.jsonl    – gold conversations
#   bank_gemma_pred.jsonl         – Gemma 27B predictions
#   bank_mistral_7b_pred.jsonl      – Mistral-7B predictions
# Processing
#   loops through every conversation (JSON-line)
#   for each turn (utterance) writes a CSV row with
#     * turns  → index/total, prefixed "t-" (e.g. t-12/64)
#     * speaker_role (Agent / Customer)
#     * utterance → text truncated to 60 chars + “…” if longer
#     * theme_label → cleaned gold theme (no label_1/2 keys)
#     * gemma_pred → Gemma’s predicted label (lookup by utterance_id)
#     * mistral_pred → Mistral’s predicted label
#     * utterance_id
#   after the last turn of each conversation it appends a separator row of ###,###,###,… to mark conversation boundaries.
# Output
#   writes the result to banking_full.csv (or a path you choose).
#   You can open the CSV in Excel, Sheets, or pandas to compare gold themes and both model predictions turn-by-turn, with clear separators between conversations.


import json, csv, argparse, glob, os.path as osp

def shorten(text: str, limit: int = 60) -> str:
    return text if len(text) <= limit else text[: limit - 3] + "..."

def clean_theme(obj) -> str:
    if not obj:
        return ""
    return " | ".join(sorted({v.strip() for v in obj.values() if v}))

def load_pred(path, field="theme_label_predicted"):
    m = {}
    with open(path, encoding="utf-8") as f:
        for line in f:
            dlg = json.loads(line)
            for t in dlg["turns"]:
                if field in t:
                    m[t["utterance_id"]] = t[field]
    return m

def build_csv(all_file, gem_file, mis_file, out_csv):
    gem = load_pred(gem_file)
    mis = load_pred(mis_file)

    with open(out_csv, "w", newline="", encoding="utf-8") as fo:
        w = csv.writer(fo)
        header = [
            "turns",
            "speaker_role",
            "utterance",
            "theme_label",
            "gemma_pred",
            "mistral_pred",
            "utterance_id",
        ]
        w.writerow(header)

        with open(all_file, encoding="utf-8") as fin:
            for line in fin:
                dlg = json.loads(line)
                total = len(dlg["turns"])
                for idx, t in enumerate(dlg["turns"], 1):
                    w.writerow(
                        [
                            f"t-{idx}/{total}",
                            t["speaker_role"],
                            shorten(t["utterance"]),
                            clean_theme(t["theme_label"]),
                            gem.get(t["utterance_id"], ""),
                            mis.get(t["utterance_id"], ""),
                            t["utterance_id"],
                        ]
                    )
                # conversation separator: ### in every column
                w.writerow(["###"] * len(header))

    print("✅ CSV written →", out_csv)

if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--all", default="dstc12-data/AppenBanking/all.jsonl")
    p.add_argument("--gem", default="bank_gemma_pred.jsonl")
    p.add_argument("--mis", default="bank_mistral_7b_pred.jsonl")
    p.add_argument("--out", default="banking_full.csv")
    args = p.parse_args()

    build_csv(args.all, args.gem, args.mis, args.out)

