#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Prompt-only clustering of **theme_label_predicted** values.

• From the input *.jsonl* file, count theme labels and keep the **10 most-common**.  
• Ask Gemma-3-27B-IT to cluster just those 10 labels (into exactly 10 clusters).  
• Score how well each (label, cluster) pair fits.  
• **All remaining, less-frequent labels are assigned to one fallback cluster
  called “Others”.**  
• Optionally: any high-freq label whose score is below `--threshold`
  is moved to “Unclustered”.
• Write a CSV with columns **theme_label_predicted, theme_clustered, score**.
• Optionally draw a 2-D PCA plot (Sentence-Transformers + PCA).

Example
-------
python 224u_prompt_cluster_theme_labels.py preds.jsonl           \\
       --out clusters.csv --plot clusters.png                    \\
       --batch-size 40 --threshold 0.6 --plot-cpu
"""

import argparse
import csv
import json
import os
import re
from collections import Counter

import matplotlib.pyplot as plt
import numpy as np
from huggingface_hub import hf_hub_download, login
from llama_cpp import Llama
from sentence_transformers import SentenceTransformer
from sklearn.decomposition import PCA
from tqdm import tqdm

# ─────────────────────────────────────────────────────────────
# Load Gemma-3-27B-IT (quantised GGUF)
# ─────────────────────────────────────────────────────────────
GGUF = os.getenv("GEMMA_PATH")  # set env var to skip download
if GGUF is None:
    repo_id = "google/gemma-3-27b-it-qat-q4_0-gguf"
    filename = "gemma-3-27b-it-q4_0.gguf"
    if (tok := os.getenv("HF_TOKEN")):
        login(token=tok, add_to_git_credential=True)
    GGUF = hf_hub_download(repo_id=repo_id,
                           filename=filename,
                           resume_download=True)

gemma = Llama(model_path=GGUF,
              n_ctx=40_960,
              n_gpu_layers=-1,
              verbose=False)

# ─────────────────────────────────────────────────────────────
# Prompt templates
# ─────────────────────────────────────────────────────────────
GROUP_PROMPT = (
    "You are an expert categorizer.\n"
    "Group the following theme labels into exactly 10 clusters.\n"
    "Give each cluster a concise name (≤15 words, Title Case, no punctuation).\n\n"
    "Return JSON like:\n"
    "[{\"label\": \"<label>\", \"cluster\": \"<cluster name>\"}, …]\n"
)

CONF_PROMPT = (
    "Rate how well a theme label fits a cluster name.\n"
    "Respond with ONLY a number between 0 and 1.\n\n"
    "Label: {lbl}\nCluster: {cl}\n<score>"
)

RX_JSON = re.compile(r"\[[\s\S]+?]")        # first JSON array in Gemma output
RX_FLOAT = re.compile(r"0\.\d+|1\.0+")
MAX_LABEL_LEN = 70                          # chars sent to LLM

# ─────────────────────────────────────────────────────────────
# Gemma helpers
# ─────────────────────────────────────────────────────────────
def gemma_chat(prompt: str,
               temperature: float = 0.2,
               max_tokens: int = 2048) -> str:
    return gemma.create_chat_completion(
        messages=[{"role": "user", "content": prompt}],
        temperature=temperature,
        max_tokens=max_tokens
    )["choices"][0]["message"]["content"].strip()


def cluster_with_prompt(labels, batch_size: int):
    """
    Returns dict: full_label -> cluster_name
    """
    # Trim very long labels so Gemma prompt stays small
    full2trim = {l: (l if len(l) <= MAX_LABEL_LEN else l[:MAX_LABEL_LEN] + "…")
                 for l in labels}
    trim2full = {v: k for k, v in full2trim.items()}

    mapping = {}
    for start in tqdm(range(0, len(labels), batch_size),
                      mininterval=1,
                      desc="Clustering batches"):
        chunk_trim = [full2trim[l] for l in labels[start:start + batch_size]]
        prompt = GROUP_PROMPT + "\n" + "\n".join(f"- {t}" for t in chunk_trim)
        resp = gemma_chat(prompt)

        m = RX_JSON.search(resp)
        if not m:
            raise ValueError("Gemma did not return parseable JSON.")

        for item in json.loads(m.group(0)):
            full = trim2full.get(item["label"])
            if full:
                mapping[full] = item["cluster"]
    return mapping


def confidence(lbl: str, cluster: str) -> float:
    """Return LLM confidence 0-1."""
    resp = gemma_chat(CONF_PROMPT.format(lbl=lbl, cl=cluster),
                      temperature=0.0,
                      max_tokens=8)
    m = RX_FLOAT.search(resp)
    return float(m.group(0)) if m else 0.5

# ─────────────────────────────────────────────────────────────
# CLI parsing
# ─────────────────────────────────────────────────────────────
def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("pred_jsonl",
                   help="Input .jsonl with theme_label_predicted in last turn")
    p.add_argument("--out", default="clustered_prompt.csv")
    p.add_argument("--plot", default=None, help="PNG output (optional)")
    p.add_argument("--batch-size", type=int, default=40)
    p.add_argument("--threshold", type=float, default=0.0,
                   help="score < threshold → Unclustered (0 = disable)")
    p.add_argument("--plot-cpu", action="store_true",
                   help="Embed plot points on CPU to avoid GPU OOM")
    p.add_argument("--embed-model",
                   default="sentence-transformers/all-mpnet-base-v2")
    return p.parse_args()

# ─────────────────────────────────────────────────────────────
# Main routine
# ─────────────────────────────────────────────────────────────
def main():
    args = parse_args()

    # 1. Count every predicted label
    counter = Counter()
    with open(args.pred_jsonl, encoding="utf-8") as fh:
        for line in fh:
            lbl = json.loads(line)["turns"][-1].get(
                "theme_label_predicted", "").strip()
            if lbl:
                counter[lbl] += 1

    if not counter:
        raise ValueError("No theme_label_predicted found.")

    top10 = [lbl for lbl, _ in counter.most_common(10)]
    print(f"Total unique labels : {len(counter):,}")
    print("Top-10 (by frequency):")
    for i, lbl in enumerate(top10, 1):
        print(f"  {i:2d}. {lbl}  (count={counter[lbl]})")

    # 2. Cluster the top-10 only
    label2cluster = cluster_with_prompt(top10,
                                        batch_size=args.batch_size)

    # 3. Confidence scoring and final cluster assignment
    rows = []
    dropped = 0
    print("Scoring confidence …")
    for lbl in tqdm(counter, mininterval=1):
        if lbl in top10:
            cluster = label2cluster.get(lbl, "Unclustered")
            score = round(confidence(lbl, cluster), 3)
            if args.threshold and score < args.threshold:
                cluster = "Unclustered"
                dropped += 1
            rows.append((lbl, cluster, score))
        else:
            rows.append((lbl, "Others", 0.00))

    if dropped:
        print(f"{dropped} high-freq labels scored below threshold → Unclustered.")

    # 4. Write CSV
    with open(args.out, "w", newline="", encoding="utf-8") as fo:
        csv.writer(fo).writerows(
            [("theme_label_predicted", "theme_clustered", "score"), *rows]
        )
    print("CSV saved →", args.out)

    # 5. Optional plot
    if args.plot:
        device = "cpu" if args.plot_cpu else "cuda"
        try:
            embedder = SentenceTransformer(args.embed_model, device=device)
        except RuntimeError:
            print("Falling back to CPU for embeddings.")
            embedder = SentenceTransformer(args.embed_model, device="cpu")

        labels = list(counter.keys())
        embs = embedder.encode(labels,
                               normalize_embeddings=True,
                               show_progress_bar=True)
        coords = PCA(n_components=2, random_state=42).fit_transform(embs)

        # Colour map per cluster
        cluster_set = sorted({cl for _, cl, _ in rows})
        cmap = {cl: (f"C{i % 9 + 1}" if cl != "Others" else "C0")
                for i, cl in enumerate(cluster_set)}

        plt.figure(figsize=(8, 6))
        for (x, y), lbl in zip(coords, labels):
            cl = next(r[1] for r in rows if r[0] == lbl)
            plt.scatter(x, y, color=cmap[cl], s=35)
        plt.title("Top-10 Clusters (+ Others) — PCA-2D")
        plt.xlabel("PCA-1")
        plt.ylabel("PCA-2")

        # Legend at the bottom
        handles = [plt.Line2D([0], [0], marker='o', color='w',
                              markerfacecolor=cmap[cl],
                              markersize=8, label=cl)
                   for cl in cluster_set]
        plt.legend(title="Cluster",
                   handles=handles,
                   bbox_to_anchor=(0.5, -0.15),
                   loc='upper center',
                   ncol=3)
        plt.tight_layout()
        plt.savefig(args.plot, dpi=160)
        print("Plot saved →", args.plot)


if __name__ == "__main__":
    main()

