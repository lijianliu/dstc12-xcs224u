#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Prompt-defined top-10 clusters; embed-based assignment for the rest.

1.  Count theme_label_predicted strings.
2.  Select the 10 most-common → send to Gemma-3-27B-IT with the original
    prompt to obtain 10 cluster names.
3.  For each of those 10 frequent labels, remember which cluster Gemma gave it.
4.  Embed **all** labels (Sentence-Transformers MPNet).
5.  For every non-top-10 label:
       • Compute cosine similarity to the *centroid* of each Gemma cluster
         (centroid = mean of embeddings of labels Gemma put there).
       • If the best sim ≥ --sim-threshold (default 0.40) → join that cluster.
       • Otherwise → cluster “Others”.
6.  Write CSV  [theme_label_predicted, theme_clustered, score]
    where score = 1.00 for the 10 frequent labels (they’re the anchors) and
    = max cosine similarity for all other labels (0 if “Others”).
7.  Optional PCA plot.

Run example
-----------
python 224u_prompt_cluster_theme_labels.py preds.jsonl           \\
       --out clusters.csv --plot clusters.png --sim-threshold 0.40
"""

import argparse
import csv
import json
import os
import re
from collections import Counter, defaultdict

import matplotlib.pyplot as plt
import numpy as np
from huggingface_hub import hf_hub_download, login
from llama_cpp import Llama
from sentence_transformers import SentenceTransformer
from sklearn.decomposition import PCA
from sklearn.metrics.pairwise import cosine_similarity
from tqdm import tqdm

# ─────────────────────────────────────────────────────────────
# Gemma loading (unchanged from your original script)
# ─────────────────────────────────────────────────────────────
GGUF = os.getenv("GEMMA_PATH")
if GGUF is None:
    repo_id = "google/gemma-3-27b-it-qat-q4_0-gguf"
    filename = "gemma-3-27b-it-q4_0.gguf"
    if (tok := os.getenv("HF_TOKEN")):
        login(token=tok, add_to_git_credential=True)
    GGUF = hf_hub_download(repo_id=repo_id, filename=filename,
                           resume_download=True)

gemma = Llama(model_path=GGUF,
              n_ctx=40_960,
              n_gpu_layers=-1,
              verbose=False)

GROUP_PROMPT = (
    "You are an expert categorizer.\n"
    "Group the following theme labels into exactly 10 clusters.\n"
    "Give each cluster a concise name (≤15 words, Title Case, no punctuation).\n\n"
    "Return JSON like:\n"
    "[{\"label\": \"<label>\", \"cluster\": \"<cluster name>\"}, …]\n"
)

RX_JSON = re.compile(r"\[[\s\S]+?]")
MAX_LABEL_LEN = 70

def gemma_chat(prompt, temperature=0.2, max_tokens=2048):
    return gemma.create_chat_completion(
        messages=[{"role": "user", "content": prompt}],
        temperature=temperature,
        max_tokens=max_tokens
    )["choices"][0]["message"]["content"].strip()

def cluster_with_prompt(labels):
    """Return dict(label -> cluster_name) using a single Gemma call."""
    full2trim = {l: (l if len(l) <= MAX_LABEL_LEN else l[:MAX_LABEL_LEN] + "…")
                 for l in labels}
    prompt = GROUP_PROMPT + "\n" + "\n".join(f"- {t}"
                                             for t in full2trim.values())
    resp = gemma_chat(prompt)
    m = RX_JSON.search(resp)
    if not m:
        raise ValueError("Gemma did not return parseable JSON.")

    mapping = {}
    trim2full = {v: k for k, v in full2trim.items()}
    for item in json.loads(m.group(0)):
        full = trim2full.get(item["label"])
        if full:
            mapping[full] = item["cluster"]
    if len(mapping) != len(labels):
        raise ValueError("Gemma response missing some labels.")
    return mapping

# ─────────────────────────────────────────────────────────────
# CLI
# ─────────────────────────────────────────────────────────────
def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("pred_jsonl")
    p.add_argument("--out", default="clustered_prompt.csv")
    p.add_argument("--plot", default=None)
    p.add_argument("--batch-size", type=int, default=40,
                    help="(ignored now: kept for backward compatibility)")    
    p.add_argument("--embed-model",
                   default="sentence-transformers/all-mpnet-base-v2")
    p.add_argument("--sim-threshold", type=float, default=0.40)
    p.add_argument("--plot-cpu", action="store_true")
    return p.parse_args()

# ─────────────────────────────────────────────────────────────
# Helpers
# ─────────────────────────────────────────────────────────────
def load_counter(path):
    c = Counter()
    with open(path, encoding="utf-8") as fh:
        for ln in fh:
            lbl = json.loads(ln)["turns"][-1].get("theme_label_predicted", "").strip()
            if lbl:
                c[lbl] += 1
    if not c:
        raise ValueError("No theme_label_predicted found.")
    return c

def embed(list_of_strings, model_name, device):
    model = SentenceTransformer(model_name, device=device)
    return model.encode(list_of_strings,
                        normalize_embeddings=True,
                        show_progress_bar=True)

# ─────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────
def main():
    args = parse_args()

    # 1. count labels
    counter = load_counter(args.pred_jsonl)
    anchors = [lbl for lbl, _ in counter.most_common(10)]
    print(f"Unique labels: {len(counter):,}")
    print("Most-frequent 10 (to send to Gemma):")
    for i, lbl in enumerate(anchors, 1):
        print(f"  {i:2d}. {lbl}  (count={counter[lbl]})")

    # 2. prompt Gemma to group those 10 labels
    print("\nGemma: generating cluster names …")
    anchor2cluster = cluster_with_prompt(anchors)
    clusters = sorted(set(anchor2cluster.values()))
    print("Gemma clusters:")
    for cl in clusters:
        members = [l for l, c in anchor2cluster.items() if c == cl]
        print(f"  {cl}: {members}")

    # 3. embed ALL labels
    print("\nEmbedding all labels …")
    device = "cpu" if args.plot_cpu else "cuda"
    labels = list(counter.keys())
    embs = embed(labels, args.embed_model, device=device)

    # 4. cluster centroids (mean of anchor embeddings per Gemma cluster)
    idx_anchor = {lbl: labels.index(lbl) for lbl in anchors}
    cluster_centroid = {}
    for cl in clusters:
        idxs = [idx_anchor[lbl] for lbl, cln in anchor2cluster.items()
                if cln == cl]
        cluster_centroid[cl] = embs[idxs].mean(axis=0, keepdims=True)

    # 5. assign each label
    rows = []
    for lbl, emb in zip(labels, embs):
        if lbl in anchor2cluster:                # one of the original 10
            rows.append((lbl, anchor2cluster[lbl], 1.00))
            continue

        # find nearest centroid
        sims = {cl: float(cosine_similarity(emb.reshape(1, -1),
                                            cluster_centroid[cl])[0, 0])
                for cl in clusters}
        best_cl, best_sim = max(sims.items(), key=lambda kv: kv[1])
        if best_sim >= args.sim_threshold:
            rows.append((lbl, best_cl, round(best_sim, 3)))
        else:
            rows.append((lbl, "Others", 0.00))

    # 6. write CSV
    with open(args.out, "w", newline="", encoding="utf-8") as fo:
        csv.writer(fo).writerows(
            [("theme_label_predicted", "theme_clustered", "score"), *rows]
        )
    print(f"\nCSV saved → {args.out}")

    # 7. optional plot
    if args.plot:
        print("Building PCA plot …")
        coords = PCA(n_components=2, random_state=42).fit_transform(embs)
        all_clusters = sorted({r[1] for r in rows},
                              key=lambda x: (0, clusters.index(x))
                              if x in clusters else (1, x))
        cmap = {cl: (f"C{i % 9 + 1}" if cl != "Others" else "C0")
                for i, cl in enumerate(all_clusters)}
        plt.figure(figsize=(8, 6))
        for (x, y), (_, cl, _) in zip(coords, rows):
            plt.scatter(x, y, color=cmap[cl], s=35)
        plt.title("Prompt-Defined Top-10 Clusters (+ Others) — PCA-2D")
        plt.xlabel("PCA-1"); plt.ylabel("PCA-2")
        handles = [plt.Line2D([0], [0], marker='o', color='w',
                              markerfacecolor=cmap[cl],
                              markersize=8, label=cl)
                   for cl in all_clusters]
        plt.legend(title="Cluster",
                   handles=handles,
                   bbox_to_anchor=(0.5, -0.15),
                   loc='upper center',
                   ncol=3)
        plt.tight_layout(); plt.savefig(args.plot, dpi=160)
        print(f"Plot saved → {args.plot}")

if __name__ == "__main__":
    main()
