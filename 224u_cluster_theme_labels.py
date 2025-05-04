#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Cluster theme_label_predicted strings into 10 groups,
name each group with local Gemma-3-27B, and output CSV with confidence scores.
"""

import json, csv, argparse, os, glob, re
from collections import defaultdict

from huggingface_hub import hf_hub_download, login
from llama_cpp import Llama
from sentence_transformers import SentenceTransformer, util
from sklearn.cluster import KMeans
import numpy as np
from tqdm import tqdm

# ---------- load / locate Gemma gguf -----------------------------------------
GGUF = os.getenv("GEMMA_PATH")
if GGUF is None:
    repo_id, fn = "google/gemma-3-27b-it-qat-q4_0-gguf", "gemma-3-27b-it-q4_0.gguf"
    if (tok := os.getenv("HF_TOKEN")):
        login(token=tok, add_to_git_credential=True)
    GGUF = hf_hub_download(repo_id=repo_id, filename=fn, resume_download=True)

gemma = Llama(model_path=GGUF, n_ctx=4096, n_gpu_layers=-1, verbose=False)
name_prompt = (
    "You are an expert at naming categories.\n"
    "Given a list of short theme labels, return ONE concise cluster name "
    "(≤9 words, Title Case, no punctuation) that best summarises them.\n\n"
    "Labels:\n{labels}\n\n<cluster_name>"
)
rx = re.compile(r"^[^\n<]{1,40}", re.S)   # first non-newline chunk


def llm_name(labels):
    txt = "\n".join(f"- {l}" for l in labels)
    out = gemma.create_chat_completion(
        messages=[{"role": "user", "content": name_prompt.format(labels=txt)}],
        temperature=0.3,
        max_tokens=16,
    )["choices"][0]["message"]["content"].strip()
    m = rx.match(out)
    return m.group(0).strip() if m else out[:40].strip()


# ---------- CLI --------------------------------------------------------------
def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("pred_jsonl", help="file produced by run_theme_detection_onecall.py")
    p.add_argument("--embed-model", default="sentence-transformers/all-mpnet-base-v2")
    p.add_argument("--clusters", type=int, default=10)
    p.add_argument("--out", default="clustered_labels.csv")
    return p.parse_args()


def main():
    args = parse_args()

    # 1) collect unique predicted labels
    labels = []
    with open(args.pred_jsonl, encoding="utf-8") as f:
        for line in f:
            dlg = json.loads(line)
            lab = dlg["turns"][-1].get("theme_label_predicted", "").strip()
            if lab:
                labels.append(lab)

    unique_labels = sorted(set(labels))
    print(f"Found {len(unique_labels)} unique predicted labels.")

    # 2) embed & cluster
    emb_model = SentenceTransformer(args.embed_model)
    embs = emb_model.encode(unique_labels, normalize_embeddings=True, show_progress_bar=True)
    kmeans = KMeans(n_clusters=args.clusters, n_init=10, random_state=42)
    kmeans.fit(embs)
    assignments = kmeans.labels_
    centroids = kmeans.cluster_centers_

    # 3) name each cluster with Gemma
    cluster_to_labels = defaultdict(list)
    for lbl, cid in zip(unique_labels, assignments):
        cluster_to_labels[cid].append(lbl)

    cluster_names = {}
    for cid, labs in tqdm(cluster_to_labels.items(), desc="Naming clusters"):
        cluster_names[cid] = llm_name(labs)

    # 4) compute confidence scores
    max_sim = 0.0
    label_rows = []
    for lbl, cid, vec in zip(unique_labels, assignments, embs):
        sim = float(util.cos_sim(vec, centroids[cid]))
        max_sim = max(max_sim, sim)
        label_rows.append((lbl, cid, sim))

    # normalise to 0–1
    for i, (lbl, cid, sim) in enumerate(label_rows):
        label_rows[i] = (lbl, cluster_names[cid], round(sim / max_sim, 3))

    # 5) write CSV
    with open(args.out, "w", newline="", encoding="utf-8") as fo:
        w = csv.writer(fo)
        w.writerow(["theme_label_predicted", "theme_clustered", "score"])
        w.writerows(sorted(label_rows, key=lambda x: x[1]))  # sort by cluster name

    print("✅ CSV saved to", args.out)


if __name__ == "__main__":
    main()

