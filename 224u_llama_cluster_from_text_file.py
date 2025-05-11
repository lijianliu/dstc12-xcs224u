#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Cluster predicted theme labels into N groups, name each group with LLaMA-4,
push labels below a similarity threshold into 'Others', and optionally plot.

Example
-------
python 224u_llama_cluster.py pulse_labels.txt \
       --clusters 10 \
       --out pulse_clusters.csv \
       --plot pulse_clusters.png
"""

import json, csv, argparse, os, re
from collections import defaultdict

import numpy as np
from sentence_transformers import SentenceTransformer, util
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
import hdbscan
import matplotlib.pyplot as plt

# LLaMA-4 model setup
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig

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

SYSTEM_PROMPT = (
    "You are an expert at naming categories.\n"
    "Given the theme labels below, return ONE concise cluster name "
    "(≤15 words, Title Case, no punctuation). Avoid too general words such as 'data', 'issue'\n\n"
)
rx_first_line = re.compile(r"^[^\n<]{1,60}", re.S)

def llm_name(labels):
    if len(labels) > 25:
        print(f"⚠️ Truncating cluster with {len(labels)} labels → 25")
        labels = labels[:25]
    joined = "\n".join(f"- {l}" for l in labels)
    prompt = f"<|system|>\n{SYSTEM_PROMPT}\n<|user|>\n{joined}\n<|assistant|>\n"
    inputs = tokenizer(prompt, return_tensors="pt", truncation=True, max_length=4096).to("cuda:0")

    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=32,
            temperature=0.3,
            top_p=0.9
        )
    raw_reply = tokenizer.decode(outputs[0], skip_special_tokens=True)
    raw_reply = raw_reply.split("<|assistant|>")[-1].strip()
    m = rx_first_line.match(raw_reply)
    return m.group(0).strip() if m else raw_reply[:60].strip()

def get_args():
    p = argparse.ArgumentParser()
    p.add_argument("label_file", help="Text file with one label per line")
    p.add_argument("--clusters", type=int, default=10, help="number of K-means clusters")
    p.add_argument("--embed-model", default="sentence-transformers/all-mpnet-base-v2")
    p.add_argument("--other-threshold", type=float, default=0.50,
                   help="labels with similarity < threshold go to 'Others'")
    p.add_argument("--out", default="clustered_labels.csv", help="output CSV")
    p.add_argument("--plot", default=None, help="PNG output for scatter plot")
    p.add_argument("--plot3d", action="store_true", help="Optional: 3D plot")
    p.add_argument("--usedbscan", action="store_true", help="Use HDBSCAN instead of KMeans")
    return p.parse_args()

def main():
    args = get_args()

    with open(args.label_file, encoding="utf-8") as f:
        labels = [line.strip() for line in f if line.strip()]
    labels = sorted(set(labels))

    if not labels:
        raise ValueError("No labels found in the text file.")
    print(f"Unique labels collected: {len(labels)}")

    embedder = SentenceTransformer(args.embed_model)
    embeddings = embedder.encode(labels, normalize_embeddings=True, show_progress_bar=True)

    if args.usedbscan:
        print("Using HDBSCAN clustering")
        clusterer = hdbscan.HDBSCAN(min_samples=5, min_cluster_size=10, metric="euclidean", cluster_selection_method="eom")
        assignments = clusterer.fit_predict(embeddings)
        cluster_to_indices = defaultdict(list)
        for i, cid in enumerate(assignments):
            if cid >= 0:
                cluster_to_indices[cid].append(i)
        centers = []
        for cid in sorted(cluster_to_indices):
            idxs = cluster_to_indices[cid]
            centers.append(np.mean([embeddings[i] for i in idxs], axis=0))
        centers = np.vstack(centers)
    else:
        print("Using K-means clustering")
        kmeans = KMeans(n_clusters=args.clusters, n_init=10, random_state=42)
        kmeans.fit(embeddings)
        assignments = kmeans.labels_
        centers = kmeans.cluster_centers_

    cluster_to_labels = defaultdict(list)
    for lbl, cid in zip(labels, assignments):
        cluster_to_labels[cid].append(lbl)
    cluster_names = {cid: llm_name(lbs) for cid, lbs in cluster_to_labels.items() if cid != -1}

    sims = util.cos_sim(np.asarray(embeddings), np.asarray(centers)).max(dim=1).values
    max_sim = sims.max().item()

    csv_rows, color_list = [], []
    for lbl, cid, sim in zip(labels, assignments, sims):
        norm = sim / max_sim if max_sim else 1.0
        if cid == -1 or norm < args.other_threshold:
            csv_rows.append((lbl, "Others", 0.00))
            color_list.append("grey")
        else:
            csv_rows.append((lbl, cluster_names[cid], round(float(norm), 3)))
            color_list.append(f"C{cid % 10}")

    with open(args.out, "w", newline="", encoding="utf-8") as fo:
        writer = csv.writer(fo)
        writer.writerow(["theme_label_predicted", "theme_clustered", "score"])
        writer.writerows(sorted(csv_rows, key=lambda x: x[1]))
    print("CSV saved →", args.out)

    if args.plot:
        coords = PCA(n_components=2, random_state=42).fit_transform(embeddings)
        plt.figure(figsize=(8, 6))
        unique_clusters = sorted({row[1] for row in csv_rows})
        color_map = {name: f"C{i % 10}" for i, name in enumerate(c for c in unique_clusters if c != "Others")}
        color_map["Others"] = "grey"
        for (x, y), (_, cl, _) in zip(coords, csv_rows):
            plt.scatter(x, y, color=color_map[cl], s=35)
        cent2d = PCA(n_components=2, random_state=42).fit_transform(centers)
        for cid, (cx, cy) in enumerate(cent2d):
            if cid in cluster_names:
                plt.scatter(cx, cy, marker="x", color="black", s=80)
        plt.title("Theme-label clusters (PCA 2-D)")
        plt.xlabel("PCA-1")
        plt.ylabel("PCA-2")
        handles = [plt.Line2D([0], [0], marker='o', color='w',
                              markerfacecolor=color_map[n], markersize=8, label=n)
                   for n in unique_clusters]
        plt.legend(title="Cluster", handles=handles, bbox_to_anchor=(1.05, 1), loc='upper left')
        plt.tight_layout()
        plt.savefig(args.plot, dpi=160)
        print("Plot saved →", args.plot)
        plt.show()

if __name__ == "__main__":
    main()

