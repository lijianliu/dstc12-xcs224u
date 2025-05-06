#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Cluster predicted theme labels into N groups, name each group with Gemma,
push labels below a similarity threshold into 'Others', and optionally plot.

Example
-------
python cluster_theme_labels.py pulse_gemma_pred.jsonl \
       --clusters 10 \
       --out pulse_clusters.csv \
       --plot pulse_clusters.png
"""

import json, csv, argparse, os, re
from collections import defaultdict

import numpy as np
from sentence_transformers import SentenceTransformer, util
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA, HDBSCAN
import matplotlib.pyplot as plt
from huggingface_hub import hf_hub_download, login
from llama_cpp import Llama


# ── Load / download Gemma-3-27B GGUF ─────────────────────────────────────────
GGUF = os.getenv("GEMMA_PATH")
if GGUF is None:
    repo_id, gguf_file = "google/gemma-3-27b-it-qat-q4_0-gguf", "gemma-3-27b-it-q4_0.gguf"
    if (tok := os.getenv("HF_TOKEN")):
        login(token=tok, add_to_git_credential=True)
    GGUF = hf_hub_download(repo_id=repo_id, filename=gguf_file, resume_download=True)

gemma = Llama(model_path=GGUF, n_ctx=4096, n_gpu_layers=-1, verbose=False)
PROMPT_TEMPLATE = (
    "You are an expert at naming categories.\n"
    "Given the theme labels below, return ONE concise cluster name "
    "(≤15 words, Title Case, no punctuation). Avoid too general words such as 'data', 'issue'\n\n"
    "{labels}\n\n<cluster_name>"
)
rx_first_line = re.compile(r"^[^\n<]{1,60}", re.S)


def llm_name(labels):
    joined = "\n".join(f"- {l}" for l in labels)
    resp = gemma.create_chat_completion(
        messages=[{"role": "user", "content": PROMPT_TEMPLATE.format(labels=joined)}],
        temperature=0.3,
        max_tokens=16,
    )["choices"][0]["message"]["content"].strip()
    m = rx_first_line.match(resp)
    return m.group(0).strip() if m else resp[:60].strip()


# ── CLI ──────────────────────────────────────────────────────────────────────
def get_args():
    p = argparse.ArgumentParser()
    p.add_argument("pred_jsonl", help="prediction JSONL file with theme_label_predicted")
    p.add_argument("--clusters", type=int, default=10, help="number of K-means clusters")
    p.add_argument("--embed-model", default="sentence-transformers/all-mpnet-base-v2")
    p.add_argument("--other-threshold", type=float, default=0.50,
                   help="labels with similarity < threshold go to 'Others'")
    p.add_argument("--out", default="clustered_labels.csv", help="output CSV")
    p.add_argument("--plot", default=None, help="PNG output for scatter plot")
    return p.parse_args()


# ── Main routine ────────────────────────────────────────────────────────────
def main():
    args = get_args()

    # 1. Collect unique predicted labels
    labels = {json.loads(l)["turns"][-1].get("theme_label_predicted", "").strip()
              for l in open(args.pred_jsonl, encoding="utf-8")}
    labels.discard("")
    labels = sorted(labels)
    if not labels:
        raise ValueError("No theme_label_predicted found in the JSONL file.")
    print(f"Unique labels collected: {len(labels)}")

    # 2. Embed labels
    embedder = SentenceTransformer(args.embed_model)
    embeddings = embedder.encode(labels, normalize_embeddings=True, show_progress_bar=True)

    # 3. K-means clustering if not using HDBSCAN
    if args.usedbscan:
        print("Using HDBSCAN clustering")
        # Optional: HDBSCAN clustering
        # Note to Lijian: Tune min_cluster_size first, then min_samples.
        # Explanation of parameters:
        # - min_samples: Minimum number of samples in a neighborhood for a point to be considered a core point.
        # - min_cluster_size: Minimum size of a cluster.
        # - metric: Distance metric to use. 'cosine' is good for text embeddings.
        # - cluster_selection_method: 'eom' for the excess of mass method, 'leaf' for the leaf method.
        # See further details: https://scikit-learn.org/stable/auto_examples/cluster/plot_hdbscan.html
        hdbscan = HDBSCAN(min_samples=5,
                          min_cluster_size=10,
                          metric="cosine",
                          store_centers="centroid",
                          cluster_selection_method="eom")
        hdbscan.fit(embeddings)
        assignments = hdbscan.labels_
        centers = hdbscan.centroids_

    else:
        print("Using K-means clustering")
        # K-means clustering
        kmeans = KMeans(n_clusters=args.clusters, n_init=10, random_state=42)
        kmeans.fit(embeddings)
        assignments = kmeans.labels_
        centers = kmeans.cluster_centers_

    # 4. Name each cluster with Gemma
    cluster_to_labels = defaultdict(list)
    for lbl, cid in zip(labels, assignments):
        cluster_to_labels[cid].append(lbl)
    cluster_names = {cid: llm_name(lbs) for cid, lbs in cluster_to_labels.items()}

    # 5. Compute similarity and assign 'Others'
    sims = util.cos_sim(np.asarray(embeddings), np.asarray(centers)).max(dim=1).values
    max_sim = sims.max().item()
    csv_rows, color_list = [], []
    for lbl, cid, vec, sim in zip(labels, assignments, embeddings, sims):
        norm = sim / max_sim
        if norm < args.other_threshold:
            csv_rows.append((lbl, "Others", 0.00))
            color_list.append("grey")
        else:
            csv_rows.append((lbl, cluster_names[cid], round(float(norm), 3)))
            color_list.append(f"C{cid % 10}")

    # 6. Write CSV
    with open(args.out, "w", newline="", encoding="utf-8") as fo:
        writer = csv.writer(fo)
        writer.writerow(["theme_label_predicted", "theme_clustered", "score"])
        writer.writerows(sorted(csv_rows, key=lambda x: x[1]))
    print("CSV saved →", args.out)

    # 7. Optional 2-D scatter plot
    if args.plot:
        coords = PCA(n_components=2, random_state=42).fit_transform(embeddings)
        plt.figure(figsize=(8, 6))

        # Color map including 'Others'
        unique_clusters = sorted({row[1] for row in csv_rows})
        color_map = {name: f"C{i % 10}" for i, name in enumerate(c for c in unique_clusters if c != "Others")}
        color_map["Others"] = "grey"

        # Plot each point
        for (x, y), (_, cl, _) in zip(coords, csv_rows):
            plt.scatter(x, y, color=color_map[cl], s=35)

        # Plot centroids (skip 'Others')
        cent2d = PCA(n_components=2, random_state=42).fit_transform(centers)
        for cid, (cx, cy) in enumerate(cent2d):
            if cid in cluster_names:
                plt.scatter(cx, cy, marker="x", color="black", s=80)

        plt.title("Theme-label clusters (PCA 2-D)")
        plt.xlabel("PCA-1")
        plt.ylabel("PCA-2")

        # Legend
        handles = [plt.Line2D([0], [0], marker='o', color='w',
                              markerfacecolor=color_map[n], markersize=8, label=n)
                   for n in unique_clusters]
        plt.legend(title="Cluster", handles=handles, bbox_to_anchor=(1.05, 1), loc='upper left')

        plt.tight_layout()
        plt.savefig(args.plot, dpi=160)
        print("Plot saved →", args.plot)
        plt.show()
    elif args.plot3d:
        coords = PCA(n_components=3, random_state=42).fit_transform(embeddings)
        fig = plt.figure(figsize=(8, 6))
        ax = fig.add_subplot(111, projection='3d')
        ax.scatter(coords[:, 0], coords[:, 1], coords[:, 2], c=color_list, s=35)

        # Plot centroids (skip 'Others')
        cent3d = PCA(n_components=3, random_state=42).fit_transform(centers)
        for cid, (cx, cy, cz) in enumerate(cent3d):
            if cid in cluster_names:
                ax.scatter(cx, cy, cz, marker="x", color="black", s=80)

        ax.set_title("Theme-label clusters (PCA 3-D)")
        ax.set_xlabel("PCA-1")
        ax.set_ylabel("PCA-2")
        ax.set_zlabel("PCA-3")

        # Legend
        handles = [plt.Line2D([0], [0], marker='o', color='w',
                              markerfacecolor=color_map[n], markersize=8, label=n)
                   for n in unique_clusters]
        ax.legend(title="Cluster", handles=handles, bbox_to_anchor=(1.05, 1), loc='upper left')
        ax.view_init(elev=20, azim=30)

        plt.tight_layout()
        plt.savefig(args.plot, dpi=160)
        print("Plot saved →", args.plot)
        plt.show()

if __name__ == "__main__":
    main()

