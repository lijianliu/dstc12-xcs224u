#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Prompt-only clustering of theme_label_predicted.

Workflow
--------
1. Read a prediction JSONL produced by run_theme_detection_onecall.py.
2. Collect the unique theme_label_predicted strings.
3. Chunk those labels (default 100 per batch) and ask Gemma-3-27B to
   • assign each label to exactly 10 clusters
   • provide a concise name for each cluster.
4. For every (label, cluster) pair ask Gemma for a 0-1 confidence score.
5. Write a CSV  [label, cluster_name, score].
6. (Optional) plot a 2-D PCA scatter chart for visual inspection.
"""

import json, csv, argparse, os, re
from collections import defaultdict

import numpy as np
from sentence_transformers import SentenceTransformer
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt
from huggingface_hub import hf_hub_download, login
from llama_cpp import Llama
from tqdm import tqdm

# -------------------------------------------------------------------- #
# Locate / download Gemma-3-27B GGUF
# -------------------------------------------------------------------- #
GGUF = os.getenv("GEMMA_PATH")           # set env to skip download
if GGUF is None:
    repo_id, file_name = "google/gemma-3-27b-it-qat-q4_0-gguf", "gemma-3-27b-it-q4_0.gguf"
    if (tok := os.getenv("HF_TOKEN")):
        login(token=tok, add_to_git_credential=True)
    GGUF = hf_hub_download(repo_id=repo_id, filename=file_name, resume_download=True)

gemma = Llama(model_path=GGUF, n_ctx=40960, n_gpu_layers=-1, verbose=False)

# -------------------------------------------------------------------- #
# Prompt templates
# -------------------------------------------------------------------- #
GROUP_PROMPT = (
    "You are an expert categorizer.\n"
    "Group the following theme labels into exactly 10 clusters.\n"
    "For EACH cluster, provide a concise name (≤4 words, Title Case, "
    "no punctuation).\n\n"
    "Return your answer as a JSON array; each element must be an object like:\n"
    "{\"label\": \"<original label>\", \"cluster\": \"<cluster name>\"}\n"
)

CONF_PROMPT = (
    "You are judging how well a theme label fits a cluster name.\n"
    "Respond with ONLY a number between 0 and 1 (1 = perfect fit).\n\n"
    "Label: {lbl}\n"
    "Cluster: {cl}\n"
    "<score>"
)

rx_json  = re.compile(r"\[[\s\S]+]")          # first JSON array in the text
rx_float = re.compile(r"0\.\d+|1\.0+")


def gemma_chat(prompt: str, temperature=0.2, max_tokens=1024) -> str:
    """Helper to call Gemma via llama-cpp."""
    return gemma.create_chat_completion(
        messages=[{"role": "user", "content": prompt}],
        temperature=temperature,
        max_tokens=max_tokens
    )["choices"][0]["message"]["content"].strip()


# -------------------------------------------------------------------- #
# Prompt-based clustering (batched)
# -------------------------------------------------------------------- #
def cluster_with_prompt(labels, batch_size=100):
    """Return {label: cluster_name}."""
    mapping = {}
    for start in range(0, len(labels), batch_size):
        batch  = labels[start:start + batch_size]
        prompt = GROUP_PROMPT + "\n".join(f"- {l}" for l in batch)
        resp   = gemma_chat(prompt, temperature=0.2, max_tokens=2048)

        m = rx_json.search(resp)
        if not m:
            print("\n–– RAW GEMMA OUTPUT (batch", start//batch_size + 1, ") ––\n",
                  resp[:600], "\n–– END ––\n")
            raise ValueError("Gemma did not return valid JSON for the batch above.")

        part = json.loads(m.group(0))
        mapping.update({d["label"]: d["cluster"] for d in part})
    return mapping


def confidence_score(label, cluster) -> float:
    """Return float in [0,1] using an LLM call."""
    resp = gemma_chat(CONF_PROMPT.format(lbl=label, cl=cluster),
                      temperature=0.0, max_tokens=8)
    m = rx_float.search(resp)
    return float(m.group(0)) if m else 0.5


# -------------------------------------------------------------------- #
# CLI
# -------------------------------------------------------------------- #
def get_args():
    p = argparse.ArgumentParser()
    p.add_argument("pred_jsonl", help="prediction file with theme_label_predicted")
    p.add_argument("--out",  default="clustered_prompt.csv", help="output CSV")
    p.add_argument("--plot", default=None, help="PNG filename for scatter plot")
    p.add_argument("--embed-model", default="sentence-transformers/all-mpnet-base-v2",
                   help="embedding model (only for plotting)")
    p.add_argument("--batch-size", type=int, default=100, help="labels per LLM call")
    return p.parse_args()


# -------------------------------------------------------------------- #
# Main
# -------------------------------------------------------------------- #
def main():
    args = get_args()

    # 1. Collect unique labels
    labels = {json.loads(line)["turns"][-1].get("theme_label_predicted", "").strip()
              for line in open(args.pred_jsonl, encoding="utf-8")}
    labels.discard("")
    labels = sorted(labels)
    if not labels:
        raise ValueError("No theme_label_predicted found.")
    print(f"Unique labels: {len(labels)}")

    # 2. Cluster via prompt
    print("→ Asking Gemma to cluster in batches …")
    label_to_cluster = cluster_with_prompt(labels, batch_size=args.batch_size)

    # 3. Confidence per label
    rows = []
    print("→ Getting confidence scores …")
    for lbl in tqdm(labels):
        cl  = label_to_cluster[lbl]
        sc  = confidence_score(lbl, cl)
        rows.append((lbl, cl, round(sc, 3)))

    # 4. Write CSV
    with open(args.out, "w", newline="", encoding="utf-8") as fo:
        w = csv.writer(fo)
        w.writerow(["theme_label_predicted", "theme_clustered", "score"])
        w.writerows(rows)
    print("✅ CSV saved →", args.out)

    # 5. Optional 2-D plot (embeddings only for visualization)
    if args.plot:
        embedder = SentenceTransformer(args.embed_model)
        embs  = embedder.encode(labels, normalize_embeddings=True, show_progress_bar=True)
        coords = PCA(n_components=2, random_state=42).fit_transform(embs)

        clusters = sorted({cl for _, cl, _ in rows})
        color_map = {cl: f"C{i%10}" for i, cl in enumerate(clusters)}
        plt.figure(figsize=(8, 6))
        for (x, y), (_, cl, _) in zip(coords, rows):
            plt.scatter(x, y, color=color_map[cl], s=35)
        plt.title("Prompt-based clusters (PCA-2D)")
        plt.xlabel("PCA-1"); plt.ylabel("PCA-2")
        handles = [plt.Line2D([0],[0], marker='o', color='w',
                              markerfacecolor=color_map[cl], markersize=8, label=cl)
                   for cl in clusters]
        plt.legend(title="Cluster", handles=handles, bbox_to_anchor=(1.05,1),
                   loc='upper left')
        plt.tight_layout(); plt.savefig(args.plot, dpi=160)
        print("🖼  Plot saved →", args.plot)


if __name__ == "__main__":
    main()

