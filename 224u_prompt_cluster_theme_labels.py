#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Prompt-only clustering of theme_label_predicted.

1. Collect unique predicted labels.
2. In batches, ask Gemma-3-27B to place each label in 1 of 10 clusters
   and give every cluster a concise name.
3. Ask Gemma for a 0-1 confidence score for each (label, cluster) pair.
4. Optionally move low-confidence labels to "Unclustered".
5. Write CSV  [label, cluster, score].
6. Optionally draw a 2-D PCA scatter plot (embeddings can run on CPU).

Example
-------
python prompt_cluster_theme_labels.py preds.jsonl           \\
       --out clusters.csv --plot clusters.png               \\
       --batch-size 40 --threshold 0.6 --plot-cpu
"""

import json, csv, re, argparse, os
from collections import defaultdict

import numpy as np
from sentence_transformers import SentenceTransformer
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt
from huggingface_hub import hf_hub_download, login
from llama_cpp import Llama
from tqdm import tqdm

# ─────────────────────────────────────────────────────────────
# Load Gemma-3-27B gguf
# ─────────────────────────────────────────────────────────────
GGUF = os.getenv("GEMMA_PATH")          # set this to skip download
if GGUF is None:
    repo_id, fn = "google/gemma-3-27b-it-qat-q4_0-gguf", "gemma-3-27b-it-q4_0.gguf"
    if (tok := os.getenv("HF_TOKEN")):
        login(token=tok, add_to_git_credential=True)
    GGUF = hf_hub_download(repo_id=repo_id, filename=fn, resume_download=True)

gemma = Llama(model_path=GGUF, n_ctx=40960, n_gpu_layers=-1, verbose=False)

# ─────────────────────────────────────────────────────────────
# Prompts
# ─────────────────────────────────────────────────────────────
GROUP_PROMPT = (
    "You are an expert categorizer.\n"
    "Group the following theme labels into exactly 10 clusters.\n"
    "Give each cluster a concise name (≤4 words, Title Case, no punctuation).\n\n"
    "Return JSON like:\n"
    "[{\"label\": \"<label>\", \"cluster\": \"<cluster name>\"}, …]\n"
)

CONF_PROMPT = (
    "Rate how well a theme label fits a cluster name.\n"
    "Respond with ONLY a number between 0 and 1.\n\n"
    "Label: {lbl}\nCluster: {cl}\n<score>"
)

rx_json  = re.compile(r"\[[\s\S]+?]")          # first JSON array
rx_float = re.compile(r"0\.\d+|1\.0+")
MAX_LABEL_LEN = 70                             # chars sent to LLM


# ─────────────────────────────────────────────────────────────
# Helpers
# ─────────────────────────────────────────────────────────────
def gemma_chat(prompt: str, temperature=0.2, max_tokens=2048) -> str:
    return gemma.create_chat_completion(
        messages=[{"role": "user", "content": prompt}],
        temperature=temperature,
        max_tokens=max_tokens
    )["choices"][0]["message"]["content"].strip()


def cluster_with_prompt(labels, batch_size: int):
    """Return dict full_label -> cluster_name using batched LLM calls."""
    # Trim long labels for the prompt
    full2trim = {l: (l if len(l) <= MAX_LABEL_LEN else l[:MAX_LABEL_LEN] + "…")
                 for l in labels}
    trim2full = {v: k for k, v in full2trim.items()}

    mapping = {}
    batches = range(0, len(labels), batch_size)
    for start in tqdm(batches, mininterval=1, desc="Clustering batches"):
        chunk_trim = [full2trim[l] for l in labels[start:start + batch_size]]
        prompt = GROUP_PROMPT + "\n" + "\n".join(f"- {t}" for t in chunk_trim)
        resp   = gemma_chat(prompt)

        m = rx_json.search(resp)
        if not m:
            raise ValueError("Gemma did not return parseable JSON.")

        for item in json.loads(m.group(0)):
            full = trim2full.get(item["label"])
            if full:
                mapping[full] = item["cluster"]
    return mapping


def confidence(lbl: str, cluster: str) -> float:
    """Return confidence 0-1 via LLM."""
    resp = gemma_chat(CONF_PROMPT.format(lbl=lbl, cl=cluster),
                      temperature=0.0, max_tokens=8)
    m = rx_float.search(resp)
    return float(m.group(0)) if m else 0.5


# ─────────────────────────────────────────────────────────────
# CLI
# ─────────────────────────────────────────────────────────────
def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("pred_jsonl")
    p.add_argument("--out",  default="clustered_prompt.csv")
    p.add_argument("--plot", default=None, help="PNG output (optional)")
    p.add_argument("--batch-size", type=int, default=40)
    p.add_argument("--threshold",  type=float, default=0.0,
                   help="score < threshold → Unclustered (0 = disable)")
    p.add_argument("--plot-cpu", action="store_true",
                   help="embed plot points on CPU to avoid GPU OOM")
    p.add_argument("--embed-model", default="sentence-transformers/all-mpnet-base-v2")
    return p.parse_args()


# ─────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────
def main():
    args = parse_args()

    # 1. load unique labels
    labels = {json.loads(line)["turns"][-1].get("theme_label_predicted", "").strip()
              for line in open(args.pred_jsonl, encoding="utf-8")}
    labels.discard("")
    labels = sorted(labels)
    if not labels:
        raise ValueError("No theme_label_predicted found.")
    print("Unique labels:", len(labels))

    # 2. clustering via prompt
    label2cluster = cluster_with_prompt(labels, batch_size=args.batch_size)

    # 3. confidence scores
    rows = []
    missing, dropped = 0, 0
    print("Scoring confidence …")
    for lbl in tqdm(labels, mininterval=1):
        cluster = label2cluster.get(lbl)
        if cluster is None:
            rows.append((lbl, "Unclustered", 0.00))
            missing += 1
            continue

        score = round(confidence(lbl, cluster), 3)
        if args.threshold and score < args.threshold:
            rows.append((lbl, "Unclustered", score))
            dropped += 1
        else:
            rows.append((lbl, cluster, score))

    if missing:
        print(f"{missing} labels were absent from Gemma output.")
    if dropped:
        print(f"{dropped} labels scored below threshold and were moved to 'Unclustered'.")

    # 4. write CSV
    with open(args.out, "w", newline="", encoding="utf-8") as fo:
        csv.writer(fo).writerows(
            [("theme_label_predicted", "theme_clustered", "score"), *rows]
        )
    print("CSV saved →", args.out)

    # 5. optional plot
    if args.plot:
        device = "cpu" if args.plot_cpu else "cuda"
        try:
            embedder = SentenceTransformer(args.embed_model, device=device)
        except RuntimeError:
            print("Falling back to CPU for embeddings.")
            embedder = SentenceTransformer(args.embed_model, device="cpu")

        embs = embedder.encode(labels, normalize_embeddings=True, show_progress_bar=True)
        coords = PCA(n_components=2, random_state=42).fit_transform(embs)

        cluster_set = sorted({cl for _, cl, _ in rows})
        cmap = {cl: f"C{i % 10}" for i, cl in enumerate(cluster_set)}
        plt.figure(figsize=(8, 6))
        for (x, y), (_, cl, _) in zip(coords, rows):
            plt.scatter(x, y, color=cmap[cl], s=35)
        plt.title("Prompt-based clusters (PCA-2D)")
        plt.xlabel("PCA-1"); plt.ylabel("PCA-2")
        handles = [plt.Line2D([0], [0], marker='o', color='w',
                              markerfacecolor=cmap[cl], markersize=8, label=cl)
                   for cl in cluster_set]
        plt.legend(title="Cluster", handles=handles,
                   bbox_to_anchor=(1.05, 1), loc='upper left')
        plt.tight_layout(); plt.savefig(args.plot, dpi=160)
        print("Plot saved →", args.plot)


if __name__ == "__main__":
    main()

