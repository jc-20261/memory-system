#!/usr/bin/env python3
r"""
similarity_check.py – Quick cosine similarity between two input terms.

Uses the same embedding model as the search system when possible:
  1. Model2Vec (minishlab/potion-base-8M)
  2. SentenceTransformer fallback (all-MiniLM-L6-v2)
"""

import numpy as np

def get_model():
    try:
        from model2vec import StaticModel
        print("Using Model2Vec")
        return StaticModel.from_pretrained("minishlab/potion-base-8M")
    except Exception:
        pass

    try:
        from sentence_transformers import SentenceTransformer
        print("Using SentenceTransformer fallback")
        return SentenceTransformer("all-MiniLM-L6-v2")
    except Exception as e:
        raise RuntimeError("No embedding model available") from e

def embed_text(model, text):
    if model.__class__.__name__ == "StaticModel":
        return np.asarray(model.encode([text])[0], dtype="float32")
    else:
        return np.asarray(model.encode(text, normalize_embeddings=True), dtype="float32")

def normalize(vec):
    norm = np.linalg.norm(vec)
    if norm == 0:
        return vec
    return vec / norm

def cosine(a, b):
    return float(np.dot(a, b))

def main():
    model = get_model()

    while True:
        t1 = input("Term 1 (or 'quit'): ").strip()
        if t1.lower() in {"quit", "exit", "q"}:
            break

        t2 = input("Term 2: ").strip()
        if not t2:
            continue

        v1 = normalize(embed_text(model, t1))
        v2 = normalize(embed_text(model, t2))

        score = cosine(v1, v2)
        print(f"Cosine similarity: {score:.6f}\n")

if __name__ == "__main__":
    main()