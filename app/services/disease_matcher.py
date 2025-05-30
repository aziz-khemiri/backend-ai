# ✅ disease_matcher.py (Fixed + Async-Compatible Ready)

"""
Loads the symptom-disease dataset and provides matching/explanation logic.
"""

import re
import pandas as pd
import json
from difflib import get_close_matches
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

# Paths
DF_PATH = "app/datasets/symptom-disease-train-dataset.csv"
MAPPING_PATH = "app/datasets/mapping.json"

# Load data
df = pd.read_csv(DF_PATH)
with open(MAPPING_PATH, "r") as f:
    mapping = json.load(f)

# Original keys for fuzzy matches
original_keys = list(mapping.keys())
original_keys_lower = [k.lower() for k in original_keys]


# Normalize disease names
def normalize_disease_key(name: str) -> str:
    key = name.lower()
    key = re.sub(r"[^a-z0-9 ]+", "", key)
    key = re.sub(r"\s+", " ", key).strip()
    return key.replace(" ", "_")


# Build lookup maps
index_to_disease = {v: k for k, v in mapping.items()}
disease_to_index = {normalize_disease_key(k): v for k, v in mapping.items()}

# Prepare symptom texts
df["text"] = df["text"].fillna("").str.strip().str.lower()

# Fit TF-IDF
vectorizer = TfidfVectorizer()
tfidf_matrix = vectorizer.fit_transform(df["text"])


def get_probable_diseases(user_input: str, top_k: int = 3) -> list[dict]:
    """Top-k probable diseases for given symptoms."""
    query = user_input.strip().lower()
    vec = vectorizer.transform([query])
    sims = cosine_similarity(vec, tfidf_matrix).flatten()
    idxs = sims.argsort()[-top_k:][::-1]
    scores = sims[idxs]

    results = []
    for i, score in zip(idxs, scores):
        label = str(df.iloc[i]["label"])
        disease = index_to_disease.get(int(label), "Unknown")
        prob = round(float(score) * 100, 2)
        snippet = df.iloc[i]["text"][:80] + "..."
        results.append(
            {
                "disease": disease,
                "probability": f"{prob}%",
                "reason": f"Matched against: '{snippet}'",
            }
        )
    return results


def get_disease_explanation(disease_key: str, raw_input: str = None) -> str | None:
    """Return a patient example for a disease, with fuzzy fallbacks."""
    key = normalize_disease_key(disease_key)
    label = disease_to_index.get(key)
    used_key = key

    if label is None:
        c1 = get_close_matches(key, disease_to_index.keys(), n=1, cutoff=0.6)
        if c1:
            used_key = c1[0]
            label = disease_to_index[used_key]
        elif raw_input:
            c2 = get_close_matches(
                raw_input.lower(), original_keys_lower, n=1, cutoff=0.6
            )
            if c2:
                orig = original_keys[original_keys_lower.index(c2[0])]
                used_key = normalize_disease_key(orig)
                label = mapping.get(orig)

    if label is None:
        return None

    entry = df[df["label"].astype(str) == str(label)]
    if entry.empty:
        return None

    example = entry.iloc[0]["text"].strip()
    display = used_key.replace("_", " ").title()
    return (
        f"Here’s a real-world symptom description related to {display}:\n\n> {example}"
    )
