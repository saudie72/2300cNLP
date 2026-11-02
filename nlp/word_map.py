"""Build the 3D word map shown on the landing page (word embeddings, Ch05).

Uses pretrained GloVe vectors (100 dimensions, Wikipedia + Gigaword), picks words
from six themes that matter to Strong Again, finds each word's nearest neighbours
by cosine similarity in the full 100-D space, then projects the words to 3D with
PCA for display.

Download the vectors (about 130 MB) once:
  curl -L -o glove100.gz https://github.com/RaRe-Technologies/gensim-data/releases/download/glove-wiki-gigaword-100/glove-wiki-gigaword-100.gz
Then:
  python3 nlp/word_map.py glove100.gz
Output: assets/word-map.json
"""
import gzip
import json
import sys
from pathlib import Path

import numpy as np
from sklearn.decomposition import PCA

GROUPS = {
    "Tiredness": ("#E0A43A", "tired exhausted fatigue weary sleepy drained sluggish lethargic weak drowsy groggy "
                  "fatigued exhaustion sleepless insomnia nap rest sleep listless worn"),
    "Energy": ("#4FC79A", "energetic energized strong vigorous lively active fit refreshed motivated eager ready "
               "powerful healthy alert awake vital dynamic upbeat invigorated enthusiastic robust"),
    "Feelings": ("#F27A9B", "happy sad calm anxious worried hopeful scared grateful frustrated nervous cheerful "
                 "lonely proud relieved stressed angry joy fear content overwhelmed confident"),
    "Body": ("#6FA8FF", "arm shoulder chest breast leg legs back neck knee hip muscles joints skin hand wrist "
             "heart lungs bones spine ankle elbow"),
    "Movement": ("#B48CF2", "walk run running jog stretch yoga squat swim swimming bike cycling dance hike lift "
                 "workout exercise exercises training marathon pilates aerobics sprint"),
    "Treatment": ("#F4F2F5", "chemo chemotherapy radiation surgery mastectomy cancer tumor oncologist doctor nurse "
                  "hospital clinic therapy recovery diagnosis treatment medication infusion biopsy scan lymph survivor"),
}
NEIGHBOURS = 6


def load(path, wanted):
    vecs = {}
    with gzip.open(path, "rt", encoding="utf8") as f:
        for i, line in enumerate(f):
            parts = line.rstrip().split(" ")
            if i == 0 and len(parts) == 2:
                continue  # word2vec-format header
            if parts[0] in wanted:
                vecs[parts[0]] = np.array(parts[1:], dtype=np.float32)
    return vecs


def main(path):
    words, group_of = [], {}
    for g, (_, ws) in GROUPS.items():
        for w in ws.split():
            if w not in group_of:
                group_of[w] = g
                words.append(w)
    vecs = load(path, set(words))
    missing = [w for w in words if w not in vecs]
    words = [w for w in words if w in vecs]
    print(f"{len(words)} words found, missing: {missing or 'none'}")

    M = np.stack([vecs[w] for w in words])
    unit = M / np.linalg.norm(M, axis=1, keepdims=True)
    sims = unit @ unit.T
    np.fill_diagonal(sims, -1)

    pca = PCA(n_components=3, random_state=0)
    P = pca.fit_transform(M)
    P = P - P.mean(axis=0)
    P = P / np.abs(P).max()
    print("explained variance (3 comps):", pca.explained_variance_ratio_.round(3), "total", pca.explained_variance_ratio_.sum().round(3))

    names = list(GROUPS)
    out = {
        "source": "GloVe 6B, 100-d (Wikipedia + Gigaword)",
        "explained_variance": round(float(pca.explained_variance_ratio_.sum()), 3),
        "groups": [{"name": g, "color": GROUPS[g][0]} for g in names],
        "words": [],
    }
    for i, w in enumerate(words):
        nn = np.argsort(sims[i])[::-1][:NEIGHBOURS]
        out["words"].append({
            "w": w, "g": names.index(group_of[w]),
            "p": [round(float(v), 4) for v in P[i]],
            "n": [int(j) for j in nn], "s": [round(float(sims[i, j]), 3) for j in nn],
        })
    dest = Path(__file__).parent.parent / "assets" / "word-map.json"
    dest.write_text(json.dumps(out, separators=(",", ":")))
    print(f"wrote {dest} ({dest.stat().st_size // 1024} KB)")
    for probe in ["tired", "energetic", "chemotherapy", "shoulder", "yoga", "anxious"]:
        i = words.index(probe)
        print(probe, "->", ", ".join(f"{words[j]} {sims[i, j]:.2f}" for j in np.argsort(sims[i])[::-1][:5]))


if __name__ == "__main__":
    main(sys.argv[1] if len(sys.argv) > 1 else "glove100.gz")
