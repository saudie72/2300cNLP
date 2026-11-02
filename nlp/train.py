"""Train the Strong Again energy check-in classifier and export it for the browser.

Pipeline (text classification, Ch02-Ch04):
  tokenize -> TF-IDF (unigrams + bigrams) -> logistic regression (3 classes)
A multinomial Naive Bayes model on the same features is trained as a baseline.

Outputs:
  ../assets/energy-model.json   weights the landing page uses to classify in the browser
  results.md                    evaluation on the hand-written test set
  parity_fixture.json           Python probabilities, used by check_parity.js to
                                confirm the JavaScript port gives the same answers
"""
import csv
import json
import warnings
import re
from pathlib import Path

import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.model_selection import cross_val_score
from sklearn.naive_bayes import MultinomialNB
from sklearn.pipeline import make_pipeline

HERE = Path(__file__).parent
warnings.filterwarnings("ignore", message="Unknown solver options")
UNSURE = 0.55  # below this top probability, recommend the gentler of the top two classes
LABELS = ["low", "steady", "strong"]


def tokenize(text):
    """Must match tokenize() in ../index.html exactly.

    Apostrophes are removed rather than split on, so negations survive as
    single tokens ("don't" -> "dont", "isn't" -> "isnt").
    """
    text = text.lower().replace("’", "'").replace("'", "")
    return [t for t in re.split(r"[^a-z0-9]+", text) if len(t) >= 2]


def load(name):
    with (HERE / "data" / name).open() as f:
        rows = list(csv.DictReader(f))
    return [r["text"] for r in rows], [r["label"] for r in rows]


def cautious(proba):
    """Asymmetric-cost decision rule, mirrored in index.html.

    Suggesting too much exercise is worse than suggesting too little, so when the
    model is unsure we pick the lower-energy class of its top two guesses.
    """
    order = np.argsort(proba)[::-1]
    if proba[order[0]] < UNSURE:
        return LABELS[min(order[0], order[1])]
    return LABELS[order[0]]


def vectorizer():
    return TfidfVectorizer(
        tokenizer=tokenize, token_pattern=None, lowercase=False,
        ngram_range=(1, 2), sublinear_tf=True, min_df=1,
    )


def main():
    X_train, y_train = load("checkins_train.csv")
    X_test, y_test = load("checkins_test.csv")

    models = {
        "Naive Bayes (baseline)": make_pipeline(vectorizer(), MultinomialNB(alpha=0.5)),
        "Logistic regression": make_pipeline(vectorizer(), LogisticRegression(C=4.0, max_iter=2000)),
    }

    lines = ["# Energy check-in classifier: results", "",
             f"Training rows: {len(X_train)} (synthetic, built by `make_dataset.py`)  ",
             f"Test rows: {len(X_test)} (hand-written separately, different phrasing)", "",
             "| Model | 5-fold CV accuracy (train) | Test accuracy |", "|---|---|---|"]
    for name, model in models.items():
        cv = cross_val_score(model, X_train, y_train, cv=5)
        model.fit(X_train, y_train)
        acc = accuracy_score(y_test, model.predict(X_test))
        lines.append(f"| {name} | {cv.mean():.3f} | {acc:.3f} |")
        print(f"{name}: cv={cv.mean():.3f} test={acc:.3f}")

    lr = models["Logistic regression"]
    vec, clf = lr.named_steps["tfidfvectorizer"], lr.named_steps["logisticregression"]
    assert list(clf.classes_) == LABELS
    pred = lr.predict(X_test)
    cm = confusion_matrix(y_test, pred, labels=LABELS)

    lines += ["", "## Logistic regression (exported model) on the test set", "",
              "```", classification_report(y_test, pred, labels=LABELS, digits=3), "```", "",
              "Confusion matrix (rows = true, columns = predicted):", "",
              "| | " + " | ".join(LABELS) + " |", "|---|---|---|---|"]
    lines += [f"| **{LABELS[i]}** | " + " | ".join(str(v) for v in row) + " |" for i, row in enumerate(cm)]

    pred_c = [cautious(p) for p in lr.predict_proba(X_test)]
    over = lambda ps: sum(LABELS.index(p) > LABELS.index(y) for y, p in zip(y_test, ps))
    lines += ["", "### Cautious rule (\"when unsure, round down\")", "",
              f"If the top probability is below {UNSURE}, the page recommends the gentler of the top two classes.", "",
              "| Decision | Test accuracy | Too-intense suggestions |", "|---|---|---|",
              f"| Plain argmax | {accuracy_score(y_test, pred):.3f} | {over(pred)} |",
              f"| Cautious rule (used on the page) | {accuracy_score(y_test, pred_c):.3f} | {over(pred_c)} |"]

    misses = [(t, y, p) for t, y, p in zip(X_test, y_test, pred) if y != p]
    lines += ["", "Misclassified test messages:", ""]
    lines += [f"- \"{t}\": true **{y}**, predicted **{p}**" for t, y, p in misses] or ["- none"]

    terms = vec.get_feature_names_out()
    lines += ["", "## Most indicative words per class", ""]
    for i, label in enumerate(LABELS):
        top = np.argsort(clf.coef_[i])[::-1][:12]
        lines.append(f"- **{label}**: " + ", ".join(f"`{terms[j]}`" for j in top))
    (HERE / "results.md").write_text("\n".join(lines) + "\n")

    model_json = {
        "labels": LABELS,
        "ngram_range": [1, 2],
        "sublinear_tf": True,
        "vocab": {t: int(i) for t, i in vec.vocabulary_.items()},
        "idf": [round(float(v), 5) for v in vec.idf_],
        "coef": [[round(float(v), 5) for v in row] for row in clf.coef_],
        "unsure_threshold": UNSURE,
        "intercept": [round(float(v), 5) for v in clf.intercept_],
    }
    out = HERE.parent / "assets" / "energy-model.json"
    out.write_text(json.dumps(model_json, separators=(",", ":")))
    print(f"exported {len(vec.vocabulary_)} features to {out} ({out.stat().st_size // 1024} KB)")

    samples = X_test + ["", "I'm not tired at all", "Chest pain and dizzy"]
    proba = lr.predict_proba(samples)
    (HERE / "parity_fixture.json").write_text(json.dumps(
        [{"text": t, "proba": [round(float(p), 6) for p in row]} for t, row in zip(samples, proba)], indent=1))


if __name__ == "__main__":
    main()
