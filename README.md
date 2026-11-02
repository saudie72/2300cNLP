# Strong Again — landing page

Static landing page (plain HTML, no build step), deployed on Netlify.

- `index.html` — the page (dark-mode toggle is a small inline script)
- `assets/` — preview video and poster image
- `favicon.svg` — logo mark
- `nlp/` — the NLP feature (see below)
- `assets/energy-model.json` — trained model weights the page loads
- `assets/word-map.json` — 3D word-map data built by `nlp/word_map.py`
- `netlify.toml` — publishes the repo root; long-cache headers for `/assets/*`

Run locally: `python3 -m http.server` and open http://localhost:8000.

Note: buttons that point to the app ("Start free trial", "Sign in", feature cards)
currently scroll to the closing call-to-action (`#get-started`) until the app URL exists.

## NLP feature: energy check-in classifier

**Course topic:** text classification with TF-IDF + logistic regression (Ch02–Ch04),
with multinomial Naive Bayes as a baseline.

In the "Try a check-in" section the model works live: it types demo messages on its own, colors
each word by the energy level it pushes toward, moves the three probability bars as each word
is added, and updates the suggestion. A visitor can click the box and type their own. The page classifies the
message as **Low / Steady / Strong** energy and suggests the matching 7 / 15 / 30-minute
session. Classification runs in the browser, so nothing typed is sent anywhere.

| Step | What happens |
|---|---|
| Tokenize | lowercase, drop apostrophes (`don't` → `dont`, so negations stay one token), split on non-alphanumerics |
| Features | TF-IDF on unigrams + bigrams, sublinear TF, L2-normalized |
| Model | 3-class logistic regression (scikit-learn), exported to JSON and re-implemented in ~40 lines of JS |
| Decision | "when unsure, round down": if top probability < 0.55, pick the gentler of the top two classes |
| Safety | messages mentioning possible warning signs (chest pain, fever, dizziness, swelling…) get a "pause and contact your care team" message instead of a session |

**Results** (full report in [`nlp/results.md`](nlp/results.md)): about 90% accuracy on 60 hand-written
test messages, vs. 88% for Naive Bayes. The data is synthetic (written for this project, no real
user data), so the test set was written separately with different phrasing to avoid scoring memorized templates.

Reproduce:

```bash
pip install -r nlp/requirements.txt
python3 nlp/make_dataset.py   # rebuild nlp/data/checkins_train.csv
python3 nlp/train.py          # train, evaluate, export assets/energy-model.json
node nlp/check_parity.js      # confirm the JS port matches scikit-learn
```

## NLP feature: 3D word map (word embeddings, Ch05)

The "Word map" section shows 127 words from six themes (tiredness, energy, feelings, body,
movement, treatment) as a rotating 3D point cloud. The positions come from pretrained **GloVe**
vectors (100 dimensions, Wikipedia + Gigaword) reduced to 3D with **PCA**. Selecting a word draws
lines to its six nearest neighbours by cosine similarity in the full 100-D space. The map tours
through words on its own until the visitor drags, clicks or searches.

The three PCA components keep about 29% of the variance, so words that look close in 3D are not
always nearest neighbours; the neighbour list and lines use the full vectors.

```bash
curl -L -o glove100.gz https://github.com/RaRe-Technologies/gensim-data/releases/download/glove-wiki-gigaword-100/glove-wiki-gigaword-100.gz
python3 nlp/word_map.py glove100.gz
```
