---
title: predict-my-data
emoji: 🐧
colorFrom: blue
colorTo: gray
sdk: gradio
app_file: app.py
pinned: false
---

# Predict a penguin's body mass

Step 15 of the Palmer Penguins homework: the Keras regression network from the notebook
(Steps 26–34), served as a permanent Gradio app on Hugging Face Spaces.

- `model.keras` — the network (Keras 3.13)
- `prep.joblib` — dict with `prep`, `problem`, `labels`, `y_mean`, `y_std`, `columns`, `typical`
  (saved with scikit-learn 1.6.1, so `requirements.txt` pins that version)
- `app.py` — one input per column, prefilled with the typical penguin, plus a Predict button;
  the prediction is un-scaled back to grams (`pred * y_std + y_mean`)

Inputs: species, island, bill_length_mm, bill_depth_mm, flipper_length_mm, sex.
