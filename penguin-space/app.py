"""Predict a Palmer penguin's body mass (g) with the Keras network from the notebook (Steps 26-34)."""
import os

os.environ.setdefault("KERAS_BACKEND", "tensorflow")

import gradio as gr
import joblib
import keras
import numpy as np
import pandas as pd

model = keras.models.load_model("model.keras")
saved = joblib.load("prep.joblib")

prep = saved["prep"]
problem = saved["problem"]
labels = saved["labels"]
y_mean, y_std = saved["y_mean"], saved["y_std"]
columns = saved["columns"]
typical = saved["typical"]

# Which columns are numbers and which are categories, taken from the fitted ColumnTransformer
num_cols, cat_choices = [], {}
for name, pipe, cols in prep.transformers_:
    if name == "numbers":
        num_cols = list(cols)
    elif name == "categories":
        encoder = pipe.steps[-1][1]
        cat_choices = {c: [str(v) for v in cats] for c, cats in zip(cols, encoder.categories_)}


def predict(*values):
    row = pd.DataFrame([dict(zip(columns, values))])
    for c in num_cols:
        row[c] = pd.to_numeric(row[c], errors="coerce")
    out = model.predict(prep.transform(row).astype("float32"), verbose=0)
    if problem == "regression":
        return f"{float(out.ravel()[0] * y_std + y_mean):,.0f} g"
    if problem == "binary":
        idx = (out.ravel() > 0.5).astype(int)
    else:
        idx = np.argmax(out, axis=1)
    return str(labels.inverse_transform(idx)[0])


fields = [
    gr.Number(value=round(float(typical[c]), 2), label=c) if c in num_cols
    else gr.Dropdown(choices=cat_choices.get(c, [str(typical[c])]), value=str(typical[c]), label=c)
    for c in columns
]

demo = gr.Interface(
    fn=predict,
    inputs=fields,
    outputs=gr.Textbox(label="Predicted body_mass_g"),
    title="Predict a penguin's body mass",
    description="Palmer Penguins · Keras network (64 → dropout 20% → 32 → 1). "
                "Every box starts at a typical penguin; change the values and click Submit.",
    submit_btn="Predict",
    flagging_mode="never",
)

if __name__ == "__main__":
    demo.launch()
