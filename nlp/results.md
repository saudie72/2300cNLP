# Energy check-in classifier: results

Training rows: 660 (synthetic, built by `make_dataset.py`)  
Test rows: 60 (hand-written separately, different phrasing)

| Model | 5-fold CV accuracy (train) | Test accuracy |
|---|---|---|
| Naive Bayes (baseline) | 0.991 | 0.883 |
| Logistic regression | 0.995 | 0.900 |

## Logistic regression (exported model) on the test set

```
              precision    recall  f1-score   support

         low      1.000     0.850     0.919        20
      steady      0.947     0.900     0.923        20
      strong      0.792     0.950     0.864        20

    accuracy                          0.900        60
   macro avg      0.913     0.900     0.902        60
weighted avg      0.913     0.900     0.902        60

```

Confusion matrix (rows = true, columns = predicted):

| | low | steady | strong |
|---|---|---|---|
| **low** | 17 | 0 | 3 |
| **steady** | 0 | 18 | 2 |
| **strong** | 0 | 1 | 19 |

### Cautious rule ("when unsure, round down")

If the top probability is below 0.55, the page recommends the gentler of the top two classes.

| Decision | Test accuracy | Too-intense suggestions |
|---|---|---|
| Plain argmax | 0.900 | 5 |
| Cautious rule (used on the page) | 0.917 | 4 |

Misclassified test messages:

- "Feeling pretty wrecked, hardly slept at all": true **low**, predicted **strong**
- "Tired to the bone today": true **low**, predicted **strong**
- "Not great, really tired": true **low**, predicted **strong**
- "Middle of the road kind of day": true **steady**, predicted **strong**
- "Kind of in between today": true **steady**, predicted **strong**
- "I'm so energetic today": true **strong**, predicted **steady**

## Most indicative words per class

- **low**: `rough`, `feel rough`, `out`, `feeling great`, `not feeling`, `sluggish`, `feel sluggish`, `down`, `exhausted`, `im exhausted`, `feel weak`, `weak`
- **steady**: `fine`, `so so`, `feel decent`, `decent`, `calm`, `feel calm`, `slept okay`, `steady`, `feel fine`, `not bad`, `balanced`, `feel balanced`
- **strong**: `im energized`, `great`, `strong`, `feel great`, `im pumped`, `pumped`, `go`, `feel fantastic`, `fantastic`, `im feeling`, `lively`, `feel lively`
