"""Build the training set for the Strong Again energy check-in classifier.

The data is synthetic: short check-in messages written for this project
(no real user data). Training sentences are assembled from hand-written
phrase banks; the test set (data/checkins_test.csv) is written separately
by hand with different phrasing, so it measures generalization instead of
memorized templates.

Labels: low (suggest 7-min session), steady (15-min), strong (30-min).
"""
import csv
import random
from pathlib import Path

random.seed(2300)

OPENERS = [
    "", "", "", "Today ", "Honestly ", "This morning ", "Right now ", "Hmm, ",
    "Checking in: ", "Not gonna lie, ", "Okay so ", "Woke up and ",
]

STATES = {
    "low": [
        "I'm exhausted", "I feel wiped out", "I barely slept last night",
        "I'm so tired I can hardly stand", "my body feels heavy",
        "I'm drained", "the fatigue is bad", "I feel foggy and slow",
        "everything aches", "I'm nauseous", "I have zero energy",
        "I'm running on empty", "I can barely get off the couch",
        "my arm is sore and stiff", "I feel weak", "I'm worn out",
        "chemo brain is real today", "I feel sluggish", "I'm really fatigued",
        "I don't have much in me", "I'm not feeling great", "I feel rough",
        "I'm low on energy", "I just want to lie down", "I'm beat",
        "I feel run down", "I'm completely spent", "my legs feel like lead",
    ],
    "steady": [
        "I feel okay", "I'm doing alright", "I feel fine", "I'm so-so",
        "I'm a little tired but okay", "I feel pretty normal", "I'm managing",
        "energy is average", "I feel decent", "I'm somewhere in the middle",
        "I'm not bad", "I'm a bit stiff but alright", "I feel steady",
        "I could do something light", "I'm moderately rested",
        "I slept okay", "I feel calm", "I'm fairly good", "it's an ordinary day",
        "I'm up for a moderate session", "I'm neither tired nor energized",
        "I'm hanging in there", "I'm a little sore but fine", "I feel balanced",
        "I'm ok, nothing special", "I have some energy", "I feel reasonably good",
    ],
    "strong": [
        "I feel amazing", "I'm full of energy", "I slept great",
        "I feel strong", "I'm ready to go", "I feel like myself again",
        "I'm energized", "I want a challenge", "I feel fantastic",
        "I'm really motivated", "I feel powerful", "I'm pumped",
        "I could go for a long session", "I'm not tired at all",
        "I feel rested and ready", "I'm buzzing with energy", "I feel great",
        "I'm raring to go", "I feel unstoppable", "I'm feeling really good",
        "I have tons of energy", "I'm ready to push a little",
        "I feel lively", "I feel refreshed", "best I've felt in weeks",
        "I'm excited to move", "I'm feeling sharp and strong",
    ],
}

CONTEXT = {
    "any": [
        "", "", "", "", ".", "!", " today.", " this morning.",
        " after my appointment.", " and the weather is nice.",
        " but it's raining.", " after coffee.",
    ],
    "low": [
        ", I think I need rest.", " after treatment yesterday.",
        ", maybe something short.", " and my head hurts.",
        ", the kids kept me up.", " since infusion.", ", just a gentle one please.",
    ],
    "steady": [
        ", nothing too intense.", ", maybe a normal session.",
        " and I want to keep my routine.", ", a medium workout sounds right.",
        " after a quiet day.", ", let's keep it steady.",
    ],
    "strong": [
        ", let's do the full workout!", " and I want to sweat.",
        ", give me the long one.", " after a great night's sleep.",
        ", I even walked the dog already.", ", bring it on!",
    ],
}

PER_CLASS = 220


def build():
    rows, seen = [], set()
    for label, states in STATES.items():
        n = 0
        while n < PER_CLASS:
            ctx = random.choice(CONTEXT["any"] + CONTEXT[label])
            text = f"{random.choice(OPENERS)}{random.choice(states)}{ctx}".strip()
            text = text[0].upper() + text[1:]
            if text in seen:
                continue
            seen.add(text)
            rows.append((text, label))
            n += 1
    random.shuffle(rows)
    return rows


if __name__ == "__main__":
    out = Path(__file__).parent / "data" / "checkins_train.csv"
    rows = build()
    with out.open("w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["text", "label"])
        w.writerows(rows)
    print(f"wrote {len(rows)} rows to {out}")
