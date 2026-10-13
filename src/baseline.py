"""TF-IDF baseline for three LLM authorship input settings.
Run from the repository root: python src/baseline.py
"""
from pathlib import Path

import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report, f1_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import make_pipeline

ROOT = Path(__file__).resolve().parents[1]
df = pd.read_csv(ROOT / "data" / "llm_responses.csv")

# One prompt ID must identify exactly one prompt; every model should answer each prompt.
if df.groupby("prompt_id")["prompt"].nunique().max() != 1:
    raise ValueError("A prompt_id is assigned to more than one prompt.")
if df.duplicated(["prompt_id", "model_name"]).any():
    raise ValueError("Duplicate prompt/model pairs found.")

# Split prompt IDs, not rows, to keep paired responses together.
train_ids, test_ids = train_test_split(
    df["prompt_id"].unique(), test_size=0.2, random_state=42
)
train = df[df["prompt_id"].isin(train_ids)]
test = df[df["prompt_id"].isin(test_ids)]

settings = {
    "output_only": (train["response"], test["response"]),
    "input_only": (train["prompt"], test["prompt"]),
    "input_output": (
        "Prompt: " + train["prompt"] + "\nResponse: " + train["response"],
        "Prompt: " + test["prompt"] + "\nResponse: " + test["response"],
    ),
}

results = []
for name, (x_train, x_test) in settings.items():
    model = make_pipeline(
        TfidfVectorizer(max_features=10000, ngram_range=(1, 2)),
        LogisticRegression(max_iter=1000),
    )
    model.fit(x_train, train["model_name"])
    predictions = model.predict(x_test)
    accuracy = accuracy_score(test["model_name"], predictions)
    macro_f1 = f1_score(test["model_name"], predictions, average="macro")
    results.append({"input_type": name, "classifier": "TF-IDF + logistic regression",
                    "accuracy": accuracy, "macro_f1": macro_f1})
    print(f"\n{name}: accuracy={accuracy:.3f}, macro-F1={macro_f1:.3f}")
    print(classification_report(test["model_name"], predictions, zero_division=0))

output = ROOT / "results" / "baseline_metrics.csv"
output.parent.mkdir(exist_ok=True)
pd.DataFrame(results).to_csv(output, index=False)
print(f"Saved results to {output}")