"""Text CNN for LLM authorship; run from repo root: python src/cnn.py."""
from collections import Counter
from copy import deepcopy
from pathlib import Path
import re

import pandas as pd
import torch
from sklearn.metrics import accuracy_score, classification_report, f1_score
from sklearn.model_selection import train_test_split
from torch import nn
from torch.utils.data import DataLoader, TensorDataset

ROOT = Path(__file__).resolve().parents[1]
df = pd.read_csv(ROOT / "data" / "llm_responses.csv")
train_ids, test_ids = train_test_split(
    df["prompt_id"].unique(), test_size=0.2, random_state=42
)
fit_ids, val_ids = train_test_split(train_ids, test_size=0.2, random_state=42)
fit = df[df["prompt_id"].isin(fit_ids)].copy()
val = df[df["prompt_id"].isin(val_ids)].copy()
test = df[df["prompt_id"].isin(test_ids)].copy()
labels = sorted(df["model_name"].unique())
label_to_id = {label: i for i, label in enumerate(labels)}
MAX_LEN = 256


def tokenize(text):
    return re.findall(r"\w+|[^\w\s]", text.lower())


def texts(frame, setting):
    if setting == "input_only":
        return frame["prompt"].tolist()
    if setting == "output_only":
        return frame["response"].tolist()
    return ("Prompt: " + frame["prompt"] + "\nResponse: " + frame["response"]).tolist()


def encode(text_list, vocab):
    rows = []
    for text in text_list:
        ids = [vocab.get(word, 1) for word in tokenize(text)[:MAX_LEN]]
        rows.append((ids + [0] * MAX_LEN)[:MAX_LEN])
    return torch.tensor(rows, dtype=torch.long)


def loader(frame, setting, vocab, shuffle=False):
    x = encode(texts(frame, setting), vocab)
    y = torch.tensor([label_to_id[name] for name in frame["model_name"]])
    return DataLoader(TensorDataset(x, y), batch_size=16, shuffle=shuffle)


class TextCNN(nn.Module):
    def __init__(self, vocab_size, classes):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, 64, padding_idx=0)
        self.convs = nn.ModuleList([nn.Conv1d(64, 64, k) for k in (3, 4, 5)])
        self.dropout = nn.Dropout(0.4)
        self.classifier = nn.Linear(64 * 3, classes)

    def forward(self, x):
        x = self.embedding(x).transpose(1, 2)
        features = [torch.relu(conv(x)).max(dim=2).values for conv in self.convs]
        return self.classifier(self.dropout(torch.cat(features, dim=1)))


def evaluate(model, data):
    model.eval()
    actual, predicted = [], []
    with torch.no_grad():
        for x, y in data:
            guesses = model(x).argmax(dim=1)
            actual.extend(y.tolist())
            predicted.extend(guesses.tolist())
    return actual, predicted


results = []
for setting in ("output_only", "input_only", "input_output"):
    torch.manual_seed(42)
    counts = Counter(word for text in texts(fit, setting) for word in tokenize(text))
    vocab = {word: i + 2 for i, (word, _) in enumerate(counts.most_common(8000))}
    train_data = loader(fit, setting, vocab, shuffle=True)
    val_data = loader(val, setting, vocab)
    test_data = loader(test, setting, vocab)

    model = TextCNN(len(vocab) + 2, len(labels))
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    loss_fn = nn.CrossEntropyLoss()
    best_score, best_weights, stale = -1, None, 0
    for epoch in range(1, 16):
        model.train()
        for x, y in train_data:
            optimizer.zero_grad()
            loss = loss_fn(model(x), y)
            loss.backward()
            optimizer.step()
        actual, predicted = evaluate(model, val_data)
        score = f1_score(actual, predicted, average="macro", zero_division=0)
        print(f"{setting} epoch {epoch}: validation macro-F1={score:.3f}")
        if score > best_score:
            best_score, best_weights, stale = score, deepcopy(model.state_dict()), 0
        else:
            stale += 1
            if stale >= 3:
                break

    model.load_state_dict(best_weights)
    actual, predicted = evaluate(model, test_data)
    accuracy = accuracy_score(actual, predicted)
    macro_f1 = f1_score(actual, predicted, average="macro", zero_division=0)
    print(f"\n{setting}: test accuracy={accuracy:.3f}, macro-F1={macro_f1:.3f}")
    print(classification_report(actual, predicted, target_names=labels, zero_division=0))
    results.append({"input_type": setting, "classifier": "Text CNN",
                    "accuracy": accuracy, "macro_f1": macro_f1})

output = ROOT / "results" / "cnn_metrics.csv"
output.parent.mkdir(exist_ok=True)
pd.DataFrame(results).to_csv(output, index=False)
print(f"Saved results to {output}")