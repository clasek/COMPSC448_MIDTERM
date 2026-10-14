# Identifying LLMs from Their Responses

**Course:** CMPSC 448  
**Author:** Christian Lasek

## Overview

This project tests whether a machine-learning classifier can identify which large language model wrote a response. The dataset contains answers from **ChatGPT, Claude, and Gemini** to the same set of prompts. The goal is not to prove that all AI-generated text can always be traced to one model; it is to see whether these three models show recognizable patterns in this specific dataset.

## Research Questions

1. **RQ1:** Can a classifier identify which LLM generated a response?
2. **RQ2:** Does including the user's prompt improve LLM identification?

## Dataset

The dataset is stored in `data/llm_responses.csv`.

- 50 shared prompts
- 3 LLMs: ChatGPT, Claude, and Gemini
- 150 total prompt-response records
- Five prompt categories: explanations, summarization/rewriting, creative writing, coding/computer science, and mathematics/logic

Each row contains:

| Column | Description |
|---|---|
| `prompt_id` | Identifier shared by the three responses to one prompt |
| `task_type` | Category of the prompt |
| `prompt` | The exact prompt sent to each model |
| `model_name` | The LLM that produced the response |
| `response` | The model's generated response |

The same prompts were given to every model. This makes the comparison fairer and means prompt-only classification should be near chance level, because a prompt does not identify which model answered it.

## Models and Input Settings

The project includes a simple baseline and the two required neural models:

- **TF-IDF + logistic regression baseline**
- **Text CNN**
- **Bidirectional LSTM (BiLSTM)**

Each classifier is evaluated using three input settings:

1. **Output only** — the LLM response
2. **Input only** — the user prompt
3. **Input + output** — the prompt and response combined

The data is split by `prompt_id`, not individual rows. This keeps all three answers to the same prompt together in either training or testing data. Ten prompts (30 responses) are reserved for testing.

## Results

| Input | Classifier | Accuracy | Macro-F1 |
|---|---|---:|---:|
| Response only | TF-IDF baseline | 63.3% | 0.601 |
| Response only | Text CNN | 63.3% | **0.611** |
| Response only | BiLSTM | 53.3% | 0.504 |
| Prompt only | TF-IDF baseline | 33.3% | 0.167 |
| Prompt only | Text CNN | 33.3% | 0.240 |
| Prompt only | BiLSTM | 33.3% | 0.285 |
| Prompt + response | TF-IDF baseline | 56.7% | 0.528 |
| Prompt + response | Text CNN | 60.0% | 0.593 |
| Prompt + response | BiLSTM | 63.3% | 0.507 |

Response-only classification performed above the three-class chance level of 33.3%. The CNN had the strongest response-only macro-F1. Adding the prompt did not consistently improve results.

## Repository Structure

```text
COMPSC448_MIDTERM/
├── README.md
├── requirements.txt
├── data/
│   ├── README.md
│   └── llm_responses.csv
├── src/
│   ├── inspect_data.py
│   ├── baseline.py
│   ├── cnn.py
│   └── rnn.py
├── results/
│   ├── baseline_metrics.csv
│   ├── cnn_metrics.csv
│   └── rnn_metrics.csv
├── models/
└── report/
    └── project_report.pdf
```

## Installation

Use Python 3.10 or newer. Install dependencies from the repository root:

```bash
python -m pip install -r requirements.txt
```

If `requirements.txt` has not been created, install the packages directly:

```bash
python -m pip install pandas scikit-learn torch
```

## Running the Project

Run each command from the repository root.

### Inspect the dataset

```bash
python src/inspect_data.py
```

This checks the columns, missing values, model balance, prompt grouping, and duplicate prompt/model pairs.

### Run the baseline

```bash
python src/baseline.py
```

This saves results to `results/baseline_metrics.csv`.

### Run the CNN

```bash
python src/cnn.py
```

This saves results to `results/cnn_metrics.csv`.

### Run the BiLSTM

```bash
python src/rnn.py
```

This saves results to `results/rnn_metrics.csv`.

## Limitations

- The final test set includes only 30 responses, so one changed prediction changes accuracy by about 3.3 percentage points.
- Results come from one dataset and one train/test split; they may not generalize to other prompts or newer model versions.
- Formatting, response length, copied interface text, refusals, or references to previous context may make a model recognizable for reasons other than writing style.
- The neural models use a maximum sequence length, so very long responses may be truncated.

## Report

The final project report is located in the `report/` folder. It describes the data collection process, model implementations, experimental results, limitations, and lessons learned.
