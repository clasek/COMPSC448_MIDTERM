# Source Code

This folder contains reusable Python code for preprocessing, model construction, training, and evaluation.

## Recommended Files

- `preprocess.py` — cleans text, creates labels, tokenizes responses, and prepares train/validation/test sets.
- `models.py` — defines the CNN and RNN/LSTM architectures.
- `train.py` — contains training loops and saves model checkpoints or training histories.
- `evaluate.py` — calculates accuracy, precision, recall, macro-F1, and confusion matrices.

## Required Models

The project must implement and evaluate:

- A convolutional neural network (CNN)
- A recurrent neural network, such as an LSTM or GRU

## Labels

The target label is `model_name`, which identifies the LLM that generated each response.

## Code Guidelines

Use clear function names, comments where useful, and consistent random seeds. Keep data loading, model definition, training, and evaluation separate so experiments can be repeated easily.