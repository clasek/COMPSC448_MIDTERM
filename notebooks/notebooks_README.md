# Notebooks

This folder contains Jupyter or Google Colab notebooks used to inspect the data, train the models, and analyze the results.

## Recommended Notebook Order

1. `01_data_inspection.ipynb` — loads the CSV and checks columns, class balance, missing values, and response lengths.
2. `02_baseline_model.ipynb` — trains a simple TF-IDF and logistic regression baseline.
3. `03_cnn_model.ipynb` — trains and evaluates the required convolutional neural network.
4. `04_rnn_model.ipynb` — trains and evaluates the required recurrent neural network, such as an LSTM or GRU.
5. `05_results_analysis.ipynb` — compares experiments and creates tables, plots, and confusion matrices.

## Input Settings

Each classifier should be evaluated using:

- Output only
- Input only
- Input plus output

## Reproducibility

Set random seeds when possible and record important settings, including:

- Train/validation/test split
- Tokenizer and vocabulary settings
- Maximum sequence length
- Batch size
- Learning rate
- Number of epochs
- Random seed

## Notes

Run the notebooks in numerical order. Do not store API keys, passwords, or private conversations in notebooks.