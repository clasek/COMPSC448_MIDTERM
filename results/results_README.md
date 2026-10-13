# Results

This folder contains the outputs produced by the experiments.

## Recommended Contents

- `metrics.csv` — accuracy, precision, recall, macro-F1, and other evaluation results.
- `figures/` — plots and visualizations.
- `confusion_matrices/` — confusion matrices for each experiment.
- `training_histories/` — optional training and validation loss or accuracy histories.

## Experiment Naming

Use descriptive names that identify the input type and classifier. For example:

- `output_only_cnn`
- `output_only_lstm`
- `input_only_cnn`
- `input_only_lstm`
- `input_output_cnn`
- `input_output_lstm`

## Main Comparison

Results should compare the CNN and RNN/LSTM using:

1. Output only
2. Input only
3. Input plus output

## Interpretation

The final report should explain which models performed best, whether the user prompt improved classification, and what limitations may affect the results.