# Identifying LLMs from Their Responses

This project investigates whether machine learning models can identify which
large language model generated a response.

## Research Questions

1. Can a classifier identify which LLM generated a response?
2. Does the user's prompt improve LLM classification?
3. Do LLM fingerprints generalize across different task types?

## Models

- Convolutional Neural Network (CNN)
- Recurrent Neural Network using LSTM

## Input Settings

- Output only
- Input only
- Input plus output

## Dataset

The dataset contains the following columns:

- `prompt_id`
- `task_type`
- `prompt`
- `model_name`
- `response`

The same prompts were given to each LLM so that the models could be compared
fairly.

## Repository Structure

- `data/` — collected dataset
- `notebooks/` — analysis and experiments
- `src/` — reusable Python code
- `results/` — metrics and figures
- `report/` — final project report

## How to Run

The notebooks can be opened and run using Google Colab or Jupyter Notebook.
