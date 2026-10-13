# Data

This folder contains the datasets used for the LLM authorship classification project.

## Files

- `llm_responses.csv` — the collected prompt-response dataset.
- `processed_data.csv` — optional cleaned or transformed version used for modeling.

## Dataset Columns

The main dataset contains:

- `prompt_id` — unique identifier for each prompt.
- `task_type` — category of the prompt, such as explanation, writing, coding, or mathematics.
- `prompt` — the exact input given to the LLM.
- `model_name` — the LLM that produced the response.
- `response` — the generated response.

## Data Collection

The same prompts were given to each LLM so that model performance could be compared fairly. The dataset should contain approximately the same number of examples for each model.

## Privacy and Licensing

No private, sensitive, or copyrighted material should be included in this folder. Any public datasets used in the project must be credited in the main project README and final report.

## Reproducibility

The original collected dataset should not be overwritten. If preprocessing changes the data, save the processed version as a separate file and document the changes.