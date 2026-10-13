from pathlib import Path
import pandas as pd

csv = Path(__file__).resolve().parents[1] / "data" / "llm_responses.csv"
df = pd.read_csv(csv)
print(df.shape, df.columns.tolist(), df.isna().sum(), df["model_name"].value_counts(), sep="\n\n")

counts = df.groupby("prompt_id")["model_name"].nunique()
print("Prompts with fewer than 3 models:", (counts != 3).sum())
print("Duplicate prompt/model pairs:",
      df.duplicated(["prompt_id", "model_name"]).sum())

prompt_counts = df.groupby("prompt")["model_name"].nunique()
print("Unique prompt texts:", len(prompt_counts))
print("Prompts missing a model:", (prompt_counts != 3).sum())

bad_ids = df.loc[df.duplicated(["prompt_id", "model_name"], keep=False), "prompt_id"].unique()
print(df.loc[df["prompt_id"].isin(bad_ids), ["prompt_id", "prompt", "model_name"]].sort_values(["prompt_id", "model_name"]).to_string(index=False))
print("IDs assigned to multiple prompts:", df.groupby("prompt_id")["prompt"].nunique().loc[lambda counts: counts > 1].to_dict())