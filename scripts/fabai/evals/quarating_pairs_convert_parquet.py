# %%
from pathlib import Path
import json

from dotenv import load_dotenv
from datasets import Dataset, load_dataset

load_dotenv(override=True)

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[2]

# %%
jsonl_file = ROOT / "data" / "qurating_pairs_full_v1.jsonl"

ds_list = []
with open(jsonl_file) as f:
    for ln in f.readlines():
        ds_list.append(json.loads(ln))

# %%
drop_keys = ["high_rank", "low_rank"]
ds = Dataset.from_list(
    [{key: val for key, val in el.items() if key not in drop_keys} for el in ds_list]
)

# %%
ds.to_parquet(ROOT / "data" / jsonl_file.with_suffix(".parquet").name)

# %%
ds.push_to_hub(repo_id="AI-for-Education/qurating-core-edu-pairs")

# %%
ds = load_dataset("AI-for-Education/qurating-core-edu-pairs", split="train")