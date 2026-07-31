# %%
import json
from pathlib import Path

import numpy as np
from datasets import Dataset, IterableDataset
from joblib import Parallel, delayed
from numpy.random import default_rng
from qurating.scoring_projects.fwe_fortified.custom_streaming_dataset import (
    load_dataset_possibly_azure,
)

# %%
url = "az://quratingscoressa/quratingfiltered-noemb/qurater_gemma-3-4b-pt_ds-ours_v2-200000"
streaming_ds = load_dataset_possibly_azure(url)


# %%
async def high_prim(batch):
    return [example > 5 for example in batch["education_level_primary_average"]]


def high_prim_shard(shard_ds: IterableDataset):
    return list(shard_ds.filter(high_prim, batched=True))


seed = 95322309
rng = default_rng(seed)

num_shards = streaming_ds.num_shards
max_shards = int(np.ceil(num_shards * 0.1))
shard_idxs = rng.permutation(num_shards)

shard_ds = streaming_ds.shard(num_shards, 0)
# shard_ds.filter(high_prim, batched=True)

p = Parallel(n_jobs=30, verbose=60)
out = p(
    delayed(high_prim_shard)(streaming_ds.shard(num_shards, i))
    for i in shard_idxs[:max_shards]
)

filtered_ds_high = Dataset.from_list([ex for shard in out for ex in shard])

# %%
url = "az://quratingscoressa/quratingfiltered-low-noemb/qurater_gemma-3-4b-pt_ds-ours_v2-200000"
streaming_ds = load_dataset_possibly_azure(url)

rng = default_rng(seed)

num_shards = streaming_ds.num_shards
max_shards = int(np.ceil(num_shards * 0.1))
shard_idxs = rng.permutation(num_shards)

shard_ds = streaming_ds.shard(num_shards, 0)
# shard_ds.filter(high_prim, batched=True)

p = Parallel(n_jobs=30, verbose=60)
out = p(
    delayed(high_prim_shard)(streaming_ds.shard(num_shards, i))
    for i in shard_idxs[:max_shards]
)

filtered_ds_low = Dataset.from_list([ex for shard in out for ex in shard])

# %%
examples_list = {}
examples_list["high"] = {}
examples_list["mid"] = {}
ds = filtered_ds_high
df = ds.to_pandas()
dims = [
    "factual_accuracy_average",
    "lesson_engagement_average",
    "pedagogical_structure_average",
]
for dim in dims:
    print(f"High: {dim}")
    top_examples = np.argsort(np.array(df[dim]))[-1::-1]
    examples_df = df.iloc[top_examples[:10]]
    examples_list["high"][dim] = [row["text"] for row in Dataset.from_pandas(examples_df)]
    examples_df = df.iloc[top_examples[-10:]]
    examples_list["mid"][dim] = [row["text"] for row in Dataset.from_pandas(examples_df)]

examples_list["low"] = {}
ds = filtered_ds_low
df = ds.to_pandas()
dims = [
    "factual_accuracy_average",
    "lesson_engagement_average",
    "pedagogical_structure_average",
]
for dim in dims:
    print(f"Low: {dim}")
    bot_examples = np.argsort(np.array(df[dim]))
    examples_df = df.iloc[bot_examples[:10]]
    examples_list["low"][dim] = [row["text"] for row in Dataset.from_pandas(examples_df)]

# %%
with open("examples.json", "w") as f:
    json.dump(examples_list, f, indent=2, ensure_ascii=False)

# %%
for highlow in ["high", "mid", "low"]:
    for dim in dims:
        outfold = Path(__file__).resolve().parent / "examples" / highlow / dim
        outfold.mkdir(exist_ok=True, parents=True)
        for i, ex in enumerate(examples_list[highlow][dim]):
            outfile = outfold / f"{i :03d}.txt"
            with open(outfile, "w") as f:
                f.write(ex)
