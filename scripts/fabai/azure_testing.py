# %%
import os
import time
from pathlib import Path
from tempfile import NamedTemporaryFile

from cloudpathlib import AzureBlobClient, CloudPath
from datasets import Dataset, IterableDataset, get_dataset_config_names
from dotenv import load_dotenv

load_dotenv(override=True)


# %%
def count_filtered_batches(subset, model_string, azure_client_kwargs):
    client = AzureBlobClient(**azure_client_kwargs)
    full_path = f"quratingfiltered-noemb/{model_string}/{subset}"
    cloud_path = CloudPath(f"az://{full_path}", client=client)
    fllist = list(cloud_path.glob(f"*{subset}_*.parquet"))
    return len(fllist)


def load_filtered_batch(
    subset, batchi, model_string, azure_client_kwargs, to_pandas=True
):
    client = AzureBlobClient(**azure_client_kwargs)
    full_path = (
        f"quratingfiltered-noemb/{model_string}/{subset}/{subset}_{batchi :04d}.parquet"
    )
    cloud_path = CloudPath(f"az://{full_path}", client=client)
    ## return null if it doesn't exist
    if not cloud_path.exists():
        return
    ## otherwise return id and average score cols
    with NamedTemporaryFile(mode="+wb") as f:
        f.write(cloud_path.read_bytes())
        f.seek(0)
        ds = Dataset.from_parquet(f.name, keep_in_memory=True)
        print(f"Len batch: {subset} - {batchi}: {len(ds)}")

    if to_pandas:
        ds_df = ds.to_pandas()
        ds.cleanup_cache_files()
        return ds_df
    else:
        return ds


# %%
configs = get_dataset_config_names("airtrain-ai/fineweb-edu-fortified")

model_name = "AI-for-Education/qurater_gemma-3-4b-pt_ds-ours_v2-200000"
model_string = [
    substr for substr in Path(model_name).parts if substr.startswith("qurater_")
]
assert len(model_string) == 1
model_string = model_string[0]

# %%
AZURE_CLIENT_KWARGS = {
    "account_url": "https://quratingscoressa.blob.core.windows.net",
    "credential": os.getenv("QURATING_SCORES_AZURE_STORAGE_KEY"),
}

# # %%
# n_batches = {}
# for subset in configs:
#     st = time.perf_counter()
#     n_batches[subset] = count_filtered_batches(
#         subset, model_string=model_string, azure_client_kwargs=AZURE_CLIENT_KWARGS
#     )
#     print(f"{subset}: {time.perf_counter() - st}")


# %%
def data_generator_sharder():
    configs = get_dataset_config_names("airtrain-ai/fineweb-edu-fortified")
    shards = []
    for subset in configs:
        print(subset)
        n_batches = count_filtered_batches(subset, model_string, AZURE_CLIENT_KWARGS)
        for batchi in range(n_batches):
            shards.append((subset, batchi))
    return shards


def data_generator(shards):
    for subset, batchi in shards:
        batch_ds = load_filtered_batch(
            subset, batchi, model_string, AZURE_CLIENT_KWARGS, to_pandas=False
        )
        for row in batch_ds:
            yield row


shards = data_generator_sharder()

batch = load_filtered_batch(
    configs[0], 0, model_string, AZURE_CLIENT_KWARGS, to_pandas=False
)
features = batch.features

# %%
ds = IterableDataset.from_generator(
    data_generator, features=features, gen_kwargs={"shards": shards}
)

ds_iter = iter(ds)

# %%
next(ds_iter)

# %%
shuff_ds = ds.shuffle(buffer_size=10000)

ds_iter = iter(shuff_ds)

# %%
next(ds_iter)

# %%
ds_take = list(shuff_ds.take(50))