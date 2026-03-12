# %%
import os
import time
from pathlib import Path
from tempfile import NamedTemporaryFile

from cloudpathlib import AzureBlobClient, CloudPath
from datasets import Dataset, IterableDataset, get_dataset_config_names
from dotenv import load_dotenv

from custom_streaming_dataset import load_dataset_possibly_azure

load_dotenv(override=True)

# %%
ds = load_dataset_possibly_azure(
    "az://quratingscoressa/quratingfiltered-noemb/qurater_gemma-3-4b-pt_ds-ours_v2-200000"
)

ds_iter = iter(ds)

# %%
next(ds_iter)

# %%
ds = load_dataset_possibly_azure(
    "airtrain-ai/fineweb-edu-fortified",
    name="CC-MAIN-2013-20",
    streaming=True,
    split="train",
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


# %%
from functools import partial
from nanotron.trainer import DistributedTrainer
from nanotron.data.processing import clm_process, get_datasets

raw_dataset = get_datasets(
    hf_dataset_or_datasets="az://quratingscoressa/quratingfiltered-noemb/qurater_gemma-3-4b-pt_ds-ours_v2-200000",
    hf_dataset_config_name=None,
    splits=["train"],
)["train"]


def gen_from_iterable_dataset(iterable_ds):
    yield from iterable_ds


ds = Dataset.from_generator(
    partial(gen_from_iterable_dataset, raw_dataset), features=raw_dataset.features
)
# %%
from torch.utils import collect_env

collect_env.main()

print(os.getenv("LOCAL_RANK"))

# %%
from fsspec import url_to_fs

fs_folder, stripped_folder_path = url_to_fs(
    "az://quratingfiltered-noemb/qurater_gemma-3-4b-pt_ds-ours_v2-200000",
    account_name="quratingscoressa",
    account_key=os.getenv("QURATING_SCORES_AZURE_STORAGE_KEY"),
)
