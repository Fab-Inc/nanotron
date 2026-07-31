# %%
import os
from pathlib import Path

from datatrove.executor import LocalPipelineExecutor
from datatrove.io import DataFolder
from datatrove.pipeline.readers import ParquetReader
from datatrove.pipeline.tokens import DocumentTokenizer, DocumentTokenizerMerger
from dotenv import load_dotenv
from transformers import AutoTokenizer

load_dotenv(override=True)
azure_kwargs = {
    "account_name": "quratingscoressa",
    "account_key": os.getenv("QURATING_SCORES_AZURE_STORAGE_KEY"),
}


# %%
dataset_path = "az://additional-datasets-preprocessed/raw/stack-edu-python/sampler_seed-48256546"
df = DataFolder(dataset_path, **azure_kwargs)
pqr = ParquetReader(data_folder=df, limit=5)

for i, doc in enumerate(pqr.run()):
    print(i)
