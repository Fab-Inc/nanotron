# %%
from dotenv import load_dotenv

from transformers import AutoTokenizer
from nanotron.data.processing import clm_process
from nanotron.custom_streaming_dataset import load_dataset_possibly_azure

load_dotenv(override=True)

# %%
dataset_path = "az://quratingscoressa/quratingfiltered-noemb/qurater_gemma-3-4b-pt_ds-ours_v2-200000"

streaming_ds = load_dataset_possibly_azure(dataset_path)