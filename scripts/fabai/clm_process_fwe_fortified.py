# %%
from dotenv import load_dotenv

from transformers import AutoTokenizer
from nanotron.data.processing import clm_process
from nanotron.custom_streaming_dataset import load_dataset_possibly_azure

load_dotenv(override=True)

# %%
dataset_path = "az://quratingscoressa/quratingfiltered-noemb/qurater_gemma-3-4b-pt_ds-ours_v2-200000"
tokenizer_path = "HuggingFaceTB/SmolLM3-3B"

fwe_ds = load_dataset_possibly_azure(dataset_path)
tokenizer = AutoTokenizer.from_pretrained(tokenizer_path)
tokenizer.padding_side = "left"
sequence_sep_tokens = [tokenizer.bos_token, tokenizer.eos_token, tokenizer.pad_token, tokenizer.unk_token]

# %%
