# %%
from functools import partial
from typing import Dict, List

import numpy as np
from datasets import Dataset, IterableDataset, Features, Sequence, Value
from dotenv import load_dotenv
from joblib import Parallel, delayed
from transformers import AutoTokenizer

from nanotron.custom_streaming_dataset import load_dataset_possibly_azure

try:
    from nanotron.data.processing import clm_process

    HAS_NANOTRON_PROC = True
except ImportError:
    HAS_NANOTRON_PROC = False

load_dotenv(override=True)

if not HAS_NANOTRON_PROC:

    def clm_process(
        raw_dataset: "Dataset",
        tokenizer,
        text_column_name: str,
        dataset_processing_num_proc_per_process: int,
        dataset_overwrite_cache: bool,
        sequence_length: int,
    ):
        """
        Concatenate all texts from raw_dataset and generate chunks of `sequence_length + 1`,
        where chunks overlap by a single token.

        Args:
            raw_dataset: Dataset containing raw text
            tokenizer: HuggingFace tokenizer
            text_column_name: Name of the column containing text data
            dataset_processing_num_proc_per_process: Number of processes for parallelization
            dataset_overwrite_cache: Whether to overwrite the cache
            sequence_length: Maximum sequence length

        Returns:
            Processed dataset with tokenized sequences
        """
        # Adapted from https://github.com/huggingface/transformers/blob/47e1676255e5dd86b9541f734cd4f4bdcbb50f4a/examples/pytorch/language-modeling/run_clm.py#L391-L439

        def group_texts(
            examples: Dict[str, List[np.ndarray]],
        ) -> Dict[str, List[np.ndarray]]:
            # Concatenate all texts.
            concatenated_examples = {k: np.concatenate(v) for k, v in examples.items()}
            total_length = len(concatenated_examples[next(iter(examples.keys()))])
            # WARNING: We drop the small remainder, we could add padding if the model supported it instead of this drop, you can
            # customize this part to your needs.
            if total_length >= sequence_length + 1:
                total_length = (
                    (total_length - 1) // sequence_length
                ) * sequence_length + 1
            # Split by chunks of sequence_length.
            result = {
                k: [
                    t[i : i + sequence_length + 1]
                    for i in range(
                        0, total_length - (sequence_length + 1), sequence_length
                    )
                ]
                for k, t in concatenated_examples.items()
            }
            return result

        def _tokenize_and_group_texts(texts: List[str]) -> Dict[str, List[np.ndarray]]:
            tokenized_batch = tokenizer.batch_encode_plus(
                texts, return_attention_mask=False, return_token_type_ids=False
            )
            tokenized_batch = {
                k: [np.array(tokenized_texts) for tokenized_texts in v]
                for k, v in tokenized_batch.items()
            }
            return group_texts(tokenized_batch)

        train_dataset = raw_dataset.map(
            _tokenize_and_group_texts,
            input_columns=text_column_name,
            remove_columns=raw_dataset.column_names,
            features=Features(
                {
                    "input_ids": Sequence(
                        feature=Value(dtype="int64"), length=sequence_length + 1
                    )
                }
            ),
            batched=True,
            num_proc=dataset_processing_num_proc_per_process,
            load_from_cache_file=not dataset_overwrite_cache,
            desc=f"Grouping texts in chunks of {sequence_length+1}",
        )
        return train_dataset


# %%
dataset_path = "az://quratingscoressa/quratingfiltered-noemb/qurater_gemma-3-4b-pt_ds-ours_v2-200000"
tokenizer_path = "HuggingFaceTB/SmolLM3-3B"

fwe_ds = load_dataset_possibly_azure(dataset_path)
tokenizer = AutoTokenizer.from_pretrained(tokenizer_path)
tokenizer.padding_side = "left"
sequence_sep_tokens = [
    tokenizer.bos_token,
    tokenizer.eos_token,
    tokenizer.pad_token,
    tokenizer.unk_token,
]


# %%
def gen_from_iterable_dataset(iterable_ds):
    yield from iterable_ds


n_shards = fwe_ds.n_shards
rng = np.random.default_rng(529779802)
shuffled_shards = rng.permutation(n_shards)


def process_shard(fwe_ds: IterableDataset, tokenizer, shard, return_result=False):
    shard_ds_stream = fwe_ds.shard(fwe_ds.n_shards, shard, contiguous=True).shuffle()
    shard_ds = Dataset.from_generator(
        partial(gen_from_iterable_dataset, shard_ds_stream),
        features=shard_ds_stream.features,
    )
    processed_shard = clm_process(
        shard_ds,
        tokenizer,
        text_column_name="text",
        dataset_processing_num_proc_per_process=1,
        dataset_overwrite_cache=True,
        sequence_length=4096,
    )
    if return_result:
        return processed_shard

ps = process_shard(fwe_ds, tokenizer, 0, True)

# with Parallel(n_jobs=30, verbose=60) as p:


