# %%
import subprocess
import threading
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from functools import partial
from multiprocessing import cpu_count
from typing import Dict, List

import numpy as np
from dotenv import load_dotenv
from joblib import Parallel, delayed
from transformers import AutoTokenizer

try:
    from nanotron.data.processing import clm_process

    HAS_NANOTRON_PROC = True
except ImportError:
    HAS_NANOTRON_PROC = False
from datasets import Dataset, Features, IterableDataset, Sequence, Value

from nanotron.custom_streaming_dataset import load_dataset_possibly_azure

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

streaming_ds = load_dataset_possibly_azure(dataset_path)

tokenizer = AutoTokenizer.from_pretrained(tokenizer_path)
tokenizer.padding_side = "left"
sequence_sep_tokens = [
    tokenizer.bos_token,
    tokenizer.eos_token,
    tokenizer.pad_token,
    tokenizer.unk_token,
]

streaming_ds._ex_iterable


# %%
def gen_from_iterable_dataset(iterable_ds):
    yield from iterable_ds


def get_shard_length(streaming_ds: IterableDataset, shard):
    shard_ds = streaming_ds.shard(
        streaming_ds.num_shards, shard, contiguous=True
    ).to_list()
    return shard, len(shard_ds)


def process_shard(streaming_ds: IterableDataset, shard, tokenizer):
    shard_ds = streaming_ds.shard(streaming_ds.num_shards, shard, contiguous=True)
    proc_ds = clm_process(
        raw_dataset=Dataset.from_list(shard_ds.to_list()),
        tokenizer=tokenizer,
        text_column_name="text",
        dataset_processing_num_proc_per_process=1,
        dataset_overwrite_cache=True,
        sequence_length=4096,
    )
    return proc_ds


n_cpus = cpu_count()

# %%
n_shards_test = 500

with Parallel(n_jobs=n_cpus - 1, verbose=60) as p:
    loaded_ds_len = p(
        delayed(get_shard_length)(streaming_ds, shard)
        # for shard in range(streaming_ds.num_shards)
        for shard in range(n_shards_test)
    )
print(loaded_ds_len)

subprocess.run("rm -fr ~/.cache", shell=True)

# %%
indices_list = []
for shardi, n in loaded_ds_len:
    indices_list.append([np.ones(shape=n) * shardi, np.arange(n)])

shuffled_indices_starter = np.hstack(indices_list).astype(int)

rng = np.random.default_rng(seed=28540236)
shuffled_indices_full = rng.permutation(shuffled_indices_starter, axis=1).T

# %%
lock_input = threading.Lock()
lock_loading = {shard: threading.Lock() for shard in range(streaming_ds.num_shards)}
lock_output = threading.Lock()

n_rows_per_output_shard = 10000
input_cache_limit = 1000

DEBUG = True


def apply_shuffle(ds: IterableDataset, shard, index, output_order):
    if DEBUG:
        print({key: len(val) for key, val in mem_cache_output.items()})
        # print(shard, index, output_order)
        # print(being_loaded)
    # manage input cache size
    # this is a trade-off between memory consumption and speed
    if len(mem_cache_input) > input_cache_limit:
        cache_nrows = {key: len(ds) for key, ds in mem_cache_input.items()}
        smallest_cache_key = sorted(cache_nrows.items(), key=lambda x: x[1])[0][0]
        with lock_input:
            mem_cache_input.pop(smallest_cache_key)
            subprocess.run("rm -fr ~/.cache", shell=True)
        #############
    while shard not in mem_cache_input:
        if shard not in being_loaded:
            with lock_loading[shard]:
                being_loaded.add(shard)
                shard_stream_ds = ds.shard(ds.num_shards, shard, contiguous=True)
                tmp_ds = shard_stream_ds.to_list()
                with lock_input:
                    mem_cache_input[shard] = tmp_ds
                being_loaded.remove(shard)
        else:
            if DEBUG:
                print(f"Waiting for shard {shard} to load")
    with lock_input:
        if shard in mem_cache_input:
            row = mem_cache_input[shard][index]
        else:
            return apply_shuffle(ds, shard, index, output_order)
    output_shard = output_order // n_rows_per_output_shard
    with lock_output:
        if output_shard not in mem_cache_output:
            mem_cache_output[output_shard] = []
        elif mem_cache_output[output_shard] is None:
            raise ValueError(
                f"{output_shard} is in the range of previously completed output shard"
            )
        else:
            mem_cache_output[output_shard].append(row)
            if len(mem_cache_output[output_shard]) >= n_rows_per_output_shard:
                print(f"Completed shard {output_shard}")
                mem_cache_output[output_shard] = None


mem_cache_input = {}
being_loaded = set()
mem_cache_output = {}
with ThreadPoolExecutor(max_workers=30) as executor:
    futures = [
        executor.submit(
            apply_shuffle,
            ds=streaming_ds,
            shard=shard,
            index=index,
            output_order=output_order,
        )
        for output_order, (shard, index) in enumerate(shuffled_indices_full)
    ]

    out = [future.result() for future in as_completed(futures)]

# %%
# with Parallel(n_jobs=n_cpus - 1, verbose=60) as p:
#     loaded_ds_list = p(
#         delayed(process_shard)(streaming_ds, shardi, tokenizer)
#         for shardi in range(streaming_ds.num_shards // 2)
#     )
