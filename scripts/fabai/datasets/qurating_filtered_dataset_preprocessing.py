# %%
import os
from pathlib import Path
from typing import Tuple

from datasets import load_dataset
from datatrove.data import Document
from datatrove.executor import LocalPipelineExecutor
from datatrove.io import DataFolder
from datatrove.pipeline.readers import ParquetReader
from datatrove.pipeline.filters.base_filter import BaseFilter
from datatrove.pipeline.tokens import DocumentTokenizer, DocumentTokenizerMerger
from datatrove.pipeline.writers.disk_base import DiskWriter
from dotenv import load_dotenv
from transformers import AutoTokenizer

# %%
class QREvalFilter(BaseFilter):
    def __init__(self, exclusion_writer: DiskWriter = None, batch_size: int = 1):
        super().__init__(exclusion_writer, batch_size)
        #### load eval dataset which will be used to remove items from training
        # data to avoid contamination
        eval_dataset = load_dataset(
            "AI-for-Education/qurating-core-edu-pairs", split="train"
        )
        self.exclude_ids = sorted(set([row["high_id"] for row in eval_dataset.to_list()]))
    
    def filter(self, doc: Document) -> bool | Tuple[bool, str]:
        return doc.id not in self.exclude_ids


# %%
# pqr = ParquetReader(
#     data_folder=df,
#     limit=5000,
# )

# for i, doc in enumerate(pqr.run()):
#     print(i)

# %%
if __name__ == "__main__":
    load_dotenv(override=True)
    azure_kwargs = {
        "account_name": "quratingscoressa",
        "account_key": os.getenv("QURATING_SCORES_AZURE_STORAGE_KEY"),
    }

    SEED = 342045735
    SEED_MERGER = 25374973

    OUTPUT_FOLDER = Path(__file__).resolve().parent / "data"

    # OUTPUT_FOLDER_MERGER = Path(__file__).resolve().parent / "data_merged"
    OUTPUT_FOLDER_MERGER = (
        "az://quratingfiltered-preprocessed"
        "/qurater_gemma-3-4b-pt_ds-ours_v2-200000"
        f"/tokenized-shuffled_seed-{SEED}_merge-seed-{SEED_MERGER}_filtered-qreval"
    )
    if isinstance(OUTPUT_FOLDER, str) and not OUTPUT_FOLDER.startswith("az://"):
        OUTPUT_FOLDER = Path(OUTPUT_FOLDER)

    if isinstance(OUTPUT_FOLDER, Path):
        OUTPUT_FOLDER.mkdir(exist_ok=True, parents=True)
    else:
        OUTPUT_FOLDER = DataFolder(OUTPUT_FOLDER, **azure_kwargs)

    if isinstance(OUTPUT_FOLDER_MERGER, str) and not OUTPUT_FOLDER_MERGER.startswith(
        "az://"
    ):
        OUTPUT_FOLDER_MERGER = Path(OUTPUT_FOLDER_MERGER)

    if isinstance(OUTPUT_FOLDER_MERGER, Path):
        OUTPUT_FOLDER_MERGER.mkdir(exist_ok=True, parents=True)
    else:
        OUTPUT_FOLDER_MERGER = DataFolder(OUTPUT_FOLDER_MERGER, **azure_kwargs)

    LOCAL_WORKING_DIR = Path(__file__).resolve().parent / "local_working"
    LOCAL_WORKING_DIR.mkdir(exist_ok=True, parents=True)

    DATASET_NAME = f"tokenized-shuffled"

    TOKENIZER_NAME = "HuggingFaceTB/SmolLM3-3B"

    dataset_path = "az://quratingfiltered-noemb/qurater_gemma-3-4b-pt_ds-ours_v2-200000"

    df = DataFolder(dataset_path, **azure_kwargs)

    tokenizer = AutoTokenizer.from_pretrained(TOKENIZER_NAME)

    dist_executor = LocalPipelineExecutor(
        pipeline=[
            ParquetReader(
                data_folder=df,
                # limit=5000,
            ),
            QREvalFilter(),
            DocumentTokenizer(
                output_folder=(
                    str(OUTPUT_FOLDER)
                    if isinstance(OUTPUT_FOLDER, Path)
                    else OUTPUT_FOLDER
                ),
                local_working_dir=(
                    str(LOCAL_WORKING_DIR)
                    if isinstance(LOCAL_WORKING_DIR, Path)
                    else LOCAL_WORKING_DIR
                ),
                save_filename=f"{DATASET_NAME}",
                tokenizer_name_or_path=TOKENIZER_NAME,
                eos_token=tokenizer.eos_token,
                shuffle_documents=True,
                seed=SEED,
            ),
        ],
        tasks=1000,
        workers=30,
    )

    # dist_executor.run()

    merge_executor = LocalPipelineExecutor(
        pipeline=[
            DocumentTokenizerMerger(
                input_folder=(
                    str(OUTPUT_FOLDER)
                    if isinstance(OUTPUT_FOLDER, Path)
                    else OUTPUT_FOLDER
                ),
                output_folder=(
                    str(OUTPUT_FOLDER_MERGER)
                    if isinstance(OUTPUT_FOLDER_MERGER, Path)
                    else OUTPUT_FOLDER_MERGER
                ),
                save_filename=f"{DATASET_NAME}",
                seed=SEED_MERGER,
                max_tokens_per_file=int(500e6),
            )
        ],
        depends=dist_executor,
    )

    merge_executor.run()
