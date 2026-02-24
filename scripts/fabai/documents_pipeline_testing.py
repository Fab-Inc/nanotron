# %%
import os
from pathlib import Path

from datatrove.executor import LocalPipelineExecutor
from datatrove.io import DataFolder
from datatrove.pipeline.readers import ParquetReader
from datatrove.pipeline.tokens import DocumentTokenizer, DocumentTokenizerMerger
from dotenv import load_dotenv
from transformers import AutoTokenizer

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

    # OUTPUT_FOLDER = Path(__file__).resolve().parent / "data"
    OUTPUT_FOLDER = (
        "az://quratingfiltered-preprocessed"
        "/qurater_gemma-3-4b-pt_ds-ours_v2-200000"
        "/tokenized-shuffled"
    )
    if isinstance(OUTPUT_FOLDER, str) and not OUTPUT_FOLDER.startswith("az://"):
        OUTPUT_FOLDER = Path(OUTPUT_FOLDER)
    else:
        OUTPUT_FOLDER = DataFolder(OUTPUT_FOLDER, **azure_kwargs)
    if isinstance(OUTPUT_FOLDER, Path):
        OUTPUT_FOLDER.mkdir(exist_ok=True, parents=True)

    OUTPUT_FOLDER_MERGER = Path(__file__).resolve().parent / "data_merged"
    OUTPUT_FOLDER_MERGER.mkdir(exist_ok=True, parents=True)

    LOCAL_WORKING_DIR = Path(__file__).resolve().parent / "local_working"
    LOCAL_WORKING_DIR.mkdir(exist_ok=True, parents=True)

    DATASET_NAME = f"tokenized-shuffled-{SEED}"

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
                save_filename=f"{DATASET_NAME}_tokenized",
                tokenizer_name_or_path=TOKENIZER_NAME,
                eos_token=tokenizer.eos_token,
                shuffle=True,
                seed=SEED,
            ),
        ],
        tasks=1000,
        workers=30,
    )

    dist_executor.run()

    # merge_executor = LocalPipelineExecutor(
    #     pipeline=[
    #         DocumentTokenizerMerger(
    #             input_folder=str(OUTPUT_FOLDER),
    #             output_folder=str(OUTPUT_FOLDER_MERGER),
    #             save_filename=f"{DATASET_NAME}",
    #             seed=8235496,
    #         )
    #     ],
    #     depends=dist_executor,
    # )

    # merge_executor.run()
