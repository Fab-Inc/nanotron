# %%
import gzip
import os
from pathlib import Path

import boto3
import pyarrow as pa
from botocore.exceptions import ClientError
from datatrove.executor import LocalPipelineExecutor
from datatrove.io import DataFolder
from datatrove.pipeline.filters import SamplerFilter
from datatrove.pipeline.formatters.base import BaseFormatter
from datatrove.pipeline.readers import HuggingFaceDatasetReader
from datatrove.pipeline.writers import ParquetWriter
from dotenv import load_dotenv


class SoftwareHeritageDownloader(BaseFormatter):

    name = "Software Heritage Downloader"

    def format(self, text: str) -> str:
        s3 = boto3.client("s3")
        bucket_name = "softwareheritage"
        key = f"content/{text}"
        try:
            obj = s3.get_object(Bucket=bucket_name, Key=key)
            with gzip.GzipFile(fileobj=obj["Body"]) as fin:
                content = fin.read().decode("utf-8", errors="ignore")
            return content
        except ClientError as e:
            if e.response["Error"]["Code"] == "NoSuchKey":
                print(f"File not found: {key}")
                return ""
            else:
                raise


# %%

if __name__ == "__main__":
    load_dotenv(override=True)
    azure_kwargs = {
        "account_name": "quratingscoressa",
        "account_key": os.getenv("QURATING_SCORES_AZURE_STORAGE_KEY"),
    }
    DS = "stack-edu-python"

    TARGET = 50_000_000_000
    FINEWEB_SCHEMA = pa.schema(
        [
            ("text", pa.string()),
            ("id", pa.string()),
            ("dump", pa.string()),
            ("url", pa.string()),
            ("date", pa.string()),
            ("file_path", pa.string()),
            ("language", pa.string()),
            ("language_score", pa.float64()),
            ("token_count", pa.int64()),
            ("score", pa.float64()),
            ("int_score", pa.int64()),
            ("dataset", pa.string()),
        ]
    )
    STACK_EDU_SCHEMA = pa.schema(
        [
            ("text", pa.string()),
            ("language", pa.string()),
            ("repo_name", pa.string()),
            ("path", pa.string()),
            ("src_encoding", pa.string()),
            ("length_bytes", pa.string()),
            ("score", pa.float64()),
            ("int_score", pa.int64()),
            ("detected_licences", pa.list_(pa.string())),
            ("license_type", pa.string()),
        ]
    )

    if DS == "stack-edu-python":
        # stack-edu-python
        savefold = "stack-edu-python"
        dataset_path = "HuggingFaceTB/stack-edu"
        dataset_options = {"split": "train", "name": "Python"}
        reader_options = {
            "text_key": "blob_id",
            "id_key": "blob_id",
            "streaming": False,
        }
        SZ = 21_800_000_000
        schema = STACK_EDU_SCHEMA
        extra_pipeline_stage = SoftwareHeritageDownloader()
    elif DS == "dclm":
        # DCLM
        savefold = "dclm_50BT"
        dataset_path = "mlfoundations/dclm-baseline-1.0-parquet"
        dataset_options = {"split": "train"}
        reader_options = {"streaming": True}
        text_key = "text"
        SZ = 3_468_923_154_406
        schema = None
        extra_pipeline_stage = None
    elif DS == "fineweb-edu":
        # Fineweb
        savefold = "fineweb-edu_50BT"
        dataset_path = "HuggingFaceFW/fineweb-edu"
        dataset_options = {"split": "train"}
        reader_options = {"streaming": True}
        SZ = 1_567_210_463_942
        schema = FINEWEB_SCHEMA
        extra_pipeline_stage = None
    elif DS == "finemath-3plus":
        # finemath-3plus
        savefold = "finemath_3plus"
        dataset_options = {"split": "train", "name": "finemath-3plus"}
        reader_options = {"streaming": True}
        dataset_path = "HuggingFaceTB/finemath"
        SZ = 34_000_000_000
        schema = None
        extra_pipeline_stage = None
    else:
        raise NotImplementedError(f"dataset {DS} not supported")

    SEED = 342045735
    SEED_MERGER = 8235496
    SEED_SAMPLER = 48256546

    # OUTPUT_FOLDER = Path(__file__).resolve().parent / "data"
    OUTPUT_FOLDER = (
        "az://additional-datasets-preprocessed"
        "/raw"
        f"/{savefold}"
        f"/sampler_seed-{SEED_SAMPLER}"
    )
    if isinstance(OUTPUT_FOLDER, str) and not OUTPUT_FOLDER.startswith("az://"):
        OUTPUT_FOLDER = Path(OUTPUT_FOLDER)

    if isinstance(OUTPUT_FOLDER, Path):
        OUTPUT_FOLDER.mkdir(exist_ok=True, parents=True)
    else:
        OUTPUT_FOLDER = DataFolder(OUTPUT_FOLDER, **azure_kwargs)

    # OUTPUT_FOLDER_MERGER = Path(__file__).resolve().parent / "data_merged"
    OUTPUT_FOLDER_MERGER = (
        "az://additional-datasets-preprocessed"
        "/raw"
        f"/{savefold}"
        f"/sampler_seed-{SEED_SAMPLER}"
        f"/tokenized-shuffled_seed-{SEED}_merge-seed_{SEED_MERGER}"
    )
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

    LOG_BASE_DIR = "az://additional-datasets-preprocessed" "/raw" f"/{savefold}" "/logs"
    # LOG_BASE_DIR = Path().resolve() / f"{savefold}" "/logs"
    if isinstance(LOG_BASE_DIR, str) and not LOG_BASE_DIR.startswith("az://"):
        LOG_BASE_DIR = Path(LOG_BASE_DIR)

    if isinstance(LOG_BASE_DIR, Path):
        logging_dir = LOG_BASE_DIR
        logging_dir.mkdir(exist_ok=True, parents=True)
        logging_dir = str(logging_dir)
    else:
        logging_dir = f"{LOG_BASE_DIR}"
        logging_dir = DataFolder(logging_dir, **azure_kwargs)

    DATASET_NAME = f"sampled-{SEED_SAMPLER}"

    output_folder = str(OUTPUT_FOLDER)
    local_working_dir = str(LOCAL_WORKING_DIR)

    ## set up pipeline with optional extra_pipeline_step
    use_pipeline = [
        HuggingFaceDatasetReader(
            dataset=dataset_path,
            dataset_options=dataset_options,
            **reader_options,
        ),
        SamplerFilter(rate=min(1.0, TARGET / SZ), seed=SEED_SAMPLER),
    ]
    if extra_pipeline_stage is not None:
        use_pipeline.append(extra_pipeline_stage)
    use_pipeline.append(
        ParquetWriter(
            output_folder=OUTPUT_FOLDER,
            output_filename=DATASET_NAME + "-${rank}",
            schema=schema,
        )
    )
    ###
    ### set up exectutor
    prev = LocalPipelineExecutor(
        pipeline=use_pipeline,
        logging_dir=logging_dir,
        tasks=1000,
        workers=100,
    )
    # run executor
    prev.run()
