# %%
from pathlib import Path

from datatrove.io import DataFolder
from datatrove.utils.dataset import DatatroveFolderDataset
from dotenv import load_dotenv

load_dotenv(override=True)

OUTPUT_FOLDER = (
    "az://quratingfiltered-preprocessed"
    "/qurater_gemma-3-4b-pt_ds-ours_v2-200000"
    "/tokenized-shuffled"
)
if isinstance(OUTPUT_FOLDER, str) and not OUTPUT_FOLDER.startswith("az://"):
    OUTPUT_FOLDER = Path(OUTPUT_FOLDER)
if isinstance(OUTPUT_FOLDER, Path):
    OUTPUT_FOLDER.mkdir(exist_ok=True, parents=True)

# %%
dt_ds = DatatroveFolderDataset(
    folder_path=(
        str(OUTPUT_FOLDER) if isinstance(OUTPUT_FOLDER, Path) else OUTPUT_FOLDER
    ),
    seq_len=4096,
)
