# %%
from pathlib import Path
import os
import subprocess
import json

from dotenv import load_dotenv

from fsspec import url_to_fs
from nanotron.config import Config
from nanotron.eval.one_job_runner import LightEvalRunner
from s3fs.core import S3FileSystem
import aiobotocore.session

ROOT = Path(__file__).resolve().parents[3]

load_dotenv(override=True)

s3_kwargs = {
    "key": os.getenv("AWS_ACCESS_KEY_ID"),
    "secret": os.getenv("AWS_SECRET_ACCESS_KEY"),
}

# %%
nanotron_config_file = ROOT / "configs" / "fabai" / "base-run-100-no-evals.yaml"
nanotron_config = Config.load_from_yaml(str(nanotron_config_file))

lighteval_config_file = (
    ROOT / "configs" / "fabai" / "lighteval" / "lighteval-config.yaml"
)
nanotron_config = Config.load_from_yaml(str(nanotron_config_file))

# %%
checkpoints_dir = "s3://qurating-checkpoints-183631302286-eu-west-2-an/base-run-100/"

if isinstance(checkpoints_dir, str):
    if checkpoints_dir.startswith("s3://"):
        s5cmd_path = str(ROOT / ".venv/bin/s5cmd")
        cmd = [s5cmd_path, "--json"]
        cmd += ["ls"]
        cmd += [checkpoints_dir]

        print(" ".join(cmd))

        output = subprocess.run(cmd, capture_output=True)
        dirlist = [
            lnjs["key"]
            for ln in output.stdout.decode().splitlines()
            if (lnjs := json.loads(ln))["type"] == "directory"
        ]
        steps = sorted(int(dirname.rstrip("/").split("/")[-1]) for dirname in dirlist)
    else:
        checkpoints_dir = Path(checkpoints_dir)
else:
    raise ValueError("checkpoints_dir must be str or Path")

if isinstance(checkpoints_dir, Path):
    all_ckpt = checkpoints_dir.rglob("config.yaml")
    steps = sorted(int(f.parent.name) for f in all_ckpt)

# %%
for step in steps:
    nanotron_config.general.step = step
    le_runner = LightEvalRunner(
        config=nanotron_config, parallel_context=nanotron_config.parallelism
    )
    ckpt_file = str(checkpoints_dir / f"{step}" / "config.yaml")
    print(ckpt_file)
    runner_input = [{"destination": ckpt_file}]
    le_runner.eval_single_checkpoint(runner_input)
