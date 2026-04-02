# %%
import json
import os
import subprocess
from pathlib import Path
from shutil import rmtree

import yaml
from dotenv import load_dotenv

from nanotron.config import Config, LightEvalConfig, get_config_from_file
from nanotron.eval.one_job_runner import LightEvalRunner

ROOT = Path(__file__).resolve().parents[3]

load_dotenv(override=True)

s3_kwargs = {
    "key": os.getenv("AWS_ACCESS_KEY_ID"),
    "secret": os.getenv("AWS_SECRET_ACCESS_KEY"),
}

# %%
nanotron_config_file = ROOT / "configs" / "fabai" / "base-run-30000.yaml"
nanotron_config = Config.load_from_yaml(str(nanotron_config_file))

lighteval_config_file = (
    ROOT / "configs" / "fabai" / "lighteval" / "lighteval-config_base-run-30000.yaml"
)
lighteval_config = get_config_from_file(
    lighteval_config_file, config_class=LightEvalConfig
)

nanotron_config.lighteval = lighteval_config
nanotron_config.lighteval.eval_config_override = str(lighteval_config_file)

# %%
checkpoints_dir = "s3://qurating-checkpoints-183631302286-eu-west-2-an/base-run-30000/"

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
    if isinstance(checkpoints_dir, Path):
        ckpt_file = str(checkpoints_dir / f"{step}" / "config.yaml")
    else:
        print("Downloading checkpoint from s3")
        local_path = ROOT / "checkpoints/smol-playbook-checkpoints" / f"{step}"
        if not local_path.exists():
            local_path.mkdir(exist_ok=True, parents=True)
            print(f"Saving to: {local_path}")
            local_path.mkdir(exist_ok=True, parents=True)
            s5cmd_path = str(ROOT / ".venv/bin/s5cmd")
            cmd = [s5cmd_path, "--json"]
            cmd += ["cp"]
            cmd += [f"{checkpoints_dir}{step}/*", str(local_path)]
            # print(" ".join(cmd))
            output = subprocess.run(cmd)#, capture_output=True)
            if output.returncode != 0:
                raise
        ckpt_file = str(local_path / "config.yaml")
        print("Done")
    print(f"Using checkpoint file: {ckpt_file}")
    runner_input = [{"destination": ckpt_file}]
    le_runner.eval_single_checkpoint(runner_input)
    # if not isinstance(checkpoints_dir, Path):
    #     rmtree(local_path)

