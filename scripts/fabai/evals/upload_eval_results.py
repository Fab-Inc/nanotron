# %%
import subprocess
from pathlib import Path

from dotenv import load_dotenv

ROOT = Path(__file__).resolve().parents[3]

load_dotenv(override=True)

s5cmd_path = str(ROOT / ".venv/bin/s5cmd")
cmd = [s5cmd_path, "--json"]
cmd += ["cp", "-s", "-u", "-sp"]
cmd += [
    f"{str(ROOT)}/eval_results/*",
    f"s3://qurating-checkpoints-183631302286-eu-west-2-an/eval_results/",
]
print(" ".join(cmd))
output = subprocess.run(cmd)  # , capture_output=True)
