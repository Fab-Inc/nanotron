# %%
import os
from pathlib import Path
import subprocess

from numpy.random import default_rng
from dotenv import load_dotenv

load_dotenv(override=True)

AWS_ACCESS_KEY_NAMES = ["AWS_ACCESS_KEY_ID", "AWS_SECRET_ACCESS_KEY"]
for key_type in AWS_ACCESS_KEY_NAMES:
    if os.getenv(key_type):
        print(f"Got {key_type}")

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[2]

# %%
s3_path = "s3://qurating-checkpoints-183631302286-eu-west-2-an/smollm-baseline-0/"
s3_bucket, s3_prefix = s3_path.replace("s3://", "").split("/", maxsplit=1)
s3_region = "eu-west-2-an"

s3_path_direct_link = f"https://s3.console.aws.amazon.com/s3/buckets/{s3_bucket}?region={s3_region}&prefix={s3_prefix}&showversions=false"

# %%
# create random numbers for test files
rng = default_rng(98238373)
test_file_dir = HERE / "s3_test_files"
for i in range(5):
    test_fl = test_file_dir / f"{i}.txt"
    with open(test_fl, "w") as f:
        seq = rng.integers(255, size=80)
        for num in seq:
            f.write(f"{num} ")

s5cmd_path = str(ROOT / ".venv/bin/s5cmd")
cmd = [s5cmd_path, "--json"]
cmd += ["cp", "--exclude", "*.lock", "--exclude", "*.lock.*"]
cmd += [str(test_file_dir), s3_path]

print (" ".join(cmd))

output = subprocess.run(cmd)

# %%
# clean up
for fl in test_file_dir.glob("*"):
    fl.unlink()
