# %%
from pathlib import Path

from dotenv import load_dotenv

from nanotron.config import Config
from nanotron.eval.one_job_runner import LightEvalRunner

ROOT = Path(__file__).resolve().parents[3]

load_dotenv(override=True)

# %%
nanotron_config_file = ROOT / "configs" / "fabai" / "base-run-100-no-evals.yaml"
nanotron_config = Config.load_from_yaml(str(nanotron_config_file))

lighteval_config_file = ROOT / "configs" / "fabai" / "lighteval" / "lighteval-config.yaml"
nanotron_config = Config.load_from_yaml(str(nanotron_config_file))

# %%
checkpoints_dir = (
    ROOT
    / "checkpoints"
    / "smol-playbook-checkpoints"
    / "qurater_gemma-3-4b-pt_ds-ours_v2-200000"
)

all_ckpt = checkpoints_dir.rglob("config.yaml")
steps = sorted(int(f.parent.name) for f in all_ckpt)
for step in steps:
    nanotron_config.general.step = step
    le_runner = LightEvalRunner(
        config=nanotron_config, parallel_context=nanotron_config.parallelism
    )
    ckpt_file = str(checkpoints_dir / f"{step}" / "config.yaml")
    print(ckpt_file)
    runner_input = [{"destination": ckpt_file}]
    le_runner.eval_single_checkpoint(runner_input)
