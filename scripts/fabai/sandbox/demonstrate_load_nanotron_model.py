# %%
import os
from pathlib import Path

from lighteval.config.lighteval_config import FullNanotronConfig
from lighteval.models.nanotron_model import NanotronLightevalModel
from lighteval.utils.parallelism import test_all_gather
from lighteval.utils.utils import EnvConfig

from nanotron import distributed as dist
from nanotron.config.config import Config
from nanotron.parallel.context import ParallelContext

ROOT = Path(__file__).resolve().parents[3]

# %%
checkpoint_path=ROOT / "checkpoints/smol-playbook-checkpoints/qurater_gemma-3-4b-pt_ds-ours_v2-200000/1000"
config = Config.load_from_yaml(str(checkpoint_path / "config.yaml"))
lighteval_config = config.lighteval
nanotron_config = FullNanotronConfig(lighteval_config=lighteval_config, nanotron_config=config)
parallel_args = nanotron_config.nanotron_config.parallelism

os.environ["WORLD_SIZE"] = "1"
os.environ["MASTER_ADDR"] = "localhost"
os.environ["MASTER_PORT"] = "6000"
os.environ["RANK"] = "0"
os.environ["LOCAL_RANK"] = "0"
dist.initialize_torch_distributed()
parallel_context = ParallelContext(
    tensor_parallel_size=parallel_args.tp,
    pipeline_parallel_size=parallel_args.pp,
    data_parallel_size=parallel_args.dp,
)
test_all_gather(parallel_context=parallel_context)

model = NanotronLightevalModel(
    checkpoint_path = str(checkpoint_path),
    nanotron_config=nanotron_config,
    parallel_context=parallel_context,
    debug_one_layer_model=False,
    model_class=None,
    env_config=EnvConfig(cache_dir=str(Path.home() / ".cache" / "huggingface"))
)
