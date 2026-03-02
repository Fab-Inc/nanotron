Make sure you install cmake and build-essential first in OS
```
sudo apt install cmake build-essential
```

```
uv sync --group nanosets --group test --group az --group fast-modeling
uv run torchrun --nproc_per_node=1 run_train.py --config-file scripts/fabai/azure_test_config.yaml
```