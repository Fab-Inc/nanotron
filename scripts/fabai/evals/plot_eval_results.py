# %%
import json
from collections import defaultdict
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns

ROOT = Path(__file__).resolve().parents[3]

# %%
results_dir = ROOT / "eval_results" / "results"
run_name = "qr-filtered-dclm-run-30000"
step_size = 1000

results_files = results_dir.rglob("*.json")

files = sorted(
    (rf for rf in results_files if run_name in rf.parent.name), key=lambda x: x.name
)

res_list_dict = defaultdict(list)
for step_file in files:
    # step_file_dir = results_dir / f"{step}"
    # step_file = sorted(step_file_dir.glob("*.json"))
    # assert len(step_file) >= 1
    # step_file = step_file[-1]
    with open(step_file, "r") as f:
        step_res = json.load(f)
    for eval, res in step_res["results"].items():
        if eval.startswith(("custom|arc_cf:", "custom|mmlu_cf:")) and "_average" not in eval:
            continue
        res_list_dict[eval].append(res)

res_df_list = []
for eval, res_list in res_list_dict.items():
    df = pd.DataFrame(
        res_list, index=list(range(step_size, (len(files) + 1) * step_size, step_size))
    )
    if "acc_norm" in df.columns:
        df["acc"] = df["acc_norm"]
        df["acc_stderr"] = df["acc_norm_stderr"]
    df.columns = pd.MultiIndex.from_product([[eval], df.columns])
    res_df_list.append(df)
res_df = pd.concat(res_df_list, axis=1)

# %%
fig, ax = plt.subplots(figsize=(24, 16))

sns.lineplot(
    res_df.loc[:, (slice(None), "acc")].droplevel(1, axis=1), ax=ax#, legend=False
)
ax.grid(visible=True, which="both")

# %%
fig, ax = plt.subplots(figsize=(24, 16))

sns.lineplot(
    res_df.loc[:, (slice(None), "acc")].droplevel(1, axis=1).mean(axis=1),
    ax=ax,
    legend=False,
)

# fig.savefig(ROOT / "eval_results" / "test_res.png", dpi=300)
