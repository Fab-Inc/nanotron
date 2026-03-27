Make sure you install cmake and build-essential first in OS
```
sudo apt install cmake build-essential
```

This next step is required to enable neme_dataset compiler to find python-config
```
source env.src
```

For running evals, need slurm (make sure you add user to group munge first)
```
sudo apt install slurmd slurmctld slurm-client munge

sudo /usr/sbin/mungekey -f
sudo chown munge:munge /etc/munge/munge.key

systemctl enable munge
systemctl restart munge

munge -n | unmunge | grep STATUS

# choose the apprpriate folder for the machine hardware (i.e. replace "h200x1")
sudo cp configs/fabai/slurm/h200x1/slurm.conf /etc/slurm/slurm.conf
sudo cp configs/fabai/slurm/h200x1/gres.conf /etc/slurm/gres.conf

sudo mkdir -p /var/lib/slurm-llnl/slurmd
sudo mkdir -p /var/lib/slurm-llnl/slurmctld
sudo chown -R slurm:slurm /var/lib/slurm-llnl/

sudo systemctl enable slurmctld slurmd
sudo systemctl restart slurmctld slurmd

# useful commands
scontrol show node localhost
squeue
# kill all jobs
scancel -u <username>

# manually launch slurmctld/slurmd with verbosity
sudo slurmctld -Dvvvv
sudo slurmd -Dvvvv

# resume node
scontrol update nodename=lhtraineval state=resume

sudo scontrol delete ReservationName=smollm
sudo scontrol create reservation ReservationName=smollm StartTime=now Duration=infinite Nodes=all Users=ubuntu
scontrol show reservation

```



```
uv sync --group nanosets --group test --group az --group s3 --group fast-modeling
uv run torchrun --nproc_per_node=1 run_train.py --config-file configs/fabai/base-run-100-no-evals.yaml
```