If we need to install nvidia drivers, first install linux-headers
```
 sudo apt-get install linux-headers-$(uname -r)
```
Then install cuda and nvidia drivers for cuda 12.4 (version 12.4 is needed to match the pytorch version in dependencies)
```
wget https://developer.download.nvidia.com/compute/cuda/12.4.0/local_installers/cuda_12.4.0_550.54.14_linux.run
sudo sh cuda_12.4.0_550.54.14_linux.run
```

If we're on multi-GPU system, verify that fabric-manager is working:
```
nvidia-smi -q -i 0 | grep -i -A 2 Fabric
```
(see: https://docs.nvidia.com/datacenter/tesla/fabric-manager-user-guide/index.html#initializing-nvswitch-and-nvlink)

Make sure you install cmake, build-essential, python3-dev (needed for kenlm build from datatrove dependency)
```
sudo apt install cmake build-essential python3-dev
```

Install project with uv
```
uv sync --group nanosets --group test --group az --group s3 --group fast-modeling
```

Verify that torch is setup correctly
```
uv run python -c "import torch; print(torch.cuda.is_available())"
```

This next step is required to enable nemo_dataset compiler to find python-config
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

# choose the apprpriate folder for the machine hardware (i.e. replace "h100x8")
sudo cp configs/fabai/slurm/h100x8/slurm.conf /etc/slurm/slurm.conf
sudo cp configs/fabai/slurm/h100x8/gres.conf /etc/slurm/gres.conf

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
uv run torchrun --nproc_per_node=1 run_train.py --config-file configs/fabai/base-run-100-no-evals.yaml
```