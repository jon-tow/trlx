#!/bin/bash
#SBATCH --job-name="trlx-ds-scaling"
#SBATCH --partition=a100-cu117
#SBATCH --nodes=2
#SBATCH --gres=gpu:8
#SBATCH --ntasks-per-node=8
#SBATCH --cpus-per-task=6
#SBATCH --hint=nomultithread 
#SBATCH --output=%x_%j.out
#SBATCH --exclusive

source /opt/hpcx/hpcx-init.sh
hpcx_load

################################################################################
# CUDA/Torch Setup
################################################################################
export NCCL_DEBUG=info
# export NCCL_IB_GID_INDEX=3 
# export NCCL_TREE_THRESHOLD=0
# export NCCL_IB_DISABLE=1
# export NCCL_IBEXT_DISABLE=1
# export NCCL_SOCKET_IFNAME="eth0"
# export CUDA_LAUNCH_BLOCKING=1
export TORCH_CPP_LOG_LEVEL=INFO
export TORCH_DISTRIBUTED_DEBUG=INFO
################################################################################


################################################################################
# MPI Setup
################################################################################
export OMPI_MCA_mtl_base_verbose=1
export OMPI_MCA_btl_base_verbose=30
# export OMPI_MCA_btl="^openib"
# export OMPI_MCA_plm_rsh_no_tree_spawn=1
# export OMPI_MCA_pml="ob1"
################################################################################


################################################################################
# Environment Setup
# TODO: Replace with your own environment setup
################################################################################
source $HOME/.bashrc
eval "$(micromamba shell hook --shell=bash)"
micromamba activate trlx
################################################################################


################################################################################
# Network Setup
################################################################################
export HOSTNAMES=$(scontrol show hostnames "$SLURM_JOB_NODELIST")
export MASTER_ADDR=$(scontrol show hostnames "$SLURM_JOB_NODELIST" | head -n 1)
export MASTER_PORT=12139
export COUNT_NODE=$(scontrol show hostnames "$SLURM_JOB_NODELIST" | wc -l)

echo "Master Addr: $MASTER_ADDR"
echo "Node Count: $COUNT_NODE"
echo "Host Names: $HOSTNAMES"

# Write the hostfile for this job
$HOME/write_hostfile.sh
export DLTS_HOSTFILE=$HOME/hostfiles/hosts_$SLURM_JOBID
echo "Hostfile: $DLTS_HOSTFILE"
HOSTNAME=`hostname`
MACHINE_RANK=`echo -e $HOSTNAMES  | python3 -c "import sys;[sys.stdout.write(str(i)) for i,line in enumerate(next(sys.stdin).split(' ')) if line.strip() == '$HOSTNAME'.strip()]"`
echo "MACHINE_RANK=$MACHINE_RANK"
################################################################################


################################################################################
# Launch
# NOTE: Make sure to `chmod 777 DeepSpeed/bin/deepspeed`
################################################################################
SCRIPT=./examples/multinode/benchmark.py
TRLX_CONFIG=./examples/multinode/configs/ppo_deepspeed.yml
ACCEL_CONFIG=./examples/multinode/configs/accelerate/deepspeed_multinode.yml
cat $ACCEL_CONFIG
# TODO: Add `--no_ssh_check` to avoid passwordless-ssh check in `DeepSpeed/launcher/runner.py`
# and avoid commenting out L400-413
rm .deepspeed_env  # `.deepspeed_env` is append-mode written to - remove it after each call. 
# export CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7  # NOTE: Needed to force CUDA_VISIBLE_DEVICES to be set
accelerate launch \
    --config_file $ACCEL_CONFIG \
    --use_deepspeed \
    --deepspeed_hostfile $DLTS_HOSTFILE \
    --machine_rank $MACHINE_RANK \
    --main_process_ip $MASTER_ADDR \
    --main_process_port $MASTER_PORT \
    $SCRIPT --config $TRLX_CONFIG
