#!/bin/bash
#SBATCH --job-name="trlx-ddp-scaling"
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
# export CUDA_LAUNCH_BLOCKING=0
export TORCH_CPP_LOG_LEVEL=INFO
export TORCH_DISTRIBUTED_DEBUG=INFO
################################################################################


################################################################################
# MPI Setup
################################################################################
export OMPI_MCA_mtl_base_verbose=1
export OMPI_MCA_btl_base_verbose=30
# export OMPI_MCA_btl="^openib"
# export OMPI_MCA_btl_tcp_if_exclude="lo,docker1"
# export OMPI_MCA_plm_rsh_no_tree_spawn=1
# export OMPI_MCA_pml="ob1"
################################################################################


################################################################################
# Network Setup
################################################################################
export HOSTNAMES=$(scontrol show hostnames "$SLURM_JOB_NODELIST")
export MASTER_ADDR=$(scontrol show hostnames "$SLURM_JOB_NODELIST" | head -n 1)
export MASTER_PORT=12134
export COUNT_NODE=$(scontrol show hostnames "$SLURM_JOB_NODELIST" | wc -l)

echo "Master Addr: $MASTER_ADDR"
echo "Node Count: $COUNT_NODE"
echo "Host Names: $HOSTNAMES"
################################################################################


###############################################################################
# Program Setup
###############################################################################
SCRIPT=examples/multinode/slurm_ddp_start.sh
mpirun --bind-to none -n $COUNT_NODE --map-by ppr:1:node $SCRIPT