#!/usr/bin/env bash
#sleep 30
#fi_info -p efa -t FI_EP_RDM

################################################################################
# Echo Environment
echo myuser=`whoami`
echo COUNT_NODE=$COUNT_NODE
echo LD_LIBRARY_PATH = $LD_LIBRARY_PATH
echo PATH = $PATH
echo which mpicc `which mpicc`
echo HOSTNAMES = $HOSTNAMES
echo hostname = `hostname`
echo MASTER_ADDR= $MASTER_ADDR
echo MASTER_PORT= $MASTER_PORT
################################################################################


################################################################################
# Set machine rank
################################################################################
H=`hostname`
MACHINE_RANK=`echo -e $HOSTNAMES  | python3 -c "import sys;[sys.stdout.write(str(i)) for i,line in enumerate(next(sys.stdin).split(' ')) if line.strip() == '$H'.strip()]"`
echo "MACHINE_RANK=$MACHINE_RANK"
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
# Launch
################################################################################
SCRIPT=./examples/multinode/benchmark.py
TRLX_CONFIG=./examples/multinode/configs/ppo_ddp.yml
ACCEL_CONFIG=./examples/multinode/configs/accelerate/ddp_multinode.yml
CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7  # NOTE: Needed to force CUDA_VISIBLE_DEVICES to be set
accelerate launch \
    --config_file $ACCEL_CONFIG \
    --multi_gpu \
    --num_processes $(( 8 * $COUNT_NODE )) \
    --num_machines $COUNT_NODE \
    --machine_rank $MACHINE_RANK \
    --main_process_ip $MASTER_ADDR \
    --main_process_port $MASTER_PORT \
    $SCRIPT --config $TRLX_CONFIG
