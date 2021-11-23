#!/bin/bash
export MASTER_PORT=29501
export MASTER_ADDR=$(scontrol show hostname ${SLURM_NODELIST} | head -n 1)

echo "using master addr"$MASTER_ADDR
echo "using part "$SLURM_PARTITION
# TODO - minibatch size must be at least the no of local gpu?
echo "running python script"
echo "world size "$SLURM_NTASKS
echo "node name "$SLURMD_NODENAME
echo "rank? "$SLURM_PROCID


python -m torch.distributed.launch --nproc_per_node=1 --nnodes=2 --master_addr=$MASTER_ADDR --master_port=$MASTER_PORT --node_rank=$SLURM_PROCID training_script.py
