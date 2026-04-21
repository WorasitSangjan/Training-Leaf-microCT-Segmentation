#!/bin/bash
#SBATCH -A als
#SBATCH -C gpu
#SBATCH -q regular
#SBATCH -t 48:00:00
#SBATCH -N 1
#SBATCH --ntasks-per-node=1
#SBATCH --gpus-per-node=4
#SBATCH -c 32
#SBATCH -J train1_unet_resnet101
#SBATCH -o /pscratch/sd/w/worasit/logs/train1_%j.out
#SBATCH -e /pscratch/sd/w/worasit/logs/train1_%j.err

mkdir -p /pscratch/sd/w/worasit/logs

module load python/3.9-24.1.0
export PYTHONPATH=/pscratch/sd/w/worasit/leafseg_venv/lib/python3.9/site-packages:$PYTHONPATH

echo "Job started: $(date)"
echo "Node: $(hostname)"
echo "GPUs: $SLURM_GPUS_ON_NODE"

torchrun --standalone --nproc_per_node=4 /pscratch/sd/w/worasit/train_1_unet_resnet101.py

echo "Job finished: $(date)"
