#!/bin/bash
#SBATCH -A als
#SBATCH -C gpu
#SBATCH -q regular
#SBATCH -t 24:00:00
#SBATCH -N 1
#SBATCH --ntasks-per-node=1
#SBATCH --gpus-per-node=4
#SBATCH -c 32
#SBATCH -o /pscratch/sd/w/worasit/logs/phase1_clean_%x_%j.out
#SBATCH -e /pscratch/sd/w/worasit/logs/phase1_clean_%x_%j.err

# Usage: sbatch -J phase1_<model> 1.1_SLURM_submit_training.sh <model> [extra train args]
#   e.g. sbatch -J phase1_eomt_vitl 1.1_SLURM_submit_training.sh eomt_vitl
#   e.g. sbatch -J phase1_eomt_vitl 1.1_SLURM_submit_training.sh eomt_vitl --class-weights 1 4.2 3.1 1.5 4.7
# Valid models: unet_resnet101 deeplab_efficientnet deeplab_mitb4
#               fpn_mitb4 fpn_mitb5 segformer mask2former eomt_vitb eomt_vitl

MODEL=$1
if [ -z "$MODEL" ]; then
  echo "ERROR: provide model name. e.g. sbatch 1.1_SLURM_submit_training.sh eomt_vitl"
  exit 1
fi
shift

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

mkdir -p /pscratch/sd/w/worasit/logs

module load python/3.9-24.1.0
export PYTHONPATH=/pscratch/sd/w/worasit/leafseg_venv/lib/python3.9/site-packages:$PYTHONPATH
echo "Job started: $(date)"
echo "Node: $(hostname)  Model: $MODEL"

torchrun --standalone --nproc_per_node=4 "$SCRIPT_DIR/1_train_multi_architecture.py" --model "$MODEL" "$@"

echo "Job finished: $(date)"
