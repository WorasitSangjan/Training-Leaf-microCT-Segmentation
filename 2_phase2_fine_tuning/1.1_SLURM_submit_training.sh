#!/bin/bash
#SBATCH -A als
#SBATCH -C gpu
#SBATCH -q regular
#SBATCH -t 48:00:00
#SBATCH -N 1
#SBATCH --ntasks-per-node=1
#SBATCH --gpus-per-node=4
#SBATCH -c 32
#SBATCH -o /pscratch/sd/w/worasit/logs/phase2_%x_%j.out
#SBATCH -e /pscratch/sd/w/worasit/logs/phase2_%x_%j.err

# Usage: sbatch -J phase2_<model> 1.1_SLURM_submit_training.sh <model>
#   e.g. sbatch -J phase2_mask2former 1.1_SLURM_submit_training.sh mask2former
# Valid models: mask2former segformer fpn_mitb4 eomt_vitl

MODEL=$1
if [ -z "$MODEL" ]; then
  echo "ERROR: provide model name. e.g. sbatch 1.1_SLURM_submit_training.sh mask2former"
  exit 1
fi

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

case "$MODEL" in
  mask2former)
    TRAIN_SCRIPT="$SCRIPT_DIR/1_train_mask2former.py"
    ;;
  segformer)
    TRAIN_SCRIPT="$SCRIPT_DIR/1_train_segformer.py"
    ;;
  fpn_mitb4)
    TRAIN_SCRIPT="$SCRIPT_DIR/1_train_fpn_mitb4.py"
    ;;
  eomt_vitl)
    TRAIN_SCRIPT="$SCRIPT_DIR/1_train_eomt_vitl.py"
    ;;
  *)
    echo "ERROR: unknown model '$MODEL'"
    echo "Valid models: mask2former segformer fpn_mitb4 eomt_vitl"
    exit 1
    ;;
esac

mkdir -p /pscratch/sd/w/worasit/logs

module load python/3.9-24.1.0
export PYTHONPATH=/pscratch/sd/w/worasit/leafseg_venv/lib/python3.9/site-packages:$PYTHONPATH
echo "Job started: $(date)"
echo "Node: $(hostname)  Model: $MODEL"
echo "Training script: $TRAIN_SCRIPT"

torchrun --standalone --nproc_per_node=4 "$TRAIN_SCRIPT"

echo "Job finished: $(date)"
