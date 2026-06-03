#!/bin/bash
# Evaluate Phase 1 clean models on test sets.
#
# Usage:
#   bash 2.1_run_standard_tests.sh <model> [test_set]
#     test_set: broadleaf | pineL | pineV | wheat | all   (default: all)
#
# Examples:
#   bash 2.1_run_standard_tests.sh deeplab_efficientnet broadleaf  # one test set
#   bash 2.1_run_standard_tests.sh deeplab_efficientnet            # all 4 test sets
#   bash 2.1_run_standard_tests.sh all                             # all 9 models, all 4 test sets
#
# Run on a compute node with venv active:
#   source /pscratch/sd/w/worasit/leafseg_venv/bin/activate

SCRATCH=/pscratch/sd/w/worasit
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
EVAL="$SCRIPT_DIR/2_evaluate.py"
CFG=$SCRATCH/configs

# Map phase1 model -> 2_evaluate.py --model name
declare -A EVAL_MODEL=(
  [unet_resnet101]=unet_resnet101
  [deeplab_efficientnet]=deeplab_efficientnet
  [deeplab_mitb4]=deeplab_mitb4
  [fpn_mitb4]=fpn_mitb4
  [fpn_mitb5]=fpn_mitb5
  [segformer]=segformer_v2
  [mask2former]=mask2former
  [eomt_vitb]=eomt_vitb
  [eomt_vitl]=eomt_vitl
)

run_test_set () {
  local M=$1; local SET=$2
  local EM=${EVAL_MODEL[$M]}
  local CKPT=$SCRATCH/outputs/phase1_clean_${M}/best_model.pth
  local PRED=$SCRATCH/predictions/phase1_clean_${M}

  if [ ! -f "$CKPT" ]; then
    echo "SKIP $M/$SET — checkpoint not found ($CKPT)"
    return
  fi

  case $SET in
    broadleaf) CFGS="$CFG/test_v_jucundum3.json $CFG/test_lantana_camara.json $CFG/test_olive_rw2.json" ;;
    pineL)     CFGS="$CFG/test_pine_pinus_araucaria_angustifolia.json" ;;
    pineV)     CFGS="$CFG/test_pine_pinus_densifora.json" ;;
    wheat)     CFGS="$CFG/test_wheat.json" ;;
    *) echo "Unknown test set: $SET (use broadleaf|pineL|pineV|wheat)"; return ;;
  esac

  echo "================================================================"
  echo "Model: $M  (eval=$EM)   Test set: $SET"
  echo "================================================================"
  python $EVAL --model $EM --checkpoint $CKPT --configs $CFGS \
      --save_predictions $PRED/$SET --per_image --tag $SET
}

eval_one_model () {
  local M=$1
  for SET in broadleaf pineL pineV wheat; do
    run_test_set $M $SET
  done
}

MODEL=$1
TEST_SET=${2:-all}

if [ "$MODEL" == "all" ]; then
  # All 9 models, all 4 test sets
  for M in unet_resnet101 deeplab_efficientnet deeplab_mitb4 fpn_mitb4 fpn_mitb5 segformer mask2former eomt_vitb eomt_vitl; do
    eval_one_model $M
  done
elif [ -n "$MODEL" ]; then
  if [ "$TEST_SET" == "all" ]; then
    eval_one_model $MODEL
  else
    run_test_set $MODEL $TEST_SET
  fi
else
  echo "Usage: bash 2.1_run_standard_tests.sh <model|all> [test_set]"
  echo "  test_set: broadleaf | pineL | pineV | wheat | all (default: all)"
  exit 1
fi

echo ""
echo "DONE. Predictions: $SCRATCH/predictions/phase1_clean_<model>/<testset>/"
