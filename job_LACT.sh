#!/bin/bash
#SBATCH --job-name=lact_libero
#SBATCH --partition=GPU-shared
#SBATCH --gres=gpu:v100-32:1
#SBATCH --time=20:00:00
#SBATCH -A cis260038p
#SBATCH --output=logs/lact_%j.out
#SBATCH --error=logs/lact_%j.err

cd /ocean/projects/cis260038p/mlee12/ACT-Tokenizer

module load anaconda3
conda activate /ocean/projects/cis260038p/mlee12/envs/aloha

export PYTHONPATH=/ocean/projects/cis260038p/mlee12/ACT-Tokenizer/detr:/ocean/projects/cis260038p/mlee12/LIBERO:$PYTHONPATH

# Shared args
POLICY_ARGS="--policy_class ACT --kl_weight 0.01 --chunk_size 50 --hidden_dim 512 --dim_feedforward 3200 --seed 0 --temporal_agg"
FAST_ARGS="--use_fast_tokens --fast_tokenizer_path ./fast_tokenizer"
LANG_ARGS="--use_language"

###############################################################################
# LAV-ACT: Language-conditioned ACT with FAST tokenization
# Trains a SINGLE model across all 10 tasks in a suite (no job array).
# The frozen CLIP text embedding tells the model which task to perform.
#
# Usage: sbatch job_LACT.sh libero_spatial
# Suites: libero_spatial, libero_object, libero_goal, libero_10
###############################################################################

SUITE=${1:-libero_spatial}

echo "========================================"
echo "LAV-ACT multi-task training: ${SUITE} (all tasks, language-conditioned)"
echo "========================================"

python3 imitate_episodes.py \
    --task_name ${SUITE} \
    --ckpt_dir ./checkpoints/${SUITE}_lact \
    $POLICY_ARGS \
    $FAST_ARGS \
    $LANG_ARGS \
    --batch_size 32 \
    --num_epochs 2000 \
    --lr 5e-4

###############################################################################
# Evaluation — uncomment below to run after training
###############################################################################

# echo "========================================"
# echo "Evaluating all tasks for ${SUITE} (LAV-ACT)"
# echo "========================================"
# python3 imitate_episodes.py \
#     --task_name ${SUITE} \
#     --ckpt_dir ./checkpoints/${SUITE}_lact \
#     $POLICY_ARGS \
#     $FAST_ARGS \
#     $LANG_ARGS \
#     --batch_size 32 \
#     --num_epochs 2000 \
#     --lr 5e-4 \
#     --eval
