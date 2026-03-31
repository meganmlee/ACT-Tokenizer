#!/bin/bash
#SBATCH --job-name=act_quest_libero
#SBATCH --partition=GPU-shared
#SBATCH --gres=gpu:v100-32:1
#SBATCH --time=12:00:00
#SBATCH -A cis260038p
#SBATCH --output=logs/act_quest_%j.out
#SBATCH --error=logs/act_quest_%j.err

cd /ocean/projects/cis260038p/mlee12/act

module load anaconda3
conda activate /ocean/projects/cis260038p/mlee12/envs/aloha

export PYTHONPATH=/ocean/projects/cis260038p/mlee12/LIBERO:$PYTHONPATH

# Shared args — note: chunk_size=32 to match QueST's skill_block_size
POLICY_ARGS="--policy_class ACT --kl_weight 10 --chunk_size 32 --hidden_dim 512 --dim_feedforward 3200 --seed 0"
QUEST_ARGS="--use_fast_tokens --fast_tokenizer_path ./quest_tokenizer"

###############################################################################
# STEP 0: Train QueST tokenizer (SkillVAE) on LIBERO action data (run once)
#         Skip this if you already have ./quest_tokenizer/
###############################################################################

# python tokenizer.py \
#     --tokenizer_type quest \
#     --dataset_path /ocean/projects/cis260038p/shared/datasets/libero/libero_90 \
#     --save_path ./quest_tokenizer \
#     --chunk_size 32 \
#     --action_dim 7 \
#     --quest_epochs 100

###############################################################################
# Per-suite multi-task training + evaluation
# Train one policy per suite, evaluate on the same suite
# Suites: libero_spatial, libero_object, libero_goal, libero_10 (Long)
###############################################################################

SUITE=${1:-libero_spatial}

# STEP 1: Train on suite
python3 imitate_episodes.py \
    --task_name ${SUITE} \
    --ckpt_dir ./checkpoints/${SUITE}_act_quest \
    $POLICY_ARGS \
    $QUEST_ARGS \
    --batch_size 32 \
    --num_epochs 800 \
    --lr 1e-5

# STEP 2: Evaluate on suite
python3 imitate_episodes.py \
    --task_name ${SUITE} \
    --ckpt_dir ./checkpoints/${SUITE}_act_quest \
    $POLICY_ARGS \
    $QUEST_ARGS \
    --batch_size 32 \
    --num_epochs 800 \
    --lr 1e-5 \
    --eval
