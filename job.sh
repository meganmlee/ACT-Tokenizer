#!/bin/bash
#SBATCH --job-name=act_libero
#SBATCH --partition=GPU-shared
#SBATCH --gres=gpu:v100-32:1
#SBATCH --time=8:00:00
#SBATCH -A cis260038p
#SBATCH --output=logs/act_libero_%j.out
#SBATCH --error=logs/act_libero_%j.err

cd /ocean/projects/cis260038p/mlee12/act

module load anaconda3
conda activate /ocean/projects/cis260038p/mlee12/envs/aloha

export PYTHONPATH=/ocean/projects/cis260038p/mlee12/LIBERO:$PYTHONPATH

# Shared args
POLICY_ARGS="--policy_class ACT --kl_weight 10 --chunk_size 50 --hidden_dim 512 --dim_feedforward 3200 --seed 0"

###############################################################################
# STEP 1: Pretrain on LIBERO-90  (~13-14 hours)
#         Skip this if you already have checkpoints/libero_90_act/policy_best.ckpt
# CHANGE the sbatch time only for this part, the longer time you request, the longer it takes to get a node assigned
###############################################################################

# python3 imitate_episodes.py \
#     --task_name libero_90 \
#     --ckpt_dir ./checkpoints/libero_90_act \
#     $POLICY_ARGS \
#     --batch_size 32 \
#     --num_epochs 800 \
#     --lr 1e-5

###############################################################################
# STEP 2: Finetune on LIBERO-10  (~1-2 hours)
#         Loads the pretrained LIBERO-90 checkpoint, then trains on LIBERO-10
#         Uses fewer epochs and same or lower learning rate
###############################################################################

# python3 imitate_episodes.py \
#     --task_name libero_10 \
#     --ckpt_dir ./checkpoints/libero_10_finetuned \
#     $POLICY_ARGS \
#     --batch_size 32 \
#     --num_epochs 500 \
#     --lr 1e-5 \
#     --resume ./checkpoints/libero_90_act/policy_best.ckpt

###############################################################################
# STEP 3: Evaluate on LIBERO-10  (~2-4 hours)
#         Runs the finetuned policy in the simulator across all 10 tasks
#         Reports per-task and average success rates
###############################################################################

python3 imitate_episodes.py \
    --task_name libero_10 \
    --ckpt_dir ./checkpoints/libero_10_finetuned \
    $POLICY_ARGS \
    --batch_size 32 \
    --num_epochs 500 \
    --lr 1e-5 \
    --eval