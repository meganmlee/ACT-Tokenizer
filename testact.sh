#!/bin/bash
#SBATCH --job-name=act_sim
#SBATCH --partition=GPU-shared
#SBATCH --gres=gpu:v100-32:1
#SBATCH --time=10:00:00
#SBATCH -A cis260038p
#SBATCH --output=logs/act_sim_%j.out
#SBATCH --error=logs/act_sim_%j.err

cd /ocean/projects/cis260038p/mlee12/ACT-Tokenizer

module load anaconda3
conda activate /ocean/projects/cis260038p/mlee12/envs/aloha

export PYTHONPATH=/ocean/projects/cis260038p/mlee12/ACT-Tokenizer/detr:$PYTHONPATH
export MUJOCO_GL=egl

POLICY_ARGS="--policy_class ACT --kl_weight 10 --chunk_size 100 --hidden_dim 512 --dim_feedforward 3200 --seed 0 --temporal_agg"

###############################################################################
# ACT sanity check on ALOHA sim data (no LIBERO required)
# Trains on sim_transfer_cube_scripted or sim_insertion_scripted.
#
# Usage: sbatch job_sim.sh [task_name]
# Tasks: sim_transfer_cube_scripted, sim_transfer_cube_human,
#        sim_insertion_scripted, sim_insertion_human
###############################################################################

TASK=${1:-sim_transfer_cube_scripted}

# echo "========================================"
# echo "ACT sim training: ${TASK}"
# echo "========================================"

# python3 imitate_episodes.py \
#     --task_name ${TASK} \
#     --ckpt_dir ./checkpoints/${TASK}_act \
#     $POLICY_ARGS \
#     --batch_size 8 \
#     --num_epochs 2000 \
#     --lr 1e-5

###############################################################################
# Evaluation — uncomment below to run after training
###############################################################################

echo "========================================"
echo "Evaluating ${TASK}"
echo "========================================"
python3 imitate_episodes.py \
    --task_name ${TASK} \
    --ckpt_dir ./checkpoints/${TASK}_act \
    $POLICY_ARGS \
    --batch_size 8 \
    --num_epochs 2000 \
    --lr 1e-5 \
    --eval
