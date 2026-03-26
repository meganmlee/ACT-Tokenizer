# ACT-Tokenizer: Action Chunking with Transformers + Discrete Action Tokenization

Fork of [ACT](https://tonyzhaozh.github.io/aloha/) that adds discrete action tokenization (FAST+) to the CVAE. The CVAE encoder embeds discrete tokens instead of raw actions, and the decoder predicts token logits. After decoding, tokens are detokenized back to continuous actions.

## Setup

```bash
conda create -n aloha python=3.8.10
conda activate aloha

# Core dependencies
pip install torch torchvision
pip install pyquaternion pyyaml rospkg pexpect
pip install mujoco==2.3.7 dm_control==1.0.14
pip install opencv-python matplotlib einops packaging h5py ipython
pip install transformers  # for FAST tokenizer

# Install DETR
cd detr && pip install -e . && cd ..

# Install LIBERO (needed for LIBERO tasks)
cd ..
git clone https://github.com/Lifelong-Robot-Learning/LIBERO.git
cd LIBERO
pip install -e .
```

When running, make sure LIBERO is on your path:
```bash
export PYTHONPATH=/path/to/LIBERO:$PYTHONPATH
```

## Running ACT (original, continuous actions)

Use `job.sh`. The three steps:

```bash
# Step 1: Pretrain on LIBERO-90
python3 imitate_episodes.py \
    --task_name libero_90 \
    --ckpt_dir ./checkpoints/libero_90_act \
    --policy_class ACT --kl_weight 10 --chunk_size 50 --hidden_dim 512 --dim_feedforward 3200 \
    --batch_size 32 --num_epochs 800 --lr 1e-5 --seed 0

# Step 2: Finetune on LIBERO-10
python3 imitate_episodes.py \
    --task_name libero_10 \
    --ckpt_dir ./checkpoints/libero_10_finetuned \
    --policy_class ACT --kl_weight 10 --chunk_size 50 --hidden_dim 512 --dim_feedforward 3200 \
    --batch_size 32 --num_epochs 500 --lr 1e-5 --seed 0 \
    --resume ./checkpoints/libero_90_act/policy_best.ckpt

# Step 3: Evaluate on LIBERO-10
python3 imitate_episodes.py \
    --task_name libero_10 \
    --ckpt_dir ./checkpoints/libero_10_finetuned \
    --policy_class ACT --kl_weight 10 --chunk_size 50 --hidden_dim 512 --dim_feedforward 3200 \
    --batch_size 32 --num_epochs 500 --lr 1e-5 --seed 0 --eval
```

## Running ACT + FAST tokenizer

Use `job_actFAST.sh`. Same structure but with an extra tokenizer training step:

```bash
# Step 0: Train FAST tokenizer on LIBERO-90 action data
python tokenizer.py \
    --dataset_path /path/to/libero_90 \
    --save_path ./fast_tokenizer \
    --chunk_size 50 --action_dim 7

# Step 1: Pretrain on LIBERO-90 with FAST tokens
python3 imitate_episodes.py \
    --task_name libero_90 \
    --ckpt_dir ./checkpoints/libero_90_act_fast \
    --policy_class ACT --kl_weight 10 --chunk_size 50 --hidden_dim 512 --dim_feedforward 3200 \
    --batch_size 32 --num_epochs 800 --lr 1e-5 --seed 0 \
    --use_fast_tokens --fast_tokenizer_path ./fast_tokenizer

# Step 2: Finetune on LIBERO-10 with FAST tokens
python3 imitate_episodes.py \
    --task_name libero_10 \
    --ckpt_dir ./checkpoints/libero_10_fast_finetuned \
    --policy_class ACT --kl_weight 10 --chunk_size 50 --hidden_dim 512 --dim_feedforward 3200 \
    --batch_size 32 --num_epochs 500 --lr 1e-5 --seed 0 \
    --use_fast_tokens --fast_tokenizer_path ./fast_tokenizer \
    --resume ./checkpoints/libero_90_act_fast/policy_best.ckpt

# Step 3: Evaluate on LIBERO-10
python3 imitate_episodes.py \
    --task_name libero_10 \
    --ckpt_dir ./checkpoints/libero_10_fast_finetuned \
    --policy_class ACT --kl_weight 10 --chunk_size 50 --hidden_dim 512 --dim_feedforward 3200 \
    --batch_size 32 --num_epochs 500 --lr 1e-5 --seed 0 \
    --use_fast_tokens --fast_tokenizer_path ./fast_tokenizer --eval
```

## Swapping tokenizers

The tokenizer is pluggable via the `ActionTokenizer` base class in `tokenizer.py`. To add a new one, subclass it and decorate with `@register_tokenizer`. The rest of the codebase loads tokenizers via `load_tokenizer(path)` which auto-dispatches based on a saved type marker.

## Repo Structure
- `imitate_episodes.py` — Train and evaluate ACT
- `policy.py` — ACT policy wrapper
- `tokenizer.py` — Action tokenizer interface + FAST+ implementation
- `detr/` — Model definitions (DETRVAE), modified from DETR
- `constants.py` — Task configs and constants
- `utils.py` — Data loading (continuous + tokenized)
- `job.sh` — SLURM job for original ACT
- `job_actFAST.sh` — SLURM job for ACT + FAST tokens
