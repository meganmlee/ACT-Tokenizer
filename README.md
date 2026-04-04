# ACT-Tokenizer: Action Chunking with Transformers + Discrete Tokenization + Language Conditioning

Fork of [ACT](https://tonyzhaozh.github.io/aloha/) with two extensions that can be used independently or together:

1. **FAST-ACT** -- Discrete action tokenization via FAST+. The CVAE encoder embeds discrete tokens instead of raw actions, and the decoder predicts token logits. After decoding, tokens are detokenized back to continuous actions.
2. **LAV-ACT** -- Language-conditioned ACT. A frozen CLIP text encoder extracts a task embedding (e.g. "pick up the ketchup"), which is projected and concatenated as an extra token in the transformer sequence. This lets a single model handle multiple tasks.

## Variants

| Variant | Tokenization | Language | Training | Flag(s) |
|---------|-------------|----------|----------|---------|
| ACT (baseline) | Continuous | No | Per-task | _(none)_ |
| FAST-ACT | FAST discrete | No | Per-task | `--use_fast_tokens` |
| LAV-ACT | FAST discrete | CLIP | Multi-task | `--use_fast_tokens --use_language` |

LAV-ACT can also be used with continuous actions (`--use_language` without `--use_fast_tokens`), but the intended configuration is FAST + language.

## Setup

```bash
conda create -n aloha python=3.8.10
conda activate aloha

# Core dependencies
pip install torch torchvision
pip install pyquaternion pyyaml rospkg pexpect
pip install mujoco==2.3.7 dm_control==1.0.14
pip install opencv-python matplotlib einops packaging h5py ipython
pip install transformers  # for FAST tokenizer + CLIP text encoder

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

## Evaluation Protocol

Per-suite training and evaluation on LIBERO, following the protocol used by OpenVLA-OFT, pi0, Dream-VLA, and MM-ACT:

- **ACT / FAST-ACT**: Train one policy per task (10 separate models per suite)
- **LAV-ACT**: Train one policy across all 10 tasks in a suite (single model, language-conditioned)
- Evaluate on the same 10 tasks with different initial conditions (50 rollouts per task)
- Report per-task success rates and suite average

| Suite | # Tasks | Episode Length | Description |
|---|---|---|---|
| LIBERO-Spatial | 10 | 300 | Spatial relationship reasoning |
| LIBERO-Object | 10 | 300 | Object recognition/manipulation |
| LIBERO-Goal | 10 | 300 | Goal-conditioned manipulation |
| LIBERO-Long | 10 | 600 | Long-horizon multi-step tasks |

## Running Experiments

### Step 0: Train FAST tokenizer (once, needed for FAST-ACT and LAV-ACT)

```bash
python tokenizer.py \
    --dataset_path /path/to/libero_90 \
    --save_path ./fast_tokenizer \
    --chunk_size 50 --action_dim 7
```

### ACT baseline (per-task, continuous actions)

```bash
sbatch job.sh libero_spatial
sbatch job.sh libero_object
sbatch job.sh libero_goal
sbatch job.sh libero_10
```

### FAST-ACT (per-task, discrete tokens)

```bash
sbatch job_actFAST.sh libero_spatial
sbatch job_actFAST.sh libero_object
sbatch job_actFAST.sh libero_goal
sbatch job_actFAST.sh libero_10
```

### LAV-ACT (multi-task, FAST tokens + CLIP language)

```bash
sbatch job_LACT.sh libero_spatial
sbatch job_LACT.sh libero_object
sbatch job_LACT.sh libero_goal
sbatch job_LACT.sh libero_10
```

### Evaluation

Eval is included in each job script as a commented-out block. Uncomment it to run eval after training completes, or submit it separately after the training job finishes.

Checkpoints and eval results are saved to:
- `checkpoints/{suite}_act/` -- ACT baseline (per-task subdirs)
- `checkpoints/{suite}_act_fast/` -- FAST-ACT (per-task subdirs)
- `checkpoints/{suite}_lact/` -- LAV-ACT (single model)

## Architecture

### ACT baseline
- **CVAE encoder**: `[CLS, qpos, action_0, ..., action_N]` -> latent z
- **Decoder memory**: `[latent, proprio, image_feat_0, ..., image_feat_HW]`
- Action queries cross-attend to memory -> continuous action predictions

### FAST-ACT
Same architecture, but actions are discrete BPE tokens (DCT + BPE encoding).
- Encoder embeds tokens via `nn.Embedding` instead of linear projection
- Decoder predicts token logits via cross-entropy instead of L1 loss

### LAV-ACT
Adds a frozen CLIP text token to both the encoder and decoder:
- **CVAE encoder**: `[CLS, text_token, qpos, action_tokens]`
- **Decoder memory**: `[latent, proprio, text_token, image_feats]`
- The text token is: `Linear(CLIP_text_encoder(task_string))` (CLIP is frozen, projection is learned)

## Swapping tokenizers

The tokenizer is pluggable via the `ActionTokenizer` base class in `tokenizer.py`. To add a new one, subclass it and decorate with `@register_tokenizer`. The rest of the codebase loads tokenizers via `load_tokenizer(path)` which auto-dispatches based on a saved type marker.

## Repo Structure
- `imitate_episodes.py` -- Train and evaluate ACT
- `policy.py` -- ACT policy wrapper
- `tokenizer.py` -- Action tokenizer interface + FAST+ implementation
- `detr/` -- Model definitions (DETRVAE), modified from DETR
- `constants.py` -- Task configs and constants
- `utils.py` -- Data loading (continuous + tokenized + language)
- `job.sh` -- SLURM job for ACT baseline (per-task, includes eval)
- `job_actFAST.sh` -- SLURM job for FAST-ACT (per-task, includes eval)
- `job_LACT.sh` -- SLURM job for LAV-ACT (multi-task, includes eval)
