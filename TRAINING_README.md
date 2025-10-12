# SPARK Training & Inference Guide

Complete guide for training and evaluating the SPARK (Slot Programs via Active Radiation for ARC) system.

## Table of Contents

1. [Overview](#overview)
2. [Phase 1: Pretrain Autoencoder](#phase-1-pretrain-autoencoder)
3. [Phase 2: Train Controller](#phase-2-train-controller)
4. [Phase 3: Inference & Evaluation](#phase-3-inference--evaluation)
5. [Model Architecture](#model-architecture)
6. [Hyperparameter Guide](#hyperparameter-guide)
7. [Tips & Best Practices](#tips--best-practices)
8. [Troubleshooting](#troubleshooting)

---

## Overview

### Training Pipeline

SPARK uses a **three-phase training approach**:

1. **Phase 1: Autoencoder Pretraining** (1-2 days)
   - Train SlotAttention encoder + renderer to reconstruct ARC grids
   - Goal: >95% reconstruction accuracy for stable slot representations
   - Frozen during Phase 2

2. **Phase 2: Controller Meta-Learning** (3-5 days)
   - Meta-learn controller + operators using REINFORCE + Reptile
   - Inner loop: Task-specific adaptation (10-20 gradient steps)
   - Outer loop: Meta-parameter updates across tasks
   - Goal: 10-30% task success rate on validation

3. **Phase 3: Inference & Evaluation**
   - Passive inference: Standard beam search
   - Active learning: Test-time adaptation per task
   - Test-time augmentation: Ensemble across spatial/color transforms

### Hardware Requirements

- **GPU**: 8GB+ VRAM (16GB+ recommended for large batches)
- **CPU**: 8+ cores for data loading
- **RAM**: 16GB+ system memory
- **Storage**: 5GB for checkpoints and logs

### Dataset Setup

Place ARC-AGI data files in `data/`:
```bash
data/
├── arc-agi_training_challenges.json      # 400 training tasks
├── arc-agi_training_solutions.json
├── arc-agi_evaluation_challenges.json    # 400 evaluation tasks
├── arc-agi_evaluation_solutions.json
└── arc-agi_test_challenges.json          # 100 test tasks (no solutions)
```

---

## Phase 1: Pretrain Autoencoder

### Goal

Train the SlotAttention autoencoder to reconstruct ARC grids with >95% accuracy. This provides:
- **Stable slot representations** (z, m, p) for downstream controller training
- **Object-centric encoding** that decomposes grids into semantic objects
- **Reconstruction capability** for rendering modified slots back to grids

### Architecture

- **Encoder**: Palette embedding → CNN features → SlotAttention → (z, m, p)
  - `z`: [B, K, 128] slot features
  - `m`: [B, K, H, W] soft masks
  - `p`: [B, K, 2] centroids

- **Renderer**: Slots → SlotDecoder (spatial broadcast) → grid logits → [B, H, W, 11]

### Basic Training

**Minimal example (quick test):**
```bash
python3 arc_nodsl/training/pretrain_autoencoder.py \
    --data_train data/arc-agi_training_challenges.json \
    --data_val data/arc-agi_evaluation_challenges.json \
    --epochs 10 \
    --batch_size 32 \
    --lr 3e-4 \
    --eval_every 2 \
    --save_every 5
```

**Expected output:**
```
Train Loss: 0.8234, Accuracy: 92.15%
Val Loss: 0.6547, Accuracy: 94.32%
✓ Saved best model (acc=94.32%)
```

### Data Augmentation Strategies

SPARK supports **three augmentation modes** to increase training data diversity:

#### Strategy 1: Spatial Augmentation (DataLoader-level)

Applies random spatial transforms (rotations, flips) to each pair.

```bash
python3 arc_nodsl/training/pretrain_autoencoder.py \
    --epochs 50 \
    --batch_size 32 \
    --augment \
    --lr 3e-4
```

**Transforms applied:** 8 spatial transforms (identity, 3 rotations, 2 flips, 2 diagonal flips)

#### Strategy 2: Color Augmentation Probability (DataLoader-level)

Applies random color permutations with probability P to each pair.

```bash
python3 arc_nodsl/training/pretrain_autoencoder.py \
    --epochs 50 \
    --batch_size 32 \
    --augment \
    --color_augment_prob 0.5 \
    --lr 3e-4
```

**Parameters:**
- `--color_augment_prob`: Probability of applying color permutation (0.0-1.0)
- Works with `--augment` for combined spatial+color augmentation

#### Strategy 3: Color Augmentation Multiplier (Training Loop-level)

Creates N additional variants per sample, each with random spatial + color transforms.

```bash
python3 arc_nodsl/training/pretrain_autoencoder.py \
    --epochs 50 \
    --batch_size 16 \
    --color_aug_multiplier 10 \
    --lr 3e-4
```

**Parameters:**
- `--color_aug_multiplier N`: Creates N additional variants per batch sample
- Each variant: random spatial transform + random color permutation
- **Effective batch size**: batch_size × (1 + multiplier)
- **Example**: `--batch_size 16 --color_aug_multiplier 10` → 176 samples per batch (16 + 16×10)

**Use this strategy for maximum data diversity** (recommended for final training).

### Recommended Training Configurations

#### Fast Training (Development)
```bash
python3 arc_nodsl/training/pretrain_autoencoder.py \
    --epochs 20 \
    --batch_size 32 \
    --augment \
    --color_augment_prob 0.3 \
    --lr 3e-4 \
    --checkpoint_dir checkpoints/dev \
    --log_dir runs/dev
```
- **Time**: ~4 hours on single GPU
- **Expected accuracy**: 92-94%

#### Standard Training (Production)
```bash
python3 arc_nodsl/training/pretrain_autoencoder.py \
    --epochs 50 \
    --batch_size 32 \
    --augment \
    --color_aug_multiplier 5 \
    --lr 3e-4 \
    --checkpoint_dir checkpoints/prod \
    --log_dir runs/prod
```
- **Time**: ~12 hours on single GPU
- **Expected accuracy**: 94-96%

#### Aggressive Training (Competition)
```bash
python3 arc_nodsl/training/pretrain_autoencoder.py \
    --epochs 100 \
    --batch_size 16 \
    --color_aug_multiplier 20 \
    --lr 3e-4 \
    --checkpoint_dir checkpoints/comp \
    --log_dir runs/comp
```
- **Time**: ~2 days on single GPU
- **Expected accuracy**: 96-98%
- **Effective multiplier**: 21x data (1 original + 20 variants)

### All Hyperparameters

```bash
python3 arc_nodsl/training/pretrain_autoencoder.py \
    --data_train <path>              # Training data JSON (default: data/arc-agi_training_challenges.json)
    --data_val <path>                # Validation data JSON (default: data/arc-agi_evaluation_challenges.json)
    --batch_size <int>               # Batch size (default: 32)
    --epochs <int>                   # Number of epochs (default: 50)
    --lr <float>                     # Learning rate (default: 3e-4)
    --num_slots <int>                # Number of slots (default: 8)
    --d_slot <int>                   # Slot dimension (default: 128)
    --num_iters <int>                # SlotAttention iterations (default: 3)
    --checkpoint_dir <path>          # Checkpoint save directory (default: checkpoints)
    --log_dir <path>                 # TensorBoard log directory (default: runs)
    --eval_every <int>               # Evaluate every N epochs (default: 5)
    --save_every <int>               # Save checkpoint every N epochs (default: 10)
    --augment                        # Enable spatial augmentation (8 transforms)
    --color_augment_prob <float>     # Color augmentation probability 0.0-1.0 (default: 0.0)
    --color_aug_multiplier <int>     # Create N variants per sample (default: 0, disabled)
```

### Monitoring Training

**TensorBoard:**
```bash
tensorboard --logdir runs/prod
```

**Key metrics:**
- `train/accuracy`: Pixel-level reconstruction accuracy (target: >95%)
- `train/recon_loss`: Cross-entropy loss (should decrease steadily)
- `train/div_loss`: Mask diversity loss (prevents slot collapse)
- `val/accuracy`: Validation accuracy (use for checkpoint selection)

### Checkpoint Selection

The best checkpoint is automatically saved as `checkpoints/autoencoder_best.pt` based on validation accuracy.

**Verify checkpoint quality:**
```python
import torch
checkpoint = torch.load('checkpoints/autoencoder_best.pt')
print(f"Validation accuracy: {checkpoint['val_acc']*100:.2f}%")
print(f"Epoch: {checkpoint['epoch']}")
```

**Important:** This checkpoint is required for Phase 2 (controller training).

---

## Phase 2: Train Controller

### Goal

Meta-learn the controller and operators to solve ARC tasks via REINFORCE + Reptile. This phase requires a pretrained autoencoder from Phase 1.

### Architecture

- **Controller**: Task-conditioned policy that selects operator sequences
  - Input: Task embedding + slot representations
  - Output: Operator indices + parameters (via beam search)

- **Operators**: Library of M=8 differentiable operators
  - Geometry: Translation, rotation, flip, scale
  - Mask morph: Dilate, erode, outline (spatial edits)
  - Color: Palette remapping

### Meta-Learning Algorithm

**Reptile-style meta-learning:**
1. **Inner loop** (per task): Adapt controller on train pairs via REINFORCE
2. **Outer loop** (across tasks): Update base controller toward adapted parameters

**Key insight:** The controller learns to quickly adapt to new tasks by meta-learning a good initialization.

### Basic Training

**Minimal example:**
```bash
python3 arc_nodsl/training/train_controller.py \
    --autoencoder_checkpoint checkpoints/autoencoder_best.pt \
    --data_train data/arc-agi_training_challenges.json \
    --meta_epochs 50 \
    --meta_batch_size 4 \
    --inner_steps 10 \
    --beam_size 8 \
    --meta_lr 1e-4 \
    --inner_lr 1e-3
```

**Expected output:**
```
Epoch 10 Summary:
  Meta-loss: -0.3245
  Test reward: 0.4821
  Train reward: 0.7234
  Test success: 12.5%
  Train success: 58.3%
✓ Saved best model (test_reward=0.4821)
```

### Data Augmentation for Meta-Learning

#### Strategy 1: Spatial Augmentation (Probabilistic)

Applies spatial transforms to each task with probability P.

```bash
python3 arc_nodsl/training/train_controller.py \
    --autoencoder_checkpoint checkpoints/autoencoder_best.pt \
    --meta_epochs 100 \
    --meta_batch_size 4 \
    --augment \
    --augment_prob 0.5 \
    --augment_exclude_identity
```

**Parameters:**
- `--augment`: Enable spatial augmentation
- `--augment_prob`: Probability of augmenting each task (default: 0.5)
- `--augment_exclude_identity`: Exclude identity transform for max diversity

#### Strategy 2: Color Augmentation Multiplier

Creates N additional task variants with random spatial + color transforms.

```bash
python3 arc_nodsl/training/train_controller.py \
    --autoencoder_checkpoint checkpoints/autoencoder_best.pt \
    --meta_epochs 100 \
    --meta_batch_size 4 \
    --color_aug_multiplier 10 \
    --augment_exclude_identity
```

**Parameters:**
- `--color_aug_multiplier N`: Creates N additional variants per task
- Each variant: random spatial transform + random color permutation
- **Effective meta-batch size**: meta_batch_size / (1 + multiplier) base tasks, expanded to meta_batch_size

**Example:**
- `--meta_batch_size 44 --color_aug_multiplier 10` → sample 4 base tasks, generate 40 variants (4×10)
- Total tasks per meta-batch: 4 + 40 = 44

### Recommended Training Configurations

#### Fast Training (Development)
```bash
python3 arc_nodsl/training/train_controller.py \
    --autoencoder_checkpoint checkpoints/autoencoder_best.pt \
    --meta_epochs 50 \
    --meta_batch_size 4 \
    --inner_steps 10 \
    --beam_size 8 \
    --meta_lr 1e-4 \
    --inner_lr 1e-3 \
    --augment \
    --augment_prob 0.5 \
    --checkpoint_dir checkpoints/dev \
    --log_dir runs/dev
```
- **Time**: ~1 day on single GPU
- **Expected test reward**: 0.3-0.5

#### Standard Training (Production)
```bash
python3 arc_nodsl/training/train_controller.py \
    --autoencoder_checkpoint checkpoints/autoencoder_best.pt \
    --meta_epochs 100 \
    --meta_batch_size 4 \
    --inner_steps 15 \
    --beam_size 16 \
    --meta_lr 1e-4 \
    --inner_lr 1e-3 \
    --color_aug_multiplier 10 \
    --augment_exclude_identity \
    --checkpoint_dir checkpoints/prod \
    --log_dir runs/prod
```
- **Time**: ~3-4 days on single GPU
- **Expected test reward**: 0.5-0.7

#### Aggressive Training (Competition)
```bash
python3 arc_nodsl/training/train_controller.py \
    --autoencoder_checkpoint checkpoints/autoencoder_best.pt \
    --meta_epochs 200 \
    --meta_batch_size 4 \
    --inner_steps 20 \
    --beam_size 16 \
    --max_operator_steps 6 \
    --meta_lr 5e-5 \
    --inner_lr 1e-3 \
    --color_aug_multiplier 50 \
    --augment_exclude_identity \
    --checkpoint_dir checkpoints/comp \
    --log_dir runs/comp
```
- **Time**: ~7 days on single GPU
- **Expected test reward**: 0.7-0.9
- **Note**: Uses 50x augmentation for maximum data diversity

### All Hyperparameters

```bash
python3 arc_nodsl/training/train_controller.py \
    --autoencoder_checkpoint <path>  # REQUIRED: Path to pretrained autoencoder
    --data_train <path>              # Training data JSON (default: data/arc-agi_training_challenges.json)
    --meta_epochs <int>              # Number of meta-training epochs (default: 100)
    --meta_batch_size <int>          # Number of tasks per meta-batch (default: 4)
    --inner_steps <int>              # Inner loop gradient steps per task (default: 10)
    --beam_size <int>                # Beam size for search (default: 8)
    --max_operator_steps <int>       # Max operator sequence length (default: 4)
    --meta_lr <float>                # Meta-learning rate (Reptile) (default: 1e-4)
    --inner_lr <float>               # Inner loop learning rate (default: 1e-3)
    --num_ops <int>                  # Number of operators in library (default: 8)
    --checkpoint_dir <path>          # Checkpoint save directory (default: checkpoints)
    --log_dir <path>                 # TensorBoard log directory (default: runs)
    --save_every <int>               # Save checkpoint every N epochs (default: 10)
    --eval_every <int>               # Evaluate every N epochs (default: 5)
    # Augmentation
    --augment                        # Enable spatial augmentation
    --augment_prob <float>           # Probability of augmenting task (default: 0.5)
    --augment_exclude_identity       # Exclude identity transform for diversity
    --color_aug_multiplier <int>     # Create N variants per task (default: 0, disabled)
```

### Understanding Meta-Loss

**Why is meta-loss negative?**

The meta-loss is **intentionally negative** because it equals `-mean_test_reward`. This converts reward maximization (higher = better) into loss minimization (lower = better) for gradient descent optimizers.

- **Negative meta-loss**: Good! (e.g., -0.85 means test reward = 0.85)
- **More negative = better performance**
- Monitor `test_reward` directly for clarity

### Monitoring Training

**TensorBoard:**
```bash
tensorboard --logdir runs/prod
```

**Key metrics:**
- `train/test_reward`: Reward on held-out test pairs (target: 0.7-0.9)
- `train/train_reward`: Reward on training pairs (should be higher)
- `train/test_success_rate`: Fraction of tasks with 100% test accuracy
- `train/train_success_rate`: Fraction of tasks with 100% train accuracy
- `epoch/meta_loss`: Negative test reward (more negative = better)

**Convergence indicators:**
- `train_reward > 0.8`: Controller can solve training pairs
- `test_reward > 0.5`: Controller generalizes to test pairs
- `test_success_rate > 10%`: Significant fraction of tasks fully solved

### Checkpoint Selection

The best checkpoint is automatically saved as `checkpoints/controller_best.pt` based on test reward.

**Verify checkpoint quality:**
```python
import torch
checkpoint = torch.load('checkpoints/controller_best.pt')
print(f"Test reward: {checkpoint['test_reward']:.3f}")
print(f"Train reward: {checkpoint['train_reward']:.3f}")
print(f"Epoch: {checkpoint['epoch']}")
```

---

## Phase 3: Inference & Evaluation

### Overview

Three inference strategies with different speed/accuracy trade-offs:

1. **Passive Inference**: Standard beam search (baseline)
2. **Active Learning**: Test-time adaptation per task (+10-20% accuracy, 10x slower)
3. **Test-Time Augmentation (TTA)**: Ensemble across transforms (+2-5% accuracy, 8x slower)

All strategies support **multi-attempt inference** (K=2 for competition).

### Strategy 1: Passive Inference (Baseline)

Standard beam search without adaptation or ensembling.

```bash
python3 arc_nodsl/evaluation/evaluate_model.py \
    --autoencoder_checkpoint checkpoints/autoencoder_best.pt \
    --controller_checkpoint checkpoints/controller_best.pt \
    --dataset data/arc-agi_evaluation_challenges.json \
    --beam_size 16 \
    --max_steps 8 \
    --num_attempts 2 \
    --output_dir evaluation_results/passive
```

**Parameters:**
- `--beam_size`: Beam width (larger = better but slower, 8-32 recommended)
- `--max_steps`: Max operator sequence length (4-8 recommended)
- `--num_attempts`: Generate K diverse predictions per test input (2 for competition)

**Output:**
```
============================================================
Computing Metrics
============================================================

Task Success Rate: 15.2% (61/400 tasks fully solved)
Train Solved Rate: 42.5% (170/400 tasks with all train pairs solved)
Test Accuracy Given Train: 35.8% (test correct when train solved)
Coverage: 42.5% (fraction of tasks where test was attempted)

✓ Saved summary to evaluation_results/passive/summary.json
✓ Saved detailed results to evaluation_results/passive/detailed_results.json
```

**When to use:**
- Fast evaluation during development
- Baseline performance measurement
- Resource-constrained inference

### Strategy 2: Active Learning (Test-Time Adaptation)

Adapts controller on each task's training pairs before predicting test pairs.

```bash
python3 arc_nodsl/evaluation/evaluate_model.py \
    --autoencoder_checkpoint checkpoints/autoencoder_best.pt \
    --controller_checkpoint checkpoints/controller_best.pt \
    --dataset data/arc-agi_evaluation_challenges.json \
    --beam_size 16 \
    --max_steps 8 \
    --num_attempts 2 \
    --active_learning \
    --adaptation_steps 20 \
    --adaptation_lr 1e-3 \
    --time_budget 60.0 \
    --beam_size_adaptation 8 \
    --output_dir evaluation_results/active
```

**Parameters:**
- `--active_learning`: Enable test-time adaptation
- `--adaptation_steps`: Max gradient steps for adaptation (10-30 recommended)
- `--adaptation_lr`: Learning rate for adaptation (1e-3 recommended)
- `--time_budget`: Max seconds per task (60-120 recommended)
- `--beam_size_adaptation`: Beam size during adaptation (smaller = faster, 8 recommended)

**Algorithm:**
1. For each task:
   a. Adapt controller on training pairs (up to `adaptation_steps` or `time_budget`)
   b. Use adapted controller to predict test pairs
   c. Revert to base controller for next task

**Expected performance:**
- **Accuracy improvement**: +10-20% over passive inference
- **Speed**: ~10x slower (60s per task with time budget)
- **Best for**: Maximizing accuracy when compute is available

**When to use:**
- Competition inference (when time permits)
- Tasks with clear patterns in training pairs
- Final evaluation for best results

### Active Learning Visualizer 🎨

**Beautiful real-time terminal UI** for visualizing active learning adaptation on individual tasks.

The visualizer provides unprecedented insight into how the controller adapts during test-time learning, displaying training pairs, predictions, metrics, and convergence in real-time with a professional Claude CLI-inspired interface.

#### Installation

```bash
cd arc-visualizer
npm install
```

**Prerequisites:** Node.js 18+, npm

#### Quick Start

```bash
# From the arc-visualizer directory
cd arc-visualizer

# Run visualizer on a specific task
npm run dev -- \
  --autoencoder ../checkpoints/autoencoder_best.pt \
  --controller ../checkpoints/controller_best.pt \
  --task-index 0 \
  --steps 20

# Or run from project root using wrapper script
./visualize.sh \
  -a checkpoints/autoencoder_best.pt \
  -c checkpoints/controller_best.pt \
  -t 00576224 \
  -s 20
```

#### Command-Line Options

```
Required:
  -a, --autoencoder <path>   Autoencoder checkpoint
  -c, --controller <path>    Controller checkpoint

Task Selection (pick one):
  -t, --task-id <id>         Task ID (e.g., "00576224")
  -i, --task-index <n>       Task index (0-based, e.g., 0)

Optional:
  -d, --dataset <path>       Dataset file (default: evaluation)
  -s, --steps <n>            Adaptation steps (default: 20)
  -b, --beam-size <n>        Beam size (default: 8)
```

#### What You'll See

The visualizer displays a live terminal UI with:

**1. Training Pairs Panel** (Left Top)
- All training pairs from the task
- Input grids, target outputs, current predictions
- Real-time accuracy for each pair (100% = ✓)
- Full 11-color ARC palette

**2. Adaptation Progress** (Left Bottom)
- Step progress bar (e.g., 15/20)
- Time elapsed vs budget
- Training pairs solved counter (e.g., 3/3 ✓)
- Status: Loading → Adapting → Converged → Complete

**3. Metrics Dashboard** (Right Top)
- Mean reward and best reward
- Mean loss trajectory
- Train accuracy percentage
- Sparkline charts: ▁▂▃▅▇█

**4. Logs** (Right Bottom)
- Real-time messages from Python backend
- Latest 5 log entries

**5. Test Results** (Bottom, when complete)
- Final test predictions
- Success indicator (🎉 if solved!)
- Competition score

#### Example Session

```bash
cd arc-visualizer
npm run dev -- \
  -a ../checkpoints/autoencoder_best.pt \
  -c ../checkpoints/controller_best.pt \
  --task-id 00576224 \
  --steps 30

# Output:
# ╭─────── ARC Active Learning Visualizer ────────╮
# │ Task: 00576224                                │
# ├─ Training Pairs ─────┬─ Metrics ─────────────┤
# │ Pair 1:              │ Mean Reward: 0.852    │
# │  ██░░ → ████░ ✓ 100% │ Train Acc: 100%       │
# │                      │ Reward: ▁▂▃▅▇█        │
# ├──────────────────────┼───────────────────────┤
# │ Progress: 20/30 ████ │ Status: ✓ Converged!  │
# ╰──────────────────────┴───────────────────────╯
```

#### Architecture

The visualizer uses a client-server architecture:
- **Frontend:** TypeScript + ink (React for terminals)
- **Backend:** Python with streaming JSON events
- **Communication:** IPC via stdout (no network needed)

See `arc-visualizer/ARCHITECTURE.md` for details.

#### Troubleshooting

**"Module not found" errors:**
```bash
cd arc-visualizer
rm -rf node_modules package-lock.json
npm install
```

**Python backend errors:**
- Verify checkpoints exist and are compatible (11-color model required)
- Check Python dependencies: `pip install torch`

**Terminal too small:**
- Minimum terminal size: 120×30 characters
- Check: `echo $COLUMNS x $LINES`

#### When to Use the Visualizer

- **Development:** Understand how controller adapts on specific tasks
- **Debugging:** Identify why certain tasks fail to converge
- **Demonstrations:** Beautiful UI for presentations/papers
- **Analysis:** Study adaptation dynamics and operator selection

**Note:** The visualizer runs active learning on a single task. For batch evaluation, use `evaluate_model.py` instead.

### Strategy 3: Test-Time Augmentation (TTA)

Ensembles predictions across spatial (and optionally color) transforms.

#### Spatial-only TTA (8 transforms)

```bash
python3 arc_nodsl/evaluation/evaluate_model.py \
    --autoencoder_checkpoint checkpoints/autoencoder_best.pt \
    --controller_checkpoint checkpoints/controller_best.pt \
    --dataset data/arc-agi_evaluation_challenges.json \
    --beam_size 16 \
    --max_steps 8 \
    --num_attempts 2 \
    --tta \
    --tta_mode majority \
    --output_dir evaluation_results/tta_spatial
```

**Parameters:**
- `--tta`: Enable test-time augmentation
- `--tta_mode`: Ensemble mode (`majority` or `first_success`)
  - `majority`: Pixel-wise majority voting across 8 predictions
  - `first_success`: Return first prediction that matches target (validation only)

**Algorithm:**
1. For each test input:
   a. Apply 8 spatial transforms (identity, rot90, rot180, rot270, flip_h, flip_v, flip_d1, flip_d2)
   b. Run inference on each transformed input
   c. Inverse transform each prediction back to original space
   d. Ensemble via majority voting

**Expected performance:**
- **Accuracy improvement**: +2-5% over passive inference
- **Speed**: 8x slower
- **Best for**: Balancing accuracy and compute

#### Spatial + Color TTA (8 × N transforms)

```bash
python3 arc_nodsl/evaluation/evaluate_model.py \
    --autoencoder_checkpoint checkpoints/autoencoder_best.pt \
    --controller_checkpoint checkpoints/controller_best.pt \
    --dataset data/arc-agi_evaluation_challenges.json \
    --beam_size 16 \
    --max_steps 8 \
    --num_attempts 2 \
    --tta \
    --tta_color \
    --tta_color_variants 4 \
    --tta_mode majority \
    --output_dir evaluation_results/tta_full
```

**Parameters:**
- `--tta_color`: Enable color TTA (requires `--tta`)
- `--tta_color_variants`: Number of color permutations (4-8 recommended)

**Algorithm:**
1. For each test input:
   a. For each of 8 spatial transforms:
      - For each of N color permutations:
        - Apply spatial + color transform
        - Run inference
        - Inverse transform (color then spatial)
   b. Ensemble via majority voting across 8×N predictions

**Expected performance:**
- **Accuracy improvement**: +3-8% over passive inference
- **Speed**: 8×N times slower (32x with N=4)
- **Best for**: Maximum accuracy (competition, unlimited compute)

**When to use TTA:**
- Final evaluation for competition
- Tasks with rotational/reflectional symmetry
- When passive inference is close but not perfect

**Note:** Cannot use `--active_learning` and `--tta` simultaneously. Choose one strategy.

### Multi-Attempt Inference (Competition Setting)

All strategies support generating K diverse predictions per test input (K=2 for ARC competition).

**Scoring:**
- **Competition score**: Task solved if ANY of K attempts is correct
- **Backward compatibility**: Task success uses best (first) attempt only

**Example output:**
```
Test pair 1: ✓ (attempt 2/2)  # Second attempt was correct
Competition score: 100% (1/1 with ANY attempt correct)
```

### Quick Evaluation (Testing)

Test evaluation pipeline on 3 tasks:

```bash
python3 arc_nodsl/evaluation/evaluate_model.py \
    --autoencoder_checkpoint checkpoints/autoencoder_best.pt \
    --controller_checkpoint checkpoints/controller_best.pt \
    --dataset data/arc-agi_evaluation_challenges.json \
    --num_tasks 3 \
    --beam_size 8 \
    --max_steps 4 \
    --verbose
```

### All Evaluation Parameters

```bash
python3 arc_nodsl/evaluation/evaluate_model.py \
    --autoencoder_checkpoint <path>  # REQUIRED: Path to pretrained autoencoder
    --controller_checkpoint <path>   # REQUIRED: Path to trained controller
    --dataset <path>                 # Dataset to evaluate (default: data/arc-agi_evaluation_challenges.json)
    --output_dir <path>              # Output directory for results (default: evaluation_results)
    --beam_size <int>                # Beam size for search (default: 16)
    --max_steps <int>                # Max operator sequence length (default: 8)
    --num_attempts <int>             # Attempts per test input (default: 2)
    --num_tasks <int>                # Evaluate only first N tasks (for testing)
    --verbose                        # Print per-task detailed output
    --no_save                        # Don't save results to disk
    # Active Learning
    --active_learning                # Enable test-time adaptation
    --adaptation_steps <int>         # Max gradient steps (default: 20)
    --adaptation_lr <float>          # Adaptation learning rate (default: 1e-3)
    --time_budget <float>            # Max seconds per task (default: 60.0)
    --beam_size_adaptation <int>     # Beam size during adaptation (default: 8)
    # Test-Time Augmentation
    --tta                            # Enable TTA ensemble
    --tta_mode <str>                 # Ensemble mode: 'majority' or 'first_success' (default: majority)
    --tta_color                      # Enable color TTA (requires --tta)
    --tta_color_variants <int>       # Number of color permutations (default: 4)
```

### Output Files

Evaluation results are saved to `output_dir/`:

**`summary.json`**: Aggregate metrics
```json
{
  "timestamp": "2025-10-12T14:30:00",
  "metrics": {
    "task_success_rate": 0.152,
    "train_solved_rate": 0.425,
    "test_accuracy_given_train": 0.358,
    "coverage": 0.425,
    "total_tasks": 400,
    "num_tasks_solved": 61,
    "num_train_solved": 170,
    "num_train_failed": 230
  },
  "tasks_solved": ["00576224", "007bbfb7", ...],
  "tasks_train_only": ["00d62c1b", ...],
  "tasks_failed": ["0a938d79", ...]
}
```

**`detailed_results.json`**: Per-task results
```json
[
  {
    "task_id": "00576224",
    "train_solved": true,
    "test_attempted": true,
    "task_success": true,
    "confidence": 1.0,
    "train_pairs": [
      {"is_solved": true, "pixel_accuracy": 1.0},
      {"is_solved": true, "pixel_accuracy": 1.0}
    ],
    "test_correct": [true]
  },
  ...
]
```

---

## Model Architecture

### Component Overview

```
Input Grid [B, H, W] (colors 0-10)
    ↓
┌─────────────────────────────────────────┐
│  ENCODER (frozen after Phase 1)         │
├─────────────────────────────────────────┤
│  PaletteEmbedding: [B, H, W] → [B, H, W, 16]  │
│  CNNFeatureExtractor: → [B, H, W, 64]         │
│  SlotAttention: → [B, K, 128]                  │
└─────────────────────────────────────────┘
    ↓
  Slots (z, m, p)
  - z: [B, K=8, 128] features
  - m: [B, K=8, H, W] masks
  - p: [B, K=8, 2] centroids
    ↓
┌─────────────────────────────────────────┐
│  CONTROLLER (trained in Phase 2)        │
├─────────────────────────────────────────┤
│  TaskEmbedder: train pairs → [B, 128]        │
│  OperatorSelector: (slots, task) → op_idx    │
│  BeamSearch: → operator sequence [T]         │
└─────────────────────────────────────────┘
    ↓
┌─────────────────────────────────────────┐
│  OPERATORS (trained in Phase 2)         │
├─────────────────────────────────────────┤
│  LatentOp × 8:                              │
│    - SetTransformer: process slots          │
│    - GeometryHead: translation, rotation    │
│    - MaskMorphHead: dilate, erode          │
│    - ColorHead: palette remapping          │
└─────────────────────────────────────────┘
    ↓
  Modified Slots (z', m', p')
    ↓
┌─────────────────────────────────────────┐
│  RENDERER (frozen after Phase 1)        │
├─────────────────────────────────────────┤
│  SlotDecoder: [B, K, 128] → [B, H, W, 11]   │
│  AlphaCompositing: weighted sum over K       │
└─────────────────────────────────────────┘
    ↓
Output Grid [B, H, W] (colors 0-10)
```

### Dimensions

| Component | Parameters | Shape | Description |
|-----------|-----------|-------|-------------|
| **Encoder** | ~500K | | |
| PaletteEmbedding | 176 | (11, 16) | Color embeddings |
| CNNFeatureExtractor | ~50K | | 4-layer CNN |
| SlotAttention | ~450K | | Iterative attention |
| **Renderer** | ~200K | | |
| SlotDecoder | ~200K | | Spatial broadcast MLP |
| **Controller** | ~1.2M | | |
| TaskEmbedder | ~100K | | Train pair encoder |
| OperatorSelector | ~1.1M | | LSTM + attention |
| **Operators** | ~1.5M | | |
| LatentOp × 8 | ~190K each | | Set transformer + heads |
| **Total** | **~3.4M** | | All parameters |

### Memory Requirements

| Configuration | GPU Memory | Batch Size | Throughput |
|---------------|-----------|------------|------------|
| Development | 4GB | 16 | ~10 samples/sec |
| Standard | 8GB | 32 | ~18 samples/sec |
| Production | 16GB | 64 | ~30 samples/sec |

---

## Hyperparameter Guide

### Autoencoder Training

| Hyperparameter | Development | Production | Competition | Notes |
|----------------|-------------|------------|-------------|-------|
| `epochs` | 20 | 50 | 100 | More epochs for better convergence |
| `batch_size` | 32 | 32 | 16 | Reduce if OOM, increase for speed |
| `lr` | 3e-4 | 3e-4 | 3e-4 | Adam learning rate |
| `num_slots` | 8 | 8 | 8 | Number of object slots (K) |
| `d_slot` | 128 | 128 | 128 | Slot feature dimension |
| `num_iters` | 3 | 3 | 3 | SlotAttention iterations |
| `augment` | ✓ | ✓ | ✗ | Spatial augmentation |
| `color_augment_prob` | 0.3 | 0.0 | 0.0 | Color probability (use with augment) |
| `color_aug_multiplier` | 0 | 5 | 20 | Color multiplier (use alone) |

**Tuning tips:**
- **Underfitting** (low train accuracy): Increase `epochs`, `num_slots`, `d_slot`
- **Overfitting** (train >> val accuracy): Increase augmentation, reduce `d_slot`
- **Slow convergence**: Increase `lr` to 5e-4, reduce `batch_size`
- **Memory issues**: Reduce `batch_size`, `num_slots`, or `color_aug_multiplier`

### Controller Training

| Hyperparameter | Development | Production | Competition | Notes |
|----------------|-------------|------------|-------------|-------|
| `meta_epochs` | 50 | 100 | 200 | Meta-training iterations |
| `meta_batch_size` | 4 | 4 | 4 | Tasks per meta-batch |
| `inner_steps` | 10 | 15 | 20 | Gradient steps per task |
| `beam_size` | 8 | 16 | 16 | Beam search width |
| `max_operator_steps` | 4 | 4 | 6 | Max operator sequence length |
| `meta_lr` | 1e-4 | 1e-4 | 5e-5 | Reptile meta-learning rate |
| `inner_lr` | 1e-3 | 1e-3 | 1e-3 | Inner loop learning rate |
| `num_ops` | 8 | 8 | 8 | Operator library size |
| `augment` | ✓ | ✗ | ✗ | Spatial augmentation |
| `augment_prob` | 0.5 | 0.0 | 0.0 | Augmentation probability |
| `color_aug_multiplier` | 0 | 10 | 50 | Color augmentation multiplier |

**Tuning tips:**
- **Low test reward** (<0.3): Increase `inner_steps`, `beam_size`, augmentation
- **High train, low test**: Add more augmentation, reduce `inner_lr`
- **Slow meta-learning**: Increase `meta_lr` to 2e-4, reduce `meta_batch_size` to 2
- **Memory issues**: Reduce `beam_size`, `meta_batch_size`, or `color_aug_multiplier`

### Inference

| Hyperparameter | Fast | Balanced | Accurate | Notes |
|----------------|------|----------|----------|-------|
| `beam_size` | 8 | 16 | 32 | Search width |
| `max_steps` | 4 | 8 | 12 | Operator sequence length |
| `num_attempts` | 1 | 2 | 2 | Attempts per test input |
| **Active Learning** | | | | |
| `adaptation_steps` | 10 | 20 | 30 | Gradient steps |
| `adaptation_lr` | 1e-3 | 1e-3 | 1e-3 | Learning rate |
| `time_budget` | 30 | 60 | 120 | Max seconds per task |
| `beam_size_adaptation` | 4 | 8 | 16 | Beam size during adaptation |
| **TTA** | | | | |
| `tta_color` | ✗ | ✗ | ✓ | Enable color TTA |
| `tta_color_variants` | - | - | 4 | Color permutations |
| `tta_mode` | - | majority | majority | Ensemble mode |

**Tuning tips:**
- **Speed priority**: Use passive inference with small beam (8-16)
- **Accuracy priority**: Use active learning with large beam (16-32)
- **Maximum accuracy**: Combine active learning + TTA (very slow)

---

## Tips & Best Practices

### Training Best Practices

1. **Always start with a quick test run**
   ```bash
   # Test autoencoder (5 minutes)
   python3 arc_nodsl/training/pretrain_autoencoder.py --epochs 1 --batch_size 8

   # Test controller (10 minutes)
   python3 arc_nodsl/training/train_controller.py \
       --autoencoder_checkpoint checkpoints/autoencoder_best.pt \
       --meta_epochs 1 --meta_batch_size 2 --inner_steps 3
   ```

2. **Monitor training with TensorBoard in real-time**
   ```bash
   tensorboard --logdir runs --port 6006
   ```

3. **Save checkpoints frequently during long runs**
   - Use `--save_every 5` for autoencoder
   - Use `--save_every 10` for controller
   - Disk space: ~50MB per autoencoder checkpoint, ~100MB per controller checkpoint

4. **Use color augmentation multiplier for final training**
   - More effective than probability-based augmentation
   - Multiplier 10-20 gives significant gains
   - Reduce `batch_size` if GPU memory is limited

5. **Verify checkpoint quality before Phase 2**
   ```python
   import torch
   ckpt = torch.load('checkpoints/autoencoder_best.pt')
   assert ckpt['val_acc'] > 0.95, "Autoencoder accuracy too low!"
   ```

### Inference Best Practices

1. **Start with passive inference as baseline**
   - Fast feedback loop
   - Measure baseline performance
   - Identify failure modes

2. **Use active learning for final evaluation**
   - +10-20% accuracy improvement
   - Especially effective for tasks with clear patterns
   - Budget 60-120s per task

3. **Consider TTA for marginal gains**
   - +2-5% accuracy improvement
   - Use spatial-only TTA for speed/accuracy balance
   - Use spatial+color TTA for maximum accuracy (competition)

4. **Generate multiple attempts (K=2) for competition**
   - Increases success rate by ~5-10%
   - No additional inference cost (uses beam diversity)

5. **Evaluate on small subset first**
   ```bash
   # Test on 10 tasks
   python3 arc_nodsl/evaluation/evaluate_model.py \
       --autoencoder_checkpoint checkpoints/autoencoder_best.pt \
       --controller_checkpoint checkpoints/controller_best.pt \
       --num_tasks 10
   ```

### Data Augmentation Strategy

**For autoencoder:**
- Development: `--augment --color_augment_prob 0.3`
- Production: `--color_aug_multiplier 5`
- Competition: `--color_aug_multiplier 20`

**For controller:**
- Development: `--augment --augment_prob 0.5`
- Production: `--color_aug_multiplier 10`
- Competition: `--color_aug_multiplier 50`

**Why multiplier > probability?**
- Multiplier creates more diverse data (random spatial + color each time)
- Probability only augments some samples (less diversity)
- Multiplier is training-loop level (consistent with controller training)

### GPU Memory Optimization

**If you encounter OOM errors:**

1. **Reduce batch size**
   ```bash
   --batch_size 16  # Down from 32
   ```

2. **Reduce color augmentation multiplier**
   ```bash
   --color_aug_multiplier 5  # Down from 10
   ```

3. **Reduce beam size (controller training)**
   ```bash
   --beam_size 8  # Down from 16
   ```

4. **Use gradient accumulation (autoencoder)**
   - Simulate large batch with smaller steps
   - Modify `pretrain_autoencoder.py` to accumulate gradients

5. **Enable mixed precision training (already enabled)**
   - Uses FP16 for forward/backward passes
   - Automatic in current implementation

---

## Troubleshooting

### Autoencoder Issues

#### Low reconstruction accuracy (<90%)

**Symptoms:** Validation accuracy stuck below 90%

**Possible causes:**
1. Insufficient training epochs
2. Too few slots or slot dimensions
3. Learning rate too high/low

**Solutions:**
```bash
# Increase epochs
--epochs 100

# Increase model capacity
--num_slots 12 --d_slot 192

# Tune learning rate
--lr 5e-4  # If underfitting
--lr 1e-4  # If unstable
```

#### Slot collapse (all masks identical)

**Symptoms:** `div_loss` near zero, all slots look the same

**Solutions:**
```bash
# Increase diversity loss weight (modify pretrain_autoencoder.py:180)
loss = recon_loss + 0.05 * div_loss  # Increase from 0.01

# Increase SlotAttention iterations
--num_iters 5
```

#### OOM during training

**Solutions:**
```bash
# Reduce batch size
--batch_size 16

# Reduce augmentation multiplier
--color_aug_multiplier 5  # Down from 10

# Monitor GPU memory
nvidia-smi -l 1  # Watch memory usage
```

### Controller Issues

#### Negative meta-loss is confusing

**Explanation:** Meta-loss = -test_reward (by design)
- More negative = better
- Monitor `test_reward` directly for clarity

#### Low test reward (<0.3)

**Symptoms:** Controller doesn't generalize to test pairs

**Possible causes:**
1. Autoencoder quality insufficient
2. Too few inner loop steps
3. Insufficient exploration (beam size)

**Solutions:**
```bash
# Verify autoencoder
python3 -c "import torch; ckpt=torch.load('checkpoints/autoencoder_best.pt'); print(f'Acc: {ckpt[\"val_acc\"]*100:.1f}%')"
# Should be >95%

# Increase inner loop adaptation
--inner_steps 20

# Increase exploration
--beam_size 16

# Add more augmentation
--color_aug_multiplier 20
```

#### High train reward, low test reward

**Symptoms:** `train_reward > 0.8`, but `test_reward < 0.4`

**Solutions:**
```bash
# Increase data augmentation
--color_aug_multiplier 20

# Reduce inner loop learning rate (less overfitting to train)
--inner_lr 5e-4

# Increase meta-batch diversity
--meta_batch_size 8
```

#### Training is very slow

**Solutions:**
```bash
# Reduce beam size during training
--beam_size 8  # Down from 16

# Reduce inner steps
--inner_steps 10  # Down from 15

# Reduce augmentation multiplier
--color_aug_multiplier 5  # Down from 10

# Use multiple GPUs (if available)
# Modify train_controller.py for DataParallel
```

### Evaluation Issues

#### Checkpoint not found

**Error:** `FileNotFoundError: checkpoints/autoencoder_best.pt`

**Solution:**
```bash
# List available checkpoints
ls checkpoints/

# Use specific checkpoint
--autoencoder_checkpoint checkpoints/autoencoder_epoch50.pt
```

#### Checkpoint incompatible (architecture mismatch)

**Error:** `RuntimeError: size mismatch for decoder.final_layer.weight`

**Cause:** Model architecture changed (e.g., 10→11 color support)

**Solution:**
- **Retrain from scratch** with new architecture
- Cannot load old checkpoints (incompatible)

#### Task success rate is 0%

**Possible causes:**
1. Using untrained/random controller
2. Autoencoder quality insufficient
3. Wrong checkpoint loaded

**Solutions:**
```bash
# Verify checkpoints
python3 -c "
import torch
ae = torch.load('checkpoints/autoencoder_best.pt')
ctrl = torch.load('checkpoints/controller_best.pt')
print(f'Autoencoder acc: {ae[\"val_acc\"]*100:.1f}%')
print(f'Controller test reward: {ctrl[\"test_reward\"]:.3f}')
"

# Expected:
# Autoencoder acc: >95%
# Controller test reward: >0.3

# Test on single task with verbose
--num_tasks 1 --verbose
```

#### Active learning fails

**Error:** Adaptation doesn't improve over passive

**Solutions:**
```bash
# Increase adaptation budget
--adaptation_steps 30
--time_budget 120

# Increase adaptation beam size
--beam_size_adaptation 16

# Verify controller supports adaptation
# (Should have been trained with inner loop)
```

### Data Issues

#### Dataset not found

**Error:** `FileNotFoundError: data/arc-agi_training_challenges.json`

**Solution:**
```bash
# Download ARC-AGI dataset
cd data/
wget https://github.com/fchollet/ARC-AGI/raw/master/data/training_challenges.json
mv training_challenges.json arc-agi_training_challenges.json
# Repeat for evaluation and test sets
```

#### Invalid color values

**Error:** `IndexError: Target 10 is out of bounds`

**Cause:** Old model with 10-color support, new data with 11 colors

**Solution:**
- **Use updated model architecture** (see commit history)
- All models now support 11 colors (0-10)
- Retrain with new architecture if using old checkpoints

---

## Summary

### Training Timeline

| Phase | Time | GPU | Checkpoint | Expected Performance |
|-------|------|-----|------------|---------------------|
| **Phase 1: Autoencoder** | 12h | 1×8GB | `autoencoder_best.pt` | 95% reconstruction |
| **Phase 2: Controller** | 3-4d | 1×16GB | `controller_best.pt` | 0.5-0.7 test reward |
| **Phase 3: Evaluation** | 2-6h | 1×8GB | - | 15-30% task success |

### Quick Start Commands

```bash
# Phase 1: Pretrain autoencoder
python3 arc_nodsl/training/pretrain_autoencoder.py \
    --epochs 50 --batch_size 32 --color_aug_multiplier 5

# Phase 2: Train controller
python3 arc_nodsl/training/train_controller.py \
    --autoencoder_checkpoint checkpoints/autoencoder_best.pt \
    --meta_epochs 100 --color_aug_multiplier 10

# Phase 3: Evaluate
python3 arc_nodsl/evaluation/evaluate_model.py \
    --autoencoder_checkpoint checkpoints/autoencoder_best.pt \
    --controller_checkpoint checkpoints/controller_best.pt \
    --beam_size 16 --num_attempts 2
```

### Expected Final Performance

| Dataset | Passive | Active | TTA | Notes |
|---------|---------|--------|-----|-------|
| Training (400 tasks) | 20-35% | 30-50% | 25-40% | Seen during meta-training |
| Evaluation (400 tasks) | 10-20% | 15-30% | 12-25% | Validation set |
| Test (100 tasks) | 5-15% | 10-25% | 8-20% | Hidden test set (no solutions) |

**Target:** 15-20% on evaluation set with passive inference is competitive.

---

## Additional Resources

- **Main README**: `/home/atgu/Projects/ARC/ProbDiff/README.md`
- **Model implementations**: `arc_nodsl/models/`
- **Training scripts**: `arc_nodsl/training/`
- **Evaluation scripts**: `arc_nodsl/evaluation/`
- **Visualization**: `python -m arc_nodsl.cli.visualize_task --task_id <id>`

For questions or issues, check the GitHub repository or contact the development team.
