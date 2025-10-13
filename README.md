# SPARC: Slot Programs via Active Radiation for ARC

An experimental approach to ARC that attempts to learn operators end-to-end rather than using hand-written rules.

<p align="center">
  <img src="visualizer.png" alt="SPARC Active Learning Visualizer" width="800">
  <br>
  <em>Real-time terminal UI showing active learning adaptation on an ARC task</em>
</p>

---

## What is SPARC?

**SPARC** (Slot Programs via Active Radiation for ARC) is an experimental machine learning system for [Abstraction and Reasoning Corpus (ARC)](https://github.com/fchollet/ARC-AGI) tasks. It explores learning neural operators rather than hand-crafting transformation rules.

**The Problem**: ARC tasks require abstract visual reasoning—identifying patterns, applying transformations, and generalizing from few examples. Traditional approaches use domain-specific languages (DSLs) with hundreds of hand-coded primitives.

**Our Approach**: SPARC attempts to learn a small library of differentiable operators (8-16) that act on object-centric representations:
1. Decompose grids into "slots" (object representations) via attention
2. Learn operators that transform these slots (geometry, mask morphology, color)
3. Train a controller to compose operators into programs
4. Use meta-learning to help the controller adapt to new tasks

**Goal**: Explore whether abstract reasoning patterns can emerge from learned compositional transformations on structured representations.

---

## How It Works

### Training Pipeline (3 Phases)

```
Phase 1: Autoencoder Pretraining (1-2 days)
┌─────────────────────────────────────┐
│ Input Grid → Encoder → K Slots     │
│ Slots → Renderer → Reconstructed   │
│ Goal: >95% pixel accuracy          │
└─────────────────────────────────────┘
         ↓
Phase 2: Controller Meta-Learning (3-5 days)
┌─────────────────────────────────────┐
│ Meta-training across 400 tasks     │
│ Inner loop: Adapt on task's train  │
│ Outer loop: Update base controller │
│ Goal: 10-30% task solve rate       │
└─────────────────────────────────────┘
         ↓
Phase 3: Inference (seconds per task)
┌─────────────────────────────────────┐
│ Passive: Beam search for operators │
│ Active: Test-time adaptation on     │
│         task training pairs         │
└─────────────────────────────────────┘
```

### Inference Process

1. **Encode**: Transform input grid → K slots `{(z, m, p)}`
   - `z`: appearance/shape features
   - `m`: soft segmentation mask
   - `p`: spatial centroid

2. **Search**: Controller proposes operator sequences
   - Beam search explores top-K candidates
   - Each candidate applies operators to slots
   - "Radiation": Stochastic variants for diversity

3. **Apply**: Operators transform slots
   - Geometry: translate, rotate, flip
   - Mask morphology: dilate, erode, outline
   - Color: palette remapping

4. **Decode**: Renderer converts modified slots → output grid

5. **Evaluate**: Compare to target, compute reward

6. **Adapt** (Active Learning): Fine-tune controller on task's training pairs before predicting test outputs

---

## Key Features

- **Learned Operators**: Attempts to learn operators from data rather than hand-coding
- **Object-Centric Representations**: Uses slot attention to decompose grids
- **Compositional Programs**: Sequences of learned operators
- **Meta-Learning**: Uses REINFORCE + Reptile for adaptation
- **Test-Time Adaptation**: Active learning on task training pairs
- **Real-Time Visualizer**: Terminal UI for debugging and visualization

---

## Architecture Overview

**Core Idea**: Learn latent operators that act on object-centric representations (slots + masks) rather than hand-coding rules. Training pairs supervise only the final output; the system attempts to discover reusable operators through gradient-based learning.

### Components

1. **Encoder (E)**: Grid (H×W) → K slots `{(z_i, m_i, p_i)}`
   - `z_i ∈ R^D`: appearance/shape features
   - `m_i ∈ [0,1]^{H×W}`: soft mask
   - `p_i ∈ R^2`: centroid/pose

2. **Operators {F_k}**: Small networks that edit (z, m, p)
   - Geometry: move, rotate, flip
   - Mask morph: dilate, erode, outline
   - Color: palette remapping

3. **Controller (C)**: Chooses operator sequence `a_{1:T}` and params `θ_{1:T}`
   - Conditioned on slots + task embedding from train pairs

4. **Renderer (R)**: Decodes modified slots → output grid

5. **Probability Radiation**: Stochastic beam search with diffusion over (a, θ)

---

## Training & Evaluation

### Quick Training Guide

```bash
# Phase 1: Pretrain autoencoder (12 hours, single GPU)
python3 arc_nodsl/training/pretrain_autoencoder.py \
    --epochs 50 --batch_size 32 --color_aug_multiplier 5

# Phase 2: Train controller (3-4 days, single GPU)
python3 arc_nodsl/training/train_controller.py \
    --autoencoder_checkpoint checkpoints/autoencoder_best.pt \
    --meta_epochs 100 --color_aug_multiplier 10

# Phase 3: Evaluate
python3 arc_nodsl/evaluation/evaluate_model.py \
    --autoencoder_checkpoint checkpoints/autoencoder_best.pt \
    --controller_checkpoint checkpoints/controller_best.pt \
    --beam_size 16 --num_attempts 2
```

### Expected Performance

*Performance varies significantly based on hyperparameters and training duration.*

| Dataset | Target Range | Notes |
|---------|-------------|-------|
| Training (400 tasks) | 10-30% | Seen during meta-training |
| Evaluation (400 tasks) | 5-20% | Validation set |
| Test (100 tasks) | 3-15% | Hidden test set |

**Note**: These are target ranges for a well-trained model. Actual performance depends heavily on autoencoder quality, controller training, and task difficulty distribution. Active learning may provide additional improvements over passive inference.

See [`TRAINING_README.md`](TRAINING_README.md) for comprehensive training guide.

---

## Quick Start

### Installation

```bash
pip install -e .
```

### Test Data Loading

```bash
python -c "from arc_nodsl.data.loader import ARCDataset; ds=ARCDataset('data/arc-agi_training_challenges.json'); print(f'Loaded {len(ds)} tasks')"
```

### Visualize a Task

```bash
python -m arc_nodsl.cli.visualize_task --task_id 00576224
```

### Active Learning Visualizer

Beautiful real-time terminal UI for visualizing test-time adaptation:

```bash
# Run from project root
./visualize.sh \
  -a checkpoints/autoencoder_best.pt \
  -c checkpoints/controller_best.pt \
  -t 00576224 \
  -s 20
```

**Features:**
- Real-time display of training pairs and predictions
- Live metrics (reward, loss, accuracy) with sparkline charts
- Convergence detection and test prediction
- Professional Claude CLI-inspired interface

See [`TRAINING_README.md`](TRAINING_README.md) (Active Learning Visualizer section) for full documentation.

---

## Project Structure

```
arc_nodsl/
├── data/           # ARC data loading, batching, augmentation
├── models/         # Encoder, operators, controller, renderer
├── inference/      # Latent search, patches, task embedding, constraints
├── training/       # Inner/outer loops, losses, optimizer
├── evaluation/     # Model evaluation, metrics, active solver
├── improve/        # Self-improvement: logging, mining, tuning
├── utils/          # Visualization, profiling
└── cli/            # Command-line tools

arc-visualizer/
├── src/            # TypeScript/React visualizer UI
├── python-backend/ # Python streaming backend
└── docs/           # Visualizer documentation
```

---

## Dataset

Place ARC-AGI files in `data/`:
- `arc-agi_training_challenges.json` (400 tasks)
- `arc-agi_training_solutions.json`
- `arc-agi_evaluation_challenges.json` (400 tasks)
- `arc-agi_evaluation_solutions.json`
- `arc-agi_test_challenges.json` (100 tasks, no solutions)

Download from: https://github.com/fchollet/ARC-AGI

---

## Requirements

- Python 3.9+
- PyTorch 2.0+ with CUDA support (recommended)
- 8GB+ GPU VRAM for training
- Node.js 18+ (for visualizer)

---

## Citation

Based on concepts from:
- Slot Attention (Locatello et al., 2020)
- REINFORCE (Williams, 1992)
- Reptile Meta-Learning (Nichol et al., 2018)
- ARC Challenge (Chollet, 2019)
