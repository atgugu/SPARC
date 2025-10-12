# SPARK: Slot Programs via Active Radiation for ARC

A no-DSL, self-discovering ARC solver that learns latent operators via neural networks and composes them to solve tasks.

## Architecture Overview

**Core Idea**: Replace hand-written rules with learned latent operators that act on object-centric representations (slots + masks). Training pairs supervise only the final output; the system discovers reusable operators through loss + sparsity/MDL pressure.

### Key Components

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

## Project Structure

```
arc_nodsl/
├── data/           # ARC data loading, batching, augmentation
├── models/         # Encoder, operators, controller, renderer
├── inference/      # Latent search, patches, task embedding, constraints
├── training/       # Inner/outer loops, losses, optimizer
├── improve/        # Self-improvement: logging, mining, tuning
├── utils/          # Visualization, profiling
└── cli/            # Command-line tools
```

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

Features:
- Real-time display of training pairs and predictions
- Live metrics (reward, loss, accuracy) with sparkline charts
- Convergence detection and test prediction
- Professional Claude CLI-inspired interface

See `TRAINING_README.md` (Active Learning Visualizer section) for full documentation.

## Development Phases

- **Phase 1** (Week 1): Data infrastructure ✓
- **Phase 2** (Weeks 2-3): Encoder/renderer + operators
- **Phase 3** (Weeks 3-4): Latent search engine
- **Phase 4** (Weeks 4-5): Training loops
- **Phase 5** (Week 6): OT factor + GPU optimization
- **Phase 6** (Week 7+): Self-improvement loop
- **Phase 7** (Week 8): Production & evaluation

## Dataset

Place ARC-AGI files in `data/`:
- `arc-agi_training_challenges.json` (1000 tasks)
- `arc-agi_training_solutions.json`
- `arc-agi_evaluation_challenges.json` (120 tasks)
- `arc-agi_evaluation_solutions.json`
- `arc-agi_test_challenges.json` (240 tasks)

## Requirements

- Python 3.9+
- PyTorch 2.0+ with CUDA support (recommended)
- 8GB+ GPU VRAM for training

## Citation

Based on concepts from:
- Slot Attention (Locatello et al., 2020)
- Program Synthesis for ARC
- Optimal Transport for visual reasoning
