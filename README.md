# SPARC: Slot Programs via Active Radiation for ARC

An experimental approach to [ARC-AGI](https://github.com/fchollet/ARC-AGI) that explores learning compositional operators end-to-end rather than hand-writing transformation rules.

<p align="center">
  <img src="visualizer.png" alt="SPARC Active Learning Visualizer" width="800">
  <br>
  <em>Real-time terminal UI showing active learning adaptation on an ARC task</em>
</p>

---

## Overview

SPARC explores whether abstract reasoning patterns can emerge from learned compositional transformations on structured, object-centric representations. Instead of encoding hundreds of hand-crafted primitives in a domain-specific language (DSL), it attempts to learn a small library of differentiable operators that act on slot-based object representations extracted via attention.

This is a research experiment, not a competitive solver. The approach is intentionally minimal: a small operator library, simple slot attention, and standard meta-learning. The goal is to investigate the feasibility of the idea, not to maximize leaderboard scores.

## Approach

1. **Decompose** grids into "slots" (object representations) using slot attention
2. **Learn** a library of latent operators that transform these slots (geometry, mask morphology, color)
3. **Train** a controller via meta-learning (REINFORCE + Reptile) to compose operator sequences
4. **Search** for solutions using beam search with stochastic variants ("probability radiation")

## Architecture

```
Input Grid [30x30]
    |
SlotEncoder (attention-based decomposition)
    |
Slots: K=8 objects x D=128 features
    |-- z: feature vectors [K, D]
    |-- m: attention masks [K, H, W]
    +-- p: centroids [K, 2]
    |
Controller (policy network, conditioned on task embedding)
    |
Operator Sequence: [op_1, op_2, ..., op_T]
    |
OperatorLibrary (learned transformations)
    |-- Geometry: translate, rotate, flip, scale
    |-- Mask: dilate, erode, outline
    +-- Color: remap palette
    |
SlotRenderer (alpha compositing)
    |
Output Grid [30x30]
```

**Encoder**: Grid (HxW) -> K slots `{(z_i, m_i, p_i)}` via iterative slot attention.

**Operators**: Small networks that edit `(z, m, p)` tuples -- geometry, mask morphology, and color operations.

**Controller**: Policy network that selects operator sequences conditioned on slots and a task embedding extracted from training pairs.

**Renderer**: Decodes modified slots back to a discrete color grid via alpha compositing.

---

## Getting Started

### Requirements

- Python 3.9+
- PyTorch 2.0+ (CUDA recommended)
- 8GB+ GPU VRAM for training
- Node.js 18+ (for the visualizer only)

### Install

```bash
pip install -e .
```

### Dataset

Download ARC-AGI data from [github.com/fchollet/ARC-AGI](https://github.com/fchollet/ARC-AGI) and place in `data/`:

```
data/
  arc-agi_training_challenges.json
  arc-agi_training_solutions.json
  arc-agi_evaluation_challenges.json
  arc-agi_evaluation_solutions.json
  arc-agi_test_challenges.json
```

### Quick Test

```bash
# Check environment
python3 -c "import torch; print(f'PyTorch: {torch.__version__}, CUDA: {torch.cuda.is_available()}')"

# Test autoencoder forward pass
python3 arc_nodsl/models/renderer.py

# Test beam search
python3 arc_nodsl/inference/latent_search.py

# Test solver pipeline (random weights)
python3 arc_nodsl/evaluation/solver.py
```

---

## Training

Training has three phases:

```bash
# Phase 1: Pretrain autoencoder
python3 arc_nodsl/training/pretrain_autoencoder.py \
    --epochs 50 --batch_size 32 --augment

# Phase 2: Meta-learn controller
python3 arc_nodsl/training/train_controller.py \
    --autoencoder_checkpoint checkpoints/autoencoder_best.pt \
    --meta_epochs 100 --augment

# Phase 3: Evaluate
python3 arc_nodsl/evaluation/evaluate_model.py \
    --autoencoder_checkpoint checkpoints/autoencoder_best.pt \
    --controller_checkpoint checkpoints/controller_best.pt \
    --beam_size 16 --num_attempts 2
```

See [`docs/training.md`](docs/training.md) for the full training guide including augmentation, active learning, and test-time augmentation options. See [`docs/pipeline.md`](docs/pipeline.md) for a detailed walkthrough of each pipeline stage.

---

## Visualizer

<p align="center">
  <img src="test_task.png" alt="ARC task visualization" width="600">
</p>

A terminal-based visualizer shows active learning adaptation in real time:

```bash
./visualize.sh \
  -a checkpoints/autoencoder_best.pt \
  -c checkpoints/controller_best.pt \
  -t 00576224 \
  -s 20
```

See [`arc-visualizer/README.md`](arc-visualizer/README.md) for setup and usage.

---

## Project Structure

```
arc_nodsl/
  models/         # Encoder, operators, controller, renderer
  inference/      # Beam search, task embedding, constraints
  training/       # Inner/outer loops, losses, pretraining
  evaluation/     # Model evaluation, metrics, active solver
  utils/          # Visualization, profiling
  cli/            # Command-line tools

arc-visualizer/   # TypeScript/React visualizer + Python backend
docs/             # Training guide, pipeline walkthrough
tests/            # Integration tests
```

---

## References

- Slot Attention: Locatello et al., "Object-Centric Learning with Slot Attention" (2020)
- REINFORCE: Williams, "Simple Statistical Gradient-Following Algorithms" (1992)
- Reptile: Nichol et al., "On First-Order Meta-Learning Algorithms" (2018)
- ARC: Chollet, "On the Measure of Intelligence" (2019)

## License

[MIT](LICENSE)
