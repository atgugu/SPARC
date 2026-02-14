# CLAUDE.md - Quick Start for Claude Code Instances

## Summary

**SPARC** is an experimental ARC solver that learns latent operators on object-centric representations instead of using hand-coded transformation rules. The system:
1. Decomposes grids into "slots" (object representations) via attention
2. Learns a library of latent operators that transform these slots
3. Uses a controller to select operator sequences via reinforcement learning
4. Searches for solutions using beam search with stochastic variants

The pipeline runs: autoencoder pretraining -> controller meta-learning -> inference.

---

## Common Commands

### Quick Tests
```bash
python3 -c "import torch; print(f'PyTorch: {torch.__version__}, CUDA: {torch.cuda.is_available()}')"
python3 arc_nodsl/models/renderer.py      # Autoencoder forward pass
python3 arc_nodsl/inference/latent_search.py  # Beam search
python3 arc_nodsl/evaluation/solver.py     # Solver pipeline (random weights)
```

### Training
```bash
# Phase 1: Pretrain autoencoder
python3 arc_nodsl/training/pretrain_autoencoder.py \
    --epochs 50 --batch_size 64 --lr 3e-4 --augment

# Phase 2: Meta-learn controller
python3 arc_nodsl/training/train_controller.py \
    --autoencoder_checkpoint checkpoints/autoencoder_best.pt \
    --meta_epochs 100 --meta_batch_size 4 --inner_steps 10 --augment

# Phase 3: Evaluate
python3 arc_nodsl/evaluation/evaluate_model.py \
    --autoencoder_checkpoint checkpoints/autoencoder_best.pt \
    --controller_checkpoint checkpoints/controller_best.pt \
    --beam_size 16 --num_attempts 2
```

See `docs/training.md` for full options (augmentation, active learning, TTA, color augmentation).

---

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
Controller (policy network)
    |
Operator Sequence: [op_1, ..., op_T]
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

### Key Decisions

1. **Fixed 30x30 padding** for batch processing. Original shapes preserved for cropping.
2. **8 slots x 128 dims**: enough capacity for most ARC tasks.
3. **Latent space search**: beam search on slot representations, not pixel space.
4. **Train-first gating**: only predict test if ALL train pairs solved exactly.
5. **Reptile meta-learning**: simpler than MAML (no second-order gradients).
6. **Support/query split**: hold out 1 train pair to mimic test generalization.
7. **Binary task success**: +0.5 bonus for exact pixel match.
8. **Multi-attempt**: K=2 attempts per test output (competition rules).

---

## Essential Files

### Pipeline
1. `docs/pipeline.md` - Complete walkthrough of all stages
2. `arc_nodsl/inference/latent_search.py` - Beam search (core algorithm)
3. `arc_nodsl/training/inner_loop.py` - REINFORCE with support/query split

### Models
4. `arc_nodsl/models/operators.py` - Latent operator transformations
5. `arc_nodsl/models/controller.py` - Policy network
6. `arc_nodsl/models/slots.py` - Slot attention

### Task Handling
7. `arc_nodsl/inference/task_embed.py` - Pattern extraction from train pairs
8. `arc_nodsl/inference/constraints.py` - Constraint extraction and filtering
9. `arc_nodsl/evaluation/metrics.py` - Evaluation metrics

---

## Key Code Patterns

### Task Data Structure
```python
task_data = {
    'task_id': str,
    'train_inputs': List[torch.Tensor],   # [30, 30] grids
    'train_outputs': List[torch.Tensor],
    'train_shapes': List[{'input': (h,w), 'output': (h,w)}],
    'test_inputs': List[torch.Tensor],
    'test_outputs': List[torch.Tensor],   # None for competition
    'test_shapes': List[dict]
}
```

### Beam Search Candidate
```python
@dataclass
class Candidate:
    slots_z: torch.Tensor      # [K, D] features
    slots_m: torch.Tensor      # [K, H, W] masks
    slots_p: torch.Tensor      # [K, 2] positions
    operator_seq: List[int]    # Applied operators
    param_seq: List[Tensor]    # Operator parameters
    score: float               # Patch similarity
    log_prob: float            # For REINFORCE
    prediction: torch.Tensor   # [H, W] output grid
```

---

## Common Issues

### PyTorch Autocast API
```python
# If you get TypeError about 'device_type':
from torch.cuda.amp import autocast
with autocast():  # instead of autocast(device_type='cuda')
```

### CUDA Out of Memory
Reduce `--batch_size` (32 instead of 64) or `--beam_size` (8 instead of 16).

### Import Errors
Check `sys.path` insertions at top of files:
```python
sys.path.insert(0, str(Path(__file__).parent.parent))
```

### Checkpoint Loading
Ensure checkpoint has expected keys: `model_state_dict`, `optimizer_state_dict`, `epoch`.

---

## Making Modifications

### Adding Operators
Edit `arc_nodsl/models/operators.py` - must return `(z_new, m_new, p_new)`.

### Adding Constraints
Edit `arc_nodsl/inference/constraints.py` - implement `check()` and `score()`.

### Modifying Rewards
Edit `arc_nodsl/training/losses.py` - reward must be scalar in `[0, 1+bonus]`.

### Changing Search
Edit `arc_nodsl/inference/latent_search.py` - beam expansion, diversity, radiation.

---

## Task Checklist

**New feature:**
- [ ] Read `docs/pipeline.md` for context
- [ ] Check if similar functionality exists
- [ ] Test on single task before full dataset

**Debugging:**
- [ ] Check tensor shapes with print/assert
- [ ] Verify model mode (train/eval) and device (CPU/CUDA)
- [ ] Use `--verbose` flags and small batch/beam sizes
