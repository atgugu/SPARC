# CLAUDE.md - Quick Start Guide for Claude Code Instances

*This guide is designed to help future Claude instances quickly become productive with the SPARK (Slot Programs via Active Radiation for ARC) codebase.*

---

## 🎯 Executive Summary

**SPARK** is a no-DSL ARC solver that learns latent operators on object-centric representations instead of using hand-coded transformation rules. The system:
1. Decomposes grids into "slots" (object representations) via attention
2. Learns a library of latent operators that transform these slots
3. Uses a controller to select operator sequences via reinforcement learning
4. Searches for solutions using beam search with "probability radiation" (stochastic variants)

**Current State**: Phase 5B complete with full evaluation system. The pipeline is functional from autoencoder pretraining through controller training to inference.

**Key Innovation**: No explicit DSL or hand-coded rules - everything is learned from input-output examples.

---

## 🚀 Common Commands Reference

### Quick Test Commands
```bash
# Test if environment is set up correctly
python3 -c "import torch; print(f'PyTorch: {torch.__version__}, CUDA: {torch.cuda.is_available()}')"

# Test data loading
python3 arc_nodsl/data/loader.py

# Test autoencoder forward pass
python3 arc_nodsl/models/renderer.py

# Test beam search
python3 arc_nodsl/inference/latent_search.py

# Test solver pipeline (with random weights)
python3 arc_nodsl/evaluation/solver.py
```

### Training Pipeline Commands

#### Stage 1: Autoencoder Pretraining (Target: >95% accuracy)
```bash
# Full training with augmentation (~50 epochs, RECOMMENDED)
python3 arc_nodsl/training/pretrain_autoencoder.py \
    --data_train data/arc-agi_training_challenges.json \
    --data_val data/arc-agi_evaluation_challenges.json \
    --epochs 50 \
    --batch_size 64 \
    --lr 3e-4 \
    --checkpoint_dir checkpoints \
    --augment

# Without augmentation (faster but less robust)
python3 arc_nodsl/training/pretrain_autoencoder.py \
    --epochs 50 \
    --batch_size 64 \
    --lr 3e-4

# Quick test (1 epoch, small batch)
python3 arc_nodsl/training/pretrain_autoencoder.py \
    --epochs 1 \
    --batch_size 16 \
    --eval_every 1 \
    --augment
```

#### Stage 2: Controller Training (Meta-Learning)
```bash
# Full training with augmentation (RECOMMENDED)
python3 arc_nodsl/training/train_controller.py \
    --autoencoder_checkpoint checkpoints/autoencoder_best.pt \
    --data_train data/arc-agi_training_challenges.json \
    --meta_epochs 100 \
    --meta_batch_size 4 \
    --inner_steps 10 \
    --augment \
    --augment_prob 0.5 \
    --augment_exclude_identity

# Without augmentation (faster but less robust)
python3 arc_nodsl/training/train_controller.py \
    --autoencoder_checkpoint checkpoints/autoencoder_best.pt \
    --meta_epochs 100 \
    --meta_batch_size 4 \
    --inner_steps 10

# Quick test with augmentation
python3 arc_nodsl/training/train_controller.py \
    --autoencoder_checkpoint checkpoints/autoencoder_best.pt \
    --meta_epochs 2 \
    --meta_batch_size 2 \
    --inner_steps 5 \
    --augment \
    --augment_prob 1.0
```

#### Stage 3: Evaluation

**Passive Inference** (default - uses fixed controller):
```bash
# Full evaluation on validation set
python3 arc_nodsl/evaluation/evaluate_model.py \
    --autoencoder_checkpoint checkpoints/autoencoder_best.pt \
    --controller_checkpoint checkpoints/controller_best.pt \
    --dataset data/arc-agi_evaluation_challenges.json \
    --beam_size 16 \
    --max_steps 8 \
    --num_attempts 2

# Quick test (3 tasks only)
python3 arc_nodsl/evaluation/evaluate_model.py \
    --autoencoder_checkpoint checkpoints/autoencoder_best.pt \
    --controller_checkpoint checkpoints/controller_best.pt \
    --num_tasks 3 \
    --verbose
```

**Test-Time Augmentation (TTA)** - highest accuracy, 8x slower:
```bash
# Full evaluation with TTA ensemble
python3 arc_nodsl/evaluation/evaluate_model.py \
    --autoencoder_checkpoint checkpoints/autoencoder_best.pt \
    --controller_checkpoint checkpoints/controller_best.pt \
    --dataset data/arc-agi_evaluation_challenges.json \
    --beam_size 16 \
    --max_steps 8 \
    --num_attempts 2 \
    --tta \
    --tta_mode majority

# Quick test with TTA (3 tasks)
python3 arc_nodsl/evaluation/evaluate_model.py \
    --autoencoder_checkpoint checkpoints/autoencoder_best.pt \
    --controller_checkpoint checkpoints/controller_best.pt \
    --num_tasks 3 \
    --tta \
    --verbose
```

**Active Inference** (NEW - adapts controller on each task):
```bash
# Full evaluation with active learning
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
    --time_budget 60 \
    --beam_size_adaptation 8

# Quick test with active learning
python3 arc_nodsl/evaluation/evaluate_model.py \
    --autoencoder_checkpoint checkpoints/autoencoder_best.pt \
    --controller_checkpoint checkpoints/controller_best.pt \
    --num_tasks 3 \
    --verbose \
    --active_learning \
    --adaptation_steps 10
```

### Inference on New Tasks
```bash
# Interactive solver
python3 arc_nodsl/cli/solve_task.py \
    --autoencoder checkpoints/autoencoder_best.pt \
    --controller checkpoints/controller_best.pt \
    --task_file path/to/task.json \
    --beam_size 16 \
    --visualize

# Batch prediction
python3 arc_nodsl/cli/predict_batch.py \
    --autoencoder checkpoints/autoencoder_best.pt \
    --controller checkpoints/controller_best.pt \
    --input_dir path/to/tasks \
    --output_dir predictions \
    --num_attempts 2
```

---

## 🧠 Active Learning / Active Inference (NEW)

### Concept

**Active Learning** is a paradigm shift from passive to active inference:

**Passive Inference (Traditional)**:
- Use a fixed controller trained via meta-learning
- At test time: just run beam search, no further learning
- Fast but limited adaptation to new tasks

**Active Inference (NEW)**:
- Use meta-learned controller as initialization
- At test time: **actively adapt** the controller on each task's train pairs
- Stop when: train solved OR time budget exceeded OR max steps
- Use adapted controller for test prediction

### Why Active Learning?

1. **Aligns with Meta-Learning Philosophy**: The whole point of meta-learning is to learn a good initialization for FAST adaptation. But we weren't doing adaptation at test time!

2. **Task-Specific Optimization**: Each ARC task has unique patterns. Active learning tailors the controller to each specific task.

3. **Better Performance**: Expected improvements:
   - Task Success Rate: +10-15% absolute
   - Train Solved Rate: +20-30% absolute
   - Convergence: 50-70% of tasks converge (train solved) during adaptation

4. **Computational Trade-off**: More time per task (20-60s adaptation) but significantly better accuracy.

### How It Works

```python
For each test task:
    1. Clone base controller (from meta-learning)
    2. Adaptation loop (max 20 steps or 60s):
        a. Sample train pair
        b. Run beam search (collect gradients)
        c. Compute rewards (accuracy + binary bonus)
        d. Update controller with REINFORCE
        e. Check if all train pairs solved → early stop
    3. Use adapted controller to predict test outputs
```

### Key Parameters

- `--active_learning`: Enable active inference
- `--adaptation_steps`: Max gradient steps (default: 20)
- `--adaptation_lr`: Learning rate for adaptation (default: 1e-3)
- `--time_budget`: Max time per task in seconds (default: 60)
- `--beam_size_adaptation`: Beam size during adaptation (default: 8, smaller = faster)

### When to Use

**Use Active Learning when**:
- You have time budget for inference (60s+ per task)
- You want maximum accuracy
- Tasks vary significantly in difficulty
- You have trained checkpoints via meta-learning

**Use Passive Inference when**:
- You need fast predictions (< 5s per task)
- You're doing initial testing/debugging
- You want to establish baseline performance

---

## 🔄 Data Augmentation (NEW)

### Concept

**Data Augmentation** applies spatial transformations (rotations, flips) to training data to improve model robustness and equivariance. ARC tasks often exhibit rotational and reflective symmetries, so augmentation helps the model learn these invariances.

### Available Transformations (8 total)

- **IDENTITY**: No transformation (baseline)
- **ROT_90, ROT_180, ROT_270**: Rotate 90°, 180°, 270° clockwise
- **FLIP_H, FLIP_V**: Flip horizontally (left-right) or vertically (top-bottom)
- **FLIP_D1, FLIP_D2**: Flip along main diagonal or anti-diagonal

During augmentation, the same random transformation is applied consistently to both input and output grids in a pair, preserving the input→output relationship.

### Integration Status

**Phase 1: Autoencoder Pretraining** ✓ IMPLEMENTED
- Augmentation integrated into batching pipeline
- Each training batch randomly transforms input-output pairs
- Expected improvement: +5-10% reconstruction accuracy
- Enables learning of rotation/flip equivariant representations

**Phase 2: Controller Training** ✓ IMPLEMENTED
- Augments entire tasks during meta-learning
- Same transform applied to all pairs in task (preserves task pattern)
- Configurable augmentation probability (default 50%)
- Expected improvement: +5-10% task success rate

**Phase 3: Test-Time Augmentation** ✓ IMPLEMENTED
- Ensemble predictions across all 8 transforms at evaluation time
- Majority voting for robust pixel-wise consensus
- Expected improvement: +2-5% accuracy
- Trade-off: 8x slower inference (5s → 40s per task)

### Color Augmentation (NEW)

In addition to spatial transformations, SPARK now supports **color augmentation** via bijective permutations of the ARC color palette (0-10). This provides massive data diversity without changing visual structure.

**Key Features:**
- **Bijective Mapping**: One-to-one color permutations (11! = 39,916,800 possible)
- **Preserves Structure**: Visual patterns remain intact, only colors change
- **ARC Palette Only**: Uses only the 11 ARC colors (0-10), no new colors introduced
- **Combinable**: Can combine with spatial augmentation for maximum diversity

**Benefits:**
- Massive data diversity (multiplier=50 creates 50 color variants per task)
- Color-invariant representations (model learns patterns, not specific colors)
- Better generalization to unseen color combinations
- Minimal computational overhead during training

### Using Augmentation

**Phase 1: Autoencoder training** (pair-level augmentation):
```bash
# Spatial only
python3 arc_nodsl/training/pretrain_autoencoder.py \
    --epochs 50 \
    --batch_size 64 \
    --augment

# Spatial + Color
python3 arc_nodsl/training/pretrain_autoencoder.py \
    --epochs 50 \
    --batch_size 64 \
    --augment \
    --color_augment_prob 0.3  # 30% chance of color permutation per pair
```

**Phase 2: Controller training** (task-level augmentation):
```bash
# Spatial only
python3 arc_nodsl/training/train_controller.py \
    --autoencoder_checkpoint checkpoints/autoencoder_best.pt \
    --meta_epochs 100 \
    --augment \
    --augment_prob 0.5 \
    --augment_exclude_identity

# Color augmentation with multiplier (MASSIVE DIVERSITY)
python3 arc_nodsl/training/train_controller.py \
    --autoencoder_checkpoint checkpoints/autoencoder_best.pt \
    --meta_epochs 100 \
    --color_aug_multiplier 50  # Each task generates 50 variants (random spatial+color)
    # Note: multiplier=50 means 51 total versions: original + 50 augmented

# Combined: Spatial + Color
python3 arc_nodsl/training/train_controller.py \
    --autoencoder_checkpoint checkpoints/autoencoder_best.pt \
    --meta_epochs 100 \
    --augment \
    --augment_prob 0.5 \
    --color_aug_multiplier 10  # More conservative multiplier with spatial aug
```

**Phase 3: Test-Time Augmentation** (color TTA):
```bash
# Spatial TTA only (8 variants)
python3 arc_nodsl/evaluation/evaluate_model.py \
    --autoencoder_checkpoint checkpoints/autoencoder_best.pt \
    --controller_checkpoint checkpoints/controller_best.pt \
    --tta \
    --tta_mode majority

# Spatial + Color TTA (8 × N variants, very expensive but most accurate)
python3 arc_nodsl/evaluation/evaluate_model.py \
    --autoencoder_checkpoint checkpoints/autoencoder_best.pt \
    --controller_checkpoint checkpoints/controller_best.pt \
    --tta \
    --tta_color \
    --tta_color_variants 4  # 8 spatial × 4 color = 32 predictions
    # WARNING: 32x slower than baseline (5s → 2.5 minutes per task)
```

**Without augmentation** (faster, but less robust):
```bash
# Autoencoder: No --augment flag
python3 arc_nodsl/training/pretrain_autoencoder.py --epochs 50

# Controller: No --augment flag
python3 arc_nodsl/training/train_controller.py --autoencoder_checkpoint checkpoints/autoencoder_best.pt
```

### Expected Benefits

1. **Better Generalization**: Model learns to handle rotated/flipped variants
2. **Data Efficiency**: 8x more training variations from same data
3. **Robustness**: Reduces overfitting to specific orientations
4. **Equivariance**: Encourages slot representations to be orientation-agnostic

### Implementation Details

**Phase 1 (Autoencoder):**
- Augmentation is applied at **collation time** in `batching.py`
- Each batch gets random transformations (different per pair)
- Validation data is **not augmented** (for stable metrics)

**Phase 2 (Controller):**
- Augmentation is applied at **task sampling time** in `train_controller.py`
- Same transform applied to ALL pairs in a task (preserves pattern)
- Probabilistic augmentation: default 50% of tasks are augmented
- Can exclude identity transform with `--augment_exclude_identity`

**Both phases:**
- Shapes are automatically updated for 90°/270° rotations (h,w) → (w,h)
- Original grid semantics are preserved (color values unchanged)

### When to Use

**Use augmentation when**:
- Training autoencoder from scratch (recommended)
- Tasks exhibit rotational/reflective patterns
- You want maximum generalization
- Training time is not a critical constraint

**Skip augmentation when**:
- Doing quick tests/debugging (faster iteration)
- Fine-tuning on specific task orientations
- Computational budget is very limited

---

## 🔄 Test-Time Augmentation (Phase 3)

### Concept

**Test-Time Augmentation (TTA)** ensembles predictions across all 8 spatial transforms at test time to improve robustness and accuracy. For each test input, the model generates predictions from 8 different orientations and aggregates them via majority voting.

### Algorithm

For each test input:
1. **Transform** input with all 8 transforms (identity, rotations, flips)
2. **Inference** on each transformed input → 8 predictions
3. **Inverse transform** predictions back to original space
4. **Ensemble** via majority voting (pixel-wise) → final prediction

### Ensemble Strategies

**Majority Voting** (default, recommended):
- For each pixel, select most common value across 8 predictions
- Robust to individual transform failures
- No model confidence required
- Simple and interpretable

**First Success** (when target available):
- Return first prediction that exactly matches target
- Fallback to majority if no exact match
- Only useful during validation (target available)

### When to Use TTA

**Use TTA when**:
- Maximum accuracy is priority
- Computational budget allows (8x slower)
- Task has rotational/reflective symmetry
- Final competition submission
- Need robust predictions

**Skip TTA when**:
- Fast inference needed (< 10s per task)
- Initial baseline evaluation
- Debugging/development
- Tasks known to have specific orientation
- Real-time constraints

### Computational Cost

| Mode | Inference Time/Task | Accuracy | Use Case |
|------|---------------------|----------|----------|
| Passive | ~5s | Baseline | Development |
| TTA | ~40s | +2-5% | Competition |
| Active | ~60s | +10-15% | Adaptation |
| Active+TTA* | ~8 min | +15-20% | Best (if time allows) |

*Note: Active+TTA is not currently supported (conflict). Choose one.

### Usage Examples

**Enable TTA for maximum accuracy**:
```bash
python3 arc_nodsl/evaluation/evaluate_model.py \
    --autoencoder_checkpoint checkpoints/autoencoder_best.pt \
    --controller_checkpoint checkpoints/controller_best.pt \
    --tta \
    --tta_mode majority
```

**Without TTA (faster baseline)**:
```bash
python3 arc_nodsl/evaluation/evaluate_model.py \
    --autoencoder_checkpoint checkpoints/autoencoder_best.pt \
    --controller_checkpoint checkpoints/controller_best.pt
    # No --tta flag
```

### Implementation Details

- **Train pairs**: NO TTA (speed priority for gating)
- **Test pairs**: YES TTA (accuracy priority)
- **Ensemble method**: Pixel-wise majority voting
- **Shape handling**: Automatically adjusts for rotated dimensions
- **Multi-attempt**: First attempt uses TTA, others use beam diversity
- **Conflict with Active**: Cannot use both TTA and active_learning simultaneously

### Expected Benefits

1. **Robustness**: Eliminates orientation-specific failures
2. **Accuracy**: +2-5% task success rate improvement
3. **Consensus**: High-confidence predictions through voting
4. **Symmetry exploitation**: Leverages rotational/reflective patterns

---

## 🏗️ Architecture Overview

### Core Pipeline: Slots → Operators → Controller → Beam Search

```
Input Grid [30×30]
    ↓
SlotEncoder (attention-based decomposition)
    ↓
Slots: K=8 objects × D=128 features
    ├─ z: feature vectors [K, D]
    ├─ m: attention masks [K, H, W]
    └─ p: centroids [K, 2]
    ↓
Controller (policy network)
    ↓
Operator Sequence: [op₁, op₂, ..., opₜ]
    ↓
OperatorLibrary (learned transformations)
    ├─ Geometry: translate, rotate, flip, scale
    ├─ Mask: dilate, erode, outline
    └─ Color: remap palette
    ↓
Modified Slots
    ↓
SlotRenderer (alpha compositing)
    ↓
Output Grid [30×30]
```

### Key Architectural Decisions

1. **Fixed 30×30 Padding**: All grids padded to 30×30 for batch processing. Original shapes preserved for cropping. Trade-off: memory vs flexibility.

2. **8 Slots × 128 Dims**: Balance between object capacity (8 is enough for most ARC tasks) and memory usage. Expandable if needed.

3. **Latent Space Search**: Beam search operates on slot representations, not grid pixels. This enables:
   - Smooth interpolation between states
   - Gradient-based optimization
   - Compositional reasoning

4. **Train-First Gating**: Only attempt test prediction if ALL train pairs are solved perfectly. Prevents hallucination on misunderstood patterns.

5. **Meta-Learning via Reptile**: Simpler than MAML (no second-order gradients), but effective for fast adaptation. Updates: θ ← θ + α·mean(θ' - θ).

6. **Support/Query Split**: Hold out 1 train pair as "query" to mimic train→test generalization. Ensures we're learning the pattern, not memorizing.

7. **Binary Task Success**: +0.5 bonus for 100% pixel-exact match encourages perfect solutions over "close enough" approximations.

8. **Multi-Attempt Evaluation**: Generate K=2 attempts per test output (competition rules). Score = 1 if ANY attempt is correct.

---

## 📁 Essential Files to Read

### Priority 1: Understand the Pipeline
1. **STEPS_DESC.md** - Complete walkthrough of all training/evaluation stages
2. **arc_nodsl/inference/latent_search.py** - Beam search algorithm (core of the system)
3. **arc_nodsl/training/inner_loop.py** - REINFORCE implementation with support/query split

### Priority 2: Understand the Models
4. **arc_nodsl/models/operators.py** - How latent operators transform slots
5. **arc_nodsl/models/controller.py** - Policy network architecture
6. **arc_nodsl/models/slots.py** - Slot attention mechanism

### Priority 3: Understand Task Handling
7. **arc_nodsl/inference/task_embed.py** - How patterns are extracted from train pairs
8. **arc_nodsl/inference/constraints.py** - Constraint extraction and filtering
9. **arc_nodsl/evaluation/metrics.py** - Evaluation metrics and task success definition

---

## 🔍 Key Code Patterns

### Pattern 1: Task Data Structure
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

### Pattern 2: Beam Search Candidate
```python
@dataclass
class Candidate:
    slots_z: torch.Tensor      # [K, D] features
    slots_m: torch.Tensor      # [K, H, W] masks
    slots_p: torch.Tensor      # [K, 2] positions
    operator_seq: List[int]    # Applied operators
    param_seq: List[Tensor]    # Operator parameters
    score: float               # Patch similarity
    log_prob: float           # For REINFORCE
    prediction: torch.Tensor   # [H, W] output grid
```

### Pattern 3: REINFORCE Reward Computation
```python
# Fuzzy reward: weighted accuracy + constraints
fuzzy_reward = 0.7 * pixel_accuracy + 0.3 * constraint_score

# Binary bonus: exact match gets extra reward
binary_bonus = 0.5 if exact_match(pred, target) else 0.0

# Total reward for REINFORCE
total_reward = fuzzy_reward + binary_bonus
```

### Pattern 4: Meta-Learning Update (Reptile)
```python
# Clone base model
theta_clone = copy.deepcopy(theta_base)

# Adapt on task (inner loop)
for step in range(inner_steps):
    loss = compute_loss(theta_clone, task)
    theta_clone.backward()
    optimizer.step()

# Compute meta-gradient
meta_grad = (theta_clone - theta_base) / inner_steps

# Meta-update
theta_base += meta_lr * mean([meta_grad for each task])
```

---

## ⚠️ Common Issues & Solutions

### Issue 1: PyTorch Autocast API Change
**Error**: `TypeError: autocast.__init__() got an unexpected keyword argument 'device_type'`

**Fix**: Update autocast usage in pretrain_autoencoder.py
```python
# Old (PyTorch ≥1.10)
with autocast(device_type='cuda', dtype=torch.float16):

# New (PyTorch <1.10)
from torch.cuda.amp import autocast
with autocast():
```

### Issue 2: CUDA Out of Memory
**Fix**: Reduce batch size or beam size
```bash
# Reduce batch size
--batch_size 32  # Instead of 64

# Reduce beam size
--beam_size 8    # Instead of 16
```

### Issue 3: Import Errors
**Fix**: Check sys.path insertions at top of files
```python
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))
```

### Issue 4: Checkpoint Loading Errors
**Fix**: Ensure checkpoint has expected keys
```python
checkpoint = torch.load(path, map_location=device)
# Check keys: 'model_state_dict', 'optimizer_state_dict', 'epoch', etc.
```

---

## 📊 Expected Performance

### After Stage 1 (Autoencoder):
- Training reconstruction: >98% accuracy
- Validation reconstruction: >95% accuracy
- Clear object segmentation in attention masks

### After Stage 2 (Controller):
- Training reward: 0.6-0.8
- Test reward: 0.4-0.6
- Task success rate: 10-30% (binary)
- Query solved rate: 20-40% (with support/query split)

### After Stage 3 (Evaluation):
- Validation task success: 10-30%
- Train solved rate: 30-50%
- Test accuracy given train: 30-60%
- Competition score: 15-35%

*Note: With random weights, expect 0% success. These are targets after proper training.*

---

## 🔧 Making Modifications

### Adding New Operators
Edit `arc_nodsl/models/operators.py`:
```python
class LatentOp(nn.Module):
    def forward(self, slots_z, slots_m, slots_p, params):
        # Add new transformation logic
        # Must return (z_new, m_new, p_new)
```

### Adding New Constraints
Edit `arc_nodsl/inference/constraints.py`:
```python
class MyConstraint(Constraint):
    def check(self, prediction, target_shape):
        # Return True if valid

    def score(self, prediction, target):
        # Return bonus score [0, 1]
```

### Modifying Reward Function
Edit `arc_nodsl/training/losses.py`:
```python
class ReinforceReward:
    def compute_reward(self, prediction, target, constraints):
        # Modify reward computation
        # Must return scalar in [0, 1+bonus]
```

### Changing Search Strategy
Edit `arc_nodsl/inference/latent_search.py`:
```python
def beam_search(...):
    # Modify beam expansion logic
    # Modify diversity selection
    # Modify probability radiation
```

---

## 📝 Recent Changes (Phase 5B+)

1. **Support/Query Split**: Training now holds out 1 train pair to mimic test generalization
2. **Binary Task Success**: Added +0.5 bonus for exact matches
3. **Multi-Attempt Support**: Generate K=2 attempts per test (competition rules)
4. **Competition Scoring**: Average of binary per-output scores
5. **Optimizer Change**: Switched from Adam to AdamW in inner_loop.py
6. **New Metrics**: Added task_success_rate, query_solved_rate, generalization_rate
7. **Active Learning**: Added test-time adaptation with ActiveARCSolver
8. **Data Augmentation Phase 1**: Integrated pair-level augmentation for autoencoder training
9. **Data Augmentation Phase 2**: Integrated task-level augmentation for controller meta-learning
10. **Data Augmentation Phase 3**: Implemented test-time augmentation ensemble with TTASolver and majority voting
11. **Color Augmentation (NEW)**: Bijective color permutations across all 3 phases:
    - Phase 1: `--color_augment_prob` for pair-level color augmentation
    - Phase 2: `--color_aug_multiplier N` creates N additional color variants per task (supports multiplier=50 for massive diversity!)
    - Phase 3: `--tta_color --tta_color_variants N` for color TTA ensemble (8×N predictions)
    - Each variant combines random spatial transform + random color permutation for maximum diversity
    - Uses only ARC's 11 colors (0-10), bijective mapping preserves structure

---

## 💡 Tips for Productivity

1. **Use Task Tool for Complex Searches**: When searching across multiple files, use the Task tool with general-purpose agent rather than multiple Grep calls.

2. **Check Background Processes**: Use BashOutput to monitor long-running training jobs.

3. **Read Docstrings**: Most functions have detailed docstrings explaining parameters and return values.

4. **Test Individual Components**: Each module has test code in `if __name__ == "__main__"` blocks.

5. **Watch Memory Usage**: ARC grids are small but beam search can consume significant memory with large beams.

6. **Understand the Metric**: Task is "solved" only if ALL outputs match exactly (100% pixels). No partial credit in competition.

---

## 🎯 Quick Task Checklist

When implementing a new feature:
- [ ] Read STEPS_DESC.md for pipeline context
- [ ] Identify which module to modify
- [ ] Check if similar functionality exists
- [ ] Write test code in `if __name__ == "__main__"`
- [ ] Test on single task before full dataset
- [ ] Update metrics if evaluation changes
- [ ] Document changes in code comments

When debugging:
- [ ] Check tensor shapes with print/assert
- [ ] Verify model is in correct mode (train/eval)
- [ ] Check device placement (CPU/CUDA)
- [ ] Monitor gradient flow if training
- [ ] Use verbose=True flags for detailed output
- [ ] Test with small batch/beam sizes first

---

## 📚 References

- **ARC-AGI Dataset**: https://github.com/fchollet/ARC-AGI
- **Slot Attention**: "Object-Centric Learning with Slot Attention" (Locatello et al.)
- **Reptile**: "On First-Order Meta-Learning Algorithms" (Nichol et al.)
- **REINFORCE**: "Simple Statistical Gradient-Following Algorithms" (Williams)

---

*This guide is a living document. Update it when making significant architectural changes or discovering important patterns.*

**Last Updated**: Phase 5B+ (Active Learning + Data Augmentation Complete + Color Augmentation Complete)
**Maintainer Note**: Full augmentation pipeline complete across all phases:
- **Spatial Augmentation**: Phase 1 (pair-level), Phase 2 (task-level), Phase 3 (TTA ensemble)
- **Color Augmentation**: Phase 1 (pair-level with `--color_augment_prob`), Phase 2 (task-level with `--color_aug_multiplier`, supports 50x!), Phase 3 (color TTA with `--tta_color`)
- All augmentation features are production-ready and support combined spatial+color for maximum diversity.