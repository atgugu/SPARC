# Complete Pipeline: Training to Inference

This document provides a comprehensive explanation of each step in the ARC solver pipeline, from raw grids to competition submissions.

---

## **Phase 1: Autoencoder Pretraining**

### **What It Does**:
Learns to decompose ARC grids into object-centric representations (slots) and reconstruct them.

### **Input**:
Raw ARC grids (30×30 integer arrays, values 0-9 representing colors)

### **Process**:

#### **1. SlotEncoder** (`arc_nodsl/models/slots.py`):
```
Input grid [30, 30]
→ Color embedding [30, 30, 16]
→ CNN features [30, 30, 64]
→ Slot Attention (K=8 iterations)
→ 8 object slots [8, 128]
```

**Purpose**: Discovers objects in the grid automatically
- Each slot represents one object (shape, color, position)
- Slot Attention learns to route features to appropriate slots
- No supervision - learns purely from reconstruction

#### **2. SlotRenderer** (`arc_nodsl/models/renderer.py`):
```
8 slots [8, 128]
→ MLP decoder per slot [8, 30, 30, 11]
→ Softmax over slots → pixel colors [30, 30]
→ Reconstructed grid
```

**Purpose**: Reconstructs the grid from slots
- Each slot generates a full-grid proposal
- Pixel-wise competition: which slot owns each pixel?
- Mask: which pixels does each slot "claim"?

#### **3. Training Loss**:
```python
reconstruction_loss = CrossEntropy(predicted, target)
# Measures: how well can we reconstruct the original grid?
```

**Success Metric**: Validation accuracy 70-80%
- Can we perfectly reconstruct held-out grids?
- This proves slots capture meaningful object structure

### **Why This Matters**:
- ARC tasks require object-level reasoning (not pixel-level)
- Slots provide a compact representation: 8×128 = 1024 numbers vs 30×30 = 900 pixels
- Operators can manipulate entire objects (move, rotate, recolor) by transforming slots

### **Output**:
`checkpoints/autoencoder_best.pt` containing:
- Trained encoder weights (frozen for controller training)
- Trained renderer weights (frozen for controller training)

### **Command**:
```bash
python3 arc_nodsl/training/pretrain_autoencoder.py \
  --data_train data/arc-agi_training_challenges.json \
  --data_val data/arc-agi_evaluation_challenges.json \
  --batch_size 32 \
  --epochs 100 \
  --lr 3e-4 \
  --eval_every 5 \
  --checkpoint_dir checkpoints
```

---

## **Phase 2: Controller Training (Meta-Learning)**

### **What It Does**:
Learns to sequence operators to solve ARC tasks via meta-learning.

### **The Meta-Learning Setup**:

#### **Outer Loop (Reptile algorithm)**:
```
For each meta-epoch:
  1. Sample batch of tasks (4 tasks)
  2. For each task:
      a. Clone base controller θ
      b. Fast adapt on task → θ'
      c. Compute difference: Δθ = θ' - θ
  3. Update base: θ ← θ + α·mean(Δθ)
```

**Purpose**: Learn a controller that can *quickly adapt* to new tasks

#### **Inner Loop (REINFORCE)**:
```
For each task:
  1. Split train pairs → support (N-1) + query (1)
  2. Build task embedding from support pairs
  3. For 10 gradient steps:
      a. Sample support pair
      b. Run beam search with current controller
      c. Compute rewards (accuracy + constraints + binary bonus)
      d. Update controller with REINFORCE
  4. Evaluate on held-out query pair
  5. Return: query_solved (binary metric)
```

### **Detailed Component Breakdown**:

#### **1. Task Embedding** (`arc_nodsl/inference/task_embed.py`):
```python
def build_task_embedding(train_pairs):
    # Extract constraints
    constraints = analyze_constraints(train_pairs)
    # - Output size: fixed or variable?
    # - Colors: subset of input colors?
    # - Blank color: always 0?
    # - Sparse: mostly background?

    # Encode I/O examples
    io_embeds = []
    for input, output in train_pairs:
        input_slots = encoder(input)   # [8, 128]
        output_slots = encoder(output) # [8, 128]
        io_embeds.append(concat(input_slots, output_slots))

    # Pool across examples
    task_embed = mean(io_embeds)  # [256]

    return {
        'task_vector': task_embed,
        'constraints': constraints
    }
```

**Purpose**: Capture "what pattern should I learn?" from train examples

#### **2. Beam Search** (`arc_nodsl/inference/latent_search.py`):
```python
def beam_search(encoder, controller, operators, renderer,
                input_grid, task_embed, beam_size=16):
    # Start: encode input
    init_slots = encoder(input_grid)  # [8, 128]
    beam = [(init_slots, score=0, ops=[])]

    for step in range(max_steps):
        # Expand beam
        candidates = []
        for slots, score, ops in beam:
            # Controller predicts operator distribution
            op_logits = controller(slots, task_embed)
            op_probs = softmax(op_logits)

            # Try top-K operators
            for op_id in top_k(op_probs, k=beam_size):
                # Apply operator
                new_slots = operators[op_id](slots)
                new_score = score + log(op_probs[op_id])
                candidates.append((new_slots, new_score, ops + [op_id]))

        # Keep best
        beam = top_k(candidates, k=beam_size, key=score)

    # Render final predictions
    predictions = []
    for slots, score, ops in beam:
        grid = renderer(slots)
        predictions.append(SearchCandidate(
            prediction=grid,
            score=score,
            operator_sequence=ops
        ))

    return sorted(predictions, key=score, reverse=True)
```

**Purpose**: Search over operator sequences to transform input → output

**Key Insight**: This is **latent space search**:
- Don't search over grids (too large: 10^900 possibilities)
- Search over slot transformations (manageable: 8^4 = 4096 sequences of length 4)

#### **3. Controller** (`arc_nodsl/models/controller.py`):
```python
class Controller(nn.Module):
    def forward(self, slots, task_embed):
        # slots: [batch, 8, 128]
        # task_embed: [batch, 256]

        # Pool slots
        slots_pooled = mean(slots, dim=1)  # [batch, 128]

        # Combine with task context
        combined = concat(slots_pooled, task_embed)  # [batch, 384]

        # Predict operator distribution
        hidden = MLP(combined)  # [batch, 256]
        op_logits = Linear(hidden)  # [batch, 8]

        return op_logits  # Which operator to apply?
```

**Purpose**: Given current state + task, predict which operator to use

#### **4. Operators** (`arc_nodsl/models/operators.py`):
```python
class OperatorLibrary(nn.Module):
    def __init__(self, num_ops=8):
        # 8 learned operators (not pre-defined!)
        self.operators = nn.ModuleList([
            SlotTransformer(d_slot=128, d_hidden=128)
            for _ in range(num_ops)
        ])

    def forward(self, slots, op_id):
        # slots: [8, 128]
        # op_id: which operator to apply

        return self.operators[op_id](slots)

class SlotTransformer(nn.Module):
    def forward(self, slots):
        # Transformer: allows slots to interact
        slots_out = TransformerEncoder(slots)
        return slots_out
```

**Purpose**: Learned transformations on slots
- Not pre-programmed (rotate, flip, etc.)
- Learned from data (whatever is useful for ARC tasks)
- Slot-to-slot interactions (e.g., "move slot A to position of slot B")

#### **5. REINFORCE Reward** (`arc_nodsl/training/losses.py`):
```python
def compute_reward(prediction, target, h, w, constraints):
    # 1. Pixel accuracy (fuzzy)
    pred_crop = prediction[:h, :w]
    target_crop = target[:h, :w]
    accuracy = (pred_crop == target_crop).float().mean()

    # 2. Constraint satisfaction
    constraint_score = constraints.score(prediction, h, w)
    # - Does output have correct size?
    # - Does output respect color constraints?
    # - Is output appropriately sparse?

    # 3. Binary bonus (Phase 5B)
    binary_bonus = 0.0
    if exact_match(prediction, target, h, w):
        binary_bonus = 0.5  # Bonus for 100% correct

    # Final reward
    return 0.7 * accuracy + 0.3 * constraint_score + binary_bonus
```

**Purpose**: Guide learning toward correct solutions
- Fuzzy component (0.7×accuracy): guides search in right direction
- Constraint component (0.3×constraint): respects task rules
- Binary bonus (+0.5): extra reward for perfect solutions

#### **6. REINFORCE Update** (`arc_nodsl/training/losses.py`):
```python
def compute_loss(log_probs, rewards):
    # log_probs: log P(action_t | state_t)
    # rewards: reward for each beam candidate

    # Baseline (reduce variance)
    baseline = running_mean(rewards)
    advantages = rewards - baseline

    # Policy gradient
    policy_loss = -(log_probs * advantages).mean()

    # Entropy regularization (encourage exploration)
    entropy_bonus = entropy(log_probs).mean()

    return policy_loss - 0.01 * entropy_bonus
```

**Purpose**: Update controller to increase probability of high-reward sequences

**Why REINFORCE?**
- Credit assignment: which operator in the sequence was good?
- Policy gradient: increase P(good sequences), decrease P(bad sequences)
- Works with discrete actions (operator selection)

### **Training Dynamics**:

#### **Early Training** (meta-epochs 1-50):
- Random operator sequences
- Low rewards (~0.1-0.3)
- Controller learns basic patterns: "output usually smaller than input", "preserve certain colors"

#### **Mid Training** (meta-epochs 50-200):
- Some tasks solved (~5-15%)
- Operators specialize: some do spatial transforms, some do color changes
- Controller learns when to use which operator

#### **Late Training** (meta-epochs 200-500):
- Consistent performance (~15-30% tasks solved)
- Good generalization: query pairs often solved when support pairs are
- Binary bonus kicks in: controller learns to aim for perfection

### **Output**:
`checkpoints/controller_best.pt` containing:
- Controller weights (operator selection policy)
- Operator library weights (8 learned transformations)

### **Command**:
```bash
python3 arc_nodsl/training/train_controller.py \
  --autoencoder_checkpoint checkpoints/autoencoder_best.pt \
  --data data/arc-agi_training_challenges.json \
  --meta_epochs 500 \
  --meta_batch_size 4 \
  --inner_steps 10 \
  --learning_rate 1e-3 \
  --beam_size 8 \
  --max_operator_steps 4 \
  --checkpoint_dir checkpoints \
  --use_train_query_split \
  --binary_bonus_weight 0.5
```

---

## **Phase 3: Evaluation on Validation Set**

### **What It Does**:
Measures performance on held-out tasks using competition-aligned metrics.

### **Process**:

#### **1. Train-First Gating** (`arc_nodsl/evaluation/solver.py`):
```python
def solve_task(task_data):
    # Step 1: Attempt train pairs
    train_predictions = []
    for train_input in task_data['train_inputs']:
        pred = beam_search(train_input, task_embed, beam_size=16)
        train_predictions.append(pred)

    # Step 2: Check if ALL train pairs perfect
    train_solved = all(
        exact_match(pred, target)
        for pred, target in zip(train_predictions, train_outputs)
    )

    # Step 3: Gate decision
    if not train_solved:
        return TaskResult(
            train_solved=False,
            test_attempted=False,
            task_success=False
        )

    # Step 4: Predict test (only if train solved)
    test_attempts = []  # K=2 attempts per test
    for test_input in task_data['test_inputs']:
        # Generate top-2 diverse predictions
        candidates = beam_search(test_input, beam_size=16)
        attempts = [candidates[0].prediction, candidates[1].prediction]
        test_attempts.append(attempts)

    # Step 5: Competition scoring
    per_output_scores = []
    for attempts, target in zip(test_attempts, test_outputs):
        # Score = 1 if ANY attempt matches
        any_correct = any(exact_match(a, target) for a in attempts)
        per_output_scores.append(1.0 if any_correct else 0.0)

    competition_score = mean(per_output_scores)

    return TaskResult(
        train_solved=True,
        test_attempted=True,
        task_success=all(per_output_scores),
        competition_score=competition_score
    )
```

**Why Gating?**
- No point predicting test if we can't solve train
- Saves computation
- Improves score (don't submit wrong guesses)

#### **2. Multi-Attempt Evaluation**:
```
For each test input:
  Generate K=2 attempts (top-2 beam candidates)

For each test output:
  Check if ANY of K attempts is 100% correct
  Score = 1 if yes, 0 if no

Task competition score = average across test outputs
Overall competition score = average across all tasks
```

**Why Multi-Attempt?**
- Competition allows 2 attempts per output
- Increases coverage: might get one attempt right
- Typical gain: +3-5% absolute improvement

### **Metrics Computed**:

1. **Task Success Rate** (binary, best attempt only):
   - % tasks where train solved AND all test outputs correct
   - Strict metric: requires best candidate to be perfect

2. **Train Solved Rate**:
   - % tasks where all train pairs are 100% correct
   - Measures: "can we learn the pattern?"

3. **Test Accuracy Given Train**:
   - Of tasks where train solved, % where test is also correct
   - Measures: "can we generalize?"

4. **Competition Score** (Phase 5B, multi-attempt):
   - Average score across tasks (allows 2 attempts)
   - Aligned with actual competition scoring
   - Usually 3-5% higher than binary task success

5. **Coverage**:
   - % tasks where we attempted test prediction
   - Same as train solved rate (due to gating)

### **Output**:
```
evaluation_results/
├── summary.json:
    {
      "task_success_rate": 0.15,
      "competition_score": 0.183,
      "train_solved_rate": 0.25,
      "test_accuracy_given_train": 0.60
    }
├── detailed_results.json:
    [
      {
        "task_id": "0934a4d8",
        "train_solved": false,
        "test_attempted": false,
        "task_success": false
      },
      ...
    ]
```

### **Command**:
```bash
python3 arc_nodsl/evaluation/evaluate_model.py \
  --autoencoder_checkpoint checkpoints/autoencoder_best.pt \
  --controller_checkpoint checkpoints/controller_best.pt \
  --dataset data/arc-agi_evaluation_challenges.json \
  --beam_size 16 \
  --max_steps 8 \
  --num_attempts 2 \
  --output_dir evaluation_results
```

---

## **Phase 4: Inference on Test Set (Competition Submission)**

### **What It Does**:
Generates predictions for competition test set (no ground truth labels).

### **Process**:

```python
def create_submission(test_dataset, solver):
    submission = {}

    for task in test_dataset:
        result = solver.solve_task(task)

        # Only submit if train solved
        if result.train_solved:
            # Format: 2 attempts per test output
            submission[task.task_id] = {
                "test_output_1": {
                    "attempt_1": to_json(result.test_attempts[0][0]),
                    "attempt_2": to_json(result.test_attempts[0][1])
                },
                "test_output_2": {
                    "attempt_1": to_json(result.test_attempts[1][0]),
                    "attempt_2": to_json(result.test_attempts[1][1])
                }
            }
        # Else: skip this task (no submission)

    return submission
```

### **Submission Strategy**:
- **Conservative**: Only submit when confident (train solved)
- **Trade-off**: Lower coverage, higher accuracy per submission
- **Expected**: Submit ~25% of tasks, with ~60% accuracy on those

### **Command** (future implementation):
```bash
python3 arc_nodsl/evaluation/create_submission.py \
  --autoencoder_checkpoint checkpoints/autoencoder_best.pt \
  --controller_checkpoint checkpoints/controller_best.pt \
  --dataset data/arc-agi_test_challenges.json \
  --beam_size 16 \
  --max_steps 8 \
  --num_attempts 2 \
  --output submission.json
```

---

## **Summary: What Gets Learned?**

### **Autoencoder Learns**:
- How to decompose grids into objects (slots)
- What features matter (shapes, colors, positions)
- How to reconstruct grids from objects

### **Controller Learns**:
- Which operators to apply in which order
- How to adapt quickly to new tasks (meta-learning)
- When to stop (implicit in beam search scoring)

### **Operators Learn**:
- Task-useful transformations (not pre-defined)
- Might learn: rotate, flip, recolor, move, copy, delete, merge
- Learned from data, not hand-coded

### **Overall System Learns**:
- Pattern recognition: "this task is about rotation"
- Composition: "first rotate, then recolor"
- Generalization: "train pattern applies to test"

---

## **Key Innovations**

1. **Object-Centric Representations (Slots)**:
   - Reduces search space from pixel-level to object-level
   - Enables compositional reasoning

2. **Latent Space Search**:
   - Search in compact slot space, not grid space
   - Makes beam search tractable

3. **Meta-Learning (Reptile)**:
   - Learns to adapt quickly to new tasks
   - Few-shot learning via fast adaptation

4. **Train-First Gating**:
   - Only predict test when train is mastered
   - Prevents wasting attempts on hopeless tasks

5. **Multi-Attempt Evaluation**:
   - Aligned with competition rules
   - Increases coverage by allowing 2 guesses

6. **Binary Task Success Bonus**:
   - Encourages perfect solutions, not just high accuracy
   - Aligns training objective with evaluation metric

---

This complete pipeline transforms the problem from "solve abstract reasoning tasks" to "learn a meta-learning controller that sequences learned operators in latent object space" - a much more tractable learning problem!
