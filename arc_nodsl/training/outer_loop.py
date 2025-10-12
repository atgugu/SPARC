"""
Outer loop: meta-learning across tasks (Reptile-style).

Updates the base controller to be better at fast adaptation.
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

import torch
import torch.nn as nn
from typing import Dict, List, Optional
from dataclasses import dataclass
import numpy as np

from arc_nodsl.training.inner_loop import InnerLoop, InnerLoopMetrics
from arc_nodsl.training.losses import ReinforceReward


@dataclass
class OuterLoopMetrics:
    """Metrics from outer loop meta-training."""
    meta_loss: float
    mean_test_reward: float
    mean_train_reward: float
    test_success_rate: float
    train_success_rate: float
    num_tasks: int
    # Binary task success metrics (Phase 5B)
    task_success_rate: float          # % tasks with query solved (100% exact match)
    query_solved_rate: float          # Same as above, for clarity
    support_solved_rate: float        # % tasks with all support pairs solved
    generalization_rate: Optional[float]  # query_solved / support_solved (when support > 0)


class OuterLoop:
    """
    Meta-learning outer loop (Reptile algorithm).

    Algorithm:
    1. Sample batch of tasks
    2. For each task:
        a. Clone base controller θ
        b. Adapt clone on train pairs → θ_i'
        c. Evaluate θ_i' on test pairs
    3. Update base controller: θ ← θ + α * mean(θ_i' - θ)

    This is simpler than full MAML (no second-order gradients) but still effective.
    """

    def __init__(
        self,
        inner_loop: InnerLoop,
        meta_learning_rate: float = 1e-4,
        meta_batch_size: int = 4,
        reward_threshold: float = 0.95,
        use_train_query_split: bool = True,  # Phase 5B: enable train/query split
        device: torch.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    ):
        """
        Initialize outer loop meta-learner.

        Args:
            inner_loop: InnerLoop instance for task adaptation
            meta_learning_rate: Meta-learning rate (α in Reptile)
            meta_batch_size: Number of tasks per meta-batch
            reward_threshold: Threshold to consider a pair "solved"
            use_train_query_split: If True, use train/query split for binary metrics
            device: Compute device
        """
        self.inner_loop = inner_loop
        self.meta_learning_rate = meta_learning_rate
        self.meta_batch_size = meta_batch_size
        self.reward_threshold = reward_threshold
        self.use_train_query_split = use_train_query_split
        self.device = device

        self.reward_computer = ReinforceReward()

    def meta_train_step(
        self,
        tasks: List[Dict],
        encoder: nn.Module,
        controller: nn.Module,
        operators: nn.Module,
        renderer: nn.Module,
        verbose: bool = False
    ) -> OuterLoopMetrics:
        """
        Perform one meta-training step on a batch of tasks.

        Args:
            tasks: List of task dicts (length = meta_batch_size)
            encoder: Pretrained encoder
            controller: Base controller to meta-train
            operators: Pretrained operators
            renderer: Pretrained renderer
            verbose: Print progress

        Returns:
            OuterLoopMetrics with meta-loss and evaluation metrics
        """
        if verbose:
            print(f"\nMeta-training step on {len(tasks)} tasks")

        # Store base controller parameters
        base_params = {name: param.clone() for name, param in controller.named_parameters()}

        # Accumulate adapted parameters for Reptile update
        adapted_params_list = []

        # Track metrics (legacy fuzzy rewards)
        all_test_rewards = []
        all_train_rewards = []
        test_success_count = 0
        train_success_count = 0
        num_test_pairs = 0
        num_train_pairs = 0

        # Track binary task success metrics (Phase 5B)
        query_solved_count = 0
        support_solved_count = 0
        both_solved_count = 0  # For generalization rate

        # Inner loop adaptation for each task
        for i, task in enumerate(tasks):
            if verbose:
                print(f"\n  Task {i+1}/{len(tasks)}: {task['task_id']}")

            # Adapt controller using train/query split if enabled
            if self.use_train_query_split and len(task['train_inputs']) >= 2:
                # Use support/query split
                adapted_controller, inner_metrics = self.inner_loop.train_on_task_with_split(
                    task, encoder, controller, operators, renderer,
                    query_index=-1,  # Last train pair as query
                    clone_controller=True,
                    verbose=verbose
                )

                # Track binary metrics
                if inner_metrics.query_solved:
                    query_solved_count += 1
                if inner_metrics.support_solved:
                    support_solved_count += 1
                if inner_metrics.query_solved and inner_metrics.support_solved:
                    both_solved_count += 1

                if verbose:
                    print(f"    Query solved: {inner_metrics.query_solved}")
                    print(f"    Support solved: {inner_metrics.support_solved}")
            else:
                # Fall back to regular training (no split)
                adapted_controller, inner_metrics = self.inner_loop.train_on_task(
                    task, encoder, controller, operators, renderer,
                    clone_controller=True,
                    verbose=verbose
                )

            # Store adapted parameters for meta-update
            adapted_params = {name: param.clone() for name, param in adapted_controller.named_parameters()}
            adapted_params_list.append(adapted_params)

            # Evaluate adapted controller on test pairs (legacy)
            test_rewards = self._evaluate_on_test(
                adapted_controller,
                task,
                encoder,
                operators,
                renderer
            )

            all_test_rewards.extend(test_rewards)
            test_success_count += sum(1 for r in test_rewards if r >= self.reward_threshold)
            num_test_pairs += len(test_rewards)

            # Also track train performance (legacy)
            all_train_rewards.extend([inner_metrics.best_reward])
            train_success_count += int(inner_metrics.success_rate > 0)
            num_train_pairs += 1

            if verbose:
                print(f"    Test reward: {np.mean(test_rewards):.3f}")
                print(f"    Train reward: {inner_metrics.mean_reward:.3f}")

        # Reptile meta-update: θ ← θ + α * mean(θ' - θ)
        with torch.no_grad():
            for name, param in controller.named_parameters():
                # Compute mean of adapted parameters
                adapted_mean = torch.stack([ap[name] for ap in adapted_params_list]).mean(dim=0)

                # Reptile update
                param.data.add_(
                    adapted_mean - base_params[name],
                    alpha=self.meta_learning_rate
                )

        # Compute metrics (legacy fuzzy rewards)
        mean_test_reward = np.mean(all_test_rewards) if all_test_rewards else 0.0
        mean_train_reward = np.mean(all_train_rewards) if all_train_rewards else 0.0
        test_success_rate = test_success_count / num_test_pairs if num_test_pairs > 0 else 0.0
        train_success_rate = train_success_count / num_train_pairs if num_train_pairs > 0 else 0.0

        # Compute binary task success metrics (Phase 5B)
        num_tasks = len(tasks)
        task_success_rate = query_solved_count / num_tasks if num_tasks > 0 else 0.0
        query_solved_rate = query_solved_count / num_tasks if num_tasks > 0 else 0.0
        support_solved_rate = support_solved_count / num_tasks if num_tasks > 0 else 0.0

        # Generalization rate: of tasks where support solved, how many also solved query?
        if support_solved_count > 0:
            generalization_rate = both_solved_count / support_solved_count
        else:
            generalization_rate = None  # No support solved, can't compute

        # Meta-loss is negative mean test reward (to minimize)
        meta_loss = -mean_test_reward

        metrics = OuterLoopMetrics(
            meta_loss=meta_loss,
            mean_test_reward=mean_test_reward,
            mean_train_reward=mean_train_reward,
            test_success_rate=test_success_rate,
            train_success_rate=train_success_rate,
            num_tasks=num_tasks,
            task_success_rate=task_success_rate,
            query_solved_rate=query_solved_rate,
            support_solved_rate=support_solved_rate,
            generalization_rate=generalization_rate
        )

        if verbose:
            print(f"\n  Meta-step complete:")
            print(f"    Meta-loss: {metrics.meta_loss:.4f}")
            print(f"    Mean test reward: {metrics.mean_test_reward:.3f}")
            print(f"    Test success rate: {metrics.test_success_rate:.1%}")
            # Phase 5B: Binary metrics
            if self.use_train_query_split:
                print(f"    [Binary] Task success rate: {metrics.task_success_rate:.1%} ({query_solved_count}/{num_tasks} tasks)")
                print(f"    [Binary] Support solved: {metrics.support_solved_rate:.1%}")
                if metrics.generalization_rate is not None:
                    print(f"    [Binary] Generalization rate: {metrics.generalization_rate:.1%}")

        return metrics

    def _evaluate_on_test(
        self,
        adapted_controller: nn.Module,
        task: Dict,
        encoder: nn.Module,
        operators: nn.Module,
        renderer: nn.Module
    ) -> List[float]:
        """
        Evaluate adapted controller on test pairs.

        Returns:
            List of rewards for each test pair
        """
        from arc_nodsl.inference.task_embed import build_task_embedding
        from arc_nodsl.inference.latent_search import beam_search

        # Check if test outputs are available
        # (They're hidden in competition data, so we fall back to train pairs)
        has_test_outputs = (
            len(task['test_outputs']) > 0 and
            task['test_outputs'][0] is not None
        )

        if not has_test_outputs:
            # No test outputs available, use train pairs as proxy
            # In real meta-learning, we'd split train into train/val
            eval_inputs = task['train_inputs']
            eval_outputs = task['train_outputs']
            eval_shapes = task['train_shapes']
        else:
            eval_inputs = task['test_inputs']
            eval_outputs = task['test_outputs']
            eval_shapes = task['test_shapes']

        # Build task embedding from train pairs
        train_pairs = []
        for i in range(len(task['train_inputs'])):
            train_pairs.append((
                task['train_inputs'][i],
                task['train_outputs'][i],
                task['train_shapes'][i]
            ))

        task_embed = build_task_embedding(
            train_pairs,
            encoder=None,
            device=self.device,
            analyze_operators=False
        )

        # Evaluate on available pairs
        rewards = []
        adapted_controller.eval()

        with torch.no_grad():
            for i in range(len(eval_inputs)):
                input_grid = eval_inputs[i].to(self.device)
                target_grid = eval_outputs[i].to(self.device)
                input_shape = eval_shapes[i]['input']
                output_shape = eval_shapes[i]['output']

                # Run beam search
                candidates = beam_search(
                    encoder,
                    adapted_controller,
                    operators,
                    renderer,
                    input_grid,
                    input_shape,
                    output_shape,
                    task_embed,
                    target_grid=target_grid,
                    beam_size=self.inner_loop.beam_size,
                    max_steps=self.inner_loop.max_operator_steps,
                    device=self.device,
                    collect_log_probs=False
                )

                # Get best candidate's reward
                if len(candidates) > 0:
                    best_cand = candidates[0]
                    reward = self.reward_computer.compute_reward(
                        best_cand.prediction,
                        target_grid,
                        best_cand.h,
                        best_cand.w,
                        constraints=task_embed['constraints'],
                        input_shape=input_shape
                    )
                    rewards.append(reward)
                else:
                    rewards.append(0.0)

        adapted_controller.train()
        return rewards


# Test code
if __name__ == "__main__":
    from arc_nodsl.data.loader import ARCDataset
    from arc_nodsl.models.slots import SlotEncoder
    from arc_nodsl.models.renderer import SlotRenderer
    from arc_nodsl.models.operators import OperatorLibrary
    from arc_nodsl.models.controller import Controller

    print("="*60)
    print("Testing OuterLoop (Meta-Learning)")
    print("="*60)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Device: {device}")

    # Load tasks
    print("\nLoading dataset...")
    dataset = ARCDataset("data/arc-agi_training_challenges.json")

    # Sample 2 tasks for fast test
    tasks = [dataset[i] for i in range(2)]
    print(f"Sampled {len(tasks)} tasks")

    # Create models (random weights)
    print("\nCreating models...")
    encoder = SlotEncoder(
        num_slots=8, d_color=16, d_feat=64, d_slot=128,
        num_iters=3, H=30, W=30
    ).to(device)

    renderer = SlotRenderer(
        d_slot=128, d_hidden=64, H=30, W=30, use_mask=True
    ).to(device)

    operators = OperatorLibrary(
        num_ops=8, d_slot=128, d_hidden=128, H=30, W=30
    ).to(device)

    controller = Controller(
        num_operators=8, d_slot=128, d_task=128,
        d_hidden=256, max_steps=4
    ).to(device)

    encoder.eval()
    renderer.eval()
    operators.eval()
    controller.train()

    # Create inner loop
    inner_loop = InnerLoop(
        num_inner_steps=2,  # Very short for fast test
        beam_size=4,
        max_operator_steps=2,
        binary_bonus_weight=0.5,  # Phase 5B: binary bonus
        device=device
    )

    # Create outer loop
    outer_loop = OuterLoop(
        inner_loop=inner_loop,
        meta_learning_rate=1e-3,
        meta_batch_size=2,
        use_train_query_split=True,  # Enable Phase 5B binary metrics
        device=device
    )

    # Run one meta-training step
    print("\nRunning meta-training step...")
    metrics = outer_loop.meta_train_step(
        tasks, encoder, controller, operators, renderer,
        verbose=True
    )

    print("\n" + "="*60)
    print("✓ Outer loop test complete!")
    print("="*60)
    print(f"Meta-loss: {metrics.meta_loss:.4f}")
    print(f"Mean test reward: {metrics.mean_test_reward:.3f}")
    print(f"Test success rate: {metrics.test_success_rate:.1%}")
    print(f"Train success rate: {metrics.train_success_rate:.1%}")
    print(f"\nBinary Task Success Metrics (Phase 5B):")
    print(f"  Task success rate: {metrics.task_success_rate:.1%}")
    print(f"  Support solved rate: {metrics.support_solved_rate:.1%}")
    if metrics.generalization_rate is not None:
        print(f"  Generalization rate: {metrics.generalization_rate:.1%}")
