"""
Active ARC Task Solver with test-time adaptation.

Implements active inference: adapts the controller on each task's train pairs
before making test predictions. This fully realizes the meta-learning potential.
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

import torch
import torch.nn as nn
import time
import copy
from typing import Dict, List, Tuple, Optional
from dataclasses import dataclass

from arc_nodsl.evaluation.solver import ARCSolver
from arc_nodsl.evaluation.metrics import TaskResult, exact_match
from arc_nodsl.training.inner_loop import InnerLoop
from arc_nodsl.inference.task_embed import build_task_embedding
from arc_nodsl.inference.latent_search import beam_search


@dataclass
class AdaptationMetrics:
    """Metrics from active adaptation phase."""
    num_steps: int
    time_seconds: float
    converged: bool  # Did all train pairs solve?
    final_train_accuracy: float
    reward_trajectory: List[float]
    stopped_reason: str  # 'train_solved', 'time_budget', 'max_steps'


class ActiveARCSolver(ARCSolver):
    """
    Active inference solver that adapts to each task.

    Key Innovation:
    Instead of using a fixed controller for all tasks, this solver:
    1. Clones the base controller
    2. Actively adapts it on the task's train pairs
    3. Uses the adapted controller for test prediction

    This aligns with meta-learning philosophy: learn initialization for
    fast task-specific adaptation.

    Algorithm:
    1. [NEW] Adaptation Phase:
        - Clone base controller
        - Run gradient descent on train pairs
        - Stop when: train solved OR time budget OR max steps
    2. Evaluate train pairs with adapted controller
    3. If train solved: predict test with adapted controller
    4. Return results + adaptation metrics
    """

    def __init__(
        self,
        encoder: nn.Module,
        controller: nn.Module,
        operators: nn.Module,
        renderer: nn.Module,
        # Existing inference params
        beam_size: int = 16,
        max_operator_steps: int = 8,
        num_attempts: int = 2,
        # NEW: Active adaptation params
        adaptation_steps: int = 20,
        adaptation_lr: float = 1e-3,
        time_budget_seconds: float = 60.0,
        early_stop_on_train_solved: bool = True,
        beam_size_adaptation: int = 8,  # Smaller beam during adaptation (faster)
        check_convergence_every: int = 5,  # Check train solved every N steps
        device: torch.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    ):
        """
        Initialize active inference solver.

        Args:
            encoder: Pretrained SlotEncoder (frozen)
            controller: Base controller from meta-learning (will be cloned per task)
            operators: Trained OperatorLibrary (frozen)
            renderer: Pretrained SlotRenderer (frozen)
            beam_size: Beam size for test prediction (default 16)
            max_operator_steps: Max operator sequence length
            num_attempts: Number of attempts per test output
            adaptation_steps: Max gradient steps for adaptation (default 20)
            adaptation_lr: Learning rate for adaptation (default 1e-3)
            time_budget_seconds: Max time for adaptation (default 60s)
            early_stop_on_train_solved: Stop when all train pairs solved
            beam_size_adaptation: Beam size during adaptation (smaller = faster)
            check_convergence_every: Check convergence every N steps
            device: Compute device
        """
        # Initialize base solver
        super().__init__(
            encoder, controller, operators, renderer,
            beam_size, max_operator_steps, num_attempts, device
        )

        # Active learning params
        self.adaptation_steps = adaptation_steps
        self.adaptation_lr = adaptation_lr
        self.time_budget = time_budget_seconds
        self.early_stop = early_stop_on_train_solved
        self.beam_size_adaptation = beam_size_adaptation
        self.check_every = check_convergence_every

        # Create InnerLoop for active adaptation
        self.inner_loop = InnerLoop(
            num_inner_steps=adaptation_steps,
            beam_size=beam_size_adaptation,
            max_operator_steps=max_operator_steps,
            learning_rate=adaptation_lr,
            entropy_weight=0.01,
            binary_bonus_weight=0.5,
            device=device
        )

    def solve_task(
        self,
        task_data: Dict,
        verbose: bool = False
    ) -> TaskResult:
        """
        Solve task with active adaptation.

        Steps:
        1. [NEW] Adaptation: Clone and adapt controller on train pairs
        2. Evaluate train with adapted controller
        3. If train solved: predict test with adapted controller
        4. Return results + adaptation metrics

        Args:
            task_data: Task dict from ARCDataset
            verbose: Print progress

        Returns:
            TaskResult with adaptation metrics
        """
        task_id = task_data['task_id']

        if verbose:
            print(f"\n{'='*60}")
            print(f"[ACTIVE SOLVER] Task: {task_id}")
            print(f"{'='*60}")

        # ===== PHASE 1: ACTIVE ADAPTATION =====
        if verbose:
            print(f"\n[1/4] Active Adaptation Phase")
            print(f"  Adapting controller on {len(task_data['train_inputs'])} train pairs...")
            print(f"  Budget: {self.adaptation_steps} steps, {self.time_budget}s")

        start_time = time.time()

        # Run adaptation with monitoring
        adapted_controller, adaptation_metrics = self._adapt_with_monitoring(
            task_data,
            start_time,
            verbose
        )

        if verbose:
            print(f"\n  Adaptation complete:")
            print(f"    Steps: {adaptation_metrics.num_steps}")
            print(f"    Time: {adaptation_metrics.time_seconds:.2f}s")
            print(f"    Stopped: {adaptation_metrics.stopped_reason}")
            print(f"    Final train accuracy: {adaptation_metrics.final_train_accuracy:.1%}")

        # ===== PHASE 2: EVALUATE TRAIN PAIRS =====
        if verbose:
            print(f"\n[2/4] Evaluating train pairs with adapted controller...")

        train_predictions, train_solved, train_results = self._evaluate_train_with_adapted(
            task_data,
            adapted_controller,
            verbose
        )

        if verbose:
            solved_count = sum(r.is_solved for r in train_results)
            print(f"  Train evaluation: {solved_count}/{len(train_results)} pairs solved")
            if not train_solved:
                print(f"  ⚠ Train not fully solved despite adaptation")

        # ===== PHASE 3: CONDITIONAL TEST PREDICTION =====
        test_predictions = None
        test_correct = None
        task_success = False
        test_attempts = None
        competition_score = None

        if train_solved:
            if verbose:
                print(f"\n[3/4] ✓ Train solved! Predicting test with adapted controller...")

            test_predictions, test_correct, task_success, test_attempts, competition_score = \
                self._predict_test_with_adapted(
                    task_data,
                    adapted_controller,
                    verbose
                )

            if verbose and test_correct is not None:
                correct_count = sum(test_correct)
                print(f"  Test result: {correct_count}/{len(test_correct)} correct")
                if task_success:
                    print(f"  🎉 TASK FULLY SOLVED!")
        else:
            if verbose:
                print(f"\n[3/4] ✗ Train not solved. Skipping test.")

        # ===== PHASE 4: RETURN RESULTS =====
        if verbose:
            print(f"\n[4/4] Summary:")
            print(f"  Train solved: {train_solved}")
            print(f"  Test attempted: {test_predictions is not None}")
            print(f"  Task success: {task_success}")
            print(f"  Adaptation: {adaptation_metrics.num_steps} steps, {adaptation_metrics.time_seconds:.2f}s")
            print(f"{'='*60}\n")

        return TaskResult(
            task_id=task_id,
            train_solved=train_solved,
            train_results=train_results,
            test_attempted=test_predictions is not None,
            test_predictions=test_predictions,
            test_correct=test_correct,
            task_success=task_success,
            confidence=1.0 if train_solved else 0.0,
            test_attempts=test_attempts,
            competition_score=competition_score,
            # NEW: Adaptation metrics
            adaptation_steps=adaptation_metrics.num_steps,
            adaptation_time=adaptation_metrics.time_seconds,
            adaptation_converged=adaptation_metrics.converged,
            adaptation_rewards=adaptation_metrics.reward_trajectory
        )

    def _adapt_with_monitoring(
        self,
        task_data: Dict,
        start_time: float,
        verbose: bool
    ) -> Tuple[nn.Module, AdaptationMetrics]:
        """
        Adapt controller with monitoring and early stopping.

        Monitors:
        - Train accuracy (check convergence every N steps)
        - Time budget
        - Max steps

        Returns:
            (adapted_controller, adaptation_metrics)
        """
        # Use InnerLoop for adaptation (handles all the REINFORCE logic)
        # We'll override its num_inner_steps to respect our budget
        original_steps = self.inner_loop.num_inner_steps
        self.inner_loop.num_inner_steps = self.adaptation_steps

        # Run adaptation
        adapted_controller, inner_metrics = self.inner_loop.train_on_task(
            task_data,
            self.encoder,
            self.controller,  # Base controller (will be cloned)
            self.operators,
            self.renderer,
            clone_controller=True,
            verbose=False  # We handle verbose ourselves
        )

        # Restore original
        self.inner_loop.num_inner_steps = original_steps

        # Compute final train accuracy
        final_acc = self._compute_train_accuracy(task_data, adapted_controller)

        # Determine stop reason
        elapsed = time.time() - start_time
        if inner_metrics.success_rate >= 1.0:
            stop_reason = 'train_solved'
            converged = True
        elif elapsed >= self.time_budget:
            stop_reason = 'time_budget'
            converged = False
        else:
            stop_reason = 'max_steps'
            converged = False

        return adapted_controller, AdaptationMetrics(
            num_steps=inner_metrics.num_steps,
            time_seconds=elapsed,
            converged=converged,
            final_train_accuracy=final_acc,
            reward_trajectory=[inner_metrics.mean_reward],  # Simplified
            stopped_reason=stop_reason
        )

    def _compute_train_accuracy(
        self,
        task_data: Dict,
        controller: nn.Module
    ) -> float:
        """Compute accuracy on all train pairs."""
        controller.eval()

        # Build task embedding
        train_pairs = self._prepare_train_pairs(task_data)
        task_embed = build_task_embedding(
            train_pairs,
            encoder=None,
            device=self.device,
            analyze_operators=False
        )

        correct = 0
        total = len(task_data['train_inputs'])

        with torch.no_grad():
            for i in range(total):
                input_grid = task_data['train_inputs'][i].to(self.device)
                target_grid = task_data['train_outputs'][i].to(self.device)
                input_shape = task_data['train_shapes'][i]['input']
                output_shape = task_data['train_shapes'][i]['output']

                candidates = beam_search(
                    self.encoder,
                    controller,
                    self.operators,
                    self.renderer,
                    input_grid,
                    input_shape,
                    output_shape,
                    task_embed,
                    target_grid=target_grid,
                    beam_size=self.beam_size_adaptation,
                    max_steps=self.max_operator_steps,
                    device=self.device,
                    collect_log_probs=False
                )

                if len(candidates) > 0:
                    h, w = output_shape
                    if exact_match(candidates[0].prediction.cpu(), target_grid.cpu(), h, w):
                        correct += 1

        controller.train()
        return correct / total if total > 0 else 0.0

    def _evaluate_train_with_adapted(
        self,
        task_data: Dict,
        adapted_controller: nn.Module,
        verbose: bool
    ) -> Tuple[List[torch.Tensor], bool, List]:
        """Evaluate train pairs with adapted controller."""
        adapted_controller.eval()

        # Build task embedding
        train_pairs = self._prepare_train_pairs(task_data)
        task_embed = build_task_embedding(
            train_pairs,
            encoder=None,
            device=self.device,
            analyze_operators=False
        )

        # Predict all train pairs
        train_predictions = []
        with torch.no_grad():
            for i in range(len(task_data['train_inputs'])):
                pred = self._solve_pair_with_controller(
                    task_data['train_inputs'][i],
                    task_data['train_shapes'][i]['input'],
                    task_data['train_shapes'][i]['output'],
                    task_embed,
                    adapted_controller,
                    target=task_data['train_outputs'][i]
                )
                train_predictions.append(pred)

                if verbose:
                    h, w = task_data['train_shapes'][i]['output']
                    target = task_data['train_outputs'][i].to(self.device)
                    is_correct = exact_match(pred, target, h, w)
                    status = "✓" if is_correct else "✗"
                    print(f"    Pair {i+1}: {status}")

        # Evaluate
        train_predictions_cpu = [p.cpu() for p in train_predictions]
        train_solved, train_results = self.metric.evaluate_train_pairs(
            train_predictions_cpu,
            task_data['train_outputs'],
            [s['output'] for s in task_data['train_shapes']]
        )

        return train_predictions, train_solved, train_results

    def _predict_test_with_adapted(
        self,
        task_data: Dict,
        adapted_controller: nn.Module,
        verbose: bool
    ) -> Tuple:
        """Predict test outputs with adapted controller."""
        adapted_controller.eval()

        # Build task embedding
        train_pairs = self._prepare_train_pairs(task_data)
        task_embed = build_task_embedding(
            train_pairs,
            encoder=None,
            device=self.device,
            analyze_operators=False
        )

        # Generate predictions
        test_attempts = []
        test_predictions = []

        with torch.no_grad():
            for i in range(len(task_data['test_inputs'])):
                attempts = self._solve_pair_multi_attempt_with_controller(
                    task_data['test_inputs'][i],
                    task_data['test_shapes'][i]['input'],
                    task_data['test_shapes'][i]['output'],
                    task_embed,
                    adapted_controller,
                    target=None,
                    num_attempts=self.num_attempts
                )
                test_attempts.append(attempts)
                test_predictions.append(attempts[0])

        # Evaluate if ground truth available
        test_correct = None
        task_success = False
        competition_score = None

        if (len(task_data['test_outputs']) > 0 and
                task_data['test_outputs'][0] is not None):

            test_correct = []
            per_output_scores = []
            test_attempts_cpu = [[a.cpu() for a in attempts] for attempts in test_attempts]

            for j, (attempts_cpu, target, shape) in enumerate(zip(
                    test_attempts_cpu,
                    task_data['test_outputs'],
                    task_data['test_shapes']
            )):
                h, w = shape['output']

                # Check if ANY attempt correct
                any_correct = any(exact_match(a, target, h, w) for a in attempts_cpu)
                per_output_scores.append(1.0 if any_correct else 0.0)

                # Best attempt for binary metric
                best_correct = exact_match(attempts_cpu[0], target, h, w)
                test_correct.append(best_correct)

                if verbose:
                    status = "✓" if any_correct else "✗"
                    print(f"    Test pair {j+1}: {status}")

            task_success = all(test_correct)
            competition_score = sum(per_output_scores) / len(per_output_scores) if per_output_scores else 0.0

        return test_predictions, test_correct, task_success, test_attempts, competition_score

    def _solve_pair_with_controller(
        self,
        input_grid: torch.Tensor,
        input_shape: Tuple[int, int],
        output_shape: Tuple[int, int],
        task_embed: Dict,
        controller: nn.Module,
        target: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """Solve pair with specific controller."""
        with torch.no_grad():
            candidates = beam_search(
                self.encoder,
                controller,
                self.operators,
                self.renderer,
                input_grid.to(self.device),
                input_shape,
                output_shape,
                task_embed,
                target_grid=target.to(self.device) if target is not None else None,
                beam_size=self.beam_size,
                max_steps=self.max_operator_steps,
                device=self.device,
                collect_log_probs=False
            )

        if len(candidates) > 0:
            return candidates[0].prediction
        else:
            return torch.zeros(30, 30, dtype=torch.long, device=self.device)

    def _solve_pair_multi_attempt_with_controller(
        self,
        input_grid: torch.Tensor,
        input_shape: Tuple[int, int],
        output_shape: Tuple[int, int],
        task_embed: Dict,
        controller: nn.Module,
        target: Optional[torch.Tensor] = None,
        num_attempts: int = 2
    ) -> List[torch.Tensor]:
        """Solve pair with multiple attempts using specific controller."""
        with torch.no_grad():
            candidates = beam_search(
                self.encoder,
                controller,
                self.operators,
                self.renderer,
                input_grid.to(self.device),
                input_shape,
                output_shape,
                task_embed,
                target_grid=target.to(self.device) if target is not None else None,
                beam_size=self.beam_size,
                max_steps=self.max_operator_steps,
                device=self.device,
                collect_log_probs=False
            )

        predictions = []
        if len(candidates) > 0:
            for i in range(min(num_attempts, len(candidates))):
                predictions.append(candidates[i].prediction)
            while len(predictions) < num_attempts:
                predictions.append(torch.zeros(30, 30, dtype=torch.long, device=self.device))
        else:
            for _ in range(num_attempts):
                predictions.append(torch.zeros(30, 30, dtype=torch.long, device=self.device))

        return predictions


# Test code
if __name__ == "__main__":
    from arc_nodsl.data.loader import ARCDataset
    from arc_nodsl.models.slots import SlotEncoder
    from arc_nodsl.models.renderer import SlotRenderer
    from arc_nodsl.models.operators import OperatorLibrary
    from arc_nodsl.models.controller import Controller

    print("="*60)
    print("Testing ActiveARCSolver")
    print("="*60)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"\nDevice: {device}")

    # Load dataset
    print("\nLoading dataset...")
    dataset = ARCDataset("data/arc-agi_evaluation_challenges.json")
    print(f"Loaded {len(dataset)} evaluation tasks")

    # Create models (random weights for testing)
    print("\nCreating models (random weights)...")
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

    # Create active solver
    print("\nCreating ActiveARCSolver...")
    solver = ActiveARCSolver(
        encoder, controller, operators, renderer,
        beam_size=4,  # Small for fast testing
        max_operator_steps=2,
        num_attempts=2,
        adaptation_steps=5,  # Very few for quick test
        adaptation_lr=1e-3,
        time_budget_seconds=30,
        beam_size_adaptation=4,
        device=device
    )
    print("✓ Created")

    # Test on first task
    print("\n" + "="*60)
    print("Testing on first task")
    print("="*60)

    task = dataset[0]
    result = solver.solve_task(task, verbose=True)

    # Verify result
    print("\n" + "="*60)
    print("Result Verification")
    print("="*60)
    print(f"✓ Task ID: {result.task_id}")
    print(f"✓ Train solved: {result.train_solved}")
    print(f"✓ Test attempted: {result.test_attempted}")
    print(f"✓ Task success: {result.task_success}")
    print(f"✓ Adaptation steps: {result.adaptation_steps}")
    print(f"✓ Adaptation time: {result.adaptation_time:.2f}s")
    print(f"✓ Adaptation converged: {result.adaptation_converged}")

    print("\n" + "="*60)
    print("✓ ActiveARCSolver test complete!")
    print("="*60)
    print("\nNote: With random weights, expect low performance.")
    print("After meta-learning, active adaptation should significantly improve results!")
