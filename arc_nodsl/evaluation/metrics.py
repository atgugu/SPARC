"""
Core evaluation metrics for ARC tasks.

Implements the "solve train first, then predict test" strategy with
task-level success metrics.
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

import torch
from typing import List, Tuple, Optional, Dict
from dataclasses import dataclass
import numpy as np


@dataclass
class PairResult:
    """Result for a single input-output pair."""
    is_solved: bool          # Exact pixel match
    pixel_accuracy: float    # For analysis (not used for gating)
    constraint_satisfied: bool  # Whether constraints are met
    h: int                   # Actual output height
    w: int                   # Actual output width


@dataclass
class TaskResult:
    """Complete result for a task."""
    task_id: str

    # Train evaluation
    train_solved: bool              # All train pairs perfect
    train_results: List[PairResult]  # Per-pair train results

    # Test evaluation (conditional on train_solved)
    test_attempted: bool            # Did we attempt test?
    test_predictions: Optional[List[torch.Tensor]]  # Predictions (if attempted)
    test_correct: Optional[List[bool]]  # Correctness (if test outputs available)

    # Overall task success
    task_success: bool              # train_solved AND all test correct
    confidence: float               # 1.0 if train solved, else 0.0

    # Phase 5B: Multi-attempt support (competition scoring)
    test_attempts: Optional[List[List[torch.Tensor]]] = None  # K attempts per test input
    competition_score: Optional[float] = None  # Competition score (0-1 per task)

    # Active Learning: Adaptation metrics (when using ActiveARCSolver)
    adaptation_steps: Optional[int] = None  # Number of gradient steps during adaptation
    adaptation_time: Optional[float] = None  # Time spent adapting (seconds)
    adaptation_converged: Optional[bool] = None  # Did train solve during adaptation?
    adaptation_rewards: Optional[List[float]] = None  # Reward trajectory during adaptation


@dataclass
class EvaluationMetrics:
    """Aggregate metrics across multiple tasks."""
    # Primary metric (what we optimize for)
    task_success_rate: float        # % tasks fully solved (train + test)

    # Diagnostic metrics
    train_solved_rate: float        # % tasks where all train pairs solved
    test_accuracy_given_train: float # Of tasks where train solved, % with all test correct
    coverage: float                 # % tasks where we attempted test

    # Granular statistics
    total_tasks: int
    num_tasks_solved: int           # Full task success
    num_train_solved: int           # Train solved (may fail test)
    num_train_failed: int           # Train not solved

    # Task lists for analysis
    tasks_solved: List[str]         # task_ids with full success
    tasks_train_only: List[str]     # Train solved but test wrong
    tasks_failed: List[str]         # Train not solved

    # Per-pair statistics (for detailed analysis)
    total_train_pairs: int
    correct_train_pairs: int
    total_test_pairs: int
    correct_test_pairs: int

    # Phase 5B: Competition scoring (multi-attempt)
    mean_competition_score: Optional[float] = None  # Average competition score across tasks
    num_competition_attempts: Optional[int] = None  # How many attempts per test output

    # Active Learning: Adaptation statistics (when using ActiveARCSolver)
    mean_adaptation_steps: Optional[float] = None  # Average steps per task
    mean_adaptation_time: Optional[float] = None  # Average time per task (seconds)
    adaptation_convergence_rate: Optional[float] = None  # % tasks that converged during adaptation


def exact_match(
    pred: torch.Tensor,
    target: torch.Tensor,
    h: int,
    w: int
) -> bool:
    """
    Check if prediction exactly matches target.

    Args:
        pred: Predicted grid [H, W]
        target: Target grid [H, W]
        h: Actual output height
        w: Actual output width

    Returns:
        True if every pixel within (h, w) matches exactly
    """
    # Check shapes
    if pred.shape != target.shape:
        return False

    # Exact match within actual output size
    return torch.all(pred[:h, :w] == target[:h, :w]).item()


def compute_pixel_accuracy(
    pred: torch.Tensor,
    target: torch.Tensor,
    h: int,
    w: int
) -> float:
    """
    Compute pixel accuracy for analysis.

    Not used for gating decisions, only for diagnostic reporting.

    Args:
        pred: Predicted grid [H, W]
        target: Target grid [H, W]
        h: Actual output height
        w: Actual output width

    Returns:
        Fraction of pixels that match (0.0 to 1.0)
    """
    if pred.shape != target.shape:
        return 0.0

    return (pred[:h, :w] == target[:h, :w]).float().mean().item()


class TaskSolvedMetric:
    """
    Determine if a task is solved based on train pairs.

    Strategy: All train pairs must be perfectly solved (exact match)
    before we attempt test prediction.
    """

    def __init__(
        self,
        require_all_train: bool = True,
        check_constraints: bool = False  # TODO: implement constraint checking
    ):
        """
        Initialize task solved metric.

        Args:
            require_all_train: If True, all train pairs must be perfect.
                              If False, could relax to majority (not recommended).
            check_constraints: If True, also check constraint satisfaction.
        """
        self.require_all_train = require_all_train
        self.check_constraints = check_constraints

    def evaluate_train_pairs(
        self,
        predictions: List[torch.Tensor],
        targets: List[torch.Tensor],
        shapes: List[Tuple[int, int]]
    ) -> Tuple[bool, List[PairResult]]:
        """
        Evaluate if all train pairs are solved.

        Args:
            predictions: List of predicted grids [H, W]
            targets: List of target grids [H, W]
            shapes: List of (h, w) actual output sizes

        Returns:
            (all_solved, list of per-pair results)
        """
        if len(predictions) != len(targets) or len(predictions) != len(shapes):
            raise ValueError("Mismatched lengths: predictions, targets, shapes must match")

        results = []

        for pred, target, (h, w) in zip(predictions, targets, shapes):
            # Check exact match
            is_solved = exact_match(pred, target, h, w)

            # Compute pixel accuracy for analysis
            pixel_acc = compute_pixel_accuracy(pred, target, h, w)

            # TODO: Check constraints
            constraint_ok = True  # Placeholder

            results.append(PairResult(
                is_solved=is_solved,
                pixel_accuracy=pixel_acc,
                constraint_satisfied=constraint_ok,
                h=h,
                w=w
            ))

        # Determine if task is solved
        if self.require_all_train:
            all_solved = all(r.is_solved for r in results)
        else:
            # Relaxed: majority solved (not recommended)
            solved_count = sum(r.is_solved for r in results)
            all_solved = solved_count >= len(results) * 0.8

        return all_solved, results

    def evaluate_test_pairs(
        self,
        predictions: List[torch.Tensor],
        targets: List[torch.Tensor],
        shapes: List[Tuple[int, int]]
    ) -> Tuple[bool, List[bool]]:
        """
        Evaluate test pair correctness.

        Args:
            predictions: List of predicted grids
            targets: List of target grids
            shapes: List of (h, w) actual output sizes

        Returns:
            (all_correct, list of per-pair correctness)
        """
        if len(predictions) != len(targets) or len(predictions) != len(shapes):
            raise ValueError("Mismatched lengths")

        correct = []
        for pred, target, (h, w) in zip(predictions, targets, shapes):
            is_correct = exact_match(pred, target, h, w)
            correct.append(is_correct)

        all_correct = all(correct)
        return all_correct, correct


def compute_evaluation_metrics(
    results: List[TaskResult]
) -> EvaluationMetrics:
    """
    Compute aggregate metrics from task results.

    Args:
        results: List of TaskResult objects

    Returns:
        EvaluationMetrics with aggregate statistics
    """
    total_tasks = len(results)

    # Count successes at different levels
    num_tasks_solved = sum(1 for r in results if r.task_success)
    num_train_solved = sum(1 for r in results if r.train_solved)
    num_train_failed = sum(1 for r in results if not r.train_solved)

    # Categorize tasks
    tasks_solved = [r.task_id for r in results if r.task_success]
    tasks_train_only = [r.task_id for r in results
                        if r.train_solved and not r.task_success]
    tasks_failed = [r.task_id for r in results if not r.train_solved]

    # Primary metrics
    task_success_rate = num_tasks_solved / total_tasks if total_tasks > 0 else 0.0
    train_solved_rate = num_train_solved / total_tasks if total_tasks > 0 else 0.0

    # Test accuracy given train was solved
    test_attempts = [r for r in results if r.train_solved and r.test_correct is not None]
    if len(test_attempts) > 0:
        test_correct_count = sum(1 for r in test_attempts if r.task_success)
        test_accuracy_given_train = test_correct_count / len(test_attempts)
    else:
        test_accuracy_given_train = 0.0

    # Coverage: fraction of tasks where we attempted test
    num_attempted = sum(1 for r in results if r.test_attempted)
    coverage = num_attempted / total_tasks if total_tasks > 0 else 0.0

    # Per-pair statistics
    total_train_pairs = sum(len(r.train_results) for r in results)
    correct_train_pairs = sum(
        sum(1 for p in r.train_results if p.is_solved)
        for r in results
    )

    # Test pair statistics (only when test_correct is available)
    total_test_pairs = 0
    correct_test_pairs = 0
    for r in results:
        if r.test_correct is not None:
            total_test_pairs += len(r.test_correct)
            correct_test_pairs += sum(r.test_correct)

    # Phase 5B: Competition scoring (multi-attempt)
    competition_scores = [r.competition_score for r in results if r.competition_score is not None]
    if competition_scores:
        mean_competition_score = sum(competition_scores) / len(competition_scores)
        # Infer num_attempts from first result with attempts
        num_attempts = None
        for r in results:
            if r.test_attempts is not None and len(r.test_attempts) > 0:
                num_attempts = len(r.test_attempts[0])  # K attempts per test input
                break
    else:
        mean_competition_score = None
        num_attempts = None

    # Active Learning: Adaptation statistics
    adaptation_steps_list = [r.adaptation_steps for r in results if r.adaptation_steps is not None]
    adaptation_time_list = [r.adaptation_time for r in results if r.adaptation_time is not None]
    adaptation_converged_list = [r.adaptation_converged for r in results if r.adaptation_converged is not None]

    if adaptation_steps_list:
        mean_adaptation_steps = sum(adaptation_steps_list) / len(adaptation_steps_list)
        mean_adaptation_time = sum(adaptation_time_list) / len(adaptation_time_list) if adaptation_time_list else None
        adaptation_convergence_rate = sum(adaptation_converged_list) / len(adaptation_converged_list) if adaptation_converged_list else None
    else:
        mean_adaptation_steps = None
        mean_adaptation_time = None
        adaptation_convergence_rate = None

    return EvaluationMetrics(
        task_success_rate=task_success_rate,
        train_solved_rate=train_solved_rate,
        test_accuracy_given_train=test_accuracy_given_train,
        coverage=coverage,
        total_tasks=total_tasks,
        num_tasks_solved=num_tasks_solved,
        num_train_solved=num_train_solved,
        num_train_failed=num_train_failed,
        tasks_solved=tasks_solved,
        tasks_train_only=tasks_train_only,
        tasks_failed=tasks_failed,
        total_train_pairs=total_train_pairs,
        correct_train_pairs=correct_train_pairs,
        total_test_pairs=total_test_pairs,
        correct_test_pairs=correct_test_pairs,
        mean_competition_score=mean_competition_score,
        num_competition_attempts=num_attempts,
        mean_adaptation_steps=mean_adaptation_steps,
        mean_adaptation_time=mean_adaptation_time,
        adaptation_convergence_rate=adaptation_convergence_rate
    )


def print_evaluation_summary(
    metrics: EvaluationMetrics,
    verbose: bool = True
):
    """
    Print a formatted summary of evaluation metrics.

    Args:
        metrics: EvaluationMetrics object
        verbose: If True, print detailed breakdown
    """
    print("\n" + "="*60)
    print("EVALUATION SUMMARY")
    print("="*60)

    print(f"\n{'PRIMARY METRICS':-^60}")
    print(f"  Task Success Rate:     {metrics.task_success_rate:6.1%}  ({metrics.num_tasks_solved}/{metrics.total_tasks} tasks)")
    print(f"  Train Solved Rate:     {metrics.train_solved_rate:6.1%}  ({metrics.num_train_solved}/{metrics.total_tasks} tasks)")
    print(f"  Test Acc (given train):{metrics.test_accuracy_given_train:6.1%}")
    print(f"  Coverage:              {metrics.coverage:6.1%}  (test attempted)")

    # Phase 5B: Competition scoring
    if metrics.mean_competition_score is not None:
        print(f"\n{'COMPETITION SCORING (Multi-Attempt)':-^60}")
        print(f"  Competition Score:     {metrics.mean_competition_score:6.1%}  (avg across tasks)")
        print(f"  Attempts per test:     {metrics.num_competition_attempts}")
        gain = (metrics.mean_competition_score - metrics.task_success_rate) * 100
        print(f"  Gain from multi-attempt: +{gain:.1f}% points")

    # Active Learning: Adaptation statistics
    if metrics.mean_adaptation_steps is not None:
        print(f"\n{'ACTIVE LEARNING ADAPTATION':-^60}")
        print(f"  Avg adaptation steps:  {metrics.mean_adaptation_steps:6.1f}")
        if metrics.mean_adaptation_time is not None:
            print(f"  Avg adaptation time:   {metrics.mean_adaptation_time:6.1f}s")
        if metrics.adaptation_convergence_rate is not None:
            print(f"  Convergence rate:      {metrics.adaptation_convergence_rate:6.1%}  (train solved during adaptation)")

    print(f"\n{'TASK BREAKDOWN':-^60}")
    print(f"  ✓ Fully Solved:        {metrics.num_tasks_solved:3d}  (train + test correct)")
    print(f"  ⚠ Train Only:          {len(metrics.tasks_train_only):3d}  (train solved, test wrong)")
    print(f"  ✗ Failed:              {metrics.num_train_failed:3d}  (train not solved)")

    print(f"\n{'PAIR-LEVEL STATISTICS':-^60}")
    train_pair_acc = metrics.correct_train_pairs / metrics.total_train_pairs if metrics.total_train_pairs > 0 else 0.0
    print(f"  Train Pairs:           {metrics.correct_train_pairs}/{metrics.total_train_pairs} ({train_pair_acc:.1%})")

    if metrics.total_test_pairs > 0:
        test_pair_acc = metrics.correct_test_pairs / metrics.total_test_pairs
        print(f"  Test Pairs:            {metrics.correct_test_pairs}/{metrics.total_test_pairs} ({test_pair_acc:.1%})")
    else:
        print(f"  Test Pairs:            N/A (no test outputs available)")

    if verbose and len(metrics.tasks_solved) > 0:
        print(f"\n{'SOLVED TASKS':-^60}")
        for task_id in metrics.tasks_solved[:10]:  # Show first 10
            print(f"  ✓ {task_id}")
        if len(metrics.tasks_solved) > 10:
            print(f"  ... and {len(metrics.tasks_solved) - 10} more")

    print("\n" + "="*60 + "\n")


# Test code
if __name__ == "__main__":
    print("="*60)
    print("Testing Evaluation Metrics")
    print("="*60)

    # Test 1: exact_match
    print("\n1. Testing exact_match()...")

    # Perfect match
    pred = torch.tensor([[1, 2], [3, 4]])
    target = torch.tensor([[1, 2], [3, 4]])
    assert exact_match(pred, target, 2, 2) == True, "Perfect match should return True"
    print("  ✓ Perfect match: PASS")

    # One pixel off
    pred = torch.tensor([[1, 2], [3, 4]])
    target = torch.tensor([[1, 2], [3, 5]])  # Last pixel different
    assert exact_match(pred, target, 2, 2) == False, "Mismatch should return False"
    print("  ✓ One pixel off: PASS")

    # Partial grid match (different h,w)
    pred = torch.tensor([[1, 2, 9], [3, 4, 9], [9, 9, 9]])
    target = torch.tensor([[1, 2, 8], [3, 4, 8], [8, 8, 8]])
    assert exact_match(pred, target, 2, 2) == True, "Should match within (2,2)"
    print("  ✓ Partial grid match: PASS")

    # Test 2: compute_pixel_accuracy
    print("\n2. Testing compute_pixel_accuracy()...")

    pred = torch.tensor([[1, 2], [3, 4]])
    target = torch.tensor([[1, 2], [3, 5]])
    acc = compute_pixel_accuracy(pred, target, 2, 2)
    assert abs(acc - 0.75) < 0.01, f"Expected 0.75, got {acc}"
    print(f"  ✓ Pixel accuracy: {acc:.2f} (3/4 correct) PASS")

    # Test 3: TaskSolvedMetric
    print("\n3. Testing TaskSolvedMetric...")

    metric = TaskSolvedMetric(require_all_train=True)

    # All train pairs solved
    preds = [
        torch.tensor([[1, 2], [3, 4]]),
        torch.tensor([[5, 6], [7, 8]])
    ]
    targets = [
        torch.tensor([[1, 2], [3, 4]]),
        torch.tensor([[5, 6], [7, 8]])
    ]
    shapes = [(2, 2), (2, 2)]

    all_solved, results = metric.evaluate_train_pairs(preds, targets, shapes)
    assert all_solved == True, "All train pairs perfect should return True"
    assert all(r.is_solved for r in results), "All pairs should be marked solved"
    print("  ✓ All train pairs solved: PASS")

    # One train pair wrong
    preds = [
        torch.tensor([[1, 2], [3, 4]]),
        torch.tensor([[5, 6], [7, 9]])  # Wrong
    ]

    all_solved, results = metric.evaluate_train_pairs(preds, targets, shapes)
    assert all_solved == False, "One wrong pair should return False"
    assert results[0].is_solved == True, "First pair should be solved"
    assert results[1].is_solved == False, "Second pair should not be solved"
    print("  ✓ One train pair wrong: PASS")

    # Test 4: compute_evaluation_metrics
    print("\n4. Testing compute_evaluation_metrics()...")

    # Create mock results
    mock_results = [
        # Task 1: Fully solved
        TaskResult(
            task_id="task1",
            train_solved=True,
            train_results=[PairResult(True, 1.0, True, 2, 2)],
            test_attempted=True,
            test_predictions=[torch.zeros(2, 2)],
            test_correct=[True],
            task_success=True,
            confidence=1.0
        ),
        # Task 2: Train solved, test wrong
        TaskResult(
            task_id="task2",
            train_solved=True,
            train_results=[PairResult(True, 1.0, True, 2, 2)],
            test_attempted=True,
            test_predictions=[torch.zeros(2, 2)],
            test_correct=[False],
            task_success=False,
            confidence=1.0
        ),
        # Task 3: Train failed
        TaskResult(
            task_id="task3",
            train_solved=False,
            train_results=[PairResult(False, 0.7, True, 2, 2)],
            test_attempted=False,
            test_predictions=None,
            test_correct=None,
            task_success=False,
            confidence=0.0
        ),
    ]

    metrics = compute_evaluation_metrics(mock_results)

    assert metrics.total_tasks == 3, "Should have 3 tasks"
    assert metrics.num_tasks_solved == 1, "Should have 1 fully solved"
    assert metrics.num_train_solved == 2, "Should have 2 with train solved"
    assert abs(metrics.task_success_rate - 1/3) < 0.01, "Task success should be 1/3"
    assert abs(metrics.train_solved_rate - 2/3) < 0.01, "Train solved should be 2/3"
    assert abs(metrics.test_accuracy_given_train - 0.5) < 0.01, "Test acc given train should be 0.5"

    print("  ✓ Aggregate metrics: PASS")
    print(f"    Task success: {metrics.task_success_rate:.1%}")
    print(f"    Train solved: {metrics.train_solved_rate:.1%}")
    print(f"    Test acc (given train): {metrics.test_accuracy_given_train:.1%}")

    # Test 5: print_evaluation_summary
    print("\n5. Testing print_evaluation_summary()...")
    print_evaluation_summary(metrics, verbose=False)

    print("\n" + "="*60)
    print("✓ All tests passed!")
    print("="*60)
