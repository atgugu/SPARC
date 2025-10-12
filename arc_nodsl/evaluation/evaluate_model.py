#!/usr/bin/env python3
"""
Comprehensive model evaluation script.

Evaluates trained SPARC model on ARC validation set with task-level metrics.
Uses train-first gating: only predict test when all train pairs are solved.
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

import torch
import torch.nn as nn
import argparse
import json
from tqdm import tqdm
from datetime import datetime
from typing import List, Dict, Optional

from arc_nodsl.data.loader import ARCDataset
from arc_nodsl.models.slots import SlotEncoder
from arc_nodsl.models.renderer import SlotRenderer
from arc_nodsl.models.operators import OperatorLibrary
from arc_nodsl.models.controller import Controller
from arc_nodsl.evaluation.solver import ARCSolver, TTASolver
from arc_nodsl.evaluation.active_solver import ActiveARCSolver
from arc_nodsl.evaluation.metrics import (
    TaskResult, compute_evaluation_metrics, print_evaluation_summary
)


def load_pretrained_autoencoder(
    checkpoint_path: str,
    device: torch.device
) -> tuple:
    """
    Load pretrained encoder and renderer from checkpoint.

    Args:
        checkpoint_path: Path to autoencoder checkpoint
        device: Compute device

    Returns:
        (encoder, renderer) tuple
    """
    checkpoint = torch.load(checkpoint_path, map_location=device)

    # Create models (match pretraining config)
    encoder = SlotEncoder(
        num_slots=8,
        d_color=16,
        d_feat=64,
        d_slot=128,
        num_iters=3,
        H=30,
        W=30
    ).to(device)

    renderer = SlotRenderer(
        d_slot=128,
        d_hidden=64,
        H=30,
        W=30,
        use_mask=True
    ).to(device)

    # Load from AutoEncoder checkpoint
    full_state = checkpoint['model_state_dict']

    # Split into encoder and renderer state dicts
    encoder_state = {}
    renderer_state = {}

    for key, value in full_state.items():
        if key.startswith('encoder.'):
            encoder_state[key[8:]] = value  # Remove 'encoder.' prefix
        elif key.startswith('renderer.'):
            renderer_state[key[9:]] = value  # Remove 'renderer.' prefix

    encoder.load_state_dict(encoder_state)
    renderer.load_state_dict(renderer_state)

    encoder.eval()
    renderer.eval()

    # Freeze weights
    for param in encoder.parameters():
        param.requires_grad = False
    for param in renderer.parameters():
        param.requires_grad = False

    print(f"✓ Loaded pretrained autoencoder from {checkpoint_path}")
    if 'val_acc' in checkpoint:
        print(f"  Validation accuracy: {checkpoint['val_acc']*100:.2f}%")

    return encoder, renderer


def load_trained_controller(
    checkpoint_path: str,
    num_ops: int,
    device: torch.device
) -> tuple:
    """
    Load trained controller and operators from checkpoint.

    Args:
        checkpoint_path: Path to controller checkpoint
        num_ops: Number of operators
        device: Compute device

    Returns:
        (controller, operators) tuple
    """
    checkpoint = torch.load(checkpoint_path, map_location=device)

    # Create models
    controller = Controller(
        num_operators=num_ops,
        d_slot=128,
        d_task=128,
        d_hidden=256,
        max_steps=4
    ).to(device)

    operators = OperatorLibrary(
        num_ops=num_ops,
        d_slot=128,
        d_hidden=128,
        H=30,
        W=30
    ).to(device)

    # Load weights
    controller.load_state_dict(checkpoint['controller_state_dict'])
    operators.load_state_dict(checkpoint['operators_state_dict'])

    controller.eval()
    operators.eval()

    # Freeze weights
    for param in controller.parameters():
        param.requires_grad = False
    for param in operators.parameters():
        param.requires_grad = False

    print(f"✓ Loaded trained controller from {checkpoint_path}")
    if 'test_reward' in checkpoint:
        print(f"  Test reward: {checkpoint['test_reward']:.3f}")

    return controller, operators


def save_evaluation_results(
    results: List[TaskResult],
    metrics: 'EvaluationMetrics',
    output_dir: Path
):
    """
    Save evaluation results to JSON files.

    Args:
        results: List of TaskResult objects
        metrics: EvaluationMetrics
        output_dir: Output directory
    """
    output_dir.mkdir(exist_ok=True, parents=True)

    # Save summary metrics
    summary = {
        'timestamp': datetime.now().isoformat(),
        'metrics': {
            'task_success_rate': metrics.task_success_rate,
            'train_solved_rate': metrics.train_solved_rate,
            'test_accuracy_given_train': metrics.test_accuracy_given_train,
            'coverage': metrics.coverage,
            'total_tasks': metrics.total_tasks,
            'num_tasks_solved': metrics.num_tasks_solved,
            'num_train_solved': metrics.num_train_solved,
            'num_train_failed': metrics.num_train_failed,
        },
        'tasks_solved': metrics.tasks_solved,
        'tasks_train_only': metrics.tasks_train_only,
        'tasks_failed': metrics.tasks_failed,
    }

    summary_path = output_dir / 'summary.json'
    with open(summary_path, 'w') as f:
        json.dump(summary, f, indent=2)
    print(f"\n✓ Saved summary to {summary_path}")

    # Save detailed per-task results
    detailed = []
    for result in results:
        detailed.append({
            'task_id': result.task_id,
            'train_solved': result.train_solved,
            'test_attempted': result.test_attempted,
            'task_success': result.task_success,
            'confidence': result.confidence,
            'train_pairs': [{
                'is_solved': r.is_solved,
                'pixel_accuracy': r.pixel_accuracy
            } for r in result.train_results],
            'test_correct': result.test_correct if result.test_correct else []
        })

    detailed_path = output_dir / 'detailed_results.json'
    with open(detailed_path, 'w') as f:
        json.dump(detailed, f, indent=2)
    print(f"✓ Saved detailed results to {detailed_path}")


def evaluate_model(
    autoencoder_checkpoint: str,
    controller_checkpoint: str,
    dataset_path: str = "data/arc-agi_evaluation_challenges.json",
    output_dir: str = "evaluation_results",
    beam_size: int = 16,
    max_steps: int = 8,
    num_ops: int = 8,
    num_attempts: int = 2,  # Phase 5B: multi-attempt (competition = 2)
    save_results: bool = True,
    num_tasks: Optional[int] = None,
    verbose: bool = False,
    # Active Learning params
    active_learning: bool = False,
    adaptation_steps: int = 20,
    adaptation_lr: float = 1e-3,
    time_budget: float = 60.0,
    beam_size_adaptation: int = 8,
    # Test-Time Augmentation params
    tta: bool = False,
    tta_mode: str = 'majority',
    tta_color: bool = False,
    tta_color_variants: int = 4
) -> 'EvaluationMetrics':
    """
    Evaluate trained model on validation set.

    Args:
        autoencoder_checkpoint: Path to pretrained autoencoder
        controller_checkpoint: Path to trained controller
        dataset_path: Path to validation dataset
        output_dir: Directory for saving results
        beam_size: Beam size for search (16 recommended for eval)
        max_steps: Max operator sequence length (8 for complex tasks)
        num_ops: Number of operators in library
        num_attempts: Number of attempts per test input (2 for competition)
        save_results: Whether to save detailed results
        num_tasks: If provided, only evaluate first N tasks (for testing)
        verbose: Print per-task progress
        active_learning: Use active inference (adapt on each task)
        adaptation_steps: Max gradient steps for adaptation
        adaptation_lr: Learning rate for adaptation
        time_budget: Max time for adaptation per task (seconds)
        beam_size_adaptation: Beam size during adaptation (smaller = faster)
        tta: Use test-time augmentation ensemble (8x slower, +2-5% accuracy)
        tta_mode: Ensemble mode ('majority' or 'first_success')
        tta_color: Use color TTA (ensemble across color permutations, requires tta=True)
        tta_color_variants: Number of color permutations for TTA (default 4)

    Returns:
        EvaluationMetrics with aggregate statistics
    """
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Device: {device}\n")

    # Load models
    print("="*60)
    print("Loading Models")
    print("="*60)

    encoder, renderer = load_pretrained_autoencoder(autoencoder_checkpoint, device)
    controller, operators = load_trained_controller(controller_checkpoint, num_ops, device)

    # Create solver (Active, TTA, or Passive)
    if active_learning and tta:
        raise ValueError("Cannot use both active_learning and tta simultaneously. Choose one.")

    if active_learning:
        print("\nCreating ActiveARCSolver (with test-time adaptation)...")
        solver = ActiveARCSolver(
            encoder, controller, operators, renderer,
            beam_size=beam_size,
            max_operator_steps=max_steps,
            num_attempts=num_attempts,
            adaptation_steps=adaptation_steps,
            adaptation_lr=adaptation_lr,
            time_budget_seconds=time_budget,
            beam_size_adaptation=beam_size_adaptation,
            device=device
        )
        print(f"✓ ActiveARCSolver initialized")
        print(f"  Inference: beam={beam_size}, max_steps={max_steps}, attempts={num_attempts}")
        print(f"  Adaptation: steps={adaptation_steps}, lr={adaptation_lr}, budget={time_budget}s, beam={beam_size_adaptation}")
    elif tta:
        print("\nCreating TTASolver (with test-time augmentation ensemble)...")
        solver = TTASolver(
            encoder, controller, operators, renderer,
            beam_size=beam_size,
            max_operator_steps=max_steps,
            num_attempts=num_attempts,
            tta_mode=tta_mode,
            tta_color=tta_color,
            tta_color_variants=tta_color_variants,
            device=device
        )
        print(f"✓ TTASolver initialized")
        print(f"  Inference: beam={beam_size}, max_steps={max_steps}, attempts={num_attempts}")
        print(f"  TTA mode: {tta_mode}")
        if tta_color:
            print(f"  Color TTA: ENABLED (variants={tta_color_variants})")
            print(f"  WARNING: {8*tta_color_variants}x slower inference due to spatial+color TTA")
        else:
            print(f"  Color TTA: DISABLED")
            print(f"  WARNING: 8x slower inference due to spatial TTA ensemble")
    else:
        print("\nCreating ARCSolver (passive inference)...")
        solver = ARCSolver(
            encoder, controller, operators, renderer,
            beam_size=beam_size,
            max_operator_steps=max_steps,
            num_attempts=num_attempts,
            device=device
        )
        print(f"✓ ARCSolver initialized (beam_size={beam_size}, max_steps={max_steps}, attempts={num_attempts})")

    # Load dataset
    print("\n" + "="*60)
    print("Loading Dataset")
    print("="*60)

    dataset = ARCDataset(dataset_path)
    if num_tasks is not None:
        dataset = dataset[:num_tasks]
        print(f"Evaluating on first {num_tasks} tasks (test mode)")
    else:
        print(f"Evaluating on {len(dataset)} tasks")

    # Evaluate each task
    print("\n" + "="*60)
    print("Running Evaluation")
    print("="*60 + "\n")

    results = []

    for task_data in tqdm(dataset, desc="Evaluating tasks"):
        result = solver.solve_task(task_data, verbose=verbose)
        results.append(result)

        # Print progress for solved tasks
        if result.task_success and not verbose:
            tqdm.write(f"✓ {result.task_id}: SOLVED!")

    # Compute aggregate metrics
    print("\n" + "="*60)
    print("Computing Metrics")
    print("="*60)

    metrics = compute_evaluation_metrics(results)

    # Print summary
    print_evaluation_summary(metrics, verbose=True)

    # Save results
    if save_results:
        output_path = Path(output_dir)
        save_evaluation_results(results, metrics, output_path)

    return metrics


def main():
    parser = argparse.ArgumentParser(
        description="Evaluate SPARC model on ARC validation set"
    )
    parser.add_argument(
        "--autoencoder_checkpoint",
        type=str,
        required=True,
        help="Path to pretrained autoencoder checkpoint"
    )
    parser.add_argument(
        "--controller_checkpoint",
        type=str,
        required=True,
        help="Path to trained controller checkpoint"
    )
    parser.add_argument(
        "--dataset",
        type=str,
        default="data/arc-agi_evaluation_challenges.json",
        help="Path to validation dataset"
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="evaluation_results",
        help="Directory for saving results"
    )
    parser.add_argument(
        "--beam_size",
        type=int,
        default=16,
        help="Beam size for search (larger = better but slower)"
    )
    parser.add_argument(
        "--max_steps",
        type=int,
        default=8,
        help="Max operator sequence length"
    )
    parser.add_argument(
        "--num_ops",
        type=int,
        default=8,
        help="Number of operators in library"
    )
    parser.add_argument(
        "--num_attempts",
        type=int,
        default=2,
        help="Number of attempts per test input (competition = 2)"
    )
    parser.add_argument(
        "--num_tasks",
        type=int,
        default=None,
        help="Evaluate only first N tasks (for testing)"
    )
    parser.add_argument(
        "--verbose",
        action="store_true",
        help="Print per-task detailed output"
    )
    parser.add_argument(
        "--no_save",
        action="store_true",
        help="Don't save results to disk"
    )
    # Active Learning arguments
    parser.add_argument(
        "--active_learning",
        action="store_true",
        help="Use active inference (adapt controller on each task)"
    )
    parser.add_argument(
        "--adaptation_steps",
        type=int,
        default=20,
        help="Max gradient steps for adaptation (default: 20)"
    )
    parser.add_argument(
        "--adaptation_lr",
        type=float,
        default=1e-3,
        help="Learning rate for adaptation (default: 1e-3)"
    )
    parser.add_argument(
        "--time_budget",
        type=float,
        default=60.0,
        help="Max time for adaptation per task in seconds (default: 60)"
    )
    parser.add_argument(
        "--beam_size_adaptation",
        type=int,
        default=8,
        help="Beam size during adaptation (smaller = faster, default: 8)"
    )
    # Test-Time Augmentation (TTA) arguments
    parser.add_argument(
        "--tta",
        action="store_true",
        help="Use test-time augmentation ensemble (8x slower, +2-5%% accuracy)"
    )
    parser.add_argument(
        "--tta_mode",
        type=str,
        default="majority",
        choices=["majority", "first_success"],
        help="TTA ensemble mode: majority voting or first success (default: majority)"
    )
    parser.add_argument(
        "--tta_color",
        action="store_true",
        help="Use color TTA (ensemble across color permutations, requires --tta)"
    )
    parser.add_argument(
        "--tta_color_variants",
        type=int,
        default=4,
        help="Number of color permutations for TTA (default: 4, used with --tta_color)"
    )

    args = parser.parse_args()

    # Run evaluation
    metrics = evaluate_model(
        autoencoder_checkpoint=args.autoencoder_checkpoint,
        controller_checkpoint=args.controller_checkpoint,
        dataset_path=args.dataset,
        output_dir=args.output_dir,
        beam_size=args.beam_size,
        max_steps=args.max_steps,
        num_ops=args.num_ops,
        num_attempts=args.num_attempts,
        save_results=not args.no_save,
        num_tasks=args.num_tasks,
        verbose=args.verbose,
        active_learning=args.active_learning,
        adaptation_steps=args.adaptation_steps,
        adaptation_lr=args.adaptation_lr,
        time_budget=args.time_budget,
        beam_size_adaptation=args.beam_size_adaptation,
        tta=args.tta,
        tta_mode=args.tta_mode,
        tta_color=args.tta_color,
        tta_color_variants=args.tta_color_variants
    )

    print("\n" + "="*60)
    print("Evaluation Complete!")
    print("="*60)
    print(f"\nPrimary Result: {metrics.task_success_rate:.1%} task success rate")
    print(f"({metrics.num_tasks_solved}/{metrics.total_tasks} tasks fully solved)\n")


def test_evaluation():
    """Test evaluation pipeline with random controller (no trained checkpoint needed)."""
    print("="*60)
    print("Testing Evaluation Pipeline")
    print("="*60 + "\n")

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # Load pretrained autoencoder
    print("Loading pretrained autoencoder...")
    encoder, renderer = load_pretrained_autoencoder(
        "checkpoints/autoencoder_best.pt",
        device
    )

    # Create random controller and operators (not trained)
    print("\nCreating random controller and operators (not trained)...")
    controller = Controller(
        num_operators=8, d_slot=128, d_task=128,
        d_hidden=256, max_steps=4
    ).to(device)
    controller.eval()

    operators = OperatorLibrary(
        num_ops=8, d_slot=128, d_hidden=128, H=30, W=30
    ).to(device)
    operators.eval()

    print("✓ Models created")

    # Create solver
    print("\nCreating ARCSolver...")
    solver = ARCSolver(
        encoder, controller, operators, renderer,
        beam_size=4,  # Small for fast testing
        max_operator_steps=2,
        num_attempts=2,  # Phase 5B: competition setting
        device=device
    )

    # Load small subset
    print("\nLoading validation dataset (3 tasks for testing)...")
    dataset = ARCDataset("data/arc-agi_evaluation_challenges.json")
    test_tasks = [dataset[i] for i in range(3)]

    # Evaluate
    print("\n" + "="*60)
    print("Running Evaluation (3 tasks)")
    print("="*60 + "\n")

    results = []
    for task_data in test_tasks:
        result = solver.solve_task(task_data, verbose=False)
        results.append(result)
        print(f"Task {result.task_id}: train_solved={result.train_solved}, task_success={result.task_success}")

    # Compute metrics
    metrics = compute_evaluation_metrics(results)
    print_evaluation_summary(metrics, verbose=False)

    print("\n✓ Evaluation pipeline test complete!")
    print("\nNote: With random weights, expect 0% success.")
    print("After training, expect 10-30% task success rate.")


if __name__ == "__main__":
    import sys
    if len(sys.argv) > 1 and sys.argv[1] == "--test":
        test_evaluation()
    else:
        main()
