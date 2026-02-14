#!/usr/bin/env python3
"""
Integration test for the full training pipeline.

Tests inner loop + outer loop + controller training without requiring
pretrained checkpoints.
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

import torch
import torch.nn as nn

from arc_nodsl.data.loader import ARCDataset
from arc_nodsl.models.slots import SlotEncoder
from arc_nodsl.models.renderer import SlotRenderer
from arc_nodsl.models.operators import OperatorLibrary
from arc_nodsl.models.controller import Controller
from arc_nodsl.training.inner_loop import InnerLoop
from arc_nodsl.training.outer_loop import OuterLoop


def test_full_pipeline():
    """
    Test the complete training pipeline with minimal epochs/steps.
    """
    print("="*60)
    print("Full Training Pipeline Integration Test")
    print("="*60)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"\nDevice: {device}")

    # Load dataset
    print("\nLoading dataset...")
    dataset = ARCDataset("data/arc-agi_training_challenges.json")
    print(f"Tasks: {len(dataset)}")

    # Sample 3 tasks for quick test
    tasks = [dataset[i] for i in range(3)]
    print(f"Test with {len(tasks)} tasks")

    # Create models (random weights, no pretraining for fast test)
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

    # Set modes
    encoder.eval()
    renderer.eval()
    operators.train()  # Training operators
    controller.train()  # Training controller

    print("\nModel configuration:")
    print(f"  Encoder: {sum(p.numel() for p in encoder.parameters()):,} params")
    print(f"  Renderer: {sum(p.numel() for p in renderer.parameters()):,} params")
    print(f"  Operators: {sum(p.numel() for p in operators.parameters()):,} params (trainable)")
    print(f"  Controller: {sum(p.numel() for p in controller.parameters()):,} params (trainable)")

    # Create training loops
    print("\nCreating training loops...")
    inner_loop = InnerLoop(
        num_inner_steps=2,  # Very short for fast test
        beam_size=4,
        max_operator_steps=2,
        learning_rate=1e-3,
        device=device
    )

    outer_loop = OuterLoop(
        inner_loop=inner_loop,
        meta_learning_rate=1e-3,
        meta_batch_size=2,
        device=device
    )

    print("\n" + "="*60)
    print("Running Meta-Training Steps")
    print("="*60)

    # Run 2 meta-training steps
    for step in range(1, 3):
        print(f"\nMeta-Training Step {step}/2")
        print("-"*60)

        # Sample tasks for this step
        if step == 1:
            batch_tasks = tasks[:2]
        else:
            batch_tasks = tasks[1:3]

        # Meta-training step
        metrics = outer_loop.meta_train_step(
            batch_tasks, encoder, controller, operators, renderer,
            verbose=True
        )

        print(f"\nStep {step} Summary:")
        print(f"  Meta-loss: {metrics.meta_loss:.4f}")
        print(f"  Test reward: {metrics.mean_test_reward:.3f}")
        print(f"  Train reward: {metrics.mean_train_reward:.3f}")
        print(f"  Test success: {metrics.test_success_rate:.1%}")
        print(f"  Train success: {metrics.train_success_rate:.1%}")

    print("\n" + "="*60)
    print("✓ Full Pipeline Test Complete!")
    print("="*60)
    print("\nAll components working:")
    print("  ✓ Data loading")
    print("  ✓ Model creation")
    print("  ✓ Inner loop (REINFORCE)")
    print("  ✓ Outer loop (Reptile)")
    print("  ✓ Meta-training integration")
    print("\nReady for full-scale training with pretrained autoencoder!")


if __name__ == "__main__":
    test_full_pipeline()
