#!/usr/bin/env python3
"""
Train the controller via meta-learning.

Requires pretrained autoencoder (encoder + renderer).
Trains controller and operators using REINFORCE + Reptile meta-learning.
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

import torch
import torch.nn as nn
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm
import argparse
from datetime import datetime
import numpy as np

from arc_nodsl.data.loader import ARCDataset
from arc_nodsl.data.augment import augment_task, random_transform, augment_task_spatial_and_colors
from arc_nodsl.models.slots import SlotEncoder
from arc_nodsl.models.renderer import SlotRenderer
from arc_nodsl.models.operators import OperatorLibrary
from arc_nodsl.models.controller import Controller
from arc_nodsl.training.inner_loop import InnerLoop
from arc_nodsl.training.outer_loop import OuterLoop


def load_pretrained_autoencoder(checkpoint_path: str, device: torch.device):
    """Load pretrained encoder and renderer from checkpoint."""
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
    # The checkpoint contains the full autoencoder, we need to extract encoder/renderer
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
    print(f"  Validation accuracy: {checkpoint.get('val_acc', 'N/A')}")

    return encoder, renderer


def main():
    parser = argparse.ArgumentParser(description="Train controller via meta-learning")
    parser.add_argument("--autoencoder_checkpoint", type=str, required=True,
                        help="Path to pretrained autoencoder checkpoint")
    parser.add_argument("--data_train", type=str, default="data/arc-agi_training_challenges.json")
    parser.add_argument("--meta_epochs", type=int, default=100,
                        help="Number of meta-training epochs")
    parser.add_argument("--meta_batch_size", type=int, default=4,
                        help="Number of tasks per meta-batch")
    parser.add_argument("--inner_steps", type=int, default=10,
                        help="Number of inner loop gradient steps per task")
    parser.add_argument("--beam_size", type=int, default=8)
    parser.add_argument("--max_operator_steps", type=int, default=4)
    parser.add_argument("--meta_lr", type=float, default=1e-4,
                        help="Meta-learning rate for Reptile")
    parser.add_argument("--inner_lr", type=float, default=1e-3,
                        help="Inner loop learning rate")
    parser.add_argument("--num_ops", type=int, default=8,
                        help="Number of operators in library")
    parser.add_argument("--checkpoint_dir", type=str, default="checkpoints")
    parser.add_argument("--log_dir", type=str, default="runs")
    parser.add_argument("--save_every", type=int, default=10)
    parser.add_argument("--eval_every", type=int, default=5)
    # Data augmentation arguments
    parser.add_argument("--augment", action="store_true",
                        help="Apply random spatial transformations to tasks (rotations, flips)")
    parser.add_argument("--augment_prob", type=float, default=0.5,
                        help="Probability of augmenting each task (default: 0.5)")
    parser.add_argument("--augment_exclude_identity", action="store_true",
                        help="Exclude identity transform for maximum diversity")
    parser.add_argument("--color_aug_multiplier", type=int, default=0,
                        help="Create N additional color-augmented variants per task (0=disabled, 50=50 variants)")
    args = parser.parse_args()

    # Setup
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_dir = Path(args.log_dir) / f"controller_{timestamp}"
    checkpoint_dir = Path(args.checkpoint_dir)
    checkpoint_dir.mkdir(exist_ok=True, parents=True)

    writer = SummaryWriter(log_dir)
    print(f"Logging to {log_dir}")

    # Load pretrained autoencoder
    print("\nLoading pretrained autoencoder...")
    encoder, renderer = load_pretrained_autoencoder(args.autoencoder_checkpoint, device)

    # Load training data
    print("\nLoading training dataset...")
    dataset = ARCDataset(args.data_train)
    print(f"Training tasks: {len(dataset)}")

    # Augmentation status
    if args.augment or args.color_aug_multiplier > 0:
        print(f"\nData Augmentation:")
        if args.augment:
            print(f"  Spatial: ENABLED (prob={args.augment_prob}, exclude_identity={args.augment_exclude_identity})")
        else:
            print(f"  Spatial: DISABLED")

        if args.color_aug_multiplier > 0:
            print(f"  Color: ENABLED (multiplier={args.color_aug_multiplier})")
            print(f"    Note: Each base task generates {args.color_aug_multiplier} additional variants")
            print(f"    Each variant has random spatial transform + random color permutation")
        else:
            print(f"  Color: DISABLED")
    else:
        print(f"\nData Augmentation: DISABLED")

    # Create trainable models
    print("\nCreating controller and operators...")
    operators = OperatorLibrary(
        num_ops=args.num_ops,
        d_slot=128,
        d_hidden=128,
        H=30,
        W=30
    ).to(device)

    controller = Controller(
        num_operators=args.num_ops,
        d_slot=128,
        d_task=128,
        d_hidden=256,
        max_steps=args.max_operator_steps
    ).to(device)

    # Count parameters
    n_ops_params = sum(p.numel() for p in operators.parameters() if p.requires_grad)
    n_ctrl_params = sum(p.numel() for p in controller.parameters() if p.requires_grad)
    print(f"Operators parameters: {n_ops_params:,}")
    print(f"Controller parameters: {n_ctrl_params:,}")
    print(f"Total trainable: {n_ops_params + n_ctrl_params:,}")

    # Create training loops
    inner_loop = InnerLoop(
        num_inner_steps=args.inner_steps,
        beam_size=args.beam_size,
        max_operator_steps=args.max_operator_steps,
        learning_rate=args.inner_lr,
        device=device
    )

    outer_loop = OuterLoop(
        inner_loop=inner_loop,
        meta_learning_rate=args.meta_lr,
        meta_batch_size=args.meta_batch_size,
        device=device
    )

    # Training state
    best_test_reward = 0.0
    global_step = 0

    print("\nStarting meta-training...")
    print(f"Meta-epochs: {args.meta_epochs}")
    print(f"Meta-batch size: {args.meta_batch_size}")
    print(f"Inner steps: {args.inner_steps}")
    print("="*60)

    for epoch in range(1, args.meta_epochs + 1):
        print(f"\nEpoch {epoch}/{args.meta_epochs}")

        # Sample random tasks for this epoch
        task_indices = np.random.choice(len(dataset), size=len(dataset), replace=False)

        epoch_metrics = {
            'meta_loss': [],
            'test_reward': [],
            'train_reward': [],
            'test_success': [],
            'train_success': []
        }

        # Meta-training steps
        num_meta_steps = len(dataset) // args.meta_batch_size
        pbar = tqdm(range(num_meta_steps), desc=f"Epoch {epoch}")

        for step in pbar:
            # Sample meta-batch with color augmentation multiplier
            if args.color_aug_multiplier > 0:
                # Sample fewer base tasks and create multiple color variants
                effective_batch_size = args.meta_batch_size // (1 + args.color_aug_multiplier)
                effective_batch_size = max(1, effective_batch_size)  # At least 1

                batch_start = step * effective_batch_size
                batch_end = batch_start + effective_batch_size
                batch_indices = task_indices[batch_start:batch_end]
                base_tasks = [dataset[i] for i in batch_indices]

                # Generate augmented variants
                tasks = []
                for task in base_tasks:
                    # Add original task
                    tasks.append(task)

                    # Add N color-augmented variants (each with random spatial + color)
                    for _ in range(args.color_aug_multiplier):
                        aug_task = augment_task_spatial_and_colors(
                            task,
                            spatial_transform=None,  # Random
                            color_perm=None,  # Random
                            exclude_identity_spatial=args.augment_exclude_identity,
                            exclude_identity_color=True,  # Always exclude identity color
                            fix_background=False
                        )
                        tasks.append(aug_task)

            else:
                # Standard batching without color multiplier
                batch_start = step * args.meta_batch_size
                batch_end = batch_start + args.meta_batch_size
                batch_indices = task_indices[batch_start:batch_end]
                tasks = [dataset[i] for i in batch_indices]

                # Apply spatial augmentation if enabled
                if args.augment:
                    augmented_tasks = []
                    for task in tasks:
                        # Probabilistically augment each task
                        if np.random.random() < args.augment_prob:
                            # Apply random transformation
                            transform = random_transform(exclude_identity=args.augment_exclude_identity)
                            aug_task = augment_task(task, transform=transform)
                            augmented_tasks.append(aug_task)
                        else:
                            # Keep original task
                            augmented_tasks.append(task)
                    tasks = augmented_tasks

            # Meta-training step
            metrics = outer_loop.meta_train_step(
                tasks, encoder, controller, operators, renderer,
                verbose=False
            )

            # Track metrics
            epoch_metrics['meta_loss'].append(metrics.meta_loss)
            epoch_metrics['test_reward'].append(metrics.mean_test_reward)
            epoch_metrics['train_reward'].append(metrics.mean_train_reward)
            epoch_metrics['test_success'].append(metrics.test_success_rate)
            epoch_metrics['train_success'].append(metrics.train_success_rate)

            # Update progress bar
            pbar.set_postfix({
                'meta_loss': f'{metrics.meta_loss:.4f}',
                'test_r': f'{metrics.mean_test_reward:.3f}',
                'train_r': f'{metrics.mean_train_reward:.3f}',
            })

            # Log to tensorboard
            writer.add_scalar('train/meta_loss', metrics.meta_loss, global_step)
            writer.add_scalar('train/test_reward', metrics.mean_test_reward, global_step)
            writer.add_scalar('train/train_reward', metrics.mean_train_reward, global_step)
            writer.add_scalar('train/test_success_rate', metrics.test_success_rate, global_step)
            writer.add_scalar('train/train_success_rate', metrics.train_success_rate, global_step)

            global_step += 1

        # Epoch summary
        mean_meta_loss = np.mean(epoch_metrics['meta_loss'])
        mean_test_reward = np.mean(epoch_metrics['test_reward'])
        mean_train_reward = np.mean(epoch_metrics['train_reward'])
        mean_test_success = np.mean(epoch_metrics['test_success'])
        mean_train_success = np.mean(epoch_metrics['train_success'])

        print(f"\nEpoch {epoch} Summary:")
        print(f"  Meta-loss: {mean_meta_loss:.4f}")
        print(f"  Test reward: {mean_test_reward:.3f}")
        print(f"  Train reward: {mean_train_reward:.3f}")
        print(f"  Test success: {mean_test_success:.1%}")
        print(f"  Train success: {mean_train_success:.1%}")

        # Log epoch summary
        writer.add_scalar('epoch/meta_loss', mean_meta_loss, epoch)
        writer.add_scalar('epoch/test_reward', mean_test_reward, epoch)
        writer.add_scalar('epoch/train_reward', mean_train_reward, epoch)
        writer.add_scalar('epoch/test_success_rate', mean_test_success, epoch)
        writer.add_scalar('epoch/train_success_rate', mean_train_success, epoch)

        # Save best model
        if mean_test_reward > best_test_reward:
            best_test_reward = mean_test_reward
            checkpoint_path = checkpoint_dir / "controller_best.pt"
            torch.save({
                'epoch': epoch,
                'controller_state_dict': controller.state_dict(),
                'operators_state_dict': operators.state_dict(),
                'test_reward': mean_test_reward,
                'train_reward': mean_train_reward,
            }, checkpoint_path)
            print(f"  ✓ Saved best model (test_reward={mean_test_reward:.3f})")

        # Save checkpoint
        if epoch % args.save_every == 0:
            checkpoint_path = checkpoint_dir / f"controller_epoch{epoch}.pt"
            torch.save({
                'epoch': epoch,
                'controller_state_dict': controller.state_dict(),
                'operators_state_dict': operators.state_dict(),
                'test_reward': mean_test_reward,
                'train_reward': mean_train_reward,
            }, checkpoint_path)
            print(f"  ✓ Saved checkpoint")

    print("\n" + "="*60)
    print("Training complete!")
    print(f"Best test reward: {best_test_reward:.3f}")
    print("="*60)

    writer.close()


if __name__ == "__main__":
    main()
