#!/usr/bin/env python3
"""
Pretrain the SlotAttention autoencoder on ARC grids.

Goal: Achieve >95% reconstruction accuracy for stable slot initialization.
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

import torch
import torch.nn as nn
from torch.cuda.amp import autocast, GradScaler
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm
import argparse
from datetime import datetime

from arc_nodsl.data.loader import ARCDataset
from arc_nodsl.data.batching import create_dataloader
from arc_nodsl.models.renderer import AutoEncoder, compute_reconstruction_loss, compute_mask_diversity_loss
from arc_nodsl.utils.profile import Timer, get_gpu_memory_stats
from arc_nodsl.data.augment import (
    augment_pair, random_transform, Transform,
    augment_pair_colors
)


def expand_batch_with_augmentations(batch: dict, multiplier: int) -> dict:
    """
    Expand a batch by creating multiple augmented variants per sample.

    Each variant gets random spatial transform + random color permutation.
    This massively increases training data diversity.

    Args:
        batch: Batch dict with keys 'inputs', 'outputs', 'shapes'
        multiplier: Number of additional variants to create per sample

    Returns:
        Expanded batch with (batch_size * (1 + multiplier)) samples
    """
    if multiplier <= 0:
        return batch

    inputs = batch['inputs']  # [B, H, W]
    outputs = batch['outputs']  # [B, H, W]
    shapes = batch['shapes']  # List of dicts with 'input' and 'output' tuples

    batch_size = inputs.shape[0]

    expanded_inputs = []
    expanded_outputs = []
    expanded_shapes = []

    # For each sample in the batch
    for i in range(batch_size):
        inp = inputs[i]
        out = outputs[i]
        shape = shapes[i]

        # Validate original tensors (autoencoder supports colors 0-10)
        if (inp < 0).any() or (inp > 10).any():
            print(f"WARNING: Original inp[{i}] has invalid values: min={inp.min().item()}, max={inp.max().item()}")
            inp = torch.clamp(inp, 0, 10)
        if (out < 0).any() or (out > 10).any():
            print(f"WARNING: Original out[{i}] has invalid values: min={out.min().item()}, max={out.max().item()}")
            out = torch.clamp(out, 0, 10)

        # Add original sample
        expanded_inputs.append(inp)
        expanded_outputs.append(out)
        expanded_shapes.append(shape)

        # Generate N augmented variants
        for _ in range(multiplier):
            # Apply random spatial transform (exclude identity for diversity)
            transform = random_transform(exclude_identity=True)
            aug_inp, aug_out, _ = augment_pair(
                inp, out,
                shape['input'], shape['output'],
                transform=transform
            )

            # Apply random color permutation (exclude identity)
            aug_inp, aug_out, _ = augment_pair_colors(
                aug_inp, aug_out,
                exclude_identity=True,
                fix_background=False
            )

            # Ensure correct dtype and validate range
            aug_inp = aug_inp.long()
            aug_out = aug_out.long()

            # Check for invalid values before clamping
            if (aug_inp < 0).any() or (aug_inp > 10).any():
                print(f"WARNING: aug_inp has invalid values: min={aug_inp.min().item()}, max={aug_inp.max().item()}")
            if (aug_out < 0).any() or (aug_out > 10).any():
                print(f"WARNING: aug_out has invalid values: min={aug_out.min().item()}, max={aug_out.max().item()}")

            # Clamp to valid color range [0, 10] for autoencoder (11 classes)
            aug_inp = torch.clamp(aug_inp, 0, 10)
            aug_out = torch.clamp(aug_out, 0, 10)

            # Update shapes if rotation changed dimensions
            if transform in [Transform.ROT_90, Transform.ROT_270,
                            Transform.FLIP_D1, Transform.FLIP_D2]:
                h_in, w_in = shape['input']
                h_out, w_out = shape['output']
                aug_shape = {
                    'input': (w_in, h_in),
                    'output': (w_out, h_out)
                }
            else:
                aug_shape = shape.copy()

            expanded_inputs.append(aug_inp)
            expanded_outputs.append(aug_out)
            expanded_shapes.append(aug_shape)

    # Stack into tensors (ensure long dtype for color indices)
    expanded_batch = {
        'inputs': torch.stack(expanded_inputs, dim=0).long(),
        'outputs': torch.stack(expanded_outputs, dim=0).long(),
        'shapes': expanded_shapes
    }

    return expanded_batch


def train_epoch(model, loader, optimizer, scaler, device, epoch, writer, timer, color_aug_multiplier=0):
    """Train for one epoch."""
    model.train()

    total_loss = 0
    total_recon_loss = 0
    total_div_loss = 0
    total_acc = 0
    n_batches = 0

    pbar = tqdm(loader, desc=f"Epoch {epoch}")

    for batch in pbar:
        with timer.measure("data_loading"):
            # Expand batch with augmented variants if multiplier enabled
            if color_aug_multiplier > 0:
                batch = expand_batch_with_augmentations(batch, color_aug_multiplier)

            inputs = batch["inputs"].to(device, non_blocking=True)
            targets = batch["outputs"].to(device, non_blocking=True)
            shapes = batch["shapes"]

        optimizer.zero_grad()

        with timer.measure("forward"):
            with autocast(dtype=torch.float16):
                # Forward
                outputs = model(inputs)

                # Compute loss per sample with original shapes
                batch_size = inputs.shape[0]
                recon_losses = []
                for i in range(batch_size):
                    h, w = shapes[i]["output"]
                    loss_i = compute_reconstruction_loss(
                        outputs["recon_logits"][i:i+1],
                        targets[i:i+1],
                        h=h, w=w
                    )
                    recon_losses.append(loss_i)

                recon_loss = torch.stack(recon_losses).mean()

                # Diversity loss
                div_loss = compute_mask_diversity_loss(outputs["slots_m"])

                # Total loss
                loss = recon_loss + 0.01 * div_loss

        with timer.measure("backward"):
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            scaler.step(optimizer)
            scaler.update()

        # Compute accuracy
        with torch.no_grad():
            recon = torch.argmax(outputs["recon_logits"], dim=-1)
            acc = (recon == targets).float().mean()

        # Update stats
        total_loss += loss.item()
        total_recon_loss += recon_loss.item()
        total_div_loss += div_loss.item()
        total_acc += acc.item()
        n_batches += 1

        # Update progress bar
        pbar.set_postfix({
            'loss': f'{loss.item():.4f}',
            'acc': f'{acc.item()*100:.1f}%',
        })

    # Log epoch stats
    avg_loss = total_loss / n_batches
    avg_recon = total_recon_loss / n_batches
    avg_div = total_div_loss / n_batches
    avg_acc = total_acc / n_batches

    if writer:
        writer.add_scalar('train/loss', avg_loss, epoch)
        writer.add_scalar('train/recon_loss', avg_recon, epoch)
        writer.add_scalar('train/div_loss', avg_div, epoch)
        writer.add_scalar('train/accuracy', avg_acc, epoch)

    return avg_loss, avg_acc


@torch.no_grad()
def evaluate(model, loader, device, epoch, writer):
    """Evaluate on validation set."""
    model.eval()

    total_loss = 0
    total_acc = 0
    n_batches = 0

    for batch in tqdm(loader, desc="Evaluating"):
        inputs = batch["inputs"].to(device)
        targets = batch["outputs"].to(device)
        shapes = batch["shapes"]

        # Forward
        outputs = model(inputs)

        # Compute loss
        batch_size = inputs.shape[0]
        recon_losses = []
        for i in range(batch_size):
            h, w = shapes[i]["output"]
            loss_i = compute_reconstruction_loss(
                outputs["recon_logits"][i:i+1],
                targets[i:i+1],
                h=h, w=w
            )
            recon_losses.append(loss_i)

        recon_loss = torch.stack(recon_losses).mean()

        # Accuracy
        recon = torch.argmax(outputs["recon_logits"], dim=-1)
        acc = (recon == targets).float().mean()

        total_loss += recon_loss.item()
        total_acc += acc.item()
        n_batches += 1

    avg_loss = total_loss / n_batches
    avg_acc = total_acc / n_batches

    if writer:
        writer.add_scalar('val/loss', avg_loss, epoch)
        writer.add_scalar('val/accuracy', avg_acc, epoch)

    return avg_loss, avg_acc


def main():
    parser = argparse.ArgumentParser(description="Pretrain SlotAttention autoencoder")
    parser.add_argument("--data_train", type=str, default="data/arc-agi_training_challenges.json")
    parser.add_argument("--data_val", type=str, default="data/arc-agi_evaluation_challenges.json")
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--num_slots", type=int, default=8)
    parser.add_argument("--d_slot", type=int, default=128)
    parser.add_argument("--num_iters", type=int, default=3)
    parser.add_argument("--checkpoint_dir", type=str, default="checkpoints")
    parser.add_argument("--log_dir", type=str, default="runs")
    parser.add_argument("--eval_every", type=int, default=5)
    parser.add_argument("--save_every", type=int, default=10)
    parser.add_argument("--augment", action="store_true", help="Apply random spatial transformations (rotations, flips)")
    parser.add_argument("--color_augment_prob", type=float, default=0.0,
                        help="Probability of applying color permutation to each pair (0.0-1.0)")
    parser.add_argument("--color_aug_multiplier", type=int, default=0,
                        help="Create N additional variants per pair with random spatial+color (0=disabled, 10=10 variants)")
    args = parser.parse_args()

    # Setup
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_dir = Path(args.log_dir) / f"pretrain_{timestamp}"
    checkpoint_dir = Path(args.checkpoint_dir)
    checkpoint_dir.mkdir(exist_ok=True, parents=True)

    writer = SummaryWriter(log_dir)
    print(f"Logging to {log_dir}")

    # Load data
    print("\nLoading datasets...")
    train_dataset = ARCDataset(args.data_train)
    val_dataset = ARCDataset(args.data_val)

    train_loader = create_dataloader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=0,  # Keep 0 to avoid multiprocessing issues
        collate_mode="flat_pairs",
        augment=args.augment,
        color_augment_prob=args.color_augment_prob
    )

    val_loader = create_dataloader(
        val_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=0,
        collate_mode="flat_pairs",
        augment=False,  # No augmentation for validation
        color_augment_prob=0.0  # No color augmentation for validation
    )

    print(f"Train: {len(train_dataset)} tasks")
    print(f"Val: {len(val_dataset)} tasks")

    # Augmentation status
    print(f"\nData Augmentation:")
    print(f"  Spatial (in DataLoader): {'ENABLED' if args.augment else 'DISABLED'}")
    if args.color_augment_prob > 0:
        print(f"  Color Probability (in DataLoader): ENABLED (prob={args.color_augment_prob})")
    else:
        print(f"  Color Probability (in DataLoader): DISABLED")

    if args.color_aug_multiplier > 0:
        print(f"  Color Multiplier (in training loop): ENABLED (multiplier={args.color_aug_multiplier})")
        print(f"    Note: Each batch sample generates {args.color_aug_multiplier} additional variants")
        print(f"    Each variant has random spatial transform + random color permutation")
        effective_multiplier = 1 + args.color_aug_multiplier
        print(f"    Effective data multiplier: {effective_multiplier}x")
    else:
        print(f"  Color Multiplier (in training loop): DISABLED")

    # Create model
    print("\nCreating model...")
    model = AutoEncoder(
        num_slots=args.num_slots,
        d_color=16,
        d_feat=64,
        d_slot=args.d_slot,
        d_hidden=64,
        num_iters=args.num_iters,
        H=30,
        W=30,
        use_mask=True
    ).to(device)

    # Count parameters
    n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Model parameters: {n_params:,}")

    # Optimizer
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-5)

    # Scheduler
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=args.epochs, eta_min=1e-6
    )

    # AMP
    scaler = GradScaler()

    # Timer
    timer = Timer()

    # Training loop
    best_val_acc = 0.0

    print("\nStarting training...")
    for epoch in range(1, args.epochs + 1):
        print(f"\n{'='*60}")
        print(f"Epoch {epoch}/{args.epochs}")
        print(f"{'='*60}")

        # Train
        train_loss, train_acc = train_epoch(
            model, train_loader, optimizer, scaler, device, epoch, writer, timer,
            color_aug_multiplier=args.color_aug_multiplier
        )

        print(f"\nTrain Loss: {train_loss:.4f}, Accuracy: {train_acc*100:.2f}%")

        # Evaluate
        if epoch % args.eval_every == 0:
            val_loss, val_acc = evaluate(model, val_loader, device, epoch, writer)
            print(f"Val Loss: {val_loss:.4f}, Accuracy: {val_acc*100:.2f}%")

            # Save best model
            if val_acc > best_val_acc:
                best_val_acc = val_acc
                checkpoint_path = checkpoint_dir / "autoencoder_best.pt"
                torch.save({
                    'epoch': epoch,
                    'model_state_dict': model.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict(),
                    'val_acc': val_acc,
                }, checkpoint_path)
                print(f"✓ Saved best model (acc={val_acc*100:.2f}%)")

        # Save checkpoint
        if epoch % args.save_every == 0:
            checkpoint_path = checkpoint_dir / f"autoencoder_epoch{epoch}.pt"
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
            }, checkpoint_path)

        # Step scheduler
        scheduler.step()

        # Print timing stats
        if epoch % 10 == 0:
            timer.print_stats()
            timer.reset()

        # GPU memory
        if torch.cuda.is_available():
            mem_stats = get_gpu_memory_stats()
            print(f"GPU Memory: {mem_stats['allocated_mb']:.0f}MB")

    print("\n" + "="*60)
    print("Training complete!")
    print(f"Best validation accuracy: {best_val_acc*100:.2f}%")
    print("="*60)

    writer.close()


if __name__ == "__main__":
    main()
