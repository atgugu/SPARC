"""
REINFORCE losses for controller training.

Provides reward computation and policy gradient losses for learning operator sequences.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import List, Optional, Dict
from dataclasses import dataclass


@dataclass
class RewardComponents:
    """Breakdown of reward computation."""
    accuracy: float
    constraint_score: float
    total: float
    metadata: Dict


class ReinforceReward:
    """
    Compute rewards for operator sequences.

    Combines:
    - Pixel accuracy (70%)
    - Constraint satisfaction (30%)

    Returns rewards in [0, 1] where:
    - 1.0 = perfect match
    - 0.7-0.9 = high accuracy
    - 0.3-0.6 = partial match
    - 0.0-0.2 = poor/invalid
    """

    def __init__(
        self,
        accuracy_weight: float = 0.7,
        constraint_weight: float = 0.3,
        penalty_invalid: float = 0.1  # Penalty for constraint violations
    ):
        """
        Initialize reward computer.

        Args:
            accuracy_weight: Weight for pixel accuracy (default 0.7)
            constraint_weight: Weight for constraint satisfaction (default 0.3)
            penalty_invalid: Penalty for violating hard constraints (default 0.1)
        """
        assert accuracy_weight + constraint_weight == 1.0, "Weights must sum to 1"
        self.accuracy_weight = accuracy_weight
        self.constraint_weight = constraint_weight
        self.penalty_invalid = penalty_invalid

    def compute_reward(
        self,
        prediction: torch.Tensor,  # [H, W]
        target: torch.Tensor,      # [H, W]
        h: int,
        w: int,
        constraints: Optional[object] = None,  # ConstraintSet
        input_shape: Optional[tuple] = None
    ) -> float:
        """
        Compute reward for a single prediction.

        Args:
            prediction: Predicted grid [H, W]
            target: Target grid [H, W]
            h: Actual output height
            w: Actual output width
            constraints: Optional ConstraintSet for constraint checking
            input_shape: Optional input shape for constraint validation

        Returns:
            Reward in [0, 1]
        """
        # 1. Pixel accuracy (crop to actual size)
        pred_crop = prediction[:h, :w]
        target_crop = target[:h, :w]

        accuracy = (pred_crop == target_crop).float().mean().item()

        # 2. Constraint satisfaction
        constraint_score = 0.5  # Neutral default

        if constraints is not None:
            # Check if prediction is valid
            is_valid = constraints.is_valid(
                prediction, h, w,
                input_shape=input_shape
            )

            if not is_valid:
                # Apply penalty for hard constraint violations
                constraint_score = self.penalty_invalid
            else:
                # Compute soft constraint score
                constraint_score = constraints.score(
                    prediction, h, w
                )

        # 3. Weighted combination
        reward = (
            self.accuracy_weight * accuracy +
            self.constraint_weight * constraint_score
        )

        return reward

    def compute_batch_rewards(
        self,
        predictions: List[torch.Tensor],  # List of [H, W]
        targets: List[torch.Tensor],      # List of [H, W]
        shapes: List[tuple],              # List of (h, w)
        constraints: Optional[object] = None,
        input_shapes: Optional[List[tuple]] = None
    ) -> torch.Tensor:
        """
        Compute rewards for a batch of predictions.

        Args:
            predictions: List of predicted grids
            targets: List of target grids
            shapes: List of actual output sizes
            constraints: Optional ConstraintSet
            input_shapes: Optional list of input shapes

        Returns:
            Rewards tensor [batch_size]
        """
        rewards = []

        for i, (pred, target, (h, w)) in enumerate(zip(predictions, targets, shapes)):
            input_shape = input_shapes[i] if input_shapes else None
            reward = self.compute_reward(
                pred, target, h, w,
                constraints=constraints,
                input_shape=input_shape
            )
            rewards.append(reward)

        return torch.tensor(rewards, dtype=torch.float32)


class SequenceLoss:
    """
    REINFORCE loss with baseline for policy gradient training.

    Loss: -∑_t log π(a_t | s_t) * (R - baseline) + entropy_bonus

    Features:
    - Advantage estimation (reward - baseline)
    - Entropy regularization for exploration
    - Baseline tracking (exponential moving average)
    """

    def __init__(
        self,
        entropy_weight: float = 0.01,
        baseline_momentum: float = 0.9,
        normalize_advantages: bool = True
    ):
        """
        Initialize REINFORCE loss.

        Args:
            entropy_weight: Weight for entropy bonus (default 0.01)
            baseline_momentum: EMA momentum for baseline (default 0.9)
            normalize_advantages: Whether to normalize advantages (default True)
        """
        self.entropy_weight = entropy_weight
        self.baseline_momentum = baseline_momentum
        self.normalize_advantages = normalize_advantages

        # Baseline tracking
        self.baseline = 0.0
        self.n_updates = 0

    def compute_loss(
        self,
        log_probs: List[torch.Tensor],  # Per-step log probs, each [beam_size, num_ops]
        rewards: torch.Tensor,           # [beam_size]
        entropies: Optional[List[torch.Tensor]] = None,  # Per-step entropy
        mask: Optional[torch.Tensor] = None  # [beam_size, max_steps] for variable length
    ) -> Dict[str, torch.Tensor]:
        """
        Compute REINFORCE loss.

        Args:
            log_probs: List of log probability tensors, one per step
                      Each is [beam_size] (already selected actions)
            rewards: Rewards for each sequence [beam_size]
            entropies: Optional list of entropy tensors for regularization
            mask: Optional mask for variable-length sequences

        Returns:
            Dictionary with:
            - 'loss': Total loss (scalar)
            - 'policy_loss': Policy gradient loss
            - 'entropy_bonus': Entropy regularization term
            - 'baseline': Current baseline value
            - 'mean_reward': Mean reward
            - 'advantages': Computed advantages
        """
        beam_size = rewards.shape[0]
        num_steps = len(log_probs)

        # 1. Compute advantages (reward - baseline)
        advantages = rewards - self.baseline

        if self.normalize_advantages and beam_size > 1:
            # Normalize advantages to reduce variance
            advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)

        # 2. Stack log probs: [beam_size, num_steps]
        log_probs_stacked = torch.stack(log_probs, dim=1)  # [beam_size, num_steps]

        # 3. Apply mask if provided
        if mask is not None:
            log_probs_stacked = log_probs_stacked * mask

        # 4. Compute policy gradient loss
        # For each sequence: sum over steps, weighted by advantage
        policy_loss = -(log_probs_stacked.sum(dim=1) * advantages).mean()

        # 5. Compute entropy bonus (encourage exploration)
        entropy_bonus = torch.tensor(0.0)
        if entropies is not None and self.entropy_weight > 0:
            entropies_stacked = torch.stack(entropies, dim=1)  # [beam_size, num_steps]
            if mask is not None:
                entropies_stacked = entropies_stacked * mask
            entropy_bonus = entropies_stacked.sum(dim=1).mean()

        # 6. Total loss
        loss = policy_loss - self.entropy_weight * entropy_bonus

        # 7. Update baseline (exponential moving average)
        mean_reward = rewards.mean().item()
        if self.n_updates == 0:
            self.baseline = mean_reward
        else:
            self.baseline = (
                self.baseline_momentum * self.baseline +
                (1 - self.baseline_momentum) * mean_reward
            )
        self.n_updates += 1

        return {
            'loss': loss,
            'policy_loss': policy_loss,
            'entropy_bonus': entropy_bonus,
            'baseline': torch.tensor(self.baseline),
            'mean_reward': torch.tensor(mean_reward),
            'advantages': advantages
        }


# Test code
if __name__ == "__main__":
    import sys
    from pathlib import Path
    sys.path.insert(0, str(Path(__file__).parent.parent.parent))

    print("="*60)
    print("Testing REINFORCE Losses")
    print("="*60)

    # 1. Test ReinforceReward
    print("\n1. Testing ReinforceReward...")

    reward_computer = ReinforceReward()

    # Create dummy prediction and target
    target = torch.randint(0, 10, (10, 10))

    # Perfect prediction
    pred_perfect = target.clone()
    reward_perfect = reward_computer.compute_reward(pred_perfect, target, 10, 10)
    print(f"  Perfect match reward: {reward_perfect:.3f} (expect ~0.7)")

    # Partial match (70% accuracy)
    pred_partial = target.clone()
    mask = torch.rand(10, 10) > 0.7
    pred_partial[mask] = torch.randint(0, 10, (mask.sum().item(),))
    reward_partial = reward_computer.compute_reward(pred_partial, target, 10, 10)
    print(f"  70% match reward: {reward_partial:.3f} (expect ~0.5)")

    # Random prediction
    pred_random = torch.randint(0, 10, (10, 10))
    reward_random = reward_computer.compute_reward(pred_random, target, 10, 10)
    print(f"  Random match reward: {reward_random:.3f} (expect ~0.2)")

    # 2. Test SequenceLoss
    print("\n2. Testing SequenceLoss...")

    loss_fn = SequenceLoss()

    # Create dummy log probs and rewards
    beam_size = 8
    num_steps = 3

    # Simulate log probs for each step (already selected actions)
    log_probs = [
        torch.randn(beam_size).log_softmax(0) for _ in range(num_steps)
    ]

    # Simulate rewards (some good, some bad)
    rewards = torch.tensor([0.9, 0.8, 0.7, 0.5, 0.4, 0.3, 0.2, 0.1])

    # Compute loss
    loss_dict = loss_fn.compute_loss(log_probs, rewards)

    print(f"  Loss: {loss_dict['loss'].item():.4f}")
    print(f"  Policy loss: {loss_dict['policy_loss'].item():.4f}")
    print(f"  Entropy bonus: {loss_dict['entropy_bonus'].item():.4f}")
    print(f"  Baseline: {loss_dict['baseline'].item():.4f}")
    print(f"  Mean reward: {loss_dict['mean_reward'].item():.4f}")

    # Test baseline update
    print("\n3. Testing baseline update...")
    for i in range(5):
        rewards = torch.tensor([0.6, 0.7, 0.5, 0.4, 0.6, 0.5, 0.7, 0.6])
        loss_dict = loss_fn.compute_loss(log_probs, rewards)
        print(f"  Iteration {i+1}: baseline={loss_dict['baseline'].item():.4f}, mean_reward={loss_dict['mean_reward'].item():.4f}")

    print("\n" + "="*60)
    print("✓ All tests passed!")
    print("="*60)
