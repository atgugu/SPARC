"""
Controller: Policy network for selecting operator sequences.

The controller takes slot representations and task embeddings as input,
and outputs:
1. Operator selection (discrete, via Gumbel-Softmax)
2. Continuous parameters for the operator
3. Stop probability

References:
- Gumbel-Softmax: Jang et al. (2017)
- Reparameterization trick: Kingma & Welling (2014)
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Tuple, Optional, Dict, List
import math


class SlotSummarizer(nn.Module):
    """
    Pool K slots into a single summary vector.
    Uses attention pooling with a learnable query.
    """

    def __init__(self, d_slot: int = 128, use_positions: bool = True):
        super().__init__()
        self.d_slot = d_slot
        self.use_positions = use_positions

        # Learnable query for attention pooling
        self.query = nn.Parameter(torch.randn(1, 1, d_slot))

        # Attention
        self.attn = nn.MultiheadAttention(d_slot, num_heads=4, batch_first=True)

        # Position encoding (optional)
        if use_positions:
            self.pos_fc = nn.Linear(2, d_slot)

    def forward(self, slots_z: torch.Tensor, slots_p: Optional[torch.Tensor] = None) -> torch.Tensor:
        """
        Args:
            slots_z: [B, K, d_slot] slot features
            slots_p: [B, K, 2] optional slot positions

        Returns:
            summary: [B, d_slot] pooled representation
        """
        B = slots_z.shape[0]

        # Add positional information if available
        if self.use_positions and slots_p is not None:
            pos_embed = self.pos_fc(slots_p)  # [B, K, d_slot]
            slots_with_pos = slots_z + pos_embed
        else:
            slots_with_pos = slots_z

        # Attention pooling with learnable query
        query = self.query.expand(B, -1, -1)  # [B, 1, d_slot]
        summary, _ = self.attn(query, slots_with_pos, slots_with_pos)  # [B, 1, d_slot]

        return summary.squeeze(1)  # [B, d_slot]


class PolicyNetwork(nn.Module):
    """
    Transformer-based policy network for sequential decision making.
    """

    def __init__(self, d_input: int, d_hidden: int = 256, n_layers: int = 2, n_heads: int = 4):
        super().__init__()
        self.d_hidden = d_hidden

        # Input projection
        self.input_proj = nn.Linear(d_input, d_hidden)

        # Transformer layers
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_hidden,
            nhead=n_heads,
            dim_feedforward=d_hidden * 2,
            batch_first=True,
            norm_first=True
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=n_layers)

        # Output projection
        self.output_proj = nn.Linear(d_hidden, d_hidden)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: [B, d_input] input features

        Returns:
            [B, d_hidden] hidden state
        """
        # Project input
        x = self.input_proj(x)  # [B, d_hidden]

        # Add batch dimension for transformer (expects [B, seq_len, d])
        x = x.unsqueeze(1)  # [B, 1, d_hidden]

        # Transform
        x = self.transformer(x)  # [B, 1, d_hidden]

        # Project output
        x = self.output_proj(x.squeeze(1))  # [B, d_hidden]

        return x


class OperatorHead(nn.Module):
    """Predict operator index (discrete choice)."""

    def __init__(self, d_hidden: int, num_operators: int = 8):
        super().__init__()
        self.fc = nn.Linear(d_hidden, num_operators)

    def forward(self, hidden: torch.Tensor) -> torch.Tensor:
        """
        Args:
            hidden: [B, d_hidden]

        Returns:
            logits: [B, num_operators]
        """
        return self.fc(hidden)


class ParamHead(nn.Module):
    """Predict continuous parameters via Gaussian."""

    def __init__(self, d_hidden: int, d_params: int = 16):
        super().__init__()
        self.fc_mu = nn.Linear(d_hidden, d_params)
        self.fc_logvar = nn.Linear(d_hidden, d_params)

    def forward(self, hidden: torch.Tensor) -> Dict[str, torch.Tensor]:
        """
        Args:
            hidden: [B, d_hidden]

        Returns:
            dict with mu, logvar: each [B, d_params]
        """
        mu = self.fc_mu(hidden)
        logvar = self.fc_logvar(hidden)
        # Clamp logvar for stability
        logvar = torch.clamp(logvar, min=-10, max=2)

        return {"mu": mu, "logvar": logvar}


class StopHead(nn.Module):
    """Predict whether to stop the sequence."""

    def __init__(self, d_hidden: int):
        super().__init__()
        self.fc = nn.Linear(d_hidden, 1)

    def forward(self, hidden: torch.Tensor) -> torch.Tensor:
        """
        Args:
            hidden: [B, d_hidden]

        Returns:
            stop_logit: [B, 1]
        """
        return self.fc(hidden)


def gumbel_softmax(logits: torch.Tensor,
                   temperature: float = 1.0,
                   hard: bool = False,
                   dim: int = -1) -> torch.Tensor:
    """
    Sample from Gumbel-Softmax distribution.

    Args:
        logits: [..., num_classes]
        temperature: Softmax temperature (lower = more discrete)
        hard: If True, return one-hot with straight-through gradients
        dim: Dimension to apply softmax

    Returns:
        [..., num_classes] sampled distribution
    """
    # Sample Gumbel noise
    gumbels = -torch.empty_like(logits).exponential_().log()  # ~Gumbel(0,1)
    gumbels = (logits + gumbels) / temperature

    # Softmax
    y_soft = F.softmax(gumbels, dim=dim)

    if hard:
        # Straight-through estimator
        index = y_soft.argmax(dim=dim, keepdim=True)
        y_hard = torch.zeros_like(logits).scatter_(dim, index, 1.0)
        # Trick: y_hard - y_soft.detach() + y_soft has hard forward, soft backward
        return y_hard - y_soft.detach() + y_soft
    else:
        return y_soft


def sample_params(mu: torch.Tensor, logvar: torch.Tensor) -> torch.Tensor:
    """
    Sample from Gaussian using reparameterization trick.

    Args:
        mu: [B, d_params] mean
        logvar: [B, d_params] log variance

    Returns:
        [B, d_params] sampled parameters
    """
    std = torch.exp(0.5 * logvar)
    eps = torch.randn_like(std)
    return mu + eps * std


def compute_entropy(logits: torch.Tensor, dim: int = -1) -> torch.Tensor:
    """
    Compute entropy of categorical distribution.

    Args:
        logits: [..., num_classes]
        dim: Dimension for softmax

    Returns:
        [...] entropy values
    """
    probs = F.softmax(logits, dim=dim)
    log_probs = F.log_softmax(logits, dim=dim)
    entropy = -(probs * log_probs).sum(dim=dim)
    return entropy


class Controller(nn.Module):
    """
    Complete controller for operator sequence generation.

    Combines slot summarizer, policy network, and output heads
    to produce operator sequences with continuous parameters.
    """

    def __init__(self,
                 num_operators: int = 8,
                 d_slot: int = 128,
                 d_task: int = 128,
                 d_hidden: int = 256,
                 d_params: int = 16,
                 n_layers: int = 2,
                 n_heads: int = 4,
                 max_steps: int = 4):
        super().__init__()
        self.num_operators = num_operators
        self.d_slot = d_slot
        self.d_task = d_task
        self.d_hidden = d_hidden
        self.d_params = d_params
        self.max_steps = max_steps

        # Components
        self.summarizer = SlotSummarizer(d_slot)

        # Policy network input: slot_summary + task_embed + history
        d_input = d_slot + d_task
        self.policy = PolicyNetwork(d_input, d_hidden, n_layers, n_heads)

        # Output heads
        self.op_head = OperatorHead(d_hidden, num_operators)
        self.param_head = ParamHead(d_hidden, d_params)
        self.stop_head = StopHead(d_hidden)

        # History embedding (for autoregressive generation)
        self.op_embed = nn.Embedding(num_operators, d_hidden // 2)
        self.history_fc = nn.Linear(d_hidden // 2, d_hidden)

    def step(self,
             slots_z: torch.Tensor,
             slots_p: torch.Tensor,
             task_embed: Optional[torch.Tensor] = None,
             prev_op: Optional[torch.Tensor] = None,
             temperature: float = 1.0) -> Dict[str, torch.Tensor]:
        """
        Single step of controller: predict next operator and parameters.

        Args:
            slots_z: [B, K, d_slot] current slot features
            slots_p: [B, K, 2] current slot positions
            task_embed: [B, d_task] optional task embedding
            prev_op: [B] optional previous operator index
            temperature: Gumbel-Softmax temperature

        Returns:
            dict with:
                - op_logits: [B, num_operators]
                - op_sample: [B, num_operators] (soft or hard)
                - params_mu: [B, d_params]
                - params_logvar: [B, d_params]
                - params_sample: [B, d_params]
                - stop_logit: [B, 1]
        """
        B = slots_z.shape[0]

        # Summarize slots
        slot_summary = self.summarizer(slots_z, slots_p)  # [B, d_slot]

        # Default task embedding if not provided
        if task_embed is None:
            task_embed = torch.zeros(B, self.d_task, device=slots_z.device)

        # Concatenate inputs
        policy_input = torch.cat([slot_summary, task_embed], dim=-1)  # [B, d_slot + d_task]

        # TODO: Add history if prev_op provided (for autoregressive)
        # For now, simple feedforward

        # Policy network
        hidden = self.policy(policy_input)  # [B, d_hidden]

        # Operator prediction
        op_logits = self.op_head(hidden)  # [B, num_operators]
        op_sample = gumbel_softmax(op_logits, temperature=temperature, hard=False)

        # Parameter prediction
        param_dist = self.param_head(hidden)
        params_mu = param_dist["mu"]
        params_logvar = param_dist["logvar"]
        params_sample = sample_params(params_mu, params_logvar)

        # Stop prediction
        stop_logit = self.stop_head(hidden)

        return {
            "op_logits": op_logits,
            "op_sample": op_sample,
            "params_mu": params_mu,
            "params_logvar": params_logvar,
            "params_sample": params_sample,
            "stop_logit": stop_logit,
        }

    def rollout(self,
                slots_z: torch.Tensor,
                slots_p: torch.Tensor,
                task_embed: Optional[torch.Tensor] = None,
                max_steps: Optional[int] = None,
                temperature: float = 1.0,
                stop_threshold: float = 0.5) -> Dict[str, List]:
        """
        Generate a full sequence of operators.

        Args:
            slots_z: [B, K, d_slot] initial slots
            slots_p: [B, K, 2] initial positions
            task_embed: [B, d_task] optional task embedding
            max_steps: Maximum sequence length (default: self.max_steps)
            temperature: Gumbel-Softmax temperature
            stop_threshold: Probability threshold for stopping

        Returns:
            dict with:
                - op_logits: list of [B, num_operators]
                - op_samples: list of [B, num_operators]
                - params_mu: list of [B, d_params]
                - params_samples: list of [B, d_params]
                - stop_probs: list of [B]
                - stopped: [B] boolean indicating which sequences stopped
        """
        if max_steps is None:
            max_steps = self.max_steps

        B = slots_z.shape[0]

        # Storage for sequences
        op_logits_seq = []
        op_samples_seq = []
        params_mu_seq = []
        params_samples_seq = []
        stop_probs_seq = []

        # Track which sequences have stopped
        stopped = torch.zeros(B, dtype=torch.bool, device=slots_z.device)

        # Current state (slots don't change during rollout, but could in future)
        current_z = slots_z
        current_p = slots_p

        for t in range(max_steps):
            # Get predictions for this step
            outputs = self.step(current_z, current_p, task_embed, temperature=temperature)

            # Store outputs
            op_logits_seq.append(outputs["op_logits"])
            op_samples_seq.append(outputs["op_sample"])
            params_mu_seq.append(outputs["params_mu"])
            params_samples_seq.append(outputs["params_sample"])

            # Check stop condition
            stop_prob = torch.sigmoid(outputs["stop_logit"].squeeze(-1))  # [B]
            stop_probs_seq.append(stop_prob)

            # Update stopped mask
            stopped = stopped | (stop_prob > stop_threshold)

            # Early termination if all stopped
            if stopped.all():
                break

            # In future: apply operator and update slots
            # For now, slots remain static during rollout

        return {
            "op_logits": op_logits_seq,
            "op_samples": op_samples_seq,
            "params_mu": params_mu_seq,
            "params_samples": params_samples_seq,
            "stop_probs": stop_probs_seq,
            "stopped": stopped,
            "num_steps": len(op_logits_seq),
        }

    def get_op_indices(self, op_samples: torch.Tensor) -> torch.Tensor:
        """
        Convert soft operator samples to hard indices.

        Args:
            op_samples: [B, num_operators] soft distribution

        Returns:
            [B] operator indices
        """
        return op_samples.argmax(dim=-1)


if __name__ == "__main__":
    # Test controller
    import sys
    from pathlib import Path
    sys.path.insert(0, str(Path(__file__).parent.parent.parent))

    print("Testing Controller...")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    # Create controller
    controller = Controller(
        num_operators=8,
        d_slot=128,
        d_task=128,
        d_hidden=256,
        d_params=16,
        n_layers=2,
        n_heads=4,
        max_steps=4
    ).to(device)

    # Count parameters
    n_params = sum(p.numel() for p in controller.parameters())
    print(f"Controller parameters: {n_params:,}")

    # Random inputs
    B = 4
    K = 8
    slots_z = torch.randn(B, K, 128, device=device)
    slots_p = torch.rand(B, K, 2, device=device) * 30
    task_embed = torch.randn(B, 128, device=device)

    print("\n=== Single Step ===")
    with torch.no_grad():
        outputs = controller.step(slots_z, slots_p, task_embed, temperature=1.0)

    print(f"Op logits shape: {outputs['op_logits'].shape}")
    print(f"Op sample shape: {outputs['op_sample'].shape}")
    print(f"Params mu shape: {outputs['params_mu'].shape}")
    print(f"Stop logit shape: {outputs['stop_logit'].shape}")

    # Check operator selection
    op_idx = controller.get_op_indices(outputs['op_sample'])
    print(f"Selected operators: {op_idx}")

    print("\n=== Rollout (Sequence Generation) ===")
    with torch.no_grad():
        sequence = controller.rollout(
            slots_z, slots_p, task_embed,
            max_steps=3,
            temperature=1.0,
            stop_threshold=0.5
        )

    print(f"Number of steps: {sequence['num_steps']}")
    print(f"Stopped: {sequence['stopped']}")

    # Get operator sequence
    op_sequence = [controller.get_op_indices(op_samples) for op_samples in sequence['op_samples']]
    print(f"Operator sequence (first batch item):")
    for t, ops in enumerate(op_sequence):
        print(f"  Step {t}: op {ops[0].item()}")

    print("\n=== Test Gumbel-Softmax ===")
    logits = torch.randn(4, 8)
    soft = gumbel_softmax(logits, temperature=1.0, hard=False)
    hard = gumbel_softmax(logits, temperature=0.5, hard=True)
    print(f"Soft sample: {soft[0]}")
    print(f"Hard sample: {hard[0]}")
    assert (hard.sum(dim=-1) - 1.0).abs().max() < 1e-5, "Hard samples should be one-hot"

    print("\n=== Test Entropy ===")
    entropy = compute_entropy(outputs['op_logits'])
    print(f"Entropy shape: {entropy.shape}")
    print(f"Mean entropy: {entropy.mean().item():.3f}")

    print("\n✓ Controller tests passed!")
