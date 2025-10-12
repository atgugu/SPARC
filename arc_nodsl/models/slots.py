"""
Slot Attention Encoder and Decoder for ARC grids.

References:
- Locatello et al. "Object-Centric Learning with Slot Attention" (2020)
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Tuple, Optional, Dict
import math


class PaletteEmbedding(nn.Module):
    """
    Embedding layer for ARC color palette (0-9).
    """

    def __init__(self, d_color: int = 16):
        super().__init__()
        self.d_color = d_color
        # 10 colors + 1 for potential padding (though we use 0=black)
        self.embedding = nn.Embedding(11, d_color)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: [B, H, W] long tensor with values 0-9

        Returns:
            [B, H, W, d_color] embedded grid
        """
        return self.embedding(x)


class CNNFeatureExtractor(nn.Module):
    """
    CNN to extract spatial features from embedded grids.
    """

    def __init__(self, d_in: int = 16, d_feat: int = 64, n_layers: int = 4):
        super().__init__()
        self.d_in = d_in
        self.d_feat = d_feat

        layers = []
        in_channels = d_in

        for i in range(n_layers):
            out_channels = d_feat if i == n_layers - 1 else d_feat // 2
            layers.extend([
                nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1),
                nn.ReLU(inplace=True),
                nn.GroupNorm(min(8, out_channels // 8), out_channels) if out_channels >= 8 else nn.Identity(),
            ])
            in_channels = out_channels

        self.cnn = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: [B, H, W, d_in]

        Returns:
            [B, H, W, d_feat]
        """
        # Conv2d expects [B, C, H, W]
        x = x.permute(0, 3, 1, 2)  # [B, d_in, H, W]
        x = self.cnn(x)  # [B, d_feat, H, W]
        x = x.permute(0, 2, 3, 1)  # [B, H, W, d_feat]
        return x


class SlotAttention(nn.Module):
    """
    Slot Attention module.

    Iteratively refines K slot representations by attending to input features.
    """

    def __init__(self, d_feat: int, d_slot: int, num_slots: int = 8, num_iters: int = 3, eps: float = 1e-8):
        super().__init__()
        self.d_feat = d_feat
        self.d_slot = d_slot
        self.num_slots = num_slots
        self.num_iters = num_iters
        self.eps = eps

        self.scale = d_slot ** -0.5

        # Linear projections
        self.to_q = nn.Linear(d_slot, d_slot, bias=False)
        self.to_k = nn.Linear(d_feat, d_slot, bias=False)
        self.to_v = nn.Linear(d_feat, d_slot, bias=False)

        # GRU for slot updates
        self.gru = nn.GRUCell(d_slot, d_slot)

        # MLP for slot refinement
        self.mlp = nn.Sequential(
            nn.Linear(d_slot, d_slot * 2),
            nn.ReLU(inplace=True),
            nn.Linear(d_slot * 2, d_slot)
        )

        # Slot initialization parameters
        self.slot_mu = nn.Parameter(torch.randn(1, 1, d_slot))
        self.slot_log_sigma = nn.Parameter(torch.zeros(1, 1, d_slot))

        # Layer norms
        self.norm_inputs = nn.LayerNorm(d_feat)
        self.norm_slots = nn.LayerNorm(d_slot)
        self.norm_mlp = nn.LayerNorm(d_slot)

    def forward(self, inputs: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Args:
            inputs: [B, H, W, d_feat]

        Returns:
            slots: [B, K, d_slot]
            attn_maps: [B, K, H*W] attention maps
        """
        B, H, W, d_feat = inputs.shape
        N = H * W

        # Flatten spatial dimensions
        inputs_flat = inputs.view(B, N, d_feat)

        # Normalize inputs
        inputs_norm = self.norm_inputs(inputs_flat)

        # Project inputs to keys and values
        k = self.to_k(inputs_norm)  # [B, N, d_slot]
        v = self.to_v(inputs_norm)  # [B, N, d_slot]

        # Initialize slots
        mu = self.slot_mu.expand(B, self.num_slots, -1)
        sigma = self.slot_log_sigma.exp().expand(B, self.num_slots, -1)
        slots = mu + sigma * torch.randn_like(mu)

        # Iterative attention
        for _ in range(self.num_iters):
            slots_prev = slots

            # Normalize slots
            slots_norm = self.norm_slots(slots)

            # Compute attention: slots attend to inputs
            q = self.to_q(slots_norm)  # [B, K, d_slot]

            # Attention scores
            attn_logits = torch.einsum('bkd,bnd->bkn', q, k) * self.scale  # [B, K, N]
            attn = F.softmax(attn_logits, dim=1)  # Softmax over slots

            # Normalize attention over slots (competition)
            attn = attn / (attn.sum(dim=-1, keepdim=True) + self.eps)  # [B, K, N]

            # Weighted sum of values
            updates = torch.einsum('bkn,bnd->bkd', attn, v)  # [B, K, d_slot]

            # GRU update
            slots = self.gru(
                updates.reshape(B * self.num_slots, self.d_slot),
                slots_prev.reshape(B * self.num_slots, self.d_slot)
            ).reshape(B, self.num_slots, self.d_slot)

            # MLP refinement
            slots = slots + self.mlp(self.norm_mlp(slots))

        # Reshape attention maps to spatial
        attn_maps = attn  # [B, K, N]

        return slots, attn_maps


class SlotEncoder(nn.Module):
    """
    Complete encoder: Palette → CNN → SlotAttention → (Z, M, P)

    Returns:
        Z: [B, K, d_slot] slot features
        M: [B, K, H, W] soft masks
        P: [B, K, 2] centroids
    """

    def __init__(self,
                 num_slots: int = 8,
                 d_color: int = 16,
                 d_feat: int = 64,
                 d_slot: int = 128,
                 num_iters: int = 3,
                 H: int = 30,
                 W: int = 30):
        super().__init__()
        self.num_slots = num_slots
        self.d_slot = d_slot
        self.H = H
        self.W = W

        # Components
        self.palette_embed = PaletteEmbedding(d_color)
        self.cnn = CNNFeatureExtractor(d_color, d_feat, n_layers=4)
        self.slot_attn = SlotAttention(d_feat, d_slot, num_slots, num_iters)

        # Positional encoding (optional, improves spatial reasoning)
        self.use_pos_encoding = True
        if self.use_pos_encoding:
            self.pos_encoding = nn.Parameter(torch.randn(1, H, W, d_feat))

    def forward(self, x: torch.Tensor) -> Dict[str, torch.Tensor]:
        """
        Args:
            x: [B, H, W] long tensor (0-9)

        Returns:
            dict with:
                - slots_z: [B, K, d_slot]
                - slots_m: [B, K, H, W] masks
                - slots_p: [B, K, 2] centroids (y, x)
        """
        B = x.shape[0]

        # Embed palette
        x_embed = self.palette_embed(x)  # [B, H, W, d_color]

        # Extract features
        feats = self.cnn(x_embed)  # [B, H, W, d_feat]

        # Add positional encoding
        if self.use_pos_encoding:
            feats = feats + self.pos_encoding

        # Slot attention
        slots_z, attn_maps = self.slot_attn(feats)  # [B, K, d_slot], [B, K, H*W]

        # Reshape attention to spatial masks
        slots_m = attn_maps.view(B, self.num_slots, self.H, self.W)  # [B, K, H, W]

        # Compute centroids from masks
        slots_p = self._compute_centroids(slots_m)  # [B, K, 2]

        return {
            "slots_z": slots_z,
            "slots_m": slots_m,
            "slots_p": slots_p,
            "attn_maps": attn_maps,
        }

    def _compute_centroids(self, masks: torch.Tensor) -> torch.Tensor:
        """
        Compute centroids from masks.

        Args:
            masks: [B, K, H, W]

        Returns:
            [B, K, 2] centroids (y, x) in range [0, H-1] × [0, W-1]
        """
        B, K, H, W = masks.shape

        # Create coordinate grids
        y_grid = torch.arange(H, dtype=masks.dtype, device=masks.device).view(1, 1, H, 1)
        x_grid = torch.arange(W, dtype=masks.dtype, device=masks.device).view(1, 1, 1, W)

        # Normalize masks
        mask_sum = masks.sum(dim=[2, 3], keepdim=True) + 1e-8

        # Compute weighted average
        y_center = (masks * y_grid).sum(dim=[2, 3]) / mask_sum.squeeze([2, 3])
        x_center = (masks * x_grid).sum(dim=[2, 3]) / mask_sum.squeeze([2, 3])

        centroids = torch.stack([y_center, x_center], dim=-1)  # [B, K, 2]

        return centroids


if __name__ == "__main__":
    # Test encoder
    print("Testing SlotEncoder...")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    # Create encoder
    encoder = SlotEncoder(
        num_slots=8,
        d_color=16,
        d_feat=64,
        d_slot=128,
        num_iters=3,
        H=30,
        W=30
    ).to(device)

    # Random input
    B = 4
    x = torch.randint(0, 11, (B, 30, 30), device=device)

    # Forward pass
    with torch.no_grad():
        outputs = encoder(x)

    print(f"\nInput shape: {x.shape}")
    print(f"Slots Z: {outputs['slots_z'].shape}")
    print(f"Slots M: {outputs['slots_m'].shape}")
    print(f"Slots P: {outputs['slots_p'].shape}")

    # Check centroids are in valid range
    assert outputs['slots_p'].min() >= 0
    assert outputs['slots_p'][:, :, 0].max() < 30  # y
    assert outputs['slots_p'][:, :, 1].max() < 30  # x

    print("\n✓ SlotEncoder tests passed!")
