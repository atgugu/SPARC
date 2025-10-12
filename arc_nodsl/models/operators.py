"""
Latent Operators for ARC transformations.

Each operator is a small network that edits (z, m, p) - the slot features,
masks, and positions - to perform transformations like:
- Geometry: move, rotate, flip
- Mask morph: dilate, erode, outline
- Color: palette remapping
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, Tuple, Optional
import math


class SetTransformer(nn.Module):
    """
    Simple set transformer for processing slots.
    Applies self-attention and cross-attention to a global summary.
    """

    def __init__(self, d_slot: int, d_hidden: int, n_heads: int = 4):
        super().__init__()
        self.d_slot = d_slot

        # Self-attention among slots
        self.self_attn = nn.MultiheadAttention(d_slot, n_heads, batch_first=True)
        self.norm1 = nn.LayerNorm(d_slot)

        # Cross-attention to global context
        self.cross_attn = nn.MultiheadAttention(d_slot, n_heads, batch_first=True)
        self.norm2 = nn.LayerNorm(d_slot)

        # MLP
        self.mlp = nn.Sequential(
            nn.Linear(d_slot, d_hidden),
            nn.ReLU(inplace=True),
            nn.Linear(d_hidden, d_slot)
        )
        self.norm3 = nn.LayerNorm(d_slot)

    def forward(self, slots: torch.Tensor, context: torch.Tensor) -> torch.Tensor:
        """
        Args:
            slots: [B, K, d_slot]
            context: [B, 1, d_slot] or [B, d_slot]

        Returns:
            [B, K, d_slot] updated slots
        """
        # Ensure context has shape [B, 1, d_slot]
        if context.dim() == 2:
            context = context.unsqueeze(1)

        # Self-attention
        attn_out, _ = self.self_attn(slots, slots, slots)
        slots = self.norm1(slots + attn_out)

        # Cross-attention to context
        cross_out, _ = self.cross_attn(slots, context, context)
        slots = self.norm2(slots + cross_out)

        # MLP
        mlp_out = self.mlp(slots)
        slots = self.norm3(slots + mlp_out)

        return slots


class GeometryHead(nn.Module):
    """
    Predict geometry transformations: translation, rotation, flip.
    """

    def __init__(self, d_slot: int):
        super().__init__()

        # Translation offset
        self.fc_translate = nn.Linear(d_slot, 2)  # (dy, dx)

        # Rotation/flip logits (4 rotations × 2 flip states = 8 possibilities)
        self.fc_rot_flip = nn.Linear(d_slot, 8)

        # Scale factor
        self.fc_scale = nn.Linear(d_slot, 1)

    def forward(self, slot_features: torch.Tensor) -> Dict[str, torch.Tensor]:
        """
        Args:
            slot_features: [B, K, d_slot]

        Returns:
            dict with:
                - delta_pos: [B, K, 2] translation offsets
                - rot_flip_logits: [B, K, 8] rotation/flip logits
                - scale: [B, K, 1] scale factors
        """
        delta_pos = self.fc_translate(slot_features)  # [B, K, 2]
        rot_flip_logits = self.fc_rot_flip(slot_features)  # [B, K, 8]
        scale = torch.sigmoid(self.fc_scale(slot_features))  # [B, K, 1], range [0, 1]

        return {
            "delta_pos": delta_pos,
            "rot_flip_logits": rot_flip_logits,
            "scale": scale,
        }


class MaskMorphHead(nn.Module):
    """
    Predict mask morphological operations: dilate, erode, outline.
    Generates a small edit field that modifies the mask.
    """

    def __init__(self, d_slot: int, H: int = 30, W: int = 30):
        super().__init__()
        self.H = H
        self.W = W

        # Project slot to spatial dimensions
        # Use a small spatial resolution for efficiency
        self.spatial_res = 8
        self.fc_spatial = nn.Linear(d_slot, self.spatial_res * self.spatial_res)

        # Upsample to full resolution
        self.upsample = nn.Sequential(
            nn.ConvTranspose2d(1, 8, kernel_size=4, stride=2, padding=1),  # 8→16
            nn.ReLU(inplace=True),
            nn.ConvTranspose2d(8, 1, kernel_size=4, stride=2, padding=1),  # 16→32, crop to 30
        )

    def forward(self, slot_features: torch.Tensor) -> torch.Tensor:
        """
        Args:
            slot_features: [B, K, d_slot]

        Returns:
            edit_fields: [B, K, H, W] spatial edit fields
        """
        B, K, _ = slot_features.shape

        # Project to spatial
        spatial = self.fc_spatial(slot_features)  # [B, K, res*res]
        spatial = spatial.view(B * K, 1, self.spatial_res, self.spatial_res)

        # Upsample
        edit_fields = self.upsample(spatial)  # [B*K, 1, 32, 32]
        edit_fields = edit_fields[:, 0, :self.H, :self.W]  # Crop to [B*K, H, W]
        edit_fields = edit_fields.view(B, K, self.H, self.W)

        return edit_fields


class ColorHead(nn.Module):
    """
    Predict color remapping: 11×11 palette permutation matrix.
    """

    def __init__(self, d_slot: int):
        super().__init__()

        # Option 1: Global color map (shared across slots)
        self.fc_color_global = nn.Linear(d_slot, 11 * 11)

        # Option 2: Per-slot color bias
        self.fc_color_bias = nn.Linear(d_slot, 11)

    def forward(self, slot_features: torch.Tensor, mode: str = "bias") -> torch.Tensor:
        """
        Args:
            slot_features: [B, K, d_slot]
            mode: "global" or "bias"

        Returns:
            If mode=="global": [B, K, 11, 11] color mapping logits
            If mode=="bias": [B, K, 11] color bias logits
        """
        if mode == "global":
            logits = self.fc_color_global(slot_features)  # [B, K, 121]
            logits = logits.view(logits.shape[0], logits.shape[1], 11, 11)
            return logits
        else:  # bias
            bias = self.fc_color_bias(slot_features)  # [B, K, 11]
            return bias


class LatentOp(nn.Module):
    """
    A single latent operator that can edit slots.

    Processes slots through a set transformer and applies transformations
    via geometry, mask morph, and color heads.
    """

    def __init__(self,
                 d_slot: int = 128,
                 d_hidden: int = 128,
                 H: int = 30,
                 W: int = 30,
                 n_heads: int = 4):
        super().__init__()
        self.d_slot = d_slot
        self.H = H
        self.W = W

        # Set transformer for processing slots
        self.set_transformer = SetTransformer(d_slot, d_hidden, n_heads)

        # Gate: which slots to edit
        self.gate_fc = nn.Linear(d_slot, 1)

        # Transformation heads
        self.geometry = GeometryHead(d_slot)
        self.mask_morph = MaskMorphHead(d_slot, H, W)
        self.color = ColorHead(d_slot)

    def forward(self,
                slots_z: torch.Tensor,
                slots_m: torch.Tensor,
                slots_p: torch.Tensor,
                task_context: Optional[torch.Tensor] = None) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, Dict]:
        """
        Args:
            slots_z: [B, K, d_slot] slot features
            slots_m: [B, K, H, W] slot masks
            slots_p: [B, K, 2] slot positions
            task_context: [B, d_slot] optional task embedding

        Returns:
            (slots_z_new, slots_m_new, slots_p_new, aux_dict)
        """
        B, K, _ = slots_z.shape

        # Create context (task embedding or mean slot)
        if task_context is None:
            context = slots_z.mean(dim=1)  # [B, d_slot]
        else:
            context = task_context

        # Process slots with set transformer
        slots_z_processed = self.set_transformer(slots_z, context)

        # Compute gates (which slots to edit)
        gates = torch.sigmoid(self.gate_fc(slots_z_processed)).squeeze(-1)  # [B, K]

        # Apply transformations
        geom_params = self.geometry(slots_z_processed)
        mask_edits = self.mask_morph(slots_z_processed)
        color_bias = self.color(slots_z_processed, mode="bias")

        # Update positions
        delta_pos = geom_params["delta_pos"]  # [B, K, 2]
        slots_p_new = slots_p + gates.unsqueeze(-1) * delta_pos

        # Update masks (additive edit field)
        mask_edits_gated = gates.unsqueeze(-1).unsqueeze(-1) * mask_edits  # [B, K, H, W]
        slots_m_logits = torch.logit(slots_m.clamp(1e-6, 1-1e-6))
        slots_m_new = torch.sigmoid(slots_m_logits + mask_edits_gated)

        # Update features (residual connection)
        slots_z_new = slots_z + gates.unsqueeze(-1) * slots_z_processed

        # Auxiliary outputs for losses
        aux = {
            "gates": gates,  # [B, K] for sparsity loss
            "delta_pos": delta_pos,  # [B, K, 2] for L1 penalty
            "mask_edits": mask_edits,  # [B, K, H, W] for perimeter penalty
            "color_bias": color_bias,  # [B, K, 11]
        }

        return slots_z_new, slots_m_new, slots_p_new, aux


class OperatorLibrary(nn.Module):
    """
    A library of M latent operators.
    """

    def __init__(self,
                 num_ops: int = 8,
                 d_slot: int = 128,
                 d_hidden: int = 128,
                 H: int = 30,
                 W: int = 30,
                 n_heads: int = 4):
        super().__init__()
        self.num_ops = num_ops

        # Create M operators
        self.operators = nn.ModuleList([
            LatentOp(d_slot, d_hidden, H, W, n_heads)
            for _ in range(num_ops)
        ])

    def forward(self, op_idx: int, *args, **kwargs):
        """
        Apply a specific operator.

        Args:
            op_idx: Index of operator to use (0 to M-1)
            *args, **kwargs: Arguments to pass to the operator

        Returns:
            Output of the operator
        """
        return self.operators[op_idx](*args, **kwargs)

    def apply_sequence(self,
                      op_indices: torch.Tensor,
                      slots_z: torch.Tensor,
                      slots_m: torch.Tensor,
                      slots_p: torch.Tensor,
                      task_context: Optional[torch.Tensor] = None) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, list]:
        """
        Apply a sequence of operators.

        Args:
            op_indices: [T] sequence of operator indices
            slots_z, slots_m, slots_p: Initial slot states
            task_context: Optional task embedding

        Returns:
            (final_z, final_m, final_p, aux_list)
        """
        aux_list = []

        for t, op_idx in enumerate(op_indices):
            op_idx = int(op_idx.item()) if torch.is_tensor(op_idx) else op_idx
            slots_z, slots_m, slots_p, aux = self.operators[op_idx](
                slots_z, slots_m, slots_p, task_context
            )
            aux_list.append(aux)

        return slots_z, slots_m, slots_p, aux_list


if __name__ == "__main__":
    # Test operators
    import sys
    from pathlib import Path
    sys.path.insert(0, str(Path(__file__).parent.parent.parent))

    print("Testing LatentOp and OperatorLibrary...")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    # Create operator library
    op_library = OperatorLibrary(
        num_ops=8,
        d_slot=128,
        d_hidden=128,
        H=30,
        W=30
    ).to(device)

    # Random slots
    B = 4
    K = 8
    slots_z = torch.randn(B, K, 128, device=device)
    slots_m = torch.rand(B, K, 30, 30, device=device)
    slots_p = torch.rand(B, K, 2, device=device) * 30

    # Test single operator
    print("\n=== Single Operator ===")
    z_new, m_new, p_new, aux = op_library(0, slots_z, slots_m, slots_p)

    print(f"Input Z: {slots_z.shape}")
    print(f"Output Z: {z_new.shape}")
    print(f"Output M: {m_new.shape}")
    print(f"Output P: {p_new.shape}")
    print(f"Gates shape: {aux['gates'].shape}")
    print(f"Gates mean: {aux['gates'].mean().item():.3f}")

    # Test sequence
    print("\n=== Operator Sequence ===")
    op_sequence = torch.tensor([0, 2, 5])  # Apply ops 0, 2, 5
    z_final, m_final, p_final, aux_list = op_library.apply_sequence(
        op_sequence, slots_z, slots_m, slots_p
    )

    print(f"Sequence length: {len(op_sequence)}")
    print(f"Final Z: {z_final.shape}")
    print(f"Number of aux dicts: {len(aux_list)}")

    print("\n✓ Operator tests passed!")
