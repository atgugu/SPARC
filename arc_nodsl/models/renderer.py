"""
Slot Renderer: Decode slots back to output grids.

Decodes each slot independently to logits + alpha, then composites them.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Tuple, Optional


class SlotDecoder(nn.Module):
    """
    Decode a single slot to per-pixel logits and alpha.
    """

    def __init__(self, d_slot: int = 128, d_hidden: int = 64, H: int = 30, W: int = 30):
        super().__init__()
        self.d_slot = d_slot
        self.H = H
        self.W = W

        # Spatial broadcast: add positional encodings
        self.pos_encoding = nn.Parameter(torch.randn(1, H, W, 2))

        # Decoder network
        self.decoder = nn.Sequential(
            nn.Linear(d_slot + 2, d_hidden),
            nn.ReLU(inplace=True),
            nn.Linear(d_hidden, d_hidden),
            nn.ReLU(inplace=True),
            nn.Linear(d_hidden, 11 + 1),  # 11 color logits (0-10) + 1 alpha
        )

    def forward(self, slot_z: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Args:
            slot_z: [B, d_slot]

        Returns:
            logits: [B, H, W, 11]
            alpha: [B, H, W]
        """
        B = slot_z.shape[0]

        # Broadcast slot to all positions
        slot_broadcast = slot_z.view(B, 1, 1, self.d_slot).expand(B, self.H, self.W, self.d_slot)

        # Concatenate with positional encoding
        pos = self.pos_encoding.expand(B, -1, -1, -1)
        decoder_input = torch.cat([slot_broadcast, pos], dim=-1)  # [B, H, W, d_slot+2]

        # Decode
        decoder_output = self.decoder(decoder_input)  # [B, H, W, 12]

        # Split into logits and alpha
        logits = decoder_output[..., :11]  # [B, H, W, 11]
        alpha = torch.sigmoid(decoder_output[..., 11])  # [B, H, W]

        return logits, alpha


class SlotRenderer(nn.Module):
    """
    Render multiple slots to a single output grid via alpha compositing.
    """

    def __init__(self, d_slot: int = 128, d_hidden: int = 64, H: int = 30, W: int = 30, use_mask: bool = True):
        super().__init__()
        self.d_slot = d_slot
        self.H = H
        self.W = W
        self.use_mask = use_mask  # Use attention masks or learned alphas

        self.slot_decoder = SlotDecoder(d_slot, d_hidden, H, W)

    def forward(self,
                slots_z: torch.Tensor,
                slots_m: Optional[torch.Tensor] = None) -> torch.Tensor:
        """
        Args:
            slots_z: [B, K, d_slot]
            slots_m: [B, K, H, W] optional attention masks

        Returns:
            logits: [B, H, W, 11] output logits
        """
        B, K, _ = slots_z.shape

        all_logits = []
        all_alphas = []

        # Decode each slot
        for k in range(K):
            logits_k, alpha_k = self.slot_decoder(slots_z[:, k])  # [B, H, W, 11], [B, H, W]

            # Use attention mask if available and enabled
            if self.use_mask and slots_m is not None:
                alpha_k = slots_m[:, k]  # [B, H, W]

            all_logits.append(logits_k)
            all_alphas.append(alpha_k)

        # Stack: [B, K, H, W, 11] and [B, K, H, W]
        all_logits = torch.stack(all_logits, dim=1)
        all_alphas = torch.stack(all_alphas, dim=1)

        # Normalize alphas to sum to 1 (competition)
        all_alphas = all_alphas / (all_alphas.sum(dim=1, keepdim=True) + 1e-8)

        # Weighted sum over slots
        output_logits = torch.einsum('bkhwc,bkhw->bhwc', all_logits, all_alphas)  # [B, H, W, 11]

        return output_logits


class AutoEncoder(nn.Module):
    """
    Complete autoencoder: SlotEncoder + SlotRenderer.
    """

    def __init__(self,
                 num_slots: int = 8,
                 d_color: int = 16,
                 d_feat: int = 64,
                 d_slot: int = 128,
                 d_hidden: int = 64,
                 num_iters: int = 3,
                 H: int = 30,
                 W: int = 30,
                 use_mask: bool = True):
        super().__init__()

        from arc_nodsl.models.slots import SlotEncoder

        self.encoder = SlotEncoder(
            num_slots=num_slots,
            d_color=d_color,
            d_feat=d_feat,
            d_slot=d_slot,
            num_iters=num_iters,
            H=H,
            W=W
        )

        self.renderer = SlotRenderer(
            d_slot=d_slot,
            d_hidden=d_hidden,
            H=H,
            W=W,
            use_mask=use_mask
        )

    def forward(self, x: torch.Tensor) -> dict:
        """
        Args:
            x: [B, H, W] input grid (0-10)

        Returns:
            dict with:
                - recon_logits: [B, H, W, 11]
                - slots_z: [B, K, d_slot]
                - slots_m: [B, K, H, W]
                - slots_p: [B, K, 2]
        """
        # Encode
        enc_outputs = self.encoder(x)

        # Render
        recon_logits = self.renderer(enc_outputs["slots_z"], enc_outputs["slots_m"])

        return {
            "recon_logits": recon_logits,
            "slots_z": enc_outputs["slots_z"],
            "slots_m": enc_outputs["slots_m"],
            "slots_p": enc_outputs["slots_p"],
        }

    def reconstruct(self, x: torch.Tensor) -> torch.Tensor:
        """
        Convenience method for reconstruction.

        Returns:
            [B, H, W] reconstructed grid (argmax)
        """
        outputs = self.forward(x)
        logits = outputs["recon_logits"]  # [B, H, W, 11]
        recon = torch.argmax(logits, dim=-1)  # [B, H, W]
        return recon


def compute_reconstruction_loss(logits: torch.Tensor,
                                target: torch.Tensor,
                                h: Optional[int] = None,
                                w: Optional[int] = None) -> torch.Tensor:
    """
    Compute cross-entropy reconstruction loss.

    Args:
        logits: [B, H, W, 11]
        target: [B, H, W] ground truth
        h, w: Original size (crop if provided)

    Returns:
        scalar loss
    """
    if h is not None and w is not None:
        logits = logits[:, :h, :w, :]
        target = target[:, :h, :w]

    # Reshape for cross entropy
    B, H, W, C = logits.shape
    logits_flat = logits.reshape(B * H * W, C)
    target_flat = target.reshape(B * H * W)

    loss = F.cross_entropy(logits_flat, target_flat, reduction='mean')
    return loss


def compute_mask_diversity_loss(slots_m: torch.Tensor) -> torch.Tensor:
    """
    Encourage masks to be different (avoid collapse).

    Args:
        slots_m: [B, K, H, W]

    Returns:
        scalar loss (lower is more diverse)
    """
    B, K, H, W = slots_m.shape

    # Flatten masks
    masks_flat = slots_m.view(B, K, H * W)  # [B, K, N]

    # Compute pairwise cosine similarity
    masks_norm = F.normalize(masks_flat, dim=2, p=2)
    sim = torch.bmm(masks_norm, masks_norm.transpose(1, 2))  # [B, K, K]

    # Penalize high off-diagonal similarity
    mask = torch.eye(K, device=sim.device).unsqueeze(0)  # [1, K, K]
    off_diag_sim = sim * (1 - mask)

    loss = off_diag_sim.abs().sum(dim=[1, 2]).mean()
    return loss


if __name__ == "__main__":
    # Test renderer and autoencoder
    import sys
    from pathlib import Path
    sys.path.insert(0, str(Path(__file__).parent.parent.parent))

    print("Testing SlotRenderer and AutoEncoder...")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    # Create autoencoder
    autoencoder = AutoEncoder(
        num_slots=8,
        d_color=16,
        d_feat=64,
        d_slot=128,
        d_hidden=64,
        num_iters=3,
        H=30,
        W=30,
        use_mask=True
    ).to(device)

    # Random input
    B = 4
    x = torch.randint(0, 11, (B, 30, 30), device=device)

    # Forward pass
    with torch.no_grad():
        outputs = autoencoder(x)

    print(f"\nInput shape: {x.shape}")
    print(f"Recon logits: {outputs['recon_logits'].shape}")
    print(f"Slots Z: {outputs['slots_z'].shape}")
    print(f"Slots M: {outputs['slots_m'].shape}")

    # Test reconstruction
    recon = autoencoder.reconstruct(x)
    print(f"Reconstructed: {recon.shape}")

    # Test loss
    loss = compute_reconstruction_loss(outputs['recon_logits'], x)
    print(f"Reconstruction loss: {loss.item():.4f}")

    # Test diversity loss
    div_loss = compute_mask_diversity_loss(outputs['slots_m'])
    print(f"Mask diversity loss: {div_loss.item():.4f}")

    # Check reconstruction accuracy (random init, will be low)
    acc = (recon == x).float().mean()
    print(f"Reconstruction accuracy: {acc.item()*100:.1f}%")

    print("\n✓ SlotRenderer and AutoEncoder tests passed!")
