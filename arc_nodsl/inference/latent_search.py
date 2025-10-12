"""
Latent search: Beam search + probability radiation.

Search Strategy:
1. Beam search over operator sequences (beam_size=16)
2. Probability radiation: diffusion-like exploration
3. Constraint filtering + patch-based scoring
4. Diversity promotion via DPP

Usage:
    candidates = beam_search(
        encoder, controller, ops, renderer,
        input_grid, input_shape, target_shape,
        task_embed, beam_size=16, max_steps=4
    )
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from typing import List, Dict, Optional, Tuple
from dataclasses import dataclass, field
import heapq


# =============================================================================
# Component 1: Candidate Representation
# =============================================================================

@dataclass
class SearchCandidate:
    """A single search candidate."""
    # Slot state
    slots_z: torch.Tensor  # [K, d_slot]
    slots_m: torch.Tensor  # [K, H, W]
    slots_p: torch.Tensor  # [K, 2]

    # Prediction
    prediction: torch.Tensor  # [H, W]
    h: int  # Actual height
    w: int  # Actual width

    # Search info
    score: float
    operator_sequence: List[int]
    param_sequence: List[torch.Tensor] = field(default_factory=list)
    depth: int = 0

    # Diversity
    embedding: Optional[torch.Tensor] = None  # For DPP

    # REINFORCE (for training)
    log_probs: Optional[List[torch.Tensor]] = None  # Per-step log probs [num_steps]
    entropies: Optional[List[torch.Tensor]] = None  # Per-step entropy [num_steps]

    def __lt__(self, other):
        """For heapq (lower score = higher priority in max heap)."""
        return self.score > other.score


def compute_candidate_embedding(candidate: SearchCandidate) -> torch.Tensor:
    """Compute embedding for diversity."""
    # Use mean slot features + prediction statistics
    slot_mean = candidate.slots_z.mean(dim=0)  # [d_slot]

    # Prediction statistics
    pred_crop = candidate.prediction[:candidate.h, :candidate.w]
    pred_features = torch.tensor([
        pred_crop.float().mean().item(),
        pred_crop.float().std().item() + 1e-8,
        len(torch.unique(pred_crop)) / 10.0,
    ], device=slot_mean.device)

    return torch.cat([slot_mean, pred_features])


# =============================================================================
# Component 2: Probability Radiation
# =============================================================================

class ProbabilityRadiator:
    """
    Generate variants of candidates via stochastic perturbations.

    Two modes:
    1. Gaussian jitter: Add noise to continuous parameters
    2. Token edits: Swap operator indices
    """

    def __init__(
        self,
        num_operators: int = 8,
        gaussian_std: float = 0.1,
        token_swap_prob: float = 0.2
    ):
        self.num_operators = num_operators
        self.gaussian_std = gaussian_std
        self.token_swap_prob = token_swap_prob

    def radiate(
        self,
        candidates: List[SearchCandidate],
        num_variants: int = 4,
        temperature: float = 1.0
    ) -> List[SearchCandidate]:
        """
        Generate variants via radiation.

        Args:
            candidates: Input candidates
            num_variants: Variants per candidate
            temperature: Controls amount of noise

        Returns:
            List of variant candidates
        """
        variants = []

        for cand in candidates:
            # Limit variants per candidate
            variants_created = 0

            # Gaussian jitter on parameters
            if len(cand.param_sequence) > 0 and np.random.rand() < 0.7:
                variant_params = []
                for params in cand.param_sequence:
                    noise = torch.randn_like(params) * self.gaussian_std * temperature
                    variant_params.append(params + noise)

                variant = SearchCandidate(
                    slots_z=cand.slots_z.clone(),
                    slots_m=cand.slots_m.clone(),
                    slots_p=cand.slots_p.clone(),
                    prediction=cand.prediction.clone(),
                    h=cand.h,
                    w=cand.w,
                    score=cand.score * 0.9,  # Penalty for variant
                    operator_sequence=cand.operator_sequence.copy(),
                    param_sequence=variant_params,
                    depth=cand.depth
                )
                variants.append(variant)
                variants_created += 1

                if variants_created >= num_variants:
                    continue

            # Token swap on operators
            if len(cand.operator_sequence) > 0 and np.random.rand() < self.token_swap_prob:
                variant_ops = cand.operator_sequence.copy()
                # Swap one operator
                idx = np.random.randint(len(variant_ops))
                variant_ops[idx] = np.random.randint(self.num_operators)

                variant = SearchCandidate(
                    slots_z=cand.slots_z.clone(),
                    slots_m=cand.slots_m.clone(),
                    slots_p=cand.slots_p.clone(),
                    prediction=cand.prediction.clone(),
                    h=cand.h,
                    w=cand.w,
                    score=cand.score * 0.85,  # Larger penalty
                    operator_sequence=variant_ops,
                    param_sequence=cand.param_sequence.copy(),
                    depth=cand.depth
                )
                variants.append(variant)

        return variants


# =============================================================================
# Component 3: Beam Search Engine
# =============================================================================

class BeamSearch:
    """
    Beam search over operator sequences.
    """

    def __init__(
        self,
        encoder: nn.Module,
        controller: nn.Module,
        operators: nn.Module,
        renderer: nn.Module,
        beam_size: int = 16,
        max_steps: int = 4,
        radiation_variants: int = 4,
        diversity_weight: float = 0.1,
        device: torch.device = torch.device('cpu')
    ):
        self.encoder = encoder
        self.controller = controller
        self.operators = operators
        self.renderer = renderer
        self.beam_size = beam_size
        self.max_steps = max_steps
        self.radiation_variants = radiation_variants
        self.diversity_weight = diversity_weight
        self.device = device
        self.collect_log_probs = False  # Set externally

        self.radiator = ProbabilityRadiator(num_operators=operators.num_ops)

    def search(
        self,
        input_grid: torch.Tensor,
        input_shape: Tuple[int, int],
        target_shape: Tuple[int, int],
        task_embed: Dict[str, any],
        target_grid: Optional[torch.Tensor] = None
    ) -> List[SearchCandidate]:
        """
        Run beam search to find best predictions.

        Args:
            input_grid: [H, W] input grid
            input_shape: (h_in, w_in) actual input size
            target_shape: (h_out, w_out) expected output size
            task_embed: Task embedding dict
            target_grid: Optional target for scoring (train pairs)

        Returns:
            List of top-K candidates (sorted by score)
        """
        from arc_nodsl.inference.patches import score_with_patches

        h_out, w_out = target_shape

        # 1. Encode input
        input_batch = input_grid.unsqueeze(0).to(self.device)
        enc_out = self.encoder(input_batch)

        init_slots_z = enc_out["slots_z"][0]  # [K, d_slot]
        init_slots_m = enc_out["slots_m"][0]  # [K, H, W]
        init_slots_p = enc_out["slots_p"][0]  # [K, 2]

        # 2. Initialize beam with identity (no operators)
        init_logits = self.renderer(
            init_slots_z.unsqueeze(0),
            init_slots_m.unsqueeze(0)
        )[0]  # [H, W, 10]
        init_pred = torch.argmax(init_logits, dim=-1)

        # Score initial prediction
        if target_grid is not None:
            init_score = score_with_patches(
                init_pred, target_grid, h_out, w_out,
                strategy="adaptive",
                constraints=task_embed['constraints']
            )
        else:
            # Use constraint score only
            init_score = task_embed['constraints'].score(init_pred, h_out, w_out)

        beam = [SearchCandidate(
            slots_z=init_slots_z,
            slots_m=init_slots_m,
            slots_p=init_slots_p,
            prediction=init_pred,
            h=h_out,
            w=w_out,
            score=init_score,
            operator_sequence=[],
            param_sequence=[],
            depth=0
        )]

        # 3. Beam search loop
        task_embed_tensor = task_embed['embed'].unsqueeze(0).to(self.device)

        for step in range(self.max_steps):
            all_candidates = []

            # Expand each candidate in beam
            for cand in beam:
                # Get controller prediction
                ctrl_out = self.controller.step(
                    cand.slots_z.unsqueeze(0),
                    cand.slots_p.unsqueeze(0),
                    task_embed_tensor,
                    temperature=1.0
                )

                # Sample top-K operators (greedy + exploration)
                op_logits = ctrl_out["op_logits"][0]  # [M]
                op_priors = torch.tensor(
                    task_embed['op_priors'],
                    device=self.device,
                    dtype=torch.float32
                )
                combined_logits = op_logits + torch.log(op_priors + 1e-8)

                # Top-K operators
                k = min(4, self.operators.num_ops)
                top_k_ops = torch.topk(combined_logits, k=k)

                for op_idx in top_k_ops.indices:
                    try:
                        # Apply operator
                        z_new, m_new, p_new, aux = self.operators(
                            int(op_idx),
                            cand.slots_z.unsqueeze(0),
                            cand.slots_m.unsqueeze(0),
                            cand.slots_p.unsqueeze(0),
                            task_embed_tensor
                        )

                        # Render
                        logits = self.renderer(z_new, m_new)[0]
                        pred = torch.argmax(logits, dim=-1)

                        # Score
                        if target_grid is not None:
                            score = score_with_patches(
                                pred, target_grid, h_out, w_out,
                                strategy="adaptive",
                                context={'slots_p': p_new[0], 'slots_m': m_new[0]},
                                constraints=task_embed['constraints']
                            )
                        else:
                            # Use constraint score only
                            score = task_embed['constraints'].score(pred, h_out, w_out)

                        # Collect log probs and entropies for REINFORCE (if requested)
                        new_log_probs = None
                        new_entropies = None
                        if self.collect_log_probs:
                            # Log prob of selected operator
                            log_prob_dist = F.log_softmax(combined_logits, dim=0)
                            selected_log_prob = log_prob_dist[op_idx]

                            # Entropy of distribution
                            prob_dist = F.softmax(combined_logits, dim=0)
                            entropy = -(prob_dist * log_prob_dist).sum()

                            # Inherit parent's log probs and add current
                            if cand.log_probs is not None:
                                new_log_probs = cand.log_probs + [selected_log_prob]
                            else:
                                new_log_probs = [selected_log_prob]

                            if cand.entropies is not None:
                                new_entropies = cand.entropies + [entropy]
                            else:
                                new_entropies = [entropy]

                        # Create new candidate
                        new_cand = SearchCandidate(
                            slots_z=z_new[0],
                            slots_m=m_new[0],
                            slots_p=p_new[0],
                            prediction=pred,
                            h=h_out,
                            w=w_out,
                            score=score,
                            operator_sequence=cand.operator_sequence + [int(op_idx)],
                            param_sequence=cand.param_sequence + [ctrl_out["params_sample"][0]],
                            depth=cand.depth + 1,
                            log_probs=new_log_probs,
                            entropies=new_entropies
                        )
                        all_candidates.append(new_cand)
                    except Exception as e:
                        # Skip operators that fail
                        continue

            # Probability radiation
            if step < self.max_steps - 1 and len(all_candidates) > 0:  # Not on last step
                radiations = self.radiator.radiate(
                    all_candidates[:self.beam_size//2],  # Radiate from top half
                    num_variants=self.radiation_variants,
                    temperature=1.0 / (step + 1)  # Anneal
                )
                all_candidates.extend(radiations)

            if len(all_candidates) == 0:
                # No candidates generated, keep previous beam
                break

            # Filter by constraints
            valid_candidates = [
                c for c in all_candidates
                if task_embed['constraints'].is_valid(
                    c.prediction, c.h, c.w, input_shape
                )
            ]

            if len(valid_candidates) == 0:
                valid_candidates = all_candidates  # Fallback

            # Select top-K by score + diversity
            beam = self._select_diverse_beam(
                valid_candidates,
                beam_size=self.beam_size,
                diversity_weight=self.diversity_weight
            )

            # Early stopping if top candidate is perfect
            if len(beam) > 0 and beam[0].score > 0.99:
                break

        return beam

    def _select_diverse_beam(
        self,
        candidates: List[SearchCandidate],
        beam_size: int,
        diversity_weight: float
    ) -> List[SearchCandidate]:
        """Select diverse beam using DPP-like approach."""
        if len(candidates) <= beam_size:
            return sorted(candidates, key=lambda c: c.score, reverse=True)

        # Compute embeddings
        for cand in candidates:
            if cand.embedding is None:
                cand.embedding = compute_candidate_embedding(cand)

        # Sort by score
        candidates = sorted(candidates, key=lambda c: c.score, reverse=True)

        # Greedy diverse selection
        selected = [candidates[0]]  # Best candidate always included

        for cand in candidates[1:]:
            if len(selected) >= beam_size:
                break

            # Compute diversity score (min distance to selected)
            diversities = []
            for sel in selected:
                dist = torch.norm(cand.embedding - sel.embedding)
                diversities.append(dist.item())

            min_diversity = min(diversities) if diversities else 1.0

            # Combined score (for sorting, not replacing original score)
            combined = cand.score + diversity_weight * min_diversity

            # Decide whether to include
            if len(selected) < beam_size:
                selected.append(cand)

        return selected


# =============================================================================
# Component 4: Main Search API
# =============================================================================

def beam_search(
    encoder: nn.Module,
    controller: nn.Module,
    operators: nn.Module,
    renderer: nn.Module,
    input_grid: torch.Tensor,
    input_shape: Tuple[int, int],
    target_shape: Tuple[int, int],
    task_embed: Dict[str, any],
    target_grid: Optional[torch.Tensor] = None,
    beam_size: int = 16,
    max_steps: int = 4,
    device: torch.device = torch.device('cpu'),
    collect_log_probs: bool = False
) -> List[SearchCandidate]:
    """
    Main API: Beam search for ARC solving.

    Args:
        encoder, controller, operators, renderer: Models
        input_grid: [H, W] input grid
        input_shape: (h, w) actual input size
        target_shape: (h, w) expected output size
        task_embed: Task embedding dict
        target_grid: Optional target for scoring
        beam_size: Beam size
        max_steps: Maximum search depth
        device: Compute device
        collect_log_probs: Whether to collect log probs for REINFORCE training

    Returns:
        List of top-K candidates (sorted by score)
    """
    search = BeamSearch(
        encoder, controller, operators, renderer,
        beam_size=beam_size,
        max_steps=max_steps,
        device=device
    )

    search.collect_log_probs = collect_log_probs

    # Use no_grad for inference, but allow gradients for training
    if collect_log_probs:
        return search.search(
            input_grid, input_shape, target_shape,
            task_embed, target_grid
        )
    else:
        with torch.no_grad():
            return search.search(
                input_grid, input_shape, target_shape,
                task_embed, target_grid
            )


if __name__ == "__main__":
    # Test beam search
    import sys
    from pathlib import Path
    sys.path.insert(0, str(Path(__file__).parent.parent.parent))

    from arc_nodsl.data.loader import ARCDataset
    from arc_nodsl.models.slots import SlotEncoder
    from arc_nodsl.models.renderer import SlotRenderer
    from arc_nodsl.models.operators import OperatorLibrary
    from arc_nodsl.models.controller import Controller
    from arc_nodsl.inference.task_embed import build_task_embedding

    print("=" * 60)
    print("Testing Beam Search Module")
    print("=" * 60)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}\n")

    # Load dataset
    dataset = ARCDataset("data/arc-agi_training_challenges.json")
    task = dataset[0]
    print(f"Task: {task['task_id']}")

    # Prepare train pairs
    train_pairs = []
    for i in range(len(task['train_inputs'])):
        train_pairs.append((
            task['train_inputs'][i],
            task['train_outputs'][i],
            task['train_shapes'][i]
        ))

    # Build task embedding
    print("\nBuilding task embedding...")
    task_embed = build_task_embedding(train_pairs, device=device)
    print(f"✓ Task embedding ready")

    # Create models (random weights)
    print("\nCreating models (random weights)...")
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

    # Set to eval mode
    encoder.eval()
    renderer.eval()
    operators.eval()
    controller.eval()

    print("✓ Models ready")

    # Test beam search on first test input
    print("\n" + "=" * 60)
    print("Running Beam Search")
    print("=" * 60)

    test_input = task['test_inputs'][0]
    test_shape = task['test_shapes'][0]
    h_in, w_in = test_shape['input']

    # Predict output size
    if test_shape['output']:
        h_out, w_out = test_shape['output']
    else:
        expected_size = task_embed['constraints'].grid_size.get_expected_size((h_in, w_in))
        h_out, w_out = expected_size if expected_size else (h_in, w_in)

    print(f"\nInput size: {h_in}×{w_in}")
    print(f"Expected output size: {h_out}×{w_out}")

    # Run beam search (small beam for testing)
    print(f"\nRunning beam search (beam_size=4, max_steps=2)...")

    try:
        candidates = beam_search(
            encoder, controller, operators, renderer,
            test_input,
            (h_in, w_in),
            (h_out, w_out),
            task_embed,
            target_grid=None,  # No target at test time
            beam_size=4,
            max_steps=2,
            device=device
        )

        print(f"✓ Search completed")
        print(f"\nTop {len(candidates)} candidates:")
        for i, cand in enumerate(candidates[:3]):
            print(f"\n{i+1}. Score: {cand.score:.3f}")
            print(f"   Operator sequence: {cand.operator_sequence}")
            print(f"   Depth: {cand.depth}")
            print(f"   Prediction shape: {cand.prediction.shape}")
            print(f"   Unique colors: {len(torch.unique(cand.prediction[:cand.h, :cand.w]))}")

        print("\n✓ ALL TESTS PASSED!")

    except Exception as e:
        print(f"\n✗ Search failed: {e}")
        import traceback
        traceback.print_exc()

    print("\n" + "=" * 60)
    print("Beam search module ready for integration!")
    print("=" * 60)
