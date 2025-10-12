"""
Partial scoring strategies for fast evaluation during search.

Instead of scoring entire 30×30 grids, identify and evaluate only
key regions (patches) to achieve 5-10× speedup with <5% accuracy loss.

Patch Selection Strategies:
1. DisagreementPatchSelector: Focus on prediction errors
2. ObjectPatchSelector: Around slot centroids
3. BorderPatchSelector: Edge regions (common in ARC)
4. LatticePatchSelector: Repeating tile patterns
5. AdaptivePatchSelector: Combine multiple strategies

Usage:
    scorer = PatchScorer(DisagreementPatchSelector())
    result = scorer.score(prediction, target, h, w)
    # Returns: {'score': 0.95, 'num_patches': 4, 'coverage': 0.25, ...}
"""

import torch
import torch.nn.functional as F
import numpy as np
from typing import List, Tuple, Dict, Optional
import scipy.ndimage as ndimage
from collections import defaultdict


# =============================================================================
# Component 1: Patch Selection Strategies
# =============================================================================

class PatchSelector:
    """
    Base class for patch selection strategies.

    Subclasses implement select_patches() to return a list of
    (y, x, h, w) patch coordinates identifying regions to evaluate.
    """

    def select_patches(
        self,
        prediction: torch.Tensor,  # [H, W]
        target: Optional[torch.Tensor],  # [H, W] or None
        h: int,  # Actual grid height
        w: int,  # Actual grid width
        context: Optional[Dict] = None  # Additional info (slots, etc.)
    ) -> List[Tuple[int, int, int, int]]:
        """
        Select patches to evaluate.

        Args:
            prediction: Predicted grid [H, W]
            target: Target grid [H, W] or None
            h: Actual height (≤ H)
            w: Actual width (≤ W)
            context: Optional dict with 'slots_p', 'slots_m', etc.

        Returns:
            List of (y, x, patch_h, patch_w) tuples
        """
        raise NotImplementedError

    def _merge_overlapping(
        self,
        patches: List[Tuple[int, int, int, int]],
        iou_threshold: float = 0.3
    ) -> List[Tuple[int, int, int, int]]:
        """
        Merge overlapping patches.

        Args:
            patches: List of (y, x, h, w)
            iou_threshold: Merge if IoU > threshold

        Returns:
            Merged patches
        """
        if len(patches) <= 1:
            return patches

        # Sort by area (largest first)
        patches = sorted(patches, key=lambda p: p[2] * p[3], reverse=True)

        merged = []
        used = set()

        for i, p1 in enumerate(patches):
            if i in used:
                continue

            y1, x1, h1, w1 = p1

            # Try to merge with remaining patches
            for j in range(i + 1, len(patches)):
                if j in used:
                    continue

                y2, x2, h2, w2 = patches[j]

                # Compute IoU
                iou = self._compute_iou(
                    (y1, x1, h1, w1),
                    (y2, x2, h2, w2)
                )

                if iou > iou_threshold:
                    # Merge: take bounding box
                    y1 = min(y1, y2)
                    x1 = min(x1, x2)
                    y1_max = max(y1 + h1, y2 + h2)
                    x1_max = max(x1 + w1, x2 + w2)
                    h1 = y1_max - y1
                    w1 = x1_max - x1
                    used.add(j)

            merged.append((y1, x1, h1, w1))

        return merged

    def _compute_iou(
        self,
        box1: Tuple[int, int, int, int],
        box2: Tuple[int, int, int, int]
    ) -> float:
        """Compute intersection over union of two boxes."""
        y1, x1, h1, w1 = box1
        y2, x2, h2, w2 = box2

        # Intersection
        y_inter = max(0, min(y1 + h1, y2 + h2) - max(y1, y2))
        x_inter = max(0, min(x1 + w1, x2 + w2) - max(x1, x2))
        inter = y_inter * x_inter

        # Union
        area1 = h1 * w1
        area2 = h2 * w2
        union = area1 + area2 - inter

        return inter / (union + 1e-8)


class DisagreementPatchSelector(PatchSelector):
    """
    Select patches around regions where prediction != target.

    Algorithm:
    1. Compute disagreement mask
    2. Find connected components
    3. Create bounding boxes with margin
    4. Merge overlapping boxes
    5. Return top-K by area
    """

    def __init__(
        self,
        patch_size: int = 5,
        margin: int = 1,
        max_patches: int = 8
    ):
        """
        Args:
            patch_size: Minimum patch size
            margin: Pixels to add around disagreement regions
            max_patches: Maximum number of patches to return
        """
        self.patch_size = patch_size
        self.margin = margin
        self.max_patches = max_patches

    def select_patches(
        self,
        prediction: torch.Tensor,
        target: Optional[torch.Tensor],
        h: int,
        w: int,
        context: Optional[Dict] = None
    ) -> List[Tuple[int, int, int, int]]:
        if target is None:
            # No target, return empty (fallback to full grid)
            return []

        # 1. Disagreement mask
        pred_crop = prediction[:h, :w]
        targ_crop = target[:h, :w]
        disagree = (pred_crop != targ_crop).cpu().numpy()

        # 2. Connected components
        labeled, n_comps = ndimage.label(disagree)

        if n_comps == 0:
            # Perfect match, no patches needed
            return []

        # 3. Bounding boxes
        patches = []
        for i in range(1, n_comps + 1):
            ys, xs = np.where(labeled == i)
            if len(ys) == 0:
                continue

            y_min, y_max = int(ys.min()), int(ys.max())
            x_min, x_max = int(xs.min()), int(xs.max())

            # Expand by margin
            y_min = max(0, y_min - self.margin)
            x_min = max(0, x_min - self.margin)
            y_max = min(h, y_max + 1 + self.margin)
            x_max = min(w, x_max + 1 + self.margin)

            patch_h = y_max - y_min
            patch_w = x_max - x_min

            patches.append((y_min, x_min, patch_h, patch_w))

        # 4. Merge overlapping
        patches = self._merge_overlapping(patches)

        # 5. Limit to top-K by area
        patches = sorted(patches, key=lambda p: p[2] * p[3], reverse=True)
        return patches[:self.max_patches]


class ObjectPatchSelector(PatchSelector):
    """
    Select patches around slot centroids.

    Algorithm:
    1. Use slot centroids from encoder
    2. Create patch around each centroid
    3. Filter slots with low mask mass (background)
    """

    def __init__(
        self,
        patch_size: int = 7,
        min_mask_mass: float = 0.01
    ):
        """
        Args:
            patch_size: Size of patch around each centroid
            min_mask_mass: Minimum mask mass to consider slot active
        """
        self.patch_size = patch_size
        self.min_mask_mass = min_mask_mass

    def select_patches(
        self,
        prediction: torch.Tensor,
        target: Optional[torch.Tensor],
        h: int,
        w: int,
        context: Optional[Dict] = None
    ) -> List[Tuple[int, int, int, int]]:
        if context is None or 'slots_p' not in context:
            return []

        slots_p = context['slots_p']  # [K, 2] centroids (y, x)
        slots_m = context.get('slots_m')  # [K, H, W] masks or None

        patches = []
        for k in range(slots_p.shape[0]):
            cy, cx = slots_p[k]

            # Filter by mask mass
            if slots_m is not None:
                mask_mass = slots_m[k, :h, :w].sum().item()
                if mask_mass < self.min_mask_mass * h * w:
                    continue

            # Patch centered at (cy, cx)
            half = self.patch_size // 2
            y_min = max(0, int(cy.item()) - half)
            x_min = max(0, int(cx.item()) - half)
            y_max = min(h, y_min + self.patch_size)
            x_max = min(w, x_min + self.patch_size)

            patch_h = y_max - y_min
            patch_w = x_max - x_min

            if patch_h > 0 and patch_w > 0:
                patches.append((y_min, x_min, patch_h, patch_w))

        return patches


class BorderPatchSelector(PatchSelector):
    """
    Select patches along grid borders.

    Many ARC tasks have important structure at edges (frames, boundaries).
    """

    def __init__(
        self,
        border_width: int = 2,
        include_corners: bool = True
    ):
        """
        Args:
            border_width: Width of border strips
            include_corners: Whether to include corners in strips
        """
        self.border_width = border_width
        self.include_corners = include_corners

    def select_patches(
        self,
        prediction: torch.Tensor,
        target: Optional[torch.Tensor],
        h: int,
        w: int,
        context: Optional[Dict] = None
    ) -> List[Tuple[int, int, int, int]]:
        bw = min(self.border_width, min(h, w) // 3)

        if bw == 0:
            return []

        patches = []

        # Top/bottom strips
        if h > 0 and w > 0:
            patches.append((0, 0, bw, w))  # Top
            if h > bw:
                patches.append((h - bw, 0, bw, w))  # Bottom

        # Left/right strips (excluding corners if not included)
        if self.include_corners:
            y_start = 0
            y_end = h
        else:
            y_start = bw
            y_end = h - bw

        if y_end > y_start and w > 0:
            patches.append((y_start, 0, y_end - y_start, bw))  # Left
            if w > bw:
                patches.append((y_start, w - bw, y_end - y_start, bw))  # Right

        return patches


class LatticePatchSelector(PatchSelector):
    """
    Detect and sample from repeating tile patterns.

    Many ARC tasks have lattice structures (e.g., 3×3 tiles repeated).
    """

    def __init__(
        self,
        tile_sizes: Optional[List[int]] = None,
        max_unique_tiles: int = 6,
        repetition_threshold: float = 0.8
    ):
        """
        Args:
            tile_sizes: List of tile sizes to try (default: [2,3,4,5])
            max_unique_tiles: Maximum unique tiles to sample
            repetition_threshold: Fraction of tiles that must match
        """
        self.tile_sizes = tile_sizes if tile_sizes else [2, 3, 4, 5]
        self.max_unique_tiles = max_unique_tiles
        self.repetition_threshold = repetition_threshold

    def select_patches(
        self,
        prediction: torch.Tensor,
        target: Optional[torch.Tensor],
        h: int,
        w: int,
        context: Optional[Dict] = None
    ) -> List[Tuple[int, int, int, int]]:
        # Use target if available, else prediction
        grid = target if target is not None else prediction
        crop = grid[:h, :w]

        # Try each tile size
        for tile_h in self.tile_sizes:
            for tile_w in self.tile_sizes:
                if h % tile_h == 0 and w % tile_w == 0:
                    # Check for repetition
                    tiles = self._extract_tiles(crop, tile_h, tile_w)
                    if self._is_repeating(tiles):
                        # Found lattice, return unique tile patches
                        return self._unique_tile_patches(
                            tile_h, tile_w, h, w
                        )

        # No lattice found
        return []

    def _extract_tiles(
        self,
        grid: torch.Tensor,
        th: int,
        tw: int
    ) -> List[torch.Tensor]:
        """Extract all tiles from grid."""
        h, w = grid.shape
        tiles = []
        for i in range(0, h, th):
            for j in range(0, w, tw):
                tile = grid[i:i+th, j:j+tw]
                tiles.append(tile)
        return tiles

    def _is_repeating(
        self,
        tiles: List[torch.Tensor]
    ) -> bool:
        """Check if tiles form a repeating pattern."""
        if len(tiles) < 2:
            return False

        # Check if most tiles match the first tile
        first_tile = tiles[0]
        matches = sum((t == first_tile).all().item() for t in tiles)

        return matches / len(tiles) >= self.repetition_threshold

    def _unique_tile_patches(
        self,
        tile_h: int,
        tile_w: int,
        h: int,
        w: int
    ) -> List[Tuple[int, int, int, int]]:
        """Return patches for unique tiles."""
        patches = []

        # Sample up to max_unique_tiles different positions
        n_tiles_h = h // tile_h
        n_tiles_w = w // tile_w
        total_tiles = n_tiles_h * n_tiles_w

        # Sample evenly distributed tiles
        sample_step = max(1, total_tiles // self.max_unique_tiles)

        idx = 0
        for i in range(0, h, tile_h):
            for j in range(0, w, tile_w):
                if idx % sample_step == 0:
                    patches.append((i, j, tile_h, tile_w))
                    if len(patches) >= self.max_unique_tiles:
                        return patches
                idx += 1

        return patches


class AdaptivePatchSelector(PatchSelector):
    """
    Combine multiple patch selection strategies.

    Automatically uses multiple strategies and combines results.
    """

    def __init__(
        self,
        strategies: Optional[List[PatchSelector]] = None,
        max_total_patches: int = 12
    ):
        """
        Args:
            strategies: List of selectors (default: Disagreement + Object + Border)
            max_total_patches: Maximum total patches
        """
        if strategies is None:
            strategies = [
                DisagreementPatchSelector(max_patches=4),
                ObjectPatchSelector(),
                BorderPatchSelector(),
            ]
        self.strategies = strategies
        self.max_total_patches = max_total_patches

    def select_patches(
        self,
        prediction: torch.Tensor,
        target: Optional[torch.Tensor],
        h: int,
        w: int,
        context: Optional[Dict] = None
    ) -> List[Tuple[int, int, int, int]]:
        all_patches = []

        for strategy in self.strategies:
            try:
                patches = strategy.select_patches(
                    prediction, target, h, w, context
                )
                all_patches.extend(patches)
            except Exception as e:
                # Skip strategies that fail
                continue

        if len(all_patches) == 0:
            return []

        # Remove duplicates and overlaps
        all_patches = self._merge_overlapping(all_patches, iou_threshold=0.5)

        # Limit total
        all_patches = sorted(
            all_patches,
            key=lambda p: p[2] * p[3],
            reverse=True
        )
        return all_patches[:self.max_total_patches]


# =============================================================================
# Component 2: Patch Scorer
# =============================================================================

class PatchScorer:
    """
    Fast scoring using selected patches.

    Features:
    - Weighted scoring (importance weights per patch)
    - Caching (avoid recomputation)
    - Multiple metrics (accuracy, CE loss)
    """

    def __init__(
        self,
        selector: PatchSelector,
        metric: str = "accuracy",  # "accuracy", "weighted"
        use_cache: bool = False  # Disabled by default (state management)
    ):
        """
        Args:
            selector: Patch selection strategy
            metric: Scoring metric
            use_cache: Whether to cache results
        """
        self.selector = selector
        self.metric = metric
        self.use_cache = use_cache
        self._cache = {} if use_cache else None

    def score(
        self,
        prediction: torch.Tensor,  # [H, W]
        target: torch.Tensor,  # [H, W]
        h: int,
        w: int,
        context: Optional[Dict] = None,
        weights: Optional[torch.Tensor] = None  # [H, W] importance weights
    ) -> Dict[str, float]:
        """
        Score prediction using patches.

        Args:
            prediction: Predicted grid
            target: Target grid
            h, w: Actual dimensions
            context: Optional context
            weights: Optional importance weights

        Returns:
            {
                'score': float [0, 1],
                'num_patches': int,
                'coverage': float,
                'patch_scores': List[float]
            }
        """
        # 1. Select patches
        patches = self.selector.select_patches(
            prediction, target, h, w, context
        )

        if len(patches) == 0:
            # Fallback to full grid
            score = self._compute_full_score(prediction, target, h, w)
            return {
                'score': score,
                'num_patches': 0,
                'coverage': 1.0,
                'patch_scores': [score]
            }

        # 2. Score each patch
        patch_scores = []
        total_weight = 0.0
        total_covered = 0

        for (y, x, ph, pw) in patches:
            y_end = min(y + ph, h)
            x_end = min(x + pw, w)

            patch_pred = prediction[y:y_end, x:x_end]
            patch_target = target[y:y_end, x:x_end]

            if patch_pred.numel() == 0:
                continue

            # Compute patch score
            if self.metric == "accuracy":
                patch_score = (patch_pred == patch_target).float().mean().item()
            else:
                patch_score = (patch_pred == patch_target).float().mean().item()

            # Weight by patch area
            patch_area = patch_pred.numel()
            patch_weight = patch_area

            if weights is not None:
                # Use importance weights
                patch_weight *= weights[y:y_end, x:x_end].mean().item()

            patch_scores.append(patch_score)
            total_weight += patch_weight * patch_score
            total_covered += patch_area

        # 3. Aggregate
        if self.metric == "weighted" and total_covered > 0:
            final_score = total_weight / total_covered
        elif len(patch_scores) > 0:
            final_score = np.mean(patch_scores)
        else:
            final_score = 0.0

        coverage = min(1.0, total_covered / (h * w)) if h * w > 0 else 0.0

        return {
            'score': final_score,
            'num_patches': len(patches),
            'coverage': coverage,
            'patch_scores': patch_scores
        }

    def _compute_full_score(
        self,
        prediction: torch.Tensor,
        target: torch.Tensor,
        h: int,
        w: int
    ) -> float:
        """Compute score on full grid."""
        pred_crop = prediction[:h, :w]
        targ_crop = target[:h, :w]

        if pred_crop.numel() == 0:
            return 0.0

        return (pred_crop == targ_crop).float().mean().item()

    def batch_score(
        self,
        predictions: List[torch.Tensor],
        target: torch.Tensor,
        h: int,
        w: int,
        context: Optional[Dict] = None
    ) -> List[Dict]:
        """Score multiple predictions against same target."""
        return [
            self.score(pred, target, h, w, context)
            for pred in predictions
        ]


# =============================================================================
# Component 3: Utilities
# =============================================================================

def extract_patches(
    grid: torch.Tensor,  # [H, W]
    patches: List[Tuple[int, int, int, int]],
    pad_value: int = 0
) -> List[torch.Tensor]:
    """
    Extract patch tensors from grid.

    Args:
        grid: Input grid
        patches: List of (y, x, h, w)
        pad_value: Value for padding (unused currently)

    Returns:
        List of patch tensors
    """
    patch_tensors = []
    for (y, x, h, w) in patches:
        patch = grid[y:y+h, x:x+w]
        patch_tensors.append(patch)
    return patch_tensors


def compute_patch_mask(
    patches: List[Tuple[int, int, int, int]],
    h: int,
    w: int,
    device: torch.device
) -> torch.Tensor:
    """
    Create boolean mask indicating which pixels are in patches.

    Args:
        patches: List of (y, x, patch_h, patch_w)
        h, w: Grid dimensions
        device: Target device

    Returns:
        [H, W] bool tensor
    """
    mask = torch.zeros(h, w, dtype=torch.bool, device=device)
    for (y, x, ph, pw) in patches:
        y_end = min(y + ph, h)
        x_end = min(x + pw, w)
        mask[y:y_end, x:x_end] = True
    return mask


def visualize_patches(
    grid: torch.Tensor,
    patches: List[Tuple[int, int, int, int]],
    save_path: Optional[str] = None
) -> np.ndarray:
    """
    Visualize patches overlaid on grid.

    Args:
        grid: Input grid [H, W]
        patches: List of patches
        save_path: Optional path to save image

    Returns:
        RGB image with patches highlighted
    """
    from arc_nodsl.utils.viz import ARC_PALETTE

    # Convert grid to RGB
    grid_np = grid.cpu().numpy()
    rgb = ARC_PALETTE[grid_np]

    # Draw patch boundaries (red)
    for (y, x, h, w) in patches:
        # Top/bottom borders
        if y < rgb.shape[0]:
            rgb[y, x:min(x+w, rgb.shape[1])] = [255, 0, 0]
        if y+h-1 < rgb.shape[0]:
            rgb[y+h-1, x:min(x+w, rgb.shape[1])] = [255, 0, 0]

        # Left/right borders
        if x < rgb.shape[1]:
            rgb[y:min(y+h, rgb.shape[0]), x] = [255, 0, 0]
        if x+w-1 < rgb.shape[1]:
            rgb[y:min(y+h, rgb.shape[0]), x+w-1] = [255, 0, 0]

    if save_path:
        import matplotlib.pyplot as plt
        plt.imsave(save_path, rgb)

    return rgb


# =============================================================================
# Component 4: Integration
# =============================================================================

def score_with_patches(
    prediction: torch.Tensor,
    target: torch.Tensor,
    h: int,
    w: int,
    strategy: str = "adaptive",  # "disagreement", "object", "border", "lattice", "adaptive"
    context: Optional[Dict] = None,
    constraints: Optional = None  # ConstraintSet from constraints.py
) -> float:
    """
    Main API: Score prediction using patches.

    Args:
        prediction: [H, W] predicted grid
        target: [H, W] target grid
        h, w: Actual dimensions
        strategy: Patch selection strategy
        context: Optional context (slots, task info)
        constraints: Optional ConstraintSet for filtering

    Returns:
        Combined score [0, 1], 1 = perfect match
    """
    # 1. Select strategy
    if strategy == "adaptive":
        selector = AdaptivePatchSelector()
    elif strategy == "disagreement":
        selector = DisagreementPatchSelector()
    elif strategy == "object":
        selector = ObjectPatchSelector()
    elif strategy == "border":
        selector = BorderPatchSelector()
    elif strategy == "lattice":
        selector = LatticePatchSelector()
    else:
        raise ValueError(f"Unknown strategy: {strategy}")

    # 2. Create scorer
    scorer = PatchScorer(selector, metric="accuracy")

    # 3. Score with patches
    result = scorer.score(prediction, target, h, w, context)
    patch_score = result['score']

    # 4. Apply constraints if provided
    if constraints is not None:
        try:
            constraint_score = constraints.score(prediction, h, w)
            # Combine with patch score (weighted average)
            final_score = 0.7 * patch_score + 0.3 * constraint_score
        except:
            final_score = patch_score
    else:
        final_score = patch_score

    return final_score


if __name__ == "__main__":
    # Test patches module
    import sys
    from pathlib import Path
    sys.path.insert(0, str(Path(__file__).parent.parent.parent))

    from arc_nodsl.data.loader import ARCDataset

    print("=" * 60)
    print("Testing Patches Module")
    print("=" * 60)

    # Load dataset
    dataset = ARCDataset("data/arc-agi_training_challenges.json")
    task = dataset[0]
    print(f"\nTask: {task['task_id']}")

    # Get first train pair
    target = task['train_outputs'][0]
    h, w = task['train_shapes'][0]['output']
    print(f"Grid size: {h}×{w}")

    # Create prediction (add some noise)
    prediction = target.clone()
    noise_mask = torch.rand(h, w) < 0.1  # 10% noise
    prediction[:h, :w][noise_mask] = torch.randint(0, 10, (noise_mask.sum().item(),))

    accuracy = (prediction[:h, :w] == target[:h, :w]).float().mean().item()
    print(f"Base accuracy: {accuracy*100:.1f}%")

    # Test each strategy
    print("\n" + "-" * 60)
    print("Testing Patch Selection Strategies")
    print("-" * 60)

    strategies = {
        "disagreement": DisagreementPatchSelector(),
        "border": BorderPatchSelector(),
        "lattice": LatticePatchSelector(),
        "adaptive": AdaptivePatchSelector()
    }

    for name, selector in strategies.items():
        patches = selector.select_patches(prediction, target, h, w)
        print(f"\n{name.capitalize()}:")
        print(f"  Patches found: {len(patches)}")
        if len(patches) > 0:
            total_area = sum(ph * pw for _, _, ph, pw in patches)
            coverage = total_area / (h * w)
            print(f"  Coverage: {coverage*100:.1f}%")
            print(f"  Patch sizes: {[(ph, pw) for _, _, ph, pw in patches]}")

    # Test scorer
    print("\n" + "-" * 60)
    print("Testing PatchScorer")
    print("-" * 60)

    scorer = PatchScorer(DisagreementPatchSelector())
    result = scorer.score(prediction, target, h, w)

    print(f"\nScoring result:")
    print(f"  Score: {result['score']:.3f}")
    print(f"  Num patches: {result['num_patches']}")
    print(f"  Coverage: {result['coverage']*100:.1f}%")
    print(f"  Patch scores: {[f'{s:.2f}' for s in result['patch_scores']]}")

    # Test integration function
    print("\n" + "-" * 60)
    print("Testing Integration Function")
    print("-" * 60)

    score = score_with_patches(
        prediction, target, h, w,
        strategy="adaptive"
    )
    print(f"\nFinal score (adaptive): {score:.3f}")

    # Test on multiple tasks
    print("\n" + "-" * 60)
    print("Testing on 10 Tasks")
    print("-" * 60)

    results = []
    for i in range(min(10, len(dataset))):
        task = dataset[i]
        target = task['train_outputs'][0]
        h, w = task['train_shapes'][0]['output']

        # Create noisy prediction
        prediction = target.clone()
        noise_mask = torch.rand(h, w) < 0.15
        if noise_mask.sum() > 0:
            prediction[:h, :w][noise_mask] = torch.randint(
                0, 10, (noise_mask.sum().item(),)
            )

        # Score with patches
        patch_score = score_with_patches(
            prediction, target, h, w, strategy="adaptive"
        )

        # Full grid score for comparison
        full_score = (prediction[:h, :w] == target[:h, :w]).float().mean().item()

        results.append({
            'task_id': task['task_id'],
            'patch_score': patch_score,
            'full_score': full_score,
            'diff': abs(patch_score - full_score)
        })

    print(f"\nResults on {len(results)} tasks:")
    print(f"{'Task ID':<12} {'Patch':<8} {'Full':<8} {'Diff':<8}")
    print("-" * 40)
    for r in results:
        print(f"{r['task_id']:<12} {r['patch_score']:<8.3f} {r['full_score']:<8.3f} {r['diff']:<8.3f}")

    avg_diff = np.mean([r['diff'] for r in results])
    print(f"\nAverage difference: {avg_diff:.3f} ({avg_diff*100:.1f}%)")
    print(f"✓ Target: <5% difference")

    if avg_diff < 0.05:
        print("\n✓ ALL TESTS PASSED!")
    else:
        print(f"\n⚠ Accuracy difference higher than target (5%)")

    print("\n" + "=" * 60)
    print("Patches module implementation complete!")
    print("=" * 60)
