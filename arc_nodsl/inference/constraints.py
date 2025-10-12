"""
Constraints extraction and checking for ARC tasks.

Extract hard and soft constraints from training pairs to guide
search and filter invalid candidates.

Types of constraints:
1. Palette: Allowed input/output colors
2. Grid Size: Output dimensions from input dimensions
3. Symmetry: Axes of symmetry
4. Object Count: Number of objects preserved/changed
"""

import torch
import numpy as np
from typing import List, Tuple, Dict, Set, Optional, Union
from dataclasses import dataclass
from collections import Counter
import scipy.ndimage as ndimage


@dataclass
class PaletteInfo:
    """Color palette information from train pairs."""
    input_colors: Set[int]
    output_colors: Set[int]
    color_mappings: Dict[int, Set[int]]  # input_color -> possible output_colors
    is_strict_mapping: bool  # True if 1-to-1 mapping


class PaletteConstraint:
    """
    Constraint on allowed colors in input/output.
    """

    def __init__(self, train_pairs: List[Tuple[torch.Tensor, torch.Tensor, Dict]]):
        """
        Args:
            train_pairs: List of (input, output, shapes) tuples
        """
        self.input_colors_all = set()
        self.output_colors_all = set()
        self.pair_mappings = []

        # Analyze each pair
        for input_grid, output_grid, shapes in train_pairs:
            h_in, w_in = shapes["input"]
            h_out, w_out = shapes["output"]

            # Get colors (crop to actual size)
            inp_colors = set(input_grid[:h_in, :w_in].unique().cpu().numpy())
            out_colors = set(output_grid[:h_out, :w_out].unique().cpu().numpy())

            self.input_colors_all.update(inp_colors)
            self.output_colors_all.update(out_colors)

            # Track per-pair mapping
            mapping = {c: set() for c in inp_colors}
            # Simple co-occurrence mapping
            for c in inp_colors:
                if c in out_colors:
                    mapping[c].add(c)
            self.pair_mappings.append(mapping)

        # Determine if strict 1-to-1 mapping exists
        self.is_strict = self._check_strict_mapping()

    def _check_strict_mapping(self) -> bool:
        """Check if there's a consistent 1-to-1 color mapping."""
        if len(self.pair_mappings) == 0:
            return False

        # Check if all pairs agree on mappings
        # For now, simple heuristic: if input/output colors are same, assume strict
        return self.input_colors_all == self.output_colors_all

    def is_valid_input(self, grid: torch.Tensor, h: int, w: int) -> bool:
        """Check if input uses only seen colors."""
        h, w = int(h), int(w)
        colors = set(grid[:h, :w].unique().cpu().numpy())
        # Allow superset (extra colors okay for input)
        return True  # Inputs can have new colors

    def is_valid_output(self, grid: torch.Tensor, h: int, w: int) -> bool:
        """Check if output uses only allowed colors."""
        h, w = int(h), int(w)
        colors = set(grid[:h, :w].unique().cpu().numpy())
        # Output must be subset of seen colors
        return colors.issubset(self.output_colors_all)

    def score_output(self, grid: torch.Tensor, h: int, w: int) -> float:
        """
        Soft score: fraction of pixels with valid colors.

        Returns: [0, 1], 1 = all valid
        """
        # Ensure h and w are Python ints (not tensors or numpy ints)
        h = int(h)
        w = int(w)
        crop = grid[:h, :w]
        valid_mask = torch.zeros_like(crop, dtype=torch.bool)

        for c in self.output_colors_all:
            valid_mask |= (crop == c)

        return valid_mask.float().mean().item()


@dataclass
class GridSizeInfo:
    """Grid size transformation pattern."""
    rule: str  # "preserve", "double", "tile_nxm", "custom"
    ratio: Optional[Tuple[float, float]]  # (h_ratio, w_ratio)
    tile_size: Optional[Tuple[int, int]]  # For tiling rules


class GridSizeConstraint:
    """
    Constraint on output grid dimensions.
    """

    def __init__(self, train_pairs: List[Tuple[torch.Tensor, torch.Tensor, Dict]]):
        self.pairs = []

        for input_grid, output_grid, shapes in train_pairs:
            h_in, w_in = shapes["input"]
            h_out, w_out = shapes["output"]
            self.pairs.append(((h_in, w_in), (h_out, w_out)))

        # Detect pattern
        self.info = self._detect_pattern()

    def _detect_pattern(self) -> GridSizeInfo:
        """Detect the size transformation rule."""
        if len(self.pairs) == 0:
            return GridSizeInfo("unknown", None, None)

        # Check if all pairs preserve size
        if all(inp == out for inp, out in self.pairs):
            return GridSizeInfo("preserve", (1.0, 1.0), None)

        # Check for constant ratio
        ratios = [(out[0] / inp[0], out[1] / inp[1]) for inp, out in self.pairs if inp[0] > 0 and inp[1] > 0]
        if ratios:
            avg_ratio = (np.mean([r[0] for r in ratios]), np.mean([r[1] for r in ratios]))
            std_ratio = (np.std([r[0] for r in ratios]), np.std([r[1] for r in ratios]))

            # If ratio is consistent
            if std_ratio[0] < 0.1 and std_ratio[1] < 0.1:
                # Check for common patterns
                if abs(avg_ratio[0] - 2.0) < 0.1 and abs(avg_ratio[1] - 2.0) < 0.1:
                    return GridSizeInfo("double", (2.0, 2.0), None)
                elif abs(avg_ratio[0] - 3.0) < 0.1 and abs(avg_ratio[1] - 3.0) < 0.1:
                    return GridSizeInfo("triple", (3.0, 3.0), None)
                else:
                    return GridSizeInfo("scale", avg_ratio, None)

        return GridSizeInfo("custom", None, None)

    def get_expected_size(self, input_shape: Tuple[int, int]) -> Optional[Tuple[int, int]]:
        """Predict output size from input size."""
        if self.info.rule == "preserve":
            return input_shape
        elif self.info.rule in ["double", "triple", "scale"] and self.info.ratio:
            h_out = int(input_shape[0] * self.info.ratio[0])
            w_out = int(input_shape[1] * self.info.ratio[1])
            return (h_out, w_out)
        else:
            # Unknown pattern, can't predict
            return None

    def is_valid_size(self, output_shape: Tuple[int, int], input_shape: Tuple[int, int]) -> bool:
        """Check if output size is valid given input size."""
        expected = self.get_expected_size(input_shape)
        if expected is None:
            return True  # Can't check, assume valid

        # Allow small tolerance
        return abs(output_shape[0] - expected[0]) <= 1 and abs(output_shape[1] - expected[1]) <= 1


class SymmetryConstraint:
    """
    Constraint on symmetry axes.
    """

    def __init__(self, train_pairs: List[Tuple[torch.Tensor, torch.Tensor, Dict]]):
        self.detected_axes = []

        # Check each pair for symmetries
        for input_grid, output_grid, shapes in train_pairs:
            h_out, w_out = shapes["output"]
            crop = output_grid[:h_out, :w_out]

            # Check vertical symmetry (left-right)
            if self._check_vertical_symmetry(crop):
                self.detected_axes.append(("vertical", w_out // 2))

            # Check horizontal symmetry (top-bottom)
            if self._check_horizontal_symmetry(crop):
                self.detected_axes.append(("horizontal", h_out // 2))

        # Keep only axes that appear in all pairs
        if self.detected_axes:
            axis_counts = Counter([ax[0] for ax in self.detected_axes])
            n_pairs = len(train_pairs)
            self.consistent_axes = [ax for ax, count in axis_counts.items() if count == n_pairs]
        else:
            self.consistent_axes = []

    def _check_vertical_symmetry(self, grid: torch.Tensor, threshold: float = 0.95) -> bool:
        """Check if grid is symmetric around vertical axis."""
        h, w = grid.shape
        if w < 2:
            return False

        left = grid[:, :w//2]
        right = grid[:, w//2:w//2+left.shape[1]].flip(1)

        if left.shape != right.shape:
            return False

        match_rate = (left == right).float().mean().item()
        return match_rate >= threshold

    def _check_horizontal_symmetry(self, grid: torch.Tensor, threshold: float = 0.95) -> bool:
        """Check if grid is symmetric around horizontal axis."""
        h, w = grid.shape
        if h < 2:
            return False

        top = grid[:h//2, :]
        bottom = grid[h//2:h//2+top.shape[0], :].flip(0)

        if top.shape != bottom.shape:
            return False

        match_rate = (top == bottom).float().mean().item()
        return match_rate >= threshold

    def check_symmetry(self, grid: torch.Tensor, h: int, w: int) -> float:
        """
        Score how well grid satisfies detected symmetries.

        Returns: [0, 1], 1 = perfectly symmetric
        """
        h, w = int(h), int(w)
        if not self.consistent_axes:
            return 1.0  # No symmetry constraint

        crop = grid[:h, :w]
        scores = []

        if "vertical" in self.consistent_axes:
            left = crop[:, :w//2]
            right = crop[:, w//2:w//2+left.shape[1]].flip(1)
            if left.shape == right.shape:
                scores.append((left == right).float().mean().item())

        if "horizontal" in self.consistent_axes:
            top = crop[:h//2, :]
            bottom = crop[h//2:h//2+top.shape[0], :].flip(0)
            if top.shape == bottom.shape:
                scores.append((top == bottom).float().mean().item())

        return np.mean(scores) if scores else 1.0


class ObjectCountConstraint:
    """
    Constraint on number of objects (connected components).
    """

    def __init__(self, train_pairs: List[Tuple[torch.Tensor, torch.Tensor, Dict]]):
        self.input_counts = []
        self.output_counts = []

        for input_grid, output_grid, shapes in train_pairs:
            h_in, w_in = shapes["input"]
            h_out, w_out = shapes["output"]

            n_in = self._count_objects(input_grid[:h_in, :w_in])
            n_out = self._count_objects(output_grid[:h_out, :w_out])

            self.input_counts.append(n_in)
            self.output_counts.append(n_out)

        # Detect pattern
        self.rule = self._detect_rule()

    def _count_objects(self, grid: torch.Tensor) -> int:
        """Count connected components (objects) in grid."""
        # Treat background (0) as not an object
        grid_np = grid.cpu().numpy()
        binary = (grid_np > 0).astype(int)

        # Connected components
        labeled, n_objects = ndimage.label(binary)
        return n_objects

    def _detect_rule(self) -> str:
        """Detect object count transformation rule."""
        if not self.input_counts:
            return "unknown"

        # Check if count is preserved
        if all(inp == out for inp, out in zip(self.input_counts, self.output_counts)):
            return "preserve"

        # Check if output is always same count
        if len(set(self.output_counts)) == 1:
            return f"constant_{self.output_counts[0]}"

        # Check for ratio
        ratios = [out / inp if inp > 0 else 0 for inp, out in zip(self.input_counts, self.output_counts)]
        if len(set(ratios)) == 1:
            return f"ratio_{ratios[0]:.1f}"

        return "custom"

    def get_expected_count(self, input_count: int) -> Optional[int]:
        """Predict output object count from input count."""
        if self.rule == "preserve":
            return input_count
        elif self.rule.startswith("constant_"):
            return int(self.rule.split("_")[1])
        elif self.rule.startswith("ratio_"):
            ratio = float(self.rule.split("_")[1])
            return int(input_count * ratio)
        else:
            return None

    def check_count(self, grid: torch.Tensor, h: int, w: int, input_count: int) -> bool:
        """Check if grid has expected object count."""
        h, w = int(h), int(w)
        actual_count = self._count_objects(grid[:h, :w])
        expected_count = self.get_expected_count(input_count)

        if expected_count is None:
            return True  # Can't check

        return actual_count == expected_count


class ConstraintSet:
    """
    Collection of constraints with filtering and scoring.
    """

    def __init__(self, constraints: Optional[List] = None):
        self.constraints = constraints or []

        # Categorize constraints
        self.palette = None
        self.grid_size = None
        self.symmetry = None
        self.object_count = None

        for c in self.constraints:
            if isinstance(c, PaletteConstraint):
                self.palette = c
            elif isinstance(c, GridSizeConstraint):
                self.grid_size = c
            elif isinstance(c, SymmetryConstraint):
                self.symmetry = c
            elif isinstance(c, ObjectCountConstraint):
                self.object_count = c

    def is_valid(self, grid: torch.Tensor, h: int, w: int, input_shape: Optional[Tuple[int, int]] = None) -> bool:
        """
        Hard constraint check: is this grid valid?

        Args:
            grid: [H, W] prediction
            h, w: Actual dimensions
            input_shape: Optional input dimensions for size constraint

        Returns: True if all hard constraints satisfied
        """
        h, w = int(h), int(w)

        # Palette constraint (hard)
        if self.palette and not self.palette.is_valid_output(grid, h, w):
            return False

        # Grid size constraint (hard)
        if self.grid_size and input_shape:
            if not self.grid_size.is_valid_size((h, w), input_shape):
                return False

        return True

    def score(self, grid: torch.Tensor, h: int, w: int, input_count: Optional[int] = None) -> float:
        """
        Soft constraint score: how well does grid satisfy constraints?

        Returns: [0, 1], 1 = perfect
        """
        # Convert h and w to Python ints robustly
        if isinstance(h, torch.Tensor):
            if h.numel() != 1:
                print(f"WARNING: h is tensor with shape {h.shape}, value {h}")
                print(f"  Type: {type(h)}, device: {h.device if isinstance(h, torch.Tensor) else 'N/A'}")
                h = int(h[0]) if h.numel() > 0 else int(h)
            else:
                h = int(h.item())
        else:
            h = int(h)

        if isinstance(w, torch.Tensor):
            if w.numel() != 1:
                print(f"WARNING: w is tensor with shape {w.shape}, value {w}")
                print(f"  Type: {type(w)}, device: {w.device if isinstance(w, torch.Tensor) else 'N/A'}")
                w = int(w[0]) if w.numel() > 0 else int(w)
            else:
                w = int(w.item())
        else:
            w = int(w)
        scores = []

        # Palette score
        if self.palette:
            scores.append(self.palette.score_output(grid, h, w))

        # Symmetry score
        if self.symmetry:
            scores.append(self.symmetry.check_symmetry(grid, h, w))

        # Object count (binary)
        if self.object_count and input_count is not None:
            scores.append(1.0 if self.object_count.check_count(grid, h, w, input_count) else 0.0)

        return np.mean(scores) if scores else 1.0

    def filter_candidates(self, candidates: List[Dict]) -> List[Dict]:
        """
        Filter candidates by hard constraints.

        Args:
            candidates: List of dicts with 'grid', 'h', 'w', 'input_shape'

        Returns: Filtered list
        """
        valid = []
        for cand in candidates:
            if self.is_valid(
                cand['grid'],
                cand.get('h', cand['grid'].shape[0]),
                cand.get('w', cand['grid'].shape[1]),
                cand.get('input_shape')
            ):
                valid.append(cand)
        return valid


def extract_constraints(
    train_pairs: List[Tuple[torch.Tensor, torch.Tensor, Dict]],
    enable_all: bool = True
) -> ConstraintSet:
    """
    Extract all constraints from training pairs.

    Args:
        train_pairs: List of (input, output, shapes) tuples
        enable_all: If True, extract all constraint types

    Returns:
        ConstraintSet with detected constraints
    """
    constraints = []

    if enable_all:
        # Palette constraint (always useful)
        constraints.append(PaletteConstraint(train_pairs))

        # Grid size constraint
        constraints.append(GridSizeConstraint(train_pairs))

        # Symmetry constraint
        constraints.append(SymmetryConstraint(train_pairs))

        # Object count constraint
        constraints.append(ObjectCountConstraint(train_pairs))

    return ConstraintSet(constraints)


if __name__ == "__main__":
    # Test constraints
    import sys
    from pathlib import Path
    sys.path.insert(0, str(Path(__file__).parent.parent.parent))

    from arc_nodsl.data.loader import ARCDataset

    print("Testing Constraints...")

    # Load dataset
    dataset = ARCDataset("data/arc-agi_training_challenges.json")

    # Test on first task
    task = dataset[0]
    print(f"\nTask: {task['task_id']}")

    # Prepare train pairs
    train_pairs = []
    for i in range(len(task['train_inputs'])):
        train_pairs.append((
            task['train_inputs'][i],
            task['train_outputs'][i],
            task['train_shapes'][i]
        ))

    # Extract constraints
    constraints = extract_constraints(train_pairs)
    print(f"Extracted {len(constraints.constraints)} constraints")

    # Test palette
    if constraints.palette:
        print(f"\nPalette:")
        print(f"  Input colors: {sorted(constraints.palette.input_colors_all)}")
        print(f"  Output colors: {sorted(constraints.palette.output_colors_all)}")
        print(f"  Strict mapping: {constraints.palette.is_strict}")

    # Test grid size
    if constraints.grid_size:
        print(f"\nGrid Size:")
        print(f"  Rule: {constraints.grid_size.info.rule}")
        if constraints.grid_size.info.ratio:
            print(f"  Ratio: {constraints.grid_size.info.ratio}")

    # Test symmetry
    if constraints.symmetry:
        print(f"\nSymmetry:")
        print(f"  Consistent axes: {constraints.symmetry.consistent_axes}")

    # Test object count
    if constraints.object_count:
        print(f"\nObject Count:")
        print(f"  Rule: {constraints.object_count.rule}")
        print(f"  Input counts: {constraints.object_count.input_counts}")
        print(f"  Output counts: {constraints.object_count.output_counts}")

    # Test validation on actual output
    test_output = task['train_outputs'][0]
    h, w = task['train_shapes'][0]["output"]
    is_valid = constraints.is_valid(test_output, h, w)
    score = constraints.score(test_output, h, w)
    print(f"\nValidation on train output:")
    print(f"  Valid: {is_valid}")
    print(f"  Score: {score:.3f}")

    print("\n✓ Constraint tests passed!")
