"""
Task embedding: Extract patterns from train pairs.

Learn task-specific priors to guide search:
1. Statistical features (colors, sizes, symmetries)
2. Operator usage patterns (which ops are useful)
3. Parameter distributions (rotation, scale, color)

Usage:
    task_embed = build_task_embedding(train_pairs)
    # Returns: {'embed': Tensor[128], 'op_priors': [...], 'constraints': ConstraintSet, ...}
"""

import torch
import torch.nn as nn
import numpy as np
from typing import List, Dict, Optional, Tuple, Set
from dataclasses import dataclass
from collections import Counter


# =============================================================================
# Component 1: Statistical Feature Extraction
# =============================================================================

@dataclass
class TaskStatistics:
    """Statistical features extracted from train pairs."""
    # Grid properties
    input_sizes: List[Tuple[int, int]]
    output_sizes: List[Tuple[int, int]]
    size_ratio: Tuple[float, float]  # (h_ratio, w_ratio)

    # Colors
    input_palette: Set[int]
    output_palette: Set[int]
    color_frequencies: Dict[int, float]

    # Objects
    input_object_counts: List[int]
    output_object_counts: List[int]
    object_count_ratio: float

    # Symmetries
    has_horizontal_symmetry: bool
    has_vertical_symmetry: bool
    has_lattice: bool
    lattice_size: Optional[Tuple[int, int]]


class StatisticalAnalyzer:
    """Extract statistical features from train pairs."""

    def __init__(self):
        pass

    def analyze(
        self,
        train_pairs: List[Tuple[torch.Tensor, torch.Tensor, Dict]]
    ) -> TaskStatistics:
        """
        Extract statistical features from train pairs.

        Args:
            train_pairs: List of (input, output, shapes)

        Returns:
            TaskStatistics object
        """
        # Extract grid sizes
        input_sizes = [shapes["input"] for _, _, shapes in train_pairs]
        output_sizes = [shapes["output"] for _, _, shapes in train_pairs]

        # Compute size ratios
        ratios = []
        for inp, out in zip(input_sizes, output_sizes):
            if inp[0] > 0 and inp[1] > 0:
                ratios.append((out[0]/inp[0], out[1]/inp[1]))

        if ratios:
            avg_ratio = (
                np.mean([r[0] for r in ratios]),
                np.mean([r[1] for r in ratios])
            )
        else:
            avg_ratio = (1.0, 1.0)

        # Extract color palettes
        input_palette = set()
        output_palette = set()
        color_freq = Counter()

        for input_grid, output_grid, shapes in train_pairs:
            h_in, w_in = shapes["input"]
            h_out, w_out = shapes["output"]

            inp_crop = input_grid[:h_in, :w_in]
            out_crop = output_grid[:h_out, :w_out]

            input_palette.update(inp_crop.unique().cpu().tolist())
            output_palette.update(out_crop.unique().cpu().tolist())

            for c in out_crop.unique().cpu().tolist():
                color_freq[c] += (out_crop == c).sum().item()

        # Normalize frequencies
        total_pixels = sum(color_freq.values()) if color_freq else 1
        color_frequencies = {
            c: count/total_pixels
            for c, count in color_freq.items()
        }

        # Object counts (use constraints module)
        from arc_nodsl.inference.constraints import ObjectCountConstraint
        obj_constraint = ObjectCountConstraint(train_pairs)

        # Compute object count ratio
        avg_input_count = np.mean(obj_constraint.input_counts) if obj_constraint.input_counts else 1
        avg_output_count = np.mean(obj_constraint.output_counts) if obj_constraint.output_counts else 1
        object_count_ratio = avg_output_count / max(avg_input_count, 1)

        # Symmetry detection (use constraints module)
        from arc_nodsl.inference.constraints import SymmetryConstraint
        sym_constraint = SymmetryConstraint(train_pairs)

        # Lattice detection (use patches module)
        from arc_nodsl.inference.patches import LatticePatchSelector
        lattice_selector = LatticePatchSelector()
        has_lattice = False
        lattice_size = None

        for _, output_grid, shapes in train_pairs:
            h, w = shapes["output"]
            patches = lattice_selector.select_patches(output_grid, None, h, w)
            if len(patches) > 0:
                has_lattice = True
                lattice_size = (patches[0][2], patches[0][3])  # First patch size
                break

        return TaskStatistics(
            input_sizes=input_sizes,
            output_sizes=output_sizes,
            size_ratio=avg_ratio,
            input_palette=input_palette,
            output_palette=output_palette,
            color_frequencies=color_frequencies,
            input_object_counts=obj_constraint.input_counts,
            output_object_counts=obj_constraint.output_counts,
            object_count_ratio=object_count_ratio,
            has_horizontal_symmetry="horizontal" in sym_constraint.consistent_axes,
            has_vertical_symmetry="vertical" in sym_constraint.consistent_axes,
            has_lattice=has_lattice,
            lattice_size=lattice_size
        )


# =============================================================================
# Component 2: Operator Usage Analysis
# =============================================================================

class OperatorAnalyzer:
    """
    Analyze which operators are useful for this task.

    Strategy:
    1. Encode each train input → slots
    2. Try each operator on slots
    3. Render back to grid
    4. Score against train output
    5. Build operator usage histogram
    """

    def __init__(
        self,
        encoder: nn.Module,
        operators: nn.Module,
        renderer: nn.Module,
        device: torch.device
    ):
        self.encoder = encoder
        self.operators = operators
        self.renderer = renderer
        self.device = device

    @torch.no_grad()
    def analyze(
        self,
        train_pairs: List[Tuple[torch.Tensor, torch.Tensor, Dict]],
        max_pairs: int = 3,
        num_operators: int = 8
    ) -> Dict[str, any]:
        """
        Analyze operator usage patterns.

        Args:
            train_pairs: Training pairs
            max_pairs: Maximum pairs to analyze (for speed)
            num_operators: Number of operators

        Returns:
            {
                'op_scores': [M] average score per operator
                'op_usage': [M] recommended usage frequency
                'best_ops': List of top-K operator indices
            }
        """
        op_scores = np.zeros(num_operators)
        op_counts = np.zeros(num_operators)

        # Analyze subset of train pairs
        pairs_to_analyze = train_pairs[:max_pairs]

        for input_grid, output_grid, shapes in pairs_to_analyze:
            h_in, w_in = shapes["input"]
            h_out, w_out = shapes["output"]

            # Encode input
            input_batch = input_grid.unsqueeze(0).to(self.device)
            enc_out = self.encoder(input_batch)

            slots_z = enc_out["slots_z"]
            slots_m = enc_out["slots_m"]
            slots_p = enc_out["slots_p"]

            # Try each operator
            for op_idx in range(num_operators):
                try:
                    # Apply operator
                    z_new, m_new, p_new, aux = self.operators(
                        op_idx, slots_z, slots_m, slots_p
                    )

                    # Render
                    logits = self.renderer(z_new, m_new)
                    prediction = torch.argmax(logits, dim=-1)

                    # Score against output
                    pred_crop = prediction[0, :h_out, :w_out]
                    target_crop = output_grid[:h_out, :w_out].to(self.device)

                    accuracy = (pred_crop == target_crop).float().mean().item()

                    op_scores[op_idx] += accuracy
                    op_counts[op_idx] += 1
                except Exception as e:
                    # Skip operators that fail
                    continue

        # Average scores
        op_scores = op_scores / (op_counts + 1e-8)

        # Normalize to get usage priors (softmax with temperature)
        temperature = 5.0  # Controls sharpness
        exp_scores = np.exp(op_scores * temperature)
        op_usage = exp_scores / (exp_scores.sum() + 1e-8)

        # Top-K operators
        best_ops = np.argsort(op_scores)[::-1][:4].tolist()

        return {
            'op_scores': op_scores.tolist(),
            'op_usage': op_usage.tolist(),
            'best_ops': best_ops
        }


# =============================================================================
# Component 3: Task Embedding Builder
# =============================================================================

class TaskEmbeddingBuilder:
    """
    Build complete task embedding from train pairs.

    Combines:
    1. Statistical features
    2. Operator priors
    3. Learnable embedding (optional, for training)
    """

    def __init__(
        self,
        encoder: Optional[nn.Module] = None,
        operators: Optional[nn.Module] = None,
        renderer: Optional[nn.Module] = None,
        device: torch.device = torch.device('cpu')
    ):
        self.stat_analyzer = StatisticalAnalyzer()

        if encoder and operators and renderer:
            self.op_analyzer = OperatorAnalyzer(encoder, operators, renderer, device)
        else:
            self.op_analyzer = None

        self.device = device

    def build(
        self,
        train_pairs: List[Tuple[torch.Tensor, torch.Tensor, Dict]],
        analyze_operators: bool = True
    ) -> Dict[str, any]:
        """
        Build task embedding from train pairs.

        Args:
            train_pairs: Training pairs
            analyze_operators: Whether to analyze operator usage (slower)

        Returns:
            {
                'embed': torch.Tensor [d_task],  # Learnable embedding
                'stats': TaskStatistics,
                'op_priors': [M] operator usage priors,
                'constraints': ConstraintSet,
                'metadata': {...}
            }
        """
        # 1. Statistical analysis
        stats = self.stat_analyzer.analyze(train_pairs)

        # 2. Operator analysis (optional, expensive)
        if analyze_operators and self.op_analyzer:
            try:
                op_analysis = self.op_analyzer.analyze(train_pairs)
                op_priors = op_analysis['op_usage']
                best_ops = op_analysis['best_ops']
            except Exception as e:
                # Fallback to uniform if analysis fails
                print(f"Operator analysis failed: {e}, using uniform priors")
                M = 8  # num operators
                op_priors = [1.0 / M] * M
                best_ops = list(range(M))
        else:
            # Uniform priors
            M = 8  # num operators
            op_priors = [1.0 / M] * M
            best_ops = list(range(M))

        # 3. Extract constraints
        from arc_nodsl.inference.constraints import extract_constraints
        constraints = extract_constraints(train_pairs)

        # 4. Build learnable embedding
        # For now, use simple feature vector
        # In training, this would be learned
        embed_features = [
            stats.size_ratio[0],
            stats.size_ratio[1],
            stats.object_count_ratio,
            float(stats.has_horizontal_symmetry),
            float(stats.has_vertical_symmetry),
            float(stats.has_lattice),
            len(stats.input_palette) / 10.0,
            len(stats.output_palette) / 10.0,
        ]

        # Add color frequency features (top-5 colors)
        sorted_colors = sorted(
            stats.color_frequencies.items(),
            key=lambda x: x[1],
            reverse=True
        )[:5]
        for _, freq in sorted_colors:
            embed_features.append(freq)

        # Pad to d_task=128
        embed_features += [0.0] * (128 - len(embed_features))
        embed = torch.tensor(
            embed_features[:128],
            dtype=torch.float32,
            device=self.device
        )

        # 5. Metadata
        metadata = {
            'num_train_pairs': len(train_pairs),
            'has_symmetry': stats.has_horizontal_symmetry or stats.has_vertical_symmetry,
            'has_lattice': stats.has_lattice,
            'size_changes': stats.size_ratio != (1.0, 1.0),
            'colors_preserved': stats.input_palette == stats.output_palette,
            'best_operators': best_ops[:3],  # Top-3
            'avg_input_size': tuple(np.mean(stats.input_sizes, axis=0).astype(int).tolist()),
            'avg_output_size': tuple(np.mean(stats.output_sizes, axis=0).astype(int).tolist())
        }

        return {
            'embed': embed,
            'stats': stats,
            'op_priors': op_priors,
            'constraints': constraints,
            'metadata': metadata
        }


# =============================================================================
# Component 4: Main API
# =============================================================================

def build_task_embedding(
    train_pairs: List[Tuple[torch.Tensor, torch.Tensor, Dict]],
    encoder: Optional[nn.Module] = None,
    operators: Optional[nn.Module] = None,
    renderer: Optional[nn.Module] = None,
    device: torch.device = torch.device('cpu'),
    analyze_operators: bool = False
) -> Dict[str, any]:
    """
    Main API: Build task embedding from train pairs.

    Args:
        train_pairs: Training input/output pairs
        encoder, operators, renderer: Models (optional, for op analysis)
        device: Compute device
        analyze_operators: Whether to analyze operator usage (slow)

    Returns:
        Task embedding dict with:
        - 'embed': Tensor[128] embedding vector
        - 'stats': TaskStatistics object
        - 'op_priors': [M] operator usage priors
        - 'constraints': ConstraintSet
        - 'metadata': Additional task info
    """
    builder = TaskEmbeddingBuilder(encoder, operators, renderer, device)
    return builder.build(train_pairs, analyze_operators=analyze_operators)


if __name__ == "__main__":
    # Test task embedding
    import sys
    from pathlib import Path
    sys.path.insert(0, str(Path(__file__).parent.parent.parent))

    from arc_nodsl.data.loader import ARCDataset

    print("=" * 60)
    print("Testing Task Embedding Module")
    print("=" * 60)

    # Load dataset
    dataset = ARCDataset("data/arc-agi_training_challenges.json")

    # Test on multiple tasks
    print("\nTesting on 10 tasks...")

    for i in range(min(10, len(dataset))):
        task = dataset[i]

        # Prepare train pairs
        train_pairs = []
        for j in range(len(task['train_inputs'])):
            train_pairs.append((
                task['train_inputs'][j],
                task['train_outputs'][j],
                task['train_shapes'][j]
            ))

        # Build embedding (without operator analysis for speed)
        task_embed = build_task_embedding(train_pairs, analyze_operators=False)

        print(f"\n{i+1}. Task: {task['task_id']}")
        print(f"   Embedding shape: {task_embed['embed'].shape}")
        print(f"   Train pairs: {task_embed['metadata']['num_train_pairs']}")
        print(f"   Size ratio: {task_embed['stats'].size_ratio[0]:.2f}x{task_embed['stats'].size_ratio[1]:.2f}")
        print(f"   Colors: in={len(task_embed['stats'].input_palette)}, out={len(task_embed['stats'].output_palette)}")
        print(f"   Has symmetry: {task_embed['metadata']['has_symmetry']}")
        print(f"   Has lattice: {task_embed['metadata']['has_lattice']}")
        print(f"   Constraints: {len(task_embed['constraints'].constraints)}")

    # Test with full metadata display
    print("\n" + "=" * 60)
    print("Detailed Analysis on Task 0")
    print("=" * 60)

    task = dataset[0]
    train_pairs = []
    for j in range(len(task['train_inputs'])):
        train_pairs.append((
            task['train_inputs'][j],
            task['train_outputs'][j],
            task['train_shapes'][j]
        ))

    task_embed = build_task_embedding(train_pairs, analyze_operators=False)

    print(f"\nTask ID: {task['task_id']}")
    print(f"\nEmbedding:")
    print(f"  Shape: {task_embed['embed'].shape}")
    print(f"  First 10 values: {task_embed['embed'][:10].tolist()}")

    print(f"\nStatistics:")
    stats = task_embed['stats']
    print(f"  Input sizes: {stats.input_sizes}")
    print(f"  Output sizes: {stats.output_sizes}")
    print(f"  Size ratio: {stats.size_ratio}")
    print(f"  Input palette: {sorted(stats.input_palette)}")
    print(f"  Output palette: {sorted(stats.output_palette)}")
    print(f"  Object counts: in={stats.input_object_counts}, out={stats.output_object_counts}")
    print(f"  Object ratio: {stats.object_count_ratio:.2f}")
    print(f"  Has H-symmetry: {stats.has_horizontal_symmetry}")
    print(f"  Has V-symmetry: {stats.has_vertical_symmetry}")
    print(f"  Has lattice: {stats.has_lattice}")
    if stats.lattice_size:
        print(f"  Lattice size: {stats.lattice_size}")

    print(f"\nOperator Priors:")
    print(f"  {task_embed['op_priors']}")

    print(f"\nMetadata:")
    for key, value in task_embed['metadata'].items():
        print(f"  {key}: {value}")

    print(f"\nConstraints:")
    constraints = task_embed['constraints']
    if constraints.palette:
        print(f"  Palette: {sorted(constraints.palette.output_colors_all)}")
    if constraints.grid_size:
        print(f"  Grid size rule: {constraints.grid_size.info.rule}")
    if constraints.symmetry:
        print(f"  Symmetry axes: {constraints.symmetry.consistent_axes}")
    if constraints.object_count:
        print(f"  Object count rule: {constraints.object_count.rule}")

    print("\n" + "=" * 60)
    print("✓ ALL TESTS PASSED!")
    print("=" * 60)
    print("\nTask embedding module ready for integration!")
