"""
ARC Task Solver with train-first gating strategy.

Implements: "Solve all train pairs perfectly → then predict test pairs"
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

import torch
import torch.nn as nn
from typing import Dict, List, Tuple, Optional

from arc_nodsl.evaluation.metrics import (
    TaskResult, PairResult, TaskSolvedMetric, exact_match
)
from arc_nodsl.inference.task_embed import build_task_embedding
from arc_nodsl.inference.latent_search import beam_search
from arc_nodsl.data.augment import (
    Transform, apply_transform, invert_transform,
    generate_color_permutation, apply_color_permutation, invert_color_permutation
)
import numpy as np


class ARCSolver:
    """
    Solve ARC tasks with train-first gating strategy.

    Algorithm:
    1. Attempt all train pairs with beam search
    2. Check if task is "solved" (all train pairs perfect)
    3. IF solved: predict test pairs with beam search
       ELSE: skip test prediction (return None)
    4. Return comprehensive TaskResult

    This ensures we only predict test when we've mastered the training pattern.
    """

    def __init__(
        self,
        encoder: nn.Module,
        controller: nn.Module,
        operators: nn.Module,
        renderer: nn.Module,
        beam_size: int = 16,          # Larger for inference
        max_operator_steps: int = 8,  # More steps for complex tasks
        num_attempts: int = 2,        # Phase 5B: K attempts per test (competition = 2)
        device: torch.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    ):
        """
        Initialize ARC solver.

        Args:
            encoder: Pretrained SlotEncoder
            controller: Trained Controller
            operators: Trained OperatorLibrary
            renderer: Pretrained SlotRenderer
            beam_size: Beam size for search (larger = better but slower)
            max_operator_steps: Max operator sequence length
            num_attempts: Number of attempts per test input (2 for competition)
            device: Compute device
        """
        self.encoder = encoder
        self.controller = controller
        self.operators = operators
        self.renderer = renderer
        self.beam_size = beam_size
        self.max_operator_steps = max_operator_steps
        self.num_attempts = num_attempts
        self.device = device

        # Metric computer
        self.metric = TaskSolvedMetric(require_all_train=True)

        # Set all models to eval mode
        self.encoder.eval()
        self.controller.eval()
        self.operators.eval()
        self.renderer.eval()

    def solve_task(
        self,
        task_data: Dict,
        verbose: bool = False
    ) -> TaskResult:
        """
        Solve a single ARC task with train-first gating.

        Steps:
        1. Build task embedding from train I/O examples
        2. Attempt all train pairs with beam search
        3. Check if all train pairs are perfectly solved
        4. IF train_solved: attempt test pairs
           ELSE: skip test prediction
        5. Return comprehensive TaskResult

        Args:
            task_data: Task dict from ARCDataset with:
                - task_id: str
                - train_inputs: List[torch.Tensor]
                - train_outputs: List[torch.Tensor]
                - train_shapes: List[dict]
                - test_inputs: List[torch.Tensor]
                - test_outputs: List[torch.Tensor] (may be None)
                - test_shapes: List[dict]
            verbose: If True, print detailed progress

        Returns:
            TaskResult with comprehensive evaluation
        """
        task_id = task_data['task_id']

        if verbose:
            print(f"\n{'='*60}")
            print(f"Solving Task: {task_id}")
            print(f"{'='*60}")

        # Step 1: Build task embedding from train pairs
        train_pairs = self._prepare_train_pairs(task_data)
        task_embed = build_task_embedding(
            train_pairs,
            encoder=None,  # Don't analyze operators (slow)
            device=self.device,
            analyze_operators=False
        )

        if verbose:
            print(f"\nTask embedding built from {len(train_pairs)} train pairs")
            if 'constraints' in task_embed and task_embed['constraints'] is not None:
                print(f"  Constraints extracted")

        # Step 2: Attempt all train pairs
        if verbose:
            print(f"\n[1/3] Attempting {len(task_data['train_inputs'])} train pairs...")

        train_predictions = []
        for i in range(len(task_data['train_inputs'])):
            pred = self._solve_pair(
                task_data['train_inputs'][i],
                task_data['train_shapes'][i]['input'],
                task_data['train_shapes'][i]['output'],
                task_embed,
                target=task_data['train_outputs'][i]  # For ranking candidates
            )
            train_predictions.append(pred)

            if verbose:
                # Quick check
                h, w = task_data['train_shapes'][i]['output']
                target = task_data['train_outputs'][i].to(self.device)
                is_correct = exact_match(pred, target, h, w)
                status = "✓" if is_correct else "✗"
                print(f"  Pair {i+1}: {status}")

        # Step 3: Check if train is solved (all pairs perfect)
        # Move predictions to CPU for evaluation
        train_predictions_cpu = [p.cpu() for p in train_predictions]
        train_solved, train_results = self.metric.evaluate_train_pairs(
            train_predictions_cpu,
            task_data['train_outputs'],
            [s['output'] for s in task_data['train_shapes']]
        )

        if verbose:
            solved_count = sum(r.is_solved for r in train_results)
            print(f"\n[2/3] Train evaluation: {solved_count}/{len(train_results)} pairs solved")
            if not train_solved:
                print(f"  ⚠ Train not fully solved. Skipping test prediction.")
                for i, r in enumerate(train_results):
                    if not r.is_solved:
                        print(f"    Pair {i+1}: {r.pixel_accuracy:.1%} accuracy (need 100%)")

        # Step 4: Conditional test prediction (GATE HERE!)
        test_predictions = None
        test_correct = None
        task_success = False

        # Phase 5B: Multi-attempt support
        test_attempts = None  # List[List[Tensor]] - K attempts per test input
        competition_score = None

        if train_solved:
            if verbose:
                print(f"\n[3/3] ✓ Train solved! Attempting {len(task_data['test_inputs'])} test pairs...")
                print(f"  Generating {self.num_attempts} attempts per test input...")

            test_attempts = []  # Outer list = test inputs, inner list = K attempts
            test_predictions = []  # Best prediction per test input (backward compat)

            for i in range(len(task_data['test_inputs'])):
                # Generate K attempts per test input
                attempts = self._solve_pair_multi_attempt(
                    task_data['test_inputs'][i],
                    task_data['test_shapes'][i]['input'],
                    task_data['test_shapes'][i]['output'],
                    task_embed,
                    target=None,  # No target for test (competition setting)
                    num_attempts=self.num_attempts
                )
                test_attempts.append(attempts)
                test_predictions.append(attempts[0])  # Best attempt for backward compat

            # Check test correctness if outputs are available
            if (len(task_data['test_outputs']) > 0 and
                    task_data['test_outputs'][0] is not None):

                test_correct = []
                per_output_scores = []  # For competition scoring

                # Move attempts to CPU for evaluation
                test_attempts_cpu = [[a.cpu() for a in attempts] for attempts in test_attempts]

                for j, (attempts_cpu, target, shape) in enumerate(zip(
                        test_attempts_cpu,
                        task_data['test_outputs'],
                        task_data['test_shapes']
                )):
                    h, w = shape['output']

                    # Competition scoring: check if ANY attempt is correct
                    any_correct = False
                    for k, attempt in enumerate(attempts_cpu):
                        if exact_match(attempt, target, h, w):
                            any_correct = True
                            if verbose and k > 0:
                                print(f"  Test pair {j+1}: ✓ (attempt {k+1}/{self.num_attempts})")
                            break

                    # For binary task success (backward compat), use best attempt
                    best_correct = exact_match(attempts_cpu[0], target, h, w)
                    test_correct.append(best_correct)

                    # Competition score: 1 if any attempt correct, 0 otherwise
                    per_output_scores.append(1.0 if any_correct else 0.0)

                    if verbose:
                        status = "✓" if any_correct else "✗"
                        print(f"  Test pair {j+1}: {status}")

                # Task success (backward compat): all BEST attempts correct
                task_success = all(test_correct)

                # Competition score: average across test outputs
                competition_score = sum(per_output_scores) / len(per_output_scores) if per_output_scores else 0.0

                if verbose:
                    correct_count = sum(test_correct)
                    print(f"\n  Test result (best attempts): {correct_count}/{len(test_correct)} correct")
                    comp_correct = sum(per_output_scores)
                    print(f"  Competition score: {competition_score:.1%} ({int(comp_correct)}/{len(per_output_scores)} with ANY attempt correct)")
                    if task_success:
                        print(f"  🎉 TASK FULLY SOLVED (best attempts)!")
            else:
                if verbose:
                    print(f"  (Test outputs not available for verification)")
        else:
            if verbose:
                print(f"\n[3/3] ✗ Train not solved. Skipping test.")

        # Step 5: Return comprehensive result
        if verbose:
            print(f"\n{'='*60}")
            print(f"Task {task_id} Summary:")
            print(f"  Train solved: {train_solved}")
            print(f"  Test attempted: {test_predictions is not None}")
            print(f"  Task success: {task_success}")
            print(f"{'='*60}\n")

        return TaskResult(
            task_id=task_id,
            train_solved=train_solved,
            train_results=train_results,
            test_attempted=test_predictions is not None,
            test_predictions=test_predictions,
            test_correct=test_correct,
            task_success=task_success,
            confidence=1.0 if train_solved else 0.0,
            test_attempts=test_attempts,
            competition_score=competition_score
        )

    def _solve_pair(
        self,
        input_grid: torch.Tensor,
        input_shape: Tuple[int, int],
        output_shape: Tuple[int, int],
        task_embed: Dict,
        target: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """
        Solve a single input-output pair with beam search.

        Args:
            input_grid: Input grid [H, W]
            input_shape: Actual input size (h, w)
            output_shape: Expected output size (h, w)
            task_embed: Task embedding dict
            target: Optional target grid for ranking candidates

        Returns:
            Predicted grid [H, W]
        """
        with torch.no_grad():
            candidates = beam_search(
                self.encoder,
                self.controller,
                self.operators,
                self.renderer,
                input_grid.to(self.device),
                input_shape,
                output_shape,
                task_embed,
                target_grid=target.to(self.device) if target is not None else None,
                beam_size=self.beam_size,
                max_steps=self.max_operator_steps,
                device=self.device,
                collect_log_probs=False  # No gradients for inference
            )

        if len(candidates) > 0:
            # Return best candidate
            return candidates[0].prediction
        else:
            # No solution found, return zeros (will be marked as wrong)
            return torch.zeros(30, 30, dtype=torch.long, device=self.device)

    def _solve_pair_multi_attempt(
        self,
        input_grid: torch.Tensor,
        input_shape: Tuple[int, int],
        output_shape: Tuple[int, int],
        task_embed: Dict,
        target: Optional[torch.Tensor] = None,
        num_attempts: int = 2
    ) -> List[torch.Tensor]:
        """
        Solve a single input-output pair with multiple attempts (Phase 5B).

        Args:
            input_grid: Input grid [H, W]
            input_shape: Actual input size (h, w)
            output_shape: Expected output size (h, w)
            task_embed: Task embedding dict
            target: Optional target grid for ranking candidates
            num_attempts: Number of diverse attempts to generate

        Returns:
            List of predicted grids [H, W] (length = num_attempts)
        """
        with torch.no_grad():
            candidates = beam_search(
                self.encoder,
                self.controller,
                self.operators,
                self.renderer,
                input_grid.to(self.device),
                input_shape,
                output_shape,
                task_embed,
                target_grid=target.to(self.device) if target is not None else None,
                beam_size=self.beam_size,
                max_steps=self.max_operator_steps,
                device=self.device,
                collect_log_probs=False
            )

        # Return top-K diverse predictions
        predictions = []
        if len(candidates) > 0:
            # Take top K candidates (already sorted by score)
            for i in range(min(num_attempts, len(candidates))):
                predictions.append(candidates[i].prediction)

            # Pad with zeros if we have fewer than K candidates
            while len(predictions) < num_attempts:
                predictions.append(torch.zeros(30, 30, dtype=torch.long, device=self.device))
        else:
            # No solution found, return K zeros
            for _ in range(num_attempts):
                predictions.append(torch.zeros(30, 30, dtype=torch.long, device=self.device))

        return predictions

    def _prepare_train_pairs(
        self,
        task_data: Dict
    ) -> List[Tuple[torch.Tensor, torch.Tensor, Dict]]:
        """
        Prepare train pairs for task embedding.

        Args:
            task_data: Task dict from ARCDataset

        Returns:
            List of (input, output, shapes) tuples
        """
        pairs = []
        for i in range(len(task_data['train_inputs'])):
            pairs.append((
                task_data['train_inputs'][i],
                task_data['train_outputs'][i],
                task_data['train_shapes'][i]
            ))
        return pairs


class TTASolver(ARCSolver):
    """
    ARC Solver with Test-Time Augmentation (TTA) ensemble.

    Extends ARCSolver with TTA for test pairs:
    - Train pairs: Use base solver (no TTA, for speed)
    - Test pairs: Ensemble predictions across all 8 transforms

    This increases computational cost 8x but improves accuracy by +2-5%.
    """

    def __init__(
        self,
        encoder: nn.Module,
        controller: nn.Module,
        operators: nn.Module,
        renderer: nn.Module,
        beam_size: int = 16,
        max_operator_steps: int = 8,
        num_attempts: int = 2,
        tta_mode: str = 'majority',
        tta_color: bool = False,
        tta_color_variants: int = 4,
        device: torch.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    ):
        """
        Initialize TTA solver.

        Args:
            encoder, controller, operators, renderer: Model components
            beam_size: Beam size for search
            max_operator_steps: Max operator sequence length
            num_attempts: Number of attempts per test input
            tta_mode: Ensemble mode ('majority' or 'first_success')
            tta_color: If True, apply color TTA (ensemble across color permutations)
            tta_color_variants: Number of color permutations to ensemble (default 4)
            device: Compute device
        """
        super().__init__(encoder, controller, operators, renderer,
                         beam_size, max_operator_steps, num_attempts, device)
        self.tta_mode = tta_mode
        self.tta_color = tta_color
        self.tta_color_variants = tta_color_variants

        if tta_mode not in ['majority', 'first_success']:
            raise ValueError(f"Unknown TTA mode: {tta_mode}. Use 'majority' or 'first_success'.")

    def _solve_pair_with_tta(
        self,
        input_grid: torch.Tensor,
        input_shape: Tuple[int, int],
        output_shape: Tuple[int, int],
        task_embed: Dict,
        target: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """
        Solve pair with test-time augmentation ensemble.

        Algorithm:
        1. For each of 8 transforms:
           a. Transform input
           b. Run inference
           c. Inverse transform prediction
        2. Ensemble via majority voting (pixel-wise)
        3. Return final prediction

        Args:
            input_grid: Input grid [H, W]
            input_shape: Actual input size (h, w)
            output_shape: Expected output size (h, w)
            task_embed: Task embedding dict
            target: Optional target for ranking/validation

        Returns:
            Final ensembled prediction [H, W]
        """
        all_predictions = []  # List of 8 predictions in original space

        h_in, w_in = input_shape
        h_out, w_out = output_shape

        for transform in Transform:
            # 1. Transform input
            input_aug = apply_transform(input_grid, transform, h_in, w_in)

            # Adjust shapes if rotation changed dimensions
            if transform in [Transform.ROT_90, Transform.ROT_270,
                           Transform.FLIP_D1, Transform.FLIP_D2]:
                input_shape_aug = (w_in, h_in)
                output_shape_aug = (w_out, h_out)
            else:
                input_shape_aug = (h_in, w_in)
                output_shape_aug = (h_out, w_out)

            # 2. Run inference on augmented input
            with torch.no_grad():
                candidates = beam_search(
                    self.encoder,
                    self.controller,
                    self.operators,
                    self.renderer,
                    input_aug.to(self.device),
                    input_shape_aug,
                    output_shape_aug,
                    task_embed,
                    target_grid=None,  # No target for test
                    beam_size=self.beam_size,
                    max_steps=self.max_operator_steps,
                    device=self.device,
                    collect_log_probs=False
                )

            if len(candidates) > 0:
                pred_aug = candidates[0].prediction
            else:
                pred_aug = torch.zeros(30, 30, dtype=torch.long, device=self.device)

            # 3. Inverse transform prediction back to original space
            inverse_trans = invert_transform(transform)
            h_out_aug, w_out_aug = output_shape_aug
            pred_original = apply_transform(pred_aug, inverse_trans, h_out_aug, w_out_aug)

            all_predictions.append(pred_original)

        # 4. Ensemble predictions
        if self.tta_mode == 'majority':
            return self._ensemble_majority_vote(all_predictions, output_shape)
        elif self.tta_mode == 'first_success':
            # Need target for this mode
            if target is not None:
                return self._ensemble_first_success(all_predictions, target, output_shape)
            else:
                # Fallback to majority if no target
                return self._ensemble_majority_vote(all_predictions, output_shape)
        else:
            raise ValueError(f"Unknown TTA mode: {self.tta_mode}")

    def _ensemble_majority_vote(
        self,
        predictions: List[torch.Tensor],
        output_shape: Tuple[int, int]
    ) -> torch.Tensor:
        """
        Ensemble predictions via pixel-wise majority voting.

        For each pixel position, select the most common color value
        across all 8 predictions.

        Args:
            predictions: List of 8 predictions [H, W] each
            output_shape: Output size (h, w)

        Returns:
            Final prediction [H, W] with majority-voted pixels
        """
        h, w = output_shape

        # Stack predictions: [8, H, W]
        stacked = torch.stack(predictions, dim=0)

        # For each pixel, find mode (most common value)
        final_pred = torch.zeros(30, 30, dtype=torch.long, device=self.device)

        for i in range(h):
            for j in range(w):
                # Get all 8 predictions for this pixel
                pixel_values = stacked[:, i, j]  # [8]

                # Find mode (most common value)
                # torch.mode returns (values, indices), we want the value
                mode_value = torch.mode(pixel_values).values.item()
                final_pred[i, j] = mode_value

        return final_pred

    def _ensemble_first_success(
        self,
        predictions: List[torch.Tensor],
        target: torch.Tensor,
        output_shape: Tuple[int, int]
    ) -> torch.Tensor:
        """
        Return first prediction that matches target exactly.
        Fallback to majority vote if none match.

        Args:
            predictions: List of 8 predictions
            target: Target grid for comparison
            output_shape: Output size (h, w)

        Returns:
            First matching prediction, or majority vote fallback
        """
        h, w = output_shape

        # Check each prediction for exact match
        for pred in predictions:
            if exact_match(pred, target, h, w):
                return pred

        # No exact match, use majority voting
        return self._ensemble_majority_vote(predictions, output_shape)

    def _solve_pair_with_spatial_and_color_tta(
        self,
        input_grid: torch.Tensor,
        input_shape: Tuple[int, int],
        output_shape: Tuple[int, int],
        task_embed: Dict,
        target: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """
        Solve pair with combined spatial and color TTA ensemble.

        Algorithm:
        1. For each of 8 spatial transforms:
           For each of N color permutations:
              a. Transform input (spatial + color)
              b. Run inference
              c. Inverse transform prediction (spatial + color)
        2. Ensemble via majority voting across 8*N predictions
        3. Return final prediction

        Args:
            input_grid: Input grid [H, W]
            input_shape: Actual input size (h, w)
            output_shape: Expected output size (h, w)
            task_embed: Task embedding dict
            target: Optional target for ranking/validation

        Returns:
            Final ensembled prediction [H, W]
        """
        all_predictions = []  # List of 8*N predictions in original space

        h_in, w_in = input_shape
        h_out, w_out = output_shape

        # Generate N random color permutations
        color_perms = [
            generate_color_permutation(exclude_identity=True)
            for _ in range(self.tta_color_variants)
        ]

        # Ensemble across spatial and color augmentations
        for spatial_transform in Transform:
            # 1. Apply spatial transformation
            input_spatial = apply_transform(input_grid, spatial_transform, h_in, w_in)

            # Adjust shapes if rotation changed dimensions
            if spatial_transform in [Transform.ROT_90, Transform.ROT_270,
                                   Transform.FLIP_D1, Transform.FLIP_D2]:
                input_shape_aug = (w_in, h_in)
                output_shape_aug = (w_out, h_out)
            else:
                input_shape_aug = (h_in, w_in)
                output_shape_aug = (h_out, w_out)

            for color_perm in color_perms:
                # 2. Apply color permutation
                input_aug = apply_color_permutation(input_spatial, color_perm)

                # 3. Run inference on augmented input
                with torch.no_grad():
                    candidates = beam_search(
                        self.encoder,
                        self.controller,
                        self.operators,
                        self.renderer,
                        input_aug.to(self.device),
                        input_shape_aug,
                        output_shape_aug,
                        task_embed,
                        target_grid=None,  # No target for test
                        beam_size=self.beam_size,
                        max_steps=self.max_operator_steps,
                        device=self.device,
                        collect_log_probs=False
                    )

                if len(candidates) > 0:
                    pred_aug = candidates[0].prediction
                else:
                    pred_aug = torch.zeros(30, 30, dtype=torch.long, device=self.device)

                # 4. Inverse color permutation
                inv_color_perm = invert_color_permutation(color_perm)
                pred_color_inv = apply_color_permutation(pred_aug, inv_color_perm)

                # 5. Inverse spatial transformation
                inverse_spatial = invert_transform(spatial_transform)
                h_out_aug, w_out_aug = output_shape_aug
                pred_original = apply_transform(pred_color_inv, inverse_spatial, h_out_aug, w_out_aug)

                all_predictions.append(pred_original)

        # 6. Ensemble predictions (8*N predictions)
        if self.tta_mode == 'majority':
            return self._ensemble_majority_vote(all_predictions, output_shape)
        elif self.tta_mode == 'first_success':
            if target is not None:
                return self._ensemble_first_success(all_predictions, target, output_shape)
            else:
                return self._ensemble_majority_vote(all_predictions, output_shape)
        else:
            raise ValueError(f"Unknown TTA mode: {self.tta_mode}")

    def _solve_pair_multi_attempt(
        self,
        input_grid: torch.Tensor,
        input_shape: Tuple[int, int],
        output_shape: Tuple[int, int],
        task_embed: Dict,
        target: Optional[torch.Tensor] = None,
        num_attempts: int = 2
    ) -> List[torch.Tensor]:
        """
        Override to use TTA ensemble for first attempt, beam diversity for others.

        Running full TTA for K attempts would be 8*K (or 8*N*K with color) inferences (expensive).
        Strategy: TTA for attempt 1, beam diversity for remaining attempts.

        Args:
            input_grid: Input grid [H, W]
            input_shape: Actual input size (h, w)
            output_shape: Expected output size (h, w)
            task_embed: Task embedding dict
            target: Optional target for ranking
            num_attempts: Number of diverse attempts to generate

        Returns:
            List of predicted grids [H, W] (length = num_attempts)
        """
        # Attempt 1: Use appropriate TTA ensemble
        if self.tta_color:
            # Use combined spatial + color TTA (8*N predictions)
            attempt_1 = self._solve_pair_with_spatial_and_color_tta(
                input_grid, input_shape, output_shape, task_embed, target
            )
        else:
            # Use spatial-only TTA (8 predictions)
            attempt_1 = self._solve_pair_with_tta(
                input_grid, input_shape, output_shape, task_embed, target
            )

        attempts = [attempt_1]

        # Remaining attempts: Use diverse beam candidates (no TTA)
        if num_attempts > 1:
            with torch.no_grad():
                candidates = beam_search(
                    self.encoder,
                    self.controller,
                    self.operators,
                    self.renderer,
                    input_grid.to(self.device),
                    input_shape,
                    output_shape,
                    task_embed,
                    target_grid=target.to(self.device) if target is not None else None,
                    beam_size=self.beam_size,
                    max_steps=self.max_operator_steps,
                    device=self.device,
                    collect_log_probs=False
                )

            # Add remaining attempts from beam candidates
            for i in range(1, num_attempts):
                if i < len(candidates):
                    attempts.append(candidates[i].prediction)
                else:
                    # Pad with zeros if not enough candidates
                    attempts.append(torch.zeros(30, 30, dtype=torch.long, device=self.device))

        return attempts


# Test code
if __name__ == "__main__":
    from arc_nodsl.data.loader import ARCDataset
    from arc_nodsl.models.slots import SlotEncoder
    from arc_nodsl.models.renderer import SlotRenderer
    from arc_nodsl.models.operators import OperatorLibrary
    from arc_nodsl.models.controller import Controller

    print("="*60)
    print("Testing ARCSolver")
    print("="*60)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"\nDevice: {device}")

    # Load dataset
    print("\nLoading dataset...")
    dataset = ARCDataset("data/arc-agi_evaluation_challenges.json")
    print(f"Loaded {len(dataset)} evaluation tasks")

    # Create models (random weights for testing)
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

    # Create solver
    print("\nCreating ARCSolver...")
    solver = ARCSolver(
        encoder, controller, operators, renderer,
        beam_size=4,  # Small for fast testing
        max_operator_steps=2,
        num_attempts=2,  # Phase 5B: competition setting
        device=device
    )

    # Test on first task
    print("\n" + "="*60)
    print("Testing on first validation task")
    print("="*60)

    task = dataset[0]
    result = solver.solve_task(task, verbose=True)

    # Verify result structure
    print("\n" + "="*60)
    print("Result Verification")
    print("="*60)
    assert result.task_id == task['task_id'], "Task ID mismatch"
    assert len(result.train_results) == len(task['train_inputs']), "Train results count mismatch"
    print(f"✓ Task ID: {result.task_id}")
    print(f"✓ Train results: {len(result.train_results)} pairs")
    print(f"✓ Train solved: {result.train_solved}")
    print(f"✓ Test attempted: {result.test_attempted}")
    print(f"✓ Task success: {result.task_success}")

    # Test gating logic
    if result.train_solved:
        assert result.test_attempted == True, "Should attempt test when train solved"
        assert result.test_predictions is not None, "Should have test predictions"
        print(f"✓ Gating logic: Test attempted (train was solved)")
    else:
        assert result.test_attempted == False, "Should NOT attempt test when train unsolved"
        assert result.test_predictions is None, "Should NOT have test predictions"
        print(f"✓ Gating logic: Test skipped (train not solved)")

    print("\n" + "="*60)
    print("✓ ARCSolver test complete!")
    print("="*60)
    print(f"\nNote: With random weights, train_solved is expected to be False.")
    print(f"After training, we expect train_solved on 10-30% of tasks.")
