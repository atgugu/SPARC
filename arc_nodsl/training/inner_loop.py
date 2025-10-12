"""
Inner loop training: single-task adaptation.

Trains the controller to solve a specific task using its train pairs as supervision.
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

import torch
import torch.nn as nn
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass

from arc_nodsl.inference.task_embed import build_task_embedding
from arc_nodsl.inference.latent_search import beam_search
from arc_nodsl.training.losses import ReinforceReward, SequenceLoss


@dataclass
class InnerLoopMetrics:
    """Metrics from inner loop training."""
    mean_reward: float
    best_reward: float
    policy_loss: float
    entropy: float
    baseline: float
    num_steps: int
    success_rate: float  # Fraction of train pairs solved
    # New: support/query split metrics
    query_solved: Optional[bool] = None  # Was query pair 100% correct?
    support_solved: Optional[bool] = None  # Were all support pairs solved?
    query_pair_idx: Optional[int] = None  # Which pair was used as query


class InnerLoop:
    """
    Single-task adaptation via REINFORCE.

    Algorithm:
    1. Build task embedding from train pairs
    2. For each inner step:
        a. Sample a train pair (input, output)
        b. Run beam search with current controller
        c. Compute rewards for all beam candidates
        d. Update controller with REINFORCE
    3. Return adapted controller + metrics
    """

    def __init__(
        self,
        num_inner_steps: int = 10,
        beam_size: int = 8,
        max_operator_steps: int = 4,
        learning_rate: float = 1e-3,
        entropy_weight: float = 0.01,
        reward_threshold: float = 0.95,  # Consider "solved" threshold
        binary_bonus_weight: float = 0.5,  # Phase 5B: bonus for 100% exact match
        device: torch.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    ):
        """
        Initialize inner loop trainer.

        Args:
            num_inner_steps: Number of gradient steps per task
            beam_size: Beam size for search
            max_operator_steps: Max operator sequence length
            learning_rate: Learning rate for controller
            entropy_weight: Weight for entropy regularization
            reward_threshold: Threshold to consider a pair "solved"
            binary_bonus_weight: Bonus added for 100% exact match (0 = disabled)
            device: Compute device
        """
        self.num_inner_steps = num_inner_steps
        self.beam_size = beam_size
        self.max_operator_steps = max_operator_steps
        self.learning_rate = learning_rate
        self.entropy_weight = entropy_weight
        self.reward_threshold = reward_threshold
        self.binary_bonus_weight = binary_bonus_weight
        self.device = device

        # Loss components
        self.reward_computer = ReinforceReward()
        self.sequence_loss = SequenceLoss(entropy_weight=entropy_weight)

    def train_on_task(
        self,
        task_data: Dict,
        encoder: nn.Module,
        controller: nn.Module,
        operators: nn.Module,
        renderer: nn.Module,
        clone_controller: bool = True,
        verbose: bool = False
    ) -> Tuple[nn.Module, InnerLoopMetrics]:
        """
        Train controller on a single task.

        Args:
            task_data: Task dict from dataset with train_inputs/outputs
            encoder: Pretrained encoder
            controller: Controller to adapt (will be cloned if clone_controller=True)
            operators: Pretrained operators
            renderer: Pretrained renderer
            clone_controller: Whether to clone controller (default True for meta-learning)
            verbose: Print progress

        Returns:
            (adapted_controller, metrics)
        """
        # 1. Clone controller if needed (for meta-learning)
        if clone_controller:
            adapted_controller = self._clone_controller(controller)
        else:
            adapted_controller = controller

        # 2. Create optimizer for adapted controller
        optimizer = torch.optim.AdamW(
            adapted_controller.parameters(),
            lr=self.learning_rate
        )

        # 3. Build task embedding from train pairs
        train_pairs = []
        for i in range(len(task_data['train_inputs'])):
            train_pairs.append((
                task_data['train_inputs'][i],
                task_data['train_outputs'][i],
                task_data['train_shapes'][i]
            ))

        task_embed = build_task_embedding(
            train_pairs,
            encoder=None,  # Don't analyze operators (slow)
            device=self.device,
            analyze_operators=False
        )

        if verbose:
            print(f"\nInner loop training on task {task_data['task_id']}")
            print(f"  Train pairs: {len(train_pairs)}")
            print(f"  Steps: {self.num_inner_steps}")

        # 4. Training loop
        all_rewards = []
        all_losses = []

        for step in range(self.num_inner_steps):
            # Sample a train pair (cycle through them)
            pair_idx = step % len(train_pairs)
            input_grid = task_data['train_inputs'][pair_idx].to(self.device)
            target_grid = task_data['train_outputs'][pair_idx].to(self.device)
            input_shape = task_data['train_shapes'][pair_idx]['input']
            output_shape = task_data['train_shapes'][pair_idx]['output']

            # Run beam search with log prob collection
            candidates = beam_search(
                encoder,
                adapted_controller,
                operators,
                renderer,
                input_grid,
                input_shape,
                output_shape,
                task_embed,
                target_grid=target_grid,
                beam_size=self.beam_size,
                max_steps=self.max_operator_steps,
                device=self.device,
                collect_log_probs=True  # NEW: track for REINFORCE
            )

            # Compute rewards for all candidates (fuzzy + binary bonus)
            rewards = []
            h, w = output_shape
            for cand in candidates:
                # Fuzzy reward (0.7 * accuracy + 0.3 * constraints)
                fuzzy_reward = self.reward_computer.compute_reward(
                    cand.prediction,
                    target_grid,
                    cand.h,
                    cand.w,
                    constraints=task_embed['constraints'],
                    input_shape=input_shape
                )

                # Binary bonus: add bonus if 100% exact match
                binary_bonus = 0.0
                if self.binary_bonus_weight > 0:
                    from arc_nodsl.evaluation.metrics import exact_match
                    if exact_match(cand.prediction.cpu(), target_grid.cpu(), h, w):
                        binary_bonus = self.binary_bonus_weight

                # Final reward = fuzzy + binary bonus
                total_reward = fuzzy_reward + binary_bonus
                rewards.append(total_reward)

            all_rewards.extend(rewards)

            # Extract log probs, entropies, and CORRESPONDING rewards
            # Filter to only candidates with valid log_probs to avoid size mismatch
            valid_indices = [i for i, cand in enumerate(candidates) if cand.log_probs is not None]

            if len(valid_indices) == 0:
                # No log probs collected (shouldn't happen, but handle gracefully)
                if verbose:
                    print(f"  Step {step+1}: No log probs collected, skipping")
                continue

            log_probs_list = [candidates[i].log_probs for i in valid_indices]
            entropies_list = [candidates[i].entropies for i in valid_indices if candidates[i].entropies is not None]
            rewards_valid = [rewards[i] for i in valid_indices]
            rewards_tensor = torch.tensor(rewards_valid, device=self.device)

            # Stack log probs: [beam_size, num_steps]
            # Each candidate has a list of log probs per step
            # Need to handle variable-length sequences with padding + mask
            max_steps = max(len(lp) for lp in log_probs_list)
            beam_size = len(log_probs_list)

            log_probs_per_step = []
            entropies_per_step = []
            mask = torch.zeros(beam_size, max_steps, device=self.device)

            for t in range(max_steps):
                step_log_probs = []
                step_entropies = []

                for i, lp in enumerate(log_probs_list):
                    if t < len(lp):
                        step_log_probs.append(lp[t])
                        mask[i, t] = 1.0  # Valid step
                        if len(entropies_list) > 0:
                            step_entropies.append(entropies_list[i][t])
                    else:
                        # Pad with zero
                        step_log_probs.append(torch.tensor(0.0, device=self.device))
                        if len(entropies_list) > 0:
                            step_entropies.append(torch.tensor(0.0, device=self.device))

                log_probs_per_step.append(torch.stack(step_log_probs))
                if len(step_entropies) > 0:
                    entropies_per_step.append(torch.stack(step_entropies))

            # Compute REINFORCE loss with mask for variable-length sequences
            loss_dict = self.sequence_loss.compute_loss(
                log_probs_per_step,
                rewards_tensor,
                entropies=entropies_per_step if len(entropies_per_step) > 0 else None,
                mask=mask
            )

            loss = loss_dict['loss']
            all_losses.append(loss.item())

            # Backprop and update
            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(adapted_controller.parameters(), max_norm=1.0)
            optimizer.step()

            if verbose and (step + 1) % max(1, self.num_inner_steps // 4) == 0:
                best_reward = max(rewards)
                mean_reward = sum(rewards) / len(rewards)
                print(f"  Step {step+1}/{self.num_inner_steps}: "
                      f"mean_reward={mean_reward:.3f}, "
                      f"best_reward={best_reward:.3f}, "
                      f"loss={loss.item():.4f}")

        # 5. Compute final metrics
        metrics = self._compute_metrics(
            all_rewards,
            all_losses,
            adapted_controller,
            task_data,
            task_embed,
            encoder,
            operators,
            renderer
        )

        if verbose:
            print(f"  Final metrics:")
            print(f"    Mean reward: {metrics.mean_reward:.3f}")
            print(f"    Best reward: {metrics.best_reward:.3f}")
            print(f"    Success rate: {metrics.success_rate:.1%}")

        return adapted_controller, metrics

    def train_on_task_with_split(
        self,
        task_data: Dict,
        encoder: nn.Module,
        controller: nn.Module,
        operators: nn.Module,
        renderer: nn.Module,
        query_index: int = -1,
        clone_controller: bool = True,
        verbose: bool = False
    ) -> Tuple[nn.Module, InnerLoopMetrics]:
        """
        Train controller with support/query split.

        This mimics the train→test structure during meta-learning by holding out
        one train pair as a "query" set (like a test pair).

        Args:
            task_data: Task dict from dataset
            encoder: Pretrained encoder
            controller: Controller to adapt
            operators: Pretrained operators
            renderer: Pretrained renderer
            query_index: Which train pair to use as query (-1 = last, default)
            clone_controller: Whether to clone controller
            verbose: Print progress

        Returns:
            (adapted_controller, metrics with query_solved)
        """
        # 1. Clone controller if needed
        if clone_controller:
            adapted_controller = self._clone_controller(controller)
        else:
            adapted_controller = controller

        # 2. Create optimizer
        optimizer = torch.optim.AdamW(
            adapted_controller.parameters(),
            lr=self.learning_rate
        )

        # 3. Split train pairs into support and query
        num_train_pairs = len(task_data['train_inputs'])
        if num_train_pairs < 2:
            # Can't split with < 2 pairs, fall back to regular training
            if verbose:
                print(f"Warning: Only {num_train_pairs} train pair(s), cannot split. Using regular training.")
            return self.train_on_task(
                task_data, encoder, controller, operators, renderer,
                clone_controller=False,  # Already cloned
                verbose=verbose
            )

        # Determine query index
        query_idx = query_index if query_index >= 0 else num_train_pairs - 1
        if query_idx >= num_train_pairs:
            query_idx = num_train_pairs - 1

        # Support indices (all except query)
        support_indices = [i for i in range(num_train_pairs) if i != query_idx]

        if verbose:
            print(f"\nInner loop training on task {task_data['task_id']} (with split)")
            print(f"  Total train pairs: {num_train_pairs}")
            print(f"  Support pairs: {len(support_indices)} (training)")
            print(f"  Query pair: 1 (index {query_idx}, held-out for evaluation)")
            print(f"  Steps: {self.num_inner_steps}")

        # 4. Build task embedding from SUPPORT ONLY (no data leakage!)
        support_pairs = []
        for i in support_indices:
            support_pairs.append((
                task_data['train_inputs'][i],
                task_data['train_outputs'][i],
                task_data['train_shapes'][i]
            ))

        task_embed = build_task_embedding(
            support_pairs,
            encoder=None,
            device=self.device,
            analyze_operators=False
        )

        # 5. Training loop on SUPPORT pairs only
        all_rewards = []
        all_losses = []

        for step in range(self.num_inner_steps):
            # Cycle through SUPPORT pairs only
            support_idx = support_indices[step % len(support_indices)]
            input_grid = task_data['train_inputs'][support_idx].to(self.device)
            target_grid = task_data['train_outputs'][support_idx].to(self.device)
            input_shape = task_data['train_shapes'][support_idx]['input']
            output_shape = task_data['train_shapes'][support_idx]['output']

            # Run beam search with log prob collection
            candidates = beam_search(
                encoder,
                adapted_controller,
                operators,
                renderer,
                input_grid,
                input_shape,
                output_shape,
                task_embed,
                target_grid=target_grid,
                beam_size=self.beam_size,
                max_steps=self.max_operator_steps,
                device=self.device,
                collect_log_probs=True
            )

            # Compute rewards (fuzzy + binary bonus)
            rewards = []
            h, w = output_shape
            for cand in candidates:
                # Fuzzy reward (0.7 * accuracy + 0.3 * constraints)
                fuzzy_reward = self.reward_computer.compute_reward(
                    cand.prediction,
                    target_grid,
                    cand.h,
                    cand.w,
                    constraints=task_embed['constraints'],
                    input_shape=input_shape
                )

                # Binary bonus: add bonus if 100% exact match
                binary_bonus = 0.0
                if self.binary_bonus_weight > 0:
                    from arc_nodsl.evaluation.metrics import exact_match
                    if exact_match(cand.prediction.cpu(), target_grid.cpu(), h, w):
                        binary_bonus = self.binary_bonus_weight

                # Final reward = fuzzy + binary bonus
                total_reward = fuzzy_reward + binary_bonus
                rewards.append(total_reward)

            all_rewards.extend(rewards)

            # Extract log probs, entropies, and CORRESPONDING rewards
            # Filter to only candidates with valid log_probs to avoid size mismatch
            valid_indices = [i for i, cand in enumerate(candidates) if cand.log_probs is not None]

            if len(valid_indices) == 0:
                if verbose:
                    print(f"  Step {step+1}: No log probs collected, skipping")
                continue

            log_probs_list = [candidates[i].log_probs for i in valid_indices]
            entropies_list = [candidates[i].entropies for i in valid_indices if candidates[i].entropies is not None]
            rewards_valid = [rewards[i] for i in valid_indices]
            rewards_tensor = torch.tensor(rewards_valid, device=self.device)

            # Stack log probs with padding for variable-length sequences
            max_steps = max(len(lp) for lp in log_probs_list)
            beam_size = len(log_probs_list)

            log_probs_per_step = []
            entropies_per_step = []
            mask = torch.zeros(beam_size, max_steps, device=self.device)

            for t in range(max_steps):
                step_log_probs = []
                step_entropies = []

                for i, lp in enumerate(log_probs_list):
                    if t < len(lp):
                        step_log_probs.append(lp[t])
                        mask[i, t] = 1.0
                        if len(entropies_list) > 0:
                            step_entropies.append(entropies_list[i][t])
                    else:
                        step_log_probs.append(torch.tensor(0.0, device=self.device))
                        if len(entropies_list) > 0:
                            step_entropies.append(torch.tensor(0.0, device=self.device))

                log_probs_per_step.append(torch.stack(step_log_probs))
                if len(step_entropies) > 0:
                    entropies_per_step.append(torch.stack(step_entropies))

            # Compute REINFORCE loss
            loss_dict = self.sequence_loss.compute_loss(
                log_probs_per_step,
                rewards_tensor,
                entropies=entropies_per_step if len(entropies_per_step) > 0 else None,
                mask=mask
            )

            loss = loss_dict['loss']
            all_losses.append(loss.item())

            # Backprop and update
            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(adapted_controller.parameters(), max_norm=1.0)
            optimizer.step()

            if verbose and (step + 1) % max(1, self.num_inner_steps // 4) == 0:
                best_reward = max(rewards)
                mean_reward = sum(rewards) / len(rewards)
                print(f"  Step {step+1}/{self.num_inner_steps}: "
                      f"mean_reward={mean_reward:.3f}, "
                      f"best_reward={best_reward:.3f}, "
                      f"loss={loss.item():.4f}")

        # 6. Evaluate on QUERY pair (held-out, mimics test)
        adapted_controller.eval()
        with torch.no_grad():
            query_input = task_data['train_inputs'][query_idx].to(self.device)
            query_target = task_data['train_outputs'][query_idx].to(self.device)
            query_input_shape = task_data['train_shapes'][query_idx]['input']
            query_output_shape = task_data['train_shapes'][query_idx]['output']

            # Run beam search on query
            query_candidates = beam_search(
                encoder,
                adapted_controller,
                operators,
                renderer,
                query_input,
                query_input_shape,
                query_output_shape,
                task_embed,
                target_grid=query_target,
                beam_size=self.beam_size,
                max_steps=self.max_operator_steps,
                device=self.device,
                collect_log_probs=False
            )

            # Check if query is 100% correct (exact match)
            query_solved = False
            if len(query_candidates) > 0:
                best_pred = query_candidates[0].prediction
                h, w = query_output_shape
                # Use exact_match from metrics
                from arc_nodsl.evaluation.metrics import exact_match
                query_solved = exact_match(best_pred.cpu(), query_target.cpu(), h, w)

        # 7. Evaluate on SUPPORT pairs to compute support_solved
        support_solved = True
        for i in support_indices:
            input_grid = task_data['train_inputs'][i].to(self.device)
            target_grid = task_data['train_outputs'][i].to(self.device)
            input_shape = task_data['train_shapes'][i]['input']
            output_shape = task_data['train_shapes'][i]['output']

            candidates = beam_search(
                encoder,
                adapted_controller,
                operators,
                renderer,
                input_grid,
                input_shape,
                output_shape,
                task_embed,
                target_grid=target_grid,
                beam_size=self.beam_size,
                max_steps=self.max_operator_steps,
                device=self.device,
                collect_log_probs=False
            )

            if len(candidates) > 0:
                from arc_nodsl.evaluation.metrics import exact_match
                h, w = output_shape
                if not exact_match(candidates[0].prediction.cpu(), target_grid.cpu(), h, w):
                    support_solved = False
                    break
            else:
                support_solved = False
                break

        adapted_controller.train()

        # 8. Compute metrics
        mean_reward = sum(all_rewards) / len(all_rewards) if all_rewards else 0.0
        best_reward = max(all_rewards) if all_rewards else 0.0
        mean_loss = sum(all_losses) / len(all_losses) if all_losses else 0.0

        # Success rate computed on support pairs (for backward compat)
        success_count = sum(1 for i in support_indices
                           if self._check_pair_solved(adapted_controller, task_data, i,
                                                     task_embed, encoder, operators, renderer))
        success_rate = success_count / len(support_indices) if support_indices else 0.0

        if verbose:
            print(f"  Final metrics:")
            print(f"    Mean reward: {mean_reward:.3f}")
            print(f"    Best reward: {best_reward:.3f}")
            print(f"    Support pairs solved: {success_count}/{len(support_indices)}")
            print(f"    Support solved: {support_solved}")
            print(f"    Query solved: {query_solved} ← KEY METRIC!")

        return adapted_controller, InnerLoopMetrics(
            mean_reward=mean_reward,
            best_reward=best_reward,
            policy_loss=mean_loss,
            entropy=0.0,
            baseline=self.sequence_loss.baseline,
            num_steps=len(all_losses),
            success_rate=success_rate,
            query_solved=query_solved,
            support_solved=support_solved,
            query_pair_idx=query_idx
        )

    def _check_pair_solved(
        self,
        controller: nn.Module,
        task_data: Dict,
        pair_idx: int,
        task_embed: Dict,
        encoder: nn.Module,
        operators: nn.Module,
        renderer: nn.Module
    ) -> bool:
        """Check if a single pair is solved (for success rate computation)."""
        controller.eval()
        with torch.no_grad():
            input_grid = task_data['train_inputs'][pair_idx].to(self.device)
            target_grid = task_data['train_outputs'][pair_idx].to(self.device)
            input_shape = task_data['train_shapes'][pair_idx]['input']
            output_shape = task_data['train_shapes'][pair_idx]['output']

            candidates = beam_search(
                encoder,
                controller,
                operators,
                renderer,
                input_grid,
                input_shape,
                output_shape,
                task_embed,
                target_grid=target_grid,
                beam_size=self.beam_size,
                max_steps=self.max_operator_steps,
                device=self.device,
                collect_log_probs=False
            )

            if len(candidates) > 0:
                reward = self.reward_computer.compute_reward(
                    candidates[0].prediction,
                    target_grid,
                    candidates[0].h,
                    candidates[0].w,
                    constraints=task_embed['constraints'],
                    input_shape=input_shape
                )
                return reward >= self.reward_threshold
        return False

    def _clone_controller(self, controller: nn.Module) -> nn.Module:
        """Create a deep copy of the controller for adaptation."""
        import copy
        cloned = copy.deepcopy(controller)
        cloned.train()
        return cloned

    def _compute_metrics(
        self,
        all_rewards: List[float],
        all_losses: List[float],
        adapted_controller: nn.Module,
        task_data: Dict,
        task_embed: Dict,
        encoder: nn.Module,
        operators: nn.Module,
        renderer: nn.Module
    ) -> InnerLoopMetrics:
        """Compute final metrics after training."""
        # Basic stats
        mean_reward = sum(all_rewards) / len(all_rewards) if all_rewards else 0.0
        best_reward = max(all_rewards) if all_rewards else 0.0
        mean_loss = sum(all_losses) / len(all_losses) if all_losses else 0.0

        # Evaluate on all train pairs to compute success rate
        success_count = 0
        num_pairs = len(task_data['train_inputs'])

        adapted_controller.eval()
        with torch.no_grad():
            for i in range(num_pairs):
                input_grid = task_data['train_inputs'][i].to(self.device)
                target_grid = task_data['train_outputs'][i].to(self.device)
                input_shape = task_data['train_shapes'][i]['input']
                output_shape = task_data['train_shapes'][i]['output']

                # Run beam search
                candidates = beam_search(
                    encoder,
                    adapted_controller,
                    operators,
                    renderer,
                    input_grid,
                    input_shape,
                    output_shape,
                    task_embed,
                    target_grid=target_grid,
                    beam_size=self.beam_size,
                    max_steps=self.max_operator_steps,
                    device=self.device,
                    collect_log_probs=False
                )

                # Check if best candidate solves the pair
                if len(candidates) > 0:
                    best_cand = candidates[0]
                    reward = self.reward_computer.compute_reward(
                        best_cand.prediction,
                        target_grid,
                        best_cand.h,
                        best_cand.w,
                        constraints=task_embed['constraints'],
                        input_shape=input_shape
                    )
                    if reward >= self.reward_threshold:
                        success_count += 1

        adapted_controller.train()

        success_rate = success_count / num_pairs if num_pairs > 0 else 0.0

        return InnerLoopMetrics(
            mean_reward=mean_reward,
            best_reward=best_reward,
            policy_loss=mean_loss,
            entropy=0.0,  # TODO: track separately
            baseline=self.sequence_loss.baseline,
            num_steps=len(all_losses),
            success_rate=success_rate
        )


# Test code
if __name__ == "__main__":
    from arc_nodsl.data.loader import ARCDataset
    from arc_nodsl.models.slots import SlotEncoder
    from arc_nodsl.models.renderer import SlotRenderer
    from arc_nodsl.models.operators import OperatorLibrary
    from arc_nodsl.models.controller import Controller

    print("="*60)
    print("Testing InnerLoop")
    print("="*60)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Device: {device}")

    # Load a task
    print("\nLoading dataset...")
    dataset = ARCDataset("data/arc-agi_training_challenges.json")
    task = dataset[0]
    print(f"Task ID: {task['task_id']}")
    print(f"Train pairs: {len(task['train_inputs'])}")

    # Create models (random weights)
    print("\nCreating models...")
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

    encoder.eval()
    renderer.eval()
    operators.eval()
    controller.train()

    # Run inner loop
    print("\nRunning inner loop (3 steps, fast test)...")
    inner_loop = InnerLoop(
        num_inner_steps=3,
        beam_size=4,
        max_operator_steps=2,
        binary_bonus_weight=0.5,  # Phase 5B: binary bonus
        device=device
    )

    adapted_controller, metrics = inner_loop.train_on_task(
        task, encoder, controller, operators, renderer,
        clone_controller=True,
        verbose=True
    )

    print("\n" + "="*60)
    print("✓ Inner loop test complete!")
    print("="*60)
    print(f"Mean reward: {metrics.mean_reward:.3f}")
    print(f"Best reward: {metrics.best_reward:.3f}")
    print(f"Success rate: {metrics.success_rate:.1%}")
    print(f"Policy loss: {metrics.policy_loss:.4f}")
