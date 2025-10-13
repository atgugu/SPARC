"""
Streaming Inner Loop - wraps InnerLoop to emit visualization events.
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

import torch
import torch.nn as nn
from typing import Dict, Optional, Tuple, List
from arc_nodsl.training.inner_loop import InnerLoop, InnerLoopMetrics
from arc_nodsl.inference.task_embed import build_task_embedding
from arc_nodsl.inference.latent_search import beam_search
from arc_nodsl.evaluation.metrics import exact_match
from event_emitter import EventEmitter


class StreamingInnerLoop(InnerLoop):
    """
    InnerLoop that emits events for real-time visualization.

    Wraps the standard InnerLoop and emits JSON events after each training step,
    allowing the JavaScript frontend to visualize the adaptation process.
    """

    def __init__(self, *args, emitter: Optional[EventEmitter] = None, emit_every: int = 2, time_budget: float = 60.0, **kwargs):
        """
        Initialize streaming inner loop.

        Args:
            *args: Forwarded to InnerLoop
            emitter: EventEmitter instance (creates new if None)
            emit_every: Emit events every N steps (default 2, to avoid spam)
            time_budget: Time budget in seconds for adaptation (default 60.0)
            **kwargs: Forwarded to InnerLoop
        """
        super().__init__(*args, **kwargs)
        self.emitter = emitter if emitter is not None else EventEmitter()
        self.emit_every = emit_every
        self.time_budget = time_budget

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
        Train on task with event emission.

        Same as parent InnerLoop.train_on_task(), but emits events for visualization.
        """
        # Emit adaptation start
        self.emitter.adaptation_start(
            max_steps=self.num_inner_steps,
            time_budget=self.time_budget,
            beam_size=self.beam_size
        )

        # Clone controller
        if clone_controller:
            adapted_controller = self._clone_controller(controller)
        else:
            adapted_controller = controller

        # Create optimizer
        optimizer = torch.optim.AdamW(
            adapted_controller.parameters(),
            lr=self.learning_rate
        )

        # Build task embedding
        train_pairs = []
        for i in range(len(task_data['train_inputs'])):
            train_pairs.append((
                task_data['train_inputs'][i],
                task_data['train_outputs'][i],
                task_data['train_shapes'][i]
            ))

        task_embed = build_task_embedding(
            train_pairs,
            encoder=None,
            device=self.device,
            analyze_operators=False
        )

        # Training loop
        all_rewards = []
        all_losses = []

        for step in range(self.num_inner_steps):
            # Sample a train pair
            pair_idx = step % len(train_pairs)
            input_grid = task_data['train_inputs'][pair_idx].to(self.device)
            target_grid = task_data['train_outputs'][pair_idx].to(self.device)
            input_shape = task_data['train_shapes'][pair_idx]['input']
            output_shape = task_data['train_shapes'][pair_idx]['output']

            # Emit step begin
            if step % self.emit_every == 0:
                self.emitter.step_begin(step + 1, pair_idx)

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
                collect_log_probs=True
            )

            # Compute rewards
            rewards = []
            h, w = output_shape
            for cand in candidates:
                fuzzy_reward = self.reward_computer.compute_reward(
                    cand.prediction,
                    target_grid,
                    cand.h,
                    cand.w,
                    constraints=task_embed['constraints'],
                    input_shape=input_shape
                )

                binary_bonus = 0.0
                if self.binary_bonus_weight > 0:
                    if exact_match(cand.prediction.cpu(), target_grid.cpu(), h, w):
                        binary_bonus = self.binary_bonus_weight

                total_reward = fuzzy_reward + binary_bonus
                rewards.append(total_reward)

            all_rewards.extend(rewards)

            # Extract log probs for REINFORCE
            valid_indices = [i for i, cand in enumerate(candidates) if cand.log_probs is not None]

            if len(valid_indices) == 0:
                continue

            log_probs_list = [candidates[i].log_probs for i in valid_indices]
            entropies_list = [candidates[i].entropies for i in valid_indices if candidates[i].entropies is not None]
            rewards_valid = [rewards[i] for i in valid_indices]
            rewards_tensor = torch.tensor(rewards_valid, device=self.device)

            # Stack log probs with padding
            max_steps_in_seq = max(len(lp) for lp in log_probs_list)
            beam_size_valid = len(log_probs_list)

            log_probs_per_step = []
            entropies_per_step = []
            mask = torch.zeros(beam_size_valid, max_steps_in_seq, device=self.device)

            for t in range(max_steps_in_seq):
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

            # Compute loss
            loss_dict = self.sequence_loss.compute_loss(
                log_probs_per_step,
                rewards_tensor,
                entropies=entropies_per_step if len(entropies_per_step) > 0 else None,
                mask=mask
            )

            loss = loss_dict['loss']
            all_losses.append(loss.item())

            # Backprop
            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(adapted_controller.parameters(), max_norm=1.0)
            optimizer.step()

            # Emit step complete with current predictions
            if (step + 1) % self.emit_every == 0 or (step + 1) == self.num_inner_steps:
                predictions, solved_count = self._get_current_predictions_and_accuracy(
                    adapted_controller,
                    task_data,
                    task_embed,
                    encoder,
                    operators,
                    renderer
                )

                mean_reward = sum(all_rewards[-10:]) / min(10, len(all_rewards)) if all_rewards else 0
                best_reward = max(all_rewards) if all_rewards else 0

                self.emitter.step_complete(
                    step=step + 1,
                    mean_reward=mean_reward,
                    best_reward=best_reward,
                    loss=loss.item(),
                    predictions=predictions,
                    accuracy=solved_count / len(task_data['train_inputs']),
                    train_solved_count=solved_count,
                    total_train=len(task_data['train_inputs'])
                )

                # Check for early convergence
                if solved_count == len(task_data['train_inputs']):
                    self.emitter.train_solved(step + 1, 1.0)
                    # Continue training to solidify (don't break early)

        # Compute final metrics
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

        # Emit adaptation complete
        self.emitter.adaptation_complete(
            final_accuracy=metrics.success_rate,
            num_steps=metrics.num_steps,
            converged=metrics.success_rate >= 1.0,
            stop_reason='train_solved' if metrics.success_rate >= 1.0 else 'max_steps'
        )

        return adapted_controller, metrics

    def _get_current_predictions_and_accuracy(
        self,
        controller: nn.Module,
        task_data: Dict,
        task_embed: Dict,
        encoder: nn.Module,
        operators: nn.Module,
        renderer: nn.Module
    ) -> Tuple[List[List[List[int]]], int]:
        """
        Get current predictions for all training pairs and count solved.

        Returns:
            (predictions as nested lists, number of solved pairs)
        """
        controller.eval()
        predictions = []
        solved_count = 0

        with torch.no_grad():
            for i in range(len(task_data['train_inputs'])):
                input_grid = task_data['train_inputs'][i].to(self.device)
                target_grid = task_data['train_outputs'][i].to(self.device)
                input_shape = task_data['train_shapes'][i]['input']
                output_shape = task_data['train_shapes'][i]['output']

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
                    pred = candidates[0].prediction
                    h, w = output_shape

                    # Check if solved
                    if exact_match(pred.cpu(), target_grid.cpu(), h, w):
                        solved_count += 1
                else:
                    pred = torch.zeros(30, 30, dtype=torch.long, device=self.device)
                    h, w = output_shape

                # Convert to nested list for JSON
                pred_list = pred[:h, :w].cpu().tolist()
                predictions.append(pred_list)

        controller.train()
        return predictions, solved_count
