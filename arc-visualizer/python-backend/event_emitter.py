"""
Event emitter for streaming JSON events to JavaScript frontend.
"""

import json
import sys
import time
from typing import Dict, Any, List


class EventEmitter:
    """Emit JSON events to stdout for JavaScript frontend consumption."""

    def __init__(self, enabled: bool = True):
        self.enabled = enabled
        self.start_time = time.time()

    def emit(self, event: str, data: Dict[str, Any]):
        """
        Emit a JSON event to stdout.

        Args:
            event: Event type name
            data: Event data dictionary
        """
        if not self.enabled:
            return

        event_obj = {
            "event": event,
            "data": data,
            "timestamp": time.time() - self.start_time
        }

        # Print JSON to stdout (JS will parse this line-by-line)
        print(json.dumps(event_obj), flush=True)

    def task_loaded(self, task_id: str, num_train: int, num_test: int, train_grids: List[Dict]):
        """Emit task loaded event with training pairs."""
        self.emit("task_loaded", {
            "task_id": task_id,
            "num_train": num_train,
            "num_test": num_test,
            "training_pairs": train_grids
        })

    def adaptation_start(self, max_steps: int, time_budget: float, beam_size: int):
        """Emit adaptation start event."""
        self.emit("adaptation_start", {
            "max_steps": max_steps,
            "time_budget": time_budget,
            "beam_size": beam_size
        })

    def step_begin(self, step: int, pair_idx: int):
        """Emit step begin event."""
        self.emit("step_begin", {
            "step": step,
            "pair_idx": pair_idx
        })

    def step_complete(self,
                     step: int,
                     mean_reward: float,
                     best_reward: float,
                     loss: float,
                     predictions: List[List[List[int]]],
                     accuracy: float,
                     train_solved_count: int,
                     total_train: int):
        """Emit step complete event with predictions and metrics."""
        self.emit("step_complete", {
            "step": step,
            "mean_reward": mean_reward,
            "best_reward": best_reward,
            "loss": loss,
            "predictions": predictions,
            "accuracy": accuracy,
            "train_solved": train_solved_count,
            "total_train": total_train
        })

    def train_solved(self, step: int, accuracy: float):
        """Emit train solved event (early convergence)."""
        self.emit("train_solved", {
            "step": step,
            "accuracy": accuracy
        })

    def adaptation_complete(self,
                          final_accuracy: float,
                          num_steps: int,
                          converged: bool,
                          stop_reason: str):
        """Emit adaptation complete event."""
        self.emit("adaptation_complete", {
            "final_accuracy": final_accuracy,
            "num_steps": num_steps,
            "converged": converged,
            "stop_reason": stop_reason
        })

    def test_start(self, num_test: int):
        """Emit test prediction start event."""
        self.emit("test_start", {
            "num_test": num_test
        })

    def test_complete(self,
                     success: bool,
                     predictions: List[List[List[int]]],
                     correct: List[bool],
                     competition_score: float):
        """Emit test complete event."""
        self.emit("test_complete", {
            "success": success,
            "predictions": predictions,
            "correct": correct,
            "competition_score": competition_score
        })

    def error(self, message: str, details: str = ""):
        """Emit error event."""
        self.emit("error", {
            "message": message,
            "details": details
        })

    def log(self, message: str, level: str = "info"):
        """Emit log event."""
        self.emit("log", {
            "message": message,
            "level": level
        })
