"""
Profiling utilities for performance monitoring.
"""

import time
import torch
from contextlib import contextmanager
from typing import Optional, Dict
import json
from pathlib import Path


class Timer:
    """Simple timer for measuring execution time."""

    def __init__(self):
        self.times = {}
        self.counts = {}

    @contextmanager
    def measure(self, name: str):
        """Context manager for timing a block of code."""
        start = time.perf_counter()
        try:
            yield
        finally:
            elapsed = time.perf_counter() - start
            if name not in self.times:
                self.times[name] = 0.0
                self.counts[name] = 0
            self.times[name] += elapsed
            self.counts[name] += 1

    def get_stats(self) -> Dict[str, Dict[str, float]]:
        """Get timing statistics."""
        stats = {}
        for name in self.times:
            total = self.times[name]
            count = self.counts[name]
            stats[name] = {
                "total": total,
                "count": count,
                "mean": total / count if count > 0 else 0.0,
            }
        return stats

    def print_stats(self):
        """Print timing statistics."""
        stats = self.get_stats()
        print("\n=== Timing Statistics ===")
        for name, stat in sorted(stats.items()):
            print(f"{name:30s}: {stat['mean']*1000:8.2f}ms (× {stat['count']:4d} = {stat['total']:8.2f}s)")

    def reset(self):
        """Reset all timers."""
        self.times = {}
        self.counts = {}


@contextmanager
def torch_profile(output_dir: str = "profiles",
                 wait: int = 1,
                 warmup: int = 1,
                 active: int = 3,
                 repeat: int = 1):
    """
    Context manager for PyTorch profiling.

    Args:
        output_dir: Directory to save profiler traces
        wait: Number of steps to wait before profiling
        warmup: Number of warmup steps
        active: Number of active profiling steps
        repeat: Number of times to repeat the cycle

    Usage:
        with torch_profile():
            for i in range(10):
                # training step
                pass
    """
    from torch.profiler import profile, schedule, tensorboard_trace_handler

    output_path = Path(output_dir)
    output_path.mkdir(exist_ok=True, parents=True)

    prof_schedule = schedule(wait=wait, warmup=warmup, active=active, repeat=repeat)

    with profile(
        activities=[
            torch.profiler.ProfilerActivity.CPU,
            torch.profiler.ProfilerActivity.CUDA,
        ] if torch.cuda.is_available() else [torch.profiler.ProfilerActivity.CPU],
        schedule=prof_schedule,
        on_trace_ready=tensorboard_trace_handler(str(output_path)),
        record_shapes=True,
        profile_memory=True,
        with_stack=True,
    ) as prof:
        yield prof

    print(f"Profiler traces saved to {output_path}")
    print(f"View with: tensorboard --logdir={output_path}")


def get_gpu_memory_stats() -> Optional[Dict[str, float]]:
    """Get current GPU memory statistics."""
    if not torch.cuda.is_available():
        return None

    stats = {
        "allocated_mb": torch.cuda.memory_allocated() / 1024**2,
        "reserved_mb": torch.cuda.memory_reserved() / 1024**2,
        "max_allocated_mb": torch.cuda.max_memory_allocated() / 1024**2,
        "max_reserved_mb": torch.cuda.max_memory_reserved() / 1024**2,
    }
    return stats


def print_gpu_memory():
    """Print GPU memory usage."""
    stats = get_gpu_memory_stats()
    if stats:
        print(f"GPU Memory: {stats['allocated_mb']:.1f}MB allocated, "
              f"{stats['reserved_mb']:.1f}MB reserved")


def reset_peak_memory_stats():
    """Reset peak memory statistics."""
    if torch.cuda.is_available():
        torch.cuda.reset_peak_memory_stats()


class MetricsLogger:
    """Simple logger for training metrics."""

    def __init__(self, log_file: Optional[str] = None):
        self.log_file = Path(log_file) if log_file else None
        self.metrics = []

    def log(self, step: int, metrics: Dict):
        """Log metrics for a step."""
        entry = {"step": step, **metrics}
        self.metrics.append(entry)

        if self.log_file:
            # Append to JSONL file
            with open(self.log_file, 'a') as f:
                f.write(json.dumps(entry) + '\n')

    def get_recent(self, n: int = 10) -> list:
        """Get the n most recent metric entries."""
        return self.metrics[-n:]

    def print_recent(self, n: int = 5):
        """Print recent metrics."""
        recent = self.get_recent(n)
        print(f"\n=== Last {len(recent)} Steps ===")
        for entry in recent:
            step = entry["step"]
            metrics_str = ", ".join(f"{k}={v:.4f}" for k, v in entry.items() if k != "step")
            print(f"Step {step:5d}: {metrics_str}")


if __name__ == "__main__":
    # Test profiling utilities
    print("Testing profiling utilities...")

    timer = Timer()

    # Simulate some work
    with timer.measure("task_A"):
        time.sleep(0.01)

    with timer.measure("task_B"):
        time.sleep(0.02)

    with timer.measure("task_A"):
        time.sleep(0.01)

    timer.print_stats()

    # GPU memory
    if torch.cuda.is_available():
        print("\n=== GPU Memory ===")
        print_gpu_memory()

        # Allocate some memory
        x = torch.randn(1000, 1000, device='cuda')
        print_gpu_memory()

    print("\n✓ Profiling tests passed!")
