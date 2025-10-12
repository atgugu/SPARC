#!/usr/bin/env python3
"""
CLI tool to visualize ARC tasks.
"""

import argparse
import sys
from pathlib import Path

# Add project to path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

import matplotlib
matplotlib.use('Agg')  # Non-interactive backend
import matplotlib.pyplot as plt

from arc_nodsl.data.loader import ARCDataset
from arc_nodsl.utils.viz import plot_task


def main():
    parser = argparse.ArgumentParser(description="Visualize ARC tasks")
    parser.add_argument(
        "--data",
        type=str,
        default="data/arc-agi_training_challenges.json",
        help="Path to challenges JSON file"
    )
    parser.add_argument(
        "--task_id",
        type=str,
        help="Specific task ID to visualize"
    )
    parser.add_argument(
        "--index",
        type=int,
        help="Task index to visualize (0-based)"
    )
    parser.add_argument(
        "--output",
        type=str,
        help="Output file path (e.g., task.png). If not provided, displays interactively."
    )
    parser.add_argument(
        "--max_examples",
        type=int,
        default=5,
        help="Maximum number of train examples to show"
    )

    args = parser.parse_args()

    # Load dataset
    print(f"Loading dataset from {args.data}...")
    dataset = ARCDataset(args.data)
    print(f"Loaded {len(dataset)} tasks")

    # Get task
    if args.task_id:
        task = dataset.get_task_by_id(args.task_id)
        if task is None:
            print(f"Error: Task ID '{args.task_id}' not found")
            sys.exit(1)
    elif args.index is not None:
        if args.index < 0 or args.index >= len(dataset):
            print(f"Error: Index {args.index} out of range (0-{len(dataset)-1})")
            sys.exit(1)
        task = dataset[args.index]
    else:
        print("Visualizing first task (use --task_id or --index to specify)")
        task = dataset[0]

    print(f"\nTask: {task['task_id']}")
    print(f"  Train pairs: {len(task['train_inputs'])}")
    print(f"  Test pairs: {len(task['test_inputs'])}")

    # Plot
    fig = plot_task(task, max_examples=args.max_examples)

    # Save or show
    if args.output:
        fig.savefig(args.output, dpi=150, bbox_inches='tight')
        print(f"\nSaved to {args.output}")
    else:
        print("\nDisplaying plot...")
        plt.show()


if __name__ == "__main__":
    main()
