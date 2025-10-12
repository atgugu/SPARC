#!/usr/bin/env python3
"""
CLI tool to list and search ARC tasks.
"""

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from arc_nodsl.data.loader import ARCDataset


def main():
    parser = argparse.ArgumentParser(description="List and search ARC tasks")
    parser.add_argument(
        "--data",
        type=str,
        default="data/arc-agi_training_challenges.json",
        help="Path to challenges JSON file"
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=20,
        help="Number of tasks to display"
    )
    parser.add_argument(
        "--search",
        type=str,
        help="Search for task ID containing this string"
    )
    parser.add_argument(
        "--min_train",
        type=int,
        help="Filter: minimum number of train pairs"
    )
    parser.add_argument(
        "--max_train",
        type=int,
        help="Filter: maximum number of train pairs"
    )
    parser.add_argument(
        "--stats",
        action="store_true",
        help="Show dataset statistics"
    )

    args = parser.parse_args()

    # Load dataset
    print(f"Loading dataset from {args.data}...")
    dataset = ARCDataset(args.data)
    print(f"Loaded {len(dataset)} tasks\n")

    # Show stats if requested
    if args.stats:
        from arc_nodsl.data.loader import compute_dataset_stats
        stats = compute_dataset_stats(dataset)
        print("=== Dataset Statistics ===")
        for k, v in stats.items():
            print(f"  {k:20s}: {v:.2f}" if isinstance(v, float) else f"  {k:20s}: {v}")
        print()

    # Filter and display tasks
    tasks_shown = 0

    for i in range(len(dataset)):
        if tasks_shown >= args.limit:
            break

        task = dataset[i]
        task_id = task["task_id"]
        n_train = len(task["train_inputs"])
        n_test = len(task["test_inputs"])

        # Apply filters
        if args.search and args.search not in task_id:
            continue

        if args.min_train and n_train < args.min_train:
            continue

        if args.max_train and n_train > args.max_train:
            continue

        # Get grid sizes
        train_shapes = task["train_shapes"]
        max_h = max(s["input"][0] for s in train_shapes)
        max_w = max(s["input"][1] for s in train_shapes)

        print(f"{i:4d}. {task_id}  |  Train: {n_train}  |  Test: {n_test}  |  Max grid: {max_h}×{max_w}")
        tasks_shown += 1

    if tasks_shown == 0:
        print("No tasks match the filters")
    else:
        print(f"\nShowing {tasks_shown} of {len(dataset)} tasks")


if __name__ == "__main__":
    main()
