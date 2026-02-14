"""
Comprehensive test of the data pipeline.
Tests loading, batching, augmentation on all training tasks.
"""

import sys
import torch
from pathlib import Path
from tqdm import tqdm

# Add project to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from arc_nodsl.data.loader import ARCDataset, compute_dataset_stats
from arc_nodsl.data.batching import create_dataloader
from arc_nodsl.data.augment import augment_task, Transform
from arc_nodsl.utils.profile import Timer


def test_loader():
    """Test basic loading of all datasets."""
    print("\n=== Testing Loader ===")

    datasets = {
        "train": "data/arc-agi_training_challenges.json",
        "eval": "data/arc-agi_evaluation_challenges.json",
        "test": "data/arc-agi_test_challenges.json",
    }

    stats_all = {}

    for name, path in datasets.items():
        print(f"\nLoading {name} dataset from {path}...")
        dataset = ARCDataset(path)
        print(f"  ✓ Loaded {len(dataset)} tasks")

        # Test accessing first task
        task = dataset[0]
        assert "task_id" in task
        assert "train_inputs" in task
        assert len(task["train_inputs"]) > 0
        print(f"  ✓ First task: {task['task_id']} with {len(task['train_inputs'])} train pairs")

        # Compute stats
        stats = compute_dataset_stats(dataset)
        stats_all[name] = stats
        print(f"  ✓ Stats: avg {stats['avg_train_pairs']:.1f} train pairs, "
              f"max grid {int(stats['max_h'])}×{int(stats['max_w'])}")

    return stats_all


def test_batching():
    """Test batching utilities."""
    print("\n=== Testing Batching ===")

    dataset = ARCDataset("data/arc-agi_training_challenges.json")

    # Test task-based batching
    print("\nTesting task-based batching...")
    loader = create_dataloader(dataset, batch_size=4, shuffle=False, collate_mode="tasks")

    n_batches = 0
    for batch in loader:
        assert len(batch["task_ids"]) <= 4
        assert len(batch["train_inputs"]) == len(batch["task_ids"])
        n_batches += 1
        if n_batches >= 3:
            break

    print(f"  ✓ Task batching works ({n_batches} batches tested)")

    # Test flat pair batching
    print("\nTesting flat pair batching...")
    loader_flat = create_dataloader(dataset, batch_size=16, shuffle=False, collate_mode="flat_pairs")

    n_batches = 0
    for batch in loader_flat:
        assert batch["inputs"].shape[0] > 0
        assert batch["inputs"].shape == batch["outputs"].shape
        assert batch["inputs"].dtype == torch.long
        n_batches += 1
        if n_batches >= 3:
            break

    print(f"  ✓ Flat pair batching works ({n_batches} batches tested)")


def test_augmentation():
    """Test augmentation on tasks."""
    print("\n=== Testing Augmentation ===")

    dataset = ARCDataset("data/arc-agi_training_challenges.json")
    task = dataset[0]

    print(f"\nOriginal task: {task['task_id']}")
    orig_inp = task["train_inputs"][0]
    orig_out = task["train_outputs"][0]
    orig_shape_in = task["train_shapes"][0]["input"]
    orig_shape_out = task["train_shapes"][0]["output"]

    print(f"  Input shape: {orig_shape_in}, Output shape: {orig_shape_out}")

    # Test all transforms
    for t in Transform:
        aug_task = augment_task(task, transform=t)
        assert aug_task["task_id"].startswith(task["task_id"])
        assert len(aug_task["train_inputs"]) == len(task["train_inputs"])

        # Check shapes are updated correctly
        aug_shape_in = aug_task["train_shapes"][0]["input"]
        if t in [Transform.ROT_90, Transform.ROT_270, Transform.FLIP_D1, Transform.FLIP_D2]:
            # Dimensions should be swapped
            assert aug_shape_in == (orig_shape_in[1], orig_shape_in[0])
        else:
            # Dimensions should be preserved
            assert aug_shape_in == orig_shape_in

    print(f"  ✓ All {len(Transform)} transforms work correctly")


def test_full_pipeline():
    """Test loading, batching, and iterating through all training tasks."""
    print("\n=== Testing Full Pipeline ===")

    dataset = ARCDataset("data/arc-agi_training_challenges.json")
    loader = create_dataloader(dataset, batch_size=8, shuffle=False, collate_mode="tasks")

    timer = Timer()
    total_tasks = 0
    total_pairs = 0

    print(f"\nIterating through all {len(dataset)} tasks...")

    for batch in tqdm(loader, desc="Processing batches"):
        with timer.measure("batch_processing"):
            n_tasks = len(batch["task_ids"])
            total_tasks += n_tasks

            for i in range(n_tasks):
                n_train = len(batch["train_inputs"][i])
                n_test = len(batch["test_inputs"][i])
                total_pairs += n_train + n_test

                # Validate tensors
                for inp in batch["train_inputs"][i]:
                    assert inp.dtype == torch.long
                    assert inp.shape == (30, 30)
                    assert inp.min() >= 0 and inp.max() <= 9

    print(f"\n  ✓ Processed {total_tasks} tasks, {total_pairs} pairs")
    timer.print_stats()


def test_gpu_loading():
    """Test loading data to GPU if available."""
    if not torch.cuda.is_available():
        print("\n=== Skipping GPU Test (CUDA not available) ===")
        return

    print("\n=== Testing GPU Loading ===")

    dataset = ARCDataset("data/arc-agi_training_challenges.json")
    loader = create_dataloader(dataset, batch_size=16, shuffle=False, collate_mode="flat_pairs")

    device = torch.device("cuda")
    timer = Timer()

    for i, batch in enumerate(loader):
        with timer.measure("cpu_to_gpu"):
            inputs = batch["inputs"].to(device)
            outputs = batch["outputs"].to(device)

        assert inputs.device.type == "cuda"
        assert outputs.device.type == "cuda"

        if i >= 10:
            break

    print(f"  ✓ GPU loading works")
    timer.print_stats()


def main():
    """Run all tests."""
    print("=" * 60)
    print("ARC Data Pipeline Tests")
    print("=" * 60)

    try:
        # Run tests
        stats = test_loader()
        test_batching()
        test_augmentation()
        test_full_pipeline()
        test_gpu_loading()

        print("\n" + "=" * 60)
        print("✓ ALL TESTS PASSED")
        print("=" * 60)

        # Print summary
        print("\nDataset Summary:")
        for name, s in stats.items():
            print(f"  {name:10s}: {s['total_tasks']:4d} tasks, "
                  f"avg {s['avg_train_pairs']:.1f} train pairs, "
                  f"avg grid {s['avg_h']:.1f}×{s['avg_w']:.1f}")

    except Exception as e:
        print(f"\n✗ TEST FAILED: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()
