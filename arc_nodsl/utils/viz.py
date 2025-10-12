"""
Visualization utilities for ARC grids, slots, and operators.
"""

import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.colors import ListedColormap
import numpy as np
import torch
from typing import List, Optional, Tuple


# ARC color palette (0-9)
ARC_COLORS = [
    '#000000',  # 0: black
    '#0074D9',  # 1: blue
    '#FF4136',  # 2: red
    '#2ECC40',  # 3: green
    '#FFDC00',  # 4: yellow
    '#AAAAAA',  # 5: grey
    '#F012BE',  # 6: magenta
    '#FF851B',  # 7: orange
    '#7FDBFF',  # 8: light blue
    '#870C25',  # 9: maroon
]

ARC_CMAP = ListedColormap(ARC_COLORS)


def plot_grid(grid: torch.Tensor,
              ax: Optional[plt.Axes] = None,
              title: str = "",
              show_grid: bool = True,
              h: Optional[int] = None,
              w: Optional[int] = None) -> plt.Axes:
    """
    Plot a single ARC grid.

    Args:
        grid: Tensor [H, W] with values 0-9
        ax: Matplotlib axis (creates new if None)
        title: Title for the plot
        show_grid: Whether to show grid lines
        h, w: Original height/width (crop padding if provided)

    Returns:
        matplotlib Axes
    """
    if ax is None:
        fig, ax = plt.subplots(1, 1, figsize=(6, 6))

    # Convert to numpy
    if isinstance(grid, torch.Tensor):
        grid_np = grid.cpu().numpy()
    else:
        grid_np = np.array(grid)

    # Crop to original size if provided
    if h is not None and w is not None:
        grid_np = grid_np[:h, :w]

    # Plot
    ax.imshow(grid_np, cmap=ARC_CMAP, vmin=0, vmax=9, interpolation='nearest')
    ax.set_title(title, fontsize=12, fontweight='bold')

    # Grid lines
    if show_grid:
        h_actual, w_actual = grid_np.shape
        ax.set_xticks(np.arange(-0.5, w_actual, 1), minor=True)
        ax.set_yticks(np.arange(-0.5, h_actual, 1), minor=True)
        ax.grid(which="minor", color="white", linestyle='-', linewidth=0.5, alpha=0.5)

    ax.set_xticks([])
    ax.set_yticks([])

    return ax


def plot_task(task_data: dict,
              max_examples: int = 3,
              figsize: Tuple[int, int] = (16, 10)) -> plt.Figure:
    """
    Plot a full ARC task (train + test pairs).

    Args:
        task_data: Task dict from ARCDataset
        max_examples: Maximum number of train examples to show
        figsize: Figure size

    Returns:
        matplotlib Figure
    """
    task_id = task_data["task_id"]
    train_inputs = task_data["train_inputs"]
    train_outputs = task_data["train_outputs"]
    test_inputs = task_data["test_inputs"]
    test_outputs = task_data["test_outputs"]
    train_shapes = task_data["train_shapes"]
    test_shapes = task_data["test_shapes"]

    n_train = min(len(train_inputs), max_examples)
    n_test = len(test_inputs)

    # Calculate grid layout
    n_rows = max(n_train, n_test)
    n_cols = 4  # train_in, train_out, test_in, test_out

    fig, axes = plt.subplots(n_rows, n_cols, figsize=figsize)
    if n_rows == 1:
        axes = axes.reshape(1, -1)

    fig.suptitle(f"Task: {task_id}", fontsize=16, fontweight='bold')

    # Plot train pairs
    for i in range(n_train):
        h_in, w_in = train_shapes[i]["input"]
        h_out, w_out = train_shapes[i]["output"]

        plot_grid(train_inputs[i], ax=axes[i, 0],
                 title=f"Train {i+1} Input", h=h_in, w=w_in)
        plot_grid(train_outputs[i], ax=axes[i, 1],
                 title=f"Train {i+1} Output", h=h_out, w=w_out)

    # Clear unused train rows
    for i in range(n_train, n_rows):
        axes[i, 0].axis('off')
        axes[i, 1].axis('off')

    # Plot test pairs
    for i in range(n_test):
        h_in, w_in = test_shapes[i]["input"]

        plot_grid(test_inputs[i], ax=axes[i, 2],
                 title=f"Test {i+1} Input", h=h_in, w=w_in)

        if test_outputs[i] is not None:
            h_out, w_out = test_shapes[i]["output"]
            plot_grid(test_outputs[i], ax=axes[i, 3],
                     title=f"Test {i+1} Output", h=h_out, w=w_out)
        else:
            axes[i, 3].text(0.5, 0.5, "?", ha='center', va='center',
                          fontsize=40, color='gray')
            axes[i, 3].set_title(f"Test {i+1} Output (Unknown)")
            axes[i, 3].axis('off')

    # Clear unused test rows
    for i in range(n_test, n_rows):
        axes[i, 2].axis('off')
        axes[i, 3].axis('off')

    plt.tight_layout()
    return fig


def plot_prediction(input_grid: torch.Tensor,
                   true_output: Optional[torch.Tensor],
                   pred_output: torch.Tensor,
                   input_shape: Tuple[int, int],
                   output_shape: Tuple[int, int],
                   title: str = "Prediction",
                   figsize: Tuple[int, int] = (15, 5)) -> plt.Figure:
    """
    Plot input, ground truth, and prediction side-by-side.

    Args:
        input_grid: Input tensor [H, W]
        true_output: Ground truth tensor [H, W] (optional)
        pred_output: Prediction tensor [H, W]
        input_shape: Original (h, w) of input
        output_shape: Original (h, w) of output
        title: Figure title
        figsize: Figure size

    Returns:
        matplotlib Figure
    """
    n_plots = 3 if true_output is not None else 2
    fig, axes = plt.subplots(1, n_plots, figsize=figsize)

    if n_plots == 2:
        axes = [axes[0], None, axes[1]]

    # Plot input
    h_in, w_in = input_shape
    plot_grid(input_grid, ax=axes[0], title="Input", h=h_in, w=w_in)

    # Plot ground truth
    if true_output is not None:
        h_out, w_out = output_shape
        plot_grid(true_output, ax=axes[1], title="Ground Truth", h=h_out, w=w_out)

    # Plot prediction
    h_out, w_out = output_shape
    plot_grid(pred_output, ax=axes[-1], title="Prediction", h=h_out, w=w_out)

    # Compute accuracy if ground truth available
    if true_output is not None:
        pred_crop = pred_output[:h_out, :w_out]
        true_crop = true_output[:h_out, :w_out]
        acc = (pred_crop == true_crop).float().mean().item()
        axes[-1].set_xlabel(f"Accuracy: {acc*100:.1f}%", fontsize=12, fontweight='bold')

    fig.suptitle(title, fontsize=14, fontweight='bold')
    plt.tight_layout()
    return fig


def plot_slots(slots_z: torch.Tensor,
               slots_m: torch.Tensor,
               input_grid: Optional[torch.Tensor] = None,
               max_slots: int = 8,
               figsize: Tuple[int, int] = (16, 8)) -> plt.Figure:
    """
    Visualize slot attention masks.

    Args:
        slots_z: Slot features [K, D]
        slots_m: Slot masks [K, H, W]
        input_grid: Original input [H, W] (optional, for overlay)
        max_slots: Maximum slots to display
        figsize: Figure size

    Returns:
        matplotlib Figure
    """
    K = min(slots_m.shape[0], max_slots)
    n_cols = 4
    n_rows = (K + n_cols - 1) // n_cols

    fig, axes = plt.subplots(n_rows, n_cols, figsize=figsize)
    axes = axes.flatten() if n_rows > 1 else [axes] if n_cols == 1 else axes

    fig.suptitle("Slot Attention Masks", fontsize=14, fontweight='bold')

    for i in range(K):
        mask = slots_m[i].cpu().numpy()

        # Show mask as heatmap
        axes[i].imshow(mask, cmap='viridis', interpolation='nearest')
        axes[i].set_title(f"Slot {i}", fontsize=10)
        axes[i].axis('off')

        # Optionally overlay on input
        if input_grid is not None:
            # Create alpha overlay
            pass  # TODO: implement overlay

    # Hide unused subplots
    for i in range(K, len(axes)):
        axes[i].axis('off')

    plt.tight_layout()
    return fig


def plot_operator_sequence(inputs: List[torch.Tensor],
                          op_names: List[str],
                          title: str = "Operator Sequence",
                          figsize: Tuple[int, int] = (18, 4)) -> plt.Figure:
    """
    Visualize a sequence of operator applications.

    Args:
        inputs: List of intermediate grids after each operator
        op_names: List of operator names
        title: Figure title
        figsize: Figure size

    Returns:
        matplotlib Figure
    """
    n_steps = len(inputs)
    fig, axes = plt.subplots(1, n_steps, figsize=figsize)

    if n_steps == 1:
        axes = [axes]

    fig.suptitle(title, fontsize=14, fontweight='bold')

    for i, (grid, op_name) in enumerate(zip(inputs, op_names)):
        plot_grid(grid, ax=axes[i], title=f"Step {i}: {op_name}")

    plt.tight_layout()
    return fig


def save_figure(fig: plt.Figure, path: str, dpi: int = 150):
    """Save a figure to disk."""
    fig.savefig(path, dpi=dpi, bbox_inches='tight')
    print(f"Saved figure to {path}")


if __name__ == "__main__":
    # Test visualization
    from arc_nodsl.data.loader import ARCDataset

    print("Testing visualization utilities...")

    dataset = ARCDataset("data/arc-agi_training_challenges.json")
    task = dataset[0]

    print(f"Visualizing task: {task['task_id']}")

    # Plot full task
    fig = plot_task(task)
    plt.show()

    print("✓ Visualization tests passed!")
