# ARC Visualizer - Usage Guide

Beautiful real-time visualization of active learning adaptation for ARC tasks.

## Installation

### Prerequisites

- Node.js 18+ and npm
- Python 3.9+ with PyTorch
- Trained ARC model checkpoints

### Setup

```bash
cd arc-visualizer

# Install Node.js dependencies
npm install

# Verify installation
npm run test
```

## Quick Start

### Basic Usage

```bash
# Run in development mode
npm run dev -- \
  --autoencoder ../checkpoints/autoencoder_best.pt \
  --controller ../checkpoints/controller_best.pt \
  --task-id 00576224 \
  --steps 20

# Or build and run
npm run build
./dist/index.js -a ../checkpoints/autoencoder_best.pt \
                -c ../checkpoints/controller_best.pt \
                -t 00576224
```

### Command-Line Options

```
Options:
  -a, --autoencoder <path>   Path to autoencoder checkpoint (required)
  -c, --controller <path>    Path to controller checkpoint (required)
  -t, --task-id <id>         Task ID to visualize
  -i, --task-index <n>       Task index (0-based, alternative to task-id)
  -d, --dataset <path>       Dataset path (default: data/arc-agi_evaluation_challenges.json)
  -s, --steps <n>            Adaptation steps (default: 20)
  -b, --beam-size <n>        Beam size for search (default: 8)
  -h, --help                 Display help
  -V, --version              Display version
```

## Examples

### Visualize Specific Task

```bash
npm run dev -- \
  -a ../checkpoints/autoencoder_best.pt \
  -c ../checkpoints/controller_best.pt \
  --task-id 007bbfb7
```

### Visualize by Index

```bash
npm run dev -- \
  -a ../checkpoints/autoencoder_best.pt \
  -c ../checkpoints/controller_best.pt \
  --task-index 0
```

### Custom Adaptation Settings

```bash
# More adaptation steps, larger beam
npm run dev -- \
  -a ../checkpoints/autoencoder_best.pt \
  -c ../checkpoints/controller_best.pt \
  --task-id 00576224 \
  --steps 30 \
  --beam-size 16
```

### Use Training Dataset

```bash
npm run dev -- \
  -a ../checkpoints/autoencoder_best.pt \
  -c ../checkpoints/controller_best.pt \
  --task-id 00576224 \
  --dataset ../data/arc-agi_training_challenges.json
```

## UI Layout

The visualizer displays information in a beautiful terminal UI:

```
╭────────────────────────────────────────╮
│              ARC                       │
│    Active Learning Visualizer          │
│    Task: 00576224                      │
╰────────────────────────────────────────╯

┌─ Training Pairs ────┐   ┌─ Metrics ──────────┐
│ Pair 1:             │   │ Mean Reward: 0.852 │
│  Input → Target vs  │   │ Best Reward: 0.952 │
│  ██░░  → ████░      │   │ Train Accuracy: 100%│
│  ░░██  → ░███░      │   │                     │
│  ✓ 100%             │   │ Reward: ▁▂▃▅▇█     │
│                     │   │ Loss:   █▇▅▃▂▁     │
├─────────────────────┤   └─────────────────────┘
│ Adaptation Progress │   ┌─ Logs ─────────────┐
│ Step: 15/20 ████░░  │   │ ✓ Models loaded    │
│ Time: 12.3s/60.0s   │   │ ✓ Task loaded      │
│ Train Solved: 3/3 ✓ │   │ Starting adaptation│
│ Status: ✓ Converged!│   │ Step 10 complete   │
└─────────────────────┘   └─────────────────────┘

╔═════════════════════════════════════════╗
║  🎉 TASK FULLY SOLVED!                  ║
║  Test: 1/1 correct (100% comp. score)   ║
╚═════════════════════════════════════════╝
```

## What You'll See

### 1. Training Pairs Panel (Left Top)
- All training pairs from the task
- Input grids, target outputs, and current predictions
- Real-time accuracy for each pair
- Color-coded grids using full ARC palette

### 2. Adaptation Progress (Left Bottom)
- Current step and progress bar
- Time elapsed vs budget
- Number of training pairs solved
- Status indicator (adapting, converged, complete)

### 3. Metrics Dashboard (Right Top)
- Mean reward and best reward
- Mean loss
- Train accuracy percentage
- Sparkline charts for reward and loss trajectories

### 4. Logs Panel (Right Bottom)
- Real-time log messages from Python backend
- Latest 5 messages displayed

### 5. Test Results (Bottom, when complete)
- Final test predictions
- Success indicator
- Competition score

## Keyboard Controls

- **Ctrl+C**: Exit the visualizer

## Understanding the Output

### Status Indicators

- **Loading...** - Loading models and dataset
- **Adapting...** - Running gradient descent on training pairs
- **Converged!** - All training pairs solved
- **Complete** - Adaptation finished (may or may not have solved all pairs)
- **Error** - Something went wrong

### Accuracy Indicators

- **✓ 100%** - Exact match (green)
- **~ 87.5%** - Partial match (yellow)
- **Training pairs must be 100% before test prediction**

### Reward/Loss Trajectories

- Sparkline charts show trends over adaptation steps
- Reward should increase: ▁▂▃▅▇█
- Loss should decrease: █▇▅▃▂▁

## Troubleshooting

### "Module not found" errors

```bash
# Reinstall dependencies
rm -rf node_modules package-lock.json
npm install
```

### Python backend errors

```bash
# Check Python dependencies
cd ..
pip install torch torchvision

# Verify checkpoints exist
ls -lh checkpoints/
```

### "Task ID not found"

```bash
# List available tasks
cd ..
python3 -c "
from arc_nodsl.data.loader import ARCDataset
ds = ARCDataset('data/arc-agi_evaluation_challenges.json')
for i, task in enumerate(ds[:10]):
    print(f'{i}: {task[\"task_id\"]}')
"
```

### Terminal too small

Minimum terminal size: 120x30 characters

```bash
# Check terminal size
echo $COLUMNS x $LINES

# Resize terminal if needed
```

## Performance Tips

1. **Faster Adaptation**: Reduce --steps (e.g., --steps 10)
2. **Smaller Beam**: Reduce --beam-size (e.g., --beam-size 4)
3. **Skip Events**: Python backend emits every 2 steps by default

## Development

### Run in Watch Mode

```bash
npm run dev -- --help
```

### Build for Production

```bash
npm run build
```

### Type Checking

```bash
npx tsc --noEmit
```

## Next Steps

After visualizing a task:
1. Try different tasks to see adaptation behavior
2. Experiment with different adaptation steps
3. Compare successful vs failed adaptations
4. Use insights to improve controller training

## Support

For issues or questions:
- Check the main TRAINING_README.md
- Verify checkpoint compatibility (must use 11-color model)
- Ensure Python backend is accessible
