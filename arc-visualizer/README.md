# ARC Visualizer

Beautiful CLI visualizer for ARC active learning adaptation.

## Installation

```bash
cd arc-visualizer
npm install
```

## Usage

```bash
# Development mode
npm run dev -- \
  --autoencoder ../checkpoints/autoencoder_best.pt \
  --controller ../checkpoints/controller_best.pt \
  --task-id 00576224 \
  --steps 20

# Build and run
npm run build
./dist/index.js -a ../checkpoints/autoencoder_best.pt \
                -c ../checkpoints/controller_best.pt \
                -t 00576224
```

## Features

- 🎨 Beautiful terminal UI with full ARC color palette
- ⚡ Real-time adaptation progress visualization
- 📊 Live metrics dashboard
- 🔄 Streaming predictions during training
- 🎯 Step-by-step convergence tracking

## Architecture

- **Frontend**: TypeScript + React (ink)
- **Backend**: Python + PyTorch
- **Communication**: Streaming JSON events over stdout
