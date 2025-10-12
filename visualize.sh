#!/usr/bin/env bash
#
# ARC Active Learning Visualizer - Wrapper Script
#
# This script allows you to run the visualizer from the project root directory.
# It automatically changes to the arc-visualizer subdirectory and forwards all arguments.
#
# Usage:
#   ./visualize.sh -a checkpoints/autoencoder_best.pt -c checkpoints/controller_best.pt -t 00576224
#

# Get the directory where this script is located
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

# Change to arc-visualizer directory
cd "$SCRIPT_DIR/arc-visualizer" || {
    echo "Error: arc-visualizer directory not found"
    exit 1
}

# Check if node_modules exists
if [ ! -d "node_modules" ]; then
    echo "Installing visualizer dependencies..."
    npm install || {
        echo "Error: npm install failed"
        exit 1
    }
fi

# Run the visualizer with all arguments forwarded
npm run dev -- "$@"
