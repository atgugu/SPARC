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

# Convert relative paths to absolute paths for checkpoints and add dataset path
ARGS=()
DATASET_SPECIFIED=false

while [[ $# -gt 0 ]]; do
    case $1 in
        -a|--autoencoder)
            # Convert relative path to absolute
            if [[ "$2" != /* ]]; then
                ARGS+=("$1" "$SCRIPT_DIR/$2")
            else
                ARGS+=("$1" "$2")
            fi
            shift 2
            ;;
        -c|--controller)
            # Convert relative path to absolute
            if [[ "$2" != /* ]]; then
                ARGS+=("$1" "$SCRIPT_DIR/$2")
            else
                ARGS+=("$1" "$2")
            fi
            shift 2
            ;;
        -d|--dataset)
            DATASET_SPECIFIED=true
            # Convert relative path to absolute
            if [[ "$2" != /* ]]; then
                ARGS+=("$1" "$SCRIPT_DIR/$2")
            else
                ARGS+=("$1" "$2")
            fi
            shift 2
            ;;
        *)
            ARGS+=("$1")
            shift
            ;;
    esac
done

# Add default dataset path if not specified
if [ "$DATASET_SPECIFIED" = false ]; then
    ARGS+=("-d" "$SCRIPT_DIR/data/arc-agi_training_challenges.json")
fi

# Run the visualizer with processed arguments
npm run dev -- "${ARGS[@]}"
