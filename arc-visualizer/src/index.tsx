#!/usr/bin/env node
/**
 * CLI entry point for ARC visualizer
 */

import React from 'react';
import { render } from 'ink';
import { Command } from 'commander';
import App from './components/App.js';

const program = new Command();

program
  .name('arc-visualizer')
  .description('Beautiful CLI visualizer for ARC active learning adaptation')
  .version('1.0.0')
  .requiredOption('-a, --autoencoder <path>', 'Path to autoencoder checkpoint')
  .requiredOption('-c, --controller <path>', 'Path to controller checkpoint')
  .option('-t, --task-id <id>', 'Task ID to visualize')
  .option('-i, --task-index <n>', 'Task index (0-based)', parseInt)
  .option('-d, --dataset <path>', 'Dataset path', 'data/arc-agi_evaluation_challenges.json')
  .option('-s, --steps <n>', 'Adaptation steps', parseInt, 20)
  .option('-b, --beam-size <n>', 'Beam size', parseInt, 8)
  .parse();

const options = program.opts();

// Validate: must specify either task-id or task-index
if (!options.taskId && options.taskIndex === undefined) {
  console.error('Error: Either --task-id or --task-index must be specified');
  process.exit(1);
}

// Render the app
const { waitUntilExit } = render(
  <App
    autoencoder={options.autoencoder}
    controller={options.controller}
    taskId={options.taskId}
    taskIndex={options.taskIndex}
    dataset={options.dataset}
    adaptationSteps={options.steps}
    beamSize={options.beamSize}
  />
);

// Wait for app to exit
waitUntilExit()
  .then(() => {
    process.exit(0);
  })
  .catch((error) => {
    console.error('Error:', error);
    process.exit(1);
  });
