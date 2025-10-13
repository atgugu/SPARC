/**
 * Main App component - orchestrates all UI components
 */

import React from 'react';
import { Box, Text } from 'ink';
import Gradient from 'ink-gradient';
import BigText from 'ink-big-text';
import TrainingPairs from './TrainingPairs.js';
import AdaptationProgress from './AdaptationProgress.js';
import MetricsDashboard from './MetricsDashboard.js';
import { usePythonRunner } from '../hooks/usePythonRunner.js';

interface AppProps {
  autoencoder: string;
  controller: string;
  taskId?: string;
  taskIndex?: number;
  dataset?: string;
  adaptationSteps: number;
  beamSize?: number;
  timeBudget?: number;
}

const App: React.FC<AppProps> = ({
  autoencoder,
  controller,
  taskId,
  taskIndex,
  dataset,
  adaptationSteps,
  beamSize,
  timeBudget,
}) => {
  const { state } = usePythonRunner({
    autoencoder,
    controller,
    taskId,
    taskIndex,
    dataset,
    adaptationSteps,
    beamSize,
    timeBudget,
    emitEvery: 2,
  });

  // Error state
  if (state.status === 'error') {
    return (
      <Box flexDirection="column" padding={1}>
        <Text bold color="red">✗ Error</Text>
        <Text color="red">{state.errorMessage}</Text>
        {state.errorDetails && (
          <Box marginTop={1}>
            <Text dimColor>{state.errorDetails}</Text>
          </Box>
        )}
      </Box>
    );
  }

  // Loading state
  if (state.status === 'idle' || state.status === 'loading') {
    return (
      <Box flexDirection="column" padding={1}>
        <Gradient name="rainbow">
          <BigText text="ARC" font="tiny" />
        </Gradient>
        <Text bold color="cyan">Active Learning Visualizer</Text>
        <Box marginTop={1}>
          <Text dimColor>Loading models and task...</Text>
        </Box>
      </Box>
    );
  }

  // Main UI
  return (
    <Box flexDirection="column" padding={1}>
      {/* Header */}
      <Box flexDirection="column" marginBottom={1}>
        <Gradient name="rainbow">
          <BigText text="ARC" font="tiny" />
        </Gradient>
        <Text bold color="cyan">Active Learning Visualizer</Text>
        <Text dimColor>Task: {state.taskId}</Text>
      </Box>

      {/* Main content */}
      <Box flexDirection="row" gap={2}>
        {/* Left column */}
        <Box flexDirection="column" width="50%" gap={1}>
          <TrainingPairs
            pairs={state.trainingPairs}
            predictions={state.currentPredictions}
          />

          <AdaptationProgress
            step={state.step}
            maxSteps={state.maxSteps}
            timeElapsed={state.timeElapsed}
            timeBudget={state.timeBudget}
            status={state.status}
            trainSolved={state.trainSolved}
            totalTrain={state.numTrain}
          />
        </Box>

        {/* Right column */}
        <Box flexDirection="column" width="50%" gap={1}>
          <MetricsDashboard
            rewardHistory={state.rewardHistory}
            lossHistory={state.lossHistory}
            trainAccuracy={state.trainAccuracy}
            trainSolved={state.trainSolved}
            totalTrain={state.numTrain}
          />

          {/* Logs */}
          <Box flexDirection="column" borderStyle="round" borderColor="gray" paddingX={1}>
            <Text bold color="gray">📜 Logs</Text>
            <Box flexDirection="column" marginTop={1}>
              {state.logs.slice(-5).map((log, i) => (
                <Text key={i} dimColor>{log}</Text>
              ))}
            </Box>
          </Box>
        </Box>
      </Box>

      {/* Test results (if complete) */}
      {state.status === 'complete' && state.testSuccess !== undefined && (
        <Box marginTop={1} flexDirection="column" borderStyle="double" borderColor="green" paddingX={1}>
          <Text bold color={state.testSuccess ? 'green' : 'yellow'}>
            {state.testSuccess ? '🎉 TASK FULLY SOLVED!' : '⚠ Partial Success'}
          </Text>
          {state.testCorrect && (
            <Text>
              Test: {state.testCorrect.filter(Boolean).length}/{state.testCorrect.length} correct
              {state.competitionScore !== undefined && (
                <Text> (Competition score: {(state.competitionScore * 100).toFixed(1)}%)</Text>
              )}
            </Text>
          )}
        </Box>
      )}

      {/* Final summary */}
      {state.status === 'complete' && (
        <Box marginTop={1}>
          <Text dimColor>
            Adaptation complete: {(state.trainAccuracy * 100).toFixed(1)}% train accuracy in {state.step} steps
          </Text>
        </Box>
      )}
    </Box>
  );
};

export default App;
