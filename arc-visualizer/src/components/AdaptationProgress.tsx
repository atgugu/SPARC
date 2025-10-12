/**
 * Adaptation Progress component - shows training progress with bars and metrics
 */

import React from 'react';
import { Box, Text } from 'ink';
import Spinner from 'ink-spinner';

interface AdaptationProgressProps {
  step: number;
  maxSteps: number;
  timeElapsed: number;
  timeBudget: number;
  status: 'idle' | 'loading' | 'loaded' | 'adapting' | 'converged' | 'complete' | 'error';
  trainSolved: number;
  totalTrain: number;
}

const AdaptationProgress: React.FC<AdaptationProgressProps> = ({
  step,
  maxSteps,
  timeElapsed,
  timeBudget,
  status,
  trainSolved,
  totalTrain,
}) => {
  const stepProgress = maxSteps > 0 ? step / maxSteps : 0;
  const timeProgress = timeBudget > 0 ? timeElapsed / timeBudget : 0;

  // Create progress bar
  const createBar = (progress: number, width: number = 30): string => {
    const filled = Math.floor(progress * width);
    const empty = width - filled;
    return '█'.repeat(filled) + '░'.repeat(empty);
  };

  // Status indicator
  const getStatusDisplay = () => {
    switch (status) {
      case 'loading':
        return <Text color="blue"><Spinner type="dots" /> Loading models...</Text>;
      case 'loaded':
        return <Text color="cyan">✓ Task loaded</Text>;
      case 'adapting':
        return <Text color="green"><Spinner type="dots" /> Adapting...</Text>;
      case 'converged':
        return <Text color="green">✓ Converged!</Text>;
      case 'complete':
        return <Text color="cyan">✓ Complete</Text>;
      case 'error':
        return <Text color="red">✗ Error</Text>;
      default:
        return <Text color="gray">Idle</Text>;
    }
  };

  return (
    <Box flexDirection="column" borderStyle="round" borderColor="cyan" paddingX={1}>
      <Text bold color="cyan">⚡ Adaptation Progress</Text>

      {/* Step progress */}
      <Box marginTop={1} flexDirection="column">
        <Text>
          Step: <Text bold>{step}</Text>/{maxSteps} {' '}
          <Text color="gray">({(stepProgress * 100).toFixed(0)}%)</Text>
        </Text>
        <Text color="cyan">{createBar(stepProgress)}</Text>
      </Box>

      {/* Time progress */}
      <Box marginTop={1} flexDirection="column">
        <Text>
          Time: <Text bold>{timeElapsed.toFixed(1)}s</Text>/{timeBudget}s {' '}
          <Text color="gray">({(timeProgress * 100).toFixed(0)}%)</Text>
        </Text>
        <Text color="yellow">{createBar(timeProgress)}</Text>
      </Box>

      {/* Train solved */}
      <Box marginTop={1}>
        <Text>
          Train Solved: <Text bold color={trainSolved === totalTrain ? 'green' : 'yellow'}>
            {trainSolved}/{totalTrain}
          </Text>
          {trainSolved === totalTrain && <Text color="green"> ✓</Text>}
        </Text>
      </Box>

      {/* Status */}
      <Box marginTop={1}>
        <Text>Status: </Text>
        {getStatusDisplay()}
      </Box>
    </Box>
  );
};

export default AdaptationProgress;
