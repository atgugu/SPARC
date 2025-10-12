/**
 * Metrics Dashboard component - displays training metrics with sparklines
 */

import React from 'react';
import { Box, Text } from 'ink';
import { sparkline } from '../utils/gridRenderer.js';

interface MetricsDashboardProps {
  rewardHistory: number[];
  lossHistory: number[];
  trainAccuracy: number;
  trainSolved: number;
  totalTrain: number;
}

const MetricsDashboard: React.FC<MetricsDashboardProps> = ({
  rewardHistory,
  lossHistory,
  trainAccuracy,
  trainSolved,
  totalTrain,
}) => {
  const meanReward = rewardHistory.length > 0
    ? rewardHistory.reduce((a, b) => a + b, 0) / rewardHistory.length
    : 0;
  const bestReward = rewardHistory.length > 0 ? Math.max(...rewardHistory) : 0;

  const meanLoss = lossHistory.length > 0
    ? lossHistory.reduce((a, b) => a + b, 0) / lossHistory.length
    : 0;

  return (
    <Box flexDirection="column" borderStyle="round" borderColor="magenta" paddingX={1}>
      <Text bold color="magenta">📊 Metrics</Text>

      {/* Reward metrics */}
      <Box marginTop={1} flexDirection="column">
        <Text>
          Mean Reward: <Text bold color="green">{meanReward.toFixed(3)}</Text>
        </Text>
        <Text>
          Best Reward: <Text bold color="green">{bestReward.toFixed(3)}</Text>
        </Text>
        <Text>
          Mean Loss: <Text bold color="red">{meanLoss.toFixed(3)}</Text>
        </Text>
      </Box>

      {/* Train accuracy */}
      <Box marginTop={1}>
        <Text>
          Train Accuracy: <Text bold color={trainAccuracy >= 1.0 ? 'green' : 'yellow'}>
            {(trainAccuracy * 100).toFixed(1)}%
          </Text>
          {trainAccuracy >= 1.0 && <Text color="green"> ✓</Text>}
        </Text>
      </Box>

      {/* Sparklines */}
      {rewardHistory.length > 0 && (
        <Box marginTop={1} flexDirection="column">
          <Text color="gray">Reward Trajectory:</Text>
          <Text color="cyan">{sparkline(rewardHistory)}</Text>
        </Box>
      )}

      {lossHistory.length > 0 && (
        <Box marginTop={1} flexDirection="column">
          <Text color="gray">Loss Trajectory:</Text>
          <Text color="red">{sparkline(lossHistory)}</Text>
        </Box>
      )}

      {/* Stats summary */}
      <Box marginTop={1} flexDirection="column">
        <Text color="gray" dimColor>
          {rewardHistory.length} steps recorded
        </Text>
      </Box>
    </Box>
  );
};

export default MetricsDashboard;
