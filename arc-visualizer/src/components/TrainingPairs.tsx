/**
 * Training Pairs component - displays all training pairs compactly
 */

import React from 'react';
import { Box, Text } from 'ink';
import ArcGrid from './ArcGrid.js';
import type { TrainingPair, Grid } from '../types/task.js';

interface TrainingPairsProps {
  pairs: TrainingPair[];
  predictions?: Grid[];
}

const TrainingPairs: React.FC<TrainingPairsProps> = ({ pairs, predictions }) => {
  return (
    <Box flexDirection="column" borderStyle="round" borderColor="blue" paddingX={1}>
      <Text bold color="blue">📝 Training Pairs</Text>

      {pairs.map((pair, idx) => {
        const pred = predictions?.[idx];

        return (
          <Box key={idx} marginTop={1} flexDirection="column">
            <Text dimColor>Pair {idx + 1}</Text>
            <Box flexDirection="row" gap={2}>
              {/* Input */}
              <Box flexDirection="column">
                <Text dimColor>Input</Text>
                <ArcGrid
                  grid={pair.input}
                  width={pair.input_shape[1]}
                  height={pair.input_shape[0]}
                  compact
                />
              </Box>

              <Text color="gray">→</Text>

              {/* Target output */}
              <Box flexDirection="column">
                <Text dimColor>Target</Text>
                <ArcGrid
                  grid={pair.output}
                  width={pair.output_shape[1]}
                  height={pair.output_shape[0]}
                  compact
                />
              </Box>

              {/* Prediction (if available) */}
              {pred && (
                <>
                  <Text color="gray">vs</Text>
                  <Box flexDirection="column">
                    <Text dimColor>Prediction</Text>
                    <ArcGrid
                      grid={pred}
                      width={pair.output_shape[1]}
                      height={pair.output_shape[0]}
                      showAccuracy
                      targetGrid={pair.output}
                      compact
                    />
                  </Box>
                </>
              )}
            </Box>
          </Box>
        );
      })}
    </Box>
  );
};

export default TrainingPairs;
