/**
 * ARC Grid component - displays a single ARC grid with optional border
 */

import React from 'react';
import { Box, Text } from 'ink';
import { renderGrid, calculateAccuracy, isExactMatch } from '../utils/gridRenderer.js';
import type { Grid } from '../types/task.js';

interface ArcGridProps {
  grid: Grid;
  width?: number;
  height?: number;
  title?: string;
  showAccuracy?: boolean;
  targetGrid?: Grid;
  compact?: boolean;
}

const ArcGrid: React.FC<ArcGridProps> = ({
  grid,
  width,
  height,
  title,
  showAccuracy = false,
  targetGrid,
  compact = false,
}) => {
  const gridRows = renderGrid(grid, width, height);

  // Calculate accuracy if target provided
  let accuracy = 0;
  let exactMatch = false;
  if (showAccuracy && targetGrid) {
    accuracy = calculateAccuracy(grid, targetGrid);
    exactMatch = isExactMatch(grid, targetGrid);
  }

  if (compact) {
    // Compact mode: no border
    return (
      <Box flexDirection="column">
        {title && <Text dimColor>{title}</Text>}
        {gridRows.map((row, i) => (
          <Text key={i}>{row}</Text>
        ))}
        {showAccuracy && targetGrid && (
          <Text color={exactMatch ? 'green' : 'yellow'}>
            {exactMatch ? '✓ ' : '~ '}
            {accuracy.toFixed(1)}%
          </Text>
        )}
      </Box>
    );
  }

  // Full mode: with border
  const w = width || (grid[0]?.length || 0);
  const borderWidth = w * 2;

  return (
    <Box flexDirection="column">
      {title && <Text color="cyan" bold>{title}</Text>}
      <Box flexDirection="column" borderStyle="round" borderColor="gray" paddingX={1}>
        {gridRows.map((row, i) => (
          <Text key={i}>{row}</Text>
        ))}
      </Box>
      {showAccuracy && targetGrid && (
        <Text color={exactMatch ? 'green' : 'yellow'}>
          {exactMatch ? '✓ Exact match' : `~ ${accuracy.toFixed(1)}% accurate`}
        </Text>
      )}
    </Box>
  );
};

export default ArcGrid;
