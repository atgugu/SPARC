/**
 * Grid rendering utilities for terminal display
 */

import chalk from 'chalk';
import type { Grid } from '../types/task.js';
import { getColorCode } from './colorPalette.js';

/**
 * Render a grid as colored ASCII art for terminal display.
 * Uses full block character (██) with background colors.
 */
export function renderGrid(grid: Grid, width?: number, height?: number): string[] {
  const h = height || grid.length;
  const w = width || (grid[0]?.length || 0);

  const rows: string[] = [];

  for (let i = 0; i < h; i++) {
    let row = '';
    for (let j = 0; w && j < w; j++) {
      const color = grid[i]?.[j] ?? 0;
      const colorCode = getColorCode(color);
      const char = '██'; // Full block
      row += chalk.hex(colorCode)(char);
    }
    rows.push(row);
  }

  return rows;
}

/**
 * Render grid with border and optional title.
 */
export function renderGridWithBorder(
  grid: Grid,
  width?: number,
  height?: number,
  title?: string
): string[] {
  const gridRows = renderGrid(grid, width, height);
  const w = width || (grid[0]?.length || 0);
  const borderWidth = w * 2 + 2;

  const result: string[] = [];

  // Top border
  if (title) {
    result.push(chalk.gray(`┌─${title.padEnd(borderWidth - 3, '─')}┐`));
  } else {
    result.push(chalk.gray('┌' + '─'.repeat(borderWidth) + '┐'));
  }

  // Grid rows
  for (const row of gridRows) {
    result.push(chalk.gray('│') + row + chalk.gray('│'));
  }

  // Bottom border
  result.push(chalk.gray('└' + '─'.repeat(borderWidth) + '┘'));

  return result;
}

/**
 * Calculate pixel accuracy between two grids.
 */
export function calculateAccuracy(pred: Grid, target: Grid): number {
  const h = Math.min(pred.length, target.length);
  if (h === 0) return 0;

  const w = Math.min(pred[0]?.length || 0, target[0]?.length || 0);
  if (w === 0) return 0;

  let correct = 0;
  let total = 0;

  for (let i = 0; i < h; i++) {
    for (let j = 0; j < w; j++) {
      if ((pred[i]?.[j] ?? 0) === (target[i]?.[j] ?? 0)) {
        correct++;
      }
      total++;
    }
  }

  return total > 0 ? (correct / total) * 100 : 0;
}

/**
 * Check if two grids are exactly equal.
 */
export function isExactMatch(pred: Grid, target: Grid): boolean {
  const h = Math.min(pred.length, target.length);
  if (h === 0) return false;

  const w = Math.min(pred[0]?.length || 0, target[0]?.length || 0);
  if (w === 0) return false;

  for (let i = 0; i < h; i++) {
    for (let j = 0; j < w; j++) {
      if ((pred[i]?.[j] ?? 0) !== (target[i]?.[j] ?? 0)) {
        return false;
      }
    }
  }

  return true;
}

/**
 * Generate a simple sparkline from an array of numbers.
 */
export function sparkline(data: number[]): string {
  if (data.length === 0) return '';

  const chars = ['▁', '▂', '▃', '▄', '▅', '▆', '▇', '█'];
  const min = Math.min(...data);
  const max = Math.max(...data);
  const range = max - min || 1;

  return data
    .map(value => {
      const normalized = (value - min) / range;
      const index = Math.floor(normalized * (chars.length - 1));
      return chars[index];
    })
    .join('');
}
