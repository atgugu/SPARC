/**
 * Event parsing utilities
 */

import type { PythonEvent } from '../types/events.js';

/**
 * Parse a JSON line from Python backend.
 */
export function parseEvent(line: string): PythonEvent | null {
  try {
    const trimmed = line.trim();
    if (!trimmed) return null;

    const event = JSON.parse(trimmed) as PythonEvent;
    return event;
  } catch (error) {
    // Not valid JSON, might be regular Python output
    return null;
  }
}

/**
 * Check if a line is a valid JSON event.
 */
export function isValidEvent(line: string): boolean {
  return parseEvent(line) !== null;
}
