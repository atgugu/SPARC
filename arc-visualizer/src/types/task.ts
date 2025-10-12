/**
 * Type definitions for ARC tasks and grids
 */

export type Grid = number[][];

export interface TrainingPair {
  input: Grid;
  output: Grid;
  input_shape: [number, number];
  output_shape: [number, number];
}

export interface AdaptationState {
  status: 'idle' | 'loading' | 'loaded' | 'adapting' | 'converged' | 'complete' | 'error';
  taskId: string;
  numTrain: number;
  numTest: number;
  trainingPairs: TrainingPair[];

  // Adaptation progress
  step: number;
  maxSteps: number;
  timeElapsed: number;
  timeBudget: number;
  beamSize: number;

  // Current predictions
  currentPredictions: Grid[];

  // Metrics
  rewardHistory: number[];
  lossHistory: number[];
  trainAccuracy: number;
  trainSolved: number;

  // Test results
  testPredictions?: Grid[];
  testCorrect?: boolean[];
  testSuccess?: boolean;
  competitionScore?: number;

  // Error
  errorMessage?: string;
  errorDetails?: string;

  // Logs
  logs: string[];
}
