/**
 * Type definitions for Python backend events
 */

export interface PythonEvent {
  event: string;
  data: any;
  timestamp: number;
}

export interface TaskLoadedEvent {
  event: 'task_loaded';
  data: {
    task_id: string;
    num_train: number;
    num_test: number;
    training_pairs: TrainingPair[];
  };
  timestamp: number;
}

export interface TrainingPair {
  input: number[][];
  output: number[][];
  input_shape: [number, number];
  output_shape: [number, number];
}

export interface AdaptationStartEvent {
  event: 'adaptation_start';
  data: {
    max_steps: number;
    time_budget: number;
    beam_size: number;
  };
  timestamp: number;
}

export interface StepCompleteEvent {
  event: 'step_complete';
  data: {
    step: number;
    mean_reward: number;
    best_reward: number;
    loss: number;
    predictions: number[][][];  // Array of grids
    accuracy: number;
    train_solved: number;
    total_train: number;
  };
  timestamp: number;
}

export interface TrainSolvedEvent {
  event: 'train_solved';
  data: {
    step: number;
    accuracy: number;
  };
  timestamp: number;
}

export interface AdaptationCompleteEvent {
  event: 'adaptation_complete';
  data: {
    final_accuracy: number;
    num_steps: number;
    converged: boolean;
    stop_reason: string;
  };
  timestamp: number;
}

export interface TestCompleteEvent {
  event: 'test_complete';
  data: {
    success: boolean;
    predictions: number[][][];
    correct: boolean[];
    competition_score: number;
  };
  timestamp: number;
}

export interface LogEvent {
  event: 'log';
  data: {
    message: string;
    level: string;
  };
  timestamp: number;
}

export interface ErrorEvent {
  event: 'error';
  data: {
    message: string;
    details: string;
  };
  timestamp: number;
}

export type TypedPythonEvent =
  | TaskLoadedEvent
  | AdaptationStartEvent
  | StepCompleteEvent
  | TrainSolvedEvent
  | AdaptationCompleteEvent
  | TestCompleteEvent
  | LogEvent
  | ErrorEvent;
