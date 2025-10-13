/**
 * Custom hook to run Python backend and manage state
 */

import { useState, useEffect, useRef } from 'react';
import { spawn, ChildProcess } from 'child_process';
import { fileURLToPath } from 'url';
import { dirname, resolve } from 'path';
import { parseEvent } from '../utils/eventParser.js';
import type { AdaptationState } from '../types/task.js';
import type { PythonEvent } from '../types/events.js';

const __filename = fileURLToPath(import.meta.url);
const __dirname = dirname(__filename);

interface PythonRunnerConfig {
  autoencoder: string;
  controller: string;
  taskId?: string;
  taskIndex?: number;
  dataset?: string;
  adaptationSteps: number;
  beamSize?: number;
  timeBudget?: number;
  emitEvery?: number;
}

export function usePythonRunner(config: PythonRunnerConfig) {
  const [state, setState] = useState<AdaptationState>({
    status: 'idle',
    taskId: '',
    numTrain: 0,
    numTest: 0,
    trainingPairs: [],
    step: 0,
    maxSteps: config.adaptationSteps,
    timeElapsed: 0,
    timeBudget: 60.0,
    beamSize: config.beamSize || 8,
    currentPredictions: [],
    rewardHistory: [],
    lossHistory: [],
    trainAccuracy: 0,
    trainSolved: 0,
    logs: [],
  });

  const processRef = useRef<ChildProcess | null>(null);
  const startTimeRef = useRef<number>(Date.now());

  useEffect(() => {
    // Build Python command - resolve path relative to package root
    const pythonPath = resolve(__dirname, '..', '..', 'python-backend', 'stream_active_learning.py');
    const args = [
      '--autoencoder_checkpoint', config.autoencoder,
      '--controller_checkpoint', config.controller,
      '--adaptation_steps', config.adaptationSteps.toString(),
    ];

    if (config.taskId) {
      args.push('--task_id', config.taskId);
    } else if (config.taskIndex !== undefined) {
      args.push('--task_index', config.taskIndex.toString());
    }

    if (config.dataset) {
      args.push('--dataset', config.dataset);
    }

    if (config.beamSize) {
      args.push('--beam_size', config.beamSize.toString());
    }

    if (config.timeBudget) {
      args.push('--time_budget', config.timeBudget.toString());
    }

    if (config.emitEvery) {
      args.push('--emit_every', config.emitEvery.toString());
    }

    // Update state to loading
    setState(prev => ({ ...prev, status: 'loading' }));

    // Spawn Python process
    const pythonProcess = spawn('python3', [pythonPath, ...args], {
      cwd: process.cwd(),
    });

    processRef.current = pythonProcess;
    startTimeRef.current = Date.now();

    // Handle stdout (JSON events)
    pythonProcess.stdout.on('data', (data) => {
      const lines = data.toString().split('\n');

      for (const line of lines) {
        if (!line.trim()) continue;

        const event = parseEvent(line);
        if (event) {
          handleEvent(event, setState, startTimeRef.current);
        }
      }
    });

    // Handle stderr (errors)
    pythonProcess.stderr.on('data', (data) => {
      const message = data.toString();
      setState(prev => ({
        ...prev,
        logs: [...prev.logs, `[ERROR] ${message}`],
      }));
    });

    // Handle process exit
    pythonProcess.on('close', (code) => {
      if (code !== 0 && code !== null) {
        setState(prev => ({
          ...prev,
          status: 'error',
          errorMessage: `Python process exited with code ${code}`,
        }));
      } else {
        // Use functional setState to avoid stale closure
        setState(prev => {
          if (prev.status !== 'complete' && prev.status !== 'error') {
            return { ...prev, status: 'complete' };
          }
          return prev;
        });
      }
    });

    // Cleanup
    return () => {
      if (processRef.current) {
        processRef.current.kill();
      }
    };
  }, [
    config.autoencoder,
    config.controller,
    config.taskId,
    config.taskIndex,
    config.dataset,
    config.adaptationSteps,
    config.beamSize,
    config.timeBudget,
    config.emitEvery,
  ]);

  return { state };
}

function handleEvent(
  event: PythonEvent,
  setState: React.Dispatch<React.SetStateAction<AdaptationState>>,
  startTime: number
) {
  switch (event.event) {
    case 'task_loaded':
      setState(prev => ({
        ...prev,
        status: 'loaded',
        taskId: event.data.task_id,
        numTrain: event.data.num_train,
        numTest: event.data.num_test,
        trainingPairs: event.data.training_pairs,
      }));
      break;

    case 'adaptation_start':
      setState(prev => ({
        ...prev,
        status: 'adapting',
        maxSteps: event.data.max_steps,
        timeBudget: event.data.time_budget,
        beamSize: event.data.beam_size,
      }));
      break;

    case 'step_complete':
      setState(prev => ({
        ...prev,
        step: event.data.step,
        currentPredictions: event.data.predictions,
        rewardHistory: [...prev.rewardHistory, event.data.mean_reward],
        lossHistory: [...prev.lossHistory, event.data.loss],
        trainAccuracy: event.data.accuracy,
        trainSolved: event.data.train_solved,
        timeElapsed: event.timestamp,
      }));
      break;

    case 'train_solved':
      setState(prev => ({
        ...prev,
        status: 'converged',
        trainAccuracy: 1.0,
      }));
      break;

    case 'adaptation_complete':
      setState(prev => ({
        ...prev,
        trainAccuracy: event.data.final_accuracy,
      }));
      break;

    case 'test_start':
      setState(prev => ({
        ...prev,
        status: 'adapting',
        logs: [...prev.logs, 'Predicting test outputs...'],
      }));
      break;

    case 'test_complete':
      setState(prev => ({
        ...prev,
        status: 'complete',
        testPredictions: event.data.predictions,
        testCorrect: event.data.correct,
        testSuccess: event.data.success,
        competitionScore: event.data.competition_score,
      }));
      break;

    case 'log':
      setState(prev => ({
        ...prev,
        logs: [...prev.logs, event.data.message],
      }));
      break;

    case 'error':
      setState(prev => ({
        ...prev,
        status: 'error',
        errorMessage: event.data.message,
        errorDetails: event.data.details,
        logs: [...prev.logs, `[ERROR] ${event.data.message}`],
      }));
      break;
  }
}
