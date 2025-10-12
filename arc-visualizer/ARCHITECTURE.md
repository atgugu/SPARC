# ARC Visualizer Architecture

## Overview

The ARC Visualizer is a TypeScript/JavaScript CLI application that provides real-time visualization of active learning adaptation for ARC tasks. It follows a client-server architecture where the JavaScript frontend renders a beautiful terminal UI while the Python backend executes the machine learning code.

## Architecture Diagram

```
┌─────────────────────────────────────────────────────────────┐
│                  JavaScript Frontend                        │
│                  (Terminal UI - ink/React)                  │
│                                                              │
│  ┌──────────┐  ┌──────────────┐  ┌────────────────────┐   │
│  │ CLI Args │→ │ usePython    │→ │ React Components   │   │
│  │ Parser   │  │ Runner Hook  │  │ (Live Rendering)   │   │
│  └──────────┘  └──────┬───────┘  └────────────────────┘   │
│                       │                                      │
│                spawn()│                                      │
│                       ↓                                      │
└───────────────────────┼──────────────────────────────────────┘
                        │
                        │ stdout (JSON events)
                        │ stderr (errors)
                        │
┌───────────────────────┼──────────────────────────────────────┐
│                       ↓                                      │
│               Python Backend                                 │
│               (PyTorch ML Code)                              │
│                                                              │
│  ┌─────────────┐  ┌──────────────┐  ┌─────────────────┐   │
│  │ Load Models │→ │ Streaming    │→ │ Event Emitter   │   │
│  │ (Encoder/   │  │ InnerLoop    │  │ (JSON to stdout)│   │
│  │ Controller) │  │ (w/Callbacks)│  └─────────────────┘   │
│  └─────────────┘  └──────────────┘                          │
└─────────────────────────────────────────────────────────────┘
```

## Component Breakdown

### JavaScript Frontend (`arc-visualizer/src/`)

#### 1. Entry Point (`index.tsx`)
- Parses command-line arguments using `commander`
- Validates inputs
- Renders the main App component using `ink`

#### 2. Main App (`components/App.tsx`)
- Orchestrates all UI components
- Manages layout and state flow
- Displays header, panels, and results

#### 3. Custom Hook (`hooks/usePythonRunner.ts`)
- **Critical component** - bridges JS and Python
- Spawns Python subprocess using Node's `child_process`
- Parses JSON events from stdout line-by-line
- Maintains React state with latest data
- Handles cleanup on exit

#### 4. UI Components (`components/`)

**ArcGrid.tsx**
- Renders individual ARC grids
- Supports compact and full modes
- Shows accuracy comparisons
- Uses chalk for terminal colors

**TrainingPairs.tsx**
- Displays all training pairs
- Shows input → target → prediction flow
- Updates in real-time during adaptation

**AdaptationProgress.tsx**
- Progress bars for steps and time
- Status indicator with spinner
- Train solved counter

**MetricsDashboard.tsx**
- Reward and loss metrics
- Sparkline charts
- Train accuracy

#### 5. Utilities (`utils/`)

**gridRenderer.ts**
- Converts number[][] grids to colored ASCII art
- Uses full block characters (██) with chalk colors
- Calculates accuracy between grids
- Generates sparklines for metrics

**colorPalette.ts**
- Defines ARC's 11-color palette
- Maps color indices to hex codes

**eventParser.ts**
- Parses JSON events from Python
- Validates event structure

#### 6. Types (`types/`)

**events.ts**
- TypeScript interfaces for all Python events
- Type-safe event handling

**task.ts**
- AdaptationState interface
- Grid and TrainingPair types

### Python Backend (`arc-visualizer/python-backend/`)

#### 1. Main Script (`stream_active_learning.py`)
- CLI entry point for Python
- Loads models from checkpoints
- Loads task from dataset
- Orchestrates the adaptation process
- Emits events at key points

#### 2. Event Emitter (`event_emitter.py`)
- Utility class for emitting JSON events
- Methods for each event type:
  - `task_loaded()`
  - `adaptation_start()`
  - `step_complete()`
  - `train_solved()`
  - `test_complete()`
  - `error()`, `log()`
- Prints JSON to stdout (captured by JS)

#### 3. Streaming Inner Loop (`streaming_inner_loop.py`)
- Extends `arc_nodsl.training.inner_loop.InnerLoop`
- Adds callback system for visualization
- Emits events after each training step
- Captures current predictions
- Zero changes to core training logic

## Communication Protocol

### Event Stream Format

Python emits newline-delimited JSON to stdout:

```json
{"event": "task_loaded", "data": {...}, "timestamp": 0.0}
{"event": "adaptation_start", "data": {...}, "timestamp": 0.1}
{"event": "step_complete", "data": {...}, "timestamp": 1.2}
...
```

JavaScript reads line-by-line, parses JSON, updates React state.

### Event Types

1. **task_loaded** - Task metadata and training pairs
2. **adaptation_start** - Adaptation parameters
3. **step_begin** - Before each gradient step
4. **step_complete** - After each gradient step (with predictions)
5. **train_solved** - All training pairs solved
6. **adaptation_complete** - Adaptation finished
7. **test_start** - Beginning test prediction
8. **test_complete** - Test results
9. **log** - Info messages
10. **error** - Error messages

## Data Flow

```
User runs CLI command
  ↓
JavaScript parses arguments
  ↓
JavaScript spawns Python subprocess
  ↓
Python loads models (emits: log events)
  ↓
Python loads task (emits: task_loaded)
  ↓
Python starts adaptation (emits: adaptation_start)
  ↓
For each training step:
  Python runs beam search + gradient descent
  Python evaluates current predictions
  Python emits: step_complete (with grids)
  JavaScript updates UI in real-time
  ↓
Python finishes adaptation (emits: adaptation_complete)
  ↓
If train solved:
  Python predicts test pairs
  Python emits: test_complete
  ↓
JavaScript displays final results
  ↓
Python exits, JavaScript waits then exits
```

## Key Design Decisions

### 1. Why Separate Frontend/Backend?

- **Python**: Has all the ML code, models, checkpoints
- **JavaScript**: Better for terminal UIs (ink is excellent)
- **Communication**: Simple JSON over stdout (no network needed)

### 2. Why Streaming Events?

- Real-time updates during long-running adaptation
- No need for websockets or HTTP server
- Simple, robust, debuggable

### 3. Why ink (React for Terminal)?

- Component-based UI (reusable, maintainable)
- React hooks for state management
- Auto-reflow on terminal resize
- Professional polish (used by Gatsby, Yarn)

### 4. Why Minimal Python Changes?

- Core ML code stays unchanged
- Only wrapper scripts needed
- Streaming is opt-in (doesn't affect production)

### 5. Why TypeScript?

- Type safety for complex event handling
- Better IDE support
- Catches bugs at compile time

## Performance Considerations

### Event Frequency

- Emit every 2 steps by default (configurable)
- Avoids flooding stdout with JSON
- Balances responsiveness vs overhead

### Grid Serialization

- Grids sent as nested arrays (number[][])
- Only active regions sent (not full 30×30)
- Efficient JSON parsing

### Process Communication

- Subprocess stdout is buffered
- Line-by-line parsing (not waiting for full output)
- Non-blocking I/O

## Testing Strategy

### Unit Tests (Future)

- `gridRenderer.ts` - Test color mapping, accuracy calculation
- `eventParser.ts` - Test JSON parsing, error handling
- `colorPalette.ts` - Verify color codes

### Integration Tests

- Spawn Python with mock events
- Verify React state updates correctly
- Test error handling

### Manual Testing

- Test with real checkpoints on various tasks
- Test on different terminal sizes
- Test interruption (Ctrl+C)

## Extensibility

### Adding New Events

1. Add event type to `events.ts`
2. Add emit method to `event_emitter.py`
3. Add case to `handleEvent()` in `usePythonRunner.ts`
4. Update UI to display new data

### Adding New UI Components

1. Create component in `components/`
2. Import and use in `App.tsx`
3. Pass data from state

### Customizing Visualization

- Modify `gridRenderer.ts` for different grid styles
- Adjust colors in `colorPalette.ts`
- Change layout in `App.tsx`

## Dependencies

### JavaScript

- **ink** - React for terminal
- **chalk** - Terminal colors
- **commander** - CLI argument parsing
- **ink-gradient**, **ink-spinner**, **ink-big-text** - UI enhancements

### Python

- **torch** - Deep learning framework
- **arc_nodsl** - ARC solver implementation
- Standard library (json, sys, argparse)

## Future Enhancements

1. **Interactive Mode** - Pause/resume adaptation
2. **Save Visualizations** - Export as HTML/SVG
3. **Multiple Tasks** - Visualize batch of tasks
4. **Operator Inspection** - Show which operators are used
5. **Beam Visualization** - Show all beam candidates
6. **Performance Profiling** - Time per step, memory usage
7. **Comparison Mode** - Compare two models side-by-side
8. **Video Export** - Record terminal session as GIF/video

## Conclusion

The ARC Visualizer provides a beautiful, real-time window into the active learning adaptation process. Its modular architecture keeps the core ML code clean while enabling rich visualization through a modern terminal UI framework.
