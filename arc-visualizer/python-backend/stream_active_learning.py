#!/usr/bin/env python3
"""
Stream active learning adaptation to JavaScript visualizer.

This script loads models, runs active adaptation on a task, and emits
JSON events to stdout for the JavaScript frontend to consume.
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

import torch
import argparse
from typing import Tuple

from arc_nodsl.data.loader import ARCDataset
from arc_nodsl.models.slots import SlotEncoder
from arc_nodsl.models.renderer import SlotRenderer
from arc_nodsl.models.operators import OperatorLibrary
from arc_nodsl.models.controller import Controller
from arc_nodsl.inference.task_embed import build_task_embedding
from arc_nodsl.inference.latent_search import beam_search
from arc_nodsl.evaluation.metrics import exact_match

from event_emitter import EventEmitter
from streaming_inner_loop import StreamingInnerLoop


def load_pretrained_autoencoder(
    checkpoint_path: str,
    device: torch.device
) -> Tuple[SlotEncoder, SlotRenderer]:
    """Load pretrained encoder and renderer from checkpoint."""
    checkpoint = torch.load(checkpoint_path, map_location=device)

    # Create models
    encoder = SlotEncoder(
        num_slots=8,
        d_color=16,
        d_feat=64,
        d_slot=128,
        num_iters=3,
        H=30,
        W=30
    ).to(device)

    renderer = SlotRenderer(
        d_slot=128,
        d_hidden=64,
        H=30,
        W=30,
        use_mask=True
    ).to(device)

    # Load from AutoEncoder checkpoint
    full_state = checkpoint['model_state_dict']

    # Split into encoder and renderer state dicts
    encoder_state = {}
    renderer_state = {}

    for key, value in full_state.items():
        if key.startswith('encoder.'):
            encoder_state[key[8:]] = value
        elif key.startswith('renderer.'):
            renderer_state[key[9:]] = value

    encoder.load_state_dict(encoder_state)
    renderer.load_state_dict(renderer_state)

    encoder.eval()
    renderer.eval()

    # Freeze weights
    for param in encoder.parameters():
        param.requires_grad = False
    for param in renderer.parameters():
        param.requires_grad = False

    return encoder, renderer


def load_trained_controller(
    checkpoint_path: str,
    num_ops: int,
    device: torch.device
) -> Tuple[Controller, OperatorLibrary]:
    """Load trained controller and operators from checkpoint."""
    checkpoint = torch.load(checkpoint_path, map_location=device)

    # Create models
    controller = Controller(
        num_operators=num_ops,
        d_slot=128,
        d_task=128,
        d_hidden=256,
        max_steps=4
    ).to(device)

    operators = OperatorLibrary(
        num_ops=num_ops,
        d_slot=128,
        d_hidden=128,
        H=30,
        W=30
    ).to(device)

    # Load weights
    controller.load_state_dict(checkpoint['controller_state_dict'])
    operators.load_state_dict(checkpoint['operators_state_dict'])

    controller.eval()  # Will be cloned for adaptation
    operators.eval()

    # Freeze operators
    for param in operators.parameters():
        param.requires_grad = False

    return controller, operators


def main():
    parser = argparse.ArgumentParser(
        description="Stream active learning adaptation to JavaScript visualizer"
    )
    parser.add_argument(
        '--autoencoder_checkpoint',
        type=str,
        required=True,
        help='Path to pretrained autoencoder checkpoint'
    )
    parser.add_argument(
        '--controller_checkpoint',
        type=str,
        required=True,
        help='Path to trained controller checkpoint'
    )
    parser.add_argument(
        '--task_id',
        type=str,
        help='Task ID to visualize'
    )
    parser.add_argument(
        '--task_index',
        type=int,
        help='Task index to visualize (0-based)'
    )
    parser.add_argument(
        '--dataset',
        type=str,
        default='data/arc-agi_evaluation_challenges.json',
        help='Path to dataset'
    )
    parser.add_argument(
        '--adaptation_steps',
        type=int,
        default=20,
        help='Number of adaptation steps'
    )
    parser.add_argument(
        '--beam_size',
        type=int,
        default=8,
        help='Beam size for search'
    )
    parser.add_argument(
        '--emit_every',
        type=int,
        default=2,
        help='Emit events every N steps'
    )

    args = parser.parse_args()

    # Device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # Create event emitter
    emitter = EventEmitter()

    try:
        # Load models
        emitter.log(f"Loading models on {device}...")
        encoder, renderer = load_pretrained_autoencoder(
            args.autoencoder_checkpoint,
            device
        )
        controller, operators = load_trained_controller(
            args.controller_checkpoint,
            num_ops=8,
            device=device
        )
        emitter.log("✓ Models loaded successfully")

        # Load dataset
        emitter.log(f"Loading dataset from {args.dataset}...")
        dataset = ARCDataset(args.dataset)

        # Get task
        if args.task_id:
            task = dataset.get_task_by_id(args.task_id)
            if task is None:
                emitter.error(f"Task ID '{args.task_id}' not found")
                sys.exit(1)
        elif args.task_index is not None:
            if args.task_index < 0 or args.task_index >= len(dataset):
                emitter.error(f"Task index {args.task_index} out of range (0-{len(dataset)-1})")
                sys.exit(1)
            task = dataset[args.task_index]
        else:
            emitter.error("Either --task_id or --task_index must be specified")
            sys.exit(1)

        task_id = task['task_id']
        emitter.log(f"✓ Loaded task: {task_id}")

        # Emit task loaded event with training pairs
        train_grids = []
        for i in range(len(task['train_inputs'])):
            h_in, w_in = task['train_shapes'][i]['input']
            h_out, w_out = task['train_shapes'][i]['output']

            train_grids.append({
                'input': task['train_inputs'][i][:h_in, :w_in].tolist(),
                'output': task['train_outputs'][i][:h_out, :w_out].tolist(),
                'input_shape': [h_in, w_in],
                'output_shape': [h_out, w_out]
            })

        emitter.task_loaded(
            task_id=task_id,
            num_train=len(task['train_inputs']),
            num_test=len(task['test_inputs']),
            train_grids=train_grids
        )

        # Create streaming inner loop
        inner_loop = StreamingInnerLoop(
            num_inner_steps=args.adaptation_steps,
            beam_size=args.beam_size,
            max_operator_steps=4,
            learning_rate=1e-3,
            entropy_weight=0.01,
            binary_bonus_weight=0.5,
            device=device,
            emitter=emitter,
            emit_every=args.emit_every
        )

        # Run adaptation
        emitter.log("Starting active adaptation...")
        adapted_controller, metrics = inner_loop.train_on_task(
            task,
            encoder,
            controller,
            operators,
            renderer,
            clone_controller=True,
            verbose=False
        )

        emitter.log(f"Adaptation complete: {metrics.success_rate*100:.1f}% train accuracy")

        # If train solved, predict test
        if metrics.success_rate >= 1.0:
            emitter.log("✓ All train pairs solved! Predicting test...")
            emitter.test_start(len(task['test_inputs']))

            # Build task embedding
            train_pairs = [(task['train_inputs'][i], task['train_outputs'][i], task['train_shapes'][i])
                          for i in range(len(task['train_inputs']))]
            task_embed = build_task_embedding(train_pairs, encoder=None, device=device, analyze_operators=False)

            # Predict test
            adapted_controller.eval()
            test_predictions = []
            test_correct = []

            with torch.no_grad():
                for i in range(len(task['test_inputs'])):
                    input_grid = task['test_inputs'][i].to(device)
                    input_shape = task['test_shapes'][i]['input']
                    output_shape = task['test_shapes'][i]['output']

                    candidates = beam_search(
                        encoder,
                        adapted_controller,
                        operators,
                        renderer,
                        input_grid,
                        input_shape,
                        output_shape,
                        task_embed,
                        target_grid=None,
                        beam_size=16,
                        max_steps=8,
                        device=device,
                        collect_log_probs=False
                    )

                    if len(candidates) > 0:
                        pred = candidates[0].prediction
                    else:
                        pred = torch.zeros(30, 30, dtype=torch.long, device=device)

                    h, w = output_shape
                    pred_list = pred[:h, :w].cpu().tolist()
                    test_predictions.append(pred_list)

                    # Check correctness if ground truth available
                    if task['test_outputs'][i] is not None:
                        target = task['test_outputs'][i]
                        is_correct = exact_match(pred.cpu(), target, h, w)
                        test_correct.append(is_correct)
                    else:
                        test_correct.append(None)

            # Emit test complete
            task_success = all(c for c in test_correct if c is not None)
            competition_score = sum(1.0 for c in test_correct if c) / len(test_correct) if test_correct else 0.0

            emitter.test_complete(
                success=task_success,
                predictions=test_predictions,
                correct=test_correct,
                competition_score=competition_score
            )

            if task_success:
                emitter.log("🎉 TASK FULLY SOLVED!")
            else:
                emitter.log(f"Test: {sum(test_correct)}/{len(test_correct)} correct")
        else:
            emitter.log(f"Train not fully solved ({metrics.success_rate*100:.1f}%), skipping test")

    except Exception as e:
        import traceback
        emitter.error(
            message=str(e),
            details=traceback.format_exc()
        )
        sys.exit(1)


if __name__ == '__main__':
    main()
