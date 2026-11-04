#!/usr/bin/env python3
"""Time synthetic joint, window-only and cached-feature training on one device.

This benchmark excludes DataLoader, disk I/O, feature extraction, Sigma-Delta
encoding and first-call warmup. It is not the total runtime of staged training.
"""
from __future__ import annotations

import argparse
import copy
import json
import math
import os
import platform
from pathlib import Path
import sys
import time

import torch
from torch.nn import functional as F

from benchmark_kfall import positive_float, positive_int, resolve_device
from train.utils import count_parameters, make_autocast, make_grad_scaler, set_seed
from train.window_pretraining import WindowPretrainer
from train_kfall_temporal import get_args, build_temporal_model


PHASES = ('joint', 'window_pretraining', 'cached_temporal')


def build_parser():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--architecture', choices=['standard', 'compact'], default='compact')
    parser.add_argument('--window-seconds', type=positive_float, default=1.)
    parser.add_argument('--batch-size', type=positive_int, default=8, help='Recordings per temporal block.')
    parser.add_argument('--sequence-length', type=positive_int, default=20)
    parser.add_argument('--pretrain-batch-size', type=positive_int, default=128, help='Independent windows per stage-one batch.')
    parser.add_argument('--device', default='auto', help='auto, cpu, cuda, or cuda:<index>')
    parser.add_argument('--warmup', type=positive_int, default=3)
    parser.add_argument('--steps', type=positive_int, default=10)
    parser.add_argument('--seed', type=int, default=0)
    parser.add_argument('--output', type=Path, default=Path('results/kfall_temporal_benchmark.json'))
    return parser


def _synchronize(device):
    if device.type == 'cuda':
        torch.cuda.synchronize(device)


def benchmark_phase(model, batch, cfg, phase, *, warmup=3, steps=10):
    """Time full Adam/backward updates; cached features must already exist.

    Each temporal update starts with state=None: these are independent
    synthetic blocks, not an emulation of state carried along an episode.
    All batch tensors are already resident on the selected device.
    """
    if phase not in PHASES:
        raise ValueError(f'Unknown benchmark phase: {phase}.')
    if warmup < 1 or steps < 1:
        raise ValueError('Warmup and timed steps must be positive.')
    device = batch['targets'].device
    if phase == 'cached_temporal':
        model.set_backbone_trainable('frozen')
    model.train()
    parameters = [parameter for parameter in model.parameters() if parameter.requires_grad]
    optimizer = torch.optim.Adam(parameters, lr=cfg.lr, weight_decay=cfg.wd)
    amp = bool(cfg.amp and device.type == 'cuda')
    scaler = make_grad_scaler(amp)
    last_loss = None

    def step():
        nonlocal last_loss
        optimizer.zero_grad(set_to_none=True)
        with make_autocast(amp):
            if phase == 'cached_temporal':
                outputs, _ = model.forward_features(batch['features'], state=None)
                logits = outputs['logits'].flatten(0, 1)
            elif phase == 'joint':
                outputs, _ = model(batch['events'], batch['offsets'], state=None)
                logits = outputs['logits'].flatten(0, 1)
            else:
                logits = model(batch['events'], offset=batch['offsets'])['head1']
            loss = F.cross_entropy(logits, batch['targets'].flatten(), label_smoothing=cfg.label_smoothing)
        scaler.scale(loss).backward()
        if cfg.grad_clip > 0:
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(parameters, cfg.grad_clip)
        scaler.step(optimizer)
        scaler.update()
        last_loss = loss.detach()

    for _ in range(warmup):
        step()
    _synchronize(device)
    baseline_allocated = None
    if device.type == 'cuda':
        torch.cuda.reset_peak_memory_stats(device)
        baseline_allocated = torch.cuda.memory_allocated(device)
    started = time.perf_counter()
    for _ in range(steps):
        step()
    _synchronize(device)
    elapsed = time.perf_counter() - started
    final_loss = float(last_loss.cpu())
    if not math.isfinite(final_loss):
        raise RuntimeError(f'Nonfinite loss in {phase} benchmark.')
    windows = batch['targets'].numel()
    total, trainable = count_parameters(model)
    report = dict(phase=phase, warmup_steps=warmup, timed_steps=steps,
        elapsed_seconds=elapsed, milliseconds_per_step=elapsed * 1000 / steps,
        windows_per_step=windows, windows_per_second=windows * steps / elapsed,
        timed_windows=windows * steps, parameters=total, trainable_parameters=trainable,
        final_synthetic_loss=final_loss, amp=amp,
        optimizer='Adam', backward=True, grad_clip=cfg.grad_clip,
        input_shapes={key: list(value.shape) for key, value in batch.items()},
        temporal_state='reset to None at every synthetic block' if phase != 'window_pretraining'
                       else 'independent windows; no recurrent layer',
        feature_extraction_in_timing=False, includes_dataloader=False,
        step_unit='optimizer update; compare windows_per_second across different batch sizes')
    if phase == 'window_pretraining':
        report['batch_windows'] = windows
    else:
        report.update(batch_recordings=int(batch['targets'].shape[0]),
                      sequence_length=int(batch['targets'].shape[1]))
    if device.type == 'cuda':
        peak = torch.cuda.max_memory_allocated(device)
        report['cuda_memory_mib'] = dict(
            allocated=torch.cuda.memory_allocated(device) / 2**20,
            reserved=torch.cuda.memory_reserved(device) / 2**20,
            allocated_before_timing=baseline_allocated / 2**20,
            peak_allocated=peak / 2**20, peak_reserved=torch.cuda.max_memory_reserved(device) / 2**20,
            peak_additional_allocated=(peak - baseline_allocated) / 2**20)
        report['memory_scope'] = 'Process PyTorch allocations, including resident inputs/template; not parameter storage alone.'
    return report


def benchmark_training_phases(cfg, device, *, pretrain_batch_size=128, warmup=3, steps=10):
    """Measure three fresh model copies against one deterministic synthetic pool."""
    if min(pretrain_batch_size, warmup, steps, cfg.batch_size, cfg.sequence_length) < 1:
        raise ValueError('Batch sizes, sequence length and step counts must be positive.')
    device = torch.device(device)
    if device.type == 'cuda':
        torch.cuda.set_device(device)
    set_seed(cfg.seed)
    template = build_temporal_model(cfg, device)
    size = max(pretrain_batch_size, cfg.batch_size * cfg.sequence_length)
    random = torch.rand((size, cfg.out_win, 6), device=device)
    events = torch.stack((random < .025, random >= .975), dim=-1).to(torch.uint8)
    del random
    offsets = torch.randn((size, 6), device=device)
    targets = torch.randint(4, (size,), device=device)
    temporal_size = cfg.batch_size * cfg.sequence_length
    temporal = dict(events=events[:temporal_size].reshape(cfg.batch_size, cfg.sequence_length, cfg.out_win, 6, 2),
                    offsets=offsets[:temporal_size].reshape(cfg.batch_size, cfg.sequence_length, 6),
                    targets=targets[:temporal_size].reshape(cfg.batch_size, cfg.sequence_length))
    phases = {}
    for phase in PHASES:
        set_seed(cfg.seed)
        model = copy.deepcopy(template)
        if phase == 'window_pretraining':
            model = WindowPretrainer(model.backbone).to(device)
            batch = dict(events=events[:pretrain_batch_size], offsets=offsets[:pretrain_batch_size],
                         targets=targets[:pretrain_batch_size])
        elif phase == 'cached_temporal':
            model.set_backbone_trainable('frozen').eval()
            with torch.inference_mode():
                extracted = model.backbone.extract_features(events[:temporal_size].float(), offsets[:temporal_size])
            # Normal tensors are needed for parameter gradients in Linear;
            # inference-mode tensors cannot be saved for backward.
            features = extracted.clone().reshape(cfg.batch_size, cfg.sequence_length, -1)
            batch = dict(features=features, targets=temporal['targets'])
        else:
            batch = temporal
        phases[phase] = benchmark_phase(model, batch, cfg, phase, warmup=warmup, steps=steps)
        del model, batch
    report = dict(benchmark='synthetic_temporal_training_phases', synthetic_only=True,
        includes_dataloader=False, includes_disk_io=False, includes_feature_extraction=False,
        is_total_staged_runtime=False,
        timing_scope='Steady-state fixed-batch optimizer updates only; feature extraction and warmup excluded. Not total staged training runtime.',
        temporal_state_scope='Independent synthetic blocks; no episode carry or padding.',
        device=str(device), device_name=torch.cuda.get_device_name(device) if device.type == 'cuda' else 'CPU',
        torch_version=str(torch.__version__), torch_cuda_version=torch.version.cuda,
        python_version=sys.version.split()[0], platform=platform.platform(),
        cpu_count=os.cpu_count(), torch_cpu_threads=torch.get_num_threads(),
        architecture=cfg.architecture, conv_channels=cfg.conv_ch, kernels=cfg.kernels,
        recurrent_hidden=cfg.recurrent_hidden, memory_reset=cfg.memory_reset,
        feature_dim=template.backbone.fused_dim, window_seconds=cfg.window_seconds,
        raw_win=cfg.raw_win, out_win=cfg.out_win, seed=cfg.seed,
        synthetic_event_density=.05,
        density_definition='Either polarity per sensor/time: 5%; ON/OFF are mutually exclusive, each 2.5%.',
        phases=phases)
    if device.type == 'cuda':
        properties = torch.cuda.get_device_properties(device)
        report['cuda_hardware'] = dict(total_memory_mib=properties.total_memory / 2**20,
            multiprocessor_count=properties.multi_processor_count,
            compute_capability=list(torch.cuda.get_device_capability(device)))
    return report


def main(argv=None):
    parser = build_parser()
    args = parser.parse_args(argv)
    try:
        device = resolve_device(args.device)
        cfg = get_args(['--architecture', args.architecture, '--window-seconds', str(args.window_seconds),
                        '--batch-size', str(args.batch_size), '--sequence-length', str(args.sequence_length),
                        '--workers', '0', '--seed', str(args.seed)])
        print(f'[benchmark] {device} | {args.architecture} | window={cfg.window_seconds:g}s | '
              f'batch={cfg.batch_size} recordings | K={cfg.sequence_length} | pretrain_batch={args.pretrain_batch_size}', flush=True)
        result = benchmark_training_phases(cfg, device, pretrain_batch_size=args.pretrain_batch_size,
                                          warmup=args.warmup, steps=args.steps)
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(result, indent=2) + '\n')
        for name, phase in result['phases'].items():
            print(f'[{name}] {phase["milliseconds_per_step"]:.2f} ms/step | '
                  f'{phase["windows_per_step"]} windows/step | {phase["windows_per_second"]:.1f} windows/s', flush=True)
        print(f'[saved] {args.output}', flush=True)
    except (ValueError, OSError, RuntimeError) as exc:
        parser.error(str(exc))


if __name__ == '__main__':
    main()
