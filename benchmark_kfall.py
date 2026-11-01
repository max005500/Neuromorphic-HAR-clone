#!/usr/bin/env python3
"""Measure SCN training throughput on a fixed synthetic batch already on device.

This isolates network/optimizer execution from dataset loading and Sigma-Delta
preprocessing. Optional profiling runs separately, after the timed measurement.
"""
from __future__ import annotations

import argparse
import json
import math
import time
from pathlib import Path
from typing import Any

import torch
import torch.nn.functional as F

from train.utils import count_parameters, extract_single_logits, make_autocast, make_grad_scaler, set_seed
from train_kfall import build_model, get_args


def positive_float(value: str) -> float:
    number = float(value)
    if not math.isfinite(number) or number <= 0:
        raise argparse.ArgumentTypeError('must be a finite positive number')
    return number


def positive_int(value: str) -> int:
    number = int(value)
    if number < 1:
        raise argparse.ArgumentTypeError('must be positive')
    return number


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--window-seconds', type=positive_float, default=2.05)
    parser.add_argument('--batch-size', type=positive_int, default=128)
    parser.add_argument('--device', default='auto', help='auto, cpu, cuda, or cuda:<index>')
    parser.add_argument('--warmup', type=positive_int, default=5)
    parser.add_argument('--steps', type=positive_int, default=20)
    parser.add_argument('--amp', action='store_true', help='Use the same CUDA FP16 autocast as training.')
    parser.add_argument('--seed', type=int, default=0)
    parser.add_argument('--profile', type=Path, default=None, metavar='TRACE_JSON',
                        help='Export a separate three-step CPU/CUDA Chrome trace and scalar operation counts.')
    parser.add_argument('--output', type=Path, default=Path('results/kfall_benchmark.json'))
    return parser


def resolve_device(value: str) -> torch.device:
    if value == 'auto':
        return torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    device = torch.device(value)
    if device.type not in {'cpu', 'cuda'}:
        raise ValueError('Only CPU or CUDA benchmarking is supported.')
    if device.type == 'cuda':
        if not torch.cuda.is_available():
            raise ValueError('CUDA was requested but is not available in this PyTorch environment.')
        if device.index is not None and device.index >= torch.cuda.device_count():
            raise ValueError('The requested CUDA device index does not exist.')
    return device


def _synchronize(device: torch.device) -> None:
    if device.type == 'cuda':
        torch.cuda.synchronize(device)


def benchmark_model(
    model: torch.nn.Module, batch: dict[str, torch.Tensor], cfg: Any, *,
    warmup: int = 5, steps: int = 20, profile_path: Path | None = None,
) -> dict[str, Any]:
    """Warm up and time actual forward/backward/Adam updates, without per-step reads."""
    if warmup < 1 or steps < 1:
        raise ValueError('warmup and steps must be positive.')
    device = batch['x'].device
    amp = bool(cfg.amp and device.type == 'cuda')
    optimizer = torch.optim.Adam((p for p in model.parameters() if p.requires_grad),
                                 lr=cfg.lr, weight_decay=cfg.wd)
    scaler = make_grad_scaler(amp)
    model.train()

    def step() -> None:
        optimizer.zero_grad(set_to_none=True)
        with make_autocast(amp):
            logits = extract_single_logits(model(batch['x'], offset=batch['offset']))
            loss = F.cross_entropy(logits, batch['y'], label_smoothing=cfg.label_smoothing)
        scaler.scale(loss).backward()
        if cfg.grad_clip and cfg.grad_clip > 0:
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), cfg.grad_clip)
        scaler.step(optimizer)
        scaler.update()

    for _ in range(warmup):
        step()
    _synchronize(device)
    if device.type == 'cuda':
        torch.cuda.reset_peak_memory_stats(device)
    started = time.perf_counter()
    for _ in range(steps):
        step()
    _synchronize(device)
    elapsed = time.perf_counter() - started
    size = int(batch['x'].shape[0])
    total, trainable = count_parameters(model)
    result = {
        'benchmark': 'synthetic_fixed_batch_training', 'includes_dataloader': False,
        'device': str(device), 'device_name': torch.cuda.get_device_name(device) if device.type == 'cuda' else 'CPU',
        'torch_version': str(torch.__version__), 'torch_cuda_version': torch.version.cuda,
        'amp': amp, 'warmup_steps': warmup, 'timed_steps': steps, 'batch_size': size,
        'event_shape': list(batch['x'].shape), 'parameters': total, 'trainable_parameters': trainable,
        'elapsed_seconds': elapsed, 'milliseconds_per_batch': elapsed * 1000 / steps,
        'samples_per_second': size * steps / elapsed,
    }
    if device.type == 'cuda':
        result['cuda_memory_mib'] = {
            'allocated': torch.cuda.memory_allocated(device) / 2**20,
            'reserved': torch.cuda.memory_reserved(device) / 2**20,
            'peak_allocated': torch.cuda.max_memory_allocated(device) / 2**20,
            'peak_reserved': torch.cuda.max_memory_reserved(device) / 2**20,
        }
    if profile_path is not None:
        activities = [torch.profiler.ProfilerActivity.CPU]
        if device.type == 'cuda':
            activities.append(torch.profiler.ProfilerActivity.CUDA)
        with torch.profiler.profile(activities=activities, record_shapes=True) as profile:
            for _ in range(3):
                step()
        _synchronize(device)
        profile_path = Path(profile_path)
        profile_path.parent.mkdir(parents=True, exist_ok=True)
        profile.export_chrome_trace(str(profile_path))
        result['profile'] = {
            'trace': str(profile_path), 'steps': 3,
            'scalar_operation_counts': {entry.key: entry.count for entry in profile.key_averages()
                                        if entry.key in {'aten::item', 'aten::_local_scalar_dense'}},
        }
    return result


def main(argv: list[str] | None = None) -> None:
    parser = build_parser()
    args = parser.parse_args(argv)
    try:
        device = resolve_device(args.device)
        cfg = get_args(['--window-seconds', str(args.window_seconds), '--batch-size', str(args.batch_size),
                        '--workers', '0', '--seed', str(args.seed)] + (['--amp'] if args.amp else []))
        set_seed(args.seed)
        model = build_model(cfg, device)
        # At most one ON/OFF event per sensor channel; density is fixed at 5%.
        random = torch.rand((args.batch_size, cfg.out_win, 6), device=device)
        events = torch.stack((random < .025, random >= .975), dim=-1).float()
        del random
        batch = {'x': events, 'offset': torch.randn((args.batch_size, 6), device=device),
                 'y': torch.randint(cfg.n_classes, (args.batch_size,), device=device)}
        print(f'[benchmark] {device} | window={args.window_seconds:g}s | batch={args.batch_size} | '
              f'warmup={args.warmup} | timed_steps={args.steps}', flush=True)
        result = benchmark_model(model, batch, cfg, warmup=args.warmup, steps=args.steps,
                                 profile_path=args.profile)
        result.update(window_seconds=cfg.window_seconds, raw_win=cfg.raw_win, out_win=cfg.out_win,
                      seed=args.seed, synthetic_event_density=.05)
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(result, indent=2) + '\n', encoding='utf-8')
    except (ValueError, RuntimeError, OSError) as exc:
        parser.error(str(exc))
    print(f"[benchmark] {result['milliseconds_per_batch']:.2f} ms/batch | "
          f"{result['samples_per_second']:.1f} samples/s")
    if 'cuda_memory_mib' in result:
        print(f"[memory] peak_allocated={result['cuda_memory_mib']['peak_allocated']:.1f} MiB")
    print(f'[saved] {args.output}')
    if 'profile' in result:
        print(f"[profile] {result['profile']}")


if __name__ == '__main__':
    main()
