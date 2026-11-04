#!/usr/bin/env python3
"""Experimental SCN + recurrent LIF on chronological KFall episodes."""
from __future__ import annotations
import argparse
import json
from pathlib import Path
import sys
import time
import numpy as np
import torch
from train_kfall import get_args as base_args, build_data
from train.utils import (set_seed, get_device, scn_model_kwargs, make_grad_scaler,
                         fit_model, format_single_epoch_line, save_json, count_parameters)
from models.temporal import TemporalSCN
from loaders.kfall_sequences import EpisodeSet, training_statistics
from train.temporal import run_temporal_epoch


def get_args(argv=None):
    argv = list(sys.argv[1:] if argv is None else argv)
    parser = argparse.ArgumentParser(description=__doc__, add_help=False)
    parser.add_argument('--no-progress', action='store_true', help='Disable tqdm epoch progress bars.')
    parser.add_argument('--memory', choices=['none', 'lif'], default='lif')
    parser.add_argument('--recurrent-hidden', type=int, default=32)
    parser.add_argument('--memory-tau', type=float, default=.9)
    parser.add_argument('--memory-threshold', type=float, default=.5)
    parser.add_argument('--sequence-length', type=int, default=20)
    parser.add_argument('--lambda-transition', type=float, default=0.)
    parser.add_argument('--lambda-boundary', type=float, default=0.)
    parser.add_argument('--transition-laplace', type=float, default=1.)
    parser.add_argument('--boundary-ignore-seconds', type=float, default=0.)
    if '--help' in argv or '-h' in argv:
        parser.print_help()
        base_args(['--help'])
    temporal, remaining = parser.parse_known_args(argv)
    cfg = base_args(['--batch-size', '8', '--workers', '0', '--stride', '10', *remaining])
    for key, value in vars(temporal).items():
        setattr(cfg, key, value)
    if cfg.n_classes != 4 or cfg.label_policy == 'legacy' or cfg.head_mode == 'dual':
        parser.error('Use complete four-class labels and single or hierarchical heads.')
    if cfg.spiking_heads:
        parser.error('The temporal module uses recurrent LIF memory with linear readouts; omit --spiking-heads.')
    if cfg.workers != 0:
        parser.error('Episode batches are sliced directly from memmaps; use --workers 0.')
    if cfg.train_per_class:
        parser.error('Chronological episodes require all windows: --train-per-class 0.')
    numeric = [cfg.memory_tau, cfg.memory_threshold, cfg.lambda_transition, cfg.lambda_boundary,
               cfg.transition_laplace, cfg.boundary_ignore_seconds]
    if (not np.isfinite(numeric).all() or cfg.sequence_length < 1 or cfg.recurrent_hidden < 1
            or not 0 <= cfg.memory_tau <= 1 or cfg.memory_threshold <= 0 or cfg.transition_laplace <= 0
            or min(cfg.lambda_transition, cfg.lambda_boundary, cfg.boundary_ignore_seconds) < 0):
        parser.error('Invalid temporal size, loss weights, leak, threshold or masking tolerance.')
    cfg.model_type = 'temporal_scn'
    cfg.temporal_protocol = 'stateful_uniform_episodes_v1'
    return cfg


def build_temporal_model(cfg, device):
    return TemporalSCN(scn_model_kwargs(cfg), memory=cfg.memory, hidden_size=cfg.recurrent_hidden,
        memory_tau=cfg.memory_tau, memory_threshold=cfg.memory_threshold,
        hierarchical=cfg.head_mode == 'hierarchical', boundary=cfg.lambda_boundary > 0).to(device)


def main(argv=None):
    cfg = get_args(argv)
    set_seed(cfg.seed)
    device = torch.device('cpu') if cfg.device == 'cpu' else get_device(cfg.gpu)
    data = build_data(cfg, device)
    # Explicit subject verification before constructing any episode batches.
    subjects = [set(data['metadata']['split_subjects'][name]) for name in ('train', 'val', 'test')]
    if any(a & b for i, a in enumerate(subjects) for b in subjects[i + 1:]):
        raise ValueError('Temporal experiments require disjoint subjects.')
    episodes = {key: EpisodeSet(value, data['metadata'], boundary_ignore_seconds=cfg.boundary_ignore_seconds)
                for key, value in data['splits'].items()}
    for key, value in episodes.items():
        print(f'[episodes:{key}] recordings={len(value.episodes)} windows={value.window_count}', flush=True)
    stats = training_statistics(episodes['train'], laplace=cfg.transition_laplace)
    if cfg.preprocess_only:
        print(f'[prepared] {Path(cfg.cache_dir) / ("kfall_event_loader_" + cfg.cache_name)}')
        return
    root = Path(cfg.out_dir)
    root.mkdir(parents=True, exist_ok=True)
    name = f'{cfg.run}_temporal_{time.strftime("%Y%m%d_%H%M%S")}'
    run = root / name
    suffix = 0
    while True:
        try:
            run.mkdir()
            break
        except FileExistsError:
            suffix += 1
            run = root / f'{name}_{suffix}'
    cfg.run = run.name
    save_json(run / 'config.json', cfg)
    save_json(run / 'dataset_metadata.json', data['metadata'])
    save_json(run / 'training_statistics.json', stats)
    model = build_temporal_model(cfg, device)
    print(f'[temporal] {cfg.memory}({cfg.recurrent_hidden}) | K={cfg.sequence_length} | '
          f'{100/cfg.stride:g} decisions/s | batch={cfg.batch_size} recordings | parameters={count_parameters(model)[0]}', flush=True)
    optimizer = torch.optim.Adam(model.parameters(), lr=cfg.lr, weight_decay=cfg.wd)
    scaler = make_grad_scaler(cfg.amp and device.type == 'cuda')
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=cfg.lr_factor,
        patience=cfg.plateau_patience, min_lr=cfg.min_lr)
    epoch = 0
    def runner(dataset, optimizer=None, scaler=None, collect_predictions=False):
        nonlocal epoch
        if optimizer is not None:
            epoch += 1
        return run_temporal_epoch(model, dataset, cfg, device, stats, optimizer=optimizer, scaler=scaler,
                                  epoch=epoch, collect=collect_predictions, phase='Train' if optimizer is not None else 'Val')
    checkpoint = run / 'checkpoints/best_model.pt'
    fit_model(model=model, train_loader=episodes['train'], val_loader=episodes['val'], cfg=cfg,
        optimizer=optimizer, scheduler=scheduler, scaler=scaler, metrics_dir=run / 'metrics',
        ckpt_path=checkpoint, run_epoch_fn=runner, format_epoch_fn=format_single_epoch_line,
        checkpoint_extra_fn=lambda: {'training_statistics': stats})
    saved = torch.load(checkpoint, map_location=device, weights_only=False)
    model.load_state_dict(saved['model_state_dict'], strict=True)
    from evaluation.kfall_temporal import evaluate_temporal
    evaluate_temporal(run, split_name='val', device=str(device))
    if not cfg.validation_only:
        evaluate_temporal(run, split_name='test', device=str(device))
    print(f'[done] saved_run={run}', flush=True)


if __name__ == '__main__':
    main()
