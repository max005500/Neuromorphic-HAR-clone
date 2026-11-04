"""Pretrain a temporal SCN backbone with an independent four-class window head."""
from __future__ import annotations

import copy
import hashlib
from argparse import Namespace
from pathlib import Path

import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader, Subset
from tqdm.auto import tqdm

from loaders.utils import balanced_class_weights
from train.utils import (fit_model, format_single_epoch_line, make_grad_scaler,
                         save_json)
from train_kfall import KFallTrainingDataset
from train_ucihar import run_epoch


class WindowPretrainer(nn.Module):
    """Share the SCN backbone; the temporary head never enters the final model."""
    def __init__(self, backbone):
        super().__init__()
        self.backbone = backbone
        self.head = nn.Linear(backbone.fused_dim, 4)

    def forward(self, events, offset=None):
        features = self.backbone.extract_features(events, offset)
        return {'head1': self.head(features)}


def supervised_window_indices(episodes, split_name):
    """Keep the same uniform endpoints and ambiguity mask as temporal training."""
    if split_name not in {'train', 'val'}:
        raise ValueError('Window pretraining uses only train and val episodes.')
    allowed = set(episodes.metadata['split_file_indices'][split_name])
    if any(ep.file_index not in allowed for ep in episodes.episodes):
        raise ValueError(f'Window pretraining {split_name} contains recordings from another partition.')
    chunks = [ep.indices[ep.valid] for ep in episodes.episodes]
    indices = np.sort(np.concatenate(chunks)) if chunks else np.empty(0, dtype=np.int64)
    if not len(indices):
        raise ValueError(f'No supervised {split_name} windows remain for backbone pretraining.')
    if len(np.unique(indices)) != len(indices):
        raise ValueError('Window pretraining endpoints must be unique.')
    return indices


def pretrain_backbone(model, train_episodes, val_episodes, cfg, device, *,
                      epochs, batch_size, out_dir):
    """Train only the shared backbone and a disposable head; restore its best val weights.

    Validation selects the best independent-window checkpoint. The temporal
    memory and readouts are absent from this optimizer and stay unchanged.
    Only valid, uniformly sampled train/val windows enter these loaders.
    """
    if epochs < 1 or batch_size < 1:
        raise ValueError('Backbone pretraining epochs and batch size must be positive.')
    train_indices = supervised_window_indices(train_episodes, 'train')
    val_indices = supervised_window_indices(val_episodes, 'val')
    train_subjects = {train_episodes.metadata['recordings'][ep.file_index]['subject']
                      for ep in train_episodes.episodes}
    val_subjects = {val_episodes.metadata['recordings'][ep.file_index]['subject']
                    for ep in val_episodes.episodes}
    if train_subjects & val_subjects:
        raise ValueError('Backbone pretraining requires disjoint train and validation subjects.')
    labels = np.asarray(train_episodes.arrays['y_h1'][train_indices], dtype=np.int64)
    class_weight = (balanced_class_weights(labels, 4) if cfg.class_weight == 'balanced' else None)
    weights = None if class_weight is None else torch.as_tensor(class_weight, device=device)

    stage_cfg = Namespace(**copy.deepcopy(vars(cfg)))
    stage_cfg.epochs, stage_cfg.batch_size = int(epochs), int(batch_size)
    stage_cfg.head_mode = 'single'
    stage_cfg.n_classes = 4
    stage_cfg.model_type = 'window_scn_pretraining'
    stage_cfg.training_stage = 'independent_window_backbone'
    stage_cfg.lr = float(getattr(cfg, 'pretrain_lr', None) or cfg.lr)
    destination = Path(out_dir)
    destination.mkdir(parents=True, exist_ok=True)
    save_json(destination / 'config.json', stage_cfg)
    np.save(destination / 'train_window_indices.npy', train_indices)
    np.save(destination / 'val_window_indices.npy', val_indices)

    if hasattr(model, 'set_backbone_trainable'):
        model.set_backbone_trainable('all')
    else:
        model.backbone.requires_grad_(True)
    window_model = WindowPretrainer(model.backbone).to(device)
    generator = torch.Generator().manual_seed(cfg.seed)
    datasets = {
        'train': Subset(KFallTrainingDataset(train_episodes.arrays, n_classes=4), train_indices.tolist()),
        'val': Subset(KFallTrainingDataset(val_episodes.arrays, n_classes=4), val_indices.tolist()),
    }
    loaders = {
        name: DataLoader(dataset, batch_size=stage_cfg.batch_size, shuffle=name == 'train',
                         num_workers=0, pin_memory=device.type == 'cuda' and not cfg.no_pin_memory,
                         generator=generator)
        for name, dataset in datasets.items()
    }
    optimizer = torch.optim.Adam(window_model.parameters(), lr=stage_cfg.lr, weight_decay=cfg.wd)
    scaler = make_grad_scaler(cfg.amp and device.type == 'cuda')
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='min', factor=cfg.lr_factor, patience=cfg.plateau_patience, min_lr=cfg.min_lr)
    epoch = 0

    def runner(loader, optimizer=None, scaler=None, collect_predictions=False):
        nonlocal epoch
        training = optimizer is not None
        if training:
            epoch += 1
        label = f'SCN {"Train" if training else "Val"} e{epoch:03d}/{stage_cfg.epochs}'
        with tqdm(loader, total=len(loader), desc=label, unit='lote', dynamic_ncols=True,
                  mininterval=.5, disable=True if getattr(cfg, 'no_progress', False) else None) as progress:
            return run_epoch(window_model, progress, device, stage_cfg, optimizer=optimizer,
                             scaler=scaler, class_weight=weights, collect_predictions=collect_predictions)

    checkpoint = destination / 'checkpoints' / 'best_model.pt'
    fitted = fit_model(model=window_model, train_loader=loaders['train'], val_loader=loaders['val'],
        cfg=stage_cfg, optimizer=optimizer, scheduler=scheduler, scaler=scaler,
        metrics_dir=destination / 'metrics', ckpt_path=checkpoint, run_epoch_fn=runner,
        format_epoch_fn=format_single_epoch_line)
    if not checkpoint.is_file():
        raise ValueError('Backbone pretraining produced no finite best validation checkpoint.')
    best = torch.load(checkpoint, map_location=device, weights_only=True)
    window_model.load_state_dict(best['model_state_dict'], strict=True)
    window_model.zero_grad(set_to_none=True)
    model.backbone.eval()
    report = dict(stage='independent_window_backbone', checkpoint=str(checkpoint.resolve()),
                  checkpoint_sha256=hashlib.sha256(checkpoint.read_bytes()).hexdigest(),
                  best_epoch=int(best['epoch']), best_val_loss=float(best['best_val_loss']),
                  completed_epochs=len(fitted['history']), epochs=int(epochs), batch_size=int(batch_size),
                  train_windows=len(train_indices), val_windows=len(val_indices),
                  train_subjects=sorted(train_subjects), val_subjects=sorted(val_subjects),
                  class_weights=None if class_weight is None else class_weight.tolist(),
                  temporary_head_discarded=True, test_consulted=False)
    save_json(destination / 'pretraining_summary.json', report)
    return report
