"""Independent-window pretraining, frozen features and optional last-block tuning."""
from __future__ import annotations

from argparse import Namespace
from pathlib import Path
import time

import torch

from loaders.kfall_features import extract_feature_cache
from train.temporal import run_temporal_epoch
from train.utils import (count_parameters, fit_model, format_single_epoch_line,
                         json_ready, make_grad_scaler, save_json)
from train.window_pretraining import pretrain_backbone


def _temporal_stage(model, train, val, cfg, device, stats, *, checkpoint, metrics_dir, name):
    """Fit a temporal stage; each stage gets a fresh optimizer and scheduler."""
    parameters = [value for value in model.parameters() if value.requires_grad]
    optimizer = torch.optim.Adam(parameters, lr=cfg.lr, weight_decay=cfg.wd)
    scaler = make_grad_scaler(False)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=cfg.lr_factor,
        patience=cfg.plateau_patience, min_lr=cfg.min_lr)
    epoch = 0

    def runner(dataset, optimizer=None, scaler=None, collect_predictions=False):
        nonlocal epoch
        if optimizer is not None:
            epoch += 1
        return run_temporal_epoch(model, dataset, cfg, device, stats, optimizer=optimizer,
            scaler=scaler, epoch=epoch, collect=collect_predictions,
            phase=f'{name} {"Train" if optimizer is not None else "Val"}')

    started = time.perf_counter()
    fitted = fit_model(model=model, train_loader=train, val_loader=val, cfg=cfg,
        optimizer=optimizer, scheduler=scheduler, scaler=scaler, metrics_dir=metrics_dir,
        ckpt_path=checkpoint, run_epoch_fn=runner, format_epoch_fn=format_single_epoch_line,
        checkpoint_extra_fn=lambda: {'training_statistics': stats, 'training_stage': name})
    if not checkpoint.is_file():
        raise ValueError(f'{name} produced no finite best validation checkpoint.')
    best = torch.load(checkpoint, map_location=device, weights_only=True)
    model.load_state_dict(best['model_state_dict'], strict=True)
    return dict(stage=name, best_val_loss=float(best['best_val_loss']), best_epoch=int(best['epoch']),
        completed_epochs=len(fitted['history']), requested_epochs=cfg.epochs,
        elapsed_seconds=time.perf_counter() - started,
        trainable_parameters=count_parameters(model)[1], checkpoint=str(checkpoint.resolve()),
        metrics_dir=str(metrics_dir.resolve()), test_consulted=False)


def fit_staged_temporal(model, episodes, cfg, device, stats, run):
    """Keep test data outside every training/feature-selection stage.

    The final checkpoint contains the complete original online architecture.
    Cached vectors are used only while its backbone is frozen. Fine-tuning
    returns to raw event windows and recomputes features after every update.
    """
    run = Path(run)
    if cfg.amp:
        raise ValueError('Staged temporal training requires FP32 features/state; AMP belongs to pretraining only.')
    summary = dict(strategy='staged', feature_dim=model.backbone.fused_dim,
        total_parameters=count_parameters(model)[0], feature_precision='float32',
        test_used_for_selection=False, stages=[])
    print('[stage 1/2] Independent-window SCN pretraining', flush=True)
    started = time.perf_counter()
    pretrain_cfg = Namespace(**{**vars(cfg), 'amp': bool(cfg.pretrain_amp)})
    pretrained = pretrain_backbone(model, episodes['train'], episodes['val'], pretrain_cfg, device,
        epochs=cfg.pretrain_epochs, batch_size=cfg.pretrain_batch_size, out_dir=run / 'stages/pretrain')
    pretrained['elapsed_seconds'] = time.perf_counter() - started
    summary['stages'].append(pretrained)

    model.set_backbone_trainable('frozen').float().eval()
    feature_root = Path(cfg.feature_cache_dir) if cfg.feature_cache_dir is not None else run / 'features'
    print(f'[features] Frozen FP32 backbone -> {model.backbone.fused_dim} features/window; train + val only', flush=True)
    started = time.perf_counter()
    cached = {split: extract_feature_cache(model, episodes[split], feature_root / split,
                batch_size=cfg.feature_batch_size, device=device, split_name=split,
                no_progress=cfg.no_progress, refresh=cfg.refresh_features)
              for split in ('train', 'val')}
    summary['feature_extraction'] = dict(elapsed_seconds=time.perf_counter() - started,
        batch_size=cfg.feature_batch_size, cache_dir=str(feature_root.resolve()),
        splits={key: dict(windows=value.window_count,
                          bytes=value.features.size * value.features.dtype.itemsize,
                          manifest=value.manifest) for key, value in cached.items()})
    print(f'[stage 2/2] Cached-feature temporal training | trainable={count_parameters(model)[1]} | backbone calls per batch=0', flush=True)
    stage2_checkpoint = run / 'stages/temporal/checkpoints/best_model.pt'
    stage2 = _temporal_stage(model, cached['train'], cached['val'], cfg, device, stats,
        checkpoint=stage2_checkpoint, metrics_dir=run / 'metrics', name='cached_temporal')
    summary['stages'].append(stage2)
    stage2['selection_val_loss'] = stage2['best_val_loss']
    selected = stage2

    if cfg.finetune_epochs:
        # Use online validation for both candidates. Different convolution
        # batch sizes can change FP32 rounding near a LIF threshold, so the
        # cached loss alone is not the strict reference for online fine-tuning.
        reference_started = time.perf_counter()
        reference = run_temporal_epoch(model, episodes['val'], cfg, device, stats,
            phase='Fine-tune reference Val')
        stage2['selection_val_loss'] = reference['loss']
        stage2['online_reference_seconds'] = time.perf_counter() - reference_started
        stage2['elapsed_seconds'] += stage2['online_reference_seconds']
        model.set_backbone_trainable('last')
        print('[fine-tune] Last convolution + temporal memory/readouts; recomputing event features', flush=True)
        tune_cfg = Namespace(**{**vars(cfg), 'epochs': cfg.finetune_epochs, 'lr': cfg.finetune_lr})
        tuned = _temporal_stage(model, episodes['train'], episodes['val'], tune_cfg, device, stats,
            checkpoint=run / 'stages/finetune/checkpoints/best_model.pt',
            metrics_dir=run / 'stages/finetune/metrics', name='last_block_finetune')
        summary['stages'].append(tuned)
        tuned['selection_val_loss'] = tuned['best_val_loss']
        if tuned['selection_val_loss'] < stage2['selection_val_loss']:
            selected = tuned

    # Select stages using the same validation loss, without consulting test.
    # Keep their local configs/history alongside a complete canonical checkpoint.
    payload = torch.load(selected['checkpoint'], map_location='cpu', weights_only=True)
    payload['stage_config'] = payload['config']
    payload['stage_best_val_loss'] = payload['best_val_loss']
    payload['best_val_loss'] = selected['selection_val_loss']
    payload['config'] = json_ready(cfg)
    payload['training_stage'] = selected['stage']
    summary['selected_stage'] = selected['stage']
    summary['selected_stage_best_val_loss'] = selected['best_val_loss']
    summary['selected_best_val_loss'] = selected['selection_val_loss']
    summary['selection_rule'] = ('Minimum online validation temporal loss across stage 2 and fine-tuning.'
                                 if cfg.finetune_epochs else 'Best cached-feature validation temporal loss.')
    summary['elapsed_seconds'] = sum(stage['elapsed_seconds'] for stage in summary['stages']) + summary['feature_extraction']['elapsed_seconds']
    summary['inference'] = 'Full online SCN + offset + recurrent LIF; no feature cache needed.'
    payload['staged_training'] = summary
    destination = run / 'checkpoints/best_model.pt'
    destination.parent.mkdir(parents=True, exist_ok=True)
    torch.save(payload, destination)
    model.load_state_dict(payload['model_state_dict'], strict=True)
    save_json(run / 'training_stages.json', summary)
    print(f'[staged] selected={selected["stage"]} | val_loss={selected["selection_val_loss"]:.6f} | checkpoint={destination}', flush=True)
    return summary
