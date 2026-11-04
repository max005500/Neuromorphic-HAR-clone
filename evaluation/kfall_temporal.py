"""Causal full-episode evaluation; every uniform window is evaluated once."""
from __future__ import annotations
import hashlib
import json
from pathlib import Path
from types import SimpleNamespace
import time
import numpy as np
import torch
from evaluation.kfall_data import load_evaluation_data, _data_config, _record_identity
from evaluation.kfall_reports import save_evaluation_reports
from loaders.kfall_sequences import EpisodeSet
from loaders.kfall_event_loader import ARRAY_KEYS
from train.temporal import run_temporal_epoch
from train.utils import count_parameters
from preprocess.utils import save_json


def data_provenance_sha256(metadata):
    """Small comparison key for data identity, independent of model/runtime."""
    manifest = {key: metadata[key] for key in (
        'version', 'dataset', 'source_signature', 'split_subjects', 'split_file_indices',
        'channels', 'units', 'fall_label_rule')}
    manifest['config'] = _data_config(metadata)
    manifest['recordings'] = [_record_identity(record) for record in metadata['recordings']]
    return hashlib.sha256(json.dumps(manifest, sort_keys=True, separators=(',', ':')).encode()).hexdigest()


def load_temporal_run(run_dir, device='auto', cache_dir=None, *, split_name='test'):
    from evaluate_kfall import _device
    from train_kfall_temporal import build_temporal_model
    run = Path(run_dir).expanduser().resolve()
    checkpoint = run / 'checkpoints/best_model.pt'
    saved = torch.load(checkpoint, map_location='cpu', weights_only=True)
    cfg = SimpleNamespace(**saved['config'])
    if getattr(cfg, 'model_type', None) != 'temporal_scn' or cfg.temporal_protocol != 'stateful_uniform_episodes_v1':
        raise ValueError('Use a checkpoint from train_kfall_temporal.py.')
    arrays, metadata, snapshot, cache = load_evaluation_data(run, saved['config'], cache_dir, split_name=split_name)
    statistics = json.loads((run / 'training_statistics.json').read_text())
    if statistics != saved['training_statistics'] or statistics['source_split'] != 'train':
        raise ValueError('Saved train-only transition statistics differ from the checkpoint.')
    if not statistics['subjects'] or not set(statistics['subjects']).issubset(snapshot['split_subjects']['train']):
        raise ValueError('Transition statistics must use only training subjects.')
    dev = _device(device)
    model = build_temporal_model(cfg, dev)
    model.load_state_dict(saved['model_state_dict'], strict=True)
    return model, cfg, saved, arrays, metadata, cache, dev, statistics


def _runs(labels, frames, class_id, stop):
    """Held-decision intervals [frame_i, frame_i+1), clipped at recording end."""
    ends = np.r_[frames[1:], stop]
    intervals = []
    for left, right, label in zip(frames, ends, labels):
        if label != class_id or right <= left:
            continue
        if intervals and intervals[-1][1] == left:
            intervals[-1] = (intervals[-1][0], int(right))
        else:
            intervals.append((int(left), int(right)))
    return intervals


def episode_metrics(predictions, arrays, metadata, split_name, *, match_seconds=1.):
    """Episode matching uses held decisions, never cross-record state or future input."""
    from preprocess.kfall_labels import record_frames, labels_from_intervals
    selected = predictions['indices']
    files = arrays['file_idx'][selected]
    results, delays, leads, transition_errors = [], [], [], []
    false_alarms = 0
    normal_seconds = 0.
    transition_truth = transition_matched = 0
    records = metadata['recordings']
    for fi in metadata['split_file_indices'][split_name]:
        record = records[fi]
        mask = files == fi
        idx = selected[mask]
        frames = np.asarray(arrays['end_frame'][idx], dtype=np.int64)
        labels = predictions['predicted'][mask]
        frames_order = np.argsort(frames)
        frames, labels = frames[frames_order], labels[frames_order]
        native = record_frames(record)
        stop = int(native[-1]) + 1
        predicted_falls = _runs(labels, frames, 0, stop) if len(frames) else []
        fall = record['label_id'] == 0
        onset = int(record['fall_onset_frame']) if fall else None
        impact = int(record['fall_impact_frame']) if fall else None
        matches = [(left, right) for left, right in predicted_falls if fall and left < impact and right > onset]
        if fall:
            if matches:
                delays.append((matches[0][0] - onset) / 100)
                leads.append((impact - matches[0][0]) / 100)
            results.append(dict(file_index=fi, file=record['file'], subject=record['subject'],
                detected=bool(matches), observable=bool(len(frames) and frames[0] < impact),
                fragments=len(matches), latency_seconds=(matches[0][0] - onset) / 100 if matches else None))
        if not len(frames):
            continue
        # Integrate reference normal time over the observed part, independent of
        # predictions. Frame intervals were saved before window segmentation.
        grid = np.arange(frames[0], stop, dtype=np.int64)
        truth = labels_from_intervals(grid, record['frame_label_intervals'])
        normal_seconds += float(np.isin(truth, [1, 2]).sum()) / 100
        for left, right in predicted_falls:
            overlaps_fall = fall and left < impact and right > onset
            if not overlaps_fall and np.isin(truth[max(0, left - frames[0]):right - frames[0]], [1, 2]).any():
                false_alarms += 1
        predicted_changes = [(int(frames[i]), int(labels[i - 1]), int(labels[i]))
                             for i in range(1, len(frames)) if labels[i] != labels[i - 1]]
        used = set()
        intervals = record['frame_label_intervals']
        for previous, current in zip(intervals, intervals[1:]):
            moment, old, new = int(current[0]), int(previous[2]), int(current[2])
            if not frames[0] <= moment < stop:
                continue
            transition_truth += 1
            eligible = [(abs(frame - moment), position, frame - moment)
                        for position, (frame, a, b) in enumerate(predicted_changes)
                        if position not in used and (a, b) == (old, new) and abs(frame - moment) <= match_seconds * 100]
            if eligible:
                _, position, delta = min(eligible)
                used.add(position)
                transition_matched += 1
                transition_errors.append(delta / 100)
    detected = sum(row['detected'] for row in results)
    observable = sum(row['observable'] for row in results)
    return dict(fall_episodes=len(results), observable_fall_episodes=observable, detected_fall_episodes=detected,
        event_recall=detected / len(results) if results else None,
        normal_observed_hours=normal_seconds / 3600,
        false_fall_alarms=false_alarms,
        false_fall_alarms_per_hour=false_alarms * 3600 / normal_seconds if normal_seconds else None,
        fragments_per_detected_fall=float(np.mean([r['fragments'] for r in results if r['detected']])) if detected else None,
        mean_detection_latency_seconds=float(np.mean(delays)) if delays else None,
        mean_preimpact_lead_seconds=float(np.mean(leads)) if leads else None,
        transition_match_tolerance_seconds=match_seconds, true_transitions=transition_truth,
        matched_transitions=transition_matched,
        transition_mae_seconds=float(np.mean(np.abs(transition_errors))) if transition_errors else None,
        mean_transition_delay_seconds=float(np.mean(transition_errors)) if transition_errors else None,
        episodes=results,
        definitions='Held predictions; a fall alert matches an episode when its interval intersects onset..impact. Unmatched alerts overlapping Normal count as false alarms. No predictions before first complete window.')


def evaluate_temporal(run_dir, *, split_name='test', device='auto', cache_dir=None, output_dir=None):
    if split_name not in {'val', 'test'}:
        raise ValueError('Evaluation split must be val or test.')
    model, cfg, saved, arrays, metadata, cache, dev, stats = load_temporal_run(run_dir, device, cache_dir, split_name=split_name)
    episodes = EpisodeSet(arrays, metadata, boundary_ignore_seconds=cfg.boundary_ignore_seconds)
    if dev.type == 'cuda':
        torch.cuda.reset_peak_memory_stats(dev)
    started = time.perf_counter()
    measured = run_temporal_epoch(model, episodes, cfg, dev, stats, collect=True, phase=split_name.capitalize())
    seconds = time.perf_counter() - started
    p = measured.pop('predictions')
    destination = Path(output_dir) if output_dir else Path(run_dir) / 'metrics' / ('test_best' if split_name == 'test' else 'validation')
    if destination.resolve().is_relative_to(cache.resolve()):
        raise ValueError('Store reports outside the read-only event cache.')
    destination.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(destination / 'episode_predictions.npz', **p)
    keep = p['valid']
    if not keep.any():
        raise ValueError('No valid classified windows after boundary masking.')
    cohort = np.column_stack((arrays['file_idx'][p['indices']], arrays['end_frame'][p['indices']], p['target'], p['valid']))
    summary = {**measured, 'dataset': 'kfall', 'split': split_name, 'model_type': cfg.model_type,
        'temporal_protocol': cfg.temporal_protocol, 'head_mode': cfg.head_mode, 'memory': cfg.memory,
        'sequence_length': cfg.sequence_length, 'recurrent_hidden': cfg.recurrent_hidden,
        'run_dir': str(Path(run_dir).resolve()), 'checkpoint': str(Path(run_dir) / 'checkpoints/best_model.pt'),
        'best_epoch': saved['epoch'], 'best_val_loss': saved['best_val_loss'],
        'n_classes': 4, 'inference_time_sec': seconds, 'parameters': count_parameters(model)[0],
        'windows_per_second': len(p['indices']) / seconds, 'required_decisions_per_second': 100 / cfg.stride,
        'milliseconds_per_decision': seconds * 1000 / len(p['indices']),
        'inference_batch_recordings': cfg.batch_size,
        'timing_scope': 'Amortized batched PyTorch wall time, not single-stream hardware latency or energy.',
        'mean_recurrent_spikes_per_window': float(p['memory_spikes'].mean()),
        'cohort_sha256': hashlib.sha256(cohort.astype('<i8').tobytes()).hexdigest(),
        'data_provenance_sha256': data_provenance_sha256(metadata),
        'observed_windows': len(p['indices']), 'masked_windows': int((~keep).sum()),
        'state_reset': 'recording boundaries; gradients detached every K windows during train',
        'decision_rule': 'root then selected branch' if cfg.head_mode == 'hierarchical' else 'four-class argmax',
        'energy_measured': False, 'output_dir': str(destination)}
    if dev.type == 'cuda':
        summary['peak_cuda_allocated_mib'] = torch.cuda.max_memory_allocated(dev) / 2**20
    report = save_evaluation_reports(destination, y_true=p['target'][keep], probabilities=p['probability'][keep],
        predicted_classes=p['predicted'][keep], window_indices=p['indices'][keep], split=arrays,
        metadata=metadata, summary=summary)
    if cfg.head_mode == 'hierarchical':
        from evaluation.kfall_hierarchy import save_hierarchy_reports
        save_hierarchy_reports(destination / 'hierarchy', p['target'][keep],
            {key: p[key + '_probability'][keep] for key in ('group', 'normal', 'event')})
    event_report = episode_metrics(p, arrays, metadata, split_name)
    save_json(destination / 'event_metrics.json', event_report)
    if cfg.memory == 'lif':
        file_ids = arrays['file_idx'][p['indices']]
        transmitted = sum(float(p['memory_spikes'][file_ids == fi][:-1].sum()) for fi in np.unique(file_ids))
        report['estimated_recurrent_synaptic_events'] = transmitted * cfg.recurrent_hidden
        report['synaptic_estimate_scope'] = 'Recurrent all-to-all LIF connections only; excludes SCN, input projections and readouts. Not measured energy.'
    report['evaluation_scope'] = 'causal_stateful_' + split_name + '_episode_classification'
    report['event_metrics'] = {key: value for key, value in event_report.items() if key != 'episodes'}
    save_json(destination / 'metrics.json', report)
    # Files remain compatible with the simple result-inspection workflow.
    run_metrics = Path(run_dir) / 'metrics'
    if output_dir is None:
        save_json(run_metrics / ('test_metrics.json' if split_name == 'test' else 'validation_metrics.json'), report)
        np.save(run_metrics / ('y_true.npy' if split_name == 'test' else 'y_true_val.npy'), p['target'][keep])
        np.save(run_metrics / ('y_pred.npy' if split_name == 'test' else 'y_pred_val.npy'), p['predicted'][keep])
    return report
