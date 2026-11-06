#!/usr/bin/env python3
"""List or replay KFall recordings with explicit recurrent-state boundaries."""
from __future__ import annotations
import argparse
from collections import OrderedDict
import copy
from pathlib import Path
import numpy as np
import torch
from animate_kfall import KFallReplayPlayer
from evaluation.kfall_temporal import load_temporal_run
from loaders.kfall_sequences import EpisodeSet
from train.temporal import run_temporal_epoch
from visualization.kfall_data import KFallWindowStore, source_frame_labels
from visualization.kfall_replay_data import ReplayData, _matches_recording, _validate_source
from preprocess.preprocess_kfall import load_record


SPLITS = ('train', 'val', 'test')


def _select_episodes(episodes, metadata, *, recording=None, subject=None, split_name='test'):
    selected = [ep for ep in episodes.episodes if
                (recording is None or _matches_recording(metadata['recordings'][ep.file_index], recording)) and
                (subject is None or metadata['recordings'][ep.file_index]['subject'].upper() == subject.upper())]
    if not selected:
        subjects = sorted({metadata['recordings'][ep.file_index]['subject'] for ep in episodes.episodes})
        reason = f'No eligible {split_name} recording matches the requested selection.'
        if recording is not None:
            matches = [fi for fi, record in enumerate(metadata['recordings']) if _matches_recording(record, recording)]
            if not matches:
                reason += f' Recording {recording!s} is absent from this run\'s recording manifest.'
            else:
                partitions = [key for key, files in metadata.get('split_file_indices', {}).items()
                              if any(fi in files for fi in matches)]
                if partitions and split_name not in partitions:
                    reason += f' It belongs to {", ".join(partitions)}; select --split {partitions[0]} to inspect it.'
                elif subject is not None:
                    reason += ' Check the --subject filter and whether the recording has a complete saved window.'
                else:
                    reason += ' It has no complete window under the saved window/context settings.'
        raise ValueError(reason + f' Available {split_name} subjects: {", ".join(subjects)}. '
                         f'Use --list-recordings --split {split_name} to list eligible files.')
    return selected


def list_temporal_recordings(run_dir, *, split_name='test', subject=None):
    """List the actual saved cohort without GPU inference, source CSVs or GUI."""
    if split_name not in SPLITS:
        raise ValueError('Replay split must be train, val or test.')
    _, cfg, _, arrays, metadata, _, _, _ = load_temporal_run(run_dir, 'cpu', split_name=split_name)
    episodes = EpisodeSet(arrays, metadata, boundary_ignore_seconds=cfg.boundary_ignore_seconds)
    selected = _select_episodes(episodes, metadata, subject=subject, split_name=split_name)
    return [dict(file=str(metadata['recordings'][ep.file_index]['file']).replace('\\', '/').rsplit('/', 1)[-1],
                 subject=metadata['recordings'][ep.file_index]['subject'], windows=len(ep.indices),
                 split=split_name) for ep in selected]


def prepare_temporal_replay(run_dir, *, recording=None, subject=None, device='auto', dataset_root=None,
                            split_name='test'):
    if split_name not in SPLITS:
        raise ValueError('Replay split must be train, val or test.')
    loaded = load_temporal_run(run_dir, device, split_name=split_name)
    return _replay_from_loaded(run_dir, loaded, recording=recording, subject=subject,
                               dataset_root=dataset_root, split_name=split_name)


def _replay_from_loaded(run_dir, loaded, *, recording=None, subject=None, dataset_root=None,
                        split_name='test', episodes=None):
    model, cfg, saved, arrays, metadata, cache, dev, stats = loaded
    episodes = (EpisodeSet(arrays, metadata, boundary_ignore_seconds=cfg.boundary_ignore_seconds)
                if episodes is None else copy.copy(episodes))
    selected = _select_episodes(episodes, metadata, recording=recording, subject=subject, split_name=split_name)
    ep = next((ep for ep in selected if metadata['recordings'][ep.file_index]['label_id'] == 0), selected[0])
    record = metadata['recordings'][ep.file_index]
    store = KFallWindowStore(cache, dataset_root=dataset_root)
    path = store._record_path(record)
    raw, frames, source = load_record(path)
    # Validate cached inputs, then recompute whole-record labels exactly once.
    # get_window() also recomputes those labels: calling it for every window
    # would repeatedly encode the complete recording.
    _validate_source(raw, frames, source, record, ep.file_index, arrays, store.cfg)
    true = source_frame_labels(raw, frames, record, store.cfg)
    episodes.episodes = [ep]
    result = run_temporal_epoch(model, episodes, cfg, dev, stats, collect=True)['predictions']
    times = (frames - frames[0]) / 100
    rows = np.asarray(arrays['start'][result['indices']]) + cfg.raw_win - 1
    onset = (record['fall_onset_frame'] - frames[0]) / 100 if record['label_id'] == 0 else None
    impact = (record['fall_impact_frame'] - frames[0]) / 100 if record['label_id'] == 0 else None
    return ReplayData(record=record, path=path, raw=raw, frames=frames, times=times,
        endpoint_rows=rows, endpoint_times=times[rows], probabilities=result['probability'],
        predicted_classes=result['predicted'], raw_win=cfg.raw_win, out_win=cfg.out_win,
        prediction_stride=cfg.stride, checkpoint=Path(run_dir) / 'checkpoints/best_model.pt',
        best_epoch=saved['epoch'], best_val_loss=saved['best_val_loss'], config={**vars(cfg), 'replay_split': split_name},
        cache_dir=cache, subject=record['subject'], label='temporal', fall_onset_time=onset,
        fall_impact_time=impact, post_impact_seconds=cfg.post_impact_seconds,
        post_impact_end_time=impact + cfg.post_impact_seconds if impact is not None else None,
        true_labels=true, hierarchy_probabilities={key: result[key + '_probability']
            for key in ('group', 'normal', 'event')} if cfg.head_mode == 'hierarchical' else None)


class TemporalReplaySource:
    """One fixed checkpoint, lazy split arrays and a small recording replay cache."""
    def __init__(self, run_dir, *, device='auto', dataset_root=None, split_name='test', max_cached_replays=3):
        if split_name not in SPLITS or max_cached_replays < 1:
            raise ValueError('Invalid replay partition or cache size.')
        self.run_dir, self.dataset_root = Path(run_dir), dataset_root
        self._base = load_temporal_run(run_dir, device, split_name=split_name)
        self._loaded = {split_name: self._base}
        self._episodes = {}
        self._replays = OrderedDict()
        self.max_cached_replays = max_cached_replays

    def _context(self, split_name):
        if split_name not in SPLITS:
            raise ValueError('Replay split must be train, val or test.')
        if split_name not in self._loaded:
            from evaluation.kfall_data import load_evaluation_data
            model, cfg, saved, _, _, cache, dev, stats = self._base
            arrays, metadata, _, cache = load_evaluation_data(self.run_dir, saved['config'], cache,
                                                             split_name=split_name)
            self._loaded[split_name] = (model, cfg, saved, arrays, metadata, cache, dev, stats)
        loaded = self._loaded[split_name]
        if split_name not in self._episodes:
            self._episodes[split_name] = EpisodeSet(loaded[3], loaded[4],
                boundary_ignore_seconds=loaded[1].boundary_ignore_seconds)
        return loaded, self._episodes[split_name]

    def records(self, split_name):
        loaded, episodes = self._context(split_name)
        metadata = loaded[4]
        return [dict(file=str(metadata['recordings'][ep.file_index]['file']).replace('\\', '/').rsplit('/', 1)[-1],
                     subject=metadata['recordings'][ep.file_index]['subject'], windows=len(ep.indices),
                     split=split_name) for ep in episodes.episodes]

    def replay(self, split_name, *, recording=None, subject=None):
        loaded, episodes = self._context(split_name)
        selected = _select_episodes(episodes, loaded[4], recording=recording, subject=subject, split_name=split_name)
        ep = next((ep for ep in selected if loaded[4]['recordings'][ep.file_index]['label_id'] == 0), selected[0])
        key = (split_name, ep.file_index)
        if key not in self._replays:
            self._replays[key] = _replay_from_loaded(self.run_dir, loaded,
                recording=loaded[4]['recordings'][ep.file_index]['file'], dataset_root=self.dataset_root,
                split_name=split_name, episodes=episodes)
            if len(self._replays) > self.max_cached_replays:
                self._replays.popitem(last=False)
        self._replays.move_to_end(key)
        return self._replays[key]


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--run-dir', type=Path, required=True)
    parser.add_argument('--split', choices=SPLITS, default='test', help='Saved partition to inspect; default test.')
    parser.add_argument('--list-recordings', action='store_true', help='List eligible CSVs without a GUI or prediction pass.')
    parser.add_argument('--recording')
    parser.add_argument('--subject')
    parser.add_argument('--dataset-root', type=Path)
    parser.add_argument('--device', default='auto')
    parser.add_argument('--sensor', choices=['acc', 'gyro'], default='acc')
    parser.add_argument('--view-seconds', type=float, default=6.)
    parser.add_argument('--speed', type=float, default=1.)
    parser.add_argument('--paused', action='store_true')
    parser.add_argument('--time', type=float, default=0.)
    parser.add_argument('--save-frame', type=Path)
    args = parser.parse_args(argv)
    if args.list_recordings:
        try:
            rows = list_temporal_recordings(args.run_dir, split_name=args.split, subject=args.subject)
            print(f'[{args.split}] recordings={len(rows)} | subjects={", ".join(sorted({row["subject"] for row in rows}))}')
            for row in rows:
                print(f'{row["file"]}\t{row["subject"]}\twindows={row["windows"]}')
        except (ValueError, OSError, RuntimeError, KeyError) as exc:
            parser.error(str(exc))
        return
    import matplotlib
    if args.save_frame:
        matplotlib.use('Agg')
    elif matplotlib.get_backend().lower() in {'agg', 'pdf', 'svg', 'ps', 'cairo', 'template'}:
        parser.error('Se necesita un backend gráfico; usa --save-frame para exportar sin escritorio.')
    from matplotlib import pyplot as plt
    try:
        if args.save_frame:
            replay = prepare_temporal_replay(args.run_dir, recording=args.recording, subject=args.subject,
                device=args.device, dataset_root=args.dataset_root, split_name=args.split)
            player = KFallReplayPlayer(replay, sensor=args.sensor, view_seconds=args.view_seconds,
                                      speed=args.speed, interactive=False)
        else:
            from visualization.kfall_temporal_browser import TemporalReplayBrowser
            source = TemporalReplaySource(args.run_dir, device=args.device, dataset_root=args.dataset_root,
                                           split_name=args.split)
            replay = source.replay(args.split, recording=args.recording, subject=args.subject)
            player = TemporalReplayBrowser(source, replay, sensor=args.sensor, view_seconds=args.view_seconds,
                                            speed=args.speed, subject=args.subject)
        player.seek(args.time)
        if args.paused:
            player.playing = False; player._render()
        if args.save_frame:
            player.save_frame(args.save_frame); plt.close(player.figure)
        else:
            plt.show()
    except (ValueError, OSError, RuntimeError, KeyError) as exc:
        parser.error(str(exc))


if __name__ == '__main__':
    main()
