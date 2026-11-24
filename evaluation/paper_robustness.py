"""Re-encode perturbed native windows without changing trained parameters."""

import numpy as np
import torch
from torch.utils.data import Dataset

from evaluation.paper_metrics import perturb_raw_signal
from preprocess.frontends import encode_window
from preprocess.interpolate_signal import derive_output_length


class PerturbedWindowDataset(Dataset):
    def __init__(self, dataset, cfg, perturbation, frames=None):
        self.dataset = dataset
        self.cfg = cfg
        self.out_win = cfg.out_win if cfg.out_win is not None else derive_output_length(cfg.raw_win, cfg.ups)
        self.perturbation = dict(perturbation)
        self.frames = frames
        expected_rate = getattr(cfg, "sample_rate_hz", 100 if getattr(cfg, "dataset", None) == "kfall" else 200)
        if self.perturbation.get("sample_rate_hz") != expected_rate:
            raise ValueError(f"Perturbation sample_rate_hz must match native {expected_rate} Hz.")
        if not len(dataset):
            raise ValueError("Cannot perturb an empty evaluation dataset.")
        self._encode(0)

    def __len__(self):
        return len(self.dataset)

    def _encode(self, index):
        item = dict(self.dataset[index])
        options = dict(self.perturbation)
        options["seed"] = int(options.get("seed", 0)) + int(index)
        signal = perturb_raw_signal(np.asarray(item["x"]), **options)
        if self.frames is not None:
            from loaders.kfall_event_loader import KFallEventLoaderConfig, signal_to_events
            config = KFallEventLoaderConfig(raw_win=self.cfg.raw_win, out_win=self.out_win,
                                            input_mode=self.cfg.input_mode, theta_sd=tuple(self.cfg.theta),
                                            dead_zone=self.cfg.dead_zone, sd_init=self.cfg.sd_init)
            encoded = signal_to_events(signal, self.frames[index], config)[0]
        else:
            encoded = encode_window(signal, mode=self.cfg.input_mode, out_win=self.out_win,
                                    theta=self.cfg.theta, dead_zone=self.cfg.dead_zone, sd_init=self.cfg.sd_init)
        item["x"] = torch.from_numpy(np.asarray(encoded, dtype=np.float32))
        item["offset"] = torch.from_numpy(np.asarray(signal[0], dtype=np.float32))
        return item

    def __getitem__(self, index):
        return self._encode(index)
