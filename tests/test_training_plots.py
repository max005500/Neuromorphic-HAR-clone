"""Verify training curves preserve epochs, missing values, and supervision."""

from __future__ import annotations

import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import numpy as np
from matplotlib.figure import Figure

from train.plotting import save_training_curves


class TrainingPlotTests(unittest.TestCase):
    def render(self, history):
        temporary = tempfile.TemporaryDirectory()
        self.addCleanup(temporary.cleanup)
        destination = Path(temporary.name) / "metrics"
        snapshots = {}
        original_savefig = Figure.savefig

        def capture(figure, filename, *args, **kwargs):
            snapshots[Path(filename).stem] = [
                {
                    "title": axis.get_title(),
                    "ylim": axis.get_ylim(),
                    "curves": [(np.array(line.get_xdata(), copy=True),
                                np.array(line.get_ydata(), dtype=float, copy=True))
                               for line in axis.lines],
                }
                for axis in figure.axes
            ]
            return original_savefig(figure, filename, *args, **kwargs)

        with patch.object(Figure, "savefig", capture):
            paths = save_training_curves(history, destination)
        return destination, paths, snapshots

    def assert_curve(self, axes, epochs, expected):
        matches = [
            (x, y) for axis in axes for x, y in axis["curves"]
            if np.array_equal(x, epochs)
            and y.shape == np.asarray(expected).shape
            and np.allclose(y, expected, equal_nan=True)
        ]
        self.assertTrue(matches, f"Expected curve {expected} at epochs {epochs}.")

    def test_single_head_outputs_pngs_with_recorded_epochs_and_values(self):
        history = [
            {"epoch": epoch, "train": {"loss": tr_loss, "acc": tr_acc},
             "val": {"loss": va_loss, "acc": va_acc}}
            for epoch, tr_loss, va_loss, tr_acc, va_acc in (
                (2, 1.3, 1.4, 0.4, 0.3),
                (5, 0.9, 1.0, 0.6, 0.55),
                (9, 0.5, 0.7, 0.8, 0.75),
            )
        ]
        destination, paths, figures = self.render(history)
        self.assertEqual(set(paths), {"accuracy", "loss"})
        for name, path in paths.items():
            self.assertEqual(Path(path), destination / f"{name}.png")
            self.assertEqual(Path(path).read_bytes()[:8], b"\x89PNG\r\n\x1a\n")
        self.assertEqual(len(figures["accuracy"]), 1)
        self.assert_curve(figures["accuracy"], [2, 5, 9], [0.4, 0.6, 0.8])
        self.assert_curve(figures["accuracy"], [2, 5, 9], [0.3, 0.55, 0.75])
        self.assert_curve(figures["loss"], [2, 5, 9], [1.3, 0.9, 0.5])
        self.assert_curve(figures["loss"], [2, 5, 9], [1.4, 1.0, 0.7])
        self.assertEqual(figures["accuracy"][0]["ylim"], (0.0, 1.0))

    def test_missing_and_nonfinite_metrics_remain_gaps(self):
        history = [
            {"epoch": 1, "train": {"loss": 1.2, "acc": 0.4},
             "val": {"loss": 1.3, "acc": 0.3}},
            {"epoch": 2, "train": {"loss": float("nan"), "acc": None},
             "val": {"loss": None}},
            {"epoch": 3, "train": {"loss": 0.6, "acc": 0.8},
             "val": {"loss": 0.8, "acc": 0.7}},
        ]
        _, _, figures = self.render(history)
        self.assert_curve(figures["accuracy"], [1, 2, 3], [0.4, np.nan, 0.8])
        self.assert_curve(figures["accuracy"], [1, 2, 3], [0.3, np.nan, 0.7])
        self.assert_curve(figures["loss"], [1, 2, 3], [1.2, np.nan, 0.6])
        self.assert_curve(figures["loss"], [1, 2, 3], [1.3, np.nan, 0.8])

    def test_supervised_dual_head_preserves_both_accuracy_curves(self):
        history = [
            {"epoch": 1,
             "train": {"loss": 1.0, "loss_h1": 0.8, "loss_h2": 0.2,
                       "acc_h1": 0.6, "acc_h2": 0.7, "n_h2": 5},
             "val": {"loss": 1.1, "loss_h1": 0.85, "loss_h2": 0.25,
                     "acc_h1": 0.5, "acc_h2": 0.65, "n_h2": 2}},
            {"epoch": 2,
             "train": {"loss": 0.7, "loss_h1": 0.6, "loss_h2": 0.1,
                       "acc_h1": 0.8, "acc_h2": 0.9, "n_h2": 6},
             "val": {"loss": 0.8, "loss_h1": 0.65, "loss_h2": 0.15,
                     "acc_h1": 0.75, "acc_h2": 0.85, "n_h2": 3}},
        ]
        _, _, figures = self.render(history)
        self.assertEqual(len(figures["accuracy"]), 2)
        for expected in ([0.6, 0.8], [0.5, 0.75], [0.7, 0.9], [0.65, 0.85]):
            self.assert_curve(figures["accuracy"], [1, 2], expected)
        self.assert_curve(figures["loss"], [1, 2], [0.2, 0.1])
        self.assert_curve(figures["loss"], [1, 2], [0.25, 0.15])

    def test_unsupervised_head_is_not_rendered_as_zero_accuracy(self):
        history = [
            {"epoch": epoch,
             "train": {"loss": 1.0, "loss_h1": 1.0, "loss_h2": 0.0,
                       "acc_h1": 0.6, "acc_h2": h2_accuracy, "n_h2": 0},
             "val": {"loss": 1.1, "loss_h1": 1.1, "loss_h2": 0.0,
                     "acc_h1": 0.5, "acc_h2": h2_accuracy, "n_h2": 0}}
            for epoch, h2_accuracy in ((1, float("nan")), (2, 0.0))
        ]
        _, _, figures = self.render(history)
        self.assertEqual(len(figures["accuracy"]), 1)
        for axis in figures["accuracy"] + figures["loss"]:
            for _, values in axis["curves"]:
                if values.shape == (2,):
                    self.assertFalse(np.all(values == 0.0), "Unsupervised head must not produce zero-valued curves.")

    def test_legacy_dual_history_can_omit_supervision_counts(self):
        history = [{"epoch": 1,
                    "train": {"loss": 1.0, "acc_h1": 0.6, "acc_h2": 0.7},
                    "val": {"loss": 1.1, "acc_h1": 0.5, "acc_h2": 0.65}}]
        _, _, figures = self.render(history)
        self.assertEqual(len(figures["accuracy"]), 2)
        self.assert_curve(figures["accuracy"], [1], [0.7])
        self.assert_curve(figures["accuracy"], [1], [0.65])

    def test_epoch_without_head2_targets_is_a_gap_in_supervised_curve(self):
        history = [
            {"epoch": epoch,
             "train": {"loss": 1.0, "acc_h1": 0.6, "acc_h2": acc, "n_h2": count},
             "val": {"loss": 1.1, "acc_h1": 0.5, "acc_h2": acc, "n_h2": count}}
            for epoch, acc, count in ((1, 0.7, 4), (2, 0.0, 0), (3, 0.9, 5))
        ]
        _, _, figures = self.render(history)
        self.assertEqual(len(figures["accuracy"]), 2)
        self.assert_curve(figures["accuracy"], [1, 2, 3], [0.7, np.nan, 0.9])

    def test_all_nonfinite_metrics_do_not_create_fabricated_zero_curves(self):
        history = [{"epoch": 1,
                    "train": {"loss": float("nan"), "acc": None},
                    "val": {"loss": float("inf"), "acc": float("nan")}}]
        _, _, figures = self.render(history)
        for axes in figures.values():
            for axis in axes:
                for _, values in axis["curves"]:
                    self.assertFalse(np.any(values == 0.0))

    def test_empty_history_does_not_create_charts(self):
        destination, paths, figures = self.render([])
        self.assertEqual(paths, {})
        self.assertEqual(figures, {})
        self.assertEqual(list(destination.glob("*.png")), [])


if __name__ == "__main__":
    unittest.main()
