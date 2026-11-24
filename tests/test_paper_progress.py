"""Progress must be visible without changing batches, results or command manifests."""

import contextlib
import io
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest

import numpy as np

from evaluate_paper_run import _progress_batches
from run_paper_experiments import _run_logged_command


class SmallLoader:
    dataset = range(5)

    def __init__(self):
        self.batches = [{"x": np.zeros((3, 6))}, {"x": np.zeros((2, 6))}]

    def __len__(self):
        return len(self.batches)

    def __iter__(self):
        return iter(self.batches)


class PaperProgressTests(unittest.TestCase):
    def test_batch_progress_counts_short_final_batch_and_preserves_objects(self):
        loader = SmallLoader()
        output = io.StringIO()
        with contextlib.redirect_stdout(output):
            batches = list(_progress_batches(loader, label="[paper:test]", every=3600))
        self.assertIs(batches[0], loader.batches[0])
        self.assertIs(batches[1], loader.batches[1])
        self.assertIn("batches=0/2 samples=0/5", output.getvalue())
        self.assertIn("batches=1/2 samples=3/5", output.getvalue())
        self.assertIn("batches=2/2 samples=5/5 progress=100.0%", output.getvalue())
        self.assertIn("rate=", output.getvalue())
        self.assertIn("ETA=", output.getvalue())

    def test_batch_is_not_reported_complete_until_processed_and_quiet_is_silent(self):
        loader = SmallLoader()
        output = io.StringIO()
        with contextlib.redirect_stdout(output):
            batches = _progress_batches(loader, label="test")
            self.assertIs(next(batches), loader.batches[0])
            batches.close()
        self.assertNotIn("batches=1/2", output.getvalue())
        output = io.StringIO()
        with contextlib.redirect_stdout(output):
            batches = list(_progress_batches(loader, label="test", enabled=False))
        self.assertEqual(len(batches), 2)
        self.assertEqual(output.getvalue(), "")

    def test_child_output_is_unbuffered_streamed_and_logged_without_mutating_command(self):
        command = [sys.executable, "-c", "import sys; print('unbuffered=' + str(sys.stdout.write_through)); print('epoch=1')"]
        original = list(command)
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "command.log"
            output = io.StringIO()
            with contextlib.redirect_stdout(output):
                _run_logged_command(command, path)
            self.assertEqual(path.read_text(), output.getvalue())
            self.assertIn("unbuffered=True", output.getvalue())
            self.assertIn("epoch=1", output.getvalue())
            self.assertEqual(command, original)
            output = io.StringIO()
            with contextlib.redirect_stdout(output):
                _run_logged_command(command, path, log_only=True)
            self.assertEqual(output.getvalue(), "")
            self.assertIn("epoch=1", path.read_text())

    def test_child_failure_is_not_reported_as_success(self):
        with tempfile.TemporaryDirectory() as directory, contextlib.redirect_stdout(io.StringIO()):
            path = Path(directory) / "failed.log"
            with self.assertRaises(subprocess.CalledProcessError) as error:
                _run_logged_command([sys.executable, "-c", "print('failed stage'); raise SystemExit(7)"], path)
            self.assertEqual(error.exception.returncode, 7)
            self.assertIn("failed stage", path.read_text())


if __name__ == "__main__":
    unittest.main()
