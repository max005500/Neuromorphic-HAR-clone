# KFall Dataset

The KFall dataset is not included in this repository.

Request and download the original sensor data and temporal fall labels from the
[official KFall site](https://sites.google.com/view/kfalldataset), following its
access and usage conditions.

Extract the files into this directory with the following structure:

```text
data/KFall/
├── sensor_data/
│   ├── SA06/
│   │   ├── S06T20R01.csv
│   │   └── ...
│   └── ...
├── label_data/
│   ├── SA06_label.xlsx
│   └── ...
└── README.md
```

Both native filenames such as `S06T20R01.csv` and the documented form
`SA06T20R01.csv` are accepted. Fall onset and impact annotations are read from
the XLSX files in `label_data/` and matched against each CSV's `FrameCounter`.

## Usage

Run these commands from the repository root, with the project dependencies installed:

```bash
# Prepare and cache train/validation/test partitions without training.
python train_kfall.py --preprocess-only --workers 0

# Prepare 2.05 / 3.30 / 4.10-second windows with the same comparison endpoints.
python compare_kfall_windows.py --mode preprocess --workers 0

# Train using the prepared event windows.
python train_kfall.py --epochs 100 --batch-size 128 --workers 4

# Interpolate observed signals from 100 to 200 Hz before event encoding.
python train_kfall.py --preprocess-only --resample-hz 200 --workers 0

# Explore thresholds using validation macro-F1, without test inference.
python tune_kfall_thresholds.py --window-seconds 2.05 --resample-hz 200 \
  --endpoint-context 410 --epochs 3 --train-per-class 128 --device cpu --workers 0

# Compare original X/Y/Z signals with the cached Sigma-Delta events.
python view_kfall.py --class fall

# Compare the same endpoint across prepared 2.05 / 3.30 / 4.10-second variants.
python view_kfall_variants.py \
  --comparison results/kfall_window_comparison_200hz_calibrated/comparison.json \
  --split test --class fall

# Evaluate an existing run's best checkpoint on its held-out test subjects.
python evaluate_kfall.py --run-dir runs/kfall_scn_YYYYMMDD_HHMMSS

# Animate a full test recording with predicted and real fall labels.
python animate_kfall.py --run-dir runs/kfall_scn_YYYYMMDD_HHMMSS
```

Subjects are assigned to disjoint partitions before windowing. Default classes
are Fall, Dynamic, Static, and Postimpact. New runs default to
`--label-policy temporal`, with targets determined by each window's last sample:

- **No fall:** Dynamic from 1.00 second after the beginning until 1.00 second
  before the end; the first and last 100 native frame positions are Static.
  Dynamic spans `[first_frame + 100, last_frame + 1 - 100)` at 100 Hz.
  Recordings lasting at most 2.00 seconds remain entirely Static.
- **Fall:** Dynamic during the 1.20 seconds before onset and during the 2.00
  seconds after the **end of Postimpact**. Fall spans `[onset, impact)`;
  Postimpact spans `[impact, impact + 1.20 s)`. Everything else is Static.

Intervals include the beginning and exclude the end, are clipped to observations,
and use the native 100 Hz frame grid. Recording coverage is
`[first_frame, last_frame + 1)`. These Dynamic/Static targets are experimental
rules; the original XLSX annotations are read and never modified. Configure
margins with `--adl-margin-seconds`, `--pre-fall-seconds`, and
`--post-recovery-seconds`.

`--label-policy motion` is an optional sensor-based alternative: outside Fall
and Postimpact, trailing variation determines Dynamic/Static. The initial
heuristics use 0.25 s lookback, 0.08 g accelerometer axis range, or 10 degrees/s
gyroscope axis range. They are not calibrated labels and are distinct from
encoder thresholds. Configure `--motion-window-seconds`,
`--motion-acc-threshold`, and `--motion-gyr-threshold` explicitly when exploring
this mode.

Existing models retain their saved label policy; missing policy metadata means
`legacy`. To reproduce original task-family labels and the original three-class
task, use `--label-policy legacy --post-impact-seconds 0`. New labels require
new caches and retraining; old model scores do not evaluate the updated targets.
Generated event caches are stored under `data/kfall_cache/`, with label settings
included in their identity.

Prepare or train updated one-second windows:

```bash
python train_kfall.py --preprocess-only \
  --window-seconds 1.00 --resample-hz 200 --endpoint-context 410 \
  --theta-file configs/kfall_theta_200hz.json \
  --label-policy temporal --post-impact-seconds 1.20 --workers 0

python train_kfall.py \
  --window-seconds 1.00 --resample-hz 200 --endpoint-context 410 \
  --theta-file configs/kfall_theta_200hz.json \
  --label-policy temporal --post-impact-seconds 1.20 \
  --epochs 100 --batch-size 32 --workers 0 --device auto \
  --run kfall_1p00s_200hz_temporal
```

The comparison context of 410 samples preserves common endpoints with longer
models. Use `--endpoint-context 100` to include natural one-second coverage.
Train all compared durations using the same new label policy.

The 200 Hz stage inserts intermediate values; acquisition and annotation frames
remain at 100 Hz. It does not create independent measurements or new recordings.
The event encoder retains its separate temporal grid. Threshold searches save
`best_theta.json`, which can be supplied to training with `--theta-file` while
keeping the same window, subject split, interpolation, and encoder settings.
The portable preset [`configs/kfall_theta_200hz.json`](../../configs/kfall_theta_200hz.json)
is included with the source code. Its calibration used legacy labels; the new
temporal targets require new validation before claiming optimal thresholds.
Files under `results/`, including generated
`best_theta.json` reports, are ignored by Git. From the repository root, use
`--theta-file configs/kfall_theta_200hz.json --resample-hz 200`; select the window
duration and endpoint context explicitly as described in the project README.

The variant viewer shows all durations together or one at a time, with X/Y/Z
signals, intermediate 200 Hz samples, reconstruction, and ON/OFF events. It
offers subject, activity, split, and sensor filters. Generate its manifest with
`compare_kfall_windows.py --mode preprocess --resample-hz 200 --theta-file
configs/kfall_theta_200hz.json --results-dir
results/kfall_window_comparison_200hz_calibrated` before opening the viewer.
Use `--save results/kfall_variant_viewer/comparison.png` for a headless export.

Only this README is versioned from `data/KFall/`; downloaded sensor files and
label workbooks are ignored by Git. See the [project README](../../README.md#kfall)
for channel selection, encoding parameters, task grouping, and evaluation limits.

## Citation

```bibtex
@article{yu2021kfall,
  title={A Large-Scale Open Motion Dataset (KFall) and Benchmark Algorithms for Detecting Pre-impact Fall of the Elderly Using Wearable Inertial Sensors},
  author={Yu, Xiaoqun and Jang, Jaehyuk and Xiong, Shuping},
  journal={Frontiers in Aging Neuroscience},
  volume={13},
  pages={692865},
  year={2021},
  doi={10.3389/fnagi.2021.692865}
}
```

### Whole-record Sigma-Delta pseudo-labels

`--label-policy spikes` encodes each complete recording once on a fixed 200 Hz
label grid, using the selected theta/dead-zone/init. Only accelerometer ON/OFF events (AccX/AccY/AccZ) form a cluster if consecutive gaps are at most 0.10 s. Clusters
whose first-to-last span is strictly greater than 0.30 s are Dynamic; remaining
unprotected samples (including no events and brief/isolated events) are Static.
Fall and Postimpact take priority and split clusters. Configure with
`--spike-label-hz`, `--spike-max-gap-seconds`, `--spike-min-duration-seconds`.

Labels are generated offline before windowing and saved as frame intervals.
A cluster may retroactively label its beginning; model inputs remain causal.
Changing encoder thresholds can change these pseudo-labels. Keep encoder/label
settings fixed for window comparisons; they are not independent ground truth
for threshold optimization. The main README includes the preparation command.

### Interactive gyroscope encoder thresholds

From the repository root run `python tune_kfall_gyro.py --cache-dir CACHE` (or
`--run-dir TRAINING_RUN`). This Matplotlib tool adjusts GyrX/GyrY/GyrZ theta in
**degrees/s**, shows measured signals, baseline/current reconstruction and ON/OFF
events, and reports per-axis RMSE/counts. Accelerometer thresholds remain fixed.
Log sliders, exact numeric inputs, linked axes and JSON export are available.
Start with train, review validation, and reserve test for final evaluation.

The exported `configs/kfall_theta_gyro_custom.json` works with `--theta-file` in
training/comparison. Preserve the remaining encoder settings; theta-file loads
only the six thresholds. Existing caches are not modified. See the main README
for commands and the distinction between theta reconstruction step and
`dead-zone * theta` event trigger.
