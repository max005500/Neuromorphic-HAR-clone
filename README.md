# Neuromorphic HAR

Event-based IMU encoding and spiking neural network training for KFall, SisFall, and UCI HAR.

The repository provides scripts to:

- generate Sigma-Delta event tensors,
- load KFall sensor CSVs and temporal fall annotations from XLSX,
- partition recordings by subject before creating event windows,
- train a KFall SCN with fall, dynamic, static, and post-impact labels,
- compare 2.05, 3.30, and 4.10-second contexts with matched subjects and endpoints,
- autotag SisFall static-posture windows,
- optimize Sigma-Delta theta values,
- train SisFall dual-head or single-head SCN models,
- train UCI HAR single-head SCN models,
- estimate normalized inference-stage energy consumption.

## Environment

Recommended:

- Python 3.10 or 3.11
- PyTorch 2.x
- CUDA GPU for full training runs; CPU is enough for preprocessing and small tests

Virtual environment:

```bash
python -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
pip install -r requirements.txt
```

Conda:

```bash
conda env create -f environment.yml
conda activate neuromorphic-har
```

For CUDA, install the PyTorch build that matches your driver and platform.

# Data layout

Datasets are not included in this repository. Download them manually and place them under `data/`.

```text
data/
  KFall/
    sensor_data/
      SA06/
        S06T20R01.csv
    label_data/
      SA06_label.xlsx
  SisFall_dataset/
  labels_transitions/
  UCI HAR Dataset/
```

## KFall

Download the sensor data and temporal labels from the [official KFall site](https://sites.google.com/view/kfalldataset). Place the extracted subject folders under `data/KFall/sensor_data/` and their XLSX workbooks under `data/KFall/label_data/`. `--label-dir` can override the label directory. See [Yu, Jang, and Xiong (2021)](https://doi.org/10.3389/fnagi.2021.692865) for the acquisition protocol and task descriptions.

Native filenames such as `S06T20R01.csv` and the documented form `SA06T20R01.csv` are both accepted. The loader reads `AccX`, `AccY`, `AccZ`, `GyrX`, `GyrY`, and `GyrZ` as floating-point values in **g** and **degrees per second**. It ignores Euler angles. It matches recordings to each subject's task/trial annotations and resolves `Fall_onset_frame` and `Fall_impact_frame` against the CSV's `FrameCounter`; frame numbers are not assumed to be row offsets. Missing, duplicate, or inconsistent fall annotations raise an error.

The activity families follow the repository's SisFall grouping: `Static` includes stable postures and posture changes. Static task IDs are `1, 2, 3, 5, 11–19`; dynamic IDs are `4, 6–10, 35, 36`; fall trials have IDs `20–34`. This grouping and the post-impact class are adaptations for the model, not native KFall activity labels.

Each window is labelled by its **last observed sample**. The default four-class task uses:

| ID | Class | Target at the window endpoint |
|---|---|---|
| 0 | `Fall` | `onset <= frame < impact` |
| 1 | `Dynamic` | Dynamic ADL task family |
| 2 | `Static` | Static/posture ADL task family |
| 3 | `Postimpact` | `impact <= frame < impact + 120` at 100 Hz |

`Postimpact` lasts 1.20 seconds from the annotated impact, including the impact
sample and excluding the end boundary. A post-impact input window can also
contain earlier falling motion; the endpoint determines its target. Pre-onset
parts of fall trials and samples after the post-impact interval have no inferred
ADL class and are excluded from this window-label protocol. Short recordings
are clipped to their available samples; no future data or padding is added.

The default context uses 205 samples at 100 Hz, stride 102, and 2048 event steps.
`--window-seconds` selects a longer context, preserving approximately the same
encoder time scale by default:

| Nominal context | Raw samples | Event steps | Event channels | Offsets |
|---|---|---|---|---|
| 2.05 s | 205 | 2048 | 12 | 6 |
| 3.30 s | 330 | 3297 | 12 | 6 |
| 4.10 s | 410 | 4096 | 12 | 6 |

`--out-win` overrides the event length. Interpolation adds no measured sensor
information. Nominal durations count samples; first-to-last sample spans are
one sampling interval shorter (e.g. 2.04 s for 205 samples at 100 Hz).

Fall windows exclude the impact sample. Postimpact windows receive class 3.
`--post-impact-seconds` changes the post-impact interval; `0` restores the legacy
three-class task and its original cache naming. The optional
`--fall-policy annotated_interval` is available only with post-impact disabled,
because assigning the impact to both classes would be ambiguous.

Evaluation uses annotation-selected windows, including phase-end anchors. Cached
`lead_time_ms` is the signed time from an endpoint to impact: positive for Fall,
zero at impact, and negative for later Postimpact windows. It is not a measured
warning time. Window accuracy alone does not establish continuous alarm quality.

KFall has no temporal **Stable Posture / Postural Transition** annotations.
`train_kfall.py` defaults to a four-class single head. The optional
`--head-mode dual` keeps the activity head with four outputs and freezes the
unsupervised posture head. Existing three-class checkpoints remain usable with
their original caches; four-class experiments train a new classifier.

The default thresholds `0.1 0.1 0.1 10 10 10` are starting values in KFall's physical units, **not optimized KFall parameters**. Do not reuse SisFall thresholds, which operate on raw sensor counts. Tune thresholds and model choices using training/validation subjects and reserve test subjects for final evaluation. This implementation does not reproduce the paper's reported accuracy or energy savings on KFall.

## SisFall

Download:

```text
https://www.kaggle.com/datasets/nvnikhil0001/sis-fall-original-dataset
```

Place the extracted subject folders inside:

```text
data/SisFall_dataset/
```

Reference:

```bibtex
@article{sucerquia2017sisfall,
  title={SisFall: A Fall and Movement Dataset},
  author={Sucerquia, Angela and L{\'o}pez, Jos{\'e} David and Vargas-Bonilla, Jes{\'u}s Francisco},
  journal={Sensors},
  volume={17},
  number={1},
  pages={198},
  year={2017},
  doi={10.3390/s17010198}
}
```

## UCI HAR

Download:

```text
https://archive.ics.uci.edu/dataset/240/human+activity+recognition+using+smartphones
```

Place the extracted dataset contents inside:

```text
data/UCI HAR Dataset/
```

Reference:

```bibtex
@inproceedings{anguita2013public,
  title={A Public Domain Dataset for Human Activity Recognition Using Smartphones},
  author={Anguita, Davide and Ghio, Alessandro and Oneto, Luca and Parra, Xavier and Reyes-Ortiz, Jorge L.},
  booktitle={Proceedings of the 21st European Symposium on Artificial Neural Networks, Computational Intelligence and Machine Learning},
  pages={437--442},
  year={2013}
}
```

## Help

All scripts expose their options through argparse:

```bash
python posture_tagging.py --help
python optimize_theta.py --help
python tune_kfall_thresholds.py --help
python train_sisfall.py --help
python train_kfall.py --help
python train_ucihar.py --help
python plot_training.py --help
python view_kfall.py --help
python evaluate_kfall.py --help
python animate_kfall.py --help
python compare_kfall_windows.py --help
python energy_report.py --help
```

## SisFall autotagging

```bash
python posture_tagging.py 
```

Useful parser options include `--refresh`, `--save-plot`, `--th-pt`, `--th-sp`, and `--margin`.

## Theta optimization

KFall, with the same subject split and temporal window rule used for training:

```bash
python optimize_theta.py --dataset kfall --data-root data --ups 10 --dead-zone 0.5 \
  --split 0.8 0.1 0.1 --seed 0 --fall-policy pre_impact \
  --window-seconds 2.05 --post-impact-seconds 1.20 --resample-hz 200
```

For KFall, UPS 10 scales the original 205-to-2048 ratio to the selected duration;
`--out-len` overrides it. Keep window duration, post-impact interval, endpoint
context, stride, subject split, and encoding settings consistent when transferring
the resulting six values to `train_kfall.py --theta ...`. Optimization uses
training/validation subjects for selection and reports test reconstruction error
afterward. Use `--post-impact-seconds 0` for the legacy protocol.

This optimizer minimizes reconstruction error, not the SNN classification error.
For thresholds selected by classification performance, use the validation search:

```bash
python tune_kfall_thresholds.py --window-seconds 2.05 --resample-hz 200 \
  --endpoint-context 410 --epochs 3 --train-per-class 128 --device cpu --workers 0
```

The search tests combinations of accelerometer and gyroscope threshold scales,
starting from `[0.1, 0.1, 0.1, 10, 10, 10]` in g and degrees/s. Each candidate
uses theta as the reconstruction step; `--dead-zone 0.5` emits an event when
the absolute difference from the reconstruction exceeds half that step.
Every candidate
trains a fresh SNN with the same initialization, subject partitions, and sampled
training windows. Validation retains all its windows. Candidates run with
`--validation-only`; test inference is reserved for the chosen configuration's
final evaluation. Selection uses validation macro-F1 by default, so every class
contributes equally. This is the best candidate in a finite search, not a claim
of a global optimum. A small training subset makes exploration faster; confirm
the selected thresholds using full training and additional seeds.

Search outputs include candidate metrics, a comparison plot, and
`best_theta.json`. Apply the saved six thresholds in a full training run using
`train_kfall.py --theta-file PATH/TO/best_theta.json --resample-hz 200`, preserving
the search window duration, endpoint context, split, and encoding settings.

The repository includes an exploratory calibration preset at
[`configs/kfall_theta_200hz.json`](configs/kfall_theta_200hz.json):
`0.025 0.025 0.025 1.25 1.25 1.25` (g for acceleration, degrees/s for rotation).
Generated files under `results/` are ignored by Git and are not available in a
fresh checkout. The preset contains the six selected thresholds and their
calibration conditions; `--theta-file` loads only the thresholds, so keep the
other settings explicit. Run from the repository root:

```bash
python compare_kfall_windows.py --mode preprocess \
  --durations 2.05 3.30 4.10 --resample-hz 200 \
  --theta-file configs/kfall_theta_200hz.json \
  --post-impact-seconds 1.20 --endpoint-policy matched --workers 0 \
  --results-dir results/kfall_window_comparison_200hz_calibrated
```

Use `--mode train --epochs 3 --batch-size 32 --device cpu` with the same settings
to train the three variants. The preset was calibrated on 2.05-second windows;
applying the same thresholds to all durations keeps that variable fixed for
the window comparison. Supply the values directly with
`--theta 0.025 0.025 0.025 1.25 1.25 1.25` if a JSON file is inconvenient.

Refine around a previous winner, or start from an `optimize_theta.py` result,
using `--theta-file` instead of six explicit `--theta` values. For example:

```bash
python tune_kfall_thresholds.py \
  --theta-file results/kfall_threshold_search/best_theta.json \
  --acc-scales 0.75 1 1.25 --gyro-scales 0.75 1 1.25 \
  --window-seconds 2.05 --resample-hz 200 --endpoint-context 410 \
  --epochs 3 --train-per-class 128 --device cpu --workers 0 \
  --reuse-completed --results-dir results/kfall_threshold_refinement
```

## KFall interpolation at 200 Hz

```bash
python train_kfall.py --preprocess-only --resample-hz 200 --workers 0
python compare_kfall_windows.py --mode preprocess --resample-hz 200 --workers 0
```

`--resample-hz 200` adds a linear interpolation stage inside each observed
window. Acquisition and annotation frame counters stay at 100 Hz; interpolated
timestamps are spaced by 5 ms. Both endpoints remain observed samples, so native
windows of 205, 330, and 410 samples produce 409, 659, and 819 intermediate
values, respectively. The nominal 2.05-second window spans 2.04 seconds between
its first and last sample. No samples beyond the prediction endpoint are used.

The intermediate analog grid and the event encoder grid are separate. Event
lengths remain 2048, 3297, and 4096 unless explicitly overridden. The original
pipeline already interpolated directly from 100 Hz to this finer event grid;
linear interpolation through a 200 Hz grid gives the same underlying curve,
apart from rounding. It adds no independent measurements or recordings and does
not increase the sensor's acquisition frequency. Caches record the extra stage
and use distinct names. The viewer also displays the intermediate samples.

## Theta optimization for SisFall and UCI HAR

SisFall:

```bash
python optimize_theta.py --dataset sisfall --data-root data --ups 5 --dead-zone 0.5
```

UCI HAR:

```bash
python optimize_theta.py --dataset ucihar --data-root data --ups 5 --dead-zone 0.5
```

Fast debug run:

```bash
python optimize_theta.py --dataset sisfall --data-root data --ups 5 --dead-zone 0.5 \
  --train-per-class 16 --val-per-class 8 --test-per-class 8 \
  --sampled-test-only --max-epochs 2 --workers 1
```

## Training

KFall, with disjoint subjects and four endpoint-labelled classes:

```bash
python train_kfall.py --dataset-root data/KFall \
  --split 0.8 0.1 0.1 --seed 0 \
  --window-seconds 2.05 --post-impact-seconds 1.20 \
  --epochs 100 --batch-size 128 --workers 4
```

Build the event cache and inspect the split metadata without training:

```bash
python train_kfall.py --preprocess-only --workers 0
```

New default caches have names identifying their context, event length, and
post-impact interval, keeping the original `kfall_event_loader_default` cache
separate. Use `--post-impact-seconds 0` to reproduce the old three-class setup.

The default subject proportions are approximately 80% training, 10% validation, and 10% test: 26/3/3 participants for all 32 KFall subjects. Subject counts are rounded to whole participants, so window counts need not follow those proportions. Every recording and overlapping window from a participant stays in that participant's split. The split and source-recording metadata are saved in the cache and copied into each training run's `dataset_metadata.json`; the seed makes the partition reproducible. Changed split/encoding settings invalidate the event cache.

SisFall also splits whole subjects by default. UCI HAR defaults to its official test subjects and holds out whole training subjects for validation (`--val-ratio`); `--case subject_split --split 0.8 0.1 0.1` repartitions all UCI subjects. Existing event caches from the previous recording/window split are rebuilt. Each cache records `split_strategy: "subject"` and the disjoint `split_subjects` lists.

SisFall dual-head:

```bash
python train_sisfall.py --head-mode dual --epochs 100 --batch-size 512 --workers 4
```

SisFall single-head:

```bash
python train_sisfall.py --head-mode single --epochs 100 --batch-size 512 --workers 4
```

UCI HAR:

```bash
python train_ucihar.py --epochs 100 --batch-size 256 --workers 4
```

Example parameter entry:

```bash
python train_sisfall.py --head-mode dual --tau 0.75 --thresh 0.5 \
  --conv-ch 32 64 64 --kernels 32 32 8 --strides 4 2 1 \
  --p-drop 0.35 --dead-zone 0.5
```

Each training run creates a separate folder under `runs/`.

All training scripts save `metrics/accuracy.png` and `metrics/loss.png` at the
end of training, including early stopping. Each plot compares training and
validation across the recorded epochs. Dual-head runs include separate panels
for supervised heads; the unannotated KFall posture head is omitted.

Generate or refresh plots for existing runs without retraining:

```bash
# All runs with a saved metrics/history.json.
python plot_training.py

# A specific run (a metrics directory or history.json path is also accepted).
python plot_training.py runs/kfall_scn_YYYYMMDD_HHMMSS
```

KFall runs save the best validation checkpoint, history, test predictions, a classification report, confusion matrices, and dataset metadata. Cached events are read on demand during training to avoid converting the complete cache to floating-point tensors in RAM.

## Compare KFall window durations

Audit coverage, build all three caches, or train all three models:

```bash
python compare_kfall_windows.py --mode inspect
python compare_kfall_windows.py --mode preprocess --workers 0
python compare_kfall_windows.py --mode train --epochs 100 --workers 4

# A short exploratory comparison on CPU.
python compare_kfall_windows.py --mode train --epochs 3 --batch-size 32 --device cpu --workers 0 \
  --results-dir results/kfall_window_comparison_3epochs_b32 --reuse-completed --parallel-runs 2
```

All durations use the same seed, subject partitions, four-class targets,
thresholds, and endpoint stride. By default, `--endpoint-policy matched` requires
410 samples of available history for **every** model. The three datasets then
contain the same `(recording, endpoint, label)` examples, with only their earlier
context differing. Endpoint/label hashes are checked before and after encoding.
The 2.05-second baseline is retrained on this same four-class task; an earlier
three-class checkpoint is not a comparable baseline.

The audit also reports each duration's natural coverage. Longer contexts can
lose falls whose impacts occur too early in a recording. On the supplied dataset
with seed 0, 4.10 seconds leaves 105 of 212 test fall recordings without a valid
pre-impact window; 3.30 seconds leaves 35, and 2.05 seconds leaves none. These
counts describe recordings, not window-level recall. `--endpoint-policy natural`
retains each duration's available examples, but then the evaluated examples differ.

With matched endpoints and the supplied dataset, every variant has 15,559 train,
1,735 validation, and 1,856 test windows. Test targets are Fall 191, Dynamic 591,
Static 667, and Postimpact 407, across the same 26/3/3 subjects.

`--event-length-policy scaled` uses the event lengths in the table above. The
optional `fixed` policy keeps `--fixed-out-win 2048` for all durations; that also
changes temporal resolution, so it is a different experiment.

Outputs under `--results-dir` include `window_plan.json`, `window_coverage.csv`,
`window_counts.png`, `natural_window_counts.png`, and per-duration cache paths in
`comparison.json`. Training additionally writes a metric comparison CSV/JSON and
`validation_comparison.png`. Checkpoints are selected by validation loss within
each run. Comparison tables do not choose a winner using test scores. Short
exploratory runs should be followed by longer training and repeated seeds before
drawing conclusions about window duration.

`--reuse-completed` checks configuration, checkpoint, and cache provenance before
reusing a finished run. `--parallel-runs` controls independent experiment
processes; the default is one. Add `--resample-hz 200` to compare all durations
with the same intermediate interpolation stage.
Reported wall times include resource contention when experiments run concurrently.

## Evaluate the best KFall checkpoint

Evaluate an existing run without training again:

```bash
python evaluate_kfall.py --run-dir runs/kfall_scn_YYYYMMDD_HHMMSS

# Force CPU inference if desired.
python evaluate_kfall.py --run-dir runs/kfall_scn_YYYYMMDD_HHMMSS --device cpu
```

The evaluator loads `checkpoints/best_model.pt`, selected by the lowest
validation loss, and reconstructs the network from the configuration embedded
in that checkpoint. It checks the event cache against the run's
`dataset_metadata.json`, including encoding parameters, source signature, and
subject/window manifests. It reads only the original test arrays and performs
inference with gradients disabled. The original CSV/XLSX files are not needed
when the matching cache is available. Missing or incompatible caches produce
an error and are not rebuilt during evaluation.

Results are saved under the run's `metrics/test_best/`:

- `metrics.json`: accuracy, loss, balanced accuracy, macro/weighted F1, per-class
  and per-subject metrics, and window-level fall precision/recall/specificity.
- `confusion_matrix.png`: counts and normalization by the true class.
- `classification_report.txt` and `.json`: class precision, recall, F1, support.
- `predictions.csv`: original test-window index, subject, recording, true and
  predicted classes, three or four softmax probabilities according to the saved
  classifier, and signed time relative to impact.
- NPY files with targets, predictions, probabilities, window indices, and
  confusion matrices.

Restrict inspection to a held-out subject or a specific test-window index:

```bash
python evaluate_kfall.py --run-dir runs/kfall_scn_YYYYMMDD_HHMMSS --subject SA07
python evaluate_kfall.py --run-dir runs/kfall_scn_YYYYMMDD_HHMMSS --window 580
```

Subset reports go into separate directories under `metrics/test_best/subsets/`.
The window index matches the test cache and `view_kfall.py --split test --window
580`. Optional `--checkpoint`, `--cache-dir`, and `--output-dir` arguments select
other locations; the checkpoint/cache must still match the run's saved dataset
configuration and partition.

Class weights come from the checkpoint's training weights. By default, the
saved training batch size and loss aggregation are reused to match the training
script's test loss. Changing `--batch-size` can change weighted loss averaging.
KFall's optional dual architecture is evaluated through its activity head.
These reports measure offline window classification; `lead_time_ms` describes
the annotation at a window endpoint and is not an achieved continuous warning
time.

## Moving signal and prediction replay

Replay one complete recording from the original held-out test subjects using
the run's best checkpoint:

```bash
python animate_kfall.py --run-dir runs/kfall_scn_YYYYMMDD_HHMMSS

# Select a specific test recording or the recording of a cached test window.
python animate_kfall.py --run-dir runs/kfall_scn_YYYYMMDD_HHMMSS \
  --recording S07T20R01.csv
python animate_kfall.py --run-dir runs/kfall_scn_YYYYMMDD_HHMMSS --window 580
```

The Matplotlib player scrolls the three accelerometer or gyroscope axes, plots
`P(Fall)`, and compares two timelines: the model's decision and the real fall
phase. Play/pause, restart, seek, previous/next decision, sensor selection, and
playback speed are available. Space toggles playback; arrow keys move between
decisions. `--view-seconds` changes the visible time span and `--paused` starts
the player paused.

Inference uses trailing windows with the checkpoint's original encoder settings.
The default prediction stride is ten raw samples (100 ms at 100 Hz), adjustable
with `--prediction-stride`. Every window is classified on this fixed grid,
including motion before fall onset and after impact; annotations do not choose
prediction times or enter the model. A decision appears at its window endpoint
and is held until the next one. Before a complete input window is available,
the model is shown as having no prediction. The displayed samples and decisions
never extend beyond the playback cursor.

Ground truth distinguishes the annotated onset-to-impact falling interval from
the post-impact period. Post-impact is shown separately because the normal
activity class there is not temporally annotated. The model's binary fall
indicator uses the argmax of its saved output classes, and the probability panel
shows softmax `P(Fall)`. Four-class runs display Postimpact separately for the
configured 1.20-second interval, then show that the remaining time is unlabelled.
Legacy three-class runs keep their original post-impact display.

Predictions are computed before playback for this visualization. Playback speed
does not measure inference latency, and this demonstration is separate from
the annotation-selected window metrics produced by `evaluate_kfall.py`.

Export without a graphical session:

```bash
python animate_kfall.py --run-dir runs/kfall_scn_YYYYMMDD_HHMMSS \
  --device cpu --fps 10 --save-gif results/kfall_replay/fall.gif
python animate_kfall.py --run-dir runs/kfall_scn_YYYYMMDD_HHMMSS \
  --time 2.5 --save-frame results/kfall_replay/fall_2p5s.png
```

Use `--subject`, `--sensor gyro`, `--device`, `--checkpoint`, `--cache-dir`, or
`--dataset-root` to select other held-out recordings or local file locations.

## Interactive KFall window viewer

Inspect the same encoded windows used by training alongside their original CSV
signals:

```bash
python view_kfall.py

# Start with pre-impact fall windows and the gyroscope's X/Y/Z axes.
python view_kfall.py --split test --class fall --sensor gyro

# Inspect a new four-class comparison cache (directory from comparison.json).
python view_kfall.py --cache-dir PATH_TO_COMPARISON_CACHE --class postimpact
```

The viewer shows three signal/event pairs. Switch between acceleration (g) and
angular velocity (degrees/s), filter by split, class, and subject, and navigate
using the slider, the actual cache index, or the Previous/Next buttons. Arrow
keys move between windows; Home/End select the first/last matching window, and
`s` saves a PNG under `results/kfall_viewer/`. Enter `Todos` to clear the subject
filter. Selecting another partition clears a subject filter that does not
belong to it.

The left column overlays the original measurements, the interpolated signal,
and an optional reconstruction from cached Sigma-Delta events. The right column
shows the actual cached ON/OFF events for each axis, displayed at +1 and -1;
the model receives their two binary polarity channels. This is signal encoding.
The onset annotation is marked when visible; the header/footer identify the
subject, source recording, window, offsets, thresholds, and annotated time
remaining before impact. The time axis covers only the observed window.

The viewer reads the cache without changing or rebuilding it. It verifies each
displayed window against its CSV, and reports mismatches instead of silently
showing different data. If needed, create the cache first with
`python train_kfall.py --preprocess-only`. Use `--cache-dir` for a different event
cache directory (the directory containing `metadata.json`), and `--dataset-root`
if the source dataset has moved.

A graphical Matplotlib backend such as TkAgg is required for the interactive
window. Export a selected window without a graphical session with:

```bash
python view_kfall.py --class fall --sensor acc \
  --save results/kfall_viewer/fall_acc.png
```

## Energy reports

All trained models for one dataset:

```bash
python energy_report.py --dataset ucihar 
python energy_report.py --dataset sisfall
python energy_report.py --dataset kfall
```

Single run:

```bash
python energy_report.py --dataset sisfall --run-dir runs/sisfall_scn_YYYYMMDD_HHMMSS
```

The estimator reports normalized network-only inference energy. It counts dense-equivalent MACs, effective SOPs from measured sparsity, LIF updates, and optional scratchpad data movement. The analog event-generation front-end is not included.

## Reproducibility

Run the preprocessing, subject-isolation, cache, and short CPU training checks with:

```bash
python -m unittest discover -s tests -v
```

Exact results are not guaranteed across runs. Differences can appear due to random seeds, hardware, CUDA/cuDNN behavior, library versions, dataset split choices, cache state, and early stopping.

## References and related code

This project builds on prior work in spiking neural networks for human activity recognition and sparsity-aware energy estimation. The SNN-HAR repository was used as a reference for the general spiking HAR modeling and training workflow, while SATA/SATA_Sim was used as a reference for normalized operation-level energy accounting in sparse SNN inference.

- Li Y, Yin R, Kim Y, Panda P. *Efficient human activity recognition with spatio-temporal spiking neural networks*. Frontiers in Neuroscience, 2023. Code: https://github.com/Intelligent-Computing-Lab-Panda/SNN_HAR
- Yin R, Moitra A, Bhattacharjee A, Kim Y, Panda P. *SATA: Sparsity-Aware Training Accelerator for Spiking Neural Networks*. IEEE TCAD, 2023. Code: https://github.com/RuokaiYin/SATA_Sim
