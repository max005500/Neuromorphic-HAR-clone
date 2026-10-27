# Neuromorphic HAR

Event-based IMU encoding and spiking neural network training for KFall, SisFall, and UCI HAR.

The repository provides scripts to:

- generate Sigma-Delta event tensors,
- load KFall sensor CSVs and temporal fall annotations from XLSX,
- partition recordings by subject before creating event windows,
- train a KFall three-class SCN for pre-impact fall detection,
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

The general activity classes follow the repository's SisFall grouping: the `Static` family includes stable postures and posture changes. Static task IDs are `1, 2, 3, 5, 11–19`; dynamic IDs are `4, 6–10, 35, 36`; fall IDs are `20–34`. This grouping is an adaptation for the model, not a native KFall three-class annotation.

The default configuration uses 205 raw samples at 100 Hz, stride 102, and linear interpolation to 2048 event steps. Sigma-Delta encoding produces `(2048, 6, 2)` binary events and six offsets per window. These settings approximate the paper's 2.05-second window; they do not give KFall additional measured temporal resolution.

`--fall-policy pre_impact` uses annotated falling windows whose last sample precedes impact. A window can include preceding normal motion as context; the impact and later signal are excluded before interpolation and encoding. `--fall-policy annotated_interval` also permits a window ending at the annotated impact sample, so its results should not be reported as strictly pre-impact.

Evaluation uses annotation-selected windows, including an extra endpoint immediately before impact; pre-onset windows from fall trials are excluded. Cached `lead_time_ms` describes the time remaining before impact at each window's endpoint. It is not measured warning time from a continuous alerting system. Window accuracy alone does not establish online sensitivity, false-alarm rate, or achieved anticipation.

KFall supplies temporal fall annotations, but does not supply temporal **Stable Posture / Postural Transition** annotations. `train_kfall.py` therefore defaults to three-class single-head classification (`Fall`, `Dynamic`, `Static`). The optional `--head-mode dual` retains the paper's model layout with the posture head frozen and its loss disabled. It does not claim posture-supervision results.

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
python train_sisfall.py --help
python train_kfall.py --help
python train_ucihar.py --help
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
  --split 0.8 0.1 0.1 --seed 0 --fall-policy pre_impact
```

For KFall, UPS 10 resolves to 2048 event steps; `--out-len` overrides this length. Copy the six resulting threshold values into `train_kfall.py --theta ...`, keeping the seed, subject proportions, window stride, temporal policy, and encoding settings consistent. Both commands default to seed 0 and the same subject partition. Threshold optimization uses held-out subjects for validation and reports the test reconstruction error after selection.

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

KFall, with disjoint subjects and pre-impact fall windows:

```bash
python train_kfall.py --dataset-root data/KFall \
  --split 0.8 0.1 0.1 --seed 0 \
  --epochs 100 --batch-size 128 --workers 4
```

Build the event cache and inspect the split metadata without training:

```bash
python train_kfall.py --preprocess-only --workers 0
```

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

KFall runs save the best validation checkpoint, history, test predictions, a classification report, confusion matrices, and dataset metadata. Cached events are read on demand during training to avoid converting the complete cache to floating-point tensors in RAM.

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
