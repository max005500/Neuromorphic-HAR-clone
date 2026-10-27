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

# Train using the prepared event windows.
python train_kfall.py --epochs 100 --batch-size 128 --workers 4

# Compare original X/Y/Z signals with the cached Sigma-Delta events.
python view_kfall.py --class fall

# Evaluate an existing run's best checkpoint on its held-out test subjects.
python evaluate_kfall.py --run-dir runs/kfall_scn_YYYYMMDD_HHMMSS

# Animate a full test recording with predicted and real fall labels.
python animate_kfall.py --run-dir runs/kfall_scn_YYYYMMDD_HHMMSS
```

Subjects are assigned to disjoint partitions before windowing. The default fall
policy excludes the impact sample and all later samples. Generated event caches
are stored separately under `data/kfall_cache/`.

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
