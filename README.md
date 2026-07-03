# Mobile Screenshot Classification

Classify which mobile app a screenshot came from, using App Store marketing
screenshots as training data.

## Repository layout

| Path | Purpose |
| --- | --- |
| `app-store-screenshot-scrape/` | Node.js scraper that lists top iOS apps and downloads their App Store screenshots |
| `training-v1/` | Original exploratory notebooks and data-cleanup scripts |
| `training/` | Reproducible, config-driven training pipeline (use this going forward) |

## Pipeline

### 1. Collect screenshots

```bash
cd app-store-screenshot-scrape
yarn install
node index.js > top_grossing_ios.csv   # list top apps (id, bundle id, title)
node get_screenshots.js                # download screenshots per app id
```

The scraper downloads with retries and bounded concurrency, and records
every image (URL, path, sha256) in `screenshots/manifest.jsonl`, so an
interrupted run can simply be re-run and resumes where it left off.
Environment variables tune it: `INPUT_CSV`, `OUT_DIR`, `CONCURRENCY`, and
`COUNTRY` (different storefronts often carry different screenshots, so
scraping several countries multiplies the dataset).

Downloads land in one directory per app id. `training-v1/rename_dirs.py`
renames those directories to app titles so the directory name becomes the
class label.

### 2. Clean and split the data

```bash
cd training-v1
python find_landscape.py

cd ../training
python dedupe.py                # report near-duplicate screenshots
python dedupe.py --delete      # remove same-class duplicates
python split.py                 # write a fixed train/val/test manifest.csv
```

`find_landscape.py` flags images that are too small or unreadable, and
letterbox-pads landscape screenshots onto a portrait canvas (orientation is
preserved rather than rotating).

`dedupe.py` finds near-identical screenshots by perceptual hash — duplicates
that straddle the train/validation split inflate validation accuracy, and
duplicates spanning two classes are label noise (reported, never deleted
automatically). `split.py` then writes a stratified `manifest.csv`; point
`data.manifest` in `training/config.yaml` at it so every run and every
machine trains against the same split. Commit the manifest to version the
dataset.

### 3. Train

```bash
cd training
pip install -r requirements.txt
python train.py --run-name my-experiment
```

All hyperparameters live in `training/config.yaml` (data directory, input
size, split fractions, backbone, learning rates, epochs). Each run saves
`model.keras`, `class_names.json`, and a copy of the config it used under
`training/models/<run-name>/`, and TensorBoard logs under
`training/logs/<run-name>/`:

```bash
tensorboard --logdir training/logs
```

### 4. Evaluate and predict

```bash
cd training
python evaluate.py models/my-experiment
python predict.py models/my-experiment path/to/screenshot.png
```

`evaluate.py` rebuilds the split from the config stored with the run. If
`data.manifest` is set, the committed manifest is the source of truth; if it is
unset, the split is derived from the seed and the current directory contents.

## Reproducibility notes

- `seed` in `config.yaml` seeds Python, NumPy, and TensorFlow
  (`tf.keras.utils.set_random_seed`). Directory-mode splits are deterministic
  only while the source directory contents are unchanged; manifest-mode splits
  are stable because the manifest is versioned.
- Dependencies are pinned in `training/requirements.txt`.
- Trained models, logs, and the screenshot dataset are not committed; each
  run directory carries the config needed to reproduce it.

The notebooks in `training-v1/` (`classification.ipynb`,
`transfer_learning.ipynb`) are kept for exploration and reference; the
`training/` scripts are the maintained path.
