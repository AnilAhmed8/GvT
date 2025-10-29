# CrossSeg-GvT: Cross-Domain Few-Shot Segmentation via Enhanced GvT

![Python](https://img.shields.io/badge/python-3.9%2B-blue)
![PyTorch](https://img.shields.io/badge/PyTorch-2.0%2B-ee4c2c)
![License: MIT](https://img.shields.io/badge/License-MIT-green)
![PRs welcome](https://img.shields.io/badge/PRs-welcome-brightgreen)

**CrossSeg-GvT** is a complete PyTorch implementation of cross-domain few-shot semantic segmentation (CD-FSS). The method follows this pipeline:

**ViT backbone → Context-Aware Memory Module (CAMM) → Meta Prompt Generator (MPG) → Cross-Domain Fusion (CDF) → domain-aware prototype integration (λ) → Enhanced Graph-based Transformer (Enhanced GvT) → Segmentation Decoder**

The repository supports **training** on: **PASCAL VOC 2012**, **SBD**, **ADE-20k**  
and **testing** on: **Cityscapes**, **Chest X-Ray (TB)**, **FSS-1000**, **ISIC2018**, **DeepGlobe**.  
A **toy episodic dataset** is included for instant sanity checks (no external data).

---

## Table of Contents
- [Features](#features)
- [Method Overview](#method-overview)
- [Datasets](#datasets)
  - [Training Datasets](#training-datasets)
  - [Testing Datasets](#testing-datasets)
  - [Folder Layouts](#folder-layouts)
- [Installation](#installation)
- [Quick Start (Toy)](#quick-start-toy)
- [Configuration](#configuration)
- [Train / Evaluate](#train--evaluate)
- [Results & Reproducibility](#results--reproducibility)
- [Troubleshooting](#troubleshooting)
- [FAQ](#faq)
- [Project Structure](#project-structure)
- [Citation](#citation)
- [License](#license)
- [Contributing](#contributing)

---

## Features
- End-to-end **CD-FSS** training & evaluation in PyTorch
- Modular implementation (CAMM, MPG, CDF, Enhanced GvT)
- Episodic **1-way k-shot** segmentation
- **mIoU** and **Dice** metrics
- **Toy** dataset for instant runs
- Generic **folder-pairs** loader (`images/` + `masks/`) for quick dataset integration
- Checkpointing, prediction overlay exports

---

## Method Overview

We implement the algorithm exactly as specified:

1. **Support Feature Extraction & Prototypes**  
   ViT backbone on support images → masked average pooling → **PrototypeNet** refinement.

2. **Query Feature Extraction & Refinement**  
   ViT on query → **CAMM** for context-aware refinement → **MPG** to generate meta prompts → **Cross-Domain Fusion** to integrate prompts.

3. **Domain-Aware Prototype Integration**  
   Scalar **λ** from fused query features scales prototypes and aggregates with fused features.

4. **Enhanced Graph-based Transformer**  
   **Enhanced GvT** performs multi-view graph smoothing (local grid + k-NN biases).

5. **Segmentation Decoder & Loss**  
   Bilinear-upsampled logits → BCE/CE loss; metrics: **mIoU** and **Dice**.

---

## Datasets

### Training Datasets
We train models under a cross-domain few-shot setting on:
- **PASCAL VOC 2012** — `10,582` images, `21` classes (20 categories + background)
- **SBD** — `11,355` images, same 20 object classes
- **ADE-20k** — `20,210` images, `150` categories across diverse scenes

### Testing Datasets
We evaluate cross-domain generalization on:
- **Cityscapes** — urban street scenes, `19` categories
- **Chest X-Ray (TB)** — tuberculosis chest X-rays (resized `1024×1024`)
- **FSS-1000** — few-shot segmentation: `1,000` categories; official test has `240` classes / `2,400` images
- **ISIC2018** — dermoscopic lesion segmentation; official training set has `2,596` images
- **DeepGlobe** — satellite segmentation with `7` categories (urban, agriculture, rangeland, forest, water, barren, unknown)

> **Default:** episodic **binary segmentation** (foreground vs background). Masks are single-channel PNG; any non-zero pixel is treated as foreground. If you want multi-class segmentation for certain datasets, see the FAQ.

### Folder Layouts

For the repo’s loaders, prepare datasets in **paired folders**:

#### VOC2012 (train)
```
VOC2012/
  JPEGImages/             # .jpg
  SegmentationClassPNG/   # .png masks (converted)
```

#### SBD (train)
```
SBD/
  images/                 # .jpg/.png
  masks/                  # .png (converted from .mat if needed)
```

#### ADE-20k (train)
```
ADE20K/
  images/
  masks/
```

#### Cityscapes / FSS-1000 / ISIC2018 / DeepGlobe / Chest X-Ray (test)
```
<DATASET_ROOT>/
  images/
  masks/                  # .png, same basenames as images
```

Alternatively, use the **generic** `folder_pairs` config pointing to:
```
data_dir/
  images/
  masks/
```

---

## Installation

### Using venv (recommended)
```bash
git clone https://github.com/<YOUR_USERNAME>/<YOUR_REPO>.git
cd <YOUR_REPO>

python -m venv .venv
# Windows:
.venv\\Scripts\\activate
# macOS / Linux:
source .venv/bin/activate

pip install -r requirements.txt
```

> **CUDA:** If you have a CUDA GPU, install the correct PyTorch build from https://pytorch.org/get-started/locally/ before `pip install -r requirements.txt` (optional but recommended).

---

## Quick Start (Toy)

Run a zero-dependency sanity check:

```bash
python train.py --config configs/toy.json
python test.py  --config configs/toy.json --checkpoint outputs/checkpoints/best.ckpt
```

Outputs:
- Checkpoints → `outputs/checkpoints/{last.ckpt,best.ckpt}`
- Predictions → `outputs/predictions/epXXX_mask.png` + `epXXX_overlay.png`

---

## Configuration

All configs live in `configs/*.json`. Common fields:

```json
{
  "experiment_name": "run_name",
  "dataset": "voc2012 | sbd | ade20k | cityscapes | fss1000 | isic2018 | deepglobe | chestxray_tb | folder_pairs | toy",

  "voc_root": "/abs/path/to/VOC2012",
  "sbd_root": "/abs/path/to/SBD",
  "ade20k_root": "/abs/path/to/ADE20K",
  "cityscapes_root": "/abs/path/to/CityscapesPrepared",
  "fss1000_root": "/abs/path/to/FSS-1000",
  "isic2018_root": "/abs/path/to/ISIC2018",
  "deepglobe_root": "/abs/path/to/DeepGlobePrepared",
  "chestxray_tb_root": "/abs/path/to/TB-ChestXRay",
  "data_dir": "/abs/path/to/AnyPairsDataset",       // for folder_pairs

  "seed": 42,
  "device": "cuda_if_available",
  "image_size": 512,
  "patch_size": 16,

  "embed_dim": 256,
  "depth": 4,
  "heads": 8,
  "mlp_ratio": 4.0,
  "dropout": 0.1,

  "k_shot": 1,
  "num_classes": 2,

  "episodes_per_epoch": 100,
  "val_episodes": 40,
  "batch_size_episodes": 1,
  "max_epochs": 20,

  "lr": 1e-4,
  "weight_decay": 0.01,
  "knn_k": 8,

  "save_dir": "outputs"
}
```

Examples to edit:
- `configs/voc2012.json` — set `"voc_root"`
- `configs/sbd.json` — set `"sbd_root"`
- `configs/ade20k.json` — set `"ade20k_root"`
- `configs/cityscapes.json` — set `"cityscapes_root"`
- `configs/folder_pairs.json` — set `"data_dir"`

---

## Train / Evaluate

### Train on training datasets
```bash
# VOC2012
python train.py --config configs/voc2012.json

# SBD
python train.py --config configs/sbd.json

# ADE-20k
python train.py --config configs/ade20k.json
```

### Evaluate on testing datasets
```bash
# Cityscapes example (uses a trained checkpoint)
python test.py --config configs/cityscapes.json \
               --checkpoint outputs/checkpoints/best.ckpt

# Other benchmarks (edit the config root path first)
python test.py --config configs/fss1000.json      --checkpoint outputs/checkpoints/best.ckpt
python test.py --config configs/isic2018.json     --checkpoint outputs/checkpoints/best.ckpt
python test.py --config configs/deepglobe.json    --checkpoint outputs/checkpoints/best.ckpt
python test.py --config configs/chestxray_tb.json --checkpoint outputs/checkpoints/best.ckpt
```

### Train/Eval with your own pairs dataset
```bash
# Prepare:
#   data_dir/images/ (RGB)
#   data_dir/masks/  (PNG, single-channel; non-zero = foreground)
python train.py --config configs/folder_pairs.json
python test.py  --config configs/folder_pairs.json --checkpoint outputs/checkpoints/best.ckpt
```

---

## Results & Reproducibility

- Metrics: **mean IoU (mIoU)** and **Dice** for binary segmentation.
- Reproducibility: seeds are fixed (PyTorch/NumPy/Python).  
- By default, evaluation averages metrics over episodes in the given split.

> To run multi-trial protocols (e.g., 5 seeds × N episodes) wrap `train.py/test.py` in a small script that iterates over seeds, then average the printed metrics.

---

## Troubleshooting

- **“No pairs found” / loader error**  
  Verify structure:
  ```
  <root>/
    images/
    masks/     # PNG masks, same basenames as images
  ```
  and your config points to the correct `*_root` or `data_dir`.

- **Masks look blank / overlays are black**  
  Ensure masks are **single-channel** PNG with foreground pixels **> 0**.

- **CUDA out of memory (OOM)**  
  Reduce `image_size`, `embed_dim`, `depth`, `heads`, or `episodes_per_epoch`.

- **GPU not used**  
  Set `"device": "cuda_if_available"` and verify `torch.cuda.is_available()` is `True`.

---

## FAQ

**Q: Can I use raw Cityscapes/ADE/ISIC folder structures?**  
A: This repo expects paired `images/` + `masks/`. If you prefer raw structures (e.g., Cityscapes `leftImg8bit` + `gtFine`), add a conversion step to produce PNG masks with the same basenames. (Open an issue if you want a built-in converter.)

**Q: How to enable multi-class segmentation?**  
A: Set `"num_classes" > 2`. Ensure your masks contain class indices [0..C-1]. The decoder already supports multi-class logits; `mean_iou` will average over classes. You may want to adjust prompt design / prototypes per class (we can provide a multi-class episodic sampler on request).

**Q: Change k-shot?**  
A: Edit `"k_shot"` in your config (e.g., 1, 5). The episodic loader samples that many support images.

---

## Project Structure

```
.
├── configs/                       # JSON configs (train/test + toy + folder_pairs)
├── src/
│   ├── data/
│   │   └── datasets.py            # Toy episodic + generic folder-pairs episodic
│   ├── engine/
│   │   ├── train_eval.py          # Train/eval loops + metrics aggregation
│   │   └── losses.py              # CE / BCE-with-logits
│   ├── models/
│   │   ├── simple_vit.py          # ViT backbone
│   │   ├── prototype_net.py       # Prototype refinement
│   │   ├── camm.py                # Context-Aware Memory Module
│   │   ├── meta_prompt.py         # Meta Prompt Generator
│   │   ├── cross_domain_fusion.py # Cross-Domain Fusion
│   │   ├── enhanced_gvt.py        # Enhanced Graph-based Transformer
│   │   └── decoder.py             # Segmentation decoder
│   └── utils/
│       ├── metrics.py             # mIoU, Dice
│       ├── visualization.py       # overlay + mask saving
│       ├── checkpoint.py          # save/load checkpoints
│       └── common.py              # seeding, device utils, dirs
├── train.py
├── test.py
└── requirements.txt
```

---

## Citation

If this repository helps your research, please cite:

```
@misc{CrossSegGvT,
  title  = {CrossSeg-GvT: Cross-Domain Few-Shot Segmentation via Enhanced GvT},
  author = {<Your Name>},
  year   = {2025},
  url    = {https://github.com/<YOUR_USERNAME>/<YOUR_REPO>}
}
```

---

## License

This repository is released under the **MIT License**. See `LICENSE` for details.

---

## Contributing

Contributions are welcome!
- Open an issue for bugs/feature requests.
- Fork → create a branch → submit a PR.
- Keep PRs focused and include brief notes on motivation and testing.

---

### Optional: Publish a Release

1. Tag & push:
   ```bash
   git tag v1.0.0
   git push origin v1.0.0
   ```
2. Draft a GitHub Release → attach your code ZIP → publish.
3. Add a social preview image (Settings → Social preview) to improve link sharing.
