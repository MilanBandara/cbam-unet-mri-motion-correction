# MR²-AttUNet — Bridging the Gap: Supervised MRI Motion Artifact Correction via Data Synthesis

Reference implementation of **MR²-AttUNet**, the multi-resolution CBAM U-Net introduced in the paper:

> **Bridging the Gap: Supervised MRI Motion Artifact Correction via Data Synthesis: A CBAM-U-Net Approach**
> U.Y.G.M.K. Bandara, W.M.V.S. Herath, Maheshi B. Dissanayake (Dept. of Electrical and Electronic Engineering, University of Peradeniya), and S.C. Weerasinghe (Teaching Hospital Peradeniya).
> Submitted to *Medical Image Analysis*.

The acronym stands for **M**otion **R**emoval × **M**ulti-**R**esolution **Att**ention **U**-**Net**.

## Overview

Magnetic Resonance Imaging (MRI) is a cornerstone of clinical diagnostics but is frequently degraded by motion artifacts that hurt diagnostic accuracy and force costly re-scans. This work proposes **MR²-AttUNet**, a physics-informed, CBAM-enhanced U-Net for retrospective motion artifact correction:

- A **Convolutional Block Attention Module (CBAM)** is embedded in a modified, multi-resolution U-Net so the network dynamically prioritises spatially and channel-wise salient features, helping it localise non-local ghosting while preserving fine anatomical detail.
- To overcome the lack of paired clinical data, we introduce a **stochastic k-space perturbation framework** that selectively corrupts phase-encoding lines in the frequency domain, generating high-fidelity paired training data from the [MR-ART](https://openneuro.org/datasets/ds004173) repository and bridging the sim-to-real gap.
- The model outperforms baseline U-Nets and state-of-the-art restoration models on PSNR/SSIM, and corrected scans were qualitatively validated by a team of consultant neurologists and a neuro-radiologist.

**Keywords:** MRI · Motion Artifact Simulation · Motion Artifact Removal · U-Net · CBAM attention.

## Repository layout

```text
cbam-unet-mri-motion-correction/
  configs/default.yaml           # hyper-parameters and paths
  src/cbam_unet/                 # importable Python package
    model.py                     # CBAM blocks + multi-resolution U-Net
    data.py                      # paired tf.data loaders
    callbacks.py                 # HistorySaver, SaveWeightsEveryN
    metrics.py                   # MSE / PSNR / SSIM helpers
    viz.py                       # qualitative sample grids and history plots
    synth/motion.py              # stochastic k-space motion synthesis
  scripts/
    synthesize_data.py           # generate paired train/val data from MR-ART
    train.py                     # train MR²-AttUNet (optional W&B)
    infer.py                     # run inference on an image or a folder
    evaluate.py                  # MSE/PSNR/SSIM on val + MR-ART real-motion test set
  notebooks/tutorial.ipynb       # slim, end-to-end smoke test
  data/                          # placeholders; bring your own MR-ART
  checkpoints/                   # weights are written here
  results/                       # plots and prediction outputs
  docs/architecture.md           # architecture description with diagram
  docs/reproducing_paper.md      # exact CLI invocations to reproduce the paper
```

> Note on naming: MR²-AttUNet is the name we use for the model in this code release. The accompanying paper refers to the same architecture as the "CBAM-U-Net" approach. The Python package is imported as `cbam_unet`.

## Installation

Python 3.10+ and a CUDA-capable GPU are recommended. The repository was tested with TensorFlow 2.15+ and Keras 3.

```bash
git clone <this-repo>
cd cbam-unet-mri-motion-correction
python -m venv .venv && source .venv/bin/activate
pip install -e .
```

Or, with conda:

```bash
conda env create -f environment.yml
conda activate cbam-unet
pip install -e .
```

## Quickstart

The pipeline is three commands:

```bash
# 1. Synthesise paired training/validation data from MR-ART originals
python scripts/synthesize_data.py \
    --mrart-root /path/to/mrart \
    --out data/

# 2. Train MR²-AttUNet
python scripts/train.py \
    --train-dir data/train \
    --val-dir   data/val \
    --epochs 100 \
    --output-dir checkpoints/

# 3. Evaluate on validation pairs and on the MR-ART real-motion test set
python scripts/evaluate.py \
    --weights checkpoints/weights/model_best_.weights.h5 \
    --val-dir  data/val \
    --test-dir data/test \
    --save-samples results/
```

To run inference on a single scan or a folder of scans:

```bash
python scripts/infer.py \
    --weights checkpoints/weights/model_best_.weights.h5 \
    --input  some_scan.png \
    --output results/predictions/
```

A walk-through of the same pipeline is available in [`notebooks/tutorial.ipynb`](notebooks/tutorial.ipynb).

## Data

This repository ships **only directory placeholders**, not image data. See [`data/README.md`](data/README.md) for the expected on-disk layout and instructions on:

- Downloading the [MR-ART](https://openneuro.org/datasets/ds004173) dataset.
- Producing the paired `Original/` + `Corrupted/` synthetic training data.
- Laying out the real-motion test set under `data/test/{Original,head_motion_1,head_motion_2}/`.

## Reproducing the paper

For exact CLI invocations, seeds, and the mapping between paper tables/figures and scripts, see [`docs/reproducing_paper.md`](docs/reproducing_paper.md). Architectural details and a diagram are in [`docs/architecture.md`](docs/architecture.md).

## Citation

If you use this code or the synthesis framework, please cite the paper:

```bibtex
@article{bandara2026bridging,
  title   = {Bridging the Gap: Supervised MRI Motion Artifact Correction via Data Synthesis: A CBAM-U-Net Approach},
  author  = {Bandara, U.Y.G.M.K. and Herath, W.M.V.S. and Dissanayake, Maheshi B. and Weerasinghe, S.C.},
  journal = {Biomedical Signal Processing and Control},
  year    = {2026},
  note    = {Under review}
}
```

A machine-readable citation is also provided in [`CITATION.cff`](CITATION.cff).

## License

MIT — see [LICENSE](LICENSE).

## Acknowledgements

We thank the contributors of the [MR-ART](https://openneuro.org/datasets/ds004173) dataset, and the consultant neurologists and neuro-radiologist who participated in the qualitative audit reported in the paper.
