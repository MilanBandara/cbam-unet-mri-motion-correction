# Reproducing the paper

This page records the exact commands, parameters, and seeds we used to produce the numbers and figures in the paper. Run all commands from the repository root with the `cbam_unet` package installed (`pip install -e .`).

## 0. Environment

- Python 3.10
- TensorFlow 2.15+, Keras 3
- One or more CUDA-capable GPUs (we used 2x NVIDIA T4 on Kaggle for training)

```bash
pip install -e .
```

## 1. Generate paired training/validation data

The paper uses the "high-motion" k-space configuration (rigid translation+rotation, 50 phase-encoding columns swapped, width 3, uniformly random region). With the MR-ART originals laid out as in [`data/README.md`](../data/README.md):

```bash
python scripts/synthesize_data.py \
    --mrart-root <mrart-root> \
    --out data/ \
    --motion-type both \
    --shift-x 2 --shift-y 0 --angle 3 \
    --lines 50 --width 3 --region random \
    --seed 0
```

The output structure is:

```text
data/train/{Original,Corrupted}/*.png
data/val/{Original,Corrupted}/*.png
```

## 2. Train the CBAM-U-Net

```bash
python scripts/train.py \
    --train-dir data/train \
    --val-dir   data/val \
    --epochs 100 \
    --batch-size 10 \
    --lr 1e-3 \
    --image-size 256 \
    --output-dir checkpoints/ \
    --save-every-n 5 \
    --seed 0
```

This trains under `tf.distribute.MirroredStrategy` (multi-GPU when available), saves the best and last weight snapshots to `checkpoints/weights/`, and writes the per-epoch history to `checkpoints/results/train_history.json`.

To enable W&B logging:

```bash
python scripts/train.py ... \
    --wandb-project train_final_model_with_CBAM \
    --wandb-run-name train_final_model_with_CBAM-run-1
```

## 3. Evaluate

```bash
python scripts/evaluate.py \
    --weights checkpoints/weights/model_best_.weights.h5 \
    --val-dir  data/val \
    --test-dir data/test \
    --max-val-pairs 250 \
    --save-samples results/
```

This prints:

- Mean MSE / PSNR / SSIM on the first 250 validation pairs (matches the validation table in the paper).
- A side-by-side summary on the MR-ART real-motion test set (`head_motion_1` and `head_motion_2`).

It also writes qualitative grids to:

- `results/validation_samples.png`
- `results/test_headmotion1_samples.png`
- `results/test_headmotion2_samples.png`

## 4. Inference on individual scans

```bash
python scripts/infer.py \
    --weights checkpoints/weights/model_best_.weights.h5 \
    --input  data/clinical_samples/ \
    --output results/clinical_predictions/
```

## Mapping to paper artefacts

| Paper artefact                              | How to reproduce                                                                                              |
|---------------------------------------------|----------------------------------------------------------------------------------------------------------------|
| Validation metrics (MSE/PSNR/SSIM)          | `scripts/evaluate.py` ("Validation pairs" block)                                                              |
| MR-ART real-motion table (motion 1 vs 2)    | `scripts/evaluate.py` ("Summary: MR-ART Real Motion Test Data")                                               |
| Qualitative validation grid                 | `results/validation_samples.png` (produced by `evaluate.py --save-samples`)                                   |
| Qualitative real-motion grids               | `results/test_headmotion1_samples.png`, `results/test_headmotion2_samples.png`                                |
| Training/validation loss curves             | `cbam_unet.viz.plot_history('checkpoints/results/train_history.json')`                                        |
| Synthetic dataset                           | `scripts/synthesize_data.py` with the parameters above                                                        |

## Notes and caveats

- Stochastic outcomes (sample selection in synthesis, training initialisation) are seeded with `--seed 0` throughout. Exact reproduction of paper numbers also depends on TensorFlow/CUDA versions and GPU determinism settings.
- Inference is performed slice-by-slice on 2-D axial PNGs, matching the paper.
