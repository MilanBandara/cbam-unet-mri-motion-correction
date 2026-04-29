# Data layout

This directory holds the datasets the scripts read from. Only directory placeholders are committed; you must populate the folders yourself.

## Expected on-disk layout

```text
data/
  train/
    Original/   *.png        # motion-free targets
    Corrupted/  *.png        # synthetic motion-corrupted inputs (i-th file pairs with i-th Original)
  val/
    Original/   *.png
    Corrupted/  *.png
  test/
    Original/        *.png   # MR-ART motion-free reference scans
    head_motion_1/   *.png   # MR-ART mild real-motion acquisitions
    head_motion_2/   *.png   # MR-ART severe real-motion acquisitions
  clinical_samples/  *.png   # optional: in-house clinical scans used for qualitative inspection
```

Filenames inside `Original/` and `Corrupted/` (resp. `Original/`, `head_motion_1/`, `head_motion_2/`) must sort to a consistent order so that the i-th file in each folder forms a paired example. The synthesis script enforces this by emitting `Corrupted/<stem>_corrupted.png` next to `Original/<stem>.png`.

## 1. Download MR-ART

The [MR-ART (Movement-Related Artefacts) dataset](https://openneuro.org/datasets/ds004173) provides paired motion-free and motion-affected T1-weighted brain MRIs (Nárai et al., 2022).

After downloading, extract the relevant axial slices to PNG (256x256 grayscale) and arrange them as:

```text
<mrart-root>/
  MRART_training_data/Original/*.png
  MRART_val_data/Original/*.png
  MRART_test_data/Original/*.png
  MRART_test_data/head_motion_1/*.png
  MRART_test_data/head_motion_2/*.png
```

(The exact split between train / val / test is described in the paper.)

## 2. Generate paired synthetic training data

Use `scripts/synthesize_data.py` to populate `data/train/{Original,Corrupted}` and `data/val/{Original,Corrupted}` from the MR-ART originals via the stochastic k-space perturbation framework:

```bash
python scripts/synthesize_data.py \
    --mrart-root <mrart-root> \
    --out data/ \
    --motion-type both --shift-x 2 --angle 3 \
    --lines 50 --width 3 --region random \
    --seed 0
```

The defaults match the "high-motion" configuration used in the paper. To explore other regimes, vary `--lines`, `--width`, and `--region` (`random`, `central`, `peripheral`, `mixed`).

## 3. Lay out the real-motion test set

Copy the MR-ART test scans into `data/test/` so they match the layout above. `scripts/evaluate.py` will then automatically report metrics for `head_motion_1` and `head_motion_2`.

## 4. Optional: clinical samples

Drop any in-house clinical PNGs into `data/clinical_samples/` for qualitative inspection via `scripts/infer.py`.

## Notes

- All scripts assume single-channel grayscale PNGs at 256x256. They will resize on read, but storing the data at the correct resolution avoids surprises.
- Image bytes are intentionally git-ignored; only `.gitkeep` placeholders are tracked.
