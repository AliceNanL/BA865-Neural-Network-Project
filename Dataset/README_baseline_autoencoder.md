# Baseline CNN Autoencoder for Screw Anomaly Detection

This project uses a simple convolutional autoencoder as a first baseline for the MVTec-style screw dataset in `/Users/nan/Desktop/BA865 Project/screw`.

## What the script does

- Trains only on normal images from `screw/train/good`
- Holds out a small validation split from those normal images
- Tests on all images in `screw/test`
- Treats reconstruction error as the anomaly score
- Saves a few example reconstructions and error maps

## Files

- Script: `/Users/nan/Desktop/BA865 Project/baseline_autoencoder.py`
- Outputs: `/Users/nan/Desktop/BA865 Project/runs/baseline_autoencoder`

## Expected packages

You will need:

- `python`
- `torch`
- `numpy`
- `Pillow`

## Run

From `/Users/nan/Desktop/BA865 Project`:

```bash
python3 baseline_autoencoder.py
```

You can also try a shorter run first:

```bash
python3 baseline_autoencoder.py --epochs 5 --batch-size 16
```

## Main outputs

- `best_model.pt`: best checkpoint by validation reconstruction loss
- `history.json`: epoch-by-epoch training and validation loss
- `metrics.json`: image-level evaluation summary
- `examples/`: input, reconstruction, and grayscale error maps

## How to interpret the results

- `mean_score_good` should be lower than `mean_score_anomaly`
- `image_auroc` closer to `1.0` is better
- Error maps should light up on defective screw regions more than on normal images

## Why this is a good first baseline

This model is intentionally simple:

- no skip connections
- no pretrained backbone
- no segmentation head

That makes it a good reference point before moving to a U-Net autoencoder or a stronger anomaly method like PatchCore or PaDiM.
