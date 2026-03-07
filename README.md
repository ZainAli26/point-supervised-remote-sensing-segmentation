# Point-Supervised Remote Sensing Segmentation

Semantic segmentation of remote sensing imagery using only **sparse point annotations** instead of full pixel-level masks. Implements the complete deep learning framework from the Landvisor project.

## Dataset

**WHDLD** (Wuhan Dense Labeling Dataset) — 6 classes, 256×256 images, 2m resolution

Classes: bare soil, building, pavement, road, vegetation, water

- 8892 training / 988 validation images
- VOC format with train/val split files

## Pipeline Overview

| Step | Experiment | What it does |
|------|-----------|--------------|
| 0 | Preprocessing | Histogram matching for color normalization |
| 1 | Baseline | Full supervision (upper bound) — each architecture trained with complete masks |
| 2 | Point Density Study | Sweeps point densities [10, 20, 30, 40, 50] per class — each architecture × self-training |
| 3 | Loss Comparison | Compares pCE variants using best pts from step 2 — each architecture × self-training |
| 4 | Final Ensemble | Inference only — loads best checkpoint per model, ensembles + TTA |

Each architecture (DeepLabV3+, U-Net, FPN) is trained **independently** through self-training in steps 2–3. The final ensemble combines their best checkpoints for inference only (no training).

## Quick Start (Local — RTX 3070 6GB)

```bash
pip install -r requirements.txt
python src/run_experiments.py
```

## Run on Google Colab (Recommended)

### Setup

```python
!git clone https://github.com/YOUR_USERNAME/point-supervised-remote-sensing-segmentation.git
%cd point-supervised-remote-sensing-segmentation
!pip install -r requirements.txt
!cp configs/config_colab.yaml configs/config.yaml
```

### Run everything at once

```bash
python src/run_experiments.py --experiment all
```

### Or run step by step (recommended to spread compute)

```bash
# Step 1: Baseline (full supervision, all architectures)
python src/run_experiments.py --experiment baseline

# Step 2: Point density study (run each model independently)
python src/run_experiments.py --experiment point_density --model deeplabv3plus_resnet50
python src/run_experiments.py --experiment point_density --model unet_resnet50
python src/run_experiments.py --experiment point_density --model fpn_resnet50

# Step 3: Loss comparison (auto-loads best pts from step 2)
python src/run_experiments.py --experiment loss_comparison --model deeplabv3plus_resnet50
python src/run_experiments.py --experiment loss_comparison --model unet_resnet50
python src/run_experiments.py --experiment loss_comparison --model fpn_resnet50

# Step 4: Final ensemble (inference only, no training)
python src/run_experiments.py --experiment final_ensemble
```

### Or run the full pipeline for one model at a time

```bash
python src/run_experiments.py --experiment all --model deeplabv3plus_resnet50
python src/run_experiments.py --experiment all --model unet_resnet50
python src/run_experiments.py --experiment all --model fpn_resnet50
python src/run_experiments.py --experiment final_ensemble
```

Each model's results are saved independently to `experiments/` so you can run them across separate Colab sessions without losing progress.

## CLI Reference

```
python src/run_experiments.py --experiment <EXPERIMENT> [--model <MODEL>]
```

| `--experiment` | Description |
|----------------|-------------|
| `all` | Run full pipeline (baseline → point density → loss comparison) |
| `baseline` | Full supervision baseline for all architectures |
| `point_density` | Point density sweep with self-training |
| `loss_comparison` | Loss function comparison with self-training |
| `final_ensemble` | Inference-only ensemble from best checkpoints + TTA |

| `--model` (optional) | Description |
|-----------------------|-------------|
| `deeplabv3plus_resnet50` | DeepLabV3+ (Colab config) |
| `unet_resnet50` | U-Net (Colab config) |
| `fpn_resnet50` | FPN (Colab config) |

Omit `--model` to run all architectures in one go.

## Output Files

```
experiments/
├── baseline_results.json              # Full supervision results
├── point_density_<model>.json         # Per-model point density results
├── point_density_summary.json         # Combined summary
├── loss_comparison_<model>.json       # Per-model loss comparison results
├── loss_comparison_summary.json       # Combined summary
├── final_ensemble_results.json        # Ensemble + TTA results
└── all_results.json                   # Final report table

visualizations/
├── point_density_<model>.png          # Point density curves per model
├── loss_comparison_<model>.png        # Loss comparison bars per model
├── ensemble_results.png               # Ensemble bar chart
├── final_progressive.png              # Progressive improvement chart
├── qual_final_ensemble.png            # Qualitative segmentation samples
├── histogram_matching.png             # Preprocessing before/after
└── results_table.txt                  # Text summary table
```

## Project Structure

```
├── src/
│   ├── losses.py              # pCE loss with focal variants
│   ├── dataset.py             # WHDLD + point label simulation
│   ├── model.py               # DeepLabV3+, U-Net, FPN (smp)
│   ├── metrics.py             # mIoU, OA, F1, Kappa
│   ├── train.py               # Trainer + TTA + EnsemblePredictor
│   ├── semi_supervised.py     # Pseudo labels + iterative self-training
│   ├── histogram_matching.py  # Color style normalization
│   ├── visualize.py           # All figures for report
│   └── run_experiments.py     # Main experiment runner
├── configs/
│   ├── config.yaml            # Local GPU config (ResNet34, patch 256)
│   └── config_colab.yaml      # Colab config (ResNet50, batch 8)
├── report/
│   └── technical_report.md    # Report template
├── requirements.txt
└── README.md
```