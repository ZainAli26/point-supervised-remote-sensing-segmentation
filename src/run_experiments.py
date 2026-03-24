"""
run_experiments.py — Main Experiment Runner
=============================================
Run all:         python src/run_experiments.py
Run one model:   python src/run_experiments.py --experiment point_density --model unet_resnet34
Run one step:    python src/run_experiments.py --experiment loss_comparison
Final ensemble:  python src/run_experiments.py --experiment final_ensemble

Pipeline:
  0. Histogram matching (preprocessing)
  1. Full supervision baseline (each architecture)
  2. Point density study (each architecture x self-training)
  3. Loss comparison (each architecture x self-training, best pts from step 2)
  4. Final ensemble: inference-only from each model's best checkpoint + TTA
  5. Report + figures

Use --model to run a single architecture independently. Per-model results
are saved to experiments/<experiment>_<model>.json and automatically merged
when all models are complete. This lets you spread compute across runs:

  python src/run_experiments.py --experiment point_density --model deeplabv3plus_resnet34
  python src/run_experiments.py --experiment point_density --model unet_resnet34
  python src/run_experiments.py --experiment point_density --model fpn_resnet34
  python src/run_experiments.py --experiment loss_comparison   # runs all, loads pts from above
  python src/run_experiments.py --experiment final_ensemble    # inference only
"""

import os
import sys
import json
import yaml
import numpy as np
import torch
from glob import glob
from PIL import Image
from datetime import datetime
from torch.utils.data import DataLoader

import matplotlib.pyplot as plt
from dataset import (WHDLDPointDataset, get_train_transform, get_val_transform)
from losses import build_loss, verify_losses
from model import build_model
from train import Trainer, TestTimeAugmentation, EnsemblePredictor
from semi_supervised import SelfTrainingPipeline
from histogram_matching import preprocess_dataset, visualize_histogram_matching
from visualize import (plot_point_density_results, plot_loss_comparison,
                       plot_per_class_iou_heatmap, plot_point_labels_demo,
                       generate_results_table, plot_ensemble_results)


def set_seed(seed):
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


# ============================================================
# SINGLE EXPERIMENT — one architecture + self-training
# ============================================================

def run_single_experiment(cfg, architecture, encoder, num_points,
                          loss_name, seed, exp_name):
    """
    Train one architecture through the full self-training pipeline.

    Returns:
        final_metrics  - metrics dict from the last self-training stage
        ckpt_path      - path to best_model.pth of that stage
    """
    set_seed(seed)

    print(f"\n{'#'*60}")
    print(f"  {exp_name}")
    print(f"  arch={architecture}, pts={num_points}, loss={loss_name}, seed={seed}")
    print(f"{'#'*60}")

    sc = cfg.get('semi_supervised', {})
    n_iter = sc.get('n_iterations', 2)
    thresholds = sc.get('thresholds', [0.95, 0.90])

    pipeline = SelfTrainingPipeline(
        cfg, num_points, loss_name, seed,
        n_iter, thresholds,
        architecture=architecture, encoder=encoder,
        exp_prefix=exp_name)
    all_metrics, _ = pipeline.run()

    # Last stage metrics (walk backwards to find the last entry with mIoU)
    final_metrics = None
    for k in reversed(list(all_metrics.keys())):
        v = all_metrics[k]
        if isinstance(v, dict) and 'mIoU' in v:
            final_metrics = v
            break

    # Checkpoint of the last self-training stage
    last_stage = 1 + n_iter
    ckpt_path = os.path.join("experiments",
                             f"{exp_name}_stage{last_stage}", "best_model.pth")
    if not os.path.exists(ckpt_path):
        # Fall back through earlier stages
        for s in range(last_stage - 1, 0, -1):
            alt = os.path.join("experiments",
                               f"{exp_name}_stage{s}", "best_model.pth")
            if os.path.exists(alt):
                ckpt_path = alt
                break

    return final_metrics, ckpt_path


# ============================================================
# BASELINE — full supervision (each architecture)
# ============================================================

def run_baseline(cfg, model_name=None):
    print("\n" + "=" * 60 + "\n  BASELINE: Full Supervision\n" + "=" * 60)

    models = _get_model_configs(cfg, model_name)
    all_names = [m['name'] for m in cfg['ensemble']['models']]
    seed = cfg['baseline']['seeds'][0]
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    patch = cfg['data']['patch_size']
    n_cls = cfg['data']['num_classes']
    bs, nw = cfg['training']['batch_size'], cfg['training']['num_workers']

    results = {}

    for mcfg in models:
        arch, enc, name = mcfg['architecture'], mcfg['encoder'], mcfg['name']
        print(f"\n  Baseline: {name}")
        set_seed(seed)

        train_ds = WHDLDPointDataset(
            cfg['data']['image_dir'], cfg['data']['mask_dir'],
            0, n_cls, transform=get_train_transform(patch), use_full_mask=True,
            split_file=cfg['data']['train_split'])
        val_ds = WHDLDPointDataset(
            cfg['data']['image_dir'], cfg['data']['mask_dir'],
            0, n_cls, transform=get_val_transform(patch), use_full_mask=True,
            split_file=cfg['data']['val_split'])
        train_loader = DataLoader(train_ds, batch_size=bs, shuffle=True,
                                  num_workers=nw, pin_memory=True, drop_last=True)
        val_loader = DataLoader(val_ds, batch_size=bs, shuffle=False,
                                num_workers=nw, pin_memory=True)

        model = build_model(arch, enc, n_cls, True)
        criterion = build_loss(cfg['baseline']['loss'], n_cls)
        tc = cfg['training']
        opt = torch.optim.AdamW(model.parameters(), lr=tc['learning_rate'],
                                weight_decay=tc['weight_decay'])
        sch = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=tc['epochs'])

        trainer = Trainer(model, criterion, opt, sch, device, cfg,
                          f"baseline_{name}")
        _, metrics = trainer.fit(train_loader, val_loader)
        results[name] = metrics
        _save_per_model_result("baseline", name, {
            k: v for k, v in metrics.items() if k != 'confusion_matrix'})
        print(f"    mIoU: {metrics['mIoU']:.2f}%")

    # Merge all available baseline results
    all_saved = _load_per_model_results("baseline", all_names)
    if all_saved:
        with open("experiments/baseline_results.json", 'w') as f:
            json.dump(all_saved, f, indent=2)
    return results


# ============================================================
# HELPERS — per-model result files
# ============================================================

def _save_per_model_result(experiment, model_name, data):
    """Save one model's results to experiments/<experiment>_<model>.json"""
    path = f"experiments/{experiment}_{model_name}.json"
    with open(path, 'w') as f:
        json.dump(data, f, indent=2)


def _load_per_model_results(experiment, model_names):
    """Load all available per-model result files and merge."""
    merged = {}
    for name in model_names:
        path = f"experiments/{experiment}_{name}.json"
        if os.path.exists(path):
            with open(path) as f:
                merged[name] = json.load(f)
    return merged


def _get_model_configs(cfg, model_name=None):
    """Return the list of model configs, filtered to one if model_name given."""
    all_models = cfg['ensemble']['models']
    if model_name is None:
        return all_models
    match = [m for m in all_models if m['name'] == model_name]
    if not match:
        available = [m['name'] for m in all_models]
        print(f"  ERROR: Unknown model '{model_name}'.")
        print(f"  Available: {available}")
        sys.exit(1)
    return match


# ============================================================
# POINT DENSITY STUDY (per-model, results saved independently)
# ============================================================

def run_point_density_study(cfg, model_name=None):
    """
    Run point density sweep. If model_name is given, only that model runs.
    Each model's results are saved to experiments/point_density_<model>.json
    so runs can happen independently and still be compared.
    """
    models = _get_model_configs(cfg, model_name)
    e = cfg['experiment1']
    all_names = [m['name'] for m in cfg['ensemble']['models']]

    scope = models[0]['name'] if model_name else "All Models"
    print("\n" + "=" * 60)
    print(f"  POINT DENSITY STUDY — {scope} + Self-Training")
    print("=" * 60)

    # results[model_name][pts] = {"mIoU": [...], "OA": [...], "F1": [...]}
    results = {}
    best_pts_per_model = {}

    for mcfg in models:
        arch, enc, name = mcfg['architecture'], mcfg['encoder'], mcfg['name']
        print(f"\n  {'='*55}")
        print(f"  Model: {name}")
        print(f"  {'='*55}")
        results[name] = {}

        for pts in e['points_per_class']:
            seed_metrics = []
            for seed in e['seeds']:
                exp_name = f"pd_{name}_pts{pts}_s{seed}"
                metrics, _ = run_single_experiment(
                    cfg, arch, enc, pts, e['loss'], seed, exp_name)
                seed_metrics.append(metrics)

            results[name][pts] = {
                "mIoU": [m['mIoU'] for m in seed_metrics],
                "OA":   [m['OA'] for m in seed_metrics],
                "F1":   [m['macro_F1'] for m in seed_metrics],
            }

        # Best pts by mean mIoU
        avg = {p: np.mean(results[name][p]["mIoU"])
               for p in e['points_per_class']}
        best = max(avg, key=avg.get)
        best_pts_per_model[name] = best

        # Save this model's results immediately
        _save_per_model_result("point_density", name, {
            "best_pts": best,
            "per_density": {
                str(p): {
                    "mIoU_mean": round(np.mean(results[name][p]["mIoU"]), 2),
                    "mIoU_std":  round(np.std(results[name][p]["mIoU"]), 2),
                }
                for p in e['points_per_class']
            },
        })
        print(f"\n    >>> Best for {name}: {best} pts/class "
              f"(mean mIoU: {avg[best]:.2f}%)")
        print(f"    Saved: experiments/point_density_{name}.json")

        # Figure for this model
        plot_point_density_results(
            {p: results[name][p]["mIoU"] for p in e['points_per_class']})
        src = "visualizations/exp1_point_density.png"
        dst = f"visualizations/point_density_{name}.png"
        if os.path.exists(src):
            os.rename(src, dst)

    # Also write a combined summary from all available per-model files
    all_saved = _load_per_model_results("point_density", all_names)
    if all_saved:
        with open("experiments/point_density_summary.json", 'w') as f:
            json.dump(all_saved, f, indent=2)

    # Merge best_pts from saved files (includes models from prior runs)
    for name, data in all_saved.items():
        if name not in best_pts_per_model:
            best_pts_per_model[name] = data['best_pts']

    # Point label demo (once)
    if not model_name:
        try:
            mf = sorted(glob(os.path.join(cfg['data']['mask_dir'], "*.png")))
            if mf:
                mask = np.array(Image.open(mf[0]))
                # Remap WHDLD 1-6 → 0-5
                n_cls = cfg['data']['num_classes']
                valid = (mask >= 1) & (mask <= n_cls)
                remapped = np.zeros_like(mask)
                remapped[valid] = mask[valid] - 1
                plot_point_labels_demo(remapped[:256, :256], e['points_per_class'])
        except Exception:
            pass

    print(f"\n  Best pts per model: {best_pts_per_model}")
    return results, best_pts_per_model


# ============================================================
# LOSS COMPARISON (per-model, results saved independently)
# ============================================================

def run_loss_comparison(cfg, best_pts_per_model, model_name=None, loss_name_filter=None):
    """
    Run loss function comparison. If model_name is given, only that model runs.
    Each model's results are saved to experiments/loss_comparison_<model>.json
    so runs can happen independently and still be compared.
    """
    models = _get_model_configs(cfg, model_name)
    e = cfg['experiment2']
    if loss_name_filter:
        if loss_name_filter not in e['losses']:
            print(f"  ERROR: Unknown loss '{loss_name_filter}'. "
                  f"Available: {e['losses']}")
            sys.exit(1)
        e = {**e, 'losses': [loss_name_filter]}
    all_names = [m['name'] for m in cfg['ensemble']['models']]

    scope = models[0]['name'] if model_name else "All Models"
    print("\n" + "=" * 60)
    print(f"  LOSS COMPARISON — {scope} + Self-Training")
    print("=" * 60)

    # results[model_name][loss] = {"mIoU": [...], "OA": [...], "F1": [...]}
    results = {}
    best_loss_per_model = {}
    best_ckpt_per_model = {}
    best_metrics_per_model = {}

    for mcfg in models:
        arch, enc, name = mcfg['architecture'], mcfg['encoder'], mcfg['name']
        pts = best_pts_per_model.get(name, 10)
        print(f"\n  {'='*55}")
        print(f"  Model: {name}  (pts={pts} from point density study)")
        print(f"  {'='*55}")
        results[name] = {}

        # Track checkpoints and metrics per loss x seed for this model
        ckpts = {}
        all_seed_metrics = {}

        for loss_name in e['losses']:
            seed_metrics = []
            ckpts[loss_name] = {}
            all_seed_metrics[loss_name] = []

            for seed in e['seeds']:
                exp_name = f"lc_{name}_{loss_name}_s{seed}"
                metrics, ckpt = run_single_experiment(
                    cfg, arch, enc, pts, loss_name, seed, exp_name)
                seed_metrics.append(metrics)
                all_seed_metrics[loss_name].append(metrics)
                ckpts[loss_name][seed] = ckpt

            results[name][loss_name] = {
                "mIoU": [m['mIoU'] for m in seed_metrics],
                "OA":   [m['OA'] for m in seed_metrics],
                "F1":   [m['macro_F1'] for m in seed_metrics],
            }

        # Best loss by mean mIoU
        avg = {l: np.mean(results[name][l]["mIoU"]) for l in e['losses']}
        best_loss = max(avg, key=avg.get)
        best_loss_per_model[name] = best_loss

        # Best checkpoint + metrics = best seed within best loss
        mious_for_best = results[name][best_loss]["mIoU"]
        best_seed_idx = int(np.argmax(mious_for_best))
        best_seed = e['seeds'][best_seed_idx]
        best_ckpt_per_model[name] = ckpts[best_loss][best_seed]
        best_metrics_per_model[name] = all_seed_metrics[best_loss][best_seed_idx]

        # Save this model's results immediately
        _save_per_model_result("loss_comparison", name, {
            "best_pts": pts,
            "best_loss": best_loss,
            "best_ckpt": best_ckpt_per_model[name],
            "per_loss": {
                l: {
                    "mIoU_mean": round(np.mean(results[name][l]["mIoU"]), 2),
                    "mIoU_std":  round(np.std(results[name][l]["mIoU"]), 2),
                }
                for l in e['losses']
            },
        })
        print(f"\n    >>> Best for {name}: loss={best_loss} "
              f"(mean mIoU: {avg[best_loss]:.2f}%)")
        print(f"        Checkpoint: {best_ckpt_per_model[name]}")
        print(f"    Saved: experiments/loss_comparison_{name}.json")

        # Figure for this model
        plot_loss_comparison(results[name])
        src = "visualizations/exp2_loss_comparison.png"
        dst = f"visualizations/loss_comparison_{name}.png"
        if os.path.exists(src):
            os.rename(src, dst)

    # Combined summary from all available per-model files
    all_saved = _load_per_model_results("loss_comparison", all_names)
    if all_saved:
        with open("experiments/loss_comparison_summary.json", 'w') as f:
            json.dump(all_saved, f, indent=2)

    # Merge from saved files (includes models from prior runs)
    for name, data in all_saved.items():
        if name not in best_loss_per_model:
            best_loss_per_model[name] = data['best_loss']
        if name not in best_ckpt_per_model:
            best_ckpt_per_model[name] = data['best_ckpt']

    # Per-class IoU heatmap for best metrics
    if best_metrics_per_model:
        try:
            plot_per_class_iou_heatmap(best_metrics_per_model)
        except Exception:
            pass

    print(f"\n  Best loss per model: {best_loss_per_model}")
    print(f"  Best checkpoints:   {best_ckpt_per_model}")
    return results, best_loss_per_model, best_ckpt_per_model


# ============================================================
# FINAL ENSEMBLE — inference only from best checkpoints + TTA
# ============================================================

def run_final_ensemble(cfg, best_pts_per_model, best_loss_per_model,
                       best_ckpt_per_model):
    """
    No training. Load each architecture's best self-trained checkpoint
    (with its own best point density + best loss) and ensemble for inference.
    """
    print("\n" + "=" * 60)
    print("  FINAL ENSEMBLE — Inference Only (No Training)")
    print("=" * 60)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    models_cfg = cfg['ensemble']['models']
    n_cls = cfg['data']['num_classes']
    patch = cfg['data']['patch_size']
    bs, nw = cfg['training']['batch_size'], cfg['training']['num_workers']

    val_ds = WHDLDPointDataset(
        cfg['data']['image_dir'], cfg['data']['mask_dir'],
        0, n_cls, transform=get_val_transform(patch), use_full_mask=True,
        split_file=cfg['data']['val_split'])
    val_loader = DataLoader(val_ds, batch_size=bs, shuffle=False,
                            num_workers=nw, pin_memory=True)

    # ── Load each model from its best checkpoint ──
    trained = {}
    for mcfg in models_cfg:
        name = mcfg['name']
        arch, enc = mcfg['architecture'], mcfg['encoder']
        ckpt_path = best_ckpt_per_model[name]

        if not os.path.exists(ckpt_path):
            print(f"\n  ERROR: Checkpoint not found for {name}: {ckpt_path}")
            sys.exit(1)

        print(f"\n  Loading {name}")
        print(f"    Best pts:  {best_pts_per_model[name]}")
        print(f"    Best loss: {best_loss_per_model[name]}")
        print(f"    Checkpoint: {ckpt_path}")

        model = build_model(arch, enc, n_cls, pretrained=False)
        ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)
        model.load_state_dict(ckpt['model_state_dict'])
        model.to(device).eval()
        trained[name] = model

    # ── Evaluate individual models ──
    print(f"\n  Evaluating {len(trained)} individual models...")
    individual_results = {}
    for mname, model in trained.items():
        tta = TestTimeAugmentation(model, device)
        no_tta_m, tta_m = tta.evaluate(val_loader, n_cls)
        individual_results[mname] = {
            "no_tta": no_tta_m,
            "with_tta": tta_m,
        }

    # ── Ensemble without TTA ──
    print(f"\n  Evaluating ensemble ({len(trained)} models)...")
    ens = EnsemblePredictor(trained, device, use_tta=False)
    r_no = ens.evaluate(val_loader, n_cls)

    # ── Ensemble with TTA ──
    use_tta = cfg.get('ensemble', {}).get('use_tta', True)
    r_tta = None
    if use_tta:
        print(f"\n  Evaluating ensemble + TTA...")
        ens_tta = EnsemblePredictor(trained, device, use_tta=True)
        r_tta = ens_tta.evaluate(val_loader, n_cls)

    # ── Print summary ──
    print(f"\n  {'='*65}")
    print(f"  FINAL RESULTS — Each model at its own best settings")
    print(f"  {'─'*65}")
    for mname in trained:
        pts = best_pts_per_model[mname]
        loss = best_loss_per_model[mname]
        miou = individual_results[mname]['no_tta']['mIoU']
        miou_tta = individual_results[mname]['with_tta']['mIoU']
        print(f"  {mname:30s} │ mIoU: {miou:6.2f}% │ +TTA: {miou_tta:6.2f}% │ "
              f"pts={pts}, loss={loss}")
    print(f"  {'─'*65}")
    ens_miou = r_no['ensemble']['mIoU']
    print(f"  {'ENSEMBLE':30s} │ mIoU: {ens_miou:6.2f}%")
    if r_tta:
        ens_tta_miou = r_tta['ensemble']['mIoU']
        print(f"  {'ENSEMBLE + TTA':30s} │ mIoU: {ens_tta_miou:6.2f}%")
    print(f"  {'='*65}")

    # ── Qualitative samples from ensemble ──
    try:
        predictor = ens_tta if r_tta else ens
        plot_segmentation_samples_from_ensemble(
            predictor, val_ds, n_samples=4)
    except Exception:
        pass

    # ── Save results ──
    results = {
        "model_configs": {
            mname: {
                "best_pts": best_pts_per_model[mname],
                "best_loss": best_loss_per_model[mname],
                "ckpt": best_ckpt_per_model[mname],
            }
            for mname in trained
        },
        "single_models": {
            n: {k: v for k, v in individual_results[n]['no_tta'].items()
                if k != 'confusion_matrix'}
            for n in trained
        },
        "single_models_tta": {
            n: {k: v for k, v in individual_results[n]['with_tta'].items()
                if k != 'confusion_matrix'}
            for n in trained
        },
        "ensemble_no_tta": {k: v for k, v in r_no['ensemble'].items()
                            if k != 'confusion_matrix'},
    }
    if r_tta:
        results["ensemble_with_tta"] = {
            k: v for k, v in r_tta['ensemble'].items()
            if k != 'confusion_matrix'}

    with open("experiments/final_ensemble_results.json", 'w') as f:
        json.dump(results, f, indent=2)

    plot_ensemble_results(results)

    # ── Progressive improvement bar chart ──
    steps = {}
    # Pick the best single model (no TTA) as step 1
    best_single_name = max(individual_results,
                           key=lambda n: individual_results[n]['no_tta']['mIoU'])
    steps["1. Best single"] = individual_results[best_single_name]['no_tta']
    steps["2. Best single + TTA"] = individual_results[best_single_name]['with_tta']
    steps["3. Ensemble"] = r_no['ensemble']
    if r_tta:
        steps["4. Ensemble + TTA"] = r_tta['ensemble']

    _plot_progressive(steps)

    return results


def _plot_progressive(steps):
    """Bar chart showing progressive improvement."""
    labels = list(steps.keys())
    miou_vals = [steps[k]['mIoU'] for k in labels]

    fig, ax = plt.subplots(figsize=(10, 5))
    colors = plt.cm.Blues(np.linspace(0.3, 0.9, len(labels)))
    bars = ax.bar(range(len(labels)), miou_vals, color=colors,
                  edgecolor='black', linewidth=0.5)

    for i, (b, v) in enumerate(zip(bars, miou_vals)):
        ax.text(b.get_x() + b.get_width() / 2, b.get_height() + 0.3,
                f"{v:.1f}%", ha='center', fontsize=10, fontweight='bold')
        if i > 0:
            d = v - miou_vals[0]
            ax.text(b.get_x() + b.get_width() / 2, b.get_height() - 2,
                    f"({'+'if d > 0 else ''}{d:.1f}%)", ha='center',
                    fontsize=8, color='green' if d > 0 else 'red')

    ax.set_xticks(range(len(labels)))
    ax.set_xticklabels([l.split('. ')[1] if '. ' in l else l for l in labels],
                       rotation=20, ha='right', fontsize=9)
    ax.set_ylabel("mIoU (%)")
    ax.set_title("Final Ensemble: Progressive Improvement",
                 fontsize=13, fontweight='bold')
    ax.grid(True, alpha=0.3, axis='y')
    if miou_vals:
        ax.set_ylim(max(0, min(miou_vals) - 5), max(miou_vals) + 5)
    plt.tight_layout()
    plt.savefig(os.path.join("visualizations", "final_progressive.png"), dpi=150)
    plt.close()
    print(f"  Saved: final_progressive.png")


def plot_segmentation_samples_from_ensemble(predictor, dataset, n_samples=4):
    """Qualitative samples using the ensemble predictor."""
    from visualize import mask_to_rgb, CLASS_NAMES, COLORS
    import matplotlib.patches as mpatches

    fig, axes = plt.subplots(n_samples, 3, figsize=(12, 4 * n_samples))
    if n_samples == 1:
        axes = axes.reshape(1, -1)
    indices = np.random.choice(len(dataset), n_samples, replace=False)

    for row, idx in enumerate(indices):
        img_t, mask = dataset[idx]
        mask_np = mask.numpy()
        pred = predictor.predict(img_t.unsqueeze(0)).squeeze().cpu().numpy()

        img_d = (img_t.permute(1, 2, 0).numpy()
                 * np.array([0.229, 0.224, 0.225])
                 + np.array([0.485, 0.456, 0.406]))
        img_d = np.clip(img_d, 0, 1)

        axes[row, 0].imshow(img_d)
        axes[row, 0].set_title("Input" if row == 0 else "")
        axes[row, 0].axis('off')
        axes[row, 1].imshow(mask_to_rgb(mask_np))
        axes[row, 1].set_title("Ground Truth" if row == 0 else "")
        axes[row, 1].axis('off')
        axes[row, 2].imshow(mask_to_rgb(pred))
        axes[row, 2].set_title("Ensemble Prediction" if row == 0 else "")
        axes[row, 2].axis('off')

    patches = [mpatches.Patch(color=np.array(COLORS[i]) / 255,
                              label=CLASS_NAMES[i]) for i in range(len(CLASS_NAMES))]
    fig.legend(handles=patches, loc='lower center', ncol=7, fontsize=9)
    plt.tight_layout(rect=[0, 0.03, 1, 1])
    plt.savefig(os.path.join("visualizations", "qual_final_ensemble.png"),
                dpi=150)
    plt.close()
    print(f"  Saved: qual_final_ensemble.png")


# ============================================================
# REPORT
# ============================================================

def generate_report(baseline_r, exp1_r, exp2_r, best_pts, best_loss,
                    ensemble_r):
    print("\n" + "=" * 60 + "\n  FINAL REPORT\n" + "=" * 60)
    all_r = {}

    # Baseline per model
    if baseline_r:
        for name, m in baseline_r.items():
            all_r[f"Baseline ({name})"] = m

    # Exp1: best density per model
    if exp1_r and best_pts:
        for name in exp1_r:
            bp = best_pts[name]
            mious = exp1_r[name][bp]["mIoU"]
            # Use the mean as a representative metric
            all_r[f"{name} pts={bp}"] = {
                "mIoU": round(np.mean(mious), 2),
                "OA": round(np.mean(exp1_r[name][bp]["OA"]), 2),
                "macro_F1": round(np.mean(exp1_r[name][bp]["F1"]), 2),
                "kappa": 0,
            }

    # Exp2: best loss per model
    if exp2_r and best_loss:
        for name in exp2_r:
            bl = best_loss[name]
            mious = exp2_r[name][bl]["mIoU"]
            all_r[f"{name} loss={bl}"] = {
                "mIoU": round(np.mean(mious), 2),
                "OA": round(np.mean(exp2_r[name][bl]["OA"]), 2),
                "macro_F1": round(np.mean(exp2_r[name][bl]["F1"]), 2),
                "kappa": 0,
            }

    # Ensemble
    if ensemble_r:
        for key in ["ensemble_no_tta", "ensemble_with_tta"]:
            if key in ensemble_r:
                label = "ENSEMBLE" if "no_tta" in key else "ENSEMBLE + TTA"
                all_r[label] = ensemble_r[key]

    generate_results_table(all_r)

    with open("experiments/all_results.json", 'w') as f:
        json.dump({k: {kk: vv for kk, vv in v.items() if kk != 'confusion_matrix'}
                   for k, v in all_r.items()}, f, indent=2)


# ============================================================
# MAIN
# ============================================================

def _apply_data_root(cfg, data_root):
    """Override all data paths to use a new root directory."""
    if data_root is None:
        return cfg
    old_root = cfg['data']['data_root']
    for key in ['data_root', 'image_dir', 'mask_dir', 'train_split', 'val_split']:
        if key in cfg['data']:
            cfg['data'][key] = cfg['data'][key].replace(old_root, data_root)
    print(f"  Data root overridden: {data_root}")
    return cfg


def main(config_path="configs/config.yaml", data_root=None):
    with open(config_path) as f:
        cfg = yaml.safe_load(f)
    cfg = _apply_data_root(cfg, data_root)

    print("=" * 60)
    print("  POINT-SUPERVISED REMOTE SENSING SEGMENTATION")
    print(f"  {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("=" * 60)

    if torch.cuda.is_available():
        print(f"  GPU: {torch.cuda.get_device_name(0)} "
              f"({torch.cuda.get_device_properties(0).total_mem / 1e9:.1f} GB)")

    verify_losses()

    if not os.path.exists(cfg['data']['image_dir']):
        print(f"\n  ERROR: Dataset not found at {cfg['data']['image_dir']}")
        print("  Download WHDLD dataset and place under data/WHDLD/")
        sys.exit(1)

    os.makedirs("experiments", exist_ok=True)
    os.makedirs("visualizations", exist_ok=True)

    # Preprocessing: Histogram matching
    original_img_dir = cfg['data']['image_dir']
    if cfg.get('preprocessing', {}).get('histogram_matching', True):
        cfg, stats = preprocess_dataset(cfg)
        if not stats.get('skipped'):
            try:
                refs = sorted(glob(os.path.join(original_img_dir, "*.jpg")))
                if refs:
                    visualize_histogram_matching(
                        original_img_dir, cfg['data']['image_dir'], refs[0])
            except Exception as e:
                print(f"  Warning: {e}")

    # 1. Baseline
    baseline_r = run_baseline(cfg)

    # 2. Experiment 1: Point density (all models x self-training)
    exp1_r, best_pts = run_point_density_study(cfg)

    # 3. Experiment 2: Loss comparison (all models x self-training)
    exp2_r, best_loss, best_ckpts = run_loss_comparison(cfg, best_pts)

    # 4. Final ensemble: inference only from best checkpoints
    ensemble_r = run_final_ensemble(cfg, best_pts, best_loss, best_ckpts)

    # 5. Report
    generate_report(baseline_r, exp1_r, exp2_r, best_pts, best_loss,
                    ensemble_r)

    print(f"\n{'='*60}")
    print(f"  ALL COMPLETE! {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"{'='*60}")


def _load_best_pts(cfg):
    """Load best_pts per model from saved point_density results."""
    all_names = [m['name'] for m in cfg['ensemble']['models']]
    saved = _load_per_model_results("point_density", all_names)
    best_pts = {name: data['best_pts'] for name, data in saved.items()}
    # Fill defaults for models not yet run
    for m in cfg['ensemble']['models']:
        if m['name'] not in best_pts:
            best_pts[m['name']] = 10
    return best_pts


def _load_best_settings(cfg):
    """Load best_pts + best_loss + best_ckpt per model from saved results."""
    all_names = [m['name'] for m in cfg['ensemble']['models']]
    saved = _load_per_model_results("loss_comparison", all_names)
    best_pts, best_loss, best_ckpts = {}, {}, {}
    for name, data in saved.items():
        best_pts[name] = data['best_pts']
        best_loss[name] = data['best_loss']
        best_ckpts[name] = data['best_ckpt']
    return best_pts, best_loss, best_ckpts


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(
        description="Point-supervised remote sensing segmentation experiments")
    parser.add_argument("--experiment", default="all",
                        choices=["all", "baseline", "point_density",
                                 "loss_comparison", "final_ensemble"])
    parser.add_argument("--model", default=None,
                        help="Run only this model (e.g. deeplabv3plus_resnet50, "
                             "unet_resnet50, fpn_resnet50). "
                             "Omit to run all models.")
    parser.add_argument("--loss", default=None,
                        help="Run only this loss (e.g. pce, pce_focal_g2). "
                             "Only applies to loss_comparison experiment.")
    parser.add_argument("--config", default="configs/config.yaml",
                        help="Path to config YAML (default: configs/config.yaml). "
                             "Use configs/config_colab.yaml on Colab.")
    parser.add_argument("--data-root", default=None,
                        help="Override data root path at runtime "
                             "(updates image_dir, mask_dir, split paths).")
    args = parser.parse_args()

    if args.experiment == "all" and args.model is None:
        main(config_path=args.config, data_root=args.data_root)
    else:
        with open(args.config) as f:
            cfg = yaml.safe_load(f)
        cfg = _apply_data_root(cfg, args.data_root)
        verify_losses()
        os.makedirs("experiments", exist_ok=True)
        os.makedirs("visualizations", exist_ok=True)
        if cfg.get('preprocessing', {}).get('histogram_matching', True):
            cfg, _ = preprocess_dataset(cfg)

        if args.experiment == "all":
            # --model with --experiment all: run full pipeline for one model
            # (point_density + loss_comparison for that model only)
            run_point_density_study(cfg, model_name=args.model)
            best_pts = _load_best_pts(cfg)
            run_loss_comparison(cfg, best_pts, model_name=args.model,
                               loss_name_filter=args.loss)
            print(f"\n  Done: {args.model}. Run --experiment final_ensemble "
                  f"after all models are complete.")

        elif args.experiment == "baseline":
            run_baseline(cfg, model_name=args.model)

        elif args.experiment == "point_density":
            run_point_density_study(cfg, model_name=args.model)

        elif args.experiment == "loss_comparison":
            best_pts = _load_best_pts(cfg)
            found = _load_per_model_results(
                "point_density",
                [m['name'] for m in cfg['ensemble']['models']])
            if found:
                print(f"  Loaded best_pts: {best_pts}")
            else:
                print(f"  No point density results found, using default pts=10")
            run_loss_comparison(cfg, best_pts, model_name=args.model,
                               loss_name_filter=args.loss)

        elif args.experiment == "final_ensemble":
            best_pts, best_loss, best_ckpts = _load_best_settings(cfg)
            if not best_loss:
                print("  ERROR: Run point_density and loss_comparison first "
                      "for all models.")
                sys.exit(1)
            missing = [m['name'] for m in cfg['ensemble']['models']
                       if m['name'] not in best_loss]
            if missing:
                print(f"  ERROR: Missing loss_comparison results for: "
                      f"{missing}")
                print(f"  Run: python src/run_experiments.py "
                      f"--experiment loss_comparison --model <name>")
                sys.exit(1)
            print(f"  Loaded settings: {best_loss}")
            run_final_ensemble(cfg, best_pts, best_loss, best_ckpts)
