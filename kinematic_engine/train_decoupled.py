"""
train_decoupled.py
=============================================================================
Task 3: Masked Multi-Task Loss Training & R^2 Decoupled Evaluation
Part of the Decoupled Hierarchical Motion-ML Pipeline.

Objective:
  1. Train DecoupledMotionModel with Masked Multi-Task Loss:
       L_total = lambda_verb * L_verb_cls + lambda_geo * L_base_geo + I(has_onomato) * lambda_style * L_style_res
  2. Compute & Report:
     - Verb Classification Accuracy (on 100% of samples)
     - Base Geometry R^2 (on 100% of samples)
     - Residual Style Dynamics R^2 (on pure onomatopoeia samples where has_onomato == True)
  3. Validate how 100% data retention (506 segments) improves geometric stability
     and sample density compared to the sparse onomatopoeia-only baseline.
=============================================================================
"""

import os
import sys
import json
import time
from pathlib import Path
import numpy as np

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from sklearn.metrics import r2_score, accuracy_score

from dataset_builder_v2 import DecoupledDatasetBuilder, DecoupledMotionDataset, VERB_CLASSES
from decoupled_motion_model import DecoupledMotionModel


class TargetNormalizer:
    """幾何・スタイル変量のための平均・分散正規化スケーラー"""
    def __init__(self):
        self.geo_mean = None
        self.geo_std = None
        self.style_mean = None
        self.style_std = None

    def fit(self, y_geo: np.ndarray, y_style: np.ndarray, has_onomato: np.ndarray):
        self.geo_mean = np.mean(y_geo, axis=0, keepdims=True)
        self.geo_std = np.std(y_geo, axis=0, keepdims=True) + 1e-6

        # Style targets are fitted primarily on onomatopoeia-valid samples
        valid_style = y_style[has_onomato] if np.sum(has_onomato) > 0 else y_style
        self.style_mean = np.mean(valid_style, axis=0, keepdims=True)
        self.style_std = np.std(valid_style, axis=0, keepdims=True) + 1e-6

    def transform_geo(self, y_geo: np.ndarray) -> np.ndarray:
        return (y_geo - self.geo_mean) / self.geo_std

    def inverse_transform_geo(self, y_geo_norm: np.ndarray) -> np.ndarray:
        return (y_geo_norm * self.geo_std) + self.geo_mean

    def transform_style(self, y_style: np.ndarray) -> np.ndarray:
        return (y_style - self.style_mean) / self.style_std

    def inverse_transform_style(self, y_style_norm: np.ndarray) -> np.ndarray:
        return (y_style_norm * self.style_std) + self.style_mean


def evaluate_model(
    model: nn.Module,
    val_loader: DataLoader,
    normalizer: TargetNormalizer,
    device: torch.device
):
    model.eval()
    all_verb_preds = []
    all_verb_trues = []

    all_geo_preds = []
    all_geo_trues = []

    style_preds_onoma = []
    style_trues_onoma = []

    with torch.no_grad():
        for batch in val_loader:
            motion = batch["motion_window"].to(device)
            verb_id = batch["verb_id"].to(device)
            y_geo = batch["y_geo"].to(device)
            y_style = batch["y_style"].to(device)
            has_onoma = batch["has_onomato"].to(device)

            out = model(motion)

            # Verb Accuracy
            preds_cls = torch.argmax(out["verb_logits"], dim=-1).cpu().numpy()
            all_verb_preds.extend(preds_cls)
            all_verb_trues.extend(verb_id.cpu().numpy())

            # Base Geometry
            pred_geo = out["base_geo_pred"].cpu().numpy()
            true_geo = y_geo.cpu().numpy()
            all_geo_preds.append(pred_geo)
            all_geo_trues.append(true_geo)

            # Residual Style (Masked for onomatopoeia-valid only)
            pred_style = out["style_pred"].cpu().numpy()
            true_style = y_style.cpu().numpy()
            mask = has_onoma.cpu().numpy()

            if np.sum(mask) > 0:
                style_preds_onoma.append(pred_style[mask])
                style_trues_onoma.append(true_style[mask])

    all_geo_preds = np.vstack(all_geo_preds)
    all_geo_trues = np.vstack(all_geo_trues)

    # Denormalize
    orig_geo_preds = normalizer.inverse_transform_geo(all_geo_preds)
    orig_geo_trues = normalizer.inverse_transform_geo(all_geo_trues)

    verb_acc = float(accuracy_score(all_verb_trues, all_verb_preds))

    # Base Geometry R^2 (per feature & overall mean)
    geo_names = ["T_torsion", "Enclosure_Area", "Flexion_Sin", "VS_Mode_Ratio"]
    geo_r2_dict = {}
    for i, name in enumerate(geo_names):
        r2 = float(r2_score(orig_geo_trues[:, i], orig_geo_preds[:, i]))
        geo_r2_dict[name] = max(-1.0, round(r2, 4))
    mean_geo_r2 = float(np.mean(list(geo_r2_dict.values())))

    # Style Residuals R^2
    style_names = ["Jerk_Spike", "Stiffness_K", "Effort_Weight", "Effort_Time", "Effort_Space", "Effort_Flow"]
    style_r2_dict = {}
    if len(style_preds_onoma) > 0:
        all_style_preds = np.vstack(style_preds_onoma)
        all_style_trues = np.vstack(style_trues_onoma)
        orig_style_preds = normalizer.inverse_transform_style(all_style_preds)
        orig_style_trues = normalizer.inverse_transform_style(all_style_trues)

        for i, name in enumerate(style_names):
            r2 = float(r2_score(orig_style_trues[:, i], orig_style_preds[:, i]))
            style_r2_dict[name] = max(-1.0, round(r2, 4))
        mean_style_r2 = float(np.mean(list(style_r2_dict.values())))
        onoma_count = len(all_style_trues)
    else:
        mean_style_r2 = 0.0
        onoma_count = 0

    return {
        "verb_accuracy": round(verb_acc, 4),
        "mean_geo_r2": round(mean_geo_r2, 4),
        "geo_r2_breakdown": geo_r2_dict,
        "mean_style_r2": round(mean_style_r2, 4),
        "style_r2_breakdown": style_r2_dict,
        "total_eval_samples": len(all_verb_trues),
        "onoma_eval_samples": onoma_count
    }


def train_decoupled_pipeline(
    epochs: int = 50,
    batch_size: int = 16,
    lr: float = 1e-3,
    lambda_verb: float = 1.0,
    lambda_geo: float = 1.2,
    lambda_style: float = 1.5,
    seed: int = 42
):
    print("=" * 80)
    print("=== [KinematicEngine] Decoupled 2-Layer Multitask Training Pipeline ===")
    print("=" * 80)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Compute Device: {device}")

    # 1. データセット構築 (全506件 100% 非破壊抽出)
    builder = DecoupledDatasetBuilder()
    samples = builder.build_dataset_samples()

    # 2. 幾何・スタイル変量の正規化
    y_geo_all = np.array([s["y_geo"] for s in samples], dtype=np.float32)
    y_style_all = np.array([s["y_style"] for s in samples], dtype=np.float32)
    has_onoma_all = np.array([s["has_onomato"] for s in samples], dtype=bool)

    normalizer = TargetNormalizer()
    normalizer.fit(y_geo_all, y_style_all, has_onoma_all)

    # 各サンプルに正規化ターゲットを適用
    y_geo_norm = normalizer.transform_geo(y_geo_all)
    y_style_norm = normalizer.transform_style(y_style_all)
    for i, s in enumerate(samples):
        s["y_geo"] = y_geo_norm[i]
        s["y_style"] = y_style_norm[i]

    # 3. Train / Val 分割 (80% / 20%)
    np.random.seed(seed)
    torch.manual_seed(seed)
    indices = np.random.permutation(len(samples))
    split = int(len(samples) * 0.8)

    train_samples = [samples[i] for i in indices[:split]]
    val_samples = [samples[i] for i in indices[split:]]

    train_ds = DecoupledMotionDataset(train_samples)
    val_ds = DecoupledMotionDataset(val_samples)

    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False)

    print(f"Data Split: Train={len(train_ds)} samples (Onoma={sum(s['has_onomato'] for s in train_samples)}), "
          f"Val={len(val_ds)} samples (Onoma={sum(s['has_onomato'] for s in val_samples)})")

    # 4. モデル構築
    model = DecoupledMotionModel(
        in_channels=33,
        num_verbs=len(VERB_CLASSES),
        geo_dim=4,
        style_dim=6,
        hidden_dim=128
    ).to(device)

    optimizer = optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs, eta_min=1e-5)

    best_val_score = -999.0
    best_metrics = {}

    print("\n--- Training Loop ---")
    for epoch in range(1, epochs + 1):
        model.train()
        running_loss = 0.0
        running_verb_loss = 0.0
        running_geo_loss = 0.0
        running_style_loss = 0.0

        for batch in train_loader:
            motion = batch["motion_window"].to(device)
            verb_id = batch["verb_id"].to(device)
            y_geo = batch["y_geo"].to(device)
            y_style = batch["y_style"].to(device)
            has_onoma = batch["has_onomato"].to(device)

            optimizer.zero_grad()
            preds = model(motion)

            targets = {
                "verb_id": verb_id,
                "y_geo": y_geo,
                "y_style": y_style,
                "has_onomato": has_onoma
            }

            loss_dict = model.compute_masked_loss(
                preds, targets,
                lambda_verb=lambda_verb,
                lambda_geo=lambda_geo,
                lambda_style=lambda_style
            )

            loss = loss_dict["total_loss"]
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=2.0)
            optimizer.step()

            running_loss += loss.item()
            running_verb_loss += loss_dict["verb_loss"].item()
            running_geo_loss += loss_dict["geo_loss"].item()
            running_style_loss += loss_dict["style_loss"].item()

        scheduler.step()

        if epoch % 5 == 0 or epoch == epochs:
            eval_res = evaluate_model(model, val_loader, normalizer, device)
            avg_loss = running_loss / len(train_loader)
            val_score = eval_res["mean_geo_r2"] + eval_res["mean_style_r2"] + eval_res["verb_accuracy"]

            print(f"Epoch [{epoch:02d}/{epochs}] Loss: {avg_loss:.4f} | "
                  f"Verb Acc: {eval_res['verb_accuracy']*100:.1f}% | "
                  f"Geo R^2: {eval_res['mean_geo_r2']:.3f} | "
                  f"Style R^2: {eval_res['mean_style_r2']:.3f}")

            if val_score > best_val_score:
                best_val_score = val_score
                best_metrics = eval_res
                # Save checkpoint
                out_path = Path(r"D:\Antigravity_Work\kinematic_engine\output\decoupled_model.pt")
                torch.save({
                    "model_state_dict": model.state_dict(),
                    "geo_mean": normalizer.geo_mean,
                    "geo_std": normalizer.geo_std,
                    "style_mean": normalizer.style_mean,
                    "style_std": normalizer.style_std,
                    "verb_classes": VERB_CLASSES,
                    "metrics": eval_res
                }, out_path)

    print("\n" + "=" * 80)
    print("=== Training Completed! Best Validation Evaluation Metrics ===")
    print("=" * 80)
    print(f"1. Global Motion Verb Classification Accuracy: {best_metrics['verb_accuracy']*100:.2f}% (All {best_metrics['total_eval_samples']} samples)")
    print(f"2. Base Kinematic Geometry R^2:               {best_metrics['mean_geo_r2']:.4f} (All samples)")
    for k, v in best_metrics["geo_r2_breakdown"].items():
        print(f"     - {k:16s}: R^2 = {v:+.4f}")
    print(f"3. Residual Style Dynamics R^2:               {best_metrics['mean_style_r2']:.4f} (Onoma valid: {best_metrics['onoma_eval_samples']} samples)")
    for k, v in best_metrics["style_r2_breakdown"].items():
        print(f"     - {k:16s}: R^2 = {v:+.4f}")

    # Output JSON summary
    summary_path = Path(r"D:\Antigravity_Work\kinematic_engine\output\decoupled_evaluation_summary.json")
    with open(summary_path, "w", encoding="utf-8") as f:
        json.dump(best_metrics, f, ensure_ascii=False, indent=2)
    print(f"\n[Saved] Checkpoint -> output/decoupled_model.pt")
    print(f"[Saved] Summary    -> output/decoupled_evaluation_summary.json")

    return best_metrics


if __name__ == "__main__":
    train_decoupled_pipeline(epochs=40, batch_size=16, lr=1e-3)
