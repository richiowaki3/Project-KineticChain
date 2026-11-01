"""
decoupled_motion_model.py
=============================================================================
Task 2: 2-Layer Decoupled Multitask Network (PyTorch)
Part of the Decoupled Hierarchical Motion-ML Pipeline.

Architecture Overview:
  1. Shared Skeleton Encoder:
     - Input: 3D Upper-body Axis series (Batch, 13 frames @ 30fps, 33 dims [11 joints x 3])
     - 1D-CNN (Temporal Convolution + BatchNorm + GELU)
     - Positional Embedding + Multi-Head Self-Attention (Transformer Encoder)
     - Aggregation -> Motion Latent Z_motion (128 dims)
  
  2. Head 1: Base Kinematics Head (Global Motion Topology):
     - Subhead 1A: Verb Classifier (12 classes, CrossEntropy Loss)
     - Subhead 1B: Base Geometry Regressor (T_torsion, Enclosure Area, Flexion, VS_mode)
     - Trained continuously on ALL 100% of samples (regardless of onomatopoeia).
  
  3. Head 2: Residual Style Head (Local Dynamics & Laban Texture Residuals):
     - Conditioned on [Z_motion + Z_verb]
     - Predicts: Jerk Spike, Virtual Stiffness K_est, Laban Effort 4-Axes (x1~x4)
     - Masked Multi-Task Loss: Computed ONLY when `has_onomato == True`
       (loss is multiplied by 0 when `has_onomato == False`).
=============================================================================
"""

import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, Any, Tuple, Optional


class PositionalEncoding(nn.Module):
    """標準正弦波 Positional Encoding"""
    def __init__(self, d_model: int, max_len: int = 64):
        super().__init__()
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        self.register_buffer("pe", pe.unsqueeze(0)) # shape (1, max_len, d_model)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (B, T, D)
        return x + self.pe[:, :x.size(1)]


class SharedSkeletonEncoder(nn.Module):
    """
    3Dノード時系列（13フレーム × 33次元）から潜在運動表現 Z_motion を抽出する。
    構造: 1D-CNN (時系列局所特徴抽出) + 2層 Transformer Encoder (大域相関) + Attention Pooling
    """
    def __init__(
        self,
        in_channels: int = 33,    # 11関節 × 3軸
        hidden_dim: int = 128,
        num_heads: int = 4,
        num_transformer_layers: int = 2,
        dropout: float = 0.1
    ):
        super().__init__()
        self.in_channels = in_channels
        self.hidden_dim = hidden_dim

        # 1. 1D Temporal Convolution Blocks
        self.conv1 = nn.Sequential(
            nn.Conv1d(in_channels, 64, kernel_size=3, padding=1),
            nn.BatchNorm1d(64),
            nn.GELU()
        )
        self.conv2 = nn.Sequential(
            nn.Conv1d(64, hidden_dim, kernel_size=3, padding=1),
            nn.BatchNorm1d(hidden_dim),
            nn.GELU()
        )

        # 2. Positional Encoding
        self.pos_encoder = PositionalEncoding(hidden_dim, max_len=32)

        # 3. Transformer Encoder
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=hidden_dim,
            nhead=num_heads,
            dim_feedforward=hidden_dim * 2,
            dropout=dropout,
            activation="gelu",
            batch_first=True
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_transformer_layers)

        # 4. Temporal Attention Pooling
        self.pool_attn = nn.Sequential(
            nn.Linear(hidden_dim, 32),
            nn.Tanh(),
            nn.Linear(32, 1)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
          x: (B, T, 33) - 13 frames @ 30fps
        Returns:
          z_motion: (B, hidden_dim) - 運動潜在表現
        """
        # (B, T, C) -> (B, C, T) for Conv1D
        x_conv = x.transpose(1, 2)
        h = self.conv1(x_conv)
        h = self.conv2(h)
        # (B, hidden_dim, T) -> (B, T, hidden_dim)
        h = h.transpose(1, 2)

        # Transformer
        h = self.pos_encoder(h)
        h_trans = self.transformer(h) # (B, T, hidden_dim)

        # Attention Weighted Pooling
        attn_weights = F.softmax(self.pool_attn(h_trans), dim=1) # (B, T, 1)
        z_motion = torch.sum(h_trans * attn_weights, dim=1)      # (B, hidden_dim)

        return z_motion


class BaseKinematicsHead(nn.Module):
    """
    Head 1: Base Kinematics Head
    動詞クラス分類（トポロジー）および大域幾何量（Base Geometry）を常時学習。
    """
    def __init__(self, in_features: int = 128, num_verbs: int = 12, geo_dim: int = 4):
        super().__init__()
        self.in_features = in_features

        # 1-A: Verb Classifier (Intermediate latent z_verb: 64D)
        self.verb_backbone = nn.Sequential(
            nn.Linear(in_features, 64),
            nn.GELU(),
            nn.Dropout(0.15)
        )
        self.verb_classifier = nn.Linear(64, num_verbs)

        # 1-B: Base Geometry Regressor (T_torsion, Enclosure, Flexion, VS_mode)
        self.geo_regressor = nn.Sequential(
            nn.Linear(in_features, 64),
            nn.GELU(),
            nn.Linear(64, geo_dim)
        )

    def forward(self, z_motion: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Returns:
          verb_logits: (B, num_verbs)
          base_geo_pred: (B, 4)
          z_verb: (B, 64) - 動詞潜在特徴
        """
        z_verb = self.verb_backbone(z_motion)
        verb_logits = self.verb_classifier(z_verb)
        base_geo_pred = self.geo_regressor(z_motion)

        return verb_logits, base_geo_pred, z_verb


class ResidualStyleHead(nn.Module):
    """
    Head 2: Residual Style Head
    動詞潜在表現と合流し、基底幾何運動からの「局所力学質感・スタイル残差」を回帰。
    Target (6D): [Jerk Spike, Stiffness K_est, Laban Weight, Laban Time, Laban Space, Laban Flow]
    """
    def __init__(self, motion_dim: int = 128, verb_feat_dim: int = 64, style_dim: int = 6):
        super().__init__()
        fusion_dim = motion_dim + verb_feat_dim

        self.mlp = nn.Sequential(
            nn.Linear(fusion_dim, 128),
            nn.GELU(),
            nn.Dropout(0.15),
            nn.Linear(128, 64),
            nn.GELU(),
            nn.Linear(64, style_dim)
        )

    def forward(self, z_motion: torch.Tensor, z_verb: torch.Tensor) -> torch.Tensor:
        """
        Args:
          z_motion: (B, 128)
          z_verb:   (B, 64)
        Returns:
          style_pred: (B, 6)
        """
        fusion = torch.cat([z_motion, z_verb], dim=-1)
        style_pred = self.mlp(fusion)
        return style_pred


class DecoupledMotionModel(nn.Module):
    """
    2層解離型マルチタスクネットワーク統合モデル
    """
    def __init__(
        self,
        in_channels: int = 33,
        num_verbs: int = 12,
        geo_dim: int = 4,
        style_dim: int = 6,
        hidden_dim: int = 128
    ):
        super().__init__()
        self.encoder = SharedSkeletonEncoder(
            in_channels=in_channels,
            hidden_dim=hidden_dim
        )
        self.base_head = BaseKinematicsHead(
            in_features=hidden_dim,
            num_verbs=num_verbs,
            geo_dim=geo_dim
        )
        self.style_head = ResidualStyleHead(
            motion_dim=hidden_dim,
            verb_feat_dim=64,
            style_dim=style_dim
        )

    def forward(self, motion_seq: torch.Tensor) -> Dict[str, torch.Tensor]:
        """
        Args:
          motion_seq: (B, 13, 33) - 3D ノード時系列 (Stroke Peak ±0.2s)
        Returns:
          Dict containing:
            verb_logits:   (B, num_verbs)
            base_geo_pred: (B, 4)
            style_pred:    (B, 6)
            z_motion:      (B, hidden_dim)
            z_verb:        (B, 64)
        """
        z_motion = self.encoder(motion_seq)
        verb_logits, base_geo_pred, z_verb = self.base_head(z_motion)
        style_pred = self.style_head(z_motion, z_verb)

        return {
            "verb_logits": verb_logits,
            "base_geo_pred": base_geo_pred,
            "style_pred": style_pred,
            "z_motion": z_motion,
            "z_verb": z_verb
        }

    def compute_masked_loss(
        self,
        preds: Dict[str, torch.Tensor],
        targets: Dict[str, torch.Tensor],
        lambda_verb: float = 1.0,
        lambda_geo: float = 1.0,
        lambda_style: float = 1.5
    ) -> Dict[str, torch.Tensor]:
        """
        Task 3 準拠 マスク付きマルチタスク損失計算:
          L_total = lambda_verb * L_verb_cls + lambda_geo * L_base_geo + I(has_onomato) * lambda_style * L_style_res
        """
        # 1. 動詞分類損失 (CrossEntropy) - 全サンプル常時適用
        verb_loss = F.cross_entropy(preds["verb_logits"], targets["verb_id"])

        # 2. 大域幾何回帰損失 (Smooth L1 / MSE) - 全サンプル常時適用
        geo_loss = F.smooth_l1_loss(preds["base_geo_pred"], targets["y_geo"])

        # 3. 局所力学質感・スタイル残差損失 (Masked Smooth L1)
        # has_onomato == True のサンプルのみ計算、False のサンプルはマスク
        has_onomato = targets["has_onomato"].float() # (B,)
        raw_style_loss_per_sample = F.smooth_l1_loss(
            preds["style_pred"], targets["y_style"], reduction="none"
        ).mean(dim=-1) # (B,)

        masked_style_loss = (raw_style_loss_per_sample * has_onomato).sum() / (has_onomato.sum() + 1e-6)

        # 総合損失
        total_loss = (
            lambda_verb * verb_loss +
            lambda_geo * geo_loss +
            lambda_style * masked_style_loss
        )

        return {
            "total_loss": total_loss,
            "verb_loss": verb_loss,
            "geo_loss": geo_loss,
            "style_loss": masked_style_loss,
            "onomato_sample_count": has_onomato.sum()
        }


if __name__ == "__main__":
    # Test network instantiation and dummy forward pass
    B, T, C = 8, 13, 33
    dummy_motion = torch.randn(B, T, C)
    model = DecoupledMotionModel()
    outputs = model(dummy_motion)

    print(f"DecoupledMotionModel Forward Test:")
    print(f"  Input: {dummy_motion.shape}")
    print(f"  verb_logits:   {outputs['verb_logits'].shape}")
    print(f"  base_geo_pred: {outputs['base_geo_pred'].shape}")
    print(f"  style_pred:    {outputs['style_pred'].shape}")
    print(f"  z_motion:      {outputs['z_motion'].shape}")

    dummy_targets = {
        "verb_id": torch.randint(0, 12, (B,)),
        "y_geo": torch.randn(B, 4),
        "y_style": torch.randn(B, 6),
        "has_onomato": torch.tensor([True, False, True, False, True, True, False, True])
    }
    losses = model.compute_masked_loss(outputs, dummy_targets)
    print(f"Loss Test:")
    print(f"  total_loss: {losses['total_loss'].item():.4f}")
    print(f"  verb_loss:  {losses['verb_loss'].item():.4f}")
    print(f"  geo_loss:   {losses['geo_loss'].item():.4f}")
    print(f"  style_loss: {losses['style_loss'].item():.4f}")
    print(f"  onomato_count in batch: {losses['onomato_sample_count'].item():.0f}/{B}")
