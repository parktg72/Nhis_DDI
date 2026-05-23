"""
Phase 3 모델 래퍼 — sklearn 인터페이스 호환

TabNet   : 즉시 사용 가능 (pytorch-tabnet)
GNN      : 실험적 (torch_geometric 필요, 환자 레벨 집계 피처 기반)
Transformer : 실험적 (torch 필요, 환자 레벨 집계 피처 기반)

모든 래퍼는 fit/predict/predict_proba + feature_importances_ 를 지원하여
ml_runner.train_model 및 cross_val_score와 호환됩니다.
"""
from __future__ import annotations

import logging
from typing import Any

import numpy as np

logger = logging.getLogger(__name__)


# ─────────────────────────────────────────────────────────────────────────────
# TabNet 래퍼
# ─────────────────────────────────────────────────────────────────────────────

class TabNetWrapper:
    """pytorch-tabnet TabNetClassifier를 sklearn 인터페이스로 래핑."""

    def __init__(
        self,
        n_d: int = 8,
        n_a: int = 8,
        n_steps: int = 3,
        gamma: float = 1.3,
        max_epochs: int = 100,
        patience: int = 15,
        batch_size: int = 1024,
        virtual_batch_size: int = 128,
        n_classes: int = 2,
        use_gpu: bool = False,
    ):
        self.n_d = n_d
        self.n_a = n_a
        self.n_steps = n_steps
        self.gamma = gamma
        self.max_epochs = max_epochs
        self.patience = patience
        self.batch_size = batch_size
        self.virtual_batch_size = virtual_batch_size
        self.n_classes = n_classes
        self.use_gpu = use_gpu
        self._model = None
        self.feature_importances_ = None
        self.classes_ = None

    def fit(self, X, y):
        from pytorch_tabnet.tab_model import TabNetClassifier

        X_arr = np.array(X, dtype=np.float32)
        y_arr = np.array(y, dtype=np.int64)
        self.classes_ = np.unique(y_arr)

        try:
            import torch
            _cuda_available = self.use_gpu and torch.cuda.is_available()
        except ImportError:
            _cuda_available = False
        device = "cuda" if _cuda_available else "cpu"
        self._model = TabNetClassifier(
            n_d=self.n_d,
            n_a=self.n_a,
            n_steps=self.n_steps,
            gamma=self.gamma,
            device_name=device,
            verbose=0,
        )
        self._model.fit(
            X_arr, y_arr,
            max_epochs=self.max_epochs,
            patience=self.patience,
            batch_size=self.batch_size,
            virtual_batch_size=self.virtual_batch_size,
        )
        self.feature_importances_ = self._model.feature_importances_
        return self

    def predict(self, X):
        X_arr = np.array(X, dtype=np.float32)
        return self._model.predict(X_arr)

    def predict_proba(self, X):
        X_arr = np.array(X, dtype=np.float32)
        return self._model.predict_proba(X_arr)

    def get_params(self, deep=True):
        return {
            "n_d": self.n_d, "n_a": self.n_a, "n_steps": self.n_steps,
            "gamma": self.gamma, "max_epochs": self.max_epochs,
            "patience": self.patience, "batch_size": self.batch_size,
            "virtual_batch_size": self.virtual_batch_size,
            "n_classes": self.n_classes, "use_gpu": self.use_gpu,
        }

    def set_params(self, **params):
        for k, v in params.items():
            setattr(self, k, v)
        return self


# ─────────────────────────────────────────────────────────────────────────────
# GNN 래퍼 (실험적)
# ─────────────────────────────────────────────────────────────────────────────

class GNNWrapper:
    """
    DEPRECATED pseudo-DL wrapper over aggregate tabular features.

    현재 피처 구조(환자 레벨 집계 16컬럼)에서는 MLP와 유사하게 동작합니다.
    운영 DL/GNN 효과를 위해서는 약물 공동처방 그래프(raw 처방 데이터)와
    drug_vocab/edge_index/model_config를 포함한 DL artifact bundle이 필요합니다.
    여기서는 간단한 2-layer GCN을 MLP 대용으로 구현합니다.
    """

    def __init__(
        self,
        hidden_dim: int = 64,
        num_layers: int = 2,
        dropout: float = 0.3,
        max_epochs: int = 50,
        lr: float = 0.001,
        n_classes: int = 2,
        use_gpu: bool = False,
    ):
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        self.dropout = dropout
        self.max_epochs = max_epochs
        self.lr = lr
        self.n_classes = n_classes
        self.use_gpu = use_gpu
        self._model = None
        self.feature_importances_ = None
        self.classes_ = None

    def fit(self, X, y):
        import torch
        import torch.nn as nn

        X_t = torch.tensor(np.array(X, dtype=np.float32))
        y_t = torch.tensor(np.array(y, dtype=np.int64))
        self.classes_ = np.unique(y)
        n_feat = X_t.shape[1]

        device = torch.device("cuda" if self.use_gpu and torch.cuda.is_available() else "cpu")

        # Simple MLP (GNN 구조의 간소화 버전 — 노드 피처만 사용)
        layers = []
        in_dim = n_feat
        for _ in range(self.num_layers):
            layers.extend([
                nn.Linear(in_dim, self.hidden_dim),
                nn.ReLU(),
                nn.Dropout(self.dropout),
            ])
            in_dim = self.hidden_dim
        layers.append(nn.Linear(in_dim, self.n_classes))
        net = nn.Sequential(*layers).to(device)

        optimizer = torch.optim.Adam(net.parameters(), lr=self.lr)
        loss_fn = nn.CrossEntropyLoss()

        X_t, y_t = X_t.to(device), y_t.to(device)
        net.train()
        for _ in range(self.max_epochs):
            optimizer.zero_grad()
            out = net(X_t)
            loss = loss_fn(out, y_t)
            loss.backward()
            optimizer.step()

        self._model = net
        self._device = device
        # 피처 중요도: 첫 번째 레이어 가중치 절댓값 합
        w = list(net.parameters())[0].detach().cpu().numpy()
        self.feature_importances_ = np.abs(w).sum(axis=0)
        return self

    def predict(self, X):
        proba = self.predict_proba(X)
        return np.argmax(proba, axis=1)

    def predict_proba(self, X):
        import torch
        self._model.eval()
        X_t = torch.tensor(np.array(X, dtype=np.float32)).to(self._device)
        with torch.no_grad():
            logits = self._model(X_t)
            proba = torch.softmax(logits, dim=1).cpu().numpy()
        return proba

    def get_params(self, deep=True):
        return {
            "hidden_dim": self.hidden_dim, "num_layers": self.num_layers,
            "dropout": self.dropout, "max_epochs": self.max_epochs,
            "lr": self.lr, "n_classes": self.n_classes, "use_gpu": self.use_gpu,
        }

    def set_params(self, **params):
        for k, v in params.items():
            setattr(self, k, v)
        return self


# ─────────────────────────────────────────────────────────────────────────────
# Temporal Transformer 래퍼 (실험적)
# ─────────────────────────────────────────────────────────────────────────────

class TemporalTransformerWrapper:
    """
    DEPRECATED pseudo-DL wrapper over aggregate tabular features.

    현재 피처 구조(환자 레벨 집계)에서는 피처를 시퀀스 토큰으로 취급합니다.
    운영 Temporal DL은 환자 처방 이력 시퀀스와 mask/padding 계약을 별도
    DL dataset으로 받아야 합니다.
    실제 시계열 효과를 위해서는 원시 처방 시퀀스 데이터 필요.
    """

    def __init__(
        self,
        d_model: int = 32,
        nhead: int = 4,
        num_layers: int = 2,
        dropout: float = 0.1,
        max_epochs: int = 50,
        lr: float = 0.001,
        n_classes: int = 2,
        use_gpu: bool = False,
    ):
        self.d_model = d_model
        self.nhead = nhead
        self.num_layers = num_layers
        self.dropout = dropout
        self.max_epochs = max_epochs
        self.lr = lr
        self.n_classes = n_classes
        self.use_gpu = use_gpu
        self._model = None
        self.feature_importances_ = None
        self.classes_ = None

    def fit(self, X, y):
        import torch
        import torch.nn as nn

        X_arr = np.array(X, dtype=np.float32)
        y_arr = np.array(y, dtype=np.int64)
        self.classes_ = np.unique(y_arr)
        n_feat = X_arr.shape[1]

        device = torch.device("cuda" if self.use_gpu and torch.cuda.is_available() else "cpu")

        # 피처를 시퀀스 길이=n_feat, d_model 차원으로 투사
        class _TFModel(nn.Module):
            def __init__(self_, n_feat, d_model, nhead, num_layers, dropout, n_classes):
                super().__init__()
                self_.proj = nn.Linear(1, d_model)
                enc_layer = nn.TransformerEncoderLayer(
                    d_model=d_model, nhead=nhead, dropout=dropout, batch_first=True,
                )
                self_.encoder = nn.TransformerEncoder(enc_layer, num_layers=num_layers)
                self_.head = nn.Linear(d_model, n_classes)

            def forward(self_, x):
                # x: (batch, n_feat) → (batch, n_feat, 1) → (batch, n_feat, d_model)
                x = x.unsqueeze(-1)
                x = self_.proj(x)
                x = self_.encoder(x)
                x = x.mean(dim=1)  # 평균 풀링
                return self_.head(x)

        net = _TFModel(n_feat, self.d_model, self.nhead, self.num_layers,
                       self.dropout, self.n_classes).to(device)
        optimizer = torch.optim.Adam(net.parameters(), lr=self.lr)
        loss_fn = nn.CrossEntropyLoss()

        X_t = torch.tensor(X_arr).to(device)
        y_t = torch.tensor(y_arr).to(device)

        net.train()
        for _ in range(self.max_epochs):
            optimizer.zero_grad()
            out = net(X_t)
            loss = loss_fn(out, y_t)
            loss.backward()
            optimizer.step()

        self._model = net
        self._device = device
        # 피처 중요도: projection 레이어 가중치
        w = net.proj.weight.detach().cpu().numpy()
        self.feature_importances_ = np.abs(w).sum(axis=0)
        if len(self.feature_importances_) == 1:
            self.feature_importances_ = np.ones(n_feat) / n_feat
        return self

    def predict(self, X):
        proba = self.predict_proba(X)
        return np.argmax(proba, axis=1)

    def predict_proba(self, X):
        import torch
        self._model.eval()
        X_t = torch.tensor(np.array(X, dtype=np.float32)).to(self._device)
        with torch.no_grad():
            logits = self._model(X_t)
            proba = torch.softmax(logits, dim=1).cpu().numpy()
        return proba

    def get_params(self, deep=True):
        return {
            "d_model": self.d_model, "nhead": self.nhead,
            "num_layers": self.num_layers, "dropout": self.dropout,
            "max_epochs": self.max_epochs, "lr": self.lr,
            "n_classes": self.n_classes, "use_gpu": self.use_gpu,
        }

    def set_params(self, **params):
        for k, v in params.items():
            setattr(self, k, v)
        return self


# ─────────────────────────────────────────────────────────────────────────────
# 팩토리
# ─────────────────────────────────────────────────────────────────────────────

def build_phase3_model(
    model_name: str,
    n_classes: int = 2,
    params: dict[str, Any] | None = None,
    use_gpu: bool = False,
    n_jobs: int = -1,
):
    """Phase 3 모델 인스턴스 생성."""
    p = params or {}

    if model_name == "tabnet":
        return TabNetWrapper(
            n_d=p.get("n_d", 8),
            n_a=p.get("n_a", 8),
            n_steps=p.get("n_steps", 3),
            gamma=p.get("gamma", 1.3),
            max_epochs=p.get("max_epochs", 100),
            patience=p.get("patience", 15),
            batch_size=p.get("batch_size", 1024),
            n_classes=n_classes,
            use_gpu=use_gpu,
        )

    elif model_name == "gnn":
        return GNNWrapper(
            hidden_dim=p.get("hidden_dim", 64),
            num_layers=p.get("num_layers", 2),
            dropout=p.get("dropout", 0.3),
            max_epochs=p.get("max_epochs", 50),
            lr=p.get("lr", 0.001),
            n_classes=n_classes,
            use_gpu=use_gpu,
        )

    elif model_name == "temporal_transformer":
        return TemporalTransformerWrapper(
            d_model=p.get("d_model", 32),
            nhead=p.get("nhead", 4),
            num_layers=p.get("num_layers", 2),
            dropout=p.get("dropout", 0.1),
            max_epochs=p.get("max_epochs", 50),
            lr=p.get("lr", 0.001),
            n_classes=n_classes,
            use_gpu=use_gpu,
        )

    raise ValueError(f"알 수 없는 Phase 3 모델: {model_name}")
