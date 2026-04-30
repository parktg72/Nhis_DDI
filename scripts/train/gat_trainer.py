"""GATTrainer — GAT 학습, Platt 보정, 저장/로드."""
from __future__ import annotations

import hashlib
import logging
import pickle
import time
from pathlib import Path
from typing import Any, Optional

import numpy as np

from .base_graph_trainer import BaseGraphTrainer
from .gat_dataset import GATDataset
from scripts.features.graph_builder import GraphBuilder

logger = logging.getLogger(__name__)

POSITIVE_SEVERITIES = {"contraindicated", "major"}
NEG_POS_RATIO = 5


class GATTrainer(BaseGraphTrainer):
    """
    GAT 훈련기.

    params dict keys
    ----------------
    hidden_dim   : GAT layer 1 hidden (default 64)
    heads        : Attention heads (default 4)
    out_dim      : GAT layer 2 output dim (default 32)
    epochs       : 최대 에폭 (default 200)
    lr           : 학습률 (default 0.001)
    patience     : 조기종료 patience (default 20)
    random_state : 재현성 시드 (default 42)
    """

    def __init__(self, params: dict, config: Any, model_dir: str | Path = "models"):
        super().__init__(params, config)
        self.model_dir = Path(model_dir)
        self._gat_model = None
        self._graph_builder: Optional[GraphBuilder] = None
        self._calibrator = None

    def fit_graph(self, dataset: GATDataset) -> "GATTrainer":
        try:
            import torch
            import torch.nn as nn
            from torch.optim import Adam
            from sklearn.metrics import roc_auc_score
        except ImportError as e:
            raise ImportError(f"의존성 없음: {e}") from e

        from .gat_model import GATModel

        seed = self.params.get("random_state", 42)
        torch.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        try:
            torch.use_deterministic_algorithms(True)
        except Exception:
            pass

        t0 = time.perf_counter()

        # 1. 그래프 빌드 (train 데이터만)
        split_attr = dataset.prescription_df.attrs.get("split") or dataset.prescription_split
        if str(split_attr).strip().lower() != "train":
            raise RuntimeError(
                "GATTrainer.fit_graph()는 train split 처방 데이터만 허용합니다. "
                "GATDataset.prescription_split='train' 또는 prescription_df.attrs['split']='train' 필요"
            )
        self._graph_builder = GraphBuilder()
        graph_data = self._graph_builder.build(dataset.prescription_df, dataset.ddi_df)
        drug_to_idx = self._graph_builder.drug_to_idx

        # 2. DDI 긍정쌍 생성
        pos_pairs = []
        for _, row in dataset.ddi_df.iterrows():
            if str(row["severity"]).lower() not in POSITIVE_SEVERITIES:
                continue
            ai = drug_to_idx.get(str(row["drug_a"]))
            bi = drug_to_idx.get(str(row["drug_b"]))
            if ai is not None and bi is not None:
                pos_pairs.append((ai, bi, 1))

        if not pos_pairs:
            logger.warning("긍정 DDI 쌍 없음 — 더미 학습 스킵, _trained=True 설정")
            self._gat_model = GATModel(
                feature_dim=graph_data.x.shape[1],
                hidden_dim=self.params.get("hidden_dim", 64),
                heads=self.params.get("heads", 4),
                out_dim=self.params.get("out_dim", 32),
            )
            self._trained = True
            return self

        # 3. 부정쌍 샘플링 (NEG_POS_RATIO:1)
        rng = np.random.default_rng(seed)
        pos_set = {(a, b) for a, b, _ in pos_pairs} | {(b, a) for a, b, _ in pos_pairs}
        n_nodes = len(drug_to_idx)
        neg_pairs = []
        attempts = 0
        target_neg = len(pos_pairs) * NEG_POS_RATIO
        while len(neg_pairs) < target_neg and attempts < target_neg * 20:
            a, b = int(rng.integers(n_nodes)), int(rng.integers(n_nodes))
            if a != b and (a, b) not in pos_set:
                neg_pairs.append((a, b, 0))
            attempts += 1

        all_pairs = pos_pairs + neg_pairs
        rng.shuffle(all_pairs)
        all_pairs = np.array(all_pairs, dtype=np.int64)

        # 4. DDI 쌍 분할: gat_val 10% / calibration 10% / train 나머지(~80%)
        # 참고: 60/20/10/10은 환자 단위 분할 기준. 쌍 단위는 gat_val+calib=20%, train=~80%.
        n = len(all_pairs)
        n_calib = max(1, int(n * 0.10))
        n_gatval = max(1, int(n * 0.10))
        n_train = n - n_calib - n_gatval

        tr_pairs = all_pairs[:n_train]
        gv_pairs = all_pairs[n_train:n_train + n_gatval]
        ca_pairs = all_pairs[n_train + n_gatval:]

        dataset.pairs_train = tr_pairs
        dataset.pairs_gat_val = gv_pairs
        dataset.pairs_calibration = ca_pairs

        # 5. 모델 초기화
        feature_dim = graph_data.x.shape[1]
        model = GATModel(
            feature_dim=feature_dim,
            hidden_dim=self.params.get("hidden_dim", 64),
            heads=self.params.get("heads", 4),
            out_dim=self.params.get("out_dim", 32),
        )
        optimizer = Adam(model.parameters(), lr=self.params.get("lr", 0.001))
        criterion = nn.BCELoss()

        x = graph_data.x
        edge_index = graph_data.edge_index

        # 6. 학습 루프 (조기종료: gat_val AUC)
        epochs = self.params.get("epochs", 200)
        patience = self.params.get("patience", 20)
        best_auc, no_improve = 0.0, 0
        best_state = None

        tr_pairs_t = torch.tensor(tr_pairs[:, :2], dtype=torch.long)
        tr_labels_t = torch.tensor(tr_pairs[:, 2], dtype=torch.float)
        gv_pairs_t = torch.tensor(gv_pairs[:, :2], dtype=torch.long)
        gv_labels = gv_pairs[:, 2]

        for epoch in range(epochs):
            model.train()
            optimizer.zero_grad()
            scores = model.score_pairs(x, edge_index, tr_pairs_t)
            loss = criterion(scores, tr_labels_t)
            loss.backward()
            optimizer.step()

            if (epoch + 1) % max(1, epochs // 20) == 0:
                model.eval()
                with torch.no_grad():
                    gv_scores = model.score_pairs(x, edge_index, gv_pairs_t).numpy()
                if len(np.unique(gv_labels)) > 1:
                    auc = roc_auc_score(gv_labels, gv_scores)
                    if auc > best_auc:
                        best_auc = auc
                        no_improve = 0
                        best_state = {k: v.clone() for k, v in model.state_dict().items()}
                    else:
                        no_improve += 1
                    if no_improve >= max(1, patience // (epochs // 20)):
                        logger.info("조기종료: epoch=%d, best_gat_val_auc=%.4f", epoch+1, best_auc)
                        break

        if best_state:
            model.load_state_dict(best_state)
        self._gat_model = model
        # EnsembleTrainer3Way.fit_gat() 가 BaseGraphTrainer.fit() wrapper 를 우회해
        # 여기를 직접 호출하므로, calibrate() / save() 가 검사하는 _trained 를 명시 set.
        self._trained = True
        logger.info("GATTrainer 학습 완료 (%.1fs): gat_val_auc=%.4f, pairs=%d",
                    time.perf_counter() - t0, best_auc, len(tr_pairs))

        # 보정 (Platt scaling) — calibration 쌍이 있으면 자동 수행
        if dataset.pairs_calibration is not None and len(dataset.pairs_calibration) > 0:
            logger.info("Platt scaling 보정 수행 (calibration 쌍 %d개)", len(dataset.pairs_calibration))
            self.calibrate(dataset.pairs_calibration)
        else:
            logger.warning("calibration 쌍 없음 — Platt scaling 미적용")

        return self

    def predict_pair_proba(self, drug_a: str, drug_b: str) -> float | None:
        """단일 약물쌍 위험 확률. 미지 약물 → None."""
        if not self._trained or self._gat_model is None:
            return None
        d2i = self._graph_builder.drug_to_idx
        ai = d2i.get(drug_a)
        bi = d2i.get(drug_b)
        if ai is None or bi is None:
            logger.warning("알 수 없는 약물 코드 — GAT 서브모델 제외: %s, %s", drug_a, drug_b)
            return None
        try:
            import torch
            self._gat_model.eval()
            data = self._graph_builder.data
            pairs_t = torch.tensor([[ai, bi]], dtype=torch.long)
            with torch.no_grad():
                raw = float(self._gat_model.score_pairs(data.x, data.edge_index, pairs_t)[0])
            if self._calibrator is not None:
                raw = float(self._calibrator.predict_proba([[raw]])[:, 1][0])
            return raw
        except Exception as e:
            logger.error("GAT 예측 오류: %s", e)
            return None

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        """serving 호환 인터페이스 — GAT는 쌍 단위 예측이므로 0.5 반환."""
        logger.warning("GATTrainer.predict_proba(numpy array) 미지원 — predict_pair_proba 사용 권장")
        return np.full(len(X), 0.5)

    def calibrate(self, calibration_pairs: np.ndarray) -> None:
        """Platt scaling 보정. calibration_pairs: [N, 3] (node_a, node_b, label)"""
        try:
            import torch
            from sklearn.linear_model import LogisticRegression
        except ImportError as e:
            raise ImportError(f"의존성 없음: {e}") from e

        if not self._trained or self._gat_model is None:
            raise RuntimeError("fit_graph() 먼저 호출하세요.")

        data = self._graph_builder.data
        pairs_t = torch.tensor(calibration_pairs[:, :2], dtype=torch.long)
        labels = calibration_pairs[:, 2].astype(int)

        self._gat_model.eval()
        with torch.no_grad():
            raw_scores = self._gat_model.score_pairs(
                data.x, data.edge_index, pairs_t
            ).numpy()

        if len(np.unique(labels)) < 2:
            logger.warning("calibration_pairs 단일 클래스 — Platt scaling 생략")
            return

        self._calibrator = LogisticRegression()
        self._calibrator.fit(raw_scores.reshape(-1, 1), labels)
        logger.info("Platt scaling 보정 완료 (calibration_pairs=%d)", len(labels))

    def save(self, path: str | Path) -> Path:
        """gat_model.pt + .sha256 저장. graph는 path.parent에 별도 저장."""
        if not self._trained:
            raise RuntimeError("fit_graph() 먼저 호출하세요.")
        try:
            import torch
        except ImportError as e:
            raise ImportError("torch 미설치") from e

        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)

        if self._calibrator is None:
            logger.warning(
                "GATTrainer.save(): _calibrator 미설정 — Platt 스케일링 미적용. "
                "calibrate() 호출 후 저장 권장."
            )

        payload = {
            "model_state": self._gat_model.state_dict(),
            "model_init_params": {
                "feature_dim": self._graph_builder.data.x.shape[1],
                "hidden_dim": self.params.get("hidden_dim", 64),
                "heads": self.params.get("heads", 4),
                "out_dim": self.params.get("out_dim", 32),
            },
            "calibrator": self._calibrator,
            "params": self.params,
            "trainer_class": self.__class__.__name__,
            "best_threshold": self.best_threshold_,
        }
        content = pickle.dumps(payload)
        path.write_bytes(content)
        sha256 = hashlib.sha256(content).hexdigest()
        path.with_suffix(path.suffix + ".sha256").write_text(f"{sha256}  {path.name}\n")

        # 그래프 아티팩트는 같은 디렉터리에 저장
        self._graph_builder.save(path.parent)
        logger.info("GATTrainer 저장: %s (sha256=%s…)", path, sha256[:16])
        return path

    @classmethod
    def load_gat(cls, path: str | Path) -> "GATTrainer":
        """gat_model.pt + gat_graph.pt sha256 검증 후 로드."""
        path = Path(path)
        if not path.exists():
            raise RuntimeError(f"gat_model.pt 없음: {path}")
        # model sha256 검증
        sha_path = path.with_suffix(path.suffix + ".sha256")
        if not sha_path.exists():
            raise RuntimeError(f"sha256 없음: {sha_path}")
        content = path.read_bytes()
        expected = sha_path.read_text().strip().split()[0]
        actual = hashlib.sha256(content).hexdigest()
        if actual != expected:
            raise RuntimeError(f"gat_model.pt sha256 불일치: {path}")

        payload = pickle.loads(content)

        from .gat_model import GATModel
        init_p = payload["model_init_params"]
        model = GATModel(**init_p)
        model.load_state_dict(payload["model_state"])
        model.eval()

        # 그래프 아티팩트 검증 로드 (RuntimeError if sha256 mismatch)
        graph_builder = GraphBuilder.load(path.parent)

        obj = cls.__new__(cls)
        obj.params = payload["params"]
        obj.config = None
        obj._gat_model = model
        obj._graph_builder = graph_builder
        obj._calibrator = payload.get("calibrator")
        obj.best_threshold_ = payload.get("best_threshold", 0.5)
        obj._trained = True
        obj.model_dir = path.parent
        obj.feature_importances_ = None
        logger.info("GATTrainer 로드: %s", path)
        return obj
