# Park 결정 사항 — 2026-05-07

**작성**: 2026-05-07 (cross-family 라운드 후속)
**근거**: park 직접 답변. 본 라운드의 두 park-dependent blocker 해소.

본 보고서는 **사실 trail 만** 포함. 결정 사항을 미래에 다시 추적할 때 참조.

---

## 1. Q1 — 배포 reality

**질문**: 실제 배포 대상이 (a) Linux 컨테이너 vs (b) Windows 폐쇄망 only?

**park 답**: **(b) Windows 폐쇄망 only**.

### 결정 사항
- Dockerfile 은 stale artifact 로 명시적 deprecated 처리 (`Dockerfile` 헤더 참조)
- P0 풀 재빌드 작업 **불필요**
- CLAUDE.md 의 "운영 PC = Windows 폐쇄망 + Python 3.12" 와 정합

### 처리 trail
- `Dockerfile` 첫 머리에 `DEPRECATED — 2026-05-07` 헤더 + 사실 인용 + 부활 시 필요 작업 명시 (본 PR commit)
- 부활 trigger 명확화: Linux 컨테이너 부활 결정 시 본 docs + Dockerfile 헤더 starting point

### 부활 시 필요 작업 (보존)
- `python:3.11-slim` → `python:3.12-slim` 전환
- `packages_linux/py311/` → `packages_linux/py312/` wheelhouse 구축
- `monitoring/` 복사 추가 (현 `serving/main.py:30` 의 import 누락 해결)
- 누락 deps 추가: `joblib`, `scikit-learn`, `torch`, `filelock` 등 (실제 serving 경로 의존성 grep 후 결정)
- `constraints-py312.txt` 기반 install (현 lock 일관)
- Dockerfile 헤더 DEPRECATED 마크 제거

---

## 2. Q2 — dup_efmdc importance artifact

**질문**: prod 학습 모델 artifact 경로 또는 직접 측정한 `dup_efmdc` feature importance 값?

**park 답**: **아직 완전 학습되지 않음** (artifact 부재, importance 측정 불가).

### 결정 사항
- 현재 `serving/predictor.py:_INTENTIONAL_FEATURE_ALLOWLIST = frozenset({"dup_efmdc"})` provisional 분류 **유지**
- serving 측 `predictor.py:650` 의 `feat["dup_efmdc"] = 0.0` 고정 그대로
- **학습 완료 시점에 importance 측정** → allowlist 유지/제거 결정

### 측정 trigger (학습 완료 시)
prod 모델 또는 최근 학습된 `.pkl`/`.joblib` artifact 입수 시:

```python
import joblib
m = joblib.load("model_prod.pkl")
fi = dict(zip(m["feature_names"], m["model"].feature_importances_))
print("dup_efmdc:", fi.get("dup_efmdc"))
print("top 10:", sorted(fi.items(), key=lambda x: -x[1])[:10])
```

### 결정 분기
- **importance ≥ 1% (상대)** → P1 promote:
  - `_INTENTIONAL_FEATURE_ALLOWLIST` 에서 `dup_efmdc` 제거
  - serving 측 `predictor.py:650` 의 0.0 고정 제거
  - DrugMaster 로드 또는 별도 lookup 추가해 실제 산출
  - 회귀 가드 추가
  - 본 docs 에 importance 측정 결과 + 결정 trail 박음
- **importance ≈ 0** → 현행 유지:
  - allowlist 그대로
  - 본 docs 에 importance 측정 결과 + 유지 결정 trail 박음

### dup_efmdc allowlist sunset 메커니즘 (별도 design)
- `FEATURE_SCHEMA_LENIENT_SUNSET_DATE` 와 같은 패턴 후보
- **단** importance 결과로 단정 결정 후 sunset 불필요 가능
- 본 결정은 importance 답 받은 뒤

---

## 3. 본 라운드 후속 진행 plan

park 답으로 두 blocker 해소된 후 다음 진행 가능 작업:

| 작업 | 가능 여부 | 비고 |
|---|---|---|
| Dockerfile deprecated 처리 | ✅ 본 PR 완료 | 헤더 마크 + 본 docs |
| dup_efmdc importance 측정 | ❌ 학습 완료 후 | trigger: prod 모델 artifact |
| dup_efmdc allowlist 결정 | ❌ importance 결과 후 | trigger: 측정 결과 |
| dup_efmdc allowlist sunset | ❌ allowlist 결정 후 | importance 결과 따라 불필요 가능 |
| 그 외 production code 변경 | ✅ park 의견 따라 | cross-family 합의 다음 작업 가능 |

---

## 4. References

### 본 라운드 commit (2026-05-07)
- `57a128e` 단일 ML schema strict — `_validate_feature_schema`, `_INTENTIONAL_FEATURE_ALLOWLIST` 도입
- `b234832` LENIENT escape hatch sunset
- `0140904` /health lenient_allowed + sunset_date visibility

### 본 라운드 docs
- `docs/ops/lenient-sunset-degraded-checklist.md` — 운영자 매뉴얼
- `docs/reports/2026-05-07-cross-family-round.md` — engineering trail
- 본 docs

### 메모리 (디시플린)
- `~/.claude/projects/.../memory/feedback_train_serving_parity_prefix.md`
- `~/.claude/projects/.../memory/feedback_primary_source_overrides_ai.md`
- `~/.claude/projects/.../memory/feedback_native_lib_test_isolation.md`
