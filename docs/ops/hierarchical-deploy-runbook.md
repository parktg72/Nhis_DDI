# 계층 모델 배포 런북 (retrain_prod_0711_hierarchy_cur)

개입 위계 재설계(2026-06-07) 반영 번들 배포 절차. 배포 = `HIERARCHICAL_MODEL_DIR` 환경변수
지정 후 serving 기동/리로드 (코드·설정 하드코딩 없음, env 전용).

## 배포 번들
```
hana_app/models/hierarchical/retrain_prod_0711_hierarchy_cur
```
- 453,677 환자, 7-class, feature_semantics_version=`rulefeat.v1`, stage1_trained=True.
- triple_whammy 키워드 큐레이션(ACEi/ARB 염형태·finerenone·morniflumate, 2026-06-07) batch 반영.
- 개입 분포: 즉각개입(Red) 0.37%(1,660) / 약사전화(Y_DDI_MAJOR) 16.7%(75,979) /
  문자안내(Y_TRIPLE) 46.3%(210,008) / 모니터링(Y_DOUBLE·단일) ~36.6% / 관여안함(No_Alert).

## 전환 절차
1. 환경변수 지정 (운영 PC):
   ```
   set HIERARCHICAL_MODEL_DIR=C:\model\MODE_11_hana\hana_app\models\hierarchical\retrain_prod_0711_hierarchy_cur
   ```
2. serving 기동(재기동) 또는 무중단 리로드:
   - 재기동: serving 프로세스 재시작 (lifespan 이 env 읽어 init_predictor).
   - 무중단: `POST /admin/reload` (ADMIN_API_KEY) — HybridPredictor.reload_hierarchical.
3. feature_semantics_version=rulefeat.v1 번들이면 서빙이 자동으로 rule_features_active=True
   → triple_whammy/위험약물 플래그를 edi→wk→DrugMaster components 로 산출(학습과 정합).

## 검증 (배포 후)
- 계층 로드 확인: 로그 "HIERARCHICAL ... loaded" + `_hierarchical.feature_semantics_version == "rulefeat.v1"`.
- 샘플 예측 응답의 `action` 이 신 위계 집합에 속하는지:
  `{즉각 개입, 약사 전화, 문자 안내, 모니터링, 관여 안 함}`.
- 금기(contraindicated) 요청 → risk_level=Red, action=즉각 개입(결정적 백스톱 `_BACKSTOP_ACTIVE`).
- in-process 검증(2026-06-07 실측): init_predictor(HIERARCHICAL_MODEL_DIR=…_hierarchy) →
  계층 로드 True, fsv=rulefeat.v1, 응답 action ∈ 위계 → **배포 검증 통과**.

## 롤백
- 직전 번들로 `HIERARCHICAL_MODEL_DIR` 재지정 후 리로드. (구 번들 정리 전이라면 이전 deliberate
  번들 사용. 단 구 번들은 개입 위계 재설계 미반영이므로 ACTION_BY_LABEL 코드와 라벨 의미가
  어긋날 수 있음 — 코드 롤백 동반 권장.)

## 주의
- 구 번들(retrain_prod_0711/_rulefeat/_redesign)은 라벨/개입 정의가 현재 코드와 불일치 →
  배포 금지(정리 대상). 신 배포는 retrain_prod_0711_hierarchy_cur 만.
- 번들은 gitignored — 운영 PC 로 별도 전송 필요.
