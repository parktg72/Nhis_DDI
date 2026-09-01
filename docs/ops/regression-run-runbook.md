# 회귀 실행 런북

개선계획 A1(main 병합)의 완료 판정은 **회귀 전량 통과**다. 그런데 이 저장소를 WSL 에서
`/mnt/c` 의 Windows 파이썬으로 돌리는 구성에서는 **스위트를 한 번에 실행하면 멈춘다.**
느린 것이 아니라 교착이다. 이 문서는 그 회피와 판정 기준을 정한다.

## 증상

`tests/test_serving` 253개를 한 프로세스로 실행하면 pytest 가 CPU 0:00 상태로 정지하고
출력이 헤더 이후 늘지 않는다. 30분을 기다려도 진행되지 않는다.

교착 시점의 스택 덤프에 **확장 모듈 187개**가 적재돼 있었다 — numpy · pandas · pyarrow ·
scipy · sklearn · **torch**. 이 상태로 실 Raw parquet 를 읽으면서 WSL↔Windows 파일시스템
경계를 넘나드는 I/O 가 교착한다. 파일 단위로 나누면 각 프로세스가 필요한 것만 적재해 완주한다.

## 실행 방법

### 파일 단위 (권장)

```bash
for f in tests/test_serving/*.py; do
  timeout 100 ./.venv/Scripts/python.exe -m pytest "$f" -q -p no:cacheprovider
done
```

- `timeout 100` — 교착을 멈춤으로 드러낸다. 가장 느린 파일이 77초이므로 100초면 충분하다.
- `-p no:cacheprovider` — `.pytest_cache` 쓰기를 막아 교차 파일시스템 쓰기를 줄인다.
- 인터프리터는 저장소 venv 의 **Windows 실행 파일**이다. 시스템 파이썬에는 `pydantic` 이 없다.

### 소규모 확인

변경 범위가 좁으면 관련 파일만 지정한다. 이쪽은 한 프로세스로 안전하다.

```bash
./.venv/Scripts/python.exe -m pytest \
  tests/test_serving/test_rule_floor_reason_merge.py \
  tests/test_serving/test_severe_backstop.py -q -p no:cacheprovider
```

## 소요 시간 (2026-09-01 실측)

전체 28파일 중 실 parquet 를 읽는 넷이 시간을 지배한다.

| 파일 | 소요 |
|---|---|
| `test_lookup_wk_callers.py` | 77초 |
| `test_ddi_train_serve_parity.py` | 52초 |
| `test_count_dup_train_serve_parity.py` | 50초 |
| `test_hierarchical_serving.py` | 21초 |
| 나머지 24개 | 각 20초 미만 |

파일 단위 전체 실행은 약 5~6분이다.

## 판정 기준

### 기준선 (2026-09-01)

`tests/test_serving` — **235 passed · 3 failed · 멈춤 0**

기존 실패 3건은 lenient 모드 관련이며 **변경 이전부터 존재한다.**

- `test_feature_schema_strict.py::test_validate_unknown_passes_lenient`
- `test_feature_schema_strict.py::test_mlmodel_load_accepts_unknown_lenient`
- `test_health_schema_drift.py::test_lenient_env_active_but_no_drift_health_ok`

`docs/ops/lenient-sunset-degraded-checklist.md` 의 lenient 걷어내기 작업과 연동된 미완 상태로
보인다. 별도 사안이므로 **회귀 판정에서 기존 실패로 계상**한다.

### 새 실패가 나왔을 때

**기존 실패인지 변경 탓인지 먼저 가른다.** 추측하지 말 것.

```bash
git stash push -- <변경한 파일>
./.venv/Scripts/python.exe -m pytest <실패한 파일> -q -p no:cacheprovider
git stash pop
```

같은 실패가 재현되면 기존 실패다. 커밋 메시지에 그 사실과 대조 방법을 남긴다 — 나중에
"그때 깨져 있었는데 왜 커밋했나"를 묻는 사람이 근거를 찾을 수 있어야 한다.

### 완료 판정

- **멈춤 0** — 하나라도 timeout 이면 통과로 계상하지 않는다
- **새 실패 0** — 기존 실패 3건 외에 늘어난 것이 없어야 한다
- 실행 방법(파일 단위 여부)과 기준선 대비 증감을 함께 기록한다

## 다른 스위트

`tests/test_rules.py` · `tests/test_etl` · `tests/test_hana_app` 도 같은 방식을 적용한다.
`tests/test_hana_app` 는 2026-07-11 기준 312건으로 서빙보다 크므로 파일 단위가 사실상 필수다.

## 주의

- **`pkill -f "pytest ..."` 를 쓰지 말 것.** 패턴이 호출한 셸까지 잡는다. PID 로 지정한다.

  ```bash
  for p in $(ps -eo pid,args | awk '/python.exe.*-m pytest/ && !/awk/ {print $1}'); do
    kill -9 "$p"
  done
  ```

- 교착한 프로세스는 스스로 끝나지 않는다. 다음 실행 전에 반드시 정리한다.
- 이 문서의 수치는 WSL + `/mnt/c` 구성 기준이다. 폐쇄망 운영 PC(네이티브 Windows)에서는
  경계를 넘지 않으므로 일괄 실행이 될 수도 있다. **확인되면 이 문서를 갱신할 것.**
