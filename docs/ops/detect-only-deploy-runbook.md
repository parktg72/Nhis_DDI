# 탐지 전용 배포 런북

이름 기반 규칙(Top-10 · QT · 고위험약)을 **켜되 즉각 개입 대상은 늘리지 않는** 배포 절차.

작성 2026-09-02 · 대상 판: A1·RS1~RS3·A1a 병합 후 (`a0_baseline_check` 가 "현판" 으로 판정하는 코드)

---

## 0. 무엇을 켜는가

| 환경변수 | 기본 | 이 배포에서 |
|---|---|---|
| `SERVING_ENABLE_EDI_NAME_RESOLUTION` | 미설정(꺼짐) | **`1`** |
| `SERVING_RULE_DETECT_ONLY` | 미설정(꺼짐) | **`1`** |
| `SERVING_RISK_FLAG_ATC_CANDIDATES` | 미설정(꺼짐) | **건드리지 않는다** |

**두 개를 반드시 함께 켠다.** 이름 해소만 켜면 즉각 개입 대상이 늘어난다(실측 300명 중 Red 68명). 탐지 전용만 켜면 해소가 없어 발화 자체가 없으므로 아무 일도 일어나지 않는다.

값은 `1`/`true`/`yes`/`on` 중 아무거나(대소문자 무관). 그 밖의 값·빈 문자열은 꺼짐이다. **호출 시점에 읽으므로 프로세스 재기동이 필요하다.**

---

## 1. 배포 전 확인

```bash
python3 scripts/ops/a0_baseline_check.py --api http://<서빙주소>/health
```

읽기 전용·표준 라이브러리만 쓴다(폐쇄망에서 venv 불요). 네 항목을 본다.

| 항목 | 통과 조건 |
|---|---|
| ① 서빙 소스 | **판정이 "현판"** — 구판이면 재배포가 선행이다 |
| ② 서빙 플래그 | `serving_flags` 키가 있고 셋 다 `false` |
| ③ 배포 번들 | 모델 파일 SHA-256 이 번들 메타와 일치 |
| ④ 개입 전달 경로 | 아래 2절 참조 |

②에서 `serving_flags` 키 자체가 없으면 **구코드가 돌고 있는 것**이다. 이 경우 플래그를 설정해도 아무 효과가 없다.

---

## 2. 개입이 실제로 어떻게 전달되는지 알고 시작한다

현재 저장소에 **SMS·메일·메시징 발송 구현은 없다.** `action` 값은 API 응답과 오프라인 산출물(CSV·DOCX)로만 사람에게 도달한다. a0 ④가 이것을 확인한다.

따라서 이 배포에서 "개입이 늘지 않는다"는 것은 **API 응답의 `action` 필드가 변하지 않는다**는 뜻이다. 응답을 소비하는 하류 시스템이 별도로 있다면 그쪽 동작을 함께 확인해야 한다.

---

## 3. 전환 절차

1. **배포 전 상태 기록** — a0 리포트를 파일로 남긴다.
   ```bash
   python3 scripts/ops/a0_baseline_check.py --api http://<서빙주소>/health > a0_before.txt
   ```
2. **환경변수 설정** — 서비스 기동 스크립트/유닛 파일에 추가한다. 셸에서만 export 하면 재기동 시 사라진다.
   ```
   SERVING_ENABLE_EDI_NAME_RESOLUTION=1
   SERVING_RULE_DETECT_ONLY=1
   ```
3. **프로세스 재기동.**
4. **기동 로그 확인** — 켜진 플래그가 경고로 찍힌다. 두 값이 모두 보여야 한다.
5. **`/health` 확인** — 아래 4절.

---

## 4. 배포 후 검증

### ① 플래그가 실제로 켜졌는가

```bash
curl -s http://<서빙주소>/health | python3 -m json.tool | grep -A4 serving_flags
```

기대(키 순서는 구현 순서를 따른다):
```json
"serving_flags": {
    "SERVING_ENABLE_EDI_NAME_RESOLUTION": true,
    "SERVING_RISK_FLAG_ATC_CANDIDATES": false,
    "SERVING_RULE_DETECT_ONLY": true
}
```

`SERVING_RULE_DETECT_ONLY` 가 `false` 인데 해소가 `true` 이면 **즉시 되돌린다**(6절). 그 상태는 즉각 개입 대상이 늘어난 상태다.

### ② 탐지가 실제로 켜졌는가

EDI 만 담은 요청 하나로 확인한다. 아래는 항응고제 + NSAID(TOP01) 조합이다.

```bash
curl -s -X POST http://<서빙주소>/predict -H 'Content-Type: application/json' -d '{
  "patient_id":"SMOKE-1","patient_age":72,
  "drugs":[{"edi_code":"645600390","total_days":14,"start_date":"2024-07-01"},
           {"edi_code":"054801360","total_days":14,"start_date":"2024-07-01"}]}'
```

기대 응답(2026-09-02 실측):

```
risk_level   Normal          ← 개입 등급은 올라가지 않는다
rule_level   Yellow          ← 규칙층은 탐지했다
risk_reasons ["TOP01: Anticoagulant_NSAIDs"]
action       null            ← 개입 지시 없음
```

세 가지를 함께 본다. `risk_reasons` 에 `TOP01` 이 있고, `rule_level` 이 올라가 있고, **`risk_level` 은 올라가지 않는다.** 셋 중 하나라도 어긋나면 플래그 조합이 의도와 다르다.

`risk_reasons` 가 비어 있으면 이름 해소가 안 된 것이다 — 참조DB(`data/processed/edi_to_wk.parquet`, 약물명 인덱스, DDI 매트릭스)가 배포본에 있는지 확인한다. `risk_level` 이 올라갔다면 탐지 전용이 꺼진 것이다(6절).

### ③ 개입 산출물이 변하지 않았는가

배포 전후 같은 요청 표본(가능하면 하루치 실청구 상위 N명)에 대해 **`action` 과 `yellow_subtype` 의 분포가 같아야 한다.** 이것이 이 배포의 핵심 주장이다.

실측 기준(하루치 상위 300명, 2026-09-02):

| | 최종 `risk_level` | Top-10 탐지 | `action` |
|---|---|---|---|
| 배포 전 (둘 다 꺼짐) | GREEN 205 / YELLOW 95 | 0 | 무조치 205 · 문자 안내 95 |
| **배포 후 (탐지 전용)** | NORMAL 205 / YELLOW 95 | **42** | **무조치 205 · 문자 안내 95** |

`risk_level` 이 GREEN → NORMAL 로 바뀌는 것은 정상이다. 개입 지시는 `action` 이 나르므로 업무는 달라지지 않는다. **`action` 분포가 달라졌다면 되돌린다.**

---

## 5. 관측 — A5 측정으로 가는 입력

이 배포의 목적은 **운영 트래픽에서 Top-10 이 실제로 얼마나 발화하는지 재는 것**(A5)이다. 개입을 늘리지 않고 그 수치를 얻는 것이 탐지 전용의 존재 이유다.

### 수집 — 서빙이 이미 기록한다

서빙은 `/predict`·`/predict/batch` 요청마다 메트릭 JSONL 에 한 줄을 남긴다(경로: `DDI_METRICS_JSONL_PATH`, 기본 `/app/data/monitoring/metrics_live.jsonl`). **별도 수집 장치를 붙일 필요가 없다.**

| 필드 | 쓰임 |
|---|---|
| `risk_level` | 실제 개입 등급 — 배포 전과 같아야 한다 |
| `rule_level` | 규칙층이 무엇을 탐지했는지. 탐지 전용에서도 그대로 실린다 |
| `rule_ids` | 발화한 규칙 ID 목록 (`TOP01`~`TOP10`, `SEV_*`, `GRADE_*`) |
| `n_reasons` | 사유 개수 |
| `partition` | 일자 — 기간 필터에 쓴다 |

`rule_ids`·`n_reasons` 는 **2026-09-02 에 추가됐다.** 그 전 레코드에는 없으므로 발화 집계에 쓸 수 없다. 집계 도구가 두 형식을 구분해 보고하며, 구형식만 있으면 발화 0 이 아니라 **집계 불가**로 끝낸다.

기록에는 **규칙 ID 만** 들어간다. 설명 문구와 약물명은 넣지 않는다 — 이 파일은 환자 단위로 누적되므로 필요한 최소치만 남긴다.

### 집계

```bash
python3 scripts/ops/a5_firing_report.py --path <메트릭 JSONL 경로>
python3 scripts/ops/a5_firing_report.py --path <경로> --since 2026-09-05 --out a5.txt
```

읽기 전용·표준 라이브러리만 쓴다(폐쇄망에서 venv 불요). 런북의 세 항목을 그대로 산출한다.

| 산출 | 내용 |
|---|---|
| ① 규칙별 발화 환자 수 | Top-10 각각. **A5 의 본체** |
| ② 환자 단위 발화율 | Top-10 중 1개 이상 붙은 비율 |
| ③ 사유 없는 Red | **A4 활성의 차단 항목.** 0 이어야 해제된다 |

종료 코드: 정상 `0` · 집계 불가(파일 없음·레코드 0건·구형식만) `2`.

무발화 규칙이 있으면 이름을 대되 **원인을 단정하지 않는다** — 해소 결함인지 실제로 그 병용이 없는 것인지는 `scripts/ops/a3_remeasure.py` 로 코퍼스 상한을 먼저 확인해야 구분된다.

관측 기간은 최소 1주. 하루치 표본은 요일 편향이 있다.

---

## 6. 롤백

**환경변수 두 개를 제거하고 재기동한다.** 그것으로 끝이다 — 코드 되돌림·재배포·데이터 복구가 필요 없다.

```bash
unset SERVING_ENABLE_EDI_NAME_RESOLUTION SERVING_RULE_DETECT_ONLY   # 기동 설정에서도 제거할 것
# 재기동 후
curl -s http://<서빙주소>/health | grep -A4 serving_flags   # 셋 다 false 확인
```

롤백 판단 기준:

| 증상 | 조치 |
|---|---|
| `action` 분포가 배포 전과 다름 | 즉시 롤백 |
| `serving_flags` 에서 탐지 전용만 꺼짐 | 즉시 롤백 |
| 사유 없는 Red 발생 | 기록 후 A4 차단 항목에 반영. 탐지 전용에서는 개입이 늘지 않으므로 즉시 롤백 대상은 아니다 |
| 응답 지연 증가 | 이름 해소가 참조DB 조회를 늘린다. 측정 후 판단 |

---

## 7. 이 배포가 하지 않는 것

- **P0-1 을 닫지 않는다.** 닫는 것은 A5 의 운영 발화 관측이다. 이 배포는 그 측정을 가능하게 할 뿐이다
- **A4(활성 방식 결정)를 대체하지 않는다.** 전면 활성은 여전히 A4 결정과 차단 항목에 걸려 있다
- **개입 용량 문제를 해결하지 않는다.** 미룰 뿐이다. 탐지 결과를 보고도 개입하지 않는 기간이 길어지면 그 자체가 설명 대상이 된다

---

## 8. 주의

- 두 플래그를 **셸에서만** export 하면 재기동 시 사라진다. 기동 설정에 넣을 것
- `SERVING_RISK_FLAG_ATC_CANDIDATES` 는 건드리지 않는다. 이 배포의 범위 밖이며 별도 검증이 필요하다
- 플래그는 호출 시점에 읽히지만, 이미 뜬 프로세스의 환경은 바뀌지 않는다. **반드시 재기동**
- a0 ②에서 `serving_flags` 키가 없으면 구코드다. 플래그 설정은 무의미하다
