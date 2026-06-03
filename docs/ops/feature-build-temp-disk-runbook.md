# 피처 빌드 임시 디스크 운영 가이드 (HANA_FEAT_TMP)

작성: 2026-06-02 · 관련: `mode_11_error.txt` 디스크풀 IOException RCA
(`docs/reports/2026-06-02_ml_dl_and_diskfull_review.md`).

## 증상

데스크톱 학습(Page 3) 또는 `build_patient_features_from_parquet` 실행 중:

```
IOException: Could not write file ".../Temp/hana_feat_xxx/_part=NN/data_0.parquet"
(디스크 공간이 부족합니다)
```

## 원인

DuckDB `COPY (SELECT * ...) TO ... (PARTITION_BY (_part))` 가 전체 raw Parquet 를
임시 디렉터리에 **통째 복제**한 뒤 파티션별로 피처를 계산한다. 피크 임시 사용량은
소스 총 크기의 **약 2배**(버퍼 + 파티션 동시 기록). 임시 경로가 시스템 드라이브
(`C:\Users\...\AppData\Local\Temp`)로 떨어지고 그 드라이브 여유가 부족하면 복제
도중 디스크가 차서 실패한다.

- 참고: 6개월 raw(`data/Raw/records_2024*.parquet` 184개) 총 ~1GB → 피크 임시 ~2GB.
  코호트가 커지면 비례 증가.
- pandas 폴백 경로(DuckDB 미설치 시)는 파일 단위 스트리밍이라 임시 풋프린트가 작다.

## 조치 — 임시 디렉터리를 넉넉한 드라이브로 지정

여유 **10GB 이상** 드라이브를 `HANA_FEAT_TMP` 환경변수로 지정한다.

```bat
set HANA_FEAT_TMP=D:\hana_tmp
```

경로 결정 우선순위 (`ml_runner._resolve_feat_tmp_base`):

1. `HANA_FEAT_TMP`
2. `HANA_TMP_DIR`
3. `hana_config.json` 의 `hana_feat_tmp` / `hana_tmp_dir` (또는 `training.*`)
4. Python `tempfile.gettempdir()` (시스템 temp)

## 사전 점검 동작 (`_preflight_temp_space`)

시작 전 `필요량 ≈ 소스 × headroom(10) + 512MB` 와 대상 드라이브 가용량을 비교한다.
부족하면 복제를 시작하지 않고 친절한 `InsufficientDiskSpaceError` 를 던지며 가용
대안 드라이브 목록을 안내한다. headroom=10 은 공유 드라이브·느린 디스크·단편화
여유를 포함한 보수값(조기·친절 실패 우선).

> **배포 주의**: 이 사전점검 로직이 빠진 구버전 배포본에서는 raw DuckDB IOException
> 이 그대로 노출된다. 운영 배포본이 `_preflight_temp_space` 를 포함하는지 확인할 것.

## 권장: BAT 안내 추가 (CRLF 보존 필수)

`install_312.bat` / `hana_app/run.bat` 는 **CRLF + `chcp 65001`** 로 저장돼야 한다
(LF 저장 시 한글 깨짐). 아래 안내문을 패키지 설치/실행 전에 추가하는 것을 권장한다.
편집 시 반드시 CRLF 를 유지할 것(antigravity-worker 또는 CRLF 보존 에디터 사용).

```bat
echo.
echo [보조] 대용량 학습용 임시 디렉터리(선택)
echo   파일이 많으면 여유 10GB+ 드라이브를 지정하세요. 예: set HANA_FEAT_TMP=D:\hana_tmp
echo.
```
