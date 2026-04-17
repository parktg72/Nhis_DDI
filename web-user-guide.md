# NHIS 다재약물 DDI 위험도 분류 시스템 — 웹 앱 사용 가이드

---

## 목차
1. [사전 준비](#1-사전-준비)
2. [패키지 설치](#2-패키지-설치)
3. [웹 앱 실행](#3-웹-앱-실행)
4. [1단계 — 연결 및 테이블 설정](#4-1단계--연결-및-테이블-설정)
5. [2단계 — 데이터 미리보기](#5-2단계--데이터-미리보기)
6. [3단계 — 모델 학습](#6-3단계--모델-학습)
7. [4단계 — 결과 분석](#7-4단계--결과-분석)
8. [문제 해결](#8-문제-해결)

---

## 1. 사전 준비

### 시스템 요구사항

| 항목 | 최소 사양 |
|------|-----------|
| OS | Windows 10 이상 (64비트) **또는** macOS 12 이상 (Apple Silicon 포함) |
| Python | 3.9 ~ 3.12 |
| RAM | 8 GB 이상 (SAS 대용량 파일 처리 시 16 GB 권장) |
| 디스크 | 5 GB 이상 여유 공간 |
| 네트워크 | 건보 내부망 (HANA DB 사용 시) |

### 폴더 구조 확인

```
MODE_11_hana/
├── hana_app/          ← 웹 앱 본체
│   ├── app.py
│   ├── run.bat        ← 실행 스크립트 (Windows)
│   ├── run.sh         ← 실행 스크립트 (Mac/Linux)
│   ├── pages/
│   │   ├── 1_🔌_연결_및_테이블설정.py
│   │   ├── 2_🔍_데이터_미리보기.py
│   │   ├── 3_🤖_모델_학습.py
│   │   └── 4_📊_결과_분석.py
│   └── core/
├── hana/              ← HANA/ML 패키지
├── packages_win/      ← 공통 패키지 (Windows)
├── packages_mac/      ← 공통 패키지 (Mac/Linux)
├── hira/              ← 심평원 약제급여목록 (xlsx)
├── data/              ← DDI 참조 데이터 (parquet)
├── install_all.bat    ← 통합 설치 스크립트 (Windows)
├── install_all.sh     ← 통합 설치 스크립트 (Mac/Linux)
└── download_all.bat   ← 통합 다운로드 스크립트
```

---

## 2. 패키지 설치

> 건보 폐쇄망은 인터넷 연결이 없으므로 **인터넷 환경에서 먼저 다운로드** 후 폐쇄망 PC에 옮겨 설치합니다.

### 2-1. 인터넷 환경에서 패키지 다운로드 (외부 PC)

**Windows:**
```bat
REM 프로젝트 루트 폴더에서 실행
download_all.bat
```

**Mac:**
```bash
# 프로젝트 루트 폴더에서 실행
bash packages_mac/download.sh
```

Python 버전별(3.9~3.12)로 `packages_win\py3X\` / `packages_mac/py3X/`, `hana\py3X\` 폴더에 저장됩니다.

### 2-2. 폐쇄망 PC에서 패키지 설치

**Windows:**

Python 버전 확인:
```bat
python --version
```

설치 (버전 예: 3.11, 가상환경 생성):
```bat
install_all.bat 311 venv
```

| 인수 | 설명 | 예시 |
|------|------|------|
| 첫 번째 | Python 버전 (39/310/311/312) | `311` |
| 두 번째 | 가상환경 사용 (`venv` 고정) | `venv` |

**Mac:**

Python 버전 확인:
```bash
python3 --version
```

설치 (버전 예: 3.11, 가상환경 생성):
```bash
bash install_all.sh --py 311 --venv
```

| 옵션 | 설명 | 예시 |
|------|------|------|
| `--py` | Python 버전 (39/310/311/312) | `--py 311` |
| `--venv` | 가상환경 자동 생성 | (플래그만) |

---

## 3. 웹 앱 실행

### Windows

```bat
REM 기본 실행 (포트 8501)
hana_app\run.bat

REM 포트 지정
hana_app\run.bat 8080

REM 가상환경 사용
hana_app\run.bat 8501 venv
```

### Mac

```bash
# 기본 실행 (포트 8501)
bash hana_app/run.sh

# 포트 지정
bash hana_app/run.sh 8080

# 가상환경 사용
bash hana_app/run.sh 8501 venv
```

실행 후 브라우저에서 접속:
```
http://localhost:8501
```

> **종료**: 실행 중인 터미널에서 `Ctrl + C`

---

## 4. 1단계 — 연결 및 테이블 설정

메인 화면 왼쪽 사이드바에서 **1️⃣ 연결 및 테이블 설정** 클릭

### 데이터 소스 선택

화면 상단 라디오 버튼에서 데이터 소스를 선택합니다.

```
○ 🗄️ SAP HANA DB   ← 건보 내부망 HANA DB 직접 연결
● 📂 SAS 파일       ← HANA DB 미연결 시 SAS 파일 폴더 사용
```

---

### HANA DB 모드

**HANA DB 연결** 탭에서 접속 정보 입력:

| 항목 | 예시 | 설명 |
|------|------|------|
| Host (IP) | `192.168.1.100` | HANA DB 서버 IP |
| Port | `30015` | 기본 포트 |
| 사용자 ID | `NHIS_USER` | DB 계정 |
| 비밀번호 | `********` | 입력 후 저장 시 암호화 |

→ **🔌 연결 테스트** 버튼 클릭 → 연결 성공 확인

**테이블 위치(HANA)** 탭에서 각 테이블 스키마·테이블명 확인:

| 키 | 기본값 |
|----|--------|
| T20 (명세서) | `NHISBDA.HHDT_TEMSBJ20` |
| T30 (원내약품) | `NHISBDA.HHDT_TEMSBJ30` |
| T40 (상병) | `NHISBDA.HHDT_TEMSBJ40` |
| T60 (원외처방) | `NHISBDA.HHDT_TEMSBJ60` |
| 요양기관 | `NHISBASE.HHDT_MDCIN_GNRL_INFO` |

> 정책에 따라 스키마·테이블명이 다를 경우 직접 수정 후 **💾 설정 저장**

---

### SAS 파일 모드

**SAS 파일 설정** 탭:

1. **SAS 파일 폴더 경로** 입력 (예: `D:\nhis_data\2023`)
2. **인코딩** 선택 (국민건강보험공단 SAS 파일 기본값: `cp949`)
3. **🔍 폴더 스캔** 버튼 → 폴더 내 `.sas7bdat` 파일 자동 인식
4. 각 테이블(T20/T30/T40/T60/요양기관)에 해당 파일 선택

```
SAS 파일명 예시:
  T20 → TEMSBJ20_202301.sas7bdat
  T30 → TEMSBJ30_202301.sas7bdat
  T40 → TEMSBJ40_202301.sas7bdat
  T60 → TEMSBJ60_202301.sas7bdat
  요양기관 → MDCIN_GNRL_INFO.sas7bdat
```

5. **💾 SAS 설정 저장**

---

### 컬럼 매핑 탭

컬럼명이 기본값과 다를 경우 조정합니다.

| 역할 | 기본 컬럼명 |
|------|------------|
| 환자 ID | `INDI_DSCM_NO` |
| 청구 키 | `CMN_KEY` |
| 주성분코드 | `WK_COMPN_CD` |
| 요양개시일 | `MDCARE_STRT_DT` |
| 투여일수 | `TOT_MCNT` |

---

## 5. 2단계 — 데이터 미리보기

왼쪽 사이드바에서 **2️⃣ 데이터 미리보기** 클릭

### 테이블 선택

드롭다운에서 확인할 테이블을 선택합니다.

```
T20 – 요양급여비용명세서
T30 – 진료내역 (원내 약품)
T40 – 상병내역
T60 – 원외처방전 내역
요양기관 현황
```

### 탭 구성

| 탭 | 기능 |
|----|------|
| 📄 샘플 데이터 | 행 수 지정 + YYYYMM 필터로 데이터 미리보기 |
| 📋 컬럼 정보 | 전체 컬럼 목록 + 매핑 상태 확인 |
| 📊 분포 분석 | 선택 컬럼의 NULL 비율·고유값 수·히스토그램 |

### YYYYMM 필터 사용법 (SAS 모드)

```
YYYYMM 필터 입력: 202301
→ MDCARE_STRT_YYYYMM이 2023년 1월인 행만 조회
```

---

## 6. 3단계 — 모델 학습

왼쪽 사이드바에서 **3️⃣ 모델 선택 및 학습** 클릭

### 1️⃣ 데이터 추출 범위

| 항목 | 설명 | 기본값 |
|------|------|--------|
| 시작 년도/월 | 학습 데이터 시작 기간 | 2023년 1월 |
| 종료 년도/월 | 학습 데이터 종료 기간 | 2023년 12월 |
| 동시복용 판단 기간 | 이 일수 내 처방을 동시복용으로 판정 | 90일 |
| 다재약물 기준 | 이 종수 이상 복용 환자만 분석 | 5종 |

### 2️⃣ 모델 선택

| 알고리즘 | 특징 |
|----------|------|
| XGBoost (권장) | 정확도 높음, 속도 보통 |
| LightGBM (빠름) | 대용량에 적합 |
| Random Forest | 해석 용이 |
| Logistic Regression | 기준선 비교용 |

**예측 타겟:**
- `이진 분류 (위험/정상)`: 위험군 탐지에 집중, **권장**
- `4분류 (Red/Yellow/Green/Normal)`: 위험 단계 세분화

### 3️⃣ 피처 선택

필요한 피처만 선택하여 모델 복잡도 조정:

| 피처 | 설명 |
|------|------|
| drug_count | 고유 약물 수 (복합제 성분 전개 포함) |
| ddi_contraindicated | 금기 DDI 쌍 수 |
| ddi_major | Major DDI 쌍 수 |
| triple_whammy | Triple Whammy 여부 |
| qt_risk_count | QT 연장 위험 약물 수 |
| dup_same_ingredient | 동일 성분 중복 수 |
| age / sex_m | 연령 / 성별 |

### 학습 실행

1. **💾 설정 저장** → 현재 설정을 파일에 저장
2. **🚀 학습 시작** → 데이터 추출 → 피처 계산 → 모델 학습 순으로 진행

```
진행 단계:
  📥 데이터 추출 중... (HANA 또는 SAS 파일)
  ⚙️ 피처 계산 중...   (DDI/중복/환자 특성)
  🤖 모델 학습 중...   (선택한 알고리즘)
  📊 결과 요약         (Accuracy / F1 / AUC)
```

> 학습 완료 후 결과는 `hana_app/results/` 폴더에 자동 저장됩니다.

---

## 7. 4단계 — 결과 분석

왼쪽 사이드바에서 **4️⃣ 결과 분석** 클릭

### 제공 분석 화면

| 섹션 | 내용 |
|------|------|
| 📊 피처 중요도 | 각 피처의 기여도 막대 그래프 |
| 🔲 혼동 행렬 | 예측 vs 실제 분류 히트맵 |
| 📈 교차검증 결과 | K-fold별 성능 지표 |
| 🎯 위험도 분포 | Red/Yellow/Green/Normal 환자 수 파이차트 |
| 📋 분류 보고서 | Precision / Recall / F1 전체 표 |
| 🏆 모델 비교 | 이전 학습 결과와 성능 비교 |

### 위험도 등급 기준

| 등급 | 기준 | 개입 수준 |
|------|------|-----------|
| 🔴 Red (고위험) | 금기 DDI ≥1건, Major DDI ≥3건, Triple Whammy, 75세↑+5종↑ | 즉각 개입 |
| 🟡 Yellow (중위험) | Major DDI 1~2건, Moderate DDI ≥2건, 동일성분 중복, 3기관↑ | 월 1회 모니터링 |
| 🟢 Green (저위험) | Minor DDI만, DDI 없이 5종↑ | 분기 1회 안내 |
| ⚪ Normal (정상) | 해당 없음 | 대상 외 |

---

## 8. 문제 해결

### 웹 앱이 실행되지 않을 때

```bat
REM streamlit 설치 확인
python -m streamlit --version

REM 설치되지 않은 경우
install_all.bat 311 venv
```

### HANA DB 연결 오류

| 증상 | 해결 방법 |
|------|-----------|
| `hdbcli` 없음 | `hana\install.bat 311 venv` 실행 |
| 연결 시간 초과 | IP·포트 확인, 건보 내부망 접속 여부 확인 |
| 인증 실패 | 계정 잠금 여부 확인, 비밀번호 재입력 |

### SAS 파일 읽기 오류

| 증상 | 해결 방법 |
|------|-----------|
| `pyreadstat` 없음 | `install_all.bat 311 venv` 재실행 |
| 인코딩 오류 | 인코딩을 `cp949` → `euc-kr` 으로 변경 |
| 파일 없음 | 폴더 경로 재확인, 파일명 오탈자 확인 |
| 메모리 오류 | Chunksize를 `100,000` → `50,000`으로 줄임 |

### 학습 오류

| 증상 | 해결 방법 |
|------|-----------|
| "다재약물 기준 충족 환자 없음" | 추출 기간 확장 또는 약물 기준 수 낮춤 (5→3) |
| 피처 선택 오류 | 최소 1개 이상 피처 선택 |
| 메모리 부족 | 추출 기간 단축 (1개월씩 나눠서 처리) |

### 포트 충돌

```bat
REM 다른 포트로 실행
hana_app\run.bat 8502
```

### 설정 초기화

설정 파일 삭제 후 재실행하면 기본값으로 초기화됩니다:
```bat
del hana_app\hana_config.json
```

---

## 데스크탑 모드 (run_desktop.bat)

### 언제 사용하나

회사 인트라넷 정책이 **3시간 미사용 시 브라우저를 자동 종료**하여 웹앱 분석 세션이 끊기는 경우. 데스크탑 모드는 pywebview 임베디드 창에서 동작하므로 브라우저 자동 종료 대상이 아니다.

### 실행

1. 프로젝트 루트에서 `run_desktop.bat` 더블클릭
2. 자동으로 `.venv_hana` 감지 → Streamlit 서버 8502 포트 기동 → pywebview 창 표시
3. 웹 모드(`hana_app\run.bat`, 포트 8501)와 **동시 실행 가능** (포트가 다르므로 충돌 없음)

### 종료

창 X 버튼 클릭 → Streamlit 서브프로세스 자동 종료 (terminate + wait 5s + kill).

### 로그 위치

`%LOCALAPPDATA%\hana_desktop\logs\desktop.log`

예: `C:\Users\<사용자>\AppData\Local\hana_desktop\logs\desktop.log`

서버 기동 실패 시 이 로그의 마지막 20줄이 콘솔에 자동 출력된다.

### exit code

| 코드 | 의미 |
| --- | --- |
| 0 | 정상 종료 |
| 1 | Streamlit 서버 기동 실패 (로그 tail 확인) |
| 2 | pywebview 미설치 — `install_312.bat venv` 재실행 필요 |
| 3 | 8502 포트에 Streamlit 이외의 프로세스 점유 중 |

### 8502 포트 점유 시

- **내 Streamlit 이 이미 떠 있는 경우**: `/_stcore/health` 헬스체크 통과 → 기존 서버 재사용 (자동).
- **다른 프로세스가 점유**: exit 3. `netstat -ano | findstr :8502` 로 PID 확인 후 해당 프로세스 종료.

### 설치 / 재설치

- 최초 설치 또는 Python 재설치 후: `install_312.bat venv` 한 번 실행 (모든 의존성 일괄 설치).
- pywebview 만 별도: `install_pywebview.bat` (레거시/진단용).

### WebView2 Runtime

pywebview 는 Windows 에서 **Edge WebView2 Runtime** 을 필요로 한다. `install_312.bat venv` 마지막 검증 단계에서 `%ProgramFiles(x86)%\Microsoft\EdgeWebView\Application` 존재를 확인하여 없을 경우 경고를 표시. 사내 표준 이미지에 포함되어 있지 않다면 설치 담당자에게 문의.

---

## 부록 — 빠른 시작 체크리스트

```
□ 1. Python 3.9~3.12 설치 확인
□ 2. install_all.bat 311 venv  실행 완료
□ 3. hana_app\run.bat 8501 venv  실행
□ 4. 브라우저에서 http://localhost:8501 접속
□ 5. 1단계: 데이터 소스 선택 (HANA 또는 SAS)
□ 6. 1단계: 연결/파일 설정 및 저장
□ 7. 2단계: 샘플 데이터 확인
□ 8. 3단계: 학습 범위·모델 설정 → 학습 시작
□ 9. 4단계: 결과 분석 확인
```
