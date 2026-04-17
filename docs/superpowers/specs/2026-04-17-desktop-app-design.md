# 데스크탑 앱(pywebview) 복구 설계 — 2026-04-17

## 배경

폐쇄망 Windows 환경에서 Streamlit 기반 `hana_app` 을 인트라넷 브라우저로 실행(`hana_app\run.bat`)할 때, 회사 정책이 3시간 미사용 시 브라우저를 자동 종료한다. 브라우저가 닫히면 그 창으로 수행 중이던 분석 세션(`st.session_state`)이 끊겨 업무가 중단된다.

해결책: 브라우저가 아닌 **pywebview 임베디드 창**(Edge WebView2 기반)에서 같은 Streamlit 앱을 표시한다. 임베디드 창은 인트라넷 브라우저 자동 종료 정책의 대상이 아니라는 점이 **이전 버전(2026-03-25) 실사용으로 이미 검증**되었다. 본 설계는 이 검증된 구조를 **Python 3.12 전환 이후 회귀된 상태**에서 복구하고, 재회귀를 방지하는 것이 목적이다.

본 스펙은 사용자가 한 번에 요청한 **두 개의 관련 작업을 번들**로 다룬다: (1) 데스크탑 앱 복구(주 작업), (2) Claude Code 반복 루틴 `auto-approve` 설정(부수 작업). 두 번째는 런타임 복구와 직접 관련은 없지만 사용자 명시 요청이므로 범위에 포함한다.

## 범위

### In
- `run_desktop.bat` 더블클릭으로 pywebview 창에 Streamlit 앱이 뜨는 상태 복원
- `install_312.bat`에 pywebview + proxy_tools 오프라인 설치 통합 (재설치 시 재회귀 방지)
- 진단성 개선(에러 로그 보존, 사일런트 fallback 제거)
- Claude Code 반복 루틴의 `auto-approve` 설정(`.claude/settings.local.json`)

### Out (YAGNI)
- 트레이 아이콘, 자동 업데이트, 다중 창
- 단일 인스턴스 락(포트 분리로 불필요)
- Electron/Tauri 재구현
- 앱 내 로컬 LLM(Ollama 등) 기능 통합

## 아키텍처

```
┌─ run_desktop.bat (더블클릭)
│   ├ .venv_hana\Scripts\python.exe 결정
│   ├ streamlit / webview import 사전 점검 → 실패 시 안내, errorlevel 설정
│   └ python.exe desktop_app.py
│       └ errorlevel 1+ 이면 pause (정상 종료 시 pause 하지 않음)
│
├─ desktop_app.py (PORT=8502)
│   ├ 포트 점유 시 헬스체크: GET http://localhost:8502/_stcore/health
│   │   ├ "ok" 응답 → "내가 띄운 Streamlit" 으로 간주 → 재사용
│   │   └ 응답 없음/다른 프로세스 → exit 3 (명확한 에러)
│   ├ Streamlit 서브프로세스 기동 (새로 띄우는 경우)
│   │   ├ stdout/stderr → %LOCALAPPDATA%\hana_desktop\logs\desktop.log
│   │   └ CREATE_NO_WINDOW (Windows)
│   ├ localhost:8502 포트 오픈까지 90초 폴링
│   ├ pywebview 창(1440×900) 생성 → http://localhost:8502
│   └ 창 closed 이벤트 → Streamlit 프로세스 terminate (atexit 로 log_fh.close())
│
└─ hana_app/run.bat (기존, PORT=8501) — 변경 없음
    └ 병행 사용 시 포트 충돌 없음
```

### 핵심 결정

1. **포트 분리** — `run.bat`(8501) vs `run_desktop.bat`(8502). 한 사용자가 두 모드를 동시에 띄울 때 충돌을 원천 차단하고 PID 소유권 추적 복잡도 제거.
2. **사일런트 브라우저 fallback 제거** — pywebview `ImportError` 시 현재는 브라우저로 넘어가 "데스크탑 창인 척 브라우저"가 열려 같은 문제가 재발한다. 복구 후에는 명확한 에러 + 로그 경로 안내 후 종료.
3. **로그 보존** — 현재 `stdout=DEVNULL`라 서버 기동 실패 원인 추적 불가. `%LOCALAPPDATA%\hana_desktop\logs\desktop.log` 로 리다이렉트하여 소스 트리는 오염시키지 않는다.
4. **설치 통합** — pywebview 설치가 별도 배치(`install_pywebview.bat`)로 분리되어 `install_312.bat venv` 재실행 시 누락된다. 4단계에 오프라인 설치 한 블록 추가.

### session_state 3시간 지속성이 의존하는 것

이 요구사항을 **보장하는 것**과 **보장하지 않는 것**을 명시한다:

- **보장**: Streamlit 서버 프로세스(Python subprocess) 가 살아있는 동안 `st.session_state` 자체는 메모리에 유지된다. pywebview 창이 인트라넷 정책의 종료 대상이 아니라는 전제에서, 창이 살아있으면 WebSocket이 유지되어 세션이 지속된다.
- **보장하지 않음(기대 동작)**: 네트워크 순단으로 WebSocket이 끊겼다가 재연결되면 **같은 세션으로 복귀할 수 있으나 보장되지 않는다** — Streamlit 세션 키는 클라이언트 쿠키 기반이라 WebView 측이 토큰을 재생성하면 새 세션이 잡힌다.
- **확실히 유실**: 창이 정상 종료되면 `on_closed` 훅에서 서버를 내리므로 세션은 유실. 프로세스가 **강제 종료/로그오프** 등으로 `on_closed` 미발동 시 유령 서버가 남을 수 있으며, 이 경우 다음 실행에서 헬스체크 후 재사용/재기동으로 회복한다.

회귀 테스트(`3시간+ 방치`)는 이 연결고리 전체를 한 번에 검증한다.

## 구현 변경 사항

### A. `install_312.bat` 수정

4단계 "Streamlit 웹앱 핵심 패키지" 블록 직후 pywebview 설치 추가:

```bat
REM ── 4.2단계: 데스크탑 앱 (pywebview) ──────────────────────────
echo.
echo [4.2/5] 데스크탑 앱 (pywebview) 설치...
%PYTHON_BIN% -m pip install --no-index %FIND_LINKS% pywebview proxy_tools
if errorlevel 1 echo [경고] pywebview 설치 실패 — run_desktop.bat 미지원 (run.bat 은 정상)
```

5단계 검증에 `import webview` 추가:

```bat
echo [데스크탑]
%PYTHON_BIN% -c "import webview" 2>nul && echo   pywebview OK || (echo   [실패] pywebview & set FAIL=1)
```

**단서**: 이 체크는 **Python 바인딩만 검증**한다. Edge WebView2 Runtime은 `webview.start()` 시점에 실패하며 이 단계에서 감지되지 않는다. 강한 런타임 검증을 원할 경우(선택) 아래 한 줄 추가:

```bat
if not exist "%ProgramFiles(x86)%\Microsoft\EdgeWebView\Application" echo   [경고] Edge WebView2 Runtime 미감지 (run_desktop.bat 실패 가능)
```

`webview.__version__` 속성은 pywebview 6.x에서 보장되지 않아 검증 줄에서 사용하지 않는다.

`install_pywebview.bat` 은 **삭제하지 않고** 레거시 호환용으로 유지하되, **설치 집합을 `install_312.bat`과 일치**시킨다(`pywebview proxy_tools` 동시). 상단에 "선택사항 — `install_312.bat venv`가 표준 경로" 주석 추가.

### B. `desktop_app.py` 수정

- `PORT = 8502` 로 변경
- `import os`, `import atexit`, `import urllib.request` 추가(현재 파일에 없음)

**모듈 레벨 상수**(ImportError 분기에서도 안전하게 참조되도록 함수 밖에서 초기화):

```python
# 모듈 최상단(클래스/함수 정의 전)
LOG_DIR = Path(os.environ.get("LOCALAPPDATA", str(ROOT))) / "hana_desktop" / "logs"
try:
    LOG_DIR.mkdir(parents=True, exist_ok=True)
    LOG_FILE = LOG_DIR / "desktop.log"
    log_fh = LOG_FILE.open("a", encoding="utf-8", errors="replace")
    atexit.register(log_fh.close)
except OSError as e:
    print(f"[WARN] 로그 파일 생성 실패({e}) — stderr 로 대체", file=sys.stderr)
    LOG_FILE = None
    log_fh = sys.stderr
```

**포트 점유 시 헬스체크** — 임의 프로세스 오연결 방지:

```python
HEALTH_URL = f"http://localhost:{PORT}/_stcore/health"

def _is_our_streamlit(timeout: float = 2.0) -> bool:
    try:
        with urllib.request.urlopen(HEALTH_URL, timeout=timeout) as r:
            return r.status == 200 and b"ok" in r.read().lower()
    except Exception:
        return False

# main():
if _port_open(PORT):
    if not _is_our_streamlit():
        print(f"[ERROR] 포트 {PORT}가 Streamlit 이외의 프로세스에 점유됨.", file=sys.stderr)
        print(f"[INFO]  종료 후 재시도하거나 PORT 충돌 원인을 확인하세요.", file=sys.stderr)
        sys.exit(3)
    already_running = True
else:
    already_running = False
```

**서브프로세스 기동**:

```python
# subprocess.Popen(..., stdout=log_fh, stderr=subprocess.STDOUT)
```

**창 닫힐 때 유령 서버 방지**:

```python
def on_closed():
    if proc and proc.poll() is None:
        proc.terminate()
        try: proc.wait(timeout=5)
        except subprocess.TimeoutExpired: proc.kill()
```

**`_run_browser()` 함수 제거**. `ImportError` 분기에서 아래로 교체:

```python
print("[ERROR] pywebview 미설치.", file=sys.stderr)
print(f"[INFO]  install_312.bat venv 를 다시 실행하거나 install_pywebview.bat 를 실행하세요.", file=sys.stderr)
if LOG_FILE:
    print(f"[INFO]  로그: {LOG_FILE}", file=sys.stderr)
sys.exit(2)
```

- 서버 기동 실패(90초 timeout) 시 `LOG_FILE` 마지막 20줄 콘솔 출력 후 exit 1
- `--server.address=localhost` 플래그 추가(현재 누락 → 외부 바인딩 방지)
- `_wait_ready` timeout 을 60 → 90초 (xgboost/lightgbm import 부하 반영)

### C. `run_desktop.bat` 수정

- UTF-8 `chcp 65001` 추가(현재 누락)
- 사전점검 3단계: venv → streamlit → webview (각 단계 실패 시 안내 + `exit /b <code>`)
- **조건부 pause**: 정상 종료 시 창을 유지하지 않고, 에러 시에만 멈춤:
  ```bat
  "%PYTHON_BIN%" "%ROOT%desktop_app.py"
  if errorlevel 1 pause
  ```
- `.venv_hana` 우선순위 유지(기존 로직 재사용)

### D. `.claude/settings.local.json` 생성/수정

현재 프로젝트 한정 `auto-approve` 규칙:

```json
{
  "permissions": {
    "allow": [
      "Bash(python -m pip install*)",
      "Bash(install_312.bat*)",
      "Bash(./install_312.bat*)",
      "Bash(install_pywebview.bat*)",
      "Bash(./install_pywebview.bat*)",
      "Bash(run_desktop.bat*)",
      "Bash(./run_desktop.bat*)",
      "Bash(./hana_app/run.bat*)",
      "Edit(install_312.bat)",
      "Edit(desktop_app.py)",
      "Edit(run_desktop.bat)",
      "Edit(install_pywebview.bat)",
      "Edit(docs/superpowers/**)"
    ],
    "deny": [
      "Bash(rm -rf*)",
      "Bash(git push --force*)",
      "Bash(rmdir /S*)"
    ]
  }
}
```

**설계 원칙**: 허용 범위는 **실제 수정 예정 파일로만 한정**한다. 넓은 `Write(**/hana_app/**)` 는 이번 작업 범위에 없으므로 **제외**(Ollama 지적 반영). Bash 와일드카드는 시작 앵커를 넣어 `rm install_312.bat` 같은 우회를 차단한다.

기존 `.claude/settings.local.json` 이 있으면 deep-merge, 없으면 신규.

**주의**: `auto-approve` 는 단순 Bash/Edit 반복 루틴만 자동 승인한다. CLAUDE.md 의 "Never without asking" 원칙(NHIS 데이터 기록, `.docx` 편집, 분석 결과 auto-commit, 코드/변수명 발명 금지)은 여전히 사전 확인이 필요하다.

### E. 문서 업데이트

`web-user-guide.md` 에 "데스크탑 모드(run_desktop.bat) 사용법" 섹션 1개 추가:
- 언제 사용하나(3시간 자동 종료 회피)
- 실행/종료 방법
- 로그 위치
- 8502 포트 점유 시 대처

## 에러 처리

| 상황 | 현 동작 | 개선 후 |
| --- | --- | --- |
| pywebview 미설치 | 브라우저 열림(사일런트 fallback) | 에러 메시지 + 로그 경로 + exit 2 |
| Streamlit 기동 실패 | 90초 후 "Server failed to start" 한 줄 | 로그 마지막 20줄 출력 + exit 1 |
| 8502 포트에 **다른 프로세스** | 포트 오픈으로 오판 → 연결 | `/_stcore/health` 헬스체크로 식별 → exit 3 |
| 8502 포트에 내 Streamlit | 포트 오픈 감지 → 재사용 | 동일(헬스체크 통과 후 의도된 재사용) |
| WebView2 Runtime 없음 | pywebview가 상세 에러로 exit | 설치 검증 단계에서 경고(선택적) + 런타임은 `webview.start()` 시점 표면화 명시 |
| venv 미생성 | `desktop_app.py` 실행 불가 | run_desktop.bat 사전점검에서 명확한 안내 |
| 강제 종료/로그오프로 `on_closed` 미발동 | 유령 서버 잔존 | 다음 실행 시 헬스체크 통과 → 기존 서버 재사용으로 자연 회복 |
| 로그 디렉터리 생성 실패 | `desktop_app.py` 크래시 | `stderr` 로 fallback (부모 프로세스의 stderr는 `run_desktop.bat` 콘솔로 보임) |

## 테스트 & 검증

### 자동(Mac에서 pytest 실행 가능)
1. `tests/test_desktop_app.py` 신규
   - `_port_open` 동작 (임의 포트)
   - `_resolve_python` venv 우선순위
   - 로그 파일 경로 계산(환경변수 `LOCALAPPDATA` 유무)
   - `_is_our_streamlit()` — mock HTTP 서버로 `ok` / 비응답 / 엉뚱한 응답 3 케이스
2. `install_312.bat` 은 shell 스크립트라 단위 테스트 불가 → 문서화로 대신

### 구현 시 확인 체크리스트 (advisor 추가 지적)
- [ ] `proxy_tools` wheel이 pywebview 6.1의 실제 의존성인지 `pip show`로 확인 — 필요 없으면 설치 커맨드에서 제외
- [ ] `install_pywebview.bat` 의 설치 집합을 `install_312.bat` 과 일치시킴 (`pywebview proxy_tools` 동시, 또는 검증 결과에 따라 `pywebview` 단독)
- [ ] `webview.__version__` 대신 `import webview` 성공만으로 OK 출력 (pywebview 6.x 속성 비보장)

### 수동 체크리스트 (Windows 폐쇄망)
- [ ] `install_312.bat venv` 실행 → 5단계 검증에 `pywebview ... OK` 출력
- [ ] `run_desktop.bat` 더블클릭 → pywebview 창에 Streamlit 홈 표시
- [ ] 같은 PC에서 `hana_app\run.bat` 도 별도 실행 → 두 창이 8501/8502로 병행
- [ ] pywebview 창에서 HANA 연결 → 분석 수행 → 결과 확인
- [ ] **3시간+ 방치** 후 복귀 → 창 살아있고 `st.session_state` 유지(이전 결과 보이는지) ← **핵심 회귀 테스트**
- [ ] 창 X 클릭 → Python 프로세스 완전 종료(작업 관리자 확인)
- [ ] `desktop.log` 에 Streamlit 로그 기록 여부 확인

## 롤백 계획

모든 변경이 비파괴적(파일 교체/추가)이므로 git revert로 1커밋 복원. 단, `.venv_hana` 에 설치된 pywebview는 남는다 — 이는 문제 없음(쓰지 않는 패키지).

## 참조

- 이전 작동 버전: `desktop_app.py` (3월 25일 타임스탬프, 실사용 검증 완료)
- pywebview 6.1 wheel: `packages_win/py312/pywebview-6.1-py3-none-any.whl`
- proxy_tools wheel: `packages_win/py312/proxy_tools-0.1.0-py3-none-any.whl`
- advisor(Opus 4.7) 리뷰 피드백 반영: 포트 분리 / stale lock 고민 제거 / 로그 경로 / WebView2 감지 / session_state 내구성 체크포인트
- 로컬 모델 2차 리뷰(Codex GPT-5.4 + Ollama GLM-4.5-Air Q5_K_M) 반영: 8502 헬스체크 필수 / session_state 문구 완화 / 조건부 pause / LOG_FILE 모듈 레벨 / WebView2 체크 단서 / auto-approve 범위 축소 / atexit close / proxy_tools 및 `__version__` 검증 체크리스트
- advisor 종합 판정: Ollama의 `log_fh.fileno()` 제안은 **반영 금지**(fd 수명이 Python GC에 종속되어 오히려 더 취약)
