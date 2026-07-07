# 65세 이상 또는 노인성 질병코드 대상 금기/DDI 기관 감사 준비 계획

> **For Hermes:** Hermes가 LO(Lead Orchestrator)로 남고, 하위 에이전트는 단계별 역할만 임시 수행한다. 구현 시 `subagent-driven-development` 절차로 task별 TDD + 2단계 리뷰를 적용한다.

**Goal:** MODE_11_hana의 고정 Raw(2024-07..12) 처방 데이터를 이용해 65세 이상 대상자와 65세 미만 중 노인성 질병코드 보유자의 금기/심각 DDI 이벤트를 분류하고, 해당 이벤트에 관여한 처방기관 번호와 기관명을 감사 가능한 형태로 산출한다.

**Architecture:** 기존 학습/서빙/feature path는 건드리지 않고 `scripts/ops/` 아래 read-only 감사 모듈을 새로 둔다. DDI severity 산정은 `scripts.etl.prescription_aggregator.ddi_pair_severities()` 및 `DrugMaster + ddi_matrix_final.parquet`를 단일 출처로 재사용하되, 기관 귀속을 위해 ops 전용 attributed-overlap 구조를 새로 만든다. 사망자 제외와 기관명 매핑은 현재 로컬 Raw만으로는 완결되지 않으므로 death/institution master 입력을 prerequisite으로 둔다.

**Tech Stack:** Python 3.12, pandas, pyarrow/parquet, pytest. 운영 PC는 Windows 폐쇄망/Python 3.12이며 WSL 검증은 `.venv_wsl` Python 3.12.3을 기본으로 한다.

---

## 0. Hermes LO 현재 확인 사실

- Repo: `/mnt/c/model/mode_11_hana` (`C:\model\MODE_11_hana`).
- Project rules: `AGENTS.md`, `CLAUDE.md` 모두 Hermes LO 원칙, HANA schema/table/column 추측 금지, Python 3.12 parity, protected artifacts 보호를 명시한다.
- Final Raw policy: 2024-07..12 daily `records_YYYYMMDD.parquet` 184개 + eligibility 500k. 2025-01 없음. Gate 5A/5B 폐기. Future-onset track freeze 유지.
- Protected paths: `data/Raw/*.parquet`, generated parquet, `mlruns/`, `out/`, `packages_win/py312/`는 명시 승인 없이 수정 금지.
- Existing dirty state: `.understand-anything/*`, `hana_app/core/report_exporter.py`, `tests/test_hana_app/test_report_exporter.py`가 이미 modified. 이 작업은 해당 파일을 건드리지 않는다.
- Read-only metadata scan:
  - `data/Raw/eligibility_demographics.parquet`: 500,000 rows, columns `patient_id`, `byear`, `age`, `sex_type`, `addr_cd`.
  - `data/Raw/eligibility_ages.parquet`: 500,000 rows, columns `patient_id`, `age`.
  - `data/Raw/records_20240701.parquet`: columns `patient_id`, `institution_id`, `bill_no`, `wk_compn_cd`, `edi_code`, `gnl_nm_cd`, `efmdc_clsf_no`, `start_date`, `end_date`, `total_days`, `dose_once`, `dose_freq`, `sick_code`, `sex`, `age_id`, `institution_type`, `source`.
  - Local `data/`에는 death parquet와 institution master/yoyang parquet가 없다.
- Existing drug/DDI artifacts:
  - `data/processed/ddi_matrix_final.parquet`: columns include `drug_a_id`, `drug_b_id`, `severity`.
  - `data/processed/hira_drug_master.parquet`: columns include `ingr_code`, `components`, `ingr_name_raw`.
  - `data/dur/dur_ddi_contraindicated_std.parquet`: DUR contraindicated DDI reference artifact.
- Confirmed layout-only HANA sources for later target-PC extraction:
  - `HHDT_DEATH`: `INDI_DSCM_NO`, `DTH_ASSMD_DT`, `DTH_HM_DT`, `DTH_BFC_DT`, etc.
  - `HHRT_MCINST_YY`: `STD_YYYY`, `MDCARE_SYM`, `INST_NM`, `ADDR`, etc.
  - `HHDV_DSES_YY`: `BYEAR`, `SEX_TYPE`, `RVSN_ADDR_CD`, etc.
- Disease-code source supplied by the user:
  - PDF folder: `/mnt/c/model/mode_11_old/disease` (`C:\model\mode_11_old\disease`).
  - PDF found/extracted: `[별표 1] 노인성 질병의 종류(제2조 관련)(노인장기요양보험법 시행령).pdf`.
  - SHA-256: `5ea16da019b2309b0bcf7fbe00d6305092de8f7cddec1334ee391883d144b3cc`.
  - Extracted KCD disease codes: `F00*`, `F01`, `F02*`, `F03`, `G30`, `I60`, `I61`, `I62`, `I63`, `I64`, `I65`, `I66`, `I67`, `I68*`, `I69`, `G20`, `G21`, `G22*`, `G23`, `U23.4`, `R25.1`, `G12`, `G13*`, `G35`.

---

## 1. Operational definitions (v1)

### 1.1 대상자: 65세 이상 또는 65세 미만 노인성 질병코드 보유자

Primary local definition:
- Use `data/Raw/eligibility_demographics.parquet.age` as the age source.
- Include patients where numeric `age >= 65`.
- Also include patients where numeric `age < 65` **and** at least one diagnosis code matches the 노인성 질병 codelist extracted from `C:\model\mode_11_old\disease` PDFs.
- Exclude rows with missing/non-numeric `patient_id` or `age` from the audit cohort and report their counts unless a later user-approved rule explicitly allows disease-code-only inclusion with unknown age.

Disease-code codelist (source PDF):
- Source path: `/mnt/c/model/mode_11_old/disease/[별표 1] 노인성 질병의 종류(제2조 관련)(노인장기요양보험법 시행령).pdf`.
- Extracted codes: `F00*`, `F01`, `F02*`, `F03`, `G30`, `I60`, `I61`, `I62`, `I63`, `I64`, `I65`, `I66`, `I67`, `I68*`, `I69`, `G20`, `G21`, `G22*`, `G23`, `U23.4`, `R25.1`, `G12`, `G13*`, `G35`.
- Matching policy: uppercase, strip whitespace, dots, and trailing `*` for comparison. Three-character category codes such as `I60`/`G30` match by prefix. Subcategory codes such as `U23.4` and `R25.1` match after dot stripping (`U234`, `R251`) and may use prefix matching only if the KCD source later confirms child codes.

Diagnosis source:
- Local Raw has `records_*.parquet.sick_code`; this can support a first-pass under-65 disease-code inclusion.
- If the study requires **all** diagnoses, not just the Raw-projected `sick_code`, require a T40/full-diagnosis export and treat Raw-only disease inclusion as provisional.

Do not use:
- `records_*.parquet.age_id` / `SUJIN_POTM_AGE_ID` as the primary 65세 filter. It is an age bucket/ID, not a confirmed exact age.

Consistency check:
- `eligibility_demographics.age` is generated in existing code as `reference_year - byear` when demographics are saved. If an implementation recomputes age from `byear`, it must use the same explicit reference year and report discrepancies instead of silently changing cohort membership.

BLOCK:
- If `eligibility_demographics.parquet` lacks `age` or the age reference year cannot be explained for the requested audit, pause and ask Hermes LO whether to use `byear`-based age.
- If no usable diagnosis source is available, continue only with the `age >= 65` cohort and report under-65 disease-code inclusion as unavailable.

### 1.2 사망자 제외

Required production definition:
- A complete run requires a death input with `patient_id` plus at least one valid death date column derived from `HHDT_DEATH`.
- Normalize `death_date` as the earliest valid date among configured death columns, defaulting to `DTH_ASSMD_DT`, `DTH_HM_DT`, `DTH_BFC_DT` when present.
- Exclude patients with `death_date <= audit_end`.
- Patients with a death row but no parseable death date are reported as a data-quality warning and excluded only if a strict `exclude_unparseable_death_rows=True` option is explicitly chosen.

Current local state:
- No local death parquet was found. Therefore a final “사망자 제외 완료” audit is HARD_STOP until death data is supplied or extracted on the target PC.

Allowed provisional mode:
- For code testing only, support `death_policy="provisional_allow_missing"`; this must mark outputs as `death_exclusion_status=unavailable` and must not be reported as final.

### 1.3 금기약물 / 심각한 약물상호작용 scope

Confirmed v1 scope:
- Treat drug-drug pair severity `Contraindicated` as “금기 DDI”.
- Treat `Major` as “심각한 약물상호작용”.
- Default result severity allowlist: `("Contraindicated", "Major")`.
- Reuse `scripts.etl.prescription_aggregator.ddi_pair_severities()` with `DrugMaster.load_parquet(data/processed/hira_drug_master.parquet, ddi_matrix_path=data/processed/ddi_matrix_final.parquet)`.

Important ambiguity:
- If “금기약물” means a single-drug elderly/PIM/Beers/STOPP-style contraindicated-medication list, that codelist is not currently confirmed in the repo. `config/drug_rules.yaml` contains DDI/top10/high-risk groups, not a validated elderly single-drug contraindication source. A separate single-drug PIM track needs an approved codelist before positives can be generated.

### 1.4 기관 attribution

Institution number:
- Use prescription `institution_id` from Raw records, corresponding to HANA `MDCARE_SYM`.

Institution name:
- Requires local or freshly extracted `HHRT_MCINST_YY`-derived master with `STD_YYYY`, `MDCARE_SYM`, `INST_NM`.
- Map by explicit `std_year`, default `audit_end.year` (`2024` for the fixed Raw audit).
- If institution name is missing, keep `institution_id`, set `institution_name=None`, and report unmatched count/rate.

DDI pair attribution:
- A DDI pair has two prescriptions. Output both sides:
  - `institution_a_id`, `institution_a_name`
  - `institution_b_id`, `institution_b_name`
  - `same_institution`
- If the two prescriptions are from different institutions, both institutions are “pair-involved prescribing institutions.” Do not infer sole responsibility without a separate clinical/operational rule.

### 1.5 Output privacy

Default final user-facing report should be institution-level aggregates, not raw patient rows.

Allowed internal row-level columns for local audit only:
- `patient_id`, `age`, `severity`, `drug_a_wk_compn`, `drug_b_wk_compn`, `drug_a_edi`, `drug_b_edi`, `institution_a_id/name`, `institution_b_id/name`, `overlap_start/end/days`, `source_a/source_b`.

User-facing aggregate columns:
- `institution_id`, `institution_name`, `severity`, `event_count`, `distinct_patient_count`, `same_institution_event_count`, `cross_institution_event_count`, `unmatched_institution_name_count`.

---

## 2. Subagent routing and dynamic role reassignment

### Phase A — Discovery/risk gate

- Hermes LO: owns user communication, sequencing, final decisions.
- AGY HQ role: `risk-auditor`.
  - Verify Python 3.12, protected-path policy, final Raw/freeze constraints, death/yoyang prerequisites.
  - Current result: local death/yoyang artifacts absent; death exclusion and institution name mapping require target-PC extraction or supplied parquet.

Dynamic reassignment:
- If death/yoyang missing: AGY role changes from `risk-auditor` to `extraction-prerequisite-planner`.
- If HANA schema ambiguity appears: AGY/Claude route returns BLOCK; no Codex implementation beyond tests/scaffolding.

### Phase B — Operational definition/spec

- Claude HQ desired role: `architect` for clinical/logic definition and leakage/schema review.
- Current limitation: Claude CLI unavailable (`claude CLI unavailable and ANTHROPIC_API_KEY not set`).
- Fallback used: Codex HQ role `technical-architect` for implementation/API/TDD plan, with Hermes manually reconciling AGY risk findings.

Dynamic reassignment:
- If Claude becomes available, send this plan for read-only logic review before production use.
- If clinical definition of “금기약물” is clarified as single-drug elderly/PIM, assign Claude/AGY to codelist provenance review before Codex implements.

### Phase C — TDD implementation

- Codex HQ role: `tdd-implementer`.
- Scope: new ops-only module and tests; no training/serving/schema changes.
- Hermes LO verifies every changed file and test output locally.

Dynamic reassignment:
- If performance/memory fails on six-month Raw: AGY role changes to `bulk-profiler` to design chunking/temp disk strategy.
- If DDI mapping drift is discovered: Codex switches from implementer to `technical-reviewer`, Hermes decides whether this is a separate DrugMaster repair task.

### Phase D — Review/validation

- Codex HQ role: `technical-reviewer` for code quality and test gaps.
- Claude HQ role, when available: `logic-reviewer` for operational definitions, death exclusion semantics, and institution attribution.
- AGY HQ role: `target-pc-validator` for HANA extraction prerequisites and closed-network runbook.

---

## 3. Proposed implementation surface

Create:
- `scripts/ops/elderly_ddi_institution_audit.py`
- `tests/test_ops/test_elderly_ddi_institution_audit.py`

Do not modify in this feature:
- `hana_app/core/ml_runner.py`
- `scripts/etl/prescription_aggregator.py`
- `scripts/etl/overlap_calculator.py`
- `serving/`, `dags/`, model bundles, generated parquet artifacts.

Core API sketch:

```python
from __future__ import annotations

from dataclasses import dataclass
from datetime import date
from pathlib import Path
from typing import Iterable, Sequence

import pandas as pd

@dataclass(frozen=True)
class AttributedOverlapPair:
    patient_id: str
    drug_a_wk_compn: str
    drug_b_wk_compn: str
    drug_a_edi: str | None
    drug_b_edi: str | None
    institution_a_id: str | None
    institution_b_id: str | None
    source_a: str | None
    source_b: str | None
    overlap_start: date
    overlap_end: date
    overlap_days: int

@dataclass(frozen=True)
class AuditPreflight:
    ok: bool
    errors: tuple[str, ...] = ()
    warnings: tuple[str, ...] = ()
    details: dict[str, object] | None = None

class DeathDataRequiredError(ValueError):
    pass

def normalize_disease_codes_from_pdf_text(text: str) -> tuple[str, ...]: ...

def patient_has_disease_code(
    diagnosis_codes: Iterable[object],
    disease_codes: Sequence[str],
) -> bool: ...

def select_target_patients(
    eligibility: pd.DataFrame,
    diagnoses: pd.DataFrame | None,
    *,
    disease_codes: Sequence[str],
    min_age: int = 65,
) -> tuple[pd.DataFrame, AuditPreflight]: ...

def normalize_death_dates(
    deaths: pd.DataFrame,
    *,
    date_columns: Sequence[str] = ("DTH_ASSMD_DT", "DTH_HM_DT", "DTH_BFC_DT"),
) -> pd.DataFrame: ...

def exclude_deceased_patients(
    cohort: pd.DataFrame,
    deaths: pd.DataFrame | None,
    *,
    audit_end: date,
    death_policy: str = "require",
) -> tuple[pd.DataFrame, AuditPreflight]: ...

def build_institution_name_map(
    yoyang: pd.DataFrame,
    *,
    std_year: str,
) -> dict[str, str]: ...

def records_to_prescriptions(records: pd.DataFrame) -> list[PrescriptionRecord]: ...

def calculate_attributed_overlaps_for_patient(
    prescriptions: list[PrescriptionRecord],
    *,
    window_days: int = 90,
    min_overlap: int = 7,
) -> list[AttributedOverlapPair]: ...

def classify_attributed_ddi_pairs(
    pairs: list[AttributedOverlapPair],
    *,
    ddi_matrix: pd.DataFrame,
    drug_master,
    institution_names: dict[str, str],
    severity_allowlist: tuple[str, ...] = ("Contraindicated", "Major"),
) -> pd.DataFrame: ...

def summarize_by_institution(events: pd.DataFrame) -> pd.DataFrame: ...
```

Notes:
- `records_to_prescriptions()` can reuse `PrescriptionRecord` from `scripts.etl.models` and must parse local Raw column names, not HANA original names.
- `calculate_attributed_overlaps_for_patient()` should copy the overlap rules from `calculate_overlaps_for_patient()` but keep source prescription institutions. Do not modify the shared ETL overlap dataclass just for this audit.
- `classify_attributed_ddi_pairs()` may convert `AttributedOverlapPair` to temporary `DrugOverlapPair` objects only for calling `ddi_pair_severities()`.

---

## 4. Bite-sized TDD tasks

### Task 1: Add target-cohort/death preflight tests

**Objective:** Lock the 65세 기준, 65세 미만 노인성 질병코드 포함, and death prerequisite semantics before implementation.

**Files:**
- Create: `tests/test_ops/test_elderly_ddi_institution_audit.py`
- Later create: `scripts/ops/elderly_ddi_institution_audit.py`

**Step 1: Write failing tests**

Test cases:
- `age == 65` is included regardless of disease code.
- `age == 64` without a matching disease code is excluded.
- `age == 64` with a matching disease code such as `I63` or dotted `R25.1` is included.
- Disease code normalization strips dots/trailing `*` and matches category prefixes (`F00*` source matches `F000` diagnosis).
- Missing/non-numeric age is excluded and counted in warnings/details unless a later explicit rule allows unknown-age disease-code-only inclusion.
- `exclude_deceased_patients(..., deaths=None, death_policy="require")` raises `DeathDataRequiredError`.
- Death date on/before `audit_end` is excluded.
- Death date after `audit_end` remains included.

**Step 2: Verify RED**

Run:

```bash
cd /mnt/c/model/mode_11_hana
.venv_wsl/bin/python -m pytest tests/test_ops/test_elderly_ddi_institution_audit.py -q
```

Expected: FAIL because module/functions do not exist.

**Step 3: Implement minimal target-cohort/death functions**

Implement only:
- `DeathDataRequiredError`
- `normalize_disease_codes_from_pdf_text()`
- `patient_has_disease_code()`
- `select_target_patients()`
- `normalize_death_dates()`
- `exclude_deceased_patients()`

**Step 4: Verify GREEN**

Same pytest command; expected: PASS for Task 1 tests.

### Task 2: Add institution master mapping tests

**Objective:** Ensure `STD_YYYY + MDCARE_SYM -> INST_NM` mapping is deterministic and missing names are explicit.

**Files:**
- Modify: `tests/test_ops/test_elderly_ddi_institution_audit.py`
- Modify: `scripts/ops/elderly_ddi_institution_audit.py`

Test cases:
- Maps only rows where `STD_YYYY == "2024"`.
- Whitespace-normalizes `MDCARE_SYM` and `INST_NM`.
- Duplicate same-year institution rows keep the last deterministic row after stable sort or raise a clear error; choose and test one policy.
- Unknown institution id is preserved downstream with `None` name.

### Task 3: Add attributed-overlap tests

**Objective:** Preserve institution A/B through DDI overlap calculation.

Test cases:
- Two different WK prescriptions from same patient and overlapping >=7 days produce one pair.
- Non-overlap or overlap <7 days produces no pair.
- Same WK pair is excluded as duplicate-drug, matching existing overlap behavior.
- Cross-institution pair has both institution IDs and `same_institution=False` downstream.
- Same-institution pair has `same_institution=True` downstream.

Implementation caution:
- Copy the algorithm shape from `scripts/etl/overlap_calculator.py` into ops module or wrap local records, but do not change the production dataclass.

### Task 4: Add DDI severity classification tests

**Objective:** Reuse project DDI severity semantics and filter only Contraindicated/Major.

Test setup:
- Use a small fake `drug_master` object with `get_ddi_ids(wk)` returning deterministic IDs.
- Use a tiny `ddi_matrix` DataFrame with `drug_a_id`, `drug_b_id`, `severity`.

Test cases:
- Contraindicated pair included.
- Major pair included.
- Moderate/Minor pair excluded by default allowlist.
- Missing DrugMaster IDs do not become positives.
- Highest severity wins if duplicate matrix rows exist for the same ID pair.

### Task 5: Add institution-level summary tests

**Objective:** Produce privacy-preserving aggregate output.

Test cases:
- A cross-institution DDI event increments both institutions’ `event_count` or explicitly increments a `pair_involved_event_count` for each institution. The policy must be named in the output.
- `distinct_patient_count` is computed per institution without exposing patient IDs.
- Missing institution names are counted.
- Empty events return a stable empty schema.

### Task 6: Add CLI only after pure functions pass

**Objective:** Provide an operator entrypoint without weakening prerequisites.

CLI contract:

```bash
.venv_wsl/bin/python -m scripts.ops.elderly_ddi_institution_audit \
  --raw-dir data/Raw \
  --death-parquet <HHDT_DEATH_export.parquet> \
  --institution-master-parquet <HHRT_MCINST_YY_export.parquet> \
  --audit-start 2024-07-01 \
  --audit-end 2024-12-31 \
  --std-year 2024 \
  --summary-csv <safe-output-path>
```

Preflight behavior:
- Missing death parquet -> non-zero exit with clear HARD_STOP message.
- Missing institution master -> non-zero exit with clear BLOCK/HARD_STOP message for institution names.
- Default output must be aggregate summary only.
- Row-level output requires an explicit `--row-level-output` flag and should warn about patient identifiers.

---

## 5. Verification commands

Focused tests:

```bash
cd /mnt/c/model/mode_11_hana
.venv_wsl/bin/python -m pytest tests/test_ops/test_elderly_ddi_institution_audit.py -q
```

Syntax:

```bash
cd /mnt/c/model/mode_11_hana
.venv_wsl/bin/python -m py_compile scripts/ops/elderly_ddi_institution_audit.py tests/test_ops/test_elderly_ddi_institution_audit.py
```

Related regression guard after implementation:

```bash
cd /mnt/c/model/mode_11_hana
.venv_wsl/bin/python -m pytest \
  tests/test_ops/test_elderly_ddi_institution_audit.py \
  tests/test_ops/test_multi_day_parquet_provider.py \
  tests/test_ops/test_eligibility_loader.py \
  tests/test_etl/test_overlap_calculator.py \
  tests/test_etl/test_drug_master.py \
  tests/test_hana_app/test_drugmaster_ddi_wiring.py \
  -q
```

Do not claim full project test success unless the broader suite is actually run and passes.

---

## 6. Target-PC / HANA prerequisite bundle

Before a final real audit can be reported, obtain one of these:

1. `HHDT_DEATH` export parquet with `INDI_DSCM_NO`, `DTH_ASSMD_DT`, `DTH_HM_DT`, `DTH_BFC_DT`.
2. `HHRT_MCINST_YY` export parquet with `STD_YYYY`, `MDCARE_SYM`, `INST_NM` at minimum.

Safe extraction rules:
- Use confirmed schema/table names only:
  - `NHISBDA.HHDT_DEATH`
  - `NHISBDA.HHRT_MCINST_YY`
- Do not store credentials in files.
- Do not overwrite existing `data/Raw/*.parquet`.
- Prefer an explicit audit input directory outside protected generated Raw, for example a user-approved path such as `audit_inputs/` or a target-PC local folder copied into the workspace.
- Record SHA-256 manifest for supplied/extracted prerequisite files.

---

## 7. Open decisions for Hermes/user before final implementation

1. Cohort boundary: confirm final target definition is `age >= 65 OR (age < 65 AND 노인성 질병코드 보유)` and whether Raw `sick_code` is sufficient or a full T40/all-diagnosis export is required.
2. Does “금기약물” mean only DDI `Contraindicated`, or also a single-drug elderly/PIM/contraindicated-medication list?
   - Default v1: DDI `Contraindicated` + `Major` interactions only.
   - Single-drug PIM requires approved codelist.
3. Death handling semantics:
   - Decide whether to exclude the entire patient if `death_date <= audit_end`, or instead censor/exclude only prescriptions/events after death date.
   - Confirm death export join key equivalence: HANA `INDI_DSCM_NO` must map to local `patient_id`, or a mapping step is required.
   - Confirm date column meaning before using earliest of `DTH_ASSMD_DT`, `DTH_HM_DT`, `DTH_BFC_DT`; if unconfirmed, prefer `DTH_ASSMD_DT` only.
4. DDI/event dedup granularity:
   - Define daily Raw prescription-row dedup key before event counting, e.g. `patient_id + bill_no + edi_code/wk_compn_cd + start_date + institution_id` or an approved alternative.
   - Define DDI overlap dedup key for institution attribution. Existing overlap logic dedups per patient `(wk_a, wk_b)` and may drop repeated or cross-institution events; audit may need per occurrence or per `(wk_pair, institution_pair)` policy.
5. DDI severity/source of truth:
   - Reconcile `ddi_matrix_final.parquet` severity route with `dur_ddi_contraindicated_std.parquet` DUR-contraindicated route before positives are generated.
   - Confirm D-code/DB-code ID space and the event key basis (`wk_compn_cd` vs `gnl_nm_cd`/`edi_code`).
6. Cross-institution DDI attribution:
   - Default v1: both institutions are pair-involved; no sole-responsibility inference.
   - Define `same_institution` behavior when one side has missing institution id/name.
7. Output level:
   - Default v1: aggregate institution summary only.
   - Row-level patient audit requires explicit local-only approval because it contains identifiers.

---

## 8. Subagent handoff summary

Tmux transport configured at `.hermes/tmux_agents/` with dedicated session `mode11_hana_agents`, panes `agy`, `claude`, `codex`, `opencode`, and `pipe-pane` logs under `.hermes/tmux_agents/current/logs/`. Hermes LO sends work through `tmux send-keys` using `.hermes/tmux_agents/tmux_lo.py` and `.hermes/tmux_agents/run_prompt.py`.

- AGY HQ completed read-only risk gate: death and institution master are prerequisites; full T40/all-diagnosis export may be needed for under-65 노인성 질병코드 inclusion; target Windows-vs-WSL/offline wheel parity must be decided.
- Claude HQ completed read-only architecture/logical review: warned that current overlap dedup semantics can make institution attribution arbitrary, daily Raw row dedup is undefined, and death join/date semantics must be confirmed.
- Codex HQ completed read-only technical validation: recommended `scripts/ops/elderly_ddi_institution_audit.py` + `tests/test_ops/test_elderly_ddi_institution_audit.py`, with pure-function TDD and no production path edits; confirmed DDI helper reuse is feasible.
- OpenCode HQ is now a standard fourth LO lane, not omitted/fallback-only: OpenCode Go provider with default `opencode-go/glm-5.2`, role-based stronger OpenCode Go models allowed for implementation/refactor/architecture/deep review, and per-dispatch override through `tmux_lo.py send opencode --role ... --model ...`.
- Hermes LO owns final reconciliation, local verification, sequencing, and user-facing reporting.

### 2026-07-03 Task 1 implementation status

- Hermes completed strict TDD for Task 1 cohort/death preflight helpers in `scripts/ops/elderly_ddi_institution_audit.py` with tests in `tests/test_ops/test_elderly_ddi_institution_audit.py`.
- RED observed: initial 6 tests failed with `ModuleNotFoundError`; review-driven RED regressions caught starred KCD code extraction after punctuation and integer/float-like `YYYYMMDD` death-date parsing.
- GREEN/verification: `tests/test_ops/test_elderly_ddi_institution_audit.py` passes (7 tests), and the focused regression bundle with existing DDI/multi-institution ops tests passes (20 tests).
- Claude tmux review verdict: PASS with minor findings; Hermes patched star-code extraction and warning status semantics.
- Codex tmux review verdict: REQUEST_CHANGES for integer/float-like death dates, then PASS after the regression and parser patch.
- Remaining implementation gates before final audit remain unchanged: death-vs-censoring policy, death ID mapping, full T40/all-diagnosis source, institution master, DDI source of truth, and event dedup granularity.

### 2026-07-03 Task 2 implementation status

- Hermes documented and re-used the project-local tmux LO procedure in `.hermes/tmux_agents/LO_PROCEDURE.md`: `pipe-pane` logs, `send-keys` dispatch through `run_prompt.py`, serialized idle checks, capture/report, and local verification before accepting subagent reports.
- Hermes completed strict TDD for Task 2 institution master helpers in `scripts/ops/elderly_ddi_institution_audit.py` with tests in `tests/test_ops/test_elderly_ddi_institution_audit.py`.
- RED observed: 4 new Task 2 tests failed with `ImportError` before implementation; Claude's minor duplicate/blank-name finding was converted into a RED regression before patching.
- Implemented policy: normalized `STD_YYYY + MDCARE_SYM -> INST_NM`; leading/trailing whitespace stripped; duplicate same-year rows are last-row-wins; if the final row has a blank `INST_NM`, the institution remains unmapped so downstream output preserves the ID with name `None`.
- Implemented downstream helper: `attach_institution_names(...)` adds name columns for one or more institution-id columns, preserves unknown IDs, and reports unmatched-name counts/IDs in `AuditPreflight`.
- Verification: `tests/test_ops/test_elderly_ddi_institution_audit.py` passes (12 tests), and the focused regression bundle with existing DDI/multi-institution ops tests passes (25 tests); syntax compilation passed.
- Claude tmux review verdict: PASS; optional minor duplicate/blank-name edge was pinned and patched.
- Codex tmux review verdict: PASS; no remaining Task 2 blockers and no production-path impact found.
- Remaining implementation gates before final audit remain unchanged: real `HHRT_MCINST_YY` input is still required for final institution names, and DDI/event attribution semantics remain for later tasks.

### 2026-07-03 Task 3 implementation status

- Hermes used the documented tmux LO procedure for Task 3: `pipe-pane` logs were active, prompts were sent through `send-keys`/`run_prompt.py`, and Hermes waited for idle/captured each agent report before continuing.
- Claude was assigned Task 3 spec review. Verdict: PASS, with required TDD cases for zipped institution attribution, older long-running prescriptions, exact 7-day/6-day boundaries, missing institution semantics, repeated occurrences, and no silent `MAX_DRUGS_PER_WINDOW` truncation.
- Hermes completed strict RED-GREEN for ops-only attributed-overlap helpers in `scripts/ops/elderly_ddi_institution_audit.py` with tests in `tests/test_ops/test_elderly_ddi_institution_audit.py`.
- Implemented `AttributedOverlapPair`, `records_to_prescriptions(...)`, and `calculate_attributed_overlaps_for_patient(...)` without modifying production ETL/training/serving code.
- Implemented policy: same `wk_compn_cd` pairs excluded; exact 7-day overlaps included; 6-day overlaps excluded; institution A/B and source A/B preserved; `same_institution` returns `True`, `False`, or `None` when either side is missing.
- Implemented occurrence-preserving zipped dedup key based on each side's `(wk_compn_cd, institution_id, source)` plus overlap dates so swapped institution assignment and repeated time episodes are not collapsed incorrectly.
- Codex technical review first returned REQUEST_CHANGES because the initial scan missed older long-running prescriptions still active in later anchor windows. Hermes added `test_attributed_overlap_counts_older_long_running_rx_active_in_later_window`, observed RED (`len(pairs) == 0`), patched active-window semantics, and reran verification.
- Codex final evidence-only re-review verdict: PASS, no remaining blocker.
- AGY operational/risk review verdict: PASS. AGY flagged scale risks for six-month Raw (`records_to_prescriptions` materialization and no production cap). Hermes removed whole-frame `.to_dict("records")` materialization; the no-cap behavior remains intentional for audit completeness but should be paired with a later per-patient anomaly/preflight gate before full-scale execution.
- Verification after the final AGY mitigation: `tests/test_ops/test_elderly_ddi_institution_audit.py`, `tests/test_ops/test_ddi_mapping_audit.py`, `tests/test_ops/test_multi_institution_label.py`, and `tests/test_etl/test_prescription_aggregator.py` pass together (`78 passed in 6.93s`); `py_compile` for the new module/test also passed.
- Remaining implementation gates before final audit: real `HHRT_MCINST_YY` institution master input, `HHDT_DEATH` death input/policy, full T40/all-diagnosis source decision, DDI source-of-truth reconciliation, daily Raw row-level dedup gate, and full-scale chunked/per-patient anomaly handling.
