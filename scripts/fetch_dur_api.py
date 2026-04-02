#!/usr/bin/env python3
"""
식약처 DUR OpenAPI 데이터 수집기

폐쇄망 반입 전 인터넷 연결 환경에서 실행하여 모든 DUR 데이터를 로컬에 저장.
수집된 parquet 파일을 폐쇄망으로 반입하여 오프라인 사용.

API: https://apis.data.go.kr/1471000/DURIrdntInfoService03
문서: drugbank/식약처 openAPI.docx

사용법:
  # 환경변수로 API 키 설정 (권장)
  export DUR_API_KEY="your_api_key_here"
  python scripts/fetch_dur_api.py

  # 특정 엔드포인트만 수집
  python scripts/fetch_dur_api.py --endpoints usjnt_taboo efcy_dplct

  # 설정 파일 지정
  python scripts/fetch_dur_api.py --config config/api_config.yaml
"""
import argparse
import os
import sys
import time
from pathlib import Path
from typing import Optional

import pandas as pd
import requests
import yaml


def load_config(config_path: str = "config/api_config.yaml") -> dict:
    with open(config_path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def get_api_key(config: dict) -> str:
    """환경변수에서 API 키 조회. 미설정 시 RuntimeError 발생."""
    env_key = config["dur_api"].get("api_key_env", "DUR_API_KEY")
    key = os.environ.get(env_key, "")
    if not key:
        raise RuntimeError(
            f"환경변수 {env_key} 미설정. "
            f"실행 전 `export {env_key}=<your_key>` 로 API 키를 설정하세요."
        )
    return key


def fetch_endpoint_page(
    session: requests.Session,
    base_url: str,
    path: str,
    api_key: str,
    page_no: int,
    num_of_rows: int,
    timeout: int,
) -> dict:
    """단일 페이지 요청."""
    url = base_url + path
    params = {
        "serviceKey": api_key,
        "pageNo": page_no,
        "numOfRows": num_of_rows,
        "type": "json",
    }
    resp = session.get(url, params=params, timeout=timeout)
    resp.raise_for_status()
    return resp.json()


def extract_items(response_json: dict) -> tuple[list, int]:
    """응답 JSON 에서 아이템 목록과 총 건수 추출."""
    try:
        body = response_json["body"]
        total_count = int(body.get("totalCount", 0))
        items = body.get("items", [])
        # items 가 dict 인 경우 (단건) → list 로 변환
        if isinstance(items, dict):
            item = items.get("item", [])
            items = item if isinstance(item, list) else [item]
        elif isinstance(items, list):
            # 각 요소가 {"item": {...}} 중첩 구조인 경우 언래핑
            if items and isinstance(items[0], dict) and list(items[0].keys()) == ["item"]:
                items = [i["item"] for i in items]
        else:
            items = []
        return items, total_count
    except (KeyError, TypeError, ValueError) as e:
        print(f"  [경고] 응답 파싱 오류: {e}")
        return [], 0


def fetch_all_pages(
    session: requests.Session,
    base_url: str,
    path: str,
    endpoint_name: str,
    api_key: str,
    num_of_rows: int = 100,
    timeout: int = 30,
    retry_count: int = 3,
    retry_delay: float = 1.0,
    rate_delay: float = 0.2,
) -> list:
    """모든 페이지를 수집하여 아이템 목록 반환."""
    all_items = []
    page_no = 1

    # 첫 페이지로 총 건수 파악
    for attempt in range(retry_count):
        try:
            resp = fetch_endpoint_page(session, base_url, path, api_key, page_no, num_of_rows, timeout)
            items, total_count = extract_items(resp)
            break
        except Exception as e:
            if attempt < retry_count - 1:
                print(f"  [재시도 {attempt+1}] {e}")
                time.sleep(retry_delay)
            else:
                print(f"  [오류] {endpoint_name} 첫 페이지 수집 실패: {e}")
                return []

    if total_count == 0:
        print(f"  [정보] {endpoint_name}: 데이터 없음")
        return []

    all_items.extend(items)
    total_pages = (total_count + num_of_rows - 1) // num_of_rows
    print(f"  총 {total_count:,} 건, {total_pages} 페이지")

    for page_no in range(2, total_pages + 1):
        time.sleep(rate_delay)
        for attempt in range(retry_count):
            try:
                resp = fetch_endpoint_page(session, base_url, path, api_key, page_no, num_of_rows, timeout)
                items, _ = extract_items(resp)
                all_items.extend(items)
                if page_no % 10 == 0:
                    print(f"  {page_no}/{total_pages} 페이지 완료 ({len(all_items):,} 건)...", end="\r", flush=True)
                break
            except Exception as e:
                if attempt < retry_count - 1:
                    time.sleep(retry_delay)
                else:
                    print(f"\n  [경고] 페이지 {page_no} 수집 실패: {e}")

    print(f"\n  수집 완료: {len(all_items):,} 건")
    return all_items


def fetch_endpoint(
    config: dict,
    endpoint_key: str,
    api_key: str,
    out_dir: Path,
) -> Optional[pd.DataFrame]:
    """특정 엔드포인트 전체 수집 후 parquet 저장."""
    dur_cfg = config["dur_api"]
    ep_cfg = dur_cfg["endpoints"].get(endpoint_key)
    if not ep_cfg:
        print(f"[오류] 엔드포인트 '{endpoint_key}' 설정 없음")
        return None

    path = ep_cfg["path"]
    desc = ep_cfg["description"]
    print(f"\n[{endpoint_key}] {desc} 수집 시작...")

    session = requests.Session()
    session.headers.update({"Accept": "application/json"})

    items = fetch_all_pages(
        session=session,
        base_url=dur_cfg["base_url"],
        path=path,
        endpoint_name=endpoint_key,
        api_key=api_key,
        num_of_rows=dur_cfg.get("max_num_of_rows", 1000),
        timeout=dur_cfg.get("request_timeout", 30),
        retry_count=dur_cfg.get("retry_count", 3),
        retry_delay=dur_cfg.get("retry_delay", 1.0),
        rate_delay=dur_cfg.get("rate_limit_delay", 0.2),
    )

    if not items:
        return None

    df = pd.json_normalize(items)

    # 출력 파일명 결정
    out_path_key = _endpoint_to_output_key(endpoint_key)
    out_path = Path(config["output"].get(out_path_key, f"data/dur/{endpoint_key}.parquet"))
    out_path.parent.mkdir(parents=True, exist_ok=True)

    df.to_parquet(out_path, index=False)
    print(f"  저장: {out_path}  ({df.shape[0]:,} rows × {df.shape[1]} cols)")
    return df


def _endpoint_to_output_key(endpoint_key: str) -> str:
    mapping = {
        "usjnt_taboo": "usjnt_taboo",
        "efcy_dplct": "efcy_dplct",
        "odsn_atent": "odsn_atent",
        "spcify_agrde_taboo": "spcify_agrde_taboo",
        "pwnm_taboo": "pwnm_taboo",
        "cpcty_atent": "cpcty_atent",
        "mdctn_pd_atent": "mdctn_pd_atent",
    }
    return mapping.get(endpoint_key, endpoint_key)


def post_process_usjnt_taboo(df: pd.DataFrame) -> pd.DataFrame:
    """병용금기 데이터 표준화 → DDI 매트릭스 형식으로 변환."""
    # 필드명 통일 (API 응답 필드명이 버전에 따라 다를 수 있음)
    rename_map = {
        "INGR_CODE": "drug_a_code",              # 성분코드 A
        "INGR_ENG_NAME": "drug_a_name",          # 영문성분명 A
        "INGR_KOR_NAME": "drug_a_name_kr",       # 한글성분명 A
        "MIXTURE_INGR_CODE": "drug_b_code",      # 성분코드 B
        "MIXTURE_INGR_ENG_NAME": "drug_b_name",  # 영문성분명 B
        "MIXTURE_INGR_KOR_NAME": "drug_b_name_kr", # 한글성분명 B
        "PROHBT_CONTENT": "prohibition_detail",
        "NOTIFICATION_DATE": "notification_date",
    }
    # 소문자 필드명도 처리
    rename_map.update({k.lower(): v for k, v in rename_map.items()})
    df = df.rename(columns={c: rename_map[c] for c in df.columns if c in rename_map})

    df["severity"] = "Contraindicated"   # 병용금기 = Contraindicated
    df["source"] = "HIRA_DUR"
    return df


def post_process_efcy_dplct(df: pd.DataFrame) -> pd.DataFrame:
    """효능군중복 데이터 표준화."""
    rename_map = {
        "INGR_CODE": "drug_code",
        "INGR_ENG_NAME": "drug_name",
        "INGR_NAME": "drug_name_kr",
        "EFFECT_CODE": "efcy_class_no",    # 효능군 분류 번호
        "SERS_NAME": "efcy_class_name",    # 효능군 이름
        "NOTIFICATION_DATE": "notification_date",
    }
    rename_map.update({k.lower(): v for k, v in rename_map.items()})
    df = df.rename(columns={c: rename_map[c] for c in df.columns if c in rename_map})
    df["source"] = "HIRA_DUR"
    return df


def print_summary(results: dict):
    """수집 결과 요약 출력."""
    print("\n" + "=" * 60)
    print("[수집 결과 요약]")
    print("=" * 60)
    for ep, df in results.items():
        if df is not None:
            print(f"  {ep:30s}: {len(df):>8,} 건")
        else:
            print(f"  {ep:30s}: 실패 또는 데이터 없음")
    print("=" * 60)
    print("\n[완료] 수집된 데이터를 폐쇄망으로 반입 후 build_ddi_matrix.py 실행하세요.")


def main():
    parser = argparse.ArgumentParser(description="식약처 DUR OpenAPI 데이터 수집기")
    parser.add_argument("--config", default="config/api_config.yaml", help="설정 파일 경로")
    parser.add_argument(
        "--endpoints",
        nargs="+",
        default=None,
        help="수집할 엔드포인트 목록 (기본: 전체). 예: usjnt_taboo efcy_dplct",
    )
    parser.add_argument("--out", default=None, help="출력 디렉토리 override")
    args = parser.parse_args()

    if not Path(args.config).exists():
        print(f"[오류] 설정 파일 없음: {args.config}")
        sys.exit(1)

    config = load_config(args.config)

    if args.out:
        # override output paths
        out_dir = Path(args.out)
        for key in config["output"]:
            filename = Path(config["output"][key]).name
            config["output"][key] = str(out_dir / filename)

    api_key = get_api_key(config)
    all_endpoints = list(config["dur_api"]["endpoints"].keys())

    # 우선순위 순 정렬
    all_endpoints.sort(key=lambda k: config["dur_api"]["endpoints"][k].get("priority", 99))

    endpoints_to_fetch = args.endpoints if args.endpoints else all_endpoints
    out_dir = Path(config["output"].get("base_dir", "data/dur"))

    print(f"[설정] Base URL: {config['dur_api']['base_url']}")
    print(f"[설정] 수집 엔드포인트: {endpoints_to_fetch}")
    print(f"[설정] 출력 디렉토리: {out_dir}")

    results = {}
    for ep_key in endpoints_to_fetch:
        df = fetch_endpoint(config, ep_key, api_key, out_dir)
        results[ep_key] = df

        # 병용금기 데이터 후처리
        if ep_key == "usjnt_taboo" and df is not None:
            df_std = post_process_usjnt_taboo(df.copy())
            std_path = out_dir / "dur_ddi_contraindicated_std.parquet"
            df_std.to_parquet(std_path, index=False)
            print(f"  표준화 저장: {std_path}")

        # 효능군중복 후처리
        if ep_key == "efcy_dplct" and df is not None:
            df_std = post_process_efcy_dplct(df.copy())
            std_path = out_dir / "dur_therapeutic_duplicate_std.parquet"
            df_std.to_parquet(std_path, index=False)
            print(f"  표준화 저장: {std_path}")

    print_summary(results)


if __name__ == "__main__":
    main()
