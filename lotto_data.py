# lotto_data.py
# -*- coding: utf-8 -*-
from __future__ import annotations

import json
import os
import re
import shutil
import tempfile
import time
from typing import Dict, Optional, Tuple

import numpy as np
import pandas as pd
import requests
from tqdm import tqdm

# Official draw JSON endpoint (works even when Content-Type is text/html).
# Some environments are more reliable when parameters are ordered as drwNo first.
BASE_URLS = [
    "https://www.dhlottery.co.kr/common.do?drwNo={drwNo}&method=getLottoNumber",
    "https://www.dhlottery.co.kr/common.do?method=getLottoNumber&drwNo={drwNo}",
]

RESULT_PAGE_URL = "https://www.dhlottery.co.kr/lt645/result"

BROWSER_HEADERS = {
    "User-Agent": (
        "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 "
        "(KHTML, like Gecko) Chrome/121.0 Safari/537.36"
    ),
    "Accept": "application/json, text/javascript, */*; q=0.01",
    "Accept-Language": "ko-KR,ko;q=0.9,en;q=0.8",
    "Referer": RESULT_PAGE_URL,
    "X-Requested-With": "XMLHttpRequest",
    "Connection": "keep-alive",
}

EMPTY_DF_COLUMNS = ["draw_no", "date", "n1", "n2", "n3", "n4", "n5", "n6", "bonus"]


def _bootstrap_session(session: requests.Session) -> None:
    # 쿠키/세션 확보(환경에 따라 API 직접 호출이 홈으로 튕기는 경우 완화)
    session.get("https://www.dhlottery.co.kr/", headers=BROWSER_HEADERS, timeout=10)
    session.get(RESULT_PAGE_URL, headers=BROWSER_HEADERS, timeout=10)


def _parse_json_forgiving(text: str) -> Optional[Dict]:
    t = (text or "").strip()
    if not t:
        return None
    # Sometimes server returns JSON with text/html content-type.
    if t[0] == "{":
        try:
            return json.loads(t)
        except Exception:
            return None
    return None


def _fetch_draw_json(drw_no: int, session: requests.Session) -> Optional[Dict]:
    """Fetch one draw JSON.

    Returns:
        dict if returnValue == success else None.

    Notes:
        - Detects redirects / non-JSON responses (HTML) and returns None.
        - Tries two URL parameter orders to maximize compatibility.
    """
    try:
        if not getattr(session, "_bootstrapped", False):
            _bootstrap_session(session)
            session._bootstrapped = True

        for tpl in BASE_URLS:
            url = tpl.format(drwNo=drw_no)
            # allow_redirects=False: detect "홈으로 튕김(3xx)" explicitly.
            r = session.get(url, headers=BROWSER_HEADERS, timeout=10, allow_redirects=False)

            if 300 <= r.status_code < 400:
                continue  # redirected -> likely blocked in this environment

            if r.status_code != 200:
                continue

            # Prefer real JSON parsing, but fall back to text parsing (Content-Type may lie)
            data = None
            try:
                data = r.json()
            except Exception:
                data = _parse_json_forgiving(r.text)

            if not isinstance(data, dict):
                continue

            if data.get("returnValue") == "success":
                return data

        return None
    except Exception:
        return None


def _latest_draw_from_result_page(session: requests.Session) -> Optional[int]:
    """Parse latest draw number from the official result page."""
    try:
        r = session.get(RESULT_PAGE_URL, headers=BROWSER_HEADERS, timeout=10)
        if r.status_code != 200:
            return None
        # The page contains a dropdown like: <option>1207회</option>
        nums = [int(x) for x in re.findall(r"(\d{1,5})회", r.text)]
        return max(nums) if nums else None
    except Exception:
        return None


def find_latest_draw(session: requests.Session, start_guess: int = 1600) -> int:
    """Find latest draw number.

    Strategy:
      1) Parse from official result page (fast & robust).
      2) Fallback: probe API by stepping down then up (older logic).
    """
    # 1) robust parse from /lt645/result
    latest = _latest_draw_from_result_page(session)
    if isinstance(latest, int) and latest > 0:
        return latest

    # 2) fallback probing (may be slow if blocked)
    hi = start_guess
    while hi > 1 and _fetch_draw_json(hi, session) is None:
        hi -= 10
    if hi < 1:
        hi = 1
    while _fetch_draw_json(hi + 1, session) is not None:
        hi += 1
    return hi


def collect_range(session: requests.Session, start_no: int, end_no: int) -> pd.DataFrame:
    rows = []
    for n in tqdm(range(start_no, end_no + 1), desc=f"Collect {start_no}-{end_no}"):
        data = _fetch_draw_json(n, session)
        if data is None:
            continue
        nums = [data.get(f"drwtNo{i}") for i in range(1, 7)]
        bonus = data.get("bnusNo")
        date = data.get("drwNoDate")
        rows.append(
            {
                "draw_no": n,
                "date": date,
                "n1": nums[0],
                "n2": nums[1],
                "n3": nums[2],
                "n4": nums[3],
                "n5": nums[4],
                "n6": nums[5],
                "bonus": bonus,
            }
        )
        time.sleep(0.01)

    # 수집 0건이면 컬럼 포함 빈 DF 반환 (KeyError 방지)
    if not rows:
        return pd.DataFrame(columns=EMPTY_DF_COLUMNS)

    return pd.DataFrame(rows).sort_values("draw_no").reset_index(drop=True)


def load_csv(csv_path: str) -> pd.DataFrame:
    if not os.path.exists(csv_path):
        os.makedirs(os.path.dirname(csv_path) or ".", exist_ok=True)
        return pd.DataFrame(columns=EMPTY_DF_COLUMNS)
    return pd.read_csv(csv_path, dtype={"draw_no": int, "date": str, "bonus": int})


def _atomic_save_csv(df: pd.DataFrame, csv_path: str):
    os.makedirs(os.path.dirname(csv_path) or ".", exist_ok=True)
    fd, tmp_path = tempfile.mkstemp(prefix="lotto_", suffix=".csv", dir=os.path.dirname(csv_path) or ".")
    os.close(fd)
    df.to_csv(tmp_path, index=False, encoding="utf-8-sig")
    if os.path.exists(csv_path):
        os.remove(csv_path)
    shutil.move(tmp_path, csv_path)


def _dedupe_sort(df: pd.DataFrame) -> pd.DataFrame:
    # CSV 비정상/컬럼 누락에도 안전 처리
    if df is None:
        return pd.DataFrame(columns=EMPTY_DF_COLUMNS)

    if "draw_no" not in df.columns:
        if "drwNo" in df.columns:
            df = df.rename(columns={"drwNo": "draw_no"})
        else:
            return pd.DataFrame(columns=EMPTY_DF_COLUMNS)

    df = df.drop_duplicates(subset=["draw_no"]).sort_values("draw_no").reset_index(drop=True)
    for c in ["n1", "n2", "n3", "n4", "n5", "n6", "bonus"]:
        if c in df.columns:
            df[c] = pd.to_numeric(df[c], errors="coerce").astype("Int64")
    return df


def incremental_update(csv_path: str) -> Tuple[pd.DataFrame, int, int]:
    with requests.Session() as session:
        latest = find_latest_draw(session)
        existing = _dedupe_sort(load_csv(csv_path))
        cur_max = int(existing["draw_no"].max()) if not existing.empty else 0
        if cur_max >= latest:
            return existing, cur_max, latest

        new_df = collect_range(session, cur_max + 1, latest)

        # 최신은 더 있는데 신규 수집이 0건이면 네트워크/차단 가능성 → 명확한 에러
        if new_df.empty and latest > cur_max:
            raise RuntimeError(
                f"동행복권 API 수집 실패: {cur_max + 1}~{latest} 구간 0건 "
                f"(환경 차단/리다이렉트/비정상 응답 가능). "
                f"브라우저에서 URL이 JSON으로 열리는지 확인하거나, 서버/네트워크(IP) 변경을 시도하세요."
            )

        merged = _dedupe_sort(pd.concat([existing, new_df], ignore_index=True))
        _atomic_save_csv(merged, csv_path)
        return merged, cur_max, latest


# ---- 분석 유틸 ----
def frequency(df: pd.DataFrame, include_bonus: bool) -> pd.Series:
    cols = ["n1", "n2", "n3", "n4", "n5", "n6"] + (["bonus"] if include_bonus else [])
    s = pd.Series(0, index=range(1, 46), dtype=int)
    for c in cols:
        s = s.add(df[c].value_counts().reindex(range(1, 46), fill_value=0), fill_value=0).astype(int)
    s.index.name = "number"
    s.name = "count"
    return s


def presence_matrix(df: pd.DataFrame, include_bonus: bool) -> pd.DataFrame:
    cols = ["n1", "n2", "n3", "n4", "n5", "n6"] + (["bonus"] if include_bonus else [])
    mat = np.zeros((len(df), 45), dtype=int)
    for i, row in df.iterrows():
        present = set(int(row[c]) for c in cols)
        for num in present:
            if 1 <= num <= 45:
                mat[i, num - 1] = 1
    out = pd.DataFrame(mat, columns=[str(i) for i in range(1, 46)])
    out.insert(0, "draw_no", df["draw_no"].values)
    out.insert(1, "date", df["date"].values)
    return out


def cooccurrence(only_num: pd.DataFrame) -> pd.DataFrame:
    X = only_num.values
    co = X.T @ X
    np.fill_diagonal(co, 0)
    return pd.DataFrame(co, index=only_num.columns, columns=only_num.columns, dtype=int)
