# lotto_data.py
# -*- coding: utf-8 -*-
from __future__ import annotations
import os, time, tempfile, shutil
from typing import Dict, Optional, Tuple
import requests, pandas as pd, numpy as np
from tqdm import tqdm

BASE_URL = "https://www.dhlottery.co.kr/common.do?method=getLottoNumber&drwNo={drwNo}"
HEADERS = {"User-Agent": "Mozilla/5.0 (LottoAnalyzer/Pro 3.0)"}

def _fetch_draw_json(drw_no: int, session: requests.Session) -> Optional[Dict]:
    try:
        r = session.get(BASE_URL.format(drwNo=drw_no), headers=HEADERS, timeout=8)
        r.raise_for_status()
        data = r.json()
        if data.get("returnValue") == "success":
            return data
        return None
    except Exception:
        return None

# lotto_data.py
def find_latest_draw(session: requests.Session, start_guess: int = 1600) -> int:
    # 1) 큰 값에서 내려오며 '존재하는 회차(앵커)' 하나를 찾는다.
    hi = start_guess
    while hi > 1 and _fetch_draw_json(hi, session) is None:
        hi -= 10  # 블록 단위 백오프 (10은 임의, 50 등으로 바꿔도 됨)

    if hi < 1:
        hi = 1

    # 2) 앵커에서 위로 전개: 존재하는 동안 +1씩 올려 '진짜 최신'까지 간다.
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
        rows.append({
            "draw_no": n, "date": date,
            "n1": nums[0], "n2": nums[1], "n3": nums[2],
            "n4": nums[3], "n5": nums[4], "n6": nums[5],
            "bonus": bonus
        })
        time.sleep(0.01)
    df = pd.DataFrame(rows).sort_values("draw_no").reset_index(drop=True)
    return df

def load_csv(csv_path: str) -> pd.DataFrame:
    if not os.path.exists(csv_path):
        os.makedirs(os.path.dirname(csv_path) or ".", exist_ok=True)
        return pd.DataFrame(columns=["draw_no","date","n1","n2","n3","n4","n5","n6","bonus"])
    return pd.read_csv(csv_path, dtype={"draw_no":int, "date":str, "bonus":int})

def _atomic_save_csv(df: pd.DataFrame, csv_path: str):
    os.makedirs(os.path.dirname(csv_path) or ".", exist_ok=True)
    fd, tmp_path = tempfile.mkstemp(prefix="lotto_", suffix=".csv", dir=os.path.dirname(csv_path) or ".")
    os.close(fd)
    df.to_csv(tmp_path, index=False, encoding="utf-8-sig")
    if os.path.exists(csv_path):
        os.remove(csv_path)
    shutil.move(tmp_path, csv_path)

def _dedupe_sort(df: pd.DataFrame) -> pd.DataFrame:
    df = df.drop_duplicates(subset=["draw_no"]).sort_values("draw_no").reset_index(drop=True)
    for c in ["n1","n2","n3","n4","n5","n6","bonus"]:
        if c in df.columns:
            df[c] = pd.to_numeric(df[c], errors="coerce").astype("Int64")
    return df

def incremental_update(csv_path: str) -> Tuple[pd.DataFrame, int, int]:
    with requests.Session() as session:
        latest = find_latest_draw(session)
        existing = load_csv(csv_path)
        existing = _dedupe_sort(existing)
        cur_max = int(existing["draw_no"].max()) if not existing.empty else 0
        if cur_max >= latest:
            return existing, cur_max, latest
        new_df = collect_range(session, cur_max + 1, latest)
        merged = pd.concat([existing, new_df], ignore_index=True)
        merged = _dedupe_sort(merged)
        _atomic_save_csv(merged, csv_path)
        return merged, cur_max, latest

# ---- 분석 유틸 ----
def frequency(df: pd.DataFrame, include_bonus: bool) -> pd.Series:
    cols = ["n1","n2","n3","n4","n5","n6"] + (["bonus"] if include_bonus else [])
    s = pd.Series(0, index=range(1,46), dtype=int)
    for c in cols:
        s = s.add(df[c].value_counts().reindex(range(1,46), fill_value=0), fill_value=0).astype(int)
    s.index.name = "number"; s.name = "count"
    return s

def presence_matrix(df: pd.DataFrame, include_bonus: bool) -> pd.DataFrame:
    cols = ["n1","n2","n3","n4","n5","n6"] + (["bonus"] if include_bonus else [])
    import numpy as np
    mat = np.zeros((len(df), 45), dtype=int)
    for i, row in df.iterrows():
        present = set(int(row[c]) for c in cols)
        for num in present:
            if 1 <= num <= 45:
                mat[i, num-1] = 1
    out = pd.DataFrame(mat, columns=[str(i) for i in range(1,46)])
    out.insert(0, "draw_no", df["draw_no"].values)
    out.insert(1, "date", df["date"].values)
    return out

def cooccurrence(only_num: pd.DataFrame) -> pd.DataFrame:
    X = only_num.values
    co = X.T @ X
    np = __import__("numpy")
    np.fill_diagonal(co, 0)
    return pd.DataFrame(co, index=only_num.columns, columns=only_num.columns, dtype=int)
