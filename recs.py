# recs.py
# -*- coding: utf-8 -*-
from __future__ import annotations
import random
from collections import Counter
from typing import List, Dict
import numpy as np, pandas as pd

def _pick_k(sorted_nums: List[int], k: int = 6) -> List[int]:
    uniq = list(dict.fromkeys(sorted_nums))
    if len(uniq) >= k:
        return sorted(uniq[:k])
    rest = [n for n in range(1,46) if n not in uniq]
    return sorted(uniq + random.sample(rest, k - len(uniq)))

def recommend_hot(freq: pd.Series, k: int = 6) -> List[int]:
    return _pick_k(freq.sort_values(ascending=False).index.tolist(), k)

def recommend_cold(freq: pd.Series, k: int = 6) -> List[int]:
    return _pick_k(freq.sort_values(ascending=True).index.tolist(), k)

def recommend_balanced(freq: pd.Series, k: int = 6) -> List[int]:
    nums_sorted = freq.sort_values(ascending=False).index.tolist()
    pick: List[int] = []
    target_odd, target_low = 3, 3
    def ok_add(n: int) -> bool:
        odd = sum(x%2 for x in pick) + (n%2)
        low = sum(1 for x in pick if x<=22) + (1 if n<=22 else 0)
        return odd <= target_odd and low <= target_low
    for n in nums_sorted:
        if n in pick: continue
        if ok_add(n): pick.append(n)
        if len(pick) == k: break
    if len(pick) < k:
        for n in nums_sorted:
            if n not in pick:
                pick.append(n)
                if len(pick) == k: break
    return sorted(pick[:k])

def recommend_weighted_recent(df: pd.DataFrame, lookback: int = 200, k: int = 6,
                              include_bonus: bool=False, seed: int = 42) -> List[int]:
    rng = np.random.default_rng(seed)
    hist = df.tail(lookback)
    cols = ["n1","n2","n3","n4","n5","n6"] + (["bonus"] if include_bonus else [])
    w = Counter()
    for _, r in hist.iterrows():
        for c in cols: w[int(r[c])] += 1
    weights = np.array([w[n] for n in range(1,46)], dtype=float)
    weights = (weights + 1.0)
    probs = weights / weights.sum()
    picks = rng.choice(np.arange(1,46), size=45, replace=False, p=probs)
    return sorted(list(picks[:k]))

def composition_metrics(nums: List[int]) -> Dict[str, int|bool|list]:
    nums = sorted(nums)
    return {
        "sum": sum(nums),
        "range": nums[-1] - nums[0],
        "odd": sum(n%2 for n in nums),
        "low": sum(1 for n in nums if n <= 22),
        "consecutive": int(any(b-a==1 for a,b in zip(nums, nums[1:]))),
        "last_digits": [n%10 for n in nums],
    }

def bonus_candidates(df: pd.DataFrame, lookback: int = 200, topk: int = 5) -> List[int]:
    hist = df.tail(lookback)
    s = hist["bonus"].value_counts().sort_values(ascending=False)
    return s.index.astype(int).tolist()[:topk]
