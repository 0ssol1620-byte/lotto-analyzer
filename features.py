# features.py
# -*- coding: utf-8 -*-
import pandas as pd
NUM_COLS = ["n1","n2","n3","n4","n5","n6"]

def _row_numbers(row):
    return [int(row[c]) for c in NUM_COLS]

def build_features(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    nums = df.apply(_row_numbers, axis=1)
    out["sum"] = nums.apply(sum)
    out["range"] = nums.apply(lambda x: max(x)-min(x))
    out["odd_cnt"] = nums.apply(lambda x: sum(n%2 for n in x))
    out["low_cnt"] = nums.apply(lambda x: sum(n<=22 for n in x))
    out["has_consecutive"] = nums.apply(lambda x: int(any(b-a==1 for a,b in zip(sorted(x), sorted(x)[1:]))))
    out["last_digit_mode"] = nums.apply(lambda x: max(range(10), key=lambda d: sum(1 for n in x if n%10==d)))
    return out

def last_digit_hist(df: pd.DataFrame) -> pd.Series:
    counts = {d:0 for d in range(10)}
    for _, r in df.iterrows():
        for c in NUM_COLS:
            counts[int(r[c])%10]+=1
    s = pd.Series(counts).sort_index()
    s.index.name="last_digit"; s.name="count"
    return s
