# rolling.py
# -*- coding: utf-8 -*-
import pandas as pd

def rolling_frequency(df: pd.DataFrame, window: int = 100, include_bonus: bool = False) -> pd.DataFrame:
    cols = ["n1","n2","n3","n4","n5","n6"] + (["bonus"] if include_bonus else [])
    idx = pd.Index(range(1,46), name="number")
    out = []
    for i in range(len(df)):
        lo = max(0, i - window + 1)
        sub = df.iloc[lo:i+1]
        s = pd.Series(0, index=idx)
        for c in cols:
            s = s.add(sub[c].value_counts().reindex(idx, fill_value=0), fill_value=0)
        out.append(s)
    R = pd.DataFrame(out, index=df["draw_no"].values)
    return R
