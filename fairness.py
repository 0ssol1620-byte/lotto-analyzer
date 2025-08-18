# fairness.py
# -*- coding: utf-8 -*-
from __future__ import annotations
from math import comb
import numpy as np, pandas as pd
from scipy.stats import chisquare, binom
from statsmodels.stats.multitest import multipletests

def chi_square_uniform(freq: pd.Series) -> dict:
    observed = freq.sort_index().values
    expected = np.full_like(observed, observed.sum() / 45.0, dtype=float)
    stat, p = chisquare(observed, f_exp=expected)
    return {"stat": float(stat), "pvalue": float(p), "total": int(observed.sum())}

def pair_significance_binomial(co_mat: pd.DataFrame, n_draws: int,
                               include_bonus: bool=False, alpha: float=0.05) -> pd.DataFrame:
    m = 7 if include_bonus else 6
    p_pair = comb(45-2, m-2) / comb(45, m)
    rows = []
    labels = list(map(int, co_mat.index))
    for i in range(len(labels)):
        for j in range(i+1, len(labels)):
            a, b = labels[i], labels[j]
            co = int(co_mat.loc[str(a), str(b)])
            exp = n_draws * p_pair
            pval = float(binom.sf(co-1, n_draws, p_pair))
            lift = co / exp if exp > 0 else np.nan
            rows.append((a,b,co,exp,pval,lift))
    df = pd.DataFrame(rows, columns=["num_a","num_b","co_count","expected","pvalue","lift"])
    rej, qvals, _, _ = multipletests(df["pvalue"].values, alpha=alpha, method="fdr_bh")
    df["qvalue"] = qvals; df["significant"] = rej
    df = df.sort_values(["significant","qvalue","lift","co_count"], ascending=[False, True, False, False]).reset_index(drop=True)
    return df
