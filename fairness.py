# fairness.py
# -*- coding: utf-8 -*-
"""
SciPy/Statsmodels 없이 동작하는 통계 유틸
- 균일성(카이제곱) 검정: Wilson–Hilferty 정규근사로 p-value 근사
- 쌍 공출현 유의성(이항 상향 단측): 정규근사(+연속성 보정)로 p-value 근사
- Benjamini–Hochberg FDR 보정: 순수 NumPy 구현
주의: 근사치 기반이라 극단적인 꼬리에서 p-value 정확도가 다소 떨어질 수 있음.
"""

from __future__ import annotations
import numpy as np
import pandas as pd
from math import comb, sqrt, log, erf

# -------- 공통: 표준정규 CDF --------
def _phi(z: float) -> float:
    """표준정규 CDF Φ(z). math.erf 사용."""
    return 0.5 * (1.0 + erf(z / sqrt(2.0)))

# -------- 1) 카이제곱 균일성 검정 (근사) --------
def chi_square_uniform(freq: pd.Series) -> dict:
    """
    번호별 관측도수(freq)가 균일분포(각 번호 동일 기대도수)와
    유의하게 다른지 카이제곱 검정 (정규근사 p-value).
    """
    obs = freq.sort_index().to_numpy(dtype=float)
    k = len(obs)                 # 카테고리 수 (보통 45)
    n = float(obs.sum())
    if n <= 0 or k <= 1:
        return {"stat": float("nan"), "pvalue": float("nan"), "total": int(n)}

    exp = n / k
    # 카이제곱 통계량
    with np.errstate(divide='ignore', invalid='ignore'):
        stat = float(np.nansum((obs - exp) ** 2 / exp))

    # Wilson-Hilferty 변환으로 상측 p-value 근사
    # Z = ((X/df)^(1/3) - (1 - 2/(9df))) / sqrt(2/(9df)) ~ N(0,1)
    df = k - 1
    if df <= 0:
        return {"stat": stat, "pvalue": float("nan"), "total": int(n)}

    y = (stat / df) ** (1.0 / 3.0) if stat > 0 else 0.0
    z = (y - (1 - 2.0 / (9.0 * df))) / sqrt(2.0 / (9.0 * df))
    # 상측 확률: P(Chi2 >= stat) ≈ 1 - Φ(z)
    pval = max(0.0, min(1.0, 1.0 - _phi(z)))
    return {"stat": stat, "pvalue": pval, "total": int(n)}

# -------- 2) 쌍 공출현 유의성 (Binomial 상향 단측, 근사) --------
def _binom_sf_normal_approx(k: int, n: int, p: float) -> float:
    """
    Binomial(n, p)에서 P(X >= k) 근사 (정규근사 + 연속성 보정).
    n이 충분히 크고 np(1-p)도 충분하면 사용 권장.
    """
    if n <= 0:
        return 1.0 if k <= 0 else 0.0
    mu = n * p
    var = n * p * (1.0 - p)
    if var <= 0:
        # p == 0 또는 1인 극단
        return 1.0 if (p == 0.0 and k <= 0) or (p == 1.0 and k <= n) else 0.0
    # 연속성 보정: k -> k - 0.5
    z = ((k - 0.5) - mu) / sqrt(var)
    # 상측 확률
    sf = 1.0 - _phi(z)
    # 수치 안정화
    return max(0.0, min(1.0, sf))

def pair_significance_binomial(
    co_mat: pd.DataFrame,
    n_draws: int,
    include_bonus: bool = False,
    alpha: float = 0.05,
) -> pd.DataFrame:
    """
    공출현 횟수(co_count)가 무작위 가정하에서 과대표현인지(상향 단측) 근사 검정.
    - 각 회차 m개 추출: m=6(보너스 제외) / m=7(보너스 포함)
    - 쌍 동시 등장 확률 p_pair = C(45-2, m-2) / C(45, m)
    - X ~ Binomial(n_draws, p_pair), p-value ≈ Normal Approx (연속성 보정)
    - 다중검정: Benjamini–Hochberg FDR 보정
    """
    m = 7 if include_bonus else 6
    p_pair = comb(45 - 2, m - 2) / comb(45, m)

    # 테이블 전개
    labels = list(map(int, co_mat.index))
    rows = []
    for i in range(len(labels)):
        for j in range(i + 1, len(labels)):
            a, b = labels[i], labels[j]
            co = int(co_mat.loc[str(a), str(b)])
            exp = n_draws * p_pair
            pval = _binom_sf_normal_approx(co, n_draws, p_pair)
            lift = co / exp if exp > 0 else np.nan
            rows.append((a, b, co, exp, pval, lift))

    df = pd.DataFrame(rows, columns=["num_a", "num_b", "co_count", "expected", "pvalue", "lift"])

    # Benjamini–Hochberg FDR (q-value) 구현
    if len(df) > 0:
        p = df["pvalue"].to_numpy()
        mtests = float(len(p))
        order = np.argsort(p)
        ranked = p[order]
        # 누적 최소화로 q 계산
        q = ranked * mtests / (np.arange(1, len(ranked) + 1))
        # 뒤에서부터 누적 최소
        for k in range(len(q) - 2, -1, -1):
            q[k] = min(q[k], q[k + 1])
        qvals = np.empty_like(q)
        qvals[order] = q
        df["qvalue"] = qvals
        df["significant"] = df["qvalue"] <= alpha
    else:
        df["qvalue"] = []
        df["significant"] = []

    df = df.sort_values(["significant", "qvalue", "lift", "co_count"],
                        ascending=[False, True, False, False]).reset_index(drop=True)
    return df
