# viz.py
# -*- coding: utf-8 -*-
from __future__ import annotations
import numpy as np
import pandas as pd
import streamlit as st
import plotly.graph_objects as go

PRIMARY = "#1F3A8A"

# -------- 공통 스타일 --------
def apply_global_style():
    st.markdown("""
    <style>
      :root { --pro-font: "Noto Sans KR","Segoe UI",system-ui,-apple-system,sans-serif; }
      html, body, [class^="css"] { font-family: var(--pro-font); }
      .metric-pro {
        border: 1px solid #263043; border-radius: 16px; padding: 14px 16px;
        background: #0F172A; box-shadow: 0 2px 6px rgba(0,0,0,.25);
      }
      .metric-title { color: #94A3B8; font-size: 12.5px; margin-bottom: 6px; }
      .metric-value { color: #E5E7EB; font-size: 22px; line-height: 1.25; word-break: keep-all; }
      .metric-subtle { color:#9CA3AF; font-size:12px; }
    </style>
    """, unsafe_allow_html=True)

def kpi_card(title: str, value: str, sub: str | None = None):
    with st.container():
        st.markdown(f"""
        <div class="metric-pro">
          <div class="metric-title">{title}</div>
          <div class="metric-value">{value}</div>
          {f'<div class="metric-subtle">{sub}</div>' if sub else ''}
        </div>
        """, unsafe_allow_html=True)

def _scale(h: int, compact: bool) -> int:
    return int(h * (0.72 if compact else 1.0))

# -------- Top-N 빈도 (세로/가로) --------
def make_top_frequency_horizontal(freq_series: pd.Series, topn: int, title: str, compact=False) -> go.Figure:
    s_desc = freq_series.sort_values(ascending=False).head(topn)
    top5 = set(s_desc.head(5).index)
    s = s_desc[::-1]
    y = [f"{i:02d}" for i in s.index]
    x = s.values
    colors = ["#3B82F6" if int(lbl) in top5 else "#60A5FA" for lbl in y]
    height = _scale(max(360, 28 * len(s) + 140), compact)
    xmax = float(max(x)) * 1.08
    fig = go.Figure(go.Bar(
        x=x, y=y, orientation="h",
        text=[f"{v:,}" for v in x], textposition="outside", cliponaxis=False,
        marker=dict(color=colors),
        hovertemplate="번호 %{y}<br>빈도 %{x:,}회<extra></extra>",
        showlegend=False
    ))
    fig.update_layout(
        title=title, height=height, margin=dict(l=10,r=20,t=50,b=12),
        template="plotly_dark", paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
        title_font=dict(size=18, color="#E5E7EB"),
        xaxis=dict(title="", range=[0, xmax], showgrid=True, gridcolor="rgba(148,163,184,.22)", zeroline=False, tickfont=dict(size=11)),
        yaxis=dict(title="", showgrid=False, tickfont=dict(size=11), categoryorder="array", categoryarray=y),
        bargap=0.18, bargroupgap=0.06
    )
    return fig

def make_top_frequency_vertical(freq_series: pd.Series, topn: int, title: str, compact=False) -> go.Figure:
    s = freq_series.sort_values(ascending=False).head(topn)
    x = [f"{i:02d}" for i in s.index]
    y = s.values
    colors = ["#60A5FA"] * len(x)
    for i in range(min(5, len(x))):
        colors[i] = "#3B82F6"
    height = _scale(340, compact)
    ymax = float(max(y)) * 1.10
    fig = go.Figure(go.Bar(
        x=x, y=y, orientation="v",
        text=[f"{v:,}" for v in y], textposition="outside", cliponaxis=False,
        marker=dict(color=colors),
        hovertemplate="번호 %{x}<br>빈도 %{y:,}회<extra></extra>",
        showlegend=False
    ))
    fig.update_layout(
        title=title, height=height, margin=dict(l=10,r=10,t=50,b=10),
        template="plotly_dark", paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
        title_font=dict(size=18, color="#E5E7EB"),
        xaxis=dict(title="", tickfont=dict(size=11)),
        yaxis=dict(title="", range=[0, ymax], showgrid=True, gridcolor="rgba(148,163,184,.22)", zeroline=False, tickfont=dict(size=11)),
        bargap=0.35, bargroupgap=0.08
    )
    return fig

# -------- 일반 히트맵 --------
def make_heatmap(matrix: pd.DataFrame, title: str, zmin=None, zmax=None,
                 colorscale="YlGnBu", compact: bool=False, height: int | None=None) -> go.Figure:
    xlabels = [f"{i:02d}" for i in range(1, matrix.shape[1]+1)]
    ylabels = [f"{i:02d}" for i in range(1, matrix.shape[0]+1)]
    h = height if height is not None else _scale(520, compact)
    fig = go.Figure(go.Heatmap(
        z=matrix.values, x=xlabels, y=ylabels,
        colorscale=colorscale, zmin=zmin, zmax=zmax,
        colorbar=dict(thickness=10, outlinewidth=0, ticks="outside", tickcolor="#94A3B8")
    ))
    fig.update_layout(
        title=title, height=h, margin=dict(l=8,r=8,t=46,b=8),
        template="plotly_dark", paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
        title_font=dict(size=18, color="#E5E7EB"),
        xaxis=dict(showgrid=False, tickfont=dict(size=9)),
        yaxis=dict(showgrid=False, tickfont=dict(size=9), autorange="reversed")
    )
    return fig

def make_corr_heatmap(corr: pd.DataFrame, title="Correlation Heatmap (Pearson)",
                      compact: bool=False, height: int | None=None) -> go.Figure:
    return make_heatmap(corr, title=title, zmin=-1, zmax=1, colorscale="RdBu",
                        compact=compact, height=height)

# -------- PRO: 상관 히트맵 가시성 강화 --------
def _reorder_corr(corr: pd.DataFrame, by_abs: bool=True) -> pd.DataFrame:
    """
    계층 클러스터링으로 변수 순서를 재정렬해 블록 구조를 드러냄.
    거리 = 1 - |r|  (by_abs=True), 아니면 1 - r를 사용.
    """
    try:
        from scipy.cluster.hierarchy import linkage, leaves_list
        from scipy.spatial.distance import squareform
        M = corr.abs().values if by_abs else corr.values
        D = 1.0 - np.clip(M, -1, 1)
        np.fill_diagonal(D, 0.0)
        Z = linkage(squareform(D, checks=False), method="average")
        order = leaves_list(Z)
        idx = corr.index.to_numpy()[order]
        return corr.loc[idx, idx]
    except Exception:
        # scipy 미사용/에러 시 원본 반환
        return corr

def make_corr_heatmap_pro(
    corr: pd.DataFrame,
    title: str = "Correlation Heatmap",
    abs_mode: bool = True,           # |r| 보기
    cluster: bool = True,            # 재정렬
    triangle: bool = True,           # 하삼각만 표시
    contrast: float = 0.25,          # 표시 범위(±r) 또는 [0, r] (abs_mode)
    compact: bool = False,
    height: int | None = None
) -> go.Figure:
    # 1) 변환
    M = corr.abs() if abs_mode else corr.copy()

    # 2) 재정렬
    if cluster:
        M = _reorder_corr(M, by_abs=abs_mode)

    # 3) 삼각 마스킹(상삼각 NaN)
    if triangle:
        mask = np.triu(np.ones_like(M.values, dtype=bool), k=1)
        Z = M.values.astype(float).copy()
        Z[mask] = np.nan
        M_plot = pd.DataFrame(Z, index=M.index, columns=M.columns)
    else:
        M_plot = M

    # 4) 색/범위 설정
    if abs_mode:
        zmin, zmax, colorscale = 0.0, float(contrast), "Viridis"
        t = f"{title} (|r|, ≤{contrast:.2f})"
    else:
        c = float(contrast)
        zmin, zmax, colorscale = -c, c, "RdBu"
        t = f"{title} (±{contrast:.2f})"

    # 5) 렌더링
    fig = make_heatmap(M_plot, title=t, zmin=zmin, zmax=zmax,
                       colorscale=colorscale, compact=compact, height=height)
    return fig

# viz.py — add this at the end

def make_top_pairs_vertical(top_df: pd.DataFrame, title: str, compact: bool=False) -> go.Figure:
    """
    Top 쌍(공출현) 세로 막대 차트
    - 입력 컬럼 호환: (num_a,num_b,co_count[,significant]) 또는 (A,B,co_count)
    - significant=True인 항목은 붉은색으로 하이라이트
    - compact=True면 높이를 축소해 요약 레이아웃에 적합
    """
    df = top_df.copy()

    if {"num_a", "num_b"}.issubset(df.columns):
        pair_label = df.apply(lambda r: f"{int(r['num_a']):02d}-{int(r['num_b']):02d}", axis=1)
        counts = df["co_count"].astype(int).values
        sig = df["significant"].values if "significant" in df.columns else [False] * len(df)
    else:
        # fallback: (A,B,co_count)
        pair_label = df.apply(lambda r: f"{int(r['A']):02d}-{int(r['B']):02d}", axis=1)
        counts = df["co_count"].astype(int).values
        sig = [False] * len(df)

    x = list(pair_label)
    y = list(counts)
    colors = ["#EF4444" if s else "#3B82F6" for s in sig]

    height = _scale(340, compact)
    ymax = float(max(y)) * 1.10 if len(y) else 1.0

    fig = go.Figure(go.Bar(
        x=x, y=y, orientation="v",
        text=[f"{v:,}" for v in y], textposition="outside", cliponaxis=False,
        marker=dict(color=colors),
        hovertemplate="쌍 %{x}<br>공출현 %{y:,}회<extra></extra>",
        showlegend=False
    ))
    fig.update_layout(
        title=title, height=height, margin=dict(l=10, r=10, t=50, b=10),
        template="plotly_dark",
        paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
        title_font=dict(size=18, color="#E5E7EB"),
        xaxis=dict(title="", tickfont=dict(size=11)),
        yaxis=dict(
            title="", range=[0, ymax], showgrid=True,
            gridcolor="rgba(148,163,184,.22)", zeroline=False, tickfont=dict(size=11)
        ),
        bargap=0.35, bargroupgap=0.08
    )
    return fig
