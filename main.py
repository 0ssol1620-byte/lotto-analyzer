# app.py
# -*- coding: utf-8 -*-
from __future__ import annotations
import os, io, numpy as np, pandas as pd, streamlit as st, plotly.express as px

from lotto_data import load_csv, incremental_update, frequency, presence_matrix, cooccurrence
from rolling import rolling_frequency
from recs import (
    recommend_hot, recommend_cold, recommend_balanced, recommend_weighted_recent,
    composition_metrics, bonus_candidates
)
from features import build_features, last_digit_hist
from fairness import chi_square_uniform, pair_significance_binomial
from viz import (
    apply_global_style, kpi_card,
    make_top_frequency_horizontal, make_top_frequency_vertical,
    make_heatmap, make_corr_heatmap_pro, make_top_pairs_vertical
)

PRIMARY = "#1F3A8A"

# -------- Page & Style --------
st.set_page_config(page_title="Lotto 6/45 Analyzer — Pro", page_icon="🎯", layout="wide")
apply_global_style()

# -------- Sidebar --------
with st.sidebar:
    st.header("⚙️ 옵션")
    include_bonus = st.toggle("보너스 포함", value=False, help="2등/흥미 비교용, 1등 분석은 보통 제외")
    topn = st.slider("Top N(빈도/쌍)", 5, 50, 25, 1)
    lookback = st.slider("최근 N회(가중·롤링)", 50, 500, 200, 10)
    compact = st.toggle("요약 콤팩트 모드", value=True, help="구성 탭의 핵심 요약 차트 높이를 축소")
    bar_dir = st.radio("Top 차트 방향", ["세로", "가로"], index=0, horizontal=True)

    st.divider()
    st.header("📦 데이터")
    data_path = st.text_input("CSV 경로", value="data/lotto_draws.csv")
    if st.button("데이터 갱신(증분)"):
        with st.spinner("최신 회차 확인 및 증분 수집 중..."):
            df_upd, prev, latest = incremental_update(data_path)
        st.session_state["df"] = df_upd
        st.success(f"완료: latest={latest}, 총 {len(df_upd):,}행 (이전 max={prev})")

# -------- Load data --------
if "df" not in st.session_state:
    os.makedirs(os.path.dirname(data_path) or ".", exist_ok=True)
    st.session_state["df"] = load_csv(data_path)
df = st.session_state["df"]
if df.empty:
    st.warning("CSV가 비어있습니다. 좌측 ‘데이터 갱신(증분)’을 눌러주세요.")
    st.stop()
latest = int(df["draw_no"].max())

# -------- KPI (잘림 없는 카드) --------
c1, c2, c3, c4 = st.columns([1.1, 1.1, 2.4, 1.1])
with c1: kpi_card("수집 회차", f"{len(df):,}")
with c2: kpi_card("최신 회차", f"{latest:,}")
with c3: kpi_card("기간", f"{df['date'].min()} → {df['date'].max()}")
with c4: kpi_card("보너스 포함", "Yes" if include_bonus else "No")

st.title("🎯 Lotto 6/45 Analyzer")

# -------- Core calculations --------
freq = frequency(df, include_bonus=include_bonus)
presence = presence_matrix(df, include_bonus=include_bonus)
only_num = presence[[str(i) for i in range(1,46)]]
co_df = cooccurrence(only_num)
corr = only_num.corr(method="pearson")

# -------- Tabs (요약 탭 제거, 추천 번호를 첫 탭으로) --------
tab_reco, tab_comp, tab_fair = st.tabs(["🎯 추천 번호", "구성(요약 · 홀짝·끝자리 등)", "공정성 체크"])

# ======================================================================
# 1) 추천 번호 탭  (첫 번째 탭)
# ======================================================================
with tab_reco:
    nums_all = list(range(1,46))

    # 추천 세트
    hot_set = recommend_hot(freq)
    cold_set = recommend_cold(freq)
    bal_set = recommend_balanced(freq)
    ai_set  = recommend_weighted_recent(df, lookback=lookback, include_bonus=include_bonus)
    sets = [
        ("🔥 HOT", hot_set, "최근 빈도 상위 기반"),
        ("❄️ COLD", cold_set, "오랫동안 드문 번호 가미"),
        ("⚖️ BALANCED", bal_set, "홀짝·저고 균형"),
        ("🤖 AI 가중", ai_set, f"최근 {lookback}회 빈도 가중 샘플링"),
    ]

    bonus_cands = bonus_candidates(df, lookback=lookback, topk=5) if include_bonus else []
    R = rolling_frequency(df, window=lookback, include_bonus=include_bonus)

    for title, picked, subtitle in sets:
        with st.container(border=True):
            cA, cB = st.columns([1,3], gap="large")

            # --- 카드 요약
            with cA:
                st.markdown(f"### {title}")
                st.markdown(f"<span style='color:#9CA3AF'>{subtitle}</span>", unsafe_allow_html=True)
                st.markdown(f"**번호**: <span style='font-size:20px'>{', '.join(f'{n:02d}' for n in picked)}</span>", unsafe_allow_html=True)
                comp = composition_metrics(picked)
                cc1, cc2, cc3 = st.columns(3)
                with cc1:
                    st.metric("합계", comp["sum"]); st.metric("홀수개수", comp["odd"])
                with cc2:
                    st.metric("범위", comp["range"]); st.metric("저번호(≤22)", comp["low"])
                with cc3:
                    st.metric("연속수 포함", "Yes" if comp["consecutive"] else "No")
                    st.caption(f"끝자리: {', '.join(map(str, comp['last_digits']))}")
                if include_bonus and bonus_cands:
                    st.info(f"보너스 후보(최근 {lookback}회 Top): {', '.join(f'{b:02d}' for b in bonus_cands)}")

            # --- 근거 시각화
            with cB:
                c1, c2 = st.columns(2, gap="large")

                # 근거①: 전체 빈도(선택번호 강조)
                colors = ["#334155"] * 45
                for n in picked: colors[n-1] = "#3B82F6"
                fig_freq = px.bar(
                    x=[str(i) for i in nums_all],
                    y=[int(freq.get(i,0)) for i in nums_all],
                    title="전체 빈도",
                )
                fig_freq.update_traces(marker_color=colors)
                fig_freq.update_layout(template="plotly_dark", paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
                                       height=320, margin=dict(l=10,r=10,t=50,b=10),
                                       xaxis_title="번호", yaxis_title="빈도", title_font=dict(size=20, color="#E5E7EB"))
                c1.plotly_chart(fig_freq, use_container_width=True)

                # 근거②: 공출현 히트맵 + 선택쌍 오버레이
                vmax = float(np.quantile(co_df.values, 0.99))
                fig_co2 = make_heatmap(co_df, title="공출현 히트맵 + 선택쌍",
                                       zmin=0, zmax=vmax, colorscale="YlGnBu", height=320)
                xs, ys = [], []
                for i in range(len(picked)):
                    for j in range(i+1, len(picked)):
                        a, b = picked[i], picked[j]
                        xs += [f"{a:02d}", f"{b:02d}"]; ys += [f"{b:02d}", f"{a:02d}"]
                if xs:
                    fig_co2.add_scatter(x=xs, y=ys, mode="markers",
                                        marker=dict(size=10, color="#EF4444"), name="선택쌍")
                c2.plotly_chart(fig_co2, use_container_width=True)

                # 근거③: 롤링 빈도(선택번호만)
                subR = R[[n for n in picked]]
                fig_roll = px.line(subR, title=f"최근 {lookback}회 롤링 빈도(선택 번호만)",
                                   labels={"index":"회차(draw_no)", "value":"빈도(창 내)"})
                fig_roll.update_layout(template="plotly_dark", paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
                                       height=300, legend_title_text="번호",
                                       margin=dict(l=10,r=10,t=50,b=10), title_font=dict(size=20, color="#E5E7EB"))
                st.plotly_chart(fig_roll, use_container_width=True)


# ======================================================================
# 2) 구성 탭  (요약 탭 내용 + 구성 분석을 한 곳에)
# ======================================================================
with tab_comp:
    # ---- (A) 핵심 요약: Top N · Top 쌍 · 히트맵 2개 ----
    st.subheader("핵심 요약")

    # 1행: Top N 빈도 + Top 쌍
    r1c1, r1c2 = st.columns(2, gap="large")
    with r1c1:
        top_title = f"Top {topn} Frequency — {'Bonus Included' if include_bonus else 'Bonus Excluded'}"
        if bar_dir == "세로":
            fig_freq_top = make_top_frequency_vertical(freq, topn=topn, title=top_title, compact=compact)
        else:
            fig_freq_top = make_top_frequency_horizontal(freq, topn=topn, title=top_title, compact=compact)
        st.plotly_chart(fig_freq_top, use_container_width=True)

    with r1c2:
        sig_pairs = pair_significance_binomial(co_df, n_draws=len(presence), include_bonus=include_bonus, alpha=0.05)
        top_pairs = sig_pairs.sort_values("co_count", ascending=False).head(topn)
        fig_tp = make_top_pairs_vertical(top_pairs, title=f"Top {topn} Co-occurring Pairs (붉은색=FDR 유의)", compact=compact)
        st.plotly_chart(fig_tp, use_container_width=True)

    # 2행: 히트맵 2개
    r2c1, r2c2 = st.columns(2, gap="large")
    with r2c1:
        vmax = float(np.quantile(co_df.values, 0.99))
        fig_co = make_heatmap(co_df, title=f"Pair Co-occurrence Heatmap — {'Bonus Included' if include_bonus else 'Bonus Excluded'}",
                              zmin=0, zmax=vmax, colorscale="YlGnBu", compact=compact)
        st.plotly_chart(fig_co, use_container_width=True)
    with r2c2:
        with st.expander("Correlation 히트맵 옵션", expanded=False):
            corr_abs = st.toggle("절대값 보기 (|r|)", value=True)
            corr_cluster = st.toggle("클러스터링 재정렬", value=True)
            corr_triangle = st.toggle("하삼각만 보기", value=True)
            corr_contrast = st.slider("표시 범위", 0.05, 1.0, 0.25, 0.05,
                                      help="절대값(|r|)일 때 [0..값], 아닐 때 [−값..+값]")

        fig_corr = make_corr_heatmap_pro(
            corr,
            title="Correlation Heatmap",
            abs_mode=corr_abs,
            cluster=corr_cluster,
            triangle=corr_triangle,
            contrast=float(corr_contrast),
            compact=compact
        )
        st.plotly_chart(fig_corr, use_container_width=True)

    # Hot/Cold 표 + 다운로드
    with st.expander("Hot / Cold 표 보기 & 다운로드", expanded=False):
        c_hot, c_cold = st.columns(2)
        with c_hot:
            st.markdown("**Top 10 Hot**")
            st.dataframe(freq.sort_values(ascending=False).head(10).rename("count").to_frame(),
                         use_container_width=True, height=280)
        with c_cold:
            st.markdown("**Top 10 Cold**")
            st.dataframe(freq.sort_values(ascending=True).head(10).rename("count").to_frame(),
                         use_container_width=True, height=280)

        c_dl1, c_dl2, c_dl3 = st.columns(3)
        with c_dl1:
            st.download_button("⬇️ 빈도 CSV", data=freq.rename("count").reset_index().to_csv(index=False).encode("utf-8-sig"),
                               file_name=f"frequency_{'with' if include_bonus else 'no'}_bonus.csv", mime="text/csv")
        with c_dl2:
            st.download_button("⬇️ 공출현 행렬 CSV", data=co_df.reset_index().to_csv(index=False).encode("utf-8-sig"),
                               file_name=f"pair_cooccurrence_{'with' if include_bonus else 'no'}_bonus.csv", mime="text/csv")
        with c_dl3:
            st.download_button("⬇️ 상관 행렬 CSV", data=corr.reset_index().to_csv(index=False).encode("utf-8-sig"),
                               file_name=f"correlation_{'with' if include_bonus else 'no'}_bonus.csv", mime="text/csv")

    st.markdown("---")

    # ---- (B) 구성 분석: 홀짝·끝자리·연속수·합계·범위 ----
    st.subheader("구성 분석 — 홀짝·끝자리·연속수·합계·범위")
    feats = build_features(df)

    c1, c2, c3 = st.columns(3)
    fig_sum = px.histogram(feats, x="sum", nbins=30, title="합계 분포")
    fig_sum.update_layout(template="plotly_dark", paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
                          height=320, title_font=dict(size=20, color="#E5E7EB"))
    c1.plotly_chart(fig_sum, use_container_width=True)

    fig_rng = px.histogram(feats, x="range", nbins=25, title="범위(최대-최소) 분포")
    fig_rng.update_layout(template="plotly_dark", paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
                          height=320, title_font=dict(size=20, color="#E5E7EB"))
    c2.plotly_chart(fig_rng, use_container_width=True)

    odd_counts = feats["odd_cnt"].value_counts().sort_index()
    fig_odd = px.bar(x=[str(i) for i in odd_counts.index], y=odd_counts.values, title="홀수 개수 분포")
    fig_odd.update_layout(template="plotly_dark", paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
                          height=320, xaxis_title="홀수 개수", yaxis_title="회수", title_font=dict(size=20, color="#E5E7EB"))
    c3.plotly_chart(fig_odd, use_container_width=True)

    st.markdown("**끝자리(Last digit) 분포**")
    ld = last_digit_hist(df)
    fig_ld = px.bar(x=[str(i) for i in ld.index], y=ld.values, title="끝자리 분포(0~9)")
    fig_ld.update_layout(template="plotly_dark", paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
                         height=360, xaxis_title="끝자리", yaxis_title="빈도", title_font=dict(size=20, color="#E5E7EB"))
    st.plotly_chart(fig_ld, use_container_width=True)

    rate_consec = feats["has_consecutive"].mean()
    st.metric("연속수 포함 비율", f"{rate_consec*100:.1f}%")

# ======================================================================
# 3) 공정성 체크 탭
# ======================================================================
with tab_fair:
    st.subheader("무작위성 가정 검정 (Uniformity & Pair Over-representation)")
    chi = chi_square_uniform(freq)
    cfa, cfb = st.columns(2)
    with cfa: st.metric("χ² 통계량", f"{chi['stat']:.2f}")
    with cfb: st.metric("p-value", f"{chi['pvalue']:.4f}")

    n_draws = len(presence)
    sig_pairs = pair_significance_binomial(co_df, n_draws=n_draws, include_bonus=include_bonus, alpha=0.05)
    st.markdown("**쌍 과대표현(상향) FDR 보정 결과 (상위 50 표시)**")
    st.dataframe(sig_pairs.head(50), use_container_width=True, height=500)
    st.caption("모형: 45개 중 6(또는 7)개 무작위 추출 가정. Binomial 상향 단측, FDR 보정(BH).")
