# app.py
# -*- coding: utf-8 -*-
from __future__ import annotations
import os, re, hashlib, datetime, requests
import numpy as np, pandas as pd, streamlit as st, plotly.express as px

from lotto_data import load_csv, frequency, presence_matrix, cooccurrence
from rolling import rolling_frequency
from recs import (
    recommend_hot, recommend_cold, recommend_balanced, recommend_weighted_recent,
    composition_metrics, bonus_candidates
)
from features import build_features, last_digit_hist
from fairness import chi_square_uniform, pair_significance_binomial
from viz import (
    apply_global_style, kpi_card,
    make_top_frequency_vertical, make_heatmap, make_corr_heatmap_pro, make_top_pairs_vertical
)

# =========================
# 0) 전역 UI 설정 & 고정 옵션
# =========================
st.set_page_config(page_title="Lotto 6/45 Analyzer — Pro", page_icon="🎯", layout="wide")
apply_global_style()

# Streamlit 기본 UI 숨김 (상단 메뉴/헤더/푸터/하단 배지)
st.markdown("""
<style>
#MainMenu {visibility: hidden;}
header {visibility: hidden;}
footer {visibility: hidden;}
/* 일부 버전에서 우하단 배지/도구 위젯 숨김 */
div[class^="viewerBadge"] {display:none !important;}
div[data-testid="stStatusWidget"] {display:none !important;}
</style>
""", unsafe_allow_html=True)

# 옵션 고정(사이드바 제거)
INCLUDE_BONUS: bool = True
TOPN: int = 50
LOOKBACK: int = 500
COMPACT: bool = True
BAR_DIR: str = "세로"  # 고정 (세로 차트만 사용)

DATA_CSV = "data/lotto_draws.csv"
MEMBERS_CSV = "data/members.csv"

# 관리자 계정 (E.164 정규화 기준)
ADMIN_NAME = "김영솔"
ADMIN_PHONE_E164 = "+821024647664"

# =========================
# 1) 회원 저장/조회 유틸 (CSV + Supabase)
# =========================
def _ensure_dirs():
    os.makedirs(os.path.dirname(DATA_CSV) or ".", exist_ok=True)
    os.makedirs(os.path.dirname(MEMBERS_CSV) or ".", exist_ok=True)

def _load_members_csv() -> pd.DataFrame:
    _ensure_dirs()
    if not os.path.exists(MEMBERS_CSV):
        cols = ["created_at", "name", "phone_e164", "phone_hash"]
        return pd.DataFrame(columns=cols)
    return pd.read_csv(MEMBERS_CSV, dtype=str)

def _save_members_csv(df: pd.DataFrame):
    _ensure_dirs()
    df.to_csv(MEMBERS_CSV, index=False, encoding="utf-8-sig")

def _normalize_e164(phone: str) -> str:
    """
    010-1234-5678 / 01012345678 / +82 10 1234 5678 등 → +821012345678
    (숫자 이외 제거 후 한국 가정)
    """
    p = re.sub(r"\D", "", phone or "")
    if not p:
        return ""
    if p.startswith("0"):
        return "+82" + p[1:]
    if p.startswith("82"):
        return "+" + p
    if phone.strip().startswith("+"):
        return phone.strip()
    return "+82" + p  # 그 외도 한국 기본

def _phone_hash(phone_e164: str) -> str:
    return hashlib.sha256((phone_e164 or "").encode("utf-8")).hexdigest()

def _supabase_enabled() -> bool:
    try:
        _ = st.secrets["supabase"]["url"]
        _ = st.secrets["supabase"]["service_role_key"]
        return True
    except Exception:
        return False

def _supabase_upsert_member(name: str, phone_e164: str, phone_hash: str) -> bool:
    """
    Supabase REST upsert (테이블: public.members)
    사전 준비:
      create table if not exists public.members (
        id uuid primary key default gen_random_uuid(),
        name text not null,
        phone_e164 text not null,
        phone_hash text unique,
        marketing_optin boolean default false,
        created_at timestamptz default now()
      );
    """
    try:
        url = st.secrets["supabase"]["url"].rstrip("/") + "/rest/v1/members"
        key = st.secrets["supabase"]["service_role_key"]
        headers = {
            "apikey": key,
            "Authorization": f"Bearer {key}",
            "Content-Type": "application/json",
            "Prefer": "resolution=merge-duplicates,return=representation"
        }
        payload = {"name": name, "phone_e164": phone_e164, "phone_hash": phone_hash}
        r = requests.post(url, headers=headers, json=payload, timeout=12)
        if r.status_code not in (200, 201):
            if r.status_code == 409:
                r2 = requests.patch(url + f"?phone_hash=eq.{phone_hash}", headers=headers, json=payload, timeout=12)
                r2.raise_for_status()
            else:
                r.raise_for_status()
        return True
    except Exception as e:
        st.info(f"Supabase 저장 건너뜀: {e}")
        return False

def register_or_login(name: str, phone: str) -> tuple[bool, str]:
    """
    이름/전화로 간편 가입+로그인.
    - 이미 존재: 로그인 처리
    - 없으면: 신규 가입(csv append + (옵션) supabase 업서트)
    return: (성공여부, 메시지)
    """
    name = (name or "").strip()
    if not name:
        return False, "이름을 입력해주세요."
    phone_e164 = _normalize_e164(phone or "")
    if not phone_e164:
        return False, "전화번호를 정확히 입력해주세요."
    ph = _phone_hash(phone_e164)

    df = _load_members_csv()
    exists = False if df.empty else ph in set(df["phone_hash"])

    if not exists:
        row = pd.DataFrame([{
            "created_at": datetime.datetime.utcnow().isoformat(),
            "name": name,
            "phone_e164": phone_e164,
            "phone_hash": ph
        }])
        df = pd.concat([df, row], ignore_index=True)
        df = df.drop_duplicates(subset=["phone_hash"], keep="first")
        _save_members_csv(df)
        if _supabase_enabled():
            _supabase_upsert_member(name, phone_e164, ph)
    # 로그인 처리(신규 또는 기존)
    st.session_state["member_name"] = name
    st.session_state["member_phone_e164"] = phone_e164
    st.session_state["logged_in"] = True
    return True, f"{name}님 환영합니다! 🎉"

# =========================
# 2) 로그인/회원 UI
# =========================
if "logged_in" not in st.session_state:
    st.session_state["logged_in"] = False

def signin_block():
    if st.session_state.get("logged_in", False):
        colA, colB = st.columns([3,1])
        with colA:
            st.success(f"✅ 로그인됨: {st.session_state.get('member_name','회원')} ({st.session_state.get('member_phone_e164','')})")
        with colB:
            if st.button("로그아웃"):
                for k in ["logged_in", "member_name", "member_phone_e164"]:
                    st.session_state.pop(k, None)
                st.rerun()
        return

    st.subheader("🔒 로그인 / 간편 가입")
    with st.form("login_form", clear_on_submit=False):
        name = st.text_input("이름", key="login_name")
        phone = st.text_input("휴대폰 번호 (예: 010-1234-5678 또는 +821012345678)", key="login_phone")
        submitted = st.form_submit_button("가입 또는 로그인")
        if submitted:
            ok, msg = register_or_login(name, phone)
            if ok:
                st.success(msg)
                st.rerun()
            else:
                st.error(msg)

def locked_box(height: int = 220, msg: str = "🔒 로그인 후 확인 가능합니다"):
    st.markdown(
        f"""
        <div style="position:relative;height:{height}px;border-radius:16px;overflow:hidden;border:1px solid #263043;background:#0F172A">
          <div style="filter:blur(4px);opacity:.6;width:100%;height:100%;background:linear-gradient(135deg,#1E293B 0%,#0B1220 100%);"></div>
          <div style="position:absolute;inset:0;display:flex;align-items:center;justify-content:center;">
            <div style="background:rgba(0,0,0,.5);border:1px solid rgba(255,255,255,.1);padding:10px 14px;border-radius:12px;color:#E5E7EB;font-size:14px">
              {msg}
            </div>
          </div>
        </div>
        """,
        unsafe_allow_html=True
    )

# =========================
# 3) 데이터 로딩 & KPI
# =========================
_ensure_dirs()
if "df" not in st.session_state:
    st.session_state["df"] = load_csv(DATA_CSV)
df = st.session_state["df"]
if df.empty:
    st.warning("원본 CSV가 비어 있습니다. `data/lotto_draws.csv`를 업로드/배포해 주세요.")
    st.stop()
latest = int(df["draw_no"].max())

# KPI
c1, c2, c3, c4 = st.columns([1.1, 1.1, 2.4, 1.1])
with c1: kpi_card("수집 회차", f"{len(df):,}")
with c2: kpi_card("최신 회차", f"{latest:,}")
with c3: kpi_card("기간", f"{df['date'].min()} → {df['date'].max()}")
with c4: kpi_card("보너스 포함", "Yes" if INCLUDE_BONUS else "No")

st.title("🎯 Lotto 6/45 Analyzer — Pro")

# =========================
# 4) 핵심 계산
# =========================
freq = frequency(df, include_bonus=INCLUDE_BONUS)
presence = presence_matrix(df, include_bonus=INCLUDE_BONUS)
only_num = presence[[str(i) for i in range(1, 46)]]
co_df = cooccurrence(only_num)
corr = only_num.corr(method="pearson")

# =========================
# 5) 관리자 여부 판별
# =========================
is_admin = (
    st.session_state.get("member_name") == ADMIN_NAME and
    st.session_state.get("member_phone_e164") == ADMIN_PHONE_E164
)

# =========================
# 6) 탭 구성 (관리자만 회원관리 탭 노출)
# =========================
if is_admin:
    tab_reco, tab_comp, tab_fair, tab_admin = st.tabs(
        ["🎯 추천 번호", "구성(요약·홀짝·끝자리 등)", "공정성 체크", "회원 관리"]
    )
else:
    tab_reco, tab_comp, tab_fair = st.tabs(
        ["🎯 추천 번호", "구성(요약·홀짝·끝자리 등)", "공정성 체크"]
    )

# -------------------------
# 6-1) 추천 번호 탭
# -------------------------
with tab_reco:
    signin_block()  # 상단 로그인 박스

    nums_all = list(range(1, 46))
    hot_set = recommend_hot(freq)
    cold_set = recommend_cold(freq)
    bal_set = recommend_balanced(freq)
    ai_set = recommend_weighted_recent(df, lookback=LOOKBACK, include_bonus=INCLUDE_BONUS)

    sets = [
        ("🔥 HOT", hot_set, "최근 빈도 상위 기반"),
        ("❄️ COLD", cold_set, "오랫동안 드문 번호 가미"),
        ("⚖️ BALANCED", bal_set, "홀짝·저고 균형"),
        ("🤖 AI 가중", ai_set, f"최근 {LOOKBACK}회 빈도 가중 샘플링"),
    ]
    bonus_cands = bonus_candidates(df, lookback=LOOKBACK, topk=5) if INCLUDE_BONUS else []
    R = rolling_frequency(df, window=LOOKBACK, include_bonus=INCLUDE_BONUS)

    logged = st.session_state.get("logged_in", False)

    for idx, (title, picked, subtitle) in enumerate(sets):
        with st.container(border=True):
            cA, cB = st.columns([1, 3], gap="large")

            with cA:
                st.markdown(f"### {title}")
                st.markdown(f"<span style='color:#9CA3AF'>{subtitle}</span>", unsafe_allow_html=True)
                if idx == 0 or logged:
                    st.markdown(f"**번호**: <span style='font-size:20px'>{', '.join(f'{n:02d}' for n in picked)}</span>",
                                unsafe_allow_html=True)
                    comp = composition_metrics(picked)
                    cc1, cc2, cc3 = st.columns(3)
                    with cc1:
                        st.metric("합계", comp["sum"]); st.metric("홀수개수", comp["odd"])
                    with cc2:
                        st.metric("범위", comp["range"]); st.metric("저번호(≤22)", comp["low"])
                    with cc3:
                        st.metric("연속수 포함", "Yes" if comp["consecutive"] else "No")
                        st.caption(f"끝자리: {', '.join(map(str, comp['last_digits']))}")
                    if INCLUDE_BONUS and bonus_cands:
                        st.info(f"보너스 후보(최근 {LOOKBACK}회 Top): {', '.join(f'{b:02d}' for b in bonus_cands)}")
                else:
                    st.markdown("**번호**: ███ ███ ███ ███ ███ ███")
                    st.caption("🔒 로그인 후 확인 가능합니다")

            with cB:
                if idx == 0 or logged:
                    c1, c2 = st.columns(2, gap="large")
                    # 빈도 막대(선택번호 강조)
                    colors = ["#334155"] * 45
                    for n in picked: colors[n - 1] = "#3B82F6"
                    fig_freq = px.bar(
                        x=[str(i) for i in nums_all],
                        y=[int(freq.get(i, 0)) for i in nums_all],
                        title="전체 빈도",
                    )
                    fig_freq.update_traces(marker_color=colors)
                    fig_freq.update_layout(template="plotly_dark", paper_bgcolor="rgba(0,0,0,0)",
                                           plot_bgcolor="rgba(0,0,0,0)", height=320,
                                           margin=dict(l=10, r=10, t=50, b=10),
                                           xaxis_title="번호", yaxis_title="빈도",
                                           title_font=dict(size=20, color="#E5E7EB"))
                    c1.plotly_chart(fig_freq, use_container_width=True)

                    # 공출현 히트맵 + 선택쌍
                    vmax = float(np.quantile(co_df.values, 0.99))
                    fig_co2 = make_heatmap(co_df, title="공출현 히트맵 + 선택쌍",
                                           zmin=0, zmax=vmax, colorscale="YlGnBu", height=320)
                    xs, ys = [], []
                    for i in range(len(picked)):
                        for j in range(i + 1, len(picked)):
                            a, b = picked[i], picked[j]
                            xs += [f"{a:02d}", f"{b:02d}"]; ys += [f"{b:02d}", f"{a:02d}"]
                    if xs:
                        fig_co2.add_scatter(x=xs, y=ys, mode="markers",
                                            marker=dict(size=10, color="#EF4444"), name="선택쌍")
                    c2.plotly_chart(fig_co2, use_container_width=True)

                    # 롤링 빈도(선택 번호만)
                    subR = R[[n for n in picked]]
                    fig_roll = px.line(subR, title=f"최근 {LOOKBACK}회 롤링 빈도(선택 번호만)",
                                       labels={"index": "회차(draw_no)", "value": "빈도(창 내)"})
                    fig_roll.update_layout(template="plotly_dark", paper_bgcolor="rgba(0,0,0,0)",
                                           plot_bgcolor="rgba(0,0,0,0)", height=300,
                                           legend_title_text="번호", margin=dict(l=10, r=10, t=50, b=10),
                                           title_font=dict(size=20, color="#E5E7EB"))
                    st.plotly_chart(fig_roll, use_container_width=True)
                else:
                    locked_box(320)

# -------------------------
# 6-2) 구성 탭
# -------------------------
with tab_comp:
    if not st.session_state.get("logged_in", False):
        st.info("🔒 로그인 후 이용 가능한 섹션입니다.")
        locked_box(420, "🔒 로그인 후 전체 분석을 확인하세요")
    else:
        st.subheader("핵심 요약")
        r1c1, r1c2 = st.columns(2, gap="large")
        with r1c1:
            top_title = f"Top {TOPN} Frequency — {'Bonus Included' if INCLUDE_BONUS else 'Bonus Excluded'}"
            fig_freq_top = make_top_frequency_vertical(freq, topn=TOPN, title=top_title, compact=COMPACT)
            st.plotly_chart(fig_freq_top, use_container_width=True)
        with r1c2:
            sig_pairs = pair_significance_binomial(co_df, n_draws=len(presence), include_bonus=INCLUDE_BONUS, alpha=0.05)
            top_pairs = sig_pairs.sort_values("co_count", ascending=False).head(TOPN)
            fig_tp = make_top_pairs_vertical(top_pairs, title=f"Top {TOPN} Co-occurring Pairs (붉은색=FDR 유의)", compact=COMPACT)
            st.plotly_chart(fig_tp, use_container_width=True)

        r2c1, r2c2 = st.columns(2, gap="large")
        with r2c1:
            vmax = float(np.quantile(co_df.values, 0.99))
            fig_co = make_heatmap(co_df, title=f"Pair Co-occurrence Heatmap — {'Bonus Included' if INCLUDE_BONUS else 'Bonus Excluded'}",
                                  zmin=0, zmax=vmax, colorscale="YlGnBu", compact=COMPACT)
            st.plotly_chart(fig_co, use_container_width=True)
        with r2c2:
            fig_corr = make_corr_heatmap_pro(
                corr, title="Correlation Heatmap",
                abs_mode=True, cluster=True, triangle=True, contrast=0.25, compact=COMPACT
            )
            st.plotly_chart(fig_corr, use_container_width=True)

        st.markdown("---")
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

# -------------------------
# 6-3) 공정성 체크 탭
# -------------------------
with tab_fair:
    if not st.session_state.get("logged_in", False):
        st.info("🔒 로그인 후 이용 가능한 섹션입니다.")
        locked_box(420, "🔒 로그인 후 공정성 검정을 확인하세요")
    else:
        st.subheader("무작위성 가정 검정 (Uniformity & Pair Over-representation)")
        chi = chi_square_uniform(freq)
        cfa, cfb = st.columns(2)
        with cfa: st.metric("χ² 통계량", f"{chi['stat']:.2f}")
        with cfb: st.metric("p-value", f"{chi['pvalue']:.4f}")

        n_draws = len(presence)
        sig_pairs = pair_significance_binomial(co_df, n_draws=n_draws, include_bonus=INCLUDE_BONUS, alpha=0.05)
        st.markdown("**쌍 과대표현(상향) FDR 보정 결과 (상위 50 표시)**")
        st.dataframe(sig_pairs.head(50), use_container_width=True, height=500)
        st.caption("모형: 45개 중 6(또는 7)개 무작위 추출 가정. Binomial 상향 단측, FDR 보정(BH).")

# -------------------------
# 6-4) 회원 관리 탭 (관리자만 존재)
# -------------------------
if is_admin:
    with tab_admin:
        st.subheader("👥 회원 관리 (관리자 전용)")
        mdf = _load_members_csv()
        st.dataframe(mdf, use_container_width=True, height=520)
        st.download_button("⬇️ 회원 CSV 다운로드",
                           data=mdf.to_csv(index=False).encode("utf-8-sig"),
                           file_name="members.csv",
                           mime="text/csv")
        st.caption("※ 전화번호는 해시 및 E.164 형식으로 저장됩니다. 실제 운영 시 보관기간/파기정책을 고지하세요.")
