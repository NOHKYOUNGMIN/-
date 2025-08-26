# app.py
# ----------------------------------------------
# ARAM PS Dashboard (Champion-centric)
# 레포 루트에 있는 CSV를 자동 탐색해서 로드합니다.
# 필요 패키지: streamlit, pandas, numpy, plotly
# ----------------------------------------------
import os, ast
from typing import List
import numpy as np
import pandas as pd
import streamlit as st
import plotly.express as px
st.set_page_config(page_title="ARAM PS Dashboard", layout="wide")
# 0) 후보 파일명들(우선순위 순)
CSV_CANDIDATES = [
    "aram_participants_with_full_runes_merged_plus.csv",
    "aram_participants_with_full_runes_merged.csv",
    "aram_participants_with_full_runes.csv",
    "aram_participants_clean_preprocessed.csv",
    "aram_participants_clean_no_dupe_items.csv",
    "aram_participants_with_items.csv",
]
# ---------- 유틸 ----------
def _yes(x) -> int:
    s = str(x).strip().lower()
    return 1 if s in ("1","true","t","yes") else 0
def _first_nonempty(*vals):
    for v in vals:
        if v is None:
            continue
        s = str(v).strip()
        if s and s.lower() not in ("nan", "none"):
            return s
    return ""
def _as_list(s):
    if isinstance(s, list):
        return s
    if not isinstance(s, str):
        return []
    s = s.strip()
    if not s:
        return []
    # 리스트형 문자열이면 파싱
    try:
        v = ast.literal_eval(s)
        if isinstance(v, list):
            return v
    except Exception:
        pass
    # 파이프/쉼표도 지원
    if "|" in s:
        return [t.strip() for t in s.split("|") if t.strip()]
    if "," in s:
        return [t.strip() for t in s.split(",") if t.strip()]
    return [s]
def _discover_csv() -> str | None:
    for name in CSV_CANDIDATES:
        if os.path.exists(name):
            return name
    return None
# ---------- 데이터 로드 ----------
@st.cache_data(show_spinner=False)
def load_df(path_or_buffer) -> pd.DataFrame:
    df = pd.read_csv(path_or_buffer)
    # win -> 0/1
    if "win" in df.columns:
        df["win_clean"] = df["win"].apply(_yes)
    else:
        df["win_clean"] = 0
    # 스펠 이름 컬럼 정규화(spell1_name/spell1)
    s1 = "spell1_name" if "spell1_name" in df.columns else ("spell1" if "spell1" in df.columns else None)
    s2 = "spell2_name" if "spell2_name" in df.columns else ("spell2" if "spell2" in df.columns else None)
    df["spell1_final"] = df[s1].astype(str) if s1 else ""
    df["spell2_final"] = df[s2].astype(str) if s2 else ""
    df["spell_combo"]  = (df["spell1_final"] + " + " + df["spell2_final"]).str.strip()
    # 아이템 문자열 정리
    for c in [c for c in df.columns if c.startswith("item")]:
        df[c] = df[c].fillna("").astype(str).str.strip()
    # 팀/상대 조합 문자열 → 리스트
    for col in ("team_champs", "enemy_champs"):
        if col in df.columns:
            df[col] = df[col].apply(_as_list)
    # 경기시간(분)
    if "game_end_min" in df.columns:
        df["duration_min"] = pd.to_numeric(df["game_end_min"], errors="coerce")
    else:
        df["duration_min"] = np.nan
    df["duration_min"] = df["duration_min"].fillna(18.0).clip(lower=6.0, upper=40.0)
    # DPM, KDA
    if "damage_total" in df.columns:
        df["dpm"] = df["damage_total"] / df["duration_min"].replace(0, np.nan)
    else:
        df["dpm"] = np.nan
    for c in ("kills","deaths","assists"):
        if c not in df.columns:
            df[c] = 0
    df["kda"] = (df["kills"] + df["assists"]) / df["deaths"].replace(0, np.nan)
    df["kda"] = df["kda"].fillna(df["kills"] + df["assists"])
    return df
# ---------- 파일 입력부 ----------
st.sidebar.title("데이터")
auto_path = _discover_csv()
st.sidebar.write(":렌즈가_오른쪽_위에_있는_확대경: 자동 검색:", auto_path if auto_path else "없음")
uploaded = st.sidebar.file_uploader("CSV 업로드(선택)", type=["csv"])
if uploaded is not None:
    df = load_df(uploaded)
elif auto_path is not None:
    df = load_df(auto_path)
else:
    st.error("레포 루트에서 CSV를 찾을 수 없습니다. CSV를 업로드해 주세요.")
    st.stop()
# ---------- 필터 ----------
st.sidebar.markdown("---")
champions = sorted(df["champion"].dropna().unique().tolist())
if not champions:
    st.error("champion 컬럼이 비어있습니다.")
    st.stop()
sel_champ = st.sidebar.selectbox("챔피언 선택", champions)
# ---------- 서브셋 & 지표 ----------
dfc = df[df["champion"] == sel_champ].copy()
total_matches = df["matchId"].nunique() if "matchId" in df.columns else len(df["matchId"])
games = len(dfc)
winrate = round(dfc["win_clean"].mean()*100, 2) if games else 0.0
pickrate = round(games/total_matches*100, 2) if total_matches else 0.0
avg_k, avg_d, avg_a = round(dfc["kills"].mean(),2), round(dfc["deaths"].mean(),2), round(dfc["assists"].mean(),2)
avg_kda = round(dfc["kda"].mean(), 2)
avg_dpm = round(dfc["dpm"].mean(), 1)
st.title(f"ARAM Dashboard — {sel_champ}")
c1, c2, c3, c4, c5 = st.columns(5)
c1.metric("게임 수", games)
c2.metric("승률(%)", winrate)
c3.metric("픽률(%)", pickrate)
c4.metric("평균 K/D/A", f"{avg_k}/{avg_d}/{avg_a}")
c5.metric("평균 DPM", avg_dpm)
# ---------- 타임라인(있으면 표시) ----------
tl_cols = ["first_blood_min","blue_first_tower_min","red_first_tower_min","game_end_min","gold_spike_min"]
if any(c in dfc.columns for c in tl_cols):
    st.subheader("타임라인 요약")
    t1, t2, t3 = st.columns(3)
    if "first_blood_min" in dfc.columns and dfc["first_blood_min"].notna().any():
        t1.metric("퍼블 평균(분)", round(dfc["first_blood_min"].mean(), 2))
    if ("blue_first_tower_min" in dfc.columns) or ("red_first_tower_min" in dfc.columns):
        bt = round(dfc["blue_first_tower_min"].dropna().mean(), 2) if "blue_first_tower_min" in dfc.columns else np.nan
        rt = round(dfc["red_first_tower_min"].dropna().mean(), 2) if "red_first_tower_min" in dfc.columns else np.nan
        t2.metric("첫 포탑 평균(블루/레드)", f"{bt} / {rt}")
    if "game_end_min" in dfc.columns and dfc["game_end_min"].notna().any():
        t3.metric("평균 게임시간(분)", round(dfc["game_end_min"].mean(), 2))
    if "gold_spike_min" in dfc.columns and dfc["gold_spike_min"].notna().any():
        fig = px.histogram(dfc, x="gold_spike_min", nbins=20, title="골드 스파이크 시각 분포(분)")
        st.plotly_chart(fig, use_container_width=True)
# ---------- 코어 아이템 구매시각 ----------
core_cols = [c for c in ["first_core_item_min","first_core_item_name",
                         "second_core_item_min","second_core_item_name"] if c in dfc.columns]
if core_cols:
    st.subheader("코어 아이템 구매 타이밍")
    a, b = st.columns(2)
    if "first_core_item_min" in dfc.columns and dfc["first_core_item_min"].notna().any():
        a.metric("1코어 평균 분", round(dfc["first_core_item_min"].mean(), 2))
        st.plotly_chart(
            px.histogram(dfc.dropna(subset=["first_core_item_min"]),
                         x="first_core_item_min", nbins=24, title="1코어 시각 분포"),
            use_container_width=True
        )
    if "second_core_item_min" in dfc.columns and dfc["second_core_item_min"].notna().any():
        b.metric("2코어 평균 분", round(dfc["second_core_item_min"].mean(), 2))
        st.plotly_chart(
            px.histogram(dfc.dropna(subset=["second_core_item_min"]),
                         x="second_core_item_min", nbins=24, title="2코어 시각 분포"),
            use_container_width=True
        )
    core_rows = []
    if "first_core_item_name" in dfc.columns:
        core_rows.append(dfc[["matchId","win_clean","first_core_item_name"]].rename(columns={"first_core_item_name":"core_item"}))
    if "second_core_item_name" in dfc.columns:
        core_rows.append(dfc[["matchId","win_clean","second_core_item_name"]].rename(columns={"second_core_item_name":"core_item"}))
    if core_rows:
        union = pd.concat(core_rows, ignore_index=True)
        union = union[union["core_item"].astype(str)!=""]
        core_stats = (union.groupby("core_item")
                      .agg(games=("matchId","count"), wins=("win_clean","sum"))
                      .reset_index())
        core_stats["win_rate"] = (core_stats["wins"]/core_stats["games"]*100).round(2)
        core_stats = core_stats.sort_values(["games","win_rate"], ascending=[False,False])
        st.dataframe(core_stats.head(20), use_container_width=True)
# ---------- 아이템 성과 ----------
st.subheader("아이템 성과(슬롯 무시, 전체 합산)")
def item_stats(sub: pd.DataFrame) -> pd.DataFrame:
    item_cols = [c for c in sub.columns if c.startswith("item")]
    rec = []
    for c in item_cols:
        rec.append(sub[["matchId","win_clean",c]].rename(columns={c:"item"}))
    u = pd.concat(rec, ignore_index=True)
    u = u[u["item"].astype(str)!=""]
    g = (u.groupby("item")
         .agg(total_picks=("matchId","count"), wins=("win_clean","sum"))
         .reset_index())
    g["win_rate"] = (g["wins"]/g["total_picks"]*100).round(2)
    g = g.sort_values(["total_picks","win_rate"], ascending=[False,False])
    return g
st.dataframe(item_stats(dfc).head(25), use_container_width=True)
# ---------- 스펠/룬 ----------
c1, c2 = st.columns(2)
with c1:
    st.subheader("스펠 조합")
    if "spell_combo" in dfc.columns and dfc["spell_combo"].str.strip().any():
        sp = (dfc.groupby("spell_combo")
              .agg(games=("matchId","count"), wins=("win_clean","sum"))
              .reset_index())
        sp["win_rate"] = (sp["wins"]/sp["games"]*100).round(2)
        sp = sp.sort_values(["games","win_rate"], ascending=[False,False])
        st.dataframe(sp.head(10), use_container_width=True)
    else:
        st.info("스펠 정보가 부족합니다.")
with c2:
    st.subheader("룬 조합(메인/보조)")
    if ("rune_core" in dfc.columns) and ("rune_sub" in dfc.columns):
        rn = (dfc.groupby(["rune_core","rune_sub"])
              .agg(games=("matchId","count"), wins=("win_clean","sum"))
              .reset_index())
        rn["win_rate"] = (rn["wins"]/rn["games"]*100).round(2)
        rn = rn.sort_values(["games","win_rate"], ascending=[False,False])
        st.dataframe(rn.head(10), use_container_width=True)
    else:
        st.info("룬 정보가 부족합니다.")
# ---------- 원본 ----------
st.subheader("원본 데이터 (필터 적용)")
show_cols = [c for c in dfc.columns if c not in ("team_champs","enemy_champs")]
st.dataframe(dfc[show_cols], use_container_width=True)
st.markdown("---")
st.caption("CSV 자동탐색 + 업로드 지원 · 누락 컬럼은 자동으로 건너뜁니다.")
