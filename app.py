# -*- coding: utf-8 -*-
# streamlit_aram_ps_clone.py
# 목표: lol.ps(ARAM 챔피언 상세)와 유사한 섹션(요약·통계·카운터·스펠/아이템·코어템·스킬·룬·종합) 구현
# 필요 패키지: streamlit, pandas, plotly

import ast
import itertools
from collections import defaultdict
from typing import List, Tuple

import pandas as pd
import streamlit as st
import plotly.express as px

st.set_page_config(page_title="ARAM PS Clone", layout="wide")

# -----------------------------
# 0) 설정
# -----------------------------
DEFAULT_CSVS = [
    "./aram_participants_clean_preprocessed.csv",
    "/mnt/data/aram_participants_clean_preprocessed.csv",
]
MIN_SHOW = 20     # 표본 하한(스펠/룬/시너지)
MIN_COMBO = 50    # 코어템 조합 하한
TOP_K = 15        # 표 표출 상한

# 간단 분류(한국어 아이템명 기준)
BOOTS = {
    "광전사의 군화","헤르메스의 발걸음","마법사의 신발","명석함의 아이오니아 장화",
    "기동력의 장화","판금 장화","닌자의 신발","헤르메스의 장화"
}
STARTER_CANDIDATES = {
    "롱소드","단검","여신의 눈물","루비 수정","사파이어 수정","원기 회복의 구슬",
    "흡혈의 낫","비에프 대검","B.F. 대검","쓸데없이 큰 지팡이","망각의 구","천 갑옷"
}
TRINKETS = {"포로 간식","허수아비"}  # ARAM 특수 아이템/장신구

# -----------------------------
# 1) 데이터 로드/정리
# -----------------------------
@st.cache_data(show_spinner=False)
def load_csv(paths: List[str]) -> pd.DataFrame:
    last_err = None
    for p in paths:
        try:
            return pd.read_csv(p)
        except Exception as e:
            last_err = e
    raise RuntimeError(f"CSV 로드 실패: {paths} / 마지막 에러: {last_err}")

@st.cache_data(show_spinner=False)
def prep_df(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()

    # 승패 0/1 정규화
    def _w(x):
        s = str(x).strip().lower()
        if s in {"1","true","t","y","yes","win"}: return 1
        if s in {"0","false","f","n","no","lose","loss"}: return 0
        try: return 1 if float(s) > 0 else 0
        except: return 0
    df["win01"] = df["win"].map(_w)

    # 아이템 공백 처리
    item_cols = [c for c in df.columns if c.startswith("item")]
    for c in item_cols:
        df[c] = df[c].fillna("").astype(str).str.strip()

    # 팀/적 챔피언 리스트 파싱
    def _to_list(s):
        if isinstance(s, str) and s.startswith("["):
            try: return ast.literal_eval(s)
            except: return []
        return []
    df["team_list"]  = df["team_champs"].map(_to_list) if "team_champs" in df else [[]]*len(df)
    df["enemy_list"] = df["enemy_champs"].map(_to_list) if "enemy_champs" in df else [[]]*len(df)

    return df

# -----------------------------
# 2) 통계 유틸
# -----------------------------
def total_matches(df: pd.DataFrame) -> int:
    return df["matchId"].nunique()

def champ_slice(df: pd.DataFrame, champ: str) -> pd.DataFrame:
    return df[df["champion"].astype(str) == str(champ)]

def compute_summary(df: pd.DataFrame, champ: str):
    sub = champ_slice(df, champ)
    tm = total_matches(df)
    games = len(sub)
    win_rate = round(sub["win01"].mean()*100, 2) if games else 0.0
    pick_rate = round(games / max(tm, 1) * 100, 2)
    return games, win_rate, pick_rate

def canonical_pair(a, b) -> Tuple[str, str]:
    a, b = str(a), str(b)
    return tuple(sorted([a, b]))

def item_frame_for(df: pd.DataFrame, champ: str) -> pd.DataFrame:
    """champ의 인벤토리 단품 레코드(장신구/빈값 제외)를 한 컬럼으로 합침"""
    sub = champ_slice(df, champ)
    items = [c for c in df.columns if c.startswith("item")]
    recs = [sub[["matchId","win01",c]].rename(columns={c:"item"}) for c in items]
    u = pd.concat(recs, ignore_index=True)
    u = u[(u["item"]!="") & (~u["item"].isin(TRINKETS))]
    return u

def simple_table(df: pd.DataFrame, count_col: str, win_col: str, total_base: int, name_col: str) -> pd.DataFrame:
    out = df.copy()
    out["승률"] = (out[win_col] / out[count_col] * 100).round(2)
    out["채택률"] = (out[count_col] / max(total_base,1) * 100).round(2)
    out.rename(columns={count_col:"게임수", name_col:"이름"}, inplace=True)
    return out[["이름","승률","채택률","게임수"]]

@st.cache_data(show_spinner=False)
def spell_stats(df: pd.DataFrame, champ: str) -> pd.DataFrame:
    sub = champ_slice(df, champ).copy()
    if ("spell1" not in sub.columns) or ("spell2" not in sub.columns): return pd.DataFrame()
    sub["pair"] = sub.apply(lambda r: canonical_pair(r["spell1"], r["spell2"]), axis=1)
    g = sub.groupby("pair").agg(게임수=("matchId","count"), 승=("win01","sum")).reset_index()
    g["spell_a"] = g["pair"].apply(lambda t: t[0]); g["spell_b"] = g["pair"].apply(lambda t: t[1])
    g = g[g["게임수"] >= MIN_SHOW]
    total_base = len(sub)
    g["승률"] = (g["승"]/g["게임수"]*100).round(2); g["채택률"] = (g["게임수"]/max(total_base,1)*100).round(2)
    return g.sort_values(["승률","게임수"], ascending=[False,False])[["spell_a","spell_b","승률","채택률","게임수"]].head(TOP_K)

@st.cache_data(show_spinner=False)
def starter_stats(df: pd.DataFrame, champ: str) -> pd.DataFrame:
    u = item_frame_for(df, champ)
    starters = u[u["item"].isin(STARTER_CANDIDATES)]
    if starters.empty: return pd.DataFrame()
    g = starters.groupby("item").agg(게임수=("matchId","count"), 승=("win01","sum")).reset_index()
    g = g[g["게임수"]>=MIN_SHOW]
    total_base = len(champ_slice(df, champ))
    return simple_table(g, "게임수", "승", total_base, "item").sort_values(["승률","게임수"], ascending=[False,False]).head(TOP_K)

@st.cache_data(show_spinner=False)
def boots_stats(df: pd.DataFrame, champ: str) -> pd.DataFrame:
    u = item_frame_for(df, champ)
    boots = u[u["item"].isin(BOOTS)]
    if boots.empty: return pd.DataFrame()
    g = boots.groupby("item").agg(게임수=("matchId","count"), 승=("win01","sum")).reset_index()
    g = g[g["게임수"]>=MIN_SHOW]
    total_base = len(champ_slice(df, champ))
    return simple_table(g, "게임수", "승", total_base, "item").sort_values(["승률","게임수"], ascending=[False,False]).head(TOP_K)

@st.cache_data(show_spinner=False)
def item_core_single(df: pd.DataFrame, champ: str) -> pd.DataFrame:
    """1코어(단품) 분포"""
    u = item_frame_for(df, champ)
    g = u.groupby("item").agg(게임수=("matchId","count"), 승=("win01","sum")).reset_index()
    g = g[g["게임수"]>=MIN_COMBO]
    total_base = len(champ_slice(df, champ))
    return simple_table(g, "게임수", "승", total_base, "item").sort_values(["승률","게임수"], ascending=[False,False]).head(TOP_K)

@st.cache_data(show_spinner=False)
def item_core_ranks(df: pd.DataFrame, champ: str, k: int) -> pd.DataFrame:
    """k코어 조합(순서 무시, 최종 보유 기준)"""
    sub = champ_slice(df, champ)
    items = [c for c in df.columns if c.startswith("item")]
    combos = defaultdict(lambda: {"cnt":0,"win":0})
    for _, r in sub.iterrows():
        inv = [str(r[c]) for c in items if str(r[c]) and str(r[c])!="nan"]
        inv = [x for x in inv if x not in TRINKETS]
        inv = list(dict.fromkeys(inv))  # 중복 제거
        if len(inv) < k: continue
        for combo in itertools.combinations(sorted(inv), k):
            combos[combo]["cnt"] += 1
            combos[combo]["win"] += int(r["win01"])
    rows = [{"조합":" + ".join(cmb), "게임수":v["cnt"], "승":v["win"]} for cmb,v in combos.items() if v["cnt"]>=MIN_COMBO]
    if not rows: return pd.DataFrame()
    g = pd.DataFrame(rows)
    total_base = len(sub)
    g["승률"] = (g["승"]/g["게임수"]*100).round(2); g["채택률"] = (g["게임수"]/max(total_base,1)*100).round(2)
    return g.sort_values(["승률","게임수"], ascending=[False,False])[["조합","승률","채택률","게임수"]].head(TOP_K)

@st.cache_data(show_spinner=False)
def rune_stats(df: pd.DataFrame, champ: str):
    sub = champ_slice(df, champ)
    if ("rune_core" in sub.columns) and ("rune_sub" in sub.columns):
        g = (sub.groupby(["rune_core","rune_sub"])
               .agg(게임수=("matchId","count"), 승=("win01","sum")).reset_index())
        g = g[g["게임수"]>=MIN_SHOW]
        total_base = len(sub)
        g["승률"] = (g["승"]/g["게임수"]*100).round(2); g["채택률"] = (g["게임수"]/max(total_base,1)*100).round(2)
        main_sub = g.sort_values(["승률","게임수"], ascending=[False,False])[["rune_core","rune_sub","승률","채택률","게임수"]].head(TOP_K)
    else:
        main_sub = pd.DataFrame()
    if "rune_shards" in sub.columns:
        s = (sub.groupby("rune_shards")
               .agg(게임수=("matchId","count"), 승=("win01","sum")).reset_index())
        s = s[s["게임수"]>=MIN_SHOW]
        total_base = len(sub)
        s["승률"] = (s["승"]/s["게임수"]*100).round(2); s["채택률"] = (s["게임수"]/max(total_base,1)*100).round(2)
        shards = s.sort_values(["승률","게임수"], ascending=[False,False])[["rune_shards","승률","채택률","게임수"]].head(TOP_K)
    else:
        shards = pd.DataFrame()
    return main_sub, shards

@st.cache_data(show_spinner=False)
def duo_synergy(df: pd.DataFrame, champ: str) -> pd.DataFrame:
    """같이 하면 좋은 챔피언(같은 팀 기준)"""
    sub = champ_slice(df, champ)[["matchId","win01","team_list"]].copy()
    base_win = round(sub["win01"].mean()*100,2) if len(sub) else 0.0
    agg = defaultdict(lambda: {"cnt":0,"win":0})
    for _, r in sub.iterrows():
        for ally in [a for a in r["team_list"] if a != champ]:
            agg[ally]["cnt"] += 1
            agg[ally]["win"] += int(r["win01"])
    rows = [{"챔피언":k,"게임수":v["cnt"],"승률":round(v["win"]/v["cnt"]*100,2),"시너지점수(Δwr)":round(v["win"]/v["cnt"]*100-base_win,2)}
            for k,v in agg.items() if v["cnt"]>=MIN_SHOW]
    if not rows: return pd.DataFrame()
    return pd.DataFrame(rows).sort_values(["승률","게임수"], ascending=[False,False]).head(30)

@st.cache_data(show_spinner=False)
def counters_vs(df: pd.DataFrame, champ: str) -> pd.DataFrame:
    """상대 상성(상대 팀 기준)"""
    sub = champ_slice(df, champ)[["matchId","win01","enemy_list"]].copy()
    base_win = round(sub["win01"].mean()*100,2) if len(sub) else 0.0
    agg = defaultdict(lambda: {"cnt":0,"win":0})
    for _, r in sub.iterrows():
        for enemy in r["enemy_list"]:
            agg[enemy]["cnt"] += 1
            agg[enemy]["win"] += int(r["win01"])
    rows = [{"상대":k,"게임수":v["cnt"],"승률":round(v["win"]/v["cnt"]*100,2),"상대효과(Δwr)":round(v["win"]/v["cnt"]*100-base_win,2)}
            for k,v in agg.items() if v["cnt"]>=MIN_SHOW]
    if not rows: return pd.DataFrame()
    return pd.DataFrame(rows).sort_values(["승률","게임수"], ascending=[False,False]).head(30)

@st.cache_data(show_spinner=False)
def overall_tier(df: pd.DataFrame) -> pd.DataFrame:
    tm = total_matches(df)
    g = (df.groupby("champion").agg(게임수=("matchId","count"), 승=("win01","sum")).reset_index())
    g = g[g["게임수"]>=MIN_SHOW]
    g["승률"] = (g["승"]/g["게임수"]*100).round(2); g["픽률"] = (g["게임수"]/max(tm,1)*100).round(2)
    return g.sort_values(["승률","게임수"], ascending=[False,False])[["champion","승률","픽률","게임수"]]

# -----------------------------
# 3) UI
# -----------------------------
st.sidebar.title("ARAM PS Controls")
uploaded = st.sidebar.file_uploader("CSV 업로드(.csv)", type=["csv"])
if uploaded:
    df = pd.read_csv(uploaded)
else:
    df = load_csv(DEFAULT_CSVS)
df = prep_df(df)

champions = sorted(df["champion"].unique().tolist())
selected = st.sidebar.selectbox("챔피언 검색", champions)

# 상단 요약
st.title(selected)
games, wr, pr = compute_summary(df, selected)
c1, c2, c3 = st.columns(3)
c1.metric("승률", f"{wr} %"); c2.metric("픽률", f"{pr} %"); c3.metric("표본수", f"{games:,}")
st.markdown("---")

# 섹션 탭
tab_summary, tab_stats, tab_counter, tab_spell_item, tab_core, tab_skill, tab_rune, tab_overall = st.tabs(
    ["요약","통계","카운터","스펠·아이템","코어템","스킬","룬","종합"]
)

# 3.1 요약
with tab_summary:
    colA, colB, colC = st.columns(3)
    sp = spell_stats(df, selected)
    if not sp.empty:
        colA.subheader("추천 스펠")
        colA.dataframe(sp.head(5), use_container_width=True)
    bt = boots_stats(df, selected)
    if not bt.empty:
        colB.subheader("2티어 신발")
        colB.dataframe(bt.head(5), use_container_width=True)
    stt = starter_stats(df, selected)
    if not stt.empty:
        colC.subheader("시작 아이템(근사치)")
        colC.dataframe(stt.head(5), use_container_width=True)
    st.subheader("같이하면 좋은 챔피언")
    duo = duo_synergy(df, selected)
    if not duo.empty:
        st.dataframe(duo.head(10), use_container_width=True)

# 3.2 통계
with tab_stats:
    st.markdown("#### 스펠, 아이템 통계")
    col1, col2 = st.columns(2)
    if not sp.empty:
        col1.markdown("**스펠**")
        col1.dataframe(sp, use_container_width=True)
    if not stt.empty:
        col2.markdown("**시작 아이템(근사치)**")
        col2.dataframe(stt, use_container_width=True)
    if not bt.empty:
        st.markdown("**2티어 신발**")
        st.dataframe(bt, use_container_width=True)

# 3.3 카운터/시너지
with tab_counter:
    c1, c2 = st.columns(2)
    duo = duo_synergy(df, selected)
    if not duo.empty:
        c1.markdown("**듀오 시너지(같이하면 좋은 챔피언)**")
        c1.dataframe(duo, use_container_width=True)
    ctr = counters_vs(df, selected)
    if not ctr.empty:
        c2.markdown("**상대 상성(상대하면 까다로운/쉬운 챔피언)**")
        c2.dataframe(ctr.sort_values("승률", ascending=False), use_container_width=True)

# 3.4 스펠·아이템 (그래프)
with tab_spell_item:
    sp = spell_stats(df, selected)
    if not sp.empty:
        st.markdown("#### 스펠 상위 조합(그래프)")
        fig = px.bar(sp.head(10), x="승률",
                     y=sp.apply(lambda r: f"{r['spell_a']} + {r['spell_b']}", axis=1),
                     orientation="h", hover_data=["채택률","게임수"])
        st.plotly_chart(fig, use_container_width=True)
    u = item_frame_for(df, selected)
    if not u.empty:
        st.markdown("#### 아이템 채택 수(단품)")
        g = (u.groupby("item").agg(게임수=("matchId","count")).reset_index()
               .sort_values("게임수", ascending=False).head(20))
        fig2 = px.bar(g, x="게임수", y="item", orientation="h")
        st.plotly_chart(fig2, use_container_width=True)

# 3.5 코어템 (1~5코어 및 조합)
with tab_core:
    st.markdown("#### 1코어")
    one = item_core_single(df, selected)
    if not one.empty: st.dataframe(one, use_container_width=True)

    col2, col3 = st.columns(2)
    two = item_core_ranks(df, selected, 2)
    if not two.empty:
        col2.markdown("#### 2코어 조합"); col2.dataframe(two, use_container_width=True)
    three = item_core_ranks(df, selected, 3)
    if not three.empty:
        col3.markdown("#### 3코어 조합"); col3.dataframe(three, use_container_width=True)

    col4, col5 = st.columns(2)
    four = item_core_ranks(df, selected, 4)
    if not four.empty:
        col4.markdown("#### 4코어 조합"); col4.dataframe(four, use_container_width=True)
    five = item_core_ranks(df, selected, 5)
    if not five.empty:
        col5.markdown("#### 5코어 조합"); col5.dataframe(five, use_container_width=True)

    st.caption("※ 코어템은 최종 보유 조합(순서 무시) 근사치입니다. 타임라인이 있으면 구매순으로 정교화 가능.")

# 3.6 스킬 (데이터 없으면 안내)
with tab_skill:
    skill_cols = [c for c in df.columns if c.lower().startswith("skill")]
    if not skill_cols:
        st.info("스킬 업그레이드 로그가 없어 이 섹션은 숨김 처리됩니다. (타임라인 제공 시 구현 가능)")
    else:
        st.write("TODO: skill order stats")

# 3.7 룬
with tab_rune:
    main_sub, shards = rune_stats(df, selected)
    if not main_sub.empty:
        st.markdown("#### 메인/보조 룬 조합")
        st.dataframe(main_sub, use_container_width=True)
    if not shards.empty:
        st.markdown("#### 파편 조합")
        st.dataframe(shards, use_container_width=True)

# 3.8 종합(전 챔피언 티어표)
with tab_overall:
    tier = overall_tier(df)
    st.dataframe(tier.head(100), use_container_width=True)

st.markdown("---")
st.caption("로컬/업로드 CSV 기반 · 챔피언 선택 시 승률/픽률/추천 스펠·아이템·코어템·룬·시너지/카운터를 확인")
