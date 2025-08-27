# app.py
# ------------------------------------------------------------------
# ARAM PS Dashboard (Streamlit 1.32+) + Data-Dragon 자동 매핑
# - 이미지 URL 직접 렌더링(st.image(url))  ✅
# - DD 버전/챔피언/아이템/스펠 매핑 자동 로드 & 캐시(ttl=1일) ✅
# - 동적 렌더링 안정화(컨테이너 단위 반복) ✅
# - 네트워크 재시도(Session+Retry) ✅
# ------------------------------------------------------------------
import os, ast, re, unicodedata, json
from typing import List
import numpy as np
import pandas as pd
import streamlit as st
import plotly.express as px
import requests
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry

st.set_page_config(page_title="ARAM PS Dashboard", layout="wide")

# ================================
# Network helpers (재시도 세션)
# ================================
def _session_with_retries() -> requests.Session:
    s = requests.Session()
    retry = Retry(
        total=3, backoff_factor=0.5,
        status_forcelist=(429, 500, 502, 503, 504),
        allowed_methods=("GET", "HEAD")
    )
    s.mount("https://", HTTPAdapter(max_retries=retry))
    s.mount("http://", HTTPAdapter(max_retries=retry))
    return s

# ================================
# Data-Dragon 버전 & 매핑 로드
# ================================
DEFAULT_DD_VER = "14.1.1"  # 네트워크 실패 시 폴백

@st.cache_data(show_spinner=False, ttl=86400)
def ddragon_version() -> str:
    try:
        s = _session_with_retries()
        return s.get(
            "https://ddragon.leagueoflegends.com/api/versions.json",
            timeout=5
        ).json()[0]
    except Exception:
        return DEFAULT_DD_VER

@st.cache_data(show_spinner=False, ttl=86400)
def load_dd_maps(ver: str):
    s = _session_with_retries()
    base = f"https://ddragon.leagueoflegends.com/cdn/{ver}/data/en_US"
    champs, items, spells = {}, {}, {}
    try:
        champs = s.get(f"{base}/champion.json", timeout=5).json()["data"]
    except Exception:
        champs = {}

    try:
        items = s.get(f"{base}/item.json", timeout=5).json()["data"]
    except Exception:
        items = {}

    try:
        spells = s.get(f"{base}/summoner.json", timeout=5).json()["data"]
    except Exception:
        spells = {}

    # --- 챔피언: name -> file(id.png), alias(normalized) -> file
    def _norm(s: str) -> str:
        s = unicodedata.normalize("NFKD", s)
        s = re.sub(r"[ '&\.:]", "", s)
        return s.lower()

    champ_name2file = {}
    champ_alias = {}
    for c in champs.values():
        cid = c.get("id", "")
        nm = c.get("name", "")
        if cid and nm:
            file = f"{cid}.png"
            champ_name2file[nm] = file
            champ_alias[_norm(nm)] = file

    # --- 아이템: name -> id
    item_name2id = {}
    for iid, v in items.items():
        nm = v.get("name")
        if nm:
            item_name2id[nm] = iid

    # --- 스펠: name -> key(SummonerFlash 등)
    spell_name2key = {}
    for v in spells.values():
        nm = v.get("name")
        sid = v.get("id")
        if nm and sid:
            spell_name2key[nm] = sid

    return {
        "champ_name2file": champ_name2file,
        "champ_alias": champ_alias,
        "item_name2id": item_name2id,
        "spell_name2key": spell_name2key,
    }

DDRAGON_VERSION = ddragon_version()
DD = load_dd_maps(DDRAGON_VERSION)

def champion_icon_url(name: str) -> str:
    key = DD["champ_name2file"].get(name)
    if not key:
        n = re.sub(r"[ '&\.:]", "", str(name)).lower()
        key = DD["champ_alias"].get(n)
    if not key:
        # 최후 폴백: 규칙 추정
        k = re.sub(r"[ '&\.:]", "", str(name))
        key = k[0].upper() + k[1:] + ".png" if k else "Aatrox.png"
    return f"https://ddragon.leagueoflegends.com/cdn/{DDRAGON_VERSION}/img/champion/{key}"

def item_icon_url(item: str) -> str:
    iid = DD["item_name2id"].get(str(item))
    if not iid:
        iid = "1001"  # Boots 폴백
    return f"https://ddragon.leagueoflegends.com/cdn/{DDRAGON_VERSION}/img/item/{iid}.png"

def spell_icon_url(spell: str) -> str:
    skey = DD["spell_name2key"].get(str(spell).strip())
    if not skey:
        skey = "SummonerFlash"
    return f"https://ddragon.leagueoflegends.com/cdn/{DDRAGON_VERSION}/img/spell/{skey}.png"

# ================================
# CSV 로더
# ================================
CSV_CANDIDATES = [
    "aram_participants_with_full_runes_merged_plus.csv",
    "aram_participants_with_full_runes_merged.csv",
    "aram_participants_with_full_runes.csv",
    "aram_participants_clean_preprocessed.csv",
    "aram_participants_clean_no_dupe_items.csv",
    "aram_participants_with_items.csv",
]

def _discover_csv():
    for f in CSV_CANDIDATES:
        if os.path.exists(f):
            return f
    return None

def _yes(x) -> int:
    return 1 if str(x).strip().lower() in ("1", "true", "t", "yes") else 0

def _as_list(s):
    if isinstance(s, list):
        return s
    if not isinstance(s, str) or not s.strip():
        return []
    try:
        v = ast.literal_eval(s)
        if isinstance(v, list):
            return v
    except Exception:
        pass
    spl = "|" if "|" in s else ("," if "," in s else None)
    return [t.strip() for t in s.split(spl)] if spl else [s]

@st.cache_data(show_spinner=False)
def load_df(buf) -> pd.DataFrame:
    df = pd.read_csv(buf)
    # 기본 파생
    df["win_clean"] = df.get("win", 0).apply(_yes)

    s1 = "spell1_name" if "spell1_name" in df.columns else "spell1"
    s2 = "spell2_name" if "spell2_name" in df.columns else "spell2"
    df["spell_combo"] = (df[s1].astype(str) + " + " + df[s2].astype(str)).str.strip()

    for c in [c for c in df.columns if c.startswith("item")]:
        df[c] = df[c].fillna("").astype(str).str.strip()

    for col in ("team_champs", "enemy_champs"):
        if col in df.columns:
            df[col] = df[col].apply(_as_list)

    df["duration_min"] = pd.to_numeric(df.get("game_end_min"), errors="coerce").fillna(18).clip(6, 40)
    df["dpm"] = df.get("damage_total", np.nan) / df["duration_min"].replace(0, np.nan)

    for k in ("kills", "deaths", "assists"):
        if k not in df.columns:
            df[k] = 0
    df["kda"] = (df["kills"] + df["assists"]) / df["deaths"].replace(0, np.nan)
    df["kda"] = df["kda"].fillna(df["kills"] + df["assists"])
    return df

# ================================
# 사이드바
# ================================
st.sidebar.header(":톱니바퀴:  설정")
auto = _discover_csv()
st.sidebar.write(":mag: 자동 검색:", auto if auto else "없음")
up = st.sidebar.file_uploader("CSV 업로드(선택)", type="csv")

df = load_df(up) if up else (load_df(auto) if auto else None)
if df is None:
    st.error("CSV 파일이 없습니다.")
    st.stop()

if "champion" not in df.columns:
    st.error("CSV에 'champion' 컬럼이 필요합니다.")
    st.stop()

champions = sorted(df["champion"].dropna().astype(str).unique())
sel = st.sidebar.selectbox(":dart: 챔피언 선택", champions)

# ================================
# 헤더 & 메트릭
# ================================
dfc = df[df["champion"] == sel]
total = df["matchId"].nunique() if "matchId" in df.columns else len(df)
games = len(dfc)
wr = round(dfc["win_clean"].mean() * 100, 2) if games else 0.0
pr = round(games / total * 100, 2) if total else 0.0

avg_k = round(dfc["kills"].mean(), 2) if games else 0.0
avg_d = round(dfc["deaths"].mean(), 2) if games else 0.0
avg_a = round(dfc["assists"].mean(), 2) if games else 0.0
avg_dpm = round(dfc["dpm"].mean(), 1) if games else 0.0

st.title(":trophy: ARAM Analytics")
mid = st.columns([2, 3, 2])[1]
with mid:
    st.image(champion_icon_url(sel), width=100)   # ✅ URL 직접 렌더링
    st.subheader(sel, divider=False)

m1, m2, m3, m4, m5 = st.columns(5)
m1.metric("게임 수", games)
m2.metric("승률", f"{wr}%")
m3.metric("픽률", f"{pr}%")
m4.metric("평균 K/D/A", f"{avg_k}/{avg_d}/{avg_a}")
m5.metric("평균 DPM", avg_dpm)

# ================================
# 본문 탭
# ================================
tab1, tab2, tab3, tab4 = st.tabs([
    ":bar_chart: 게임 분석",
    ":crossed_swords: 아이템 & 스펠",
    ":stopwatch: 타임라인",
    ":clipboard: 상세 데이터"
])

with tab1:
    if "first_blood_min" in dfc.columns and dfc["first_blood_min"].notna().any():
        st.metric("퍼블 평균 분", round(dfc["first_blood_min"].mean(), 2))
    if "game_end_min" in dfc.columns and dfc["game_end_min"].notna().any():
        st.metric("평균 게임 시간", round(dfc["game_end_min"].mean(), 2))

with tab2:
    if games == 0:
        st.info("선택한 챔피언의 게임 데이터가 없습니다.")
    else:
        left, right = st.columns(2)

        # ----- 아이템 성과
        with left:
            st.subheader(":shield: 아이템 성과")
            item_cols = [c for c in dfc.columns if c.startswith("item")]
            if item_cols:
                pieces = []
                mid_col = "matchId" if "matchId" in dfc.columns else None
                for c in item_cols:
                    tmp = dfc[[c, "win_clean"]].copy()
                    if mid_col:
                        tmp[mid_col] = dfc[mid_col].values
                    tmp = tmp.rename(columns={c: "item"})
                    pieces.append(tmp)
                rec = pd.concat(pieces, ignore_index=True)

                g = (
                    rec[rec["item"].astype(str) != ""]
                    .groupby("item", as_index=False)
                    .agg(total=("win_clean", "size"), wins=("win_clean", "sum"))
                )
                g["win_rate"] = (g["wins"] / g["total"] * 100).round(2)
                g = g.sort_values(["total", "win_rate"], ascending=[False, False]).head(10).reset_index(drop=True)

                for i, r in g.iterrows():
                    block = st.container()  # ✅ 고유 컨테이너
                    c_icon, c_name, c_pick, c_wr = block.columns([1, 4, 2, 2])
                    with c_icon:
                        st.image(item_icon_url(str(r["item"])), width=32)
                    with c_name:
                        st.write(str(r["item"]))
                    with c_pick:
                        st.write(f"{int(r['total'])} 게임")
                    with c_wr:
                        st.write(f"{r['win_rate']}%")
                    st.divider()
            else:
                st.info("아이템 컬럼(item1, item2, …)이 없습니다.")

        # ----- 스펠 조합
        with right:
            st.subheader(":sparkles: 스펠 조합")
            if "spell_combo" in dfc.columns:
                sp = (
                    dfc.groupby("spell_combo", as_index=False)
                    .agg(games=("win_clean", "size"), wins=("win_clean", "sum"))
                )
                sp["win_rate"] = (sp["wins"] / sp["games"] * 100).round(2)
                sp = sp.sort_values(["games", "win_rate"], ascending=[False, False]).head(8).reset_index(drop=True)

                for i, r in sp.iterrows():
                    s1, s2 = [t.strip() for t in str(r["spell_combo"]).split("+")]
                    block = st.container()  # ✅ 고유 컨테이너
                    col_i, col_n, col_v = block.columns([2, 3, 2])
                    with col_i:
                        st.image(spell_icon_url(s1), width=28)
                        st.image(spell_icon_url(s2), width=28)
                    with col_n:
                        st.write(str(r["spell_combo"]))
                    with col_v:
                        st.write(f"{r['win_rate']}%\n{int(r['games'])}G")
                    st.divider()
            else:
                st.info("스펠 정보가 없습니다.")

with tab3:
    if "first_core_item_min" in dfc.columns and dfc["first_core_item_min"].notna().any():
        st.metric("1코어 평균 분", round(dfc["first_core_item_min"].mean(), 2))
        fig = px.histogram(dfc, x="first_core_item_min", nbins=24, title="1코어 시점")
        fig.update_layout(plot_bgcolor="#1E2328", paper_bgcolor="#1E2328", font_color="#F0E6D2")
        st.plotly_chart(fig, use_container_width=True)
    else:
        st.info("1코어 타이밍 데이터가 없습니다.")

with tab4:
    drop_cols = [c for c in ("team_champs", "enemy_champs") if c in dfc.columns]
    st.dataframe(dfc.drop(columns=drop_cols, errors="ignore"), use_container_width=True)

st.caption(f"Data-Dragon v{DDRAGON_VERSION} · {len(champions)} 챔프 · {total} 경기")
