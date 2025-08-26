# app.py
import os
import glob
from pathlib import Path
import streamlit as st
import pandas as pd
import requests
import plotly.express as px

# ============================
# 기본 설정
# ============================
st.set_page_config(page_title="ARAM PS Dashboard", layout="wide")
DEFAULT_CSV_NAME = "aram_participants_clean_preprocessed.csv"
DD_VER = "13.17.1"  # Data Dragon 버전

# ============================
# 데이터 소스 선택 UI
# ============================
st.sidebar.header("데이터 설정")
data_source = st.sidebar.radio(
    "데이터 소스",
    ["리포지토리 파일", "URL에서 불러오기", "CSV 업로드"],
    index=0,
)

repo_path_input = url_input = uploaded = None

if data_source == "리포지토리 파일":
    # 클라우드에서 작업 디렉토리가 다를 수 있으므로 경로를 직접 받도록 함
    repo_path_input = st.sidebar.text_input(
        "리포지토리 내 CSV 경로",
        value=DEFAULT_CSV_NAME,
        help="예: ./aram_participants_clean_preprocessed.csv, ./data/aram_*.csv",
    )
elif data_source == "URL에서 불러오기":
    url_input = st.sidebar.text_input(
        "CSV URL (https)",
        value="",
        placeholder="https://raw.githubusercontent.com/your/repo/main/aram.csv",
        help="GitHub raw 등 HTTPS URL 권장",
    )
else:
    uploaded = st.sidebar.file_uploader("CSV 업로드", type=["csv"])

# ============================
# 안전한 CSV 로더
# ============================
@st.cache_data(show_spinner=False)
def load_csv_from_path(path: str) -> pd.DataFrame:
    return pd.read_csv(path)

@st.cache_data(show_spinner=False)
def load_csv_from_url(url: str) -> pd.DataFrame:
    return pd.read_csv(url)

def find_first_match(pattern: str) -> str | None:
    matches = glob.glob(pattern, recursive=True)
    return matches[0] if matches else None

def resolve_repo_path(user_input: str) -> str | None:
    """
    리포지토리 파일 모드에서 입력값을 해석:
    - 정확한 파일 경로면 그대로 사용
    - 와일드카드면 첫 매칭 사용
    - 상대경로 흔한 루트 후보도 점검
    """
    p = Path(user_input)
    if p.exists() and p.is_file():
        return str(p)

    # 와일드카드(패턴) 지원
    if any(ch in user_input for ch in ["*", "?", "[", "]"]):
        hit = find_first_match(user_input)
        if hit:
            return hit

    # 흔한 루트 후보에서 탐색
    candidates = []
    roots = [
        ".", "./data", "/mount/src", "/mount/src/-", "/app", str(Path.cwd())
    ]
    for r in roots:
        candidates.append(str(Path(r) / user_input))
    for c in candidates:
        if Path(c).exists():
            return c

    return None

def load_data_smart(
    uploaded_file=None, repo_path: str | None = None, url: str | None = None
) -> pd.DataFrame:
    if uploaded_file is not None:
        # 업로드 파일은 캐시 없이 바로 로드(핸들 객체 캐시키 이슈 회피)
        return pd.read_csv(uploaded_file)
    if url:
        return load_csv_from_url(url)
    if repo_path:
        return load_csv_from_path(repo_path)
    raise FileNotFoundError("데이터 소스를 찾지 못했습니다.")

# ============================
# Data Dragon 리소스 (HTTPS)
# ============================
@st.cache_data(show_spinner=False)
def load_dd_resources(version: str):
    # 아이템
    item_json = requests.get(
        f"https://ddragon.leagueoflegends.com/cdn/{version}/data/ko_KR/item.json"
    ).json()
    item_name_to_img = {v["name"]: v["image"]["full"] for v in item_json["data"].values()}

    # 스펠
    spell_json = requests.get(
        f"https://ddragon.leagueoflegends.com/cdn/{version}/data/ko_KR/summoner.json"
    ).json()
    spell_name_to_img = {v["name"]: v["image"]["full"] for v in spell_json["data"].values()}

    # 룬 (perk-images 경로)
    rune_json = requests.get(
        "https://ddragon.leagueoflegends.com/cdn/13.17.1/data/ko_KR/runesReforged.json"
    ).json()
    rune_name_to_icon = {}
    for tree in rune_json:
        rune_name_to_icon[tree["name"]] = tree["icon"]
        for slot in tree["slots"]:
            for rune in slot["runes"]:
                rune_name_to_icon[rune["name"]] = rune["icon"]
    return item_name_to_img, spell_name_to_img, rune_name_to_icon

def item_icon_url(item_name: str, mp: dict) -> str | None:
    f = mp.get(item_name)
    return f"https://ddragon.leagueoflegends.com/cdn/{DD_VER}/img/item/{f}" if f else None

def spell_icon_url(spell_name: str, mp: dict) -> str | None:
    f = mp.get(spell_name)
    return f"https://ddragon.leagueoflegends.com/cdn/{DD_VER}/img/spell/{f}" if f else None

def rune_icon_url(rune_name: str, mp: dict) -> str | None:
    f = mp.get(rune_name)
    return f"https://ddragon.leagueoflegends.com/cdn/img/{f}" if f else None

# ============================
# UI: 데이터 로드 시도
# ============================
try:
    resolved_path = None
    if data_source == "리포지토리 파일":
        resolved_path = resolve_repo_path(repo_path_input)
        if not resolved_path:
            st.warning(
                f"리포지토리에서 파일을 찾지 못했습니다: `{repo_path_input}`\n"
                "정확한 경로를 입력하거나, URL/업로드 옵션을 사용하세요."
            )
        df = load_data_smart(uploaded_file=None, repo_path=resolved_path, url=None)
    elif data_source == "URL에서 불러오기":
        if not url_input.strip():
            st.stop()
        df = load_data_smart(uploaded_file=None, repo_path=None, url=url_input.strip())
    else:
        if uploaded is None:
            st.stop()
        df = load_data_smart(uploaded_file=uploaded, repo_path=None, url=None)

except FileNotFoundError as e:
    st.error(
        "CSV 파일을 열 수 없습니다. 다음 중 하나를 시도하세요:\n"
        "1) 사이드바에서 'CSV 업로드'로 파일 업로드\n"
        "2) 'URL에서 불러오기'로 HTTPS CSV 주소 입력\n"
        f"3) '리포지토리 파일'에 정확한 경로 입력 (예: ./{DEFAULT_CSV_NAME})"
    )
    st.stop()

# ============================
# 데이터 정리
# ============================
# 필수 컬럼 체크
required_cols = {"champion", "win", "matchId"}
missing = [c for c in required_cols if c not in df.columns]
if missing:
    st.error(f"필수 컬럼이 없습니다: {missing}. CSV 스키마를 확인하세요.")
    st.stop()

# 이진 승패 컬럼 정리
df["win_clean"] = df["win"].apply(lambda x: 1 if str(x).lower() in ("1", "true", "t", "yes") else 0)

# 문자열 컬럼 정리
item_cols = [c for c in df.columns if c.startswith("item")]
for c in item_cols:
    df[c] = df[c].fillna("").astype(str).str.strip()
for c in ["spell1", "spell2", "rune_core", "rune_sub"]:
    if c in df.columns:
        df[c] = df[c].astype(str).str.strip()

# ============================
# Data Dragon 맵 로드
# ============================
item_map, spell_map, rune_map = load_dd_resources(DD_VER)

# ============================
# 레이아웃 컴포넌트 (아이콘 그리드)
# ============================
def show_item_grid(stats_df: pd.DataFrame, n_cols: int = 5):
    st.subheader("Recommended Items")
    wrap = st.container()     # ✅ 컨테이너로 격리
    cols = wrap.columns(n_cols, gap="small")
    for i, row in stats_df.iterrows():
        with cols[i % n_cols]:
            url = item_icon_url(row["item"], item_map)
            if url:
                st.image(url, width=36)
            st.markdown(f"**{row['item']}**")
            st.caption(f"WR {row['win_rate']}% · PR {row['pick_rate']}%")
    st.divider()

def show_spell_pairs(spells_df: pd.DataFrame, n_cols: int = 5):
    st.subheader("Recommended Spell Combos")
    wrap = st.container()
    cols = wrap.columns(n_cols, gap="small")
    for i, row in spells_df.iterrows():
        with cols[i % n_cols]:
            c1, c2 = st.columns([1, 1])
            u1 = spell_icon_url(row["spell1"], spell_map) if "spell1" in row else None
            u2 = spell_icon_url(row["spell2"], spell_map) if "spell2" in row else None
            if u1: c1.image(u1, width=28)
            if u2: c2.image(u2, width=28)
            name1 = row["spell1"] if "spell1" in row else "?"
            name2 = row["spell2"] if "spell2" in row else "?"
            st.markdown(f"**{name1} + {name2}**")
            st.caption(f"WR {row['win_rate']}% · PR {row['pick_rate']}%")
    st.divider()

def show_rune_pairs(runes_df: pd.DataFrame, n_cols: int = 5):
    st.subheader("Recommended Runes")
    wrap = st.container()
    cols = wrap.columns(n_cols, gap="small")
    for i, row in runes_df.iterrows():
        with cols[i % n_cols]:
            c1, c2 = st.columns([1, 1])
            u1 = rune_icon_url(row["rune_core"], rune_map) if "rune_core" in row else None
            u2 = rune_icon_url(row["rune_sub"], rune_map) if "rune_sub" in row else None
            if u1: c1.image(u1, width=32)
            if u2: c2.image(u2, width=28)
            name1 = row["rune_core"] if "rune_core" in row else "?"
            name2 = row["rune_sub"] if "rune_sub" in row else "?"
            st.markdown(f"**{name1} + {name2}**")
            st.caption(f"WR {row['win_rate']}% · PR {row['pick_rate']}%")
    st.divider()

# ============================
# 챔피언 선택
# ============================
champion_list = sorted(df["champion"].astype(str).unique())
if not champion_list:
    st.error("champion 값이 비어 있습니다. CSV 내용을 확인하세요.")
    st.stop()

selected_champion = st.sidebar.selectbox("Select Champion", champion_list)
champ_df = df[df["champion"].astype(str) == str(selected_champion)]

st.title(f"Champion: {selected_champion}")

# ============================
# 통계 집계
# ============================
# 아이템
if item_cols:
    records = []
    for c in item_cols:
        sub = champ_df[["matchId", "win_clean", c]].rename(columns={c: "item"})
        records.append(sub)
    items_long = pd.concat(records, ignore_index=True)
    items_long = items_long[items_long["item"].astype(str).str.len() > 0]

    item_stats = (
        items_long.groupby("item")
        .agg(total_picks=("matchId", "count"), wins=("win_clean", "sum"))
        .reset_index()
    )
    if not item_stats.empty:
        item_stats["win_rate"] = (item_stats["wins"] / item_stats["total_picks"] * 100).round(2)
        item_stats["pick_rate"] = (
            item_stats["total_picks"] / champ_df["matchId"].nunique() * 100
        ).round(2)
        item_stats = item_stats.sort_values(["win_rate", "pick_rate"], ascending=False).head(10)
        show_item_grid(item_stats, n_cols=5)
else:
    st.info("item* 컬럼이 없어 아이템 추천을 건너뜁니다.")

# 스펠
if {"spell1", "spell2"}.issubset(champ_df.columns):
    spell_stats = (
        champ_df.groupby(["spell1", "spell2"])
        .agg(total_games=("matchId", "count"), wins=("win_clean", "sum"))
        .reset_index()
    )
    if not spell_stats.empty:
        spell_stats["win_rate"] = (spell_stats["wins"] / spell_stats["total_games"] * 100).round(2)
        spell_stats["pick_rate"] = (
            spell_stats["total_games"] / champ_df["matchId"].nunique() * 100
        ).round(2)
        spell_stats = spell_stats.sort_values(["win_rate", "pick_rate"], ascending=False).head(10)
        show_spell_pairs(spell_stats, n_cols=5)
else:
    st.info("spell1, spell2 컬럼이 없어 스펠 추천을 건너뜁니다.")

# 룬
if {"rune_core", "rune_sub"}.issubset(champ_df.columns):
    rune_stats = (
        champ_df.groupby(["rune_core", "rune_sub"])
        .agg(total_games=("matchId", "count"), wins=("win_clean", "sum"))
        .reset_index()
    )
    if not rune_stats.empty:
        rune_stats["win_rate"] = (rune_stats["wins"] / rune_stats["total_games"] * 100).round(2)
        rune_stats["pick_rate"] = (
            rune_stats["total_games"] / champ_df["matchId"].nunique() * 100
        ).round(2)
        rune_stats = rune_stats.sort_values(["win_rate", "pick_rate"], ascending=False).head(10)
        show_rune_pairs(rune_stats, n_cols=5)
else:
    st.info("rune_core, rune_sub 컬럼이 없어 룬 추천을 건너뜁니다.")

# ============================
# 뒤쪽 그래프 (깨지지 않아야 정상)
# ============================
st.subheader("Win Rate by Top Items")
if item_cols and 'item_stats' in locals() and not item_stats.empty:
    fig = px.bar(item_stats, x="item", y="win_rate", text="win_rate")
    fig.update_traces(textposition="outside")
    fig.update_layout(xaxis_title="", yaxis_title="Win Rate (%)", margin=dict(t=30, b=30))
    st.plotly_chart(fig, use_container_width=True)  # ✅ 레이아웃 안전
else:
    st.caption("아이템 통계가 없어 그래프를 생략합니다.")
