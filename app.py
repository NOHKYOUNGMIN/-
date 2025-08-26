# streamlit_app.py
import streamlit as st
import pandas as pd
import requests
import plotly.express as px

# -----------------------------
# 기본 설정
# -----------------------------
st.set_page_config(page_title="ARAM PS Dashboard", layout="wide")
CSV_PATH = "./aram_participants_clean_preprocessed.csv"
DD_VER = "13.17.1"  # Data Dragon 버전 (변수로 관리)

# -----------------------------
# 데이터 로드
# -----------------------------
@st.cache_data
def load_data(path: str) -> pd.DataFrame:
    df = pd.read_csv(path)
    # 이진 승패 컬럼 정리
    df["win_clean"] = df["win"].apply(lambda x: 1 if str(x).lower() in ("1", "true", "t", "yes") else 0)
    # 문자열 컬럼 정리
    item_cols = [c for c in df.columns if c.startswith("item")]
    for c in item_cols:
        df[c] = df[c].fillna("").astype(str).str.strip()
    for c in ["spell1", "spell2", "rune_core", "rune_sub"]:
        df[c] = df[c].astype(str).str.strip()
    return df

df = load_data(CSV_PATH)

# -----------------------------
# Data Dragon 리소스 (HTTPS 사용)
# -----------------------------
@st.cache_data
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

    # 룬 (아이콘은 /cdn/img/ 경로 하위의 perk-images/* 파일 경로가 온다)
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

item_name_to_img, spell_name_to_img, rune_name_to_icon = load_dd_resources(DD_VER)

# -----------------------------
# 이미지 URL 유틸
# -----------------------------
def item_icon_url(item_name: str) -> str | None:
    f = item_name_to_img.get(item_name)
    return f"https://ddragon.leagueoflegends.com/cdn/{DD_VER}/img/item/{f}" if f else None

def spell_icon_url(spell_name: str) -> str | None:
    f = spell_name_to_img.get(spell_name)
    return f"https://ddragon.leagueoflegends.com/cdn/{DD_VER}/img/spell/{f}" if f else None

def rune_icon_url(rune_name: str) -> str | None:
    f = rune_name_to_icon.get(rune_name)
    return f"https://ddragon.leagueoflegends.com/cdn/img/{f}" if f else None  # runes는 /cdn/img/ 고정

# -----------------------------
# 레이아웃 컴포넌트 (아이콘 그리드)
# -----------------------------
def show_item_grid(stats_df: pd.DataFrame, n_cols: int = 5):
    st.subheader("Recommended Items")
    wrap = st.container()     # ✅ 컨테이너로 아이콘 섹션 격리
    cols = wrap.columns(n_cols, gap="small")

    for i, row in stats_df.iterrows():
        with cols[i % n_cols]:
            url = item_icon_url(row["item"])
            if url:
                st.image(url, width=36)
            st.markdown(f"**{row['item']}**")
            st.caption(f"WR {row['win_rate']}% · PR {row['pick_rate']}%")
    st.divider()              # 아래 그래프와 시각적 구분

def show_spell_pairs(spells_df: pd.DataFrame, n_cols: int = 5):
    st.subheader("Recommended Spell Combos")
    wrap = st.container()
    cols = wrap.columns(n_cols, gap="small")

    for i, row in spells_df.iterrows():
        with cols[i % n_cols]:
            c1, c2 = st.columns([1, 1])
            u1, u2 = spell_icon_url(row["spell1"]), spell_icon_url(row["spell2"])
            if u1: c1.image(u1, width=28)
            if u2: c2.image(u2, width=28)
            st.markdown(f"**{row['spell1']} + {row['spell2']}**")
            st.caption(f"WR {row['win_rate']}% · PR {row['pick_rate']}%")
    st.divider()

def show_rune_pairs(runes_df: pd.DataFrame, n_cols: int = 5):
    st.subheader("Recommended Runes")
    wrap = st.container()
    cols = wrap.columns(n_cols, gap="small")

    for i, row in runes_df.iterrows():
        with cols[i % n_cols]:
            c1, c2 = st.columns([1, 1])
            u1, u2 = rune_icon_url(row["rune_core"]), rune_icon_url(row["rune_sub"])
            if u1: c1.image(u1, width=32)
            if u2: c2.image(u2, width=28)
            st.markdown(f"**{row['rune_core']} + {row['rune_sub']}**")
            st.caption(f"WR {row['win_rate']}% · PR {row['pick_rate']}%")
    st.divider()

# -----------------------------
# 챔피언 선택
# -----------------------------
champion_list = sorted(df["champion"].unique())
selected_champion = st.sidebar.selectbox("Select Champion", champion_list)
champ_df = df[df["champion"] == selected_champion]

st.title(f"Champion: {selected_champion}")

# -----------------------------
# 통계 집계 (아이템 / 스펠 / 룬)
# -----------------------------
# 아이템 집계: item* 컬럼 펼치기
item_cols = [c for c in champ_df.columns if c.startswith("item")]
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
item_stats["win_rate"] = (item_stats["wins"] / item_stats["total_picks"] * 100).round(2)
item_stats["pick_rate"] = (
    item_stats["total_picks"] / champ_df["matchId"].nunique() * 100
).round(2)
item_stats = item_stats.sort_values(["win_rate", "pick_rate"], ascending=False).head(10)

# 스펠
spell_stats = (
    champ_df.groupby(["spell1", "spell2"])
    .agg(total_games=("matchId", "count"), wins=("win_clean", "sum"))
    .reset_index()
)
spell_stats["win_rate"] = (spell_stats["wins"] / spell_stats["total_games"] * 100).round(2)
spell_stats["pick_rate"] = (
    spell_stats["total_games"] / champ_df["matchId"].nunique() * 100
).round(2)
spell_stats = spell_stats.sort_values(["win_rate", "pick_rate"], ascending=False).head(10)

# 룬
rune_stats = (
    champ_df.groupby(["rune_core", "rune_sub"])
    .agg(total_games=("matchId", "count"), wins=("win_clean", "sum"))
    .reset_index()
)
rune_stats["win_rate"] = (rune_stats["wins"] / rune_stats["total_games"] * 100).round(2)
rune_stats["pick_rate"] = (
    rune_stats["total_games"] / champ_df["matchId"].nunique() * 100
).round(2)
rune_stats = rune_stats.sort_values(["win_rate", "pick_rate"], ascending=False).head(10)

# -----------------------------
# 아이콘 + 텍스트 그리드 출력 (HTML 없이)
# -----------------------------
show_item_grid(item_stats, n_cols=5)
show_spell_pairs(spell_stats, n_cols=5)
show_rune_pairs(rune_stats, n_cols=5)

# -----------------------------
# (검증용) 뒤쪽 그래프 예시 - 안 깨져야 정상
# -----------------------------
st.subheader("Win Rate by Top Items (Check layout)")
fig = px.bar(item_stats, x="item", y="win_rate", text="win_rate")
fig.update_traces(textposition="outside")
fig.update_layout(xaxis_title="", yaxis_title="Win Rate (%)", margin=dict(t=30, b=30))
st.plotly_chart(fig, use_container_width=True)  # ✅ 레이아웃 안전
