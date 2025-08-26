# app.py
# =========================================================
# ARAM PS Dashboard — 로컬 에셋(assets/)만으로 아이콘 표시
#   - 레이아웃: st.columns + st.image + st.caption (HTML 금지)
#   - 그래프: use_container_width=True
#   - 아이콘은 최초 1회 Data Dragon에서 assets/로 프리패치 후, 표시에는 로컬만 사용
# 요구 패키지: streamlit, pandas, numpy, plotly, requests
# =========================================================
import os, ast, json
from pathlib import Path
import numpy as np
import pandas as pd
import streamlit as st
import plotly.express as px
import requests

st.set_page_config(page_title="ARAM PS Dashboard", layout="wide")

# ------------------------
# 설정
# ------------------------
DD_VER = "13.17.1"
LOCALE = "ko_KR"
ASSET_DIR = Path("assets")
DD_DIR = ASSET_DIR / "dd" / DD_VER
ITEM_DIR  = ASSET_DIR / "items"
SPELL_DIR = ASSET_DIR / "spells"
RUNE_DIR  = ASSET_DIR / "runes"
CHAMP_DIR = ASSET_DIR / "champs"
for d in [DD_DIR, ITEM_DIR, SPELL_DIR, RUNE_DIR, CHAMP_DIR]:
    d.mkdir(parents=True, exist_ok=True)

# ------------------------
# 배경(옵션) - CSS만 주입 (레이아웃 충돌 없음)
# ------------------------
def set_background():
    st.markdown("""
    <style>
    :root{--brand1:#34d399;--brand2:#22c55e;--brand3:#16a34a;--bg:#fff;}
    .stApp{background:linear-gradient(to bottom,var(--bg) 0%,var(--bg) 72%,var(--brand1) 72%,var(--brand2) 86%,var(--brand3) 100%);}
    </style>
    """, unsafe_allow_html=True)
set_background()

# ------------------------
# CSV 자동 탐색 + 업로드
# ------------------------
CSV_CANDIDATES = [
    "aram_participants_with_full_runes_merged_plus.csv",
    "aram_participants_with_full_runes_merged.csv",
    "aram_participants_with_full_runes.csv",
    "aram_participants_clean_preprocessed.csv",
    "aram_participants_clean_no_dupe_items.csv",
    "aram_participants_with_items.csv",
]

def _discover_csv():
    for n in CSV_CANDIDATES:
        if os.path.exists(n):
            return n
    return None

def _yes(x)->int:
    s=str(x).strip().lower()
    return 1 if s in ("1","true","t","yes") else 0

def _as_list(s):
    if isinstance(s,list): return s
    if not isinstance(s,str): return []
    s=s.strip()
    if not s: return []
    try:
        v=ast.literal_eval(s)
        if isinstance(v,list): return v
    except: pass
    if "|" in s: return [t.strip() for t in s.split("|") if t.strip()]
    if "," in s: return [t.strip() for t in s.split(",") if t.strip()]
    return [s]

@st.cache_data(show_spinner=False)
def load_df(path_or_buffer)->pd.DataFrame:
    df=pd.read_csv(path_or_buffer)
    # 파생 지표/정리
    df["win_clean"]=df["win"].apply(_yes) if "win" in df.columns else 0
    s1 = "spell1_name" if "spell1_name" in df.columns else ("spell1" if "spell1" in df.columns else None)
    s2 = "spell2_name" if "spell2_name" in df.columns else ("spell2" if "spell2" in df.columns else None)
    df["spell1_final"]=df[s1].astype(str) if s1 else ""
    df["spell2_final"]=df[s2].astype(str) if s2 else ""
    df["spell_combo"]=(df["spell1_final"]+" + "+df["spell2_final"]).str.strip()
    for c in [c for c in df.columns if c.startswith("item")]:
        df[c]=df[c].fillna("").astype(str).str.strip()
    for c in ("team_champs","enemy_champs"):
        if c in df.columns: df[c]=df[c].apply(_as_list)
    if "game_end_min" in df.columns:
        df["duration_min"]=pd.to_numeric(df["game_end_min"],errors="coerce")
    else:
        df["duration_min"]=np.nan
    df["duration_min"]=df["duration_min"].fillna(18.0).clip(6.0,40.0)
    if "damage_total" in df.columns:
        df["dpm"]=df["damage_total"]/df["duration_min"].replace(0,np.nan)
    else:
        df["dpm"]=np.nan
    for c in ("kills","deaths","assists"):
        if c not in df.columns: df[c]=0
    df["kda"]=(df["kills"]+df["assists"])/df["deaths"].replace(0,np.nan)
    df["kda"]=df["kda"].fillna(df["kills"]+df["assists"])
    return df

st.sidebar.title("데이터")
auto=_discover_csv()
st.sidebar.write(":mag: 자동 검색:", auto if auto else "없음")
up=st.sidebar.file_uploader("CSV 업로드(선택)", type=["csv"])
if up is not None:
    df=load_df(up)
elif auto is not None:
    df=load_df(auto)
else:
    st.error("레포 루트에서 CSV를 찾지 못했습니다. 업로드해 주세요.")
    st.stop()

# ------------------------
# Data Dragon JSON 프리패치 & 로컬 매핑
#   - JSON이 없으면 원본에서 1회 다운로드 (표시에는 로컬만 사용)
# ------------------------
def _download(url: str, out: Path):
    out.parent.mkdir(parents=True, exist_ok=True)
    r=requests.get(url, timeout=20)
    r.raise_for_status()
    out.write_bytes(r.content)

def ensure_dd_json():
    files = {
        "item.json":      f"https://ddragon.leagueoflegends.com/cdn/{DD_VER}/data/{LOCALE}/item.json",
        "summoner.json":  f"https://ddragon.leagueoflegends.com/cdn/{DD_VER}/data/{LOCALE}/summoner.json",
        "runesReforged.json": f"https://ddragon.leagueoflegends.com/cdn/{DD_VER}/data/{LOCALE}/runesReforged.json",
        "champion.json":  f"https://ddragon.leagueoflegends.com/cdn/{DD_VER}/data/{LOCALE}/champion.json",
    }
    for fname, url in files.items():
        path = DD_DIR / fname
        if not path.exists():
            _download(url, path)

def load_dd_maps_local():
    ensure_dd_json()
    item_json = json.loads((DD_DIR/"item.json").read_text(encoding="utf-8"))
    spell_json = json.loads((DD_DIR/"summoner.json").read_text(encoding="utf-8"))
    rune_json  = json.loads((DD_DIR/"runesReforged.json").read_text(encoding="utf-8"))
    champ_json = json.loads((DD_DIR/"champion.json").read_text(encoding="utf-8"))["data"]

    item_map = {v["name"]: v["image"]["full"] for v in item_json["data"].values()}
    spell_map = {v["name"]: v["image"]["full"] for v in spell_json["data"].values()}
    rune_map  = {}
    for tree in rune_json:
        rune_map[tree["name"]] = tree["icon"]
        for slot in tree["slots"]:
            for r in slot["runes"]:
                rune_map[r["name"]] = r["icon"]
    champ_name_to_img = {v["name"]: v["image"]["full"] for v in champ_json.values()}
    champ_id_to_img   = {k: v["image"]["full"] for k,v in champ_json.items()}
    return item_map, spell_map, rune_map, champ_name_to_img, champ_id_to_img

item_map, spell_map, rune_map, champ_name_img, champ_id_img = load_dd_maps_local()

# ------------------------
# 필요 에셋만 선별 다운로드 → assets/
#   - 표시에는 로컬 파일 경로만 사용
# ------------------------
def collect_needed_names(df_all: pd.DataFrame):
    items=set()
    for c in [c for c in df_all.columns if c.startswith("item")]:
        items |= set(df_all[c].dropna().astype(str).str.strip())
    spells=set(df_all.get("spell1", df_all.get("spell1_name", pd.Series(dtype=str))).dropna().astype(str))
    spells |= set(df_all.get("spell2", df_all.get("spell2_name", pd.Series(dtype=str))).dropna().astype(str))
    runes=set(df_all.get("rune_core", pd.Series(dtype=str)).dropna().astype(str))
    runes |= set(df_all.get("rune_sub",  pd.Series(dtype=str)).dropna().astype(str))
    champs=set(df_all.get("champion", pd.Series(dtype=str)).dropna().astype(str))
    # 팀/상대에 챔프 리스트가 있으면 포함
    if "team_champs" in df_all.columns:
        for lst in df_all["team_champs"].dropna().tolist():
            for x in _as_list(lst): champs.add(str(x))
    if "enemy_champs" in df_all.columns:
        for lst in df_all["enemy_champs"].dropna().tolist():
            for x in _as_list(lst): champs.add(str(x))
    return items, spells, runes, champs

def ensure_assets(df_all: pd.DataFrame):
    items, spells, runes, champs = collect_needed_names(df_all)
    # 다운로드(이미 있으면 건너뜀)
    total = len(items)+len(spells)+len(runes)+len(champs)
    prog = st.sidebar.progress(0, text="아이콘 에셋 준비 중...")
    done=0

    for name in items:
        fn=item_map.get(str(name).strip())
        if fn:
            out=ITEM_DIR/fn
            if not out.exists():
                _download(f"https://ddragon.leagueoflegends.com/cdn/{DD_VER}/img/item/{fn}", out)
        done+=1; prog.progress(min(done/ max(total,1),1.0))

    for name in spells:
        fn=spell_map.get(str(name).strip())
        if fn:
            out=SPELL_DIR/fn
            if not out.exists():
                _download(f"https://ddragon.leagueoflegends.com/cdn/{DD_VER}/img/spell/{fn}", out)
        done+=1; prog.progress(min(done/ max(total,1),1.0))

    for name in runes:
        fn=rune_map.get(str(name).strip())
        if fn:
            out=RUNE_DIR/Path(fn).name
            if not out.exists():
                _download(f"https://ddragon.leagueoflegends.com/cdn/img/{fn}", out)
        done+=1; prog.progress(min(done/ max(total,1),1.0))

    for name in champs:
        key=str(name).strip()
        fn=champ_name_img.get(key) or champ_id_img.get(key)
        if fn:
            out=CHAMP_DIR/fn
            if not out.exists():
                _download(f"https://ddragon.leagueoflegends.com/cdn/{DD_VER}/img/champion/{fn}", out)
        done+=1; prog.progress(min(done/ max(total,1),1.0))

    prog.empty()
    st.sidebar.success("아이콘 에셋 준비 완료(assets/). 이제부터는 로컬 파일만 사용!")

# 사이드바 버튼으로 수동 준비 가능(최초 1회 추천)
if st.sidebar.button("아이콘 에셋 준비/갱신"):
    ensure_assets(df)

# 앱이 알아서 필요시 준비하도록(최초 접근에서 자동)
if not any([*ITEM_DIR.glob("*"), *SPELL_DIR.glob("*"), *RUNE_DIR.glob("*"), *CHAMP_DIR.glob("*")]):
    with st.spinner("아이콘 에셋을 준비하고 있습니다..."):
        ensure_assets(df)

# ------------------------
# 로컬 경로 리졸버 (표시에만 사용)
# ------------------------
def path_item(name:str):
    fn=item_map.get(name); p=ITEM_DIR/fn if fn else None
    return str(p) if p and p.exists() else None

def path_spell(name:str):
    fn=spell_map.get(name); p=SPELL_DIR/fn if fn else None
    return str(p) if p and p.exists() else None

def path_rune(name:str):
    fn=rune_map.get(name)
    p=RUNE_DIR/Path(fn).name if fn else None
    return str(p) if p and p.exists() else None

def path_champ(name:str):
    fn=champ_name_img.get(name) or champ_id_img.get(name)
    p=CHAMP_DIR/fn if fn else None
    return str(p) if p and p.exists() else None

# ------------------------
# 필터 & KPI
# ------------------------
champs=sorted(df["champion"].dropna().astype(str).unique())
if not champs: st.error("champion 컬럼이 비어있습니다."); st.stop()
sel=st.sidebar.selectbox("챔피언 선택", champs)
dfc=df[df["champion"].astype(str)==str(sel)].copy()

total_matches = df["matchId"].nunique() if "matchId" in df.columns else len(df)
games=len(dfc)
winrate=round(dfc["win_clean"].mean()*100,2) if games else 0.0
pickrate=round(games/max(total_matches,1)*100,2)
avg_k,avg_d,avg_a = round(dfc["kills"].mean(),2), round(dfc["deaths"].mean(),2), round(dfc["assists"].mean(),2)
avg_kda=round(dfc["kda"].mean(),2)
avg_dpm=round(dfc["dpm"].mean(),1)

lc, mc, rc = st.columns([1,3,1])
with lc:
    icon=path_champ(sel)
    if icon: st.image(icon, width=64)
with mc:
    st.title(f"ARAM Dashboard — {sel}")
m1,m2,m3,m4,m5 = st.columns(5)
m1.metric("게임 수", games)
m2.metric("승률(%)", winrate)
m3.metric("픽률(%)", pickrate)
m4.metric("평균 K/D/A", f"{avg_k}/{avg_d}/{avg_a}")
m5.metric("평균 DPM", avg_dpm)

# =========================================================
# 지표 함수들 (아이콘 그리드 + 표/그래프)
# =========================================================
def top_items_for_icons(sub: pd.DataFrame, top_k=15):
    item_cols=[c for c in sub.columns if c.startswith("item")]
    if not item_cols: return pd.DataFrame(columns=["item","win_rate","pick_rate"])
    rec=[]
    for c in item_cols:
        rec.append(sub[["matchId","win_clean",c]].rename(columns={c:"item"}))
    u=pd.concat(rec,ignore_index=True)
    u=u[u["item"].astype(str)!=""]
    g=(u.groupby("item").agg(total_picks=("matchId","count"), wins=("win_clean","sum")).reset_index())
    g["win_rate"]=(g["wins"]/g["total_picks"]*100).round(2)
    denom=sub["matchId"].nunique() if "matchId" in sub.columns else len(sub)
    g["pick_rate"]=((g["total_picks"]/max(denom,1))*100).round(2)
    return g.sort_values(["win_rate","pick_rate"],ascending=[False,False]).head(top_k)

def show_item_grid(stats_df, n_cols=5, title="아이템"):
    st.markdown(f"**{title}**")
    box=st.container(); cols=box.columns(n_cols, gap="small")
    for i,row in stats_df.iterrows():
        with cols[i % n_cols]:
            p=path_item(str(row["item"]))
            if p: st.image(p, width=36)
            st.markdown(f"**{row['item']}**")
            st.caption(f"WR {row['win_rate']}% · PR {row['pick_rate']}%")
    st.divider()

def show_spell_pairs(sub: pd.DataFrame, top_k=10, n_cols=5):
    if not {"spell1_final","spell2_final"}.issubset(sub.columns): 
        st.caption("스펠 정보 없음"); return
    g=(sub.groupby(["spell1_final","spell2_final"])
        .agg(total_games=("matchId","count"), wins=("win_clean","sum")).reset_index())
    g["win_rate"]=(g["wins"]/g["total_games"]*100).round(2)
    denom=sub["matchId"].nunique() if "matchId" in sub.columns else len(sub)
    g["pick_rate"]=((g["total_games"]/max(denom,1))*100).round(2)
    g=g.sort_values(["win_rate","pick_rate"],ascending=[False,False]).head(top_k)
    st.markdown("**스펠 조합**")
    wrap=st.container(); cols=wrap.columns(n_cols, gap="small")
    for i,row in g.iterrows():
        with cols[i % n_cols]:
            c1,c2=st.columns([1,1])
            p1,p2=path_spell(str(row["spell1_final"])), path_spell(str(row["spell2_final"]))
            if p1: c1.image(p1, width=28)
            if p2: c2.image(p2, width=28)
            st.markdown(f"**{row['spell1_final']} + {row['spell2_final']}**")
            st.caption(f"WR {row['win_rate']}% · PR {row['pick_rate']}%")
    st.divider()

def show_rune_pairs(sub: pd.DataFrame, top_k=10, n_cols=5):
    if not {"rune_core","rune_sub"}.issubset(sub.columns):
        st.caption("룬 정보 없음"); return
    g=(sub.groupby(["rune_core","rune_sub"])
        .agg(total_games=("matchId","count"), wins=("win_clean","sum")).reset_index())
    g["win_rate"]=(g["wins"]/g["total_games"]*100).round(2)
    denom=sub["matchId"].nunique() if "matchId" in sub.columns else len(sub)
    g["pick_rate"]=((g["total_games"]/max(denom,1))*100).round(2)
    g=g.sort_values(["win_rate","pick_rate"],ascending=[False,False]).head(top_k)
    st.markdown("**룬 조합(메인/보조)**")
    wrap=st.container(); cols=wrap.columns(n_cols, gap="small")
    for i,row in g.iterrows():
        with cols[i % n_cols]:
            c1,c2=st.columns([1,1])
            p1,p2=path_rune(str(row["rune_core"])), path_rune(str(row["rune_sub"]))
            if p1: c1.image(p1, width=32)
            if p2: c2.image(p2, width=28)
            st.markdown(f"**{row['rune_core']} + {row['rune_sub']}**")
            st.caption(f"WR {row['win_rate']}% · PR {row['pick_rate']}%")
    st.divider()

def build_order_stats(sub: pd.DataFrame, top_k=12):
    cols= [c for c in ["first_core_item_name","second_core_item_name"] if c in sub.columns]
    if len(cols)<1: return pd.DataFrame(columns=["core1","core2","games","win_rate"])
    df2=pd.DataFrame()
    if set(cols)=={"first_core_item_name","second_core_item_name"}:
        df2=sub[["matchId","win_clean","first_core_item_name","second_core_item_name"]].rename(
            columns={"first_core_item_name":"core1","second_core_item_name":"core2"}
        )
    else:
        df2=sub[["matchId","win_clean",cols[0]]].rename(columns={cols[0]:"core1"})
        df2["core2"]=""
    df2=df2[(df2["core1"].astype(str)!="") | (df2["core2"].astype(str)!="")]
    g=(df2.groupby(["core1","core2"])
        .agg(games=("matchId","count"), wins=("win_clean","sum")).reset_index())
    g["win_rate"]=(g["wins"]/g["games"]*100).round(2)
    return g.sort_values(["games","win_rate"],ascending=[False,False]).head(top_k)

def show_build_order_grid(g, n_cols=4):
    st.markdown("**빌드 오더(1코어 → 2코어)**")
    wrap=st.container(); cols=wrap.columns(n_cols, gap="small")
    for i,row in g.iterrows():
        with cols[i % n_cols]:
            c1,c2=st.columns([1,1])
            p1,p2=path_item(str(row["core1"])), path_item(str(row["core2"]))
            if p1: c1.image(p1, width=32)
            if p2: c2.image(p2, width=32)
            st.markdown(f"**{row['core1']} → {row['core2']}**")
            st.caption(f"{int(row['games'])}G · WR {row['win_rate']}%")
    st.divider()

def explode_enemy(sub: pd.DataFrame):
    if "enemy_champs" not in sub.columns: return pd.DataFrame(columns=["enemy"])
    x=sub[["matchId","win_clean","enemy_champs"]].copy()
    x=x.explode("enemy_champs").rename(columns={"enemy_champs":"enemy"}).dropna()
    x["enemy"]=x["enemy"].astype(str)
    return x

def explode_team(sub: pd.DataFrame, self_name: str):
    if "team_champs" not in sub.columns: return pd.DataFrame(columns=["ally"])
    x=sub[["matchId","win_clean","team_champs"]].copy()
    x=x.explode("team_champs").rename(columns={"team_champs":"ally"}).dropna()
    x["ally"]=x["ally"].astype(str)
    x=x[x["ally"]!=self_name]
    return x

def show_matchups(sub: pd.DataFrame, self_name: str, top_k=12, n_cols=4):
    em=explode_enemy(sub)
    if em.empty: st.caption("상대전 데이터 없음"); return
    g=(em.groupby("enemy")
        .agg(games=("matchId","count"), wins=("win_clean","sum")).reset_index())
    g["win_rate"]=(g["wins"]/g["games"]*100).round(2)
    g=g.sort_values(["games","win_rate"],ascending=[False,True]).head(top_k)
    st.markdown("**상대전(자주 만나는 챔피언)**")
    wrap=st.container(); cols=wrap.columns(n_cols, gap="small")
    for i,row in g.iterrows():
        with cols[i % n_cols]:
            p=path_champ(str(row["enemy"]))
            if p: st.image(p, width=40)
            st.markdown(f"**{row['enemy']}**")
            st.caption(f"{int(row['games'])}G · WR {row['win_rate']}%")
    st.divider()

def show_synergy(sub: pd.DataFrame, self_name: str, top_k=12, n_cols=4):
    tm=explode_team(sub, self_name=self_name)
    if tm.empty: st.caption("시너지 데이터 없음"); return
    g=(tm.groupby("ally")
        .agg(games=("matchId","count"), wins=("win_clean","sum")).reset_index())
    g["win_rate"]=(g["wins"]/g["games"]*100).round(2)
    g=g.sort_values(["games","win_rate"],ascending=[False,False]).head(top_k)
    st.markdown("**시너지(같이 잘 맞는 챔피언)**")
    wrap=st.container(); cols=wrap.columns(n_cols, gap="small")
    for i,row in g.iterrows():
        with cols[i % n_cols]:
            p=path_champ(str(row["ally"]))
            if p: st.image(p, width=40)
            st.markdown(f"**{row['ally']}**")
            st.caption(f"{int(row['games'])}G · WR {row['win_rate']}%")
    st.divider()

def show_timeline(sub: pd.DataFrame):
    tl_cols=["first_blood_min","blue_first_tower_min","red_first_tower_min","game_end_min","gold_spike_min"]
    if not any(c in sub.columns for c in tl_cols): st.caption("타임라인 데이터 없음"); return
    t1,t2,t3=st.columns(3)
    if "first_blood_min" in sub.columns and sub["first_blood_min"].notna().any():
        t1.metric("퍼블 평균(분)", round(sub["first_blood_min"].mean(),2))
    if ("blue_first_tower_min" in sub.columns) or ("red_first_tower_min" in sub.columns):
        bt = round(sub["blue_first_tower_min"].dropna().mean(),2) if "blue_first_tower_min" in sub.columns else np.nan
        rt = round(sub["red_first_tower_min"].dropna().mean(),2) if "red_first_tower_min" in sub.columns else np.nan
        t2.metric("첫 포탑 평균(블루/레드)", f"{bt} / {rt}")
    if "game_end_min" in sub.columns and sub["game_end_min"].notna().any():
        t3.metric("평균 게임시간(분)", round(sub["game_end_min"].mean(),2))
    if "gold_spike_min" in sub.columns and sub["gold_spike_min"].notna().any():
        fig=px.histogram(sub, x="gold_spike_min", nbins=20, title="골드 스파이크 시각 분포(분)")
        st.plotly_chart(fig, use_container_width=True)

def show_length_bucket(sub: pd.DataFrame):
    if "duration_min" not in sub.columns: return
    bins=[0,10,15,20,25,40]
    labels=["≤10","10–15","15–20","20–25",">25"]
    x=sub.copy()
    x["len_bucket"]=pd.cut(x["duration_min"], bins=bins, labels=labels, include_lowest=True)
    g=(x.groupby("len_bucket").agg(games=("matchId","count"), wins=("win_clean","sum")).reset_index())
    g["win_rate"]=(g["wins"]/g["games"]*100).round(2)
    fig=px.bar(g, x="len_bucket", y="win_rate", text="win_rate", title="게임 길이 구간별 승률")
    fig.update_traces(textposition="outside")
    st.plotly_chart(fig, use_container_width=True)

def show_distributions(sub: pd.DataFrame):
    l,r=st.columns(2)
    with l:
        if "kda" in sub.columns:
            fig=px.histogram(sub, x="kda", nbins=30, title="KDA 분포")
            st.plotly_chart(fig, use_container_width=True)
    with r:
        if "dpm" in sub.columns and sub["dpm"].notna().any():
            fig=px.histogram(sub, x="dpm", nbins=30, title="DPM 분포")
            st.plotly_chart(fig, use_container_width=True)

# =========================================================
# 탭 배치
# =========================================================
tab1, tab2, tab3, tab4, tab5 = st.tabs(
    ["개요", "아이템/빌드", "스펠 · 룬", "상대전/시너지", "타임라인 · 지속시간"]
)

with tab1:
    st.subheader("핵심 지표 개요")
    show_distributions(dfc)

with tab2:
    st.subheader("아이템 & 빌드")
    it = top_items_for_icons(dfc, top_k=15)
    if not it.empty:
        show_item_grid(it, n_cols=5, title="상위 아이템(WR/PR)")
        fig=px.bar(it.sort_values("win_rate",ascending=False).head(10),
                   x="item", y="win_rate", text="win_rate", title="Top Items — Win Rate")
        fig.update_traces(textposition="outside")
        st.plotly_chart(fig, use_container_width=True)
    bo = build_order_stats(dfc, top_k=12)
    if not bo.empty:
        show_build_order_grid(bo, n_cols=4)

with tab3:
    st.subheader("스펠 · 룬")
    show_spell_pairs(dfc, top_k=10, n_cols=5)
    show_rune_pairs(dfc, top_k=10, n_cols=5)

with tab4:
    st.subheader("상대전 & 시너지")
    show_matchups(dfc, self_name=str(sel), top_k=12, n_cols=4)
    show_synergy(dfc, self_name=str(sel), top_k=12, n_cols=4)

with tab5:
    st.subheader("타임라인 · 지속시간")
    show_timeline(dfc)
    show_length_bucket(dfc)

# 원본(필터 적용)
st.markdown("---")
st.subheader("원본 데이터")
show_cols=[c for c in dfc.columns if c not in ("team_champs","enemy_champs")]
st.dataframe(dfc[show_cols], use_container_width=True)
