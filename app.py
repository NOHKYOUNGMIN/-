"""
Streamlit ARAM PS Dashboard — Champion-centric
File: streamlit_aram_ps_app_champion.py
"""

import streamlit as st
import pandas as pd
import plotly.express as px
from typing import List

st.set_page_config(page_title="ARAM PS Dashboard", layout="wide")

CSV_PATH = "./aram_participants_clean_preprocessed.csv"

# --- 데이터 로드 ---
@st.cache_data
def load_data(path: str) -> pd.DataFrame:
    df = pd.read_csv(path)
    # win 컬럼 정리
    if 'win' in df.columns:
        df['win_clean'] = df['win'].apply(lambda x: 1 if str(x).lower() in ('1','true','t','yes') else 0)
    else:
        df['win_clean'] = 0
    item_cols = [c for c in df.columns if c.startswith('item')]
    for c in item_cols:
        df[c] = df[c].fillna('').astype(str).str.strip()
    return df

# --- 챔피언 통계 ---
@st.cache_data
def compute_champion_stats(df: pd.DataFrame, champion: str) -> pd.DataFrame:
    df_champ = df[df['champion']==champion].copy()
    total_matches = df['matchId'].nunique()
    total_games = len(df_champ)
    win_rate = round(df_champ['win_clean'].mean()*100,2)
    pick_rate = round(total_games/total_matches*100,2)
    return pd.DataFrame({'champion':[champion],'total_games':[total_games],'win_rate':[win_rate],'pick_rate':[pick_rate]})

# --- 아이템 통계 ---
@st.cache_data
def compute_item_stats(df: pd.DataFrame) -> pd.DataFrame:
    item_cols = [c for c in df.columns if c.startswith('item')]
    records = []
    for c in item_cols:
        tmp = df[['matchId','win_clean',c]].rename(columns={c:'item'})
        records.append(tmp)
    union = pd.concat(records, axis=0, ignore_index=True)
    union = union[union['item'].astype(str) != '']
    stats = (union.groupby('item')
             .agg(total_picks=('matchId','count'), wins=('win_clean','sum'))
             .reset_index())
    stats['win_rate'] = (stats['wins']/stats['total_picks']*100).round(2)
    total_matches = df['matchId'].nunique()
    stats['pick_rate'] = (stats['total_picks']/total_matches*100).round(2)
    stats = stats.sort_values('win_rate', ascending=False)
    return stats

# --- Spell 통계 ---
@st.cache_data
def compute_spell_stats(df: pd.DataFrame) -> pd.DataFrame:
    stats = df.groupby(['spell1','spell2']).agg(total_games=('matchId','count'), wins=('win_clean','sum')).reset_index()
    stats['win_rate'] = (stats['wins']/stats['total_games']*100).round(2)
    stats['pick_rate'] = (stats['total_games']/df['matchId'].nunique()*100).round(2)
    stats = stats.sort_values('win_rate', ascending=False)
    return stats

# --- Rune 통계 ---
@st.cache_data
def compute_rune_stats(df: pd.DataFrame) -> pd.DataFrame:
    stats = df.groupby(['rune_core','rune_sub']).agg(total_games=('matchId','count'), wins=('win_clean','sum')).reset_index()
    stats['win_rate'] = (stats['wins']/stats['total_games']*100).round(2)
    stats['pick_rate'] = (stats['total_games']/df['matchId'].nunique()*100).round(2)
    stats = stats.sort_values('win_rate', ascending=False)
    return stats

# --- 로드 ---
with st.spinner('Loading data...'):
    df = load_data(CSV_PATH)

# --- 사이드바 ---
st.sidebar.title('ARAM PS Controls')
champion_list = sorted(df['champion'].unique().tolist())
selected_champion = st.sidebar.selectbox('Select Champion', champion_list)

# --- 챔피언별 요약 ---
st.title(f"Champion: {selected_champion}")
champ_summary = compute_champion_stats(df, selected_champion)
st.metric("Games Played", champ_summary['total_games'].values[0])
st.metric("Win Rate (%)", champ_summary['win_rate'].values[0])
st.metric("Pick Rate (%)", champ_summary['pick_rate'].values[0])

# --- 추천 아이템 ---
st.subheader('Recommended Items')
items = compute_item_stats(df[df['champion']==selected_champion])
st.dataframe(items.head(20))

# --- 추천 스펠 ---
st.subheader('Recommended Spell Combos')
spells = compute_spell_stats(df[df['champion']==selected_champion])
st.dataframe(spells.head(10))

# --- 추천 룬 ---
st.subheader('Recommended Rune Combos')
runes = compute_rune_stats(df[df['champion']==selected_champion])
st.dataframe(runes.head(10))

# --- Raw Data ---
st.subheader('Raw Data (Filtered)')
st.dataframe(df[df['champion']==selected_champion])

st.markdown('---')
st.write('앱: 로컬 CSV 기반, 특정 챔피언 선택 시 승률, 픽률, 추천 아이템/스펠/룬 확인 가능')
