# app.py
# ----------------------------------------------
# ARAM PS Dashboard (Champion-centric)
# 레포 루트에 있는 CSV를 자동 탐색해서 로드합니다.
# 필요 패키지: streamlit, pandas, numpy, plotly, requests
# ----------------------------------------------
import os, ast
from typing import List
import numpy as np
import pandas as pd
import streamlit as st
import plotly.express as px
import requests  # ✅ 아이콘 맵 로드를 위해 추가

st.set_page_config(page_title="ARAM PS Dashboard", layout="wide")

# Data Dragon 버전(한 곳에서 관리)
DD_VER = "13.17.1"

# 0) 후보 파일명들(우선순위 순)
CSV_CANDIDATES = [
    "aram_participants_with_full_runes_merged_plus.csv",
    "aram_participants_with_full_runes_merged.csv",
    "aram_participants_with_full_runes.csv",
    "aram_participants_clean_preprocessed.csv",
    "aram_participants_clean_no_dupe_items.csv",
    "aram_participants_with_items.csv",
]

# --- Background Themes (CSS only / no HTML tables) ---
import streamlit as st

def set_background(theme: str = "split_bottom_green"):
    """
    theme 옵션:
      - split_bottom_green : 화이트 배경 + 화면 하단 28% 정도 초록 그라데이션 밴드
      - soft_full_gradient : 화면 전체에 아주 연한 초록 그라데이션
      - white_top_bar      : 화이트 배경 + 상단 얇은 초록 바(헤더 느낌)
    """
    css_map = {
        "split_bottom_green": f"""
        <style>
        :root {{
            --brand-1: #34d399; /* teal/emerald */
            --brand-2: #22c55e; /* green */
            --brand-3: #16a34a; /* darker green */
            --bg-white: #ffffff;
        }}
        /* Streamlit 기본 컨테이너만 타겟팅 → Plotly/레이아웃 영향 최소화 */
        .stApp {{
            background:
                linear-gradient(
                    to bottom,
                    var(--bg-white) 0%,
                    var(--bg-white) 72%,
                    var(--brand-1) 72%,
                    var(--brand-2) 86%,
                    var(--brand-3) 100%
                );
        }}
        </style>
        """,

        "soft_full_gradient": f"""
        <style>
        :root {{
            --g1: #eafff4;
            --g2: #d9ffe9;
            --g3: #c7ffd9;
        }}
        .stApp {{
            background: linear-gradient(180deg, var(--g1) 0%, var(--g2) 60%, var(--g3) 100%);
        }}
        </style>
        """,

        "white_top_bar": f"""
        <style>
        :root {{
            --brand-2: #22c55e;
        }}
        .stApp {{
            background: #ffffff;
            position: relative;
        }}
        /* 상단 얇은 바 */
        .stApp::before {{
            content: "";
            position: fixed;
            top: 0; left: 0; right: 0;
            height: 8px;
            background: linear-gradient(90deg, #34d399, var(--brand-2), #16a34a);
            z-index: 0; /* 본문 위에 뜨되 컴포넌트와 겹치지 않게 */
        }}
        </style>
        """,
    }

    st.markdown(css_map.get(theme, css_map["split_bottom_green"]), unsafe_allow_html=True)
# 사이드바 스위처 (원하면 고정값으로 호출해도 OK)
bg_choice = st.sidebar.selectbox(
    "배경 테마",
    ("split_bottom_green", "soft_full_gradient", "white_top_bar"),
    format_func=lambda s: {
        "split_bottom_green": "화이트 + 하단 그린 그라데이션",
        "soft_full_gradient": "연한 그린 전체 그라데이션",
        "white_top_bar": "화이트 + 상단 바",
    }[s],
)
set_background(bg_choice)
