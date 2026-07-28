"""AI異常検知の閾値・評価を説明する Streamlit アプリ（構想A）。

判定規則: 異常スコア >= 閾値 → NG判定

タブの中身は `ui/tab_*.py` が持つ（このファイルは組み立てのみ）。
- タブ①: 疑似OK/NG分布 + 閾値スライダー
- タブ②: 混同行列 + Recall/Precision/F1
- タブ③: ROC / PR / F1 曲線 + 閾値決定モード
"""

from __future__ import annotations

from pathlib import Path

import streamlit as st

from ui import sidebar, tab_curves, tab_manual, tab_matrix, tab_threshold

# 実行時 cwd に依存しないよう、このファイルからの相対で解決する
LOGO_PATH = Path(__file__).parent / "02.Logo_Images" / "TR_inc_logo.png"

st.set_page_config(
    page_title="AI異常検知：閾値と評価指標",
    page_icon="📊",
    layout="wide",
)
st.logo(str(LOGO_PATH), size="large")

data = sidebar.render()

st.title("AI異常検知：閾値と評価指標")
st.markdown(
    "> 幸福な家庭はみな同じように似ているが、"
    "不幸な家庭は不幸なさまもそれぞれ違うものだ。\n"
    ">\n"
    "> — トルストイ『アンナ・カレーニナ』"
)
st.caption(
    "異常検知もこれと同じ。**良品はどれも似ている**が、**欠陥は一つひとつ違う**。"
    "だから AI は「良品らしさ」だけを学び、そこから外れた度合いを **異常スコア** にする。"
    "そのスコアが **閾値** を超えたものを NG と判定する — "
    "AIは最初から OK / NG を答えているわけではない。"
)

tab_1, tab_2, tab_3, tab_help = st.tabs(
    ["① 閾値とは何か", "② 混同行列と評価指標", "③ ROC / PR / F1 曲線", "📖 使い方"]
)

with tab_1:
    tab_threshold.render(data)

with tab_2:
    tab_matrix.render(data)

with tab_3:
    tab_curves.render(data)

with tab_help:
    tab_manual.render()
