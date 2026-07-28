"""📖 使い方タブ — マニュアルの埋め込みと別ウィンドウ表示。"""

from __future__ import annotations

import base64
from pathlib import Path

import streamlit as st

# 使い方マニュアルは static/ に置く（manuals/build_manual_html.py で生成）。
# - 別ウィンドウで開く用: Streamlit 静的配信の URL（データURIは別窓で開けないため）
# - タブ内の埋め込み用   : 同じ HTML を data URI 化（相対URLだと埋め込みが解決できないため）
# 実行時 cwd に依存しないよう、このファイルからの相対（= プロジェクト直下）で解決する。
MANUAL_PATH = Path(__file__).resolve().parents[1] / "static" / "manual.html"
MANUAL_URL = "app/static/manual.html"


@st.cache_data
def _manual_src(mtime: float) -> str:
    """マニュアル HTML を data URI 化する。mtime をキーにしてキャッシュする。"""
    if MANUAL_PATH.exists():
        html = MANUAL_PATH.read_text(encoding="utf-8")
    else:
        html = "<p>マニュアルが見つかりません。</p>"
    b64 = base64.b64encode(html.encode("utf-8")).decode("ascii")
    return f"data:text/html;charset=utf-8;base64,{b64}"


def load_manual_src() -> str:
    """埋め込み用マニュアルの data URI。

    引数なしでキャッシュすると、マニュアルを差し替えてもプロセスが生き続けた場合に
    古い内容が残る（静的配信はディスク直読みなので新しく、埋め込みだけ古くなる）。
    更新時刻をキーに渡して、ファイルが変われば必ず作り直す。
    """
    mtime = MANUAL_PATH.stat().st_mtime if MANUAL_PATH.exists() else 0.0
    return _manual_src(mtime)


def render() -> None:
    st.markdown(
        f'<a href="{MANUAL_URL}" target="_blank" rel="noopener" '
        'style="display:inline-block;padding:8px 18px;border-radius:8px;'
        'background:#2a78d6;color:#fff;font-weight:700;text-decoration:none;'
        'font-family:Meiryo,sans-serif;">🗗 使い方を別ウィンドウで開く</a>'
        '<span style="color:#898781;font-size:13px;margin-left:12px;">'
        'デュアルモニタの方は、開いたタブを別画面へ移すと見やすくなります。</span>',
        unsafe_allow_html=True,
    )
    st.write("")
    st.iframe(load_manual_src(), height=820)
