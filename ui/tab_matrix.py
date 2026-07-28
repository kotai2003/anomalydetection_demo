"""タブ② 混同行列と評価指標。"""

from __future__ import annotations

import math

import plotly.graph_objects as go
import streamlit as st

from ui.sidebar import AppData
from ui.theme import FONT, INK, INK_MUTED, INK_SECONDARY, SURFACE, pct

# 混同行列の各セルに入れる説明: (正式名称, 略号, 日本語訳, 現場での意味)
CONFUSION_CELLS = (
    (
        ("True Negative", "TN", "真陰性", "良品を正しく通過"),
        ("False Positive", "FP", "偽陽性", "<b>過検知</b>：良品を止めた"),
    ),
    (
        ("False Negative", "FN", "偽陰性", "<b>見逃し</b>：欠陥を通した"),
        ("True Positive", "TP", "真陽性", "欠陥を正しく停止"),
    ),
)


def confusion_figure(cm, threshold: float) -> go.Figure:
    """混同行列を Blues ヒートマップで描く（社内の既存出力と同じ形式）。

    行 = 実際 (True) / 列 = AI判定 (Predicted)、左上を TN として
    Good/Bad が左上→右下に並ぶ向きに揃える。
    """
    z = [[cm.tn, cm.fp], [cm.fn, cm.tp]]
    labels = ["OK（良品）", "NG（欠陥）"]
    zmax = max(max(row) for row in z) or 1

    fig = go.Figure(
        go.Heatmap(
            z=z,
            x=labels,
            y=labels,
            colorscale="Blues",
            zmin=0,
            zmax=zmax,
            xgap=2,
            ygap=2,
            colorbar=dict(
                thickness=14,
                outlinewidth=0,
                tickformat="d",
                dtick=1 if zmax <= 10 else None,
                tickfont=dict(color=INK_MUTED, size=12),
            ),
            hovertemplate="実際 %{y}／AI判定 %{x}<br>%{z} 件<extra></extra>",
        )
    )

    for row, values in enumerate(z):
        for col, value in enumerate(values):
            full_name, code, japanese, meaning = CONFUSION_CELLS[row][col]
            # 濃いセルの上では白抜きにする（seaborn の annot と同じ振る舞い）。
            # 小さい文字も載せるので、切替点は本文が 4.5:1 を保てる濃さに置く。
            fig.add_annotation(
                x=labels[col],
                y=labels[row],
                showarrow=False,
                align="center",
                text=(
                    f'<span style="font-size:34px"><b>{value}</b></span>'
                    f'<br><span style="font-size:14px">{full_name}（{code}）</span>'
                    f'<br><span style="font-size:14px">{japanese}</span>'
                    f'<br><span style="font-size:13px">{meaning}</span>'
                ),
                font=dict(
                    family=FONT,
                    color="#ffffff" if value / zmax > 0.65 else INK,
                ),
            )

    fig.update_layout(
        title=dict(
            text=f"混同行列（閾値 {threshold:.2f}）",
            font=dict(family=FONT, color=INK, size=16),
            x=0,
        ),
        height=560,
        margin=dict(t=56, r=16, b=56, l=16),
        paper_bgcolor=SURFACE,
        plot_bgcolor=SURFACE,
        font=dict(family=FONT, color=INK_SECONDARY, size=13),
    )
    fig.update_xaxes(
        title_text="AI判定（Predicted）", side="bottom", showgrid=False,
        tickfont=dict(color=INK_SECONDARY, size=13),
    )
    fig.update_yaxes(
        title_text="実際（True）", autorange="reversed", showgrid=False,
        tickfont=dict(color=INK_SECONDARY, size=13),
    )
    return fig


def render(data: AppData) -> None:
    m = data.metrics

    matrix_col, _ = st.columns([2, 1])
    matrix_col.plotly_chart(confusion_figure(data.cm, data.threshold))
    st.caption(
        "※ タブ①と同じデータ・同じ閾値。閾値を動かすと 4 つの数が連動して変わる。"
        "まずは略語ではなく、**過検知**（良品を止める誤り）と"
        "**見逃し**（欠陥を通す誤り）の 2 つで捉えるとよい。"
    )

    st.divider()

    cols = st.columns(5)
    cols[0].metric("Recall（検出率）", pct(m["recall"]), help="欠陥のうち、何%を止められたか")
    cols[1].metric("Specificity", pct(m["specificity"]), help="良品のうち、何%を正しく通せたか")
    cols[2].metric("Precision", pct(m["precision"]), help="NG と判定したもののうち、何%が本当に欠陥か")
    cols[3].metric("Accuracy", pct(m["accuracy"]), help="全サンプルのうち、何%が正解だったか")
    cols[4].metric("F1 Score", "—" if math.isnan(m["f1"]) else f"{m['f1']:.3f}",
                   help="Precision と Recall の調和平均")

    st.caption(
        "「—」は分母が 0 で計算できないことを示す（例: NG と判定したものが 1 件も無いときの Precision）。"
    )
