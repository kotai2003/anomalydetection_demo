"""タブ① 閾値とは何か — 分布ヒストグラムと個票散布図。"""

from __future__ import annotations

import numpy as np
import plotly.graph_objects as go
import streamlit as st
from plotly.subplots import make_subplots

from ui.sidebar import AppData
from ui.theme import (
    FONT,
    GRID,
    INK,
    INK_MUTED,
    INK_SECONDARY,
    NG_COLOR,
    OK_COLOR,
    SURFACE,
)


def score_figure(
    ok_scores: np.ndarray, ng_scores: np.ndarray, threshold: float, seed: int
) -> go.Figure:
    """上段: スコア分布ヒストグラム / 下段: 個々のサンプル点。"""
    fig = make_subplots(
        rows=2,
        cols=1,
        shared_xaxes=True,
        row_heights=[0.68, 0.32],
        vertical_spacing=0.07,
    )

    bins = dict(start=0.0, end=1.0, size=0.025)
    for scores, name, color in (
        (ok_scores, "OK（良品）", OK_COLOR),
        (ng_scores, "NG（欠陥）", NG_COLOR),
    ):
        fig.add_trace(
            go.Histogram(
                x=scores,
                name=name,
                marker=dict(color=color, line=dict(width=0)),
                opacity=0.72,
                xbins=bins,
                hovertemplate=f"{name}<br>異常スコア %{{x}}<br>%{{y}} 件<extra></extra>",
            ),
            row=1,
            col=1,
        )

    # 下段の個票。閾値を動かしても点が跳ねないよう、ジッタもシード固定。
    jitter_rng = np.random.default_rng(seed + 1)
    for scores, name, color, base in (
        (ok_scores, "OK（良品）", OK_COLOR, 1.0),
        (ng_scores, "NG（欠陥）", NG_COLOR, 0.0),
    ):
        fig.add_trace(
            go.Scatter(
                x=scores,
                y=base + jitter_rng.uniform(-0.17, 0.17, len(scores)),
                mode="markers",
                name=name,
                marker=dict(
                    color=color, size=9, line=dict(width=2, color=SURFACE), opacity=0.9
                ),
                showlegend=False,
                hovertemplate=f"{name}<br>異常スコア %{{x:.3f}}<extra></extra>",
            ),
            row=2,
            col=1,
        )

    for row in (1, 2):
        fig.add_vrect(
            x0=threshold,
            x1=1.0,
            fillcolor="rgba(11,11,11,0.045)",
            line_width=0,
            layer="below",
            row=row,
            col=1,
        )
        fig.add_vline(
            x=threshold, line=dict(color=INK, width=2, dash="dash"), row=row, col=1
        )

    fig.add_annotation(
        x=threshold, y=1.02, yref="y domain", text=f"閾値 {threshold:.2f}",
        showarrow=False, yanchor="bottom",
        font=dict(color=INK, size=13, family=FONT), row=1, col=1,
    )
    fig.add_annotation(
        x=0.02, y=1.62, text="← OK判定", showarrow=False, xanchor="left",
        font=dict(color=INK_MUTED, size=12, family=FONT), row=2, col=1,
    )
    fig.add_annotation(
        x=0.98, y=1.62, text="NG判定 →", showarrow=False, xanchor="right",
        font=dict(color=INK_MUTED, size=12, family=FONT), row=2, col=1,
    )

    fig.update_layout(
        barmode="overlay",
        bargap=0.06,
        height=520,
        margin=dict(t=56, r=16, b=48, l=56),
        paper_bgcolor=SURFACE,
        plot_bgcolor=SURFACE,
        font=dict(family=FONT, color=INK_SECONDARY, size=13),
        legend=dict(
            orientation="h", yanchor="bottom", y=1.03, xanchor="left", x=0,
            bgcolor="rgba(0,0,0,0)",
        ),
        hovermode="closest",
    )
    fig.update_xaxes(
        range=[0, 1], gridcolor=GRID, zeroline=False, linecolor="#c3c2b7",
        tickfont=dict(color=INK_MUTED),
    )
    fig.update_xaxes(title_text="異常スコア（学習した良品からの隔たり）", row=2, col=1)
    fig.update_yaxes(
        title_text="件数", gridcolor=GRID, zeroline=False, showline=False,
        tickfont=dict(color=INK_MUTED), row=1, col=1,
    )
    fig.update_yaxes(
        range=[-0.55, 1.8], tickvals=[0, 1], ticktext=["NG（欠陥）", "OK（良品）"],
        showgrid=False, zeroline=False, tickfont=dict(color=INK_SECONDARY), row=2, col=1,
    )
    return fig


def scatter_figure(
    ok_scores: np.ndarray, ng_scores: np.ndarray, threshold: float
) -> go.Figure:
    """サンプル番号 × 異常スコアの散布図。閾値はここでは横線になる。

    OK を先、NG を後ろに並べる（塊で見えた方が分離の有無を掴みやすい）。
    """
    fig = go.Figure()

    offset = 0
    for scores, name, color in (
        (ok_scores, "OK（良品）", OK_COLOR),
        (ng_scores, "NG（欠陥）", NG_COLOR),
    ):
        fig.add_trace(
            go.Scatter(
                x=np.arange(offset, offset + len(scores)),
                y=scores,
                mode="markers",
                name=name,
                marker=dict(
                    color=color, size=10, line=dict(width=2, color=SURFACE), opacity=0.9
                ),
                hovertemplate=(
                    f"{name}<br>サンプル #%{{x}}<br>異常スコア %{{y:.3f}}<extra></extra>"
                ),
            )
        )
        offset += len(scores)

    fig.add_hrect(
        y0=threshold, y1=1.0, fillcolor="rgba(11,11,11,0.045)", line_width=0,
        layer="below",
    )
    fig.add_hline(y=threshold, line=dict(color=INK, width=2, dash="dash"))
    fig.add_annotation(
        x=1.0, y=threshold, xref="x domain", text=f"閾値 {threshold:.2f}",
        showarrow=False, xanchor="right", yanchor="bottom",
        font=dict(color=INK, size=13, family=FONT),
    )

    fig.update_layout(
        height=520,
        margin=dict(t=56, r=16, b=48, l=56),
        paper_bgcolor=SURFACE,
        plot_bgcolor=SURFACE,
        font=dict(family=FONT, color=INK_SECONDARY, size=13),
        legend=dict(
            orientation="h", yanchor="bottom", y=1.03, xanchor="left", x=0,
            bgcolor="rgba(0,0,0,0)",
        ),
        hovermode="closest",
    )
    fig.update_xaxes(
        title_text="サンプル番号", gridcolor=GRID, zeroline=False,
        linecolor="#c3c2b7", tickfont=dict(color=INK_MUTED),
    )
    fig.update_yaxes(
        title_text="異常スコア", range=[0, 1], gridcolor=GRID, zeroline=False,
        linecolor="#c3c2b7", tickfont=dict(color=INK_MUTED),
    )
    return fig


def render(data: AppData) -> None:
    hist_col, scatter_col = st.columns(2)
    with hist_col:
        st.caption("**分布で見る** — 閾値は縦線。左が OK 判定、右が NG 判定。")
        st.plotly_chart(
            score_figure(data.ok_scores, data.ng_scores, data.threshold, data.seed)
        )
    with scatter_col:
        st.caption("**1 サンプルずつ見る** — 閾値は横線。線より上が NG 判定。")
        st.plotly_chart(scatter_figure(data.ok_scores, data.ng_scores, data.threshold))

    left, right = st.columns(2)
    left.metric(
        "過検知（良品を止めた数）",
        f"{data.cm.fp} / {data.ok_spec.n} 件",
        help="良品なのに NG と判定された数。再検査・歩留まり低下につながる。",
    )
    right.metric(
        "見逃し（欠陥を通した数）",
        f"{data.cm.fn} / {data.ng_spec.n} 件",
        help="欠陥なのに OK と判定された数。後工程や市場への流出につながる。",
    )

    st.info(
        "**閾値を左（低く）へ動かすと** 見逃しは減るが、過検知が増える。\n\n"
        "**閾値を右（高く）へ動かすと** 過検知は減るが、見逃しが増える。\n\n"
        "両方を同時にゼロにできる閾値は、分布が重なっている限り存在しない。"
        "したがって閾値は数学的な最適値ではなく、**品質方針と生産条件で決める運用条件**である。"
    )
