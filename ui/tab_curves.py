"""タブ③ ROC / PR / F1 曲線と、閾値の決め方。"""

from __future__ import annotations

import math

import numpy as np
import plotly.graph_objects as go
import streamlit as st

from core.curves import (
    THRESHOLD_MODES,
    average_precision,
    choose_threshold,
    roc_auc,
    sweep,
)
from core.metrics import confusion_at, metrics_at
from ui.sidebar import AppData
from ui.theme import (
    F1_COLOR,
    FONT,
    GRID,
    INK,
    INK_MUTED,
    INK_SECONDARY,
    MARKER_RED,
    NG_COLOR,
    OK_COLOR,
    SURFACE,
    pct,
)


def _curve_layout(fig: go.Figure, title: str) -> None:
    """3つの曲線図で共通のレイアウト。"""
    fig.update_layout(
        title=dict(text=title, font=dict(family=FONT, color=INK, size=15), x=0),
        height=380,
        margin=dict(t=48, r=16, b=44, l=52),
        paper_bgcolor=SURFACE,
        plot_bgcolor=SURFACE,
        font=dict(family=FONT, color=INK_SECONDARY, size=12),
        legend=dict(
            orientation="h", yanchor="bottom", y=1.0, xanchor="right", x=1.0,
            bgcolor="rgba(0,0,0,0)",
        ),
        hovermode="closest",
    )
    fig.update_xaxes(gridcolor=GRID, zeroline=False, linecolor="#c3c2b7",
                     tickfont=dict(color=INK_MUTED))
    fig.update_yaxes(gridcolor=GRID, zeroline=False, linecolor="#c3c2b7",
                     tickfont=dict(color=INK_MUTED))


def _threshold_marker(fig: go.Figure, x: float, y: float, name: str) -> None:
    """現在採用中の閾値の位置を示す赤い点。"""
    fig.add_trace(
        go.Scatter(
            x=[x], y=[y], mode="markers", name=name,
            marker=dict(color=MARKER_RED, size=14, symbol="circle",
                        line=dict(width=2, color=SURFACE)),
            hovertemplate=f"{name}<br>(%{{x:.3f}}, %{{y:.3f}})<extra></extra>",
        )
    )


def roc_figure(sw, auc: float, fpr_star: float, recall_star: float) -> go.Figure:
    """ROC 曲線。横軸=FPR（過検知率）, 縦軸=TPR（Recall）。"""
    fig = go.Figure()
    fig.add_trace(
        go.Scatter(
            x=[0, 1], y=[0, 1], mode="lines", name="ランダム（AUC 0.5）",
            line=dict(color=INK_MUTED, width=1.5, dash="dot"), hoverinfo="skip",
        )
    )
    fig.add_trace(
        go.Scatter(
            x=sw.fpr, y=sw.recall, mode="lines", name="ROC",
            line=dict(color=OK_COLOR, width=2.5),
            hovertemplate="FPR %{x:.3f}<br>Recall %{y:.3f}<extra></extra>",
        )
    )
    _threshold_marker(fig, fpr_star, recall_star, "採用中の閾値")
    auc_text = "—" if math.isnan(auc) else f"{auc:.3f}"
    fig.add_annotation(
        x=0.97, y=0.05, xref="x", yref="y", text=f"AUC = {auc_text}",
        showarrow=False, xanchor="right",
        font=dict(color=INK, size=14, family=FONT),
        bgcolor="rgba(252,252,251,0.8)",
    )
    _curve_layout(fig, "ROC 曲線")
    fig.update_xaxes(title_text="FPR（過検知率）", range=[-0.02, 1.02])
    fig.update_yaxes(title_text="TPR（Recall＝検出率）", range=[-0.02, 1.02])
    return fig


def pr_figure(sw, ap: float, prevalence: float,
              recall_star: float, precision_star: float) -> go.Figure:
    """PR 曲線。横軸=Recall, 縦軸=Precision。"""
    mask = ~np.isnan(sw.precision)
    fig = go.Figure()
    fig.add_hline(
        y=prevalence, line=dict(color=INK_MUTED, width=1.5, dash="dot"),
        annotation_text=f"欠陥割合 {prevalence:.2f}", annotation_position="bottom right",
        annotation_font=dict(color=INK_MUTED, size=11, family=FONT),
    )
    fig.add_trace(
        go.Scatter(
            x=sw.recall[mask], y=sw.precision[mask], mode="lines", name="PR",
            line=dict(color=NG_COLOR, width=2.5),
            hovertemplate="Recall %{x:.3f}<br>Precision %{y:.3f}<extra></extra>",
        )
    )
    if not math.isnan(precision_star):
        _threshold_marker(fig, recall_star, precision_star, "採用中の閾値")
    ap_text = "—" if math.isnan(ap) else f"{ap:.3f}"
    fig.add_annotation(
        x=0.03, y=0.05, xref="x", yref="y", text=f"AP = {ap_text}",
        showarrow=False, xanchor="left",
        font=dict(color=INK, size=14, family=FONT),
        bgcolor="rgba(252,252,251,0.8)",
    )
    _curve_layout(fig, "PR 曲線")
    fig.update_xaxes(title_text="Recall（検出率）", range=[-0.02, 1.02])
    fig.update_yaxes(title_text="Precision（適合率）", range=[-0.02, 1.02])
    return fig


def _threshold_vline(fig: go.Figure, t_star: float) -> None:
    """閾値ごとの図に「採用中の閾値」の縦線を引く。"""
    fig.add_vline(x=t_star, line=dict(color=INK, width=2, dash="dash"))
    fig.add_annotation(
        x=t_star, y=1.02, yref="y", text=f"採用中の閾値 {t_star:.2f}",
        showarrow=False, yanchor="bottom", xanchor="center",
        font=dict(color=INK, size=12, family=FONT),
    )


def f1_figure(
    sw, t_star: float, t_f1max: float, f1max_value: float,
    show_pr: bool, show_f1: bool,
) -> go.Figure:
    """閾値ごとの指標曲線。show_pr で Recall・Precision、show_f1 で F1 を出し分ける。"""
    fig = go.Figure()
    # Recall・Precision は先に、F1 より細く描いて F1 を前面に立たせる
    if show_pr:
        for values, name, color in (
            (sw.recall, "Recall（検出率）", OK_COLOR),
            (sw.precision, "Precision（適合率）", NG_COLOR),
        ):
            fig.add_trace(
                go.Scatter(
                    x=sw.thresholds, y=values, mode="lines", name=name,
                    line=dict(color=color, width=2),
                    hovertemplate=f"{name}<br>閾値 %{{x:.3f}}<br>%{{y:.3f}}<extra></extra>",
                )
            )
    if show_f1:
        fig.add_trace(
            go.Scatter(
                x=sw.thresholds, y=sw.f1, mode="lines", name="F1",
                line=dict(color=F1_COLOR, width=3),
                hovertemplate="F1<br>閾値 %{x:.3f}<br>%{y:.3f}<extra></extra>",
            )
        )
        if not math.isnan(f1max_value):
            fig.add_trace(
                go.Scatter(
                    x=[t_f1max], y=[f1max_value], mode="markers", name="F1 最大",
                    marker=dict(color=F1_COLOR, size=13, symbol="diamond",
                                line=dict(width=1.5, color=SURFACE)),
                    hovertemplate=f"F1 最大<br>閾値 {t_f1max:.3f}<br>F1 {f1max_value:.3f}"
                                  "<extra></extra>",
                )
            )
    _threshold_vline(fig, t_star)
    _curve_layout(fig, "F1 曲線（閾値ごと）")
    fig.update_xaxes(title_text="閾値", range=[0, 1])
    fig.update_yaxes(
        title_text="F1 Score" if show_f1 and not show_pr else "指標の値",
        range=[-0.02, 1.08],
    )
    return fig


def render(data: AppData) -> None:
    ok_scores, ng_scores = data.ok_scores, data.ng_scores

    st.caption(
        "分布は同じまま、**閾値をどう決めるか**で曲線上の位置と各指標がどう動くかを見る。"
        "ROC / PR の★、F1 曲線の縦線は、下で選んだ決め方で決まる **採用中の閾値** を指す。"
    )

    mode = st.radio(
        "閾値の決め方",
        list(THRESHOLD_MODES),
        format_func=THRESHOLD_MODES.get,
        horizontal=True,
        key="mode",
        help="「手動設定」はサイドバーの閾値スライダーをそのまま使う。",
    )

    fpr_cap, cost_fp, cost_fn = 0.05, 1.0, 10.0
    if mode == "fpr_cap":
        # スライダーは % 単位（0〜50%）。内部の FPR は 0〜1 の割合なので /100 する。
        fpr_cap_pct = st.slider(
            "許容する過検知率（FPR）の上限", 0, 50, 5, 1,
            format="%d%%",
            help="良品をこの割合まで NG にしてよい、という上限。低いほど過検知に厳しい。",
        )
        fpr_cap = fpr_cap_pct / 100.0
    elif mode == "cost_min":
        c1, c2 = st.columns(2)
        cost_fn = c1.number_input("見逃し 1 件のコスト", 0.0, 1000.0, 10.0, 1.0,
                                  help="欠陥を市場へ流す損失。過検知より大きく設定するのが普通。")
        cost_fp = c2.number_input("過検知 1 件のコスト", 0.0, 1000.0, 1.0, 1.0,
                                  help="良品を止めて再検査する損失。")
        st.caption(
            "※ ここで最小化しているのは **この評価データの中での件数コスト**"
            f"（{cost_fp:.0f}×過検知件数 ＋ {cost_fn:.0f}×見逃し件数）。"
            "**実ラインの期待コストが最小になる閾値とは一致しない。** "
            "このデータの欠陥割合は "
            f"{data.ng_spec.n / (data.ok_spec.n + data.ng_spec.n):.0%} だが、"
            "実ラインの不良率は"
            "多くの場合 数% 以下で、桁が違うためである。"
            "実運用の閾値を決めるときは、想定不良率で重みを付け直す必要がある"
            "（同じ話をタブ④が F1 について扱っている）。"
        )

    sw = sweep(ok_scores, ng_scores)
    auc = roc_auc(ok_scores, ng_scores)
    ap = average_precision(ok_scores, ng_scores)
    prevalence = data.ng_spec.n / (data.ok_spec.n + data.ng_spec.n)

    t_star = choose_threshold(
        mode, sw, manual=data.threshold, fpr_cap=fpr_cap, cost_fp=cost_fp, cost_fn=cost_fn
    )
    t_f1max = choose_threshold("f1_max", sw, manual=data.threshold)
    f1max_value = float(np.nanmax(sw.f1)) if not np.all(np.isnan(sw.f1)) else math.nan

    cm_star = confusion_at(ok_scores, ng_scores, t_star)
    m_star = metrics_at(cm_star)
    fpr_star = cm_star.fp / (cm_star.fp + cm_star.tn)

    summary = st.columns(5)
    summary[0].metric("採用中の閾値", f"{t_star:.3f}",
                      help="下で選んだ決め方から決まる閾値")
    summary[1].metric("過検知（FP）", f"{cm_star.fp} 件")
    summary[2].metric("見逃し（FN）", f"{cm_star.fn} 件")
    summary[3].metric("Recall", pct(m_star["recall"]))
    summary[4].metric("Precision", pct(m_star["precision"]))

    if mode != "manual":
        st.caption(
            f"※ この閾値 {t_star:.3f} はタブ③内だけの検討用。"
            "タブ①②とサイドバーは引き続きスライダーの値を使う。"
        )

    left, right = st.columns(2)
    left.plotly_chart(roc_figure(sw, auc, fpr_star, m_star["recall"]))
    right.plotly_chart(pr_figure(sw, ap, prevalence, m_star["recall"], m_star["precision"]))

    tcol1, tcol2 = st.columns(2)
    show_f1 = tcol1.toggle("F1 を表示", value=True, key="show_f1")
    show_pr = tcol2.toggle("Recall・Precision を表示", value=True, key="show_pr")
    if not show_f1 and not show_pr:
        st.info("表示する曲線を選んでください（F1 / Recall・Precision）。")
    else:
        st.plotly_chart(
            f1_figure(sw, t_star, t_f1max, f1max_value, show_pr, show_f1)
        )

    st.info(
        "**F1 最大**は数学的にバランスの良い点だが、製造現場での最適とは限らない。"
        "**見逃しゼロ優先**にすると過検知が跳ね上がり、**過検知率の上限**を厳しくすると"
        "見逃しが増える。どれを採るかは、見逃しと過検知どちらの損失が大きいかという"
        "**品質方針**で決まる。"
    )

    st.divider()
    st.markdown("#### F1 Score と Recall・Precision の関係")

    eq_col, txt_col = st.columns([1, 1])
    with eq_col:
        st.latex(r"\text{Recall} = \frac{TP}{TP + FN}")
        st.latex(r"\text{Precision} = \frac{TP}{TP + FP}")
        st.latex(
            r"F_1 = 2 \cdot \frac{\text{Precision} \cdot \text{Recall}}"
            r"{\text{Precision} + \text{Recall}}"
        )
    with txt_col:
        st.markdown(
            "- **Recall（検出率）** … 欠陥のうち何割を捕まえたか。"
            "閾値を **下げる** と必ず上がる。\n"
            "- **Precision（適合率）** … NG と判定したうち何割が本当に欠陥か。"
            "閾値を **上げる** と上がりやすい"
            "（ただし NG 判定から外れるのが過検知か真の欠陥かで上下するので、"
            "**単調に上がるとは限らない**）。\n"
            "- **F1 Score** … この 2 つの **調和平均**。"
            "**両方が高いときだけ** 高くなる。"
        )

    st.markdown(
        "調和平均は **低い方に強く引っ張られる**。"
        "たとえば Precision = 1.0、Recall = 0.1 のとき："
    )
    st.latex(
        r"\text{算術平均} = \frac{1.0 + 0.1}{2} = 0.55"
        r"\quad \text{に対して} \quad "
        r"F_1 = 2 \cdot \frac{1.0 \times 0.1}{1.0 + 0.1} \approx 0.18"
    )
    st.markdown(
        "だから「見逃しゼロだが過検知だらけ（Recall 高・Precision 低）」も、"
        "「過検知ゼロだが見逃し多数（Precision 高・Recall 低）」も F1 は伸びない。"
        "このデータでは、F1 曲線に重ねた "
        "**Recall（青）と Precision（橙）が交差するあたり**で "
        "**F1 が最大**になっている。ただし F1 の定義は "
        "Precision ＝ Recall を要求しないので、"
        "**交点が必ず F1 最大点になるわけではない**。"
    )
