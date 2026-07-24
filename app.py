"""AI異常検知の閾値・評価を説明する Streamlit アプリ（構想A）。

判定規則: 異常スコア >= 閾値 → NG判定
- タブ①: 疑似OK/NG分布 + 閾値スライダー
- タブ②: 混同行列 + Recall/Precision/F1
- タブ③: ROC / PR / F1 曲線 + 閾値決定モード
"""

from __future__ import annotations

import math
from pathlib import Path

import numpy as np
import plotly.graph_objects as go
import streamlit as st
from plotly.subplots import make_subplots

from core.curves import (
    THRESHOLD_MODES,
    average_precision,
    choose_threshold,
    roc_auc,
    sweep,
)
from core.distributions import SHAPES, DistributionSpec, generate_scores
from core.metrics import confusion_at, metrics_at

# --- 配色（dataviz 準拠。OK/NG は青/橙＝色覚多様性に安全なペア） -------------
OK_COLOR = "#2a78d6"
NG_COLOR = "#eb6834"
# 曲線用の第3色（aqua）。Recall=青 / Precision=橙 / F1=aqua で色覚に安全な3色。
F1_COLOR = "#1baf7a"
# 採用中の閾値マーカー（ROC / PR 上の赤い点）
MARKER_RED = "#d03b3b"
INK = "#0b0b0b"
INK_SECONDARY = "#52514e"
INK_MUTED = "#898781"
GRID = "#e1e0d9"
SURFACE = "#fcfcfb"

FONT = '"Meiryo", "Hiragino Sans", "Noto Sans JP", system-ui, sans-serif'

# 実行時 cwd に依存しないよう、このファイルからの相対で解決する
LOGO_PATH = Path(__file__).parent / "02.Logo_Images" / "TR_inc_logo.png"

st.set_page_config(
    page_title="AI異常検知：閾値と評価指標",
    page_icon="📊",
    layout="wide",
)
st.logo(str(LOGO_PATH), size="large")


# --- データ生成 -------------------------------------------------------------
def make_scores(
    ok: DistributionSpec, ng: DistributionSpec, seed: int
) -> tuple[np.ndarray, np.ndarray]:
    """OK/NG の異常スコアを生成する。

    シードを固定しておかないと閾値スライダーを動かすたびに分布が引き直され、
    「閾値だけを変えた」比較にならない。
    """
    rng = np.random.default_rng(seed)
    return generate_scores(ok, rng), generate_scores(ng, rng)


def distribution_controls(defaults: dict, key: str) -> DistributionSpec:
    """サイドバーに片側分の分布パラメータ入力を並べる。"""
    n = st.slider("サンプル数", 2, 300, defaults["n"], key=f"{key}_n")
    center = st.slider("中心位置", 0.0, 1.0, defaults["center"], 0.01, key=f"{key}_c")
    spread = st.slider("ばらつき", 0.01, 0.30, defaults["spread"], 0.01, key=f"{key}_s")
    shape = st.selectbox(
        "分布形状",
        list(SHAPES),
        format_func=SHAPES.get,
        key=f"{key}_shape",
    )
    n_outlier = st.slider("外れ値の数", 0, 20, 0, key=f"{key}_no")
    outlier_center = st.slider(
        "外れ値の位置", 0.0, 1.0, defaults["outlier_center"], 0.01, key=f"{key}_oc"
    )
    return DistributionSpec(
        n=n,
        center=center,
        spread=spread,
        shape=shape,
        n_outlier=n_outlier,
        outlier_center=outlier_center,
    )


# --- 図 ---------------------------------------------------------------------
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


def pct(value: float) -> str:
    return "—" if math.isnan(value) else f"{value * 100:.1f}%"


# --- Step2: ROC / PR / F1 曲線 ---------------------------------------------
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


# --- サイドバー -------------------------------------------------------------
st.sidebar.title("操作パネル")

threshold = st.sidebar.slider(
    "閾値", 0.0, 1.0, 0.44, 0.01, help="異常スコアがこの値以上なら NG と判定する"
)

# デフォルトは「サンプル100枚超・AUC≈0.97 の分離良好な検出器」。
# 既定閾値 0.44 で過検知・見逃しとも1割程度に収まる、実案件で見せても
# 不安を与えない「良い検出器」を初期状態にする（重なりを見たいときは
# サイドバーで中心を近づける／ばらつきを広げる）。
with st.sidebar.expander("OK（良品）分布", expanded=False):
    ok_spec = distribution_controls(
        {"n": 120, "center": 0.31, "spread": 0.09, "outlier_center": 0.75}, "ok"
    )
with st.sidebar.expander("NG（欠陥）分布", expanded=False):
    ng_spec = distribution_controls(
        {"n": 100, "center": 0.57, "spread": 0.10, "outlier_center": 0.25}, "ng"
    )

with st.sidebar.expander("乱数", expanded=False):
    if "seed" not in st.session_state:
        st.session_state.seed = 0
    st.write(f"シード: `{st.session_state.seed}`")
    if st.button("サンプルを引き直す", width="stretch"):
        st.session_state.seed += 1

ok_scores, ng_scores = make_scores(ok_spec, ng_spec, st.session_state.seed)
cm = confusion_at(ok_scores, ng_scores, threshold)
m = metrics_at(cm)

# --- 本体 -------------------------------------------------------------------
st.title("AI異常検知：閾値と評価指標")
st.caption(
    "AIは最初から OK / NG を答えているのではない。"
    "各サンプルに「学習した良品とどれくらい違うか」を表す**異常スコア**を付け、"
    "そのスコアが**閾値**を超えたものを NG と判定している。"
)

tab_threshold, tab_matrix, tab_curves = st.tabs(
    ["① 閾値とは何か", "② 混同行列と評価指標", "③ ROC / PR / F1 曲線"]
)

with tab_threshold:
    hist_col, scatter_col = st.columns(2)
    with hist_col:
        st.caption("**分布で見る** — 閾値は縦線。左が OK 判定、右が NG 判定。")
        st.plotly_chart(
            score_figure(ok_scores, ng_scores, threshold, st.session_state.seed)
        )
    with scatter_col:
        st.caption("**1 サンプルずつ見る** — 閾値は横線。線より上が NG 判定。")
        st.plotly_chart(scatter_figure(ok_scores, ng_scores, threshold))

    left, right = st.columns(2)
    left.metric(
        "過検知（良品を止めた数）",
        f"{cm.fp} / {ok_spec.n} 件",
        help="良品なのに NG と判定された数。再検査・歩留まり低下につながる。",
    )
    right.metric(
        "見逃し（欠陥を通した数）",
        f"{cm.fn} / {ng_spec.n} 件",
        help="欠陥なのに OK と判定された数。後工程や市場への流出につながる。",
    )

    st.info(
        "**閾値を左（低く）へ動かすと** 見逃しは減るが、過検知が増える。\n\n"
        "**閾値を右（高く）へ動かすと** 過検知は減るが、見逃しが増える。\n\n"
        "両方を同時にゼロにできる閾値は、分布が重なっている限り存在しない。"
        "したがって閾値は数学的な最適値ではなく、**品質方針と生産条件で決める運用条件**である。"
    )

with tab_matrix:
    matrix_col, _ = st.columns([2, 1])
    matrix_col.plotly_chart(confusion_figure(cm, threshold))
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

with tab_curves:
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

    sw = sweep(ok_scores, ng_scores)
    auc = roc_auc(ok_scores, ng_scores)
    ap = average_precision(ok_scores, ng_scores)
    prevalence = ng_spec.n / (ok_spec.n + ng_spec.n)

    t_star = choose_threshold(
        mode, sw, manual=threshold, fpr_cap=fpr_cap, cost_fp=cost_fp, cost_fn=cost_fn
    )
    t_f1max = choose_threshold("f1_max", sw, manual=threshold)
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
            "閾値を **下げる** と上がる。\n"
            "- **Precision（適合率）** … NG と判定したうち何割が本当に欠陥か。"
            "閾値を **上げる** と上がる。\n"
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
        "F1 曲線に重ねた **Recall（青）と Precision（橙）が交差するあたり** ＝"
        "両者のバランスが取れた位置で、**F1 は最大** になる。"
    )
