"""タブ④ 少数サンプルの問題 — 同じ検出器でも、選んだサンプルで結果が変わる。

タブ①〜③は「1 回測った結果」を見てきた。ここで見るのは
**その測り方で本当に測れているのか**。母集団（ラインの真の実力）から
何度も抜き取り、そのたびに閾値を決め直して評価する。
"""

from __future__ import annotations

from dataclasses import astuple

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st

from core.distributions import DistributionSpec
from core.intervals import rule_of_three, wilson
from core.sampling import (
    EVAL_OK_NG_RATIO,
    POOL_NG_SIZE,
    POOL_OK_SIZE,
    PRESET_CONDITIONS,
    condition_label,
    make_pool,
    pool_reference,
    run_conditions,
    summarize,
)
from ui.sidebar import AppData
from ui.theme import (
    F1_COLOR,
    FONT,
    GRID,
    INK,
    INK_MUTED,
    INK_SECONDARY,
    LARGE_N_COLOR,
    NG_COLOR,
    OK_COLOR,
    SMALL_N_COLOR,
    SURFACE,
    pct,
)

# 母集団は「ラインの真の実力」なので、サイドバーの「引き直す」では動かさない。
# 分布の設定を変えたときだけ変わればよいので、専用の固定シードを使う。
POOL_SEED = 20240724
# 反復実験のシード。固定して、同じ設定なら誰が開いても同じ図になるようにする。
EXPERIMENT_SEED = 0

# 箱ひげ図に出す指標（内部キー: (表示名, 色)）
BOX_METRICS = {
    "recall": ("Recall（検出率）", OK_COLOR),
    "f1": ("F1 Score", F1_COLOR),
    "specificity": ("Specificity（良品を通せた率）", NG_COLOR),
}


@st.cache_data(show_spinner="抜き取りを反復中…")
def _run_experiments(
    ok_params: tuple, ng_params: tuple, conditions: tuple, n_repeat: int
) -> tuple[pd.DataFrame, dict]:
    """母集団を作り、条件ごとに反復抽出する。設定が同じならキャッシュを返す。"""
    rng = np.random.default_rng(POOL_SEED)
    ok_pool = make_pool(DistributionSpec(*ok_params), POOL_OK_SIZE, rng)
    ng_pool = make_pool(DistributionSpec(*ng_params), POOL_NG_SIZE, rng)

    experiments = run_conditions(
        ok_pool, ng_pool, conditions, n_repeat, seed=EXPERIMENT_SEED
    )
    return experiments, pool_reference(ok_pool, ng_pool)


def _box_layout(fig: go.Figure, title: str, y_title: str) -> None:
    fig.update_layout(
        title=dict(text=title, font=dict(family=FONT, color=INK, size=15), x=0),
        height=420,
        margin=dict(t=52, r=16, b=52, l=60),
        paper_bgcolor=SURFACE,
        plot_bgcolor=SURFACE,
        font=dict(family=FONT, color=INK_SECONDARY, size=12),
        showlegend=False,
        boxgap=0.45,
    )
    fig.update_xaxes(
        title_text="評価に使ったサンプル数（左ほど少ない）", showgrid=False,
        linecolor="#c3c2b7", tickfont=dict(color=INK_SECONDARY, size=13),
    )
    fig.update_yaxes(
        title_text=y_title, gridcolor=GRID, zeroline=False, linecolor="#c3c2b7",
        tickfont=dict(color=INK_MUTED),
    )


def _reference_line(fig: go.Figure, value: float, text: str) -> None:
    """母集団での「正解」を示す横線。各試行はこれを少数サンプルから当てにいく。"""
    if np.isnan(value):
        return
    fig.add_hline(
        y=value, line=dict(color=INK, width=2, dash="dash"),
        annotation_text=text, annotation_position="top left",
        annotation_font=dict(color=INK, size=12, family=FONT),
    )


def _add_boxes(fig: go.Figure, experiments: pd.DataFrame, column: str, color: str) -> None:
    for condition, group in experiments.groupby("condition", sort=False):
        fig.add_trace(
            go.Box(
                y=group[column],
                name=condition,
                marker=dict(color=color, size=4, opacity=0.35),
                line=dict(color=color, width=1.8),
                fillcolor="rgba(0,0,0,0)",
                boxpoints="all",
                jitter=0.5,
                pointpos=0,
                hovertemplate=f"{condition}<br>%{{y:.3f}}<extra></extra>",
            )
        )


def threshold_box_figure(experiments: pd.DataFrame, ref_threshold: float) -> go.Figure:
    """条件ごとの「選ばれた閾値」の散らばり。点 1 つ = 1 回の抜き取り。"""
    fig = go.Figure()
    # 閾値はアプリ全体で黒い破線として描いてきたので、ここも無彩色で通す。
    _add_boxes(fig, experiments, "threshold", INK_SECONDARY)
    _reference_line(fig, ref_threshold, f"母集団での正解 {ref_threshold:.3f}")
    _box_layout(fig, "① 少数サンプルから決めた閾値", "F1 最大となった閾値")
    return fig


def metric_box_figure(
    experiments: pd.DataFrame, metric_key: str, ref_value: float
) -> go.Figure:
    """条件ごとの評価値の散らばり。"""
    label, color = BOX_METRICS[metric_key]
    fig = go.Figure()
    _add_boxes(fig, experiments, metric_key, color)
    _reference_line(fig, ref_value, f"母集団での真の値 {ref_value:.3f}")
    _box_layout(fig, f"② その少数サンプルで報告される {label}", label)
    fig.update_yaxes(range=[-0.04, 1.06])
    return fig


def overlay_figure(
    experiments: pd.DataFrame, small_label: str, large_label: str
) -> go.Figure:
    """少数条件と多数条件の閾値分布を重ねる。同系色の濃淡で「同じものの枚数違い」を示す。"""
    fig = go.Figure()
    bins = dict(start=0.0, end=1.0, size=0.02)
    for label, color in ((small_label, SMALL_N_COLOR), (large_label, LARGE_N_COLOR)):
        values = experiments.loc[experiments["condition"] == label, "threshold"]
        fig.add_trace(
            go.Histogram(
                x=values,
                name=label,
                marker=dict(color=color, line=dict(width=0)),
                opacity=0.75,
                xbins=bins,
                hovertemplate=f"{label}<br>閾値 %{{x}}<br>%{{y}} 回<extra></extra>",
            )
        )

    fig.update_layout(
        barmode="overlay",
        bargap=0.05,
        title=dict(
            text=f"③ 決まった閾値のばらつき: {small_label} と {large_label}",
            font=dict(family=FONT, color=INK, size=15), x=0,
        ),
        height=340,
        margin=dict(t=52, r=16, b=48, l=60),
        paper_bgcolor=SURFACE,
        plot_bgcolor=SURFACE,
        font=dict(family=FONT, color=INK_SECONDARY, size=12),
        legend=dict(
            orientation="h", yanchor="bottom", y=1.0, xanchor="right", x=1.0,
            bgcolor="rgba(0,0,0,0)",
        ),
    )
    fig.update_xaxes(
        title_text="F1 最大となった閾値", gridcolor=GRID, zeroline=False,
        linecolor="#c3c2b7", tickfont=dict(color=INK_MUTED),
    )
    fig.update_yaxes(
        title_text="その閾値が選ばれた回数", gridcolor=GRID, zeroline=False,
        showline=False, tickfont=dict(color=INK_MUTED),
    )
    return fig


def _summary_table(summary: pd.DataFrame) -> pd.DataFrame:
    """条件別サマリを表示用の文字列表へ整える。"""
    return pd.DataFrame({
        "評価サンプル数": summary["condition"],
        "閾値の中央値": summary["threshold_median"].map("{:.3f}".format),
        "閾値の5〜95%範囲": [
            f"{lo:.3f} 〜 {hi:.3f}（幅 {hi - lo:.3f}）"
            for lo, hi in zip(summary["threshold_p05"], summary["threshold_p95"])
        ],
        "Recall の中央値": summary["recall_median"].map(pct),
        "Recall の下位5%": summary["recall_p05"].map(pct),
        "Recall 100% と報告された割合": summary["recall_100_rate"].map(pct),
    })


def _confidence_table(summary: pd.DataFrame) -> pd.DataFrame:
    """「全数検出＝Recall 100%」を、NG 枚数ごとにどこまで信じてよいか。"""
    rows = []
    for _, row in summary.iterrows():
        n_ng = int(row["n_ng"])
        low, _ = wilson(n_ng, n_ng)
        upper = rule_of_three(n_ng)
        rows.append({
            "評価サンプル数": row["condition"],
            "NG 枚数": f"{n_ng} 枚",
            "全数検出したときの表示": "Recall 100%",
            "実際の Recall の95%下限": pct(low),
            "見逃し率の95%上限（3の法則）": (
                "評価不能" if np.isnan(upper) or upper >= 1.0 else pct(upper)
            ),
        })
    return pd.DataFrame(rows)


def render(data: AppData) -> None:
    st.caption(
        "ここまでは「1 回測った結果」を見てきた。このタブで見るのは "
        "**その測り方で本当に測れているのか**。"
        f"サイドバーの分布を**ラインの真の実力**（良品 {POOL_OK_SIZE:,} 個・"
        f"欠陥 {POOL_NG_SIZE:,} 個の母集団）とみなし、そこから決まった枚数だけ"
        "抜き取って、**そのサンプルだけで F1 最大の閾値を決めて評価する** — "
        "という現場の手順を何度も繰り返す。"
    )

    left, right = st.columns([3, 2])
    with left:
        chosen = st.multiselect(
            "比べる評価サンプル数",
            options=[condition_label(n_ok, n_ng) for n_ok, n_ng in PRESET_CONDITIONS],
            default=[condition_label(n_ok, n_ng) for n_ok, n_ng in PRESET_CONDITIONS],
            help=(
                f"評価用の OK:NG は {EVAL_OK_NG_RATIO}:1 に固定している。"
                "比率まで変えると、閾値の違いがサンプル数のせいなのか"
                "不良割合のせいなのか区別できなくなるため。"
            ),
        )
    with right:
        n_repeat = st.slider(
            "抜き取りを繰り返す回数", 20, 500, 100, 20,
            help="1 回 = 1 度の抜き取り検査。何度も繰り返して結果の散らばりを見る。",
        )

    conditions = tuple(
        (n_ok, n_ng)
        for n_ok, n_ng in PRESET_CONDITIONS
        if condition_label(n_ok, n_ng) in chosen
    )
    if len(conditions) < 2:
        st.info("比べる評価サンプル数を 2 つ以上選んでください。")
        return

    experiments, ref = _run_experiments(
        astuple(data.ok_spec), astuple(data.ng_spec), conditions, n_repeat
    )
    if experiments.empty:
        st.warning("この分布では閾値を決められる試行がありませんでした。")
        return
    summary = summarize(experiments)

    st.divider()
    st.markdown("#### この検出器の「真の実力」（母集団で測った値）")
    cols = st.columns(4)
    cols[0].metric("正解の閾値", f"{ref['threshold']:.3f}",
                   help="母集団全体で F1 が最大になる閾値。各試行はこれを当てにいっている。")
    cols[1].metric("真の Recall", pct(ref["recall"]))
    cols[2].metric("真の Specificity", pct(ref["specificity"]))
    cols[3].metric("真の F1", f"{ref['f1']:.3f}")
    st.caption(
        "この 4 つが「答え」。以下の各試行は、少数のサンプルだけから"
        "この値を推し量ろうとしている。**答えからどれだけずれるか**が問題になる。"
    )

    st.divider()

    box_left, box_right = st.columns(2)
    with box_left:
        st.plotly_chart(threshold_box_figure(experiments, ref["threshold"]))
    with box_right:
        metric_key = st.radio(
            "見る指標",
            list(BOX_METRICS),
            format_func=lambda k: BOX_METRICS[k][0],
            horizontal=True,
            key="sampling_metric",
        )
        st.plotly_chart(
            metric_box_figure(experiments, metric_key, ref[metric_key])
        )
    st.caption(
        f"点 1 つが 1 回の抜き取り（各条件 {n_repeat} 回）。"
        "**黒い破線が正解**。左の条件ほど箱が縦に長い＝"
        "同じラインを測っているのに、たまたま選んだサンプルで答えが変わっている。"
    )

    st.plotly_chart(
        overlay_figure(experiments, summary["condition"].iloc[0],
                       summary["condition"].iloc[-1])
    )

    st.divider()
    st.markdown("#### 条件別のまとめ")
    st.dataframe(_summary_table(summary), hide_index=True, width="stretch")

    smallest, largest = summary.iloc[0], summary.iloc[-1]
    small_width = smallest["threshold_p95"] - smallest["threshold_p05"]
    large_width = largest["threshold_p95"] - largest["threshold_p05"]
    st.info(
        f"**{smallest['condition']}** では、閾値の 9 割が "
        f"{smallest['threshold_p05']:.2f}〜{smallest['threshold_p95']:.2f}（幅 {small_width:.2f}）"
        f"に散らばり、{pct(smallest['recall_100_rate'])} の試行が "
        "**「Recall 100%、見逃しゼロ」と報告**した。\n\n"
        f"**{largest['condition']}** では幅が {large_width:.2f} まで狭まり、"
        f"Recall 100% と報告されたのは {pct(largest['recall_100_rate'])} だけになる。\n\n"
        f"どちらも同じ検出器で、真の Recall は {pct(ref['recall'])} のまま変わっていない。"
        "**サンプルを増やして良くなるのは検出器ではなく、測定の確かさ**である。"
    )

    st.divider()
    st.markdown("#### 「見逃しゼロ」をどこまで信じてよいか")
    st.caption(
        "同じ「Recall 100%」でも、2 枚中 2 枚と 60 枚中 60 枚では意味が違う。"
        "全数検出できたとき、真の Recall がどこまで低い可能性が残るかを示す。"
    )
    st.dataframe(_confidence_table(summary), hide_index=True, width="stretch")
    st.caption(
        "「3の法則」… n 件で 1 件も見逃さなかったとき、見逃し率の 95% 上限は"
        " およそ **3 ÷ n**。n が小さいと上限が 100% を下回らず、"
        "**性能を評価できていない**ことになる。"
    )

    st.warning(
        "**初期評価と量産の性能保証は別物である。** 少数サンプルでの"
        "「見逃し 0 件」は、*確認した範囲では見逃しが無かった* という事実であって、"
        "*量産でも見逃さない* という保証ではない。"
        "量産へ持ち込む閾値は、少数サンプルで決めた暫定値ではなく、"
        "十分な枚数の独立した評価データを通過した固定値である必要がある。"
    )
