"""タブ④ 少数サンプルの問題 — 同じ検出器でも、選んだサンプルで結果が変わる。

**このタブだけ、見ているデータがタブ①〜③と違う。**

タブ①〜③ … サイドバーで作った 1 回分の抜き取り（既定 220 枚）を、そのまま
             全データとして扱う。
タブ④     … その手前に「ラインの真の実力」＝母集団を置き、そこから何度も
             抜き取る。閾値は毎回その抜き取りだけから決め直す。

この違いを黙って切り替えると誤解を生むので、画面でも最初に母集団を見せ、
タブ①〜③との対比表を置いてから実験に入る。
"""

from __future__ import annotations

import re
import unicodedata
from dataclasses import astuple

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st

from core.curves import roc_auc, sweep
from core.distributions import DistributionSpec
from core.intervals import rule_of_three, wilson
from core.metrics import confusion_at, metrics_at
from core.sampling import (
    DEFAULT_OK_NG_RATIO,
    DEFAULT_POOL_OK_SIZE,
    POOL_OK_SIZE_OPTIONS,
    PRESET_OK_COUNTS,
    conditions_for,
    make_pool,
    max_eval_ok,
    pool_ng_size,
    pool_reference,
    run_conditions,
    summarize,
)
from ui.sidebar import AppData

# 「正解の閾値がどう決まったか」は、タブ②③で使ったのと同じ図を母集団に
# 当てて見せる。同じ道具の使い回しであること自体が説明になるので、
# 「図はそのタブが持つ」という原則をここだけ意図的に外して借りている。
from ui.tab_curves import f1_figure, roc_figure
from ui.tab_matrix import confusion_figure
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

# 反復実験のシード。固定して、同じ設定なら誰が開いても同じ図になるようにする。
EXPERIMENT_SEED = 0

# 「任意の枚数を追加」の上限。母集団を最大にしたときの抜き取り上限に合わせる。
# 実際に使える上限は母集団サイズで変わるが、可変にするとウィジェット ID が
# 変わって入力がリセットされるため、ここは固定にして採否は入力後に判定する。
ADDABLE_MAX = max_eval_ok(max(POOL_OK_SIZE_OPTIONS))

# 箱ひげ図に出す指標（内部キー: (表示名, 色)）
BOX_METRICS = {
    "recall": ("Recall（検出率）", OK_COLOR),
    "f1": ("F1 Score", F1_COLOR),
    "specificity": ("Specificity（良品を通せた率）", NG_COLOR),
}


@st.cache_data(show_spinner=False)
def _build_pool(
    ok_params: tuple, ng_params: tuple, ratio: int, pool_ok_size: int, pool_seed: int
) -> tuple[np.ndarray, np.ndarray, dict]:
    """母集団を作り、その F1最大閾値と指標（＝「正解」）も一緒に返す。"""
    rng = np.random.default_rng(pool_seed)
    ok_pool = make_pool(DistributionSpec(*ok_params), pool_ok_size, rng)
    ng_pool = make_pool(
        DistributionSpec(*ng_params), pool_ng_size(ratio, pool_ok_size), rng
    )
    return ok_pool, ng_pool, pool_reference(ok_pool, ng_pool)


@st.cache_data(show_spinner="抜き取りを反復中…")
def _run_experiments(
    ok_pool: np.ndarray, ng_pool: np.ndarray, conditions: tuple, n_repeat: int
) -> pd.DataFrame:
    return run_conditions(ok_pool, ng_pool, conditions, n_repeat, seed=EXPERIMENT_SEED)


@st.cache_data(show_spinner=False)
def _pool_sweep(ok_pool: np.ndarray, ng_pool: np.ndarray):
    """母集団に対する閾値スイープと AUC。タブ③と同じ core.curves を使う。"""
    return sweep(ok_pool, ng_pool), roc_auc(ok_pool, ng_pool)


def _derivation(ok_pool: np.ndarray, ng_pool: np.ndarray, threshold: float) -> None:
    """「正解の閾値」がどう決まったかを ROC → 混同行列 → F1 の順で見せる。"""
    sw, auc = _pool_sweep(ok_pool, ng_pool)
    cm = confusion_at(ok_pool, ng_pool, threshold)
    m = metrics_at(cm)
    fpr = cm.fp / (cm.fp + cm.tn)
    f1_max = float(np.nanmax(sw.f1)) if not np.all(np.isnan(sw.f1)) else np.nan

    st.caption(
        "タブ②③で使ったのと同じ 3 つの図を、今度は**母集団に対して**当てている。"
        "違うのは対象だけで、やっていることは同じ。"
    )

    roc_col, cm_col = st.columns(2)
    with roc_col:
        st.markdown("**ステップ1: この検出器の実力を見る（ROC 曲線）**")
        st.plotly_chart(roc_figure(sw, auc, fpr, m["recall"]))
        st.caption(
            f"AUC = {auc:.3f}。閾値をどこに置くかとは無関係な、"
            "**検出器そのものの実力**。赤い点が下で選ばれる正解の閾値の位置。"
        )
    with cm_col:
        st.markdown("**ステップ2: ある閾値で切った内訳を見る（混同行列）**")
        st.plotly_chart(
            confusion_figure(
                cm, threshold, height=380,
                title=f"母集団を閾値 {threshold:.3f} で切った結果",
            )
        )
        st.caption(
            f"過検知 {cm.fp:,} 件・見逃し {cm.fn:,} 件。"
            "閾値を動かすとこの 4 つの数が動き、そこから Recall・Precision、"
            "そして **F1 が計算される**。"
        )

    f1_col, note_col = st.columns([2, 1])
    with f1_col:
        st.markdown("**ステップ3: 全部の閾値で F1 を計算し、頂点を採る（F1 曲線）**")
        st.plotly_chart(f1_figure(sw, threshold, threshold, f1_max, True, True))
    with note_col:
        st.write("")
        st.markdown(
            "閾値を 0 から 1 まで動かし、**そのつどステップ2 の混同行列を作って "
            "F1 を計算**したものがこの曲線。\n\n"
            f"F1 が最大（{f1_max:.3f}）になったのが **閾値 {threshold:.3f}** で、"
            "これを「正解」と呼んでいる。\n\n"
            "Recall（青）は閾値を下げるほど上がり、Precision（橙）は上げるほど"
            "上がる。**両方が釣り合うあたり**が頂点になる。"
        )
    st.info(
        "**「正解」と呼んでいるのは、あくまで F1 を最大にする閾値である。** "
        "見逃しを何としても避けたいなら、もっと左（低い閾値）を選ぶことになる。"
        "どこを選ぶかは品質方針の問題で、数学が決めるわけではない — "
        "これはタブ③で見たとおり。ここでは**少数サンプルでも同じ狙いを"
        "定められるか**を問題にしているので、狙いを F1 最大に固定している。"
    )


def _parse_ok_counts(raw_values, limit: int) -> tuple[list[int], list[str]]:
    """multiselect の選択（既定は int、手入力は文字列）を OK 枚数へ揃える。

    戻り値は (採用した枚数, 却下した入力の説明)。

    手入力の受け取りは**寛容にする**。`accept_new_options` で入った値は、
    次の再描画で format_func を通り「OK 150 枚」という表示文字列に化けて
    戻ってくる。`int()` だけで受けると、追加できたはずの枚数が次の操作で
    消える。数字さえ含んでいれば拾う — 「150」「150枚」「OK 150 枚」は
    すべて 150 として扱う。
    """
    counts: list[int] = []
    rejected: list[str] = []
    for value in raw_values:
        # IME が入ったままだと全角数字になりやすい。桁区切りのカンマも来る。
        text = unicodedata.normalize("NFKC", str(value)).replace(",", "")
        digits = re.findall(r"\d+", text)
        if len(digits) != 1:
            rejected.append(f"「{value}」から枚数を読み取れません（例: 150）")
            continue
        n = int(digits[0])
        if n < 2:
            rejected.append(f"{n} 枚は少なすぎます（2 枚以上）")
        elif n > limit:
            rejected.append(f"{n} 枚は多すぎます（母集団の半分 {limit:,} 枚まで）")
        else:
            counts.append(n)
    return sorted(set(counts)), rejected


# --- 図 ---------------------------------------------------------------------
def pool_figure(
    ok_pool: np.ndarray, ng_pool: np.ndarray, threshold: float, normalize: bool
) -> go.Figure:
    """母集団そのものの分布。タブ①のヒストグラムと同じ見方で、枚数だけが違う。"""
    fig = go.Figure()
    histnorm = "percent" if normalize else None
    for scores, name, color in (
        (ok_pool, "OK（良品）", OK_COLOR),
        (ng_pool, "NG（欠陥）", NG_COLOR),
    ):
        fig.add_trace(
            go.Histogram(
                x=scores,
                name=f"{name} {len(scores):,} 個",
                marker=dict(color=color, line=dict(width=0)),
                opacity=0.72,
                xbins=dict(start=0.0, end=1.0, size=0.02),
                histnorm=histnorm,
                hovertemplate=f"{name}<br>異常スコア %{{x}}<br>%{{y}}<extra></extra>",
            )
        )

    if not np.isnan(threshold):
        fig.add_vrect(
            x0=threshold, x1=1.0, fillcolor="rgba(11,11,11,0.045)",
            line_width=0, layer="below",
        )
        fig.add_vline(x=threshold, line=dict(color=INK, width=2, dash="dash"))
        fig.add_annotation(
            x=threshold, y=1.02, yref="y domain",
            text=f"母集団での正解の閾値 {threshold:.3f}",
            showarrow=False, yanchor="bottom",
            font=dict(color=INK, size=13, family=FONT),
        )

    fig.update_layout(
        barmode="overlay",
        bargap=0.05,
        height=380,
        margin=dict(t=56, r=16, b=48, l=60),
        paper_bgcolor=SURFACE,
        plot_bgcolor=SURFACE,
        font=dict(family=FONT, color=INK_SECONDARY, size=13),
        legend=dict(
            orientation="h", yanchor="bottom", y=1.03, xanchor="left", x=0,
            bgcolor="rgba(0,0,0,0)",
        ),
    )
    fig.update_xaxes(
        title_text="異常スコア", range=[0, 1], gridcolor=GRID, zeroline=False,
        linecolor="#c3c2b7", tickfont=dict(color=INK_MUTED),
    )
    fig.update_yaxes(
        title_text="各群に占める割合(%)" if normalize else "個数",
        gridcolor=GRID, zeroline=False, showline=False,
        tickfont=dict(color=INK_MUTED),
    )
    return fig


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


# --- 表 ---------------------------------------------------------------------
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


# --- 画面 -------------------------------------------------------------------
def _pool_controls(data: AppData) -> tuple[int, int]:
    """母集団の作り方（大きさ・OK:NG比・引き直し）。戻り値は (比, OK側サイズ)。"""
    if "pool_seed" not in st.session_state:
        st.session_state.pool_seed = 0

    left, mid, right = st.columns([2, 2, 2])
    with left:
        pool_ok_size = st.select_slider(
            "母集団の大きさ（良品側）",
            options=list(POOL_OK_SIZE_OPTIONS),
            value=DEFAULT_POOL_OK_SIZE,
            format_func=lambda n: f"{n:,} 個",
            key="pool_ok_size",
            help="このラインが作る良品の総数とみなす数。欠陥側は右の比から決まる。",
        )
    with mid:
        ratio = st.slider(
            "母集団と評価データの OK:NG 比", 1, 20, DEFAULT_OK_NG_RATIO, 1,
            format="%d : 1",
            key="pool_ratio",
            help=(
                "良品と欠陥の比。母集団と抜き取りで同じ比を使う（違う比にすると、"
                "母集団で決めた閾値が抜き取りの「正解」でなくなる）。"
            ),
        )
    with right:
        st.write("")
        if st.button(
            "母集団を引き直す", width="stretch",
            help=(
                "分布の設定はそのままに、別の母集団を作り直す＝「同じ実力の別ライン」。"
                "結論が特定の母集団に依存していないことを確かめられる。"
            ),
        ):
            st.session_state.pool_seed += 1
        st.caption(f"母集団シード: `{st.session_state.pool_seed}`")

    return ratio, pool_ok_size


def _eval_controls(limit: int) -> tuple[list, int]:
    """比べる評価サンプル数と反復回数。戻り値は (multiselect の生の選択, 反復回数)。

    枚数の追加口は 2 つ用意する。枠に直接打つ方法（`accept_new_options`）は
    「OK 150 枚」と表示されている一覧に対して `150` としか打てず気づきにくいので、
    数値入力＋ボタンという明示的な導線を主にする。
    """
    if "extra_ok_counts" not in st.session_state:
        st.session_state.extra_ok_counts = []
    if "eval_ok_counts" not in st.session_state:
        st.session_state.eval_ok_counts = list(PRESET_OK_COUNTS)

    list_col, add_col, repeat_col = st.columns([3, 2, 2])

    # 追加処理は multiselect より前に置くこと。ウィジェット生成後に
    # session_state["eval_ok_counts"] を書き換えると Streamlit が例外を投げる。
    with add_col:
        num_col, button_col = st.columns([3, 2])
        # max_value に可変値を入れるとウィジェット ID が変わって入力が
        # リセットされるので、上限は固定にして採否は _parse_ok_counts で判定する。
        new_count = num_col.number_input(
            "任意の枚数を追加", min_value=2, max_value=ADDABLE_MAX,
            value=150, step=10, key="new_ok_count",
        )
        button_col.markdown("<div style='height:28px'></div>", unsafe_allow_html=True)
        if button_col.button("＋ 追加", width="stretch"):
            n = int(new_count)
            if n > limit:
                st.warning(f"{n} 枚は多すぎます（母集団の半分 {limit:,} 枚まで）")
            else:
                if n not in PRESET_OK_COUNTS:
                    st.session_state.extra_ok_counts = sorted(
                        set(st.session_state.extra_ok_counts) | {n}
                    )
                if n not in st.session_state.eval_ok_counts:
                    st.session_state.eval_ok_counts = [
                        *st.session_state.eval_ok_counts, n
                    ]

    with repeat_col:
        n_repeat = st.slider(
            "抜き取りを繰り返す回数", 20, 500, 100, 20,
            key="eval_repeat",
            help="1 回 = 1 度の抜き取り検査。何度も繰り返して結果の散らばりを見る。",
        )

    with list_col:
        # help に母集団サイズなど変わる値を入れないこと。ウィジェット ID が
        # 変わって選択がリセットされる（上限は下の caption 側で伝える）。
        options = sorted(set(PRESET_OK_COUNTS) | set(st.session_state.extra_ok_counts))
        raw_counts = st.multiselect(
            "比べる評価サンプル数（OK 枚数）",
            options=options,
            format_func=lambda n: f"OK {n} 枚",
            accept_new_options=True,
            key="eval_ok_counts",
            help="不要な条件は各チップの × で外せる。NG 枚数は上の OK:NG 比から決まる。",
        )

    st.caption(
        f"**一覧に無い枚数を試すには** … 右の「任意の枚数を追加」に数字を入れて "
        "**＋ 追加**（例: 150 → OK150 枚の条件が加わる）。"
        "枠内に直接 `150` と打って Enter でも入る（**数字だけ**。"
        "「OK150枚」のように打っても数字を拾う）。"
        f"1 条件あたり OK 2 〜 {limit:,} 枚まで（母集団の半分が上限）。"
    )
    return raw_counts, n_repeat


def _comparison_table(data: AppData, ratio: int, pool_ok_size: int) -> None:
    """タブ①〜③と何が違うのかを正面から並べる。"""
    sample_total = data.ok_spec.n + data.ng_spec.n
    sample_prevalence = data.ng_spec.n / sample_total
    pool_ng = pool_ng_size(ratio, pool_ok_size)
    pool_prevalence = pool_ng / (pool_ok_size + pool_ng)

    st.markdown(
        "| | タブ①〜③ | **タブ④（このタブ）** |\n"
        "|---|---|---|\n"
        f"| 見ているデータ | 抜き取り **1 回分**<br>良品{data.ok_spec.n}・欠陥{data.ng_spec.n} = "
        f"{sample_total} 枚 | **母集団**（ラインの実力そのもの）<br>"
        f"良品{pool_ok_size:,}・欠陥{pool_ng:,} = {pool_ok_size + pool_ng:,} 個 |\n"
        f"| 欠陥の割合 | {sample_prevalence:.1%} | {pool_prevalence:.1%} |\n"
        "| 閾値 | その 1 回のデータから決める | 母集団から決める ＝ **正解** |\n"
        "| サンプルを引き直すと | 数字が変わる | **変わらない**（実力は不変） |\n"
    )
    st.caption(
        "サイドバーの**中心位置・ばらつき・分布形状・外れ値**は、どちらにも同じように効く"
        "（＝ラインの実力の設定）。サイドバーの**サンプル数**が効くのはタブ①〜③だけで、"
        "このタブの枚数は下で別に指定する。"
    )


def _difference_note(data: AppData) -> None:
    sample_ratio = data.ok_spec.n / data.ng_spec.n
    with st.expander("なぜタブ③の「採用中の閾値」と数字が一致しないのか"):
        st.markdown(
            "上の表のとおり**見ているデータと欠陥割合が違う**ためで、"
            "どちらかが誤りではない。\n\n"
            "**F1 最大の閾値は、評価データの欠陥割合で動く。** F1 は Precision を"
            "含むため、欠陥割合が下がると同じ閾値でも Precision が下がり、"
            "最適点は高い閾値側へずれる。上の「OK:NG 比」を "
            f"**{sample_ratio:.0f}:1**（＝サイドバーと同じ比）に近づけると、"
            "タブ③の値へ寄っていく。\n\n"
            "これは表示上の都合ではなく、量産評価で効く事実である。"
            "評価用に NG を集めたデータセットの欠陥割合（数十%）と、"
            "実ラインの不良率（多くは数%以下）は桁が違う。"
            "**評価セットで F1 最大にした閾値は、実ラインでの最適閾値ではない。**"
        )


def render(data: AppData) -> None:
    st.info(
        "**このタブは、ここまでの 3 つとは見ているデータが違う。** "
        "タブ①〜③ではサイドバーで作った 1 回分の抜き取りを全データとして扱ってきた。"
        "ここではその手前に**ラインの真の実力（母集団）**を置き、"
        "そこから決まった枚数だけ抜き取って、"
        "**そのサンプルだけで F1 最大の閾値を決めて評価する** — "
        "という現場の手順を何度も繰り返す。"
    )

    st.markdown("### 1. まず、母集団（ラインの真の実力）を見る")
    ratio, pool_ok_size = _pool_controls(data)

    ok_pool, ng_pool, ref = _build_pool(
        astuple(data.ok_spec), astuple(data.ng_spec),
        ratio, pool_ok_size, st.session_state.pool_seed,
    )

    fig_col, metric_col = st.columns([3, 1])
    with fig_col:
        normalize = st.toggle(
            "縦軸を各群 100% に揃える", value=False, key="pool_normalize",
            help="欠陥は良品より少ないので、そのままだと山が低く見える。形を比べたいとき用。",
        )
        st.plotly_chart(pool_figure(ok_pool, ng_pool, ref["threshold"], normalize))
    with metric_col:
        st.markdown("**この検出器の「答え」**")
        st.metric("正解の閾値", f"{ref['threshold']:.3f}",
                  help="母集団全体で F1 が最大になる閾値。各試行はこれを当てにいく。")
        st.metric("真の Recall", pct(ref["recall"]))
        st.metric("真の Specificity", pct(ref["specificity"]))
        st.metric("真の F1", f"{ref['f1']:.3f}")
    st.caption(
        "**現場ではこの図は手に入らない。** 母集団を全部検査できるなら AI は要らない。"
        "実際に見えるのは、ここから抜き取った数十枚だけである。"
        "以下では「答えを知っている状態」で、少数の抜き取りが答えをどれだけ"
        "外すかを確かめる。"
    )

    with st.expander(
        f"▸ この「正解の閾値 {ref['threshold']:.3f}」はどう決まったのか"
        "（ROC 曲線 → 混同行列 → F1 曲線）"
    ):
        _derivation(ok_pool, ng_pool, ref["threshold"])

    st.divider()
    st.markdown("### 2. タブ①〜③とどこが違うのか")
    _comparison_table(data, ratio, pool_ok_size)
    _difference_note(data)

    st.divider()
    st.markdown("### 3. その母集団から、何度も抜き取ってみる")

    limit = max_eval_ok(pool_ok_size)
    raw_counts, n_repeat = _eval_controls(limit)

    ok_counts, rejected = _parse_ok_counts(raw_counts, limit)
    for message in rejected:
        st.warning(message)

    # 直接入力で入った枚数を正式な選択肢へ昇格させる。こうしておくと次の
    # 再描画で「OK 150 枚」→ 150 と読み戻せる（表示文字列のまま残らない）。
    unknown = set(ok_counts) - set(PRESET_OK_COUNTS) - set(st.session_state.extra_ok_counts)
    if unknown:
        st.session_state.extra_ok_counts = sorted(
            set(st.session_state.extra_ok_counts) | unknown
        )

    conditions = conditions_for(ok_counts, ratio)
    if len(conditions) < 2:
        st.info("比べる評価サンプル数を 2 つ以上選んでください。")
        return

    st.caption(
        "この設定で比べる条件: "
        + " / ".join(f"**OK{n_ok}・NG{n_ng}**" for n_ok, n_ng in conditions)
    )

    experiments = _run_experiments(ok_pool, ng_pool, conditions, n_repeat)
    if experiments.empty:
        st.warning("この分布では閾値を決められる試行がありませんでした。")
        return
    summary = summarize(experiments)

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
