"""サイドバー（操作パネル）と、全タブが共有するデータの組み立て。"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import streamlit as st

from core.distributions import SHAPES, DistributionSpec, generate_scores
from core.metrics import Confusion, confusion_at, metrics_at


@dataclass(frozen=True)
class AppData:
    """サイドバーで決まる、全タブ共通の入力一式。"""

    threshold: float
    ok_spec: DistributionSpec
    ng_spec: DistributionSpec
    ok_scores: np.ndarray
    ng_scores: np.ndarray
    cm: Confusion
    metrics: dict[str, float]
    seed: int


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


def render() -> AppData:
    """サイドバーを描画し、全タブが使うデータを返す。"""
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
        st.caption(
            "シードは“くじの番号”。同じ番号なら毎回同じサンプルが出る（再現できる）。"
            "固定してあるので、閾値を動かしても点は動かず境目だけが動く。"
        )
        if st.button(
            "サンプルを引き直す",
            width="stretch",
            help=(
                "分布の設定（数・中心・ばらつき・形）はそのままに、別のサンプルを"
                "引き直す＝「別の日にもう一度、同じラインから抜き取り検査した」イメージ。"
                "ラインの実力は同じでも、たまたま選ばれた製品が違うと結果の数字は少しブレる。"
            ),
        ):
            st.session_state.seed += 1
        st.caption(
            "同じ設定でも、選ぶサンプル次第で成績は少し変わる — を体験するボタンです。"
        )

    seed = st.session_state.seed
    ok_scores, ng_scores = make_scores(ok_spec, ng_spec, seed)
    cm = confusion_at(ok_scores, ng_scores, threshold)

    return AppData(
        threshold=threshold,
        ok_spec=ok_spec,
        ng_spec=ng_spec,
        ok_scores=ok_scores,
        ng_scores=ng_scores,
        cm=cm,
        metrics=metrics_at(cm),
        seed=seed,
    )
