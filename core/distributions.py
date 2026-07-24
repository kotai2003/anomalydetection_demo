"""異常スコア分布の疑似生成。

異常スコアは 0.0〜1.0 に収める。分布形状を変えても「中心位置」「ばらつき」の
意味が変わらないよう、形状ごとの変量を平均0・標準偏差1に正規化してから
center / spread を適用する。
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

# キー: 内部識別子, 値: UI 表示名
SHAPES: dict[str, str] = {
    "normal": "正規分布",
    "right_skew": "右裾の長い分布",
    "left_skew": "左裾の長い分布",
    "bimodal": "二山分布",
    "uniform": "一様分布",
}

# 外れ値のばらつき（外れ値位置まわりの微小ジッタ）
_OUTLIER_JITTER = 0.02


@dataclass(frozen=True)
class DistributionSpec:
    """片側（OK または NG）のスコア分布パラメータ。"""

    n: int
    center: float
    spread: float
    shape: str = "normal"
    n_outlier: int = 0
    outlier_center: float = 0.9


def _standard_variates(shape: str, size: int, rng: np.random.Generator) -> np.ndarray:
    """平均0・標準偏差1へ正規化した、形状違いの変量を返す。"""
    if shape == "normal":
        return rng.standard_normal(size)
    if shape == "right_skew":
        # gamma(k=2) は 平均2・分散2
        return (rng.gamma(2.0, 1.0, size) - 2.0) / np.sqrt(2.0)
    if shape == "left_skew":
        return -(rng.gamma(2.0, 1.0, size) - 2.0) / np.sqrt(2.0)
    if shape == "bimodal":
        # ±1 を中心とする2山。分散 = 1 + 0.35^2
        peaks = rng.choice([-1.0, 1.0], size=size)
        return (peaks + 0.35 * rng.standard_normal(size)) / np.sqrt(1.0 + 0.35**2)
    if shape == "uniform":
        return rng.uniform(-np.sqrt(3.0), np.sqrt(3.0), size)
    raise ValueError(f"未知の分布形状: {shape}")


def generate_scores(spec: DistributionSpec, rng: np.random.Generator) -> np.ndarray:
    """spec に従って異常スコアを n 件生成する（0.0〜1.0 にクリップ）。

    外れ値はサンプルを追加するのではなく、n 件のうち n_outlier 件を
    置き換える（良品10枚のうち1枚が外れ値、という現場の見え方に合わせる）。
    """
    scores = spec.center + spec.spread * _standard_variates(spec.shape, spec.n, rng)

    n_outlier = min(spec.n_outlier, spec.n)
    if n_outlier > 0:
        index = rng.choice(spec.n, size=n_outlier, replace=False)
        scores[index] = spec.outlier_center + _OUTLIER_JITTER * rng.standard_normal(
            n_outlier
        )

    return np.clip(scores, 0.0, 1.0)
