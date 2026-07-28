"""少数サンプルでの抜き取り評価を、繰り返し模擬する。

現場の手順をそのまま繰り返す:

    ラインの真の実力（母集団プール）から OK n枚・NG n枚を抜き取る
      → その少数サンプルだけで F1 最大の閾値を決める
      → その少数サンプルで Recall などを報告する

これを何度も繰り返すと、**同じラインの同じ検出器でも**、たまたま選ばれた
サンプル次第で閾値も評価値も変わることが分布として見える。

判定規則は core.metrics と同じ **スコア >= 閾値 → NG判定**、閾値グリッドは
core.curves の GRID を共有する（タブ③の曲線と同じ閾値の上で議論するため）。
"""

from __future__ import annotations

from dataclasses import replace

import numpy as np
import pandas as pd

from .curves import GRID
from .distributions import DistributionSpec, generate_scores

# 評価データの OK:NG 比。**条件によってこの比を変えてはいけない。**
# F1 が最大になる閾値は不良割合で動くため、比率まで一緒に動かすと、
# 閾値の違いが「サンプル数のせい」なのか「OK:NG 比のせい」なのか
# 区別できなくなる（見たいのは前者だけ）。
EVAL_OK_NG_RATIO = 5

# 母集団プールの既定サイズ。「真の実力」を十分な精度で表せて、かつ
# 反復実験が対話的な速さで終わる大きさ。抜き取りと同じ OK:NG 比で持つ
# ことで、母集団の F1最大閾値がそのまま各試行の「正解」になる。
POOL_OK_SIZE = 2000
POOL_NG_SIZE = POOL_OK_SIZE // EVAL_OK_NG_RATIO

# 反復実験で使う評価サンプル数。OK 側は構想書の 10/20/50/100/最大に合わせ、
# NG 側は上の比率から決まる。先頭は初期評価で実際に採られがちな OK10・NG2。
PRESET_CONDITIONS: tuple[tuple[int, int], ...] = tuple(
    (n_ok, n_ok // EVAL_OK_NG_RATIO) for n_ok in (10, 20, 50, 100, 300)
)


def condition_label(n_ok: int, n_ng: int) -> str:
    return f"OK{n_ok}・NG{n_ng}"


def make_pool(spec: DistributionSpec, size: int, rng: np.random.Generator) -> np.ndarray:
    """spec の分布形状のまま、母集団サイズ分のスコアを生成する。

    外れ値は「枚数」ではなく「割合」を保つ。spec が 120枚中2枚の外れ値なら、
    母集団でも同じ 1.7% になるようにスケールする（枚数のまま 2 件にすると
    母集団では無視できる量になり、抜き取りに出てこなくなる）。
    """
    n_outlier = round(spec.n_outlier * size / spec.n) if spec.n else 0
    return generate_scores(replace(spec, n=size, n_outlier=n_outlier), rng)


def _counts_on_grid(
    ok_scores: np.ndarray, ng_scores: np.ndarray, grid: np.ndarray
) -> tuple[np.ndarray, np.ndarray]:
    """全閾値での (FP, TP) をまとめて数える。

    `sum(score >= t)` を閾値ごとに数え直すと反復実験では遅すぎるので、
    ソート済み配列への二分探索で一度に求める。数え方は confusion_at と同一。
    """
    ok_sorted = np.sort(np.asarray(ok_scores, dtype=float))
    ng_sorted = np.sort(np.asarray(ng_scores, dtype=float))
    fp = len(ok_sorted) - np.searchsorted(ok_sorted, grid, side="left")
    tp = len(ng_sorted) - np.searchsorted(ng_sorted, grid, side="left")
    return fp, tp


def _f1_on_grid(fp: np.ndarray, tp: np.ndarray, n_ng: int) -> np.ndarray:
    """全閾値での F1。NG と判定したものが 0 件の閾値は NaN（metrics_at と同じ）。"""
    fn = n_ng - tp
    denominator = 2 * tp + fp + fn
    with np.errstate(invalid="ignore", divide="ignore"):
        f1 = np.where(denominator > 0, 2 * tp / denominator, np.nan)
    # tp + fp == 0 は Precision が計算不能 → F1 も計算不能
    return np.where(tp + fp == 0, np.nan, f1)


def f1_max_threshold(ok_scores, ng_scores, grid: np.ndarray = GRID) -> float:
    """F1 が最大になる閾値。core.curves.choose_threshold("f1_max") と同じ値を返す。"""
    fp, tp = _counts_on_grid(ok_scores, ng_scores, grid)
    f1 = _f1_on_grid(fp, tp, len(ng_scores))
    if np.all(np.isnan(f1)):
        return float("nan")
    return float(grid[np.nanargmax(f1)])


def _metrics_row(fp: int, tp: int, n_ok: int, n_ng: int) -> dict[str, float]:
    """1 試行分の混同行列と指標。分母 0 は NaN（core.metrics と同じ規則）。"""
    tn, fn = n_ok - fp, n_ng - tp
    recall = tp / n_ng if n_ng else np.nan
    specificity = tn / n_ok if n_ok else np.nan
    precision = tp / (tp + fp) if (tp + fp) else np.nan
    accuracy = (tp + tn) / (n_ok + n_ng) if (n_ok + n_ng) else np.nan
    if np.isnan(precision) or np.isnan(recall):
        f1 = np.nan
    elif precision + recall == 0:
        f1 = 0.0
    else:
        f1 = 2 * precision * recall / (precision + recall)
    return {
        "tn": tn, "fp": fp, "fn": fn, "tp": tp,
        "recall": recall, "specificity": specificity,
        "precision": precision, "accuracy": accuracy, "f1": f1,
    }


def repeat_experiment(
    ok_pool: np.ndarray,
    ng_pool: np.ndarray,
    n_ok: int,
    n_ng: int,
    n_repeat: int,
    *,
    seed: int = 0,
    grid: np.ndarray = GRID,
) -> pd.DataFrame:
    """母集団から n_ok / n_ng 枚を非復元抽出 → F1最大の閾値決定 → 評価、を n_repeat 回。

    1 行 = 1 回の抜き取り試行。閾値も評価値も**その少数サンプルだけ**から出す
    （現場で少数サンプルから閾値を決めて報告する状況の再現）。
    """
    # 条件ごとに独立かつ再現可能な乱数列にする（同じ条件なら常に同じ結果）。
    rng = np.random.default_rng([seed, n_ok, n_ng])

    rows = []
    for trial in range(n_repeat):
        ok = rng.choice(ok_pool, size=n_ok, replace=False)
        ng = rng.choice(ng_pool, size=n_ng, replace=False)

        fp_grid, tp_grid = _counts_on_grid(ok, ng, grid)
        f1_grid = _f1_on_grid(fp_grid, tp_grid, n_ng)
        if np.all(np.isnan(f1_grid)):
            continue
        best = int(np.nanargmax(f1_grid))

        row = {"trial": trial, "n_ok": n_ok, "n_ng": n_ng,
               "condition": condition_label(n_ok, n_ng),
               "threshold": float(grid[best])}
        row.update(_metrics_row(int(fp_grid[best]), int(tp_grid[best]), n_ok, n_ng))
        rows.append(row)

    return pd.DataFrame(rows)


def run_conditions(
    ok_pool: np.ndarray,
    ng_pool: np.ndarray,
    conditions,
    n_repeat: int,
    *,
    seed: int = 0,
) -> pd.DataFrame:
    """複数の (n_ok, n_ng) 条件をまとめて回し、1本の DataFrame にする。"""
    frames = [
        repeat_experiment(ok_pool, ng_pool, n_ok, n_ng, n_repeat, seed=seed)
        for n_ok, n_ng in conditions
    ]
    if not frames:
        return pd.DataFrame()
    return pd.concat(frames, ignore_index=True)


def pool_reference(
    ok_pool: np.ndarray, ng_pool: np.ndarray, grid: np.ndarray = GRID
) -> dict[str, float]:
    """母集団全体で F1 最大にしたときの閾値と指標＝各試行が当てにいっている「正解」。"""
    threshold = f1_max_threshold(ok_pool, ng_pool, grid)
    if np.isnan(threshold):
        return {"threshold": np.nan}
    fp, tp = _counts_on_grid(ok_pool, ng_pool, np.asarray([threshold]))
    row = _metrics_row(int(fp[0]), int(tp[0]), len(ok_pool), len(ng_pool))
    return {"threshold": threshold, **row}


def summarize(experiments: pd.DataFrame) -> pd.DataFrame:
    """条件ごとの要約。ばらつきは 5〜95 パーセンタイル幅で見る。

    `recall_100_rate` = Recall がちょうど 100% になった試行の割合。
    「見逃しゼロ」がどれくらい簡単に出てしまうかを示す中心的な数字。
    """
    if experiments.empty:
        return pd.DataFrame()

    rows = []
    for condition, group in experiments.groupby("condition", sort=False):
        rows.append({
            "condition": condition,
            "n_ok": int(group["n_ok"].iloc[0]),
            "n_ng": int(group["n_ng"].iloc[0]),
            "trials": len(group),
            "threshold_median": group["threshold"].median(),
            "threshold_p05": group["threshold"].quantile(0.05),
            "threshold_p95": group["threshold"].quantile(0.95),
            "recall_median": group["recall"].median(),
            "recall_p05": group["recall"].quantile(0.05),
            "recall_100_rate": float((group["recall"] >= 1.0).mean()),
            "f1_median": group["f1"].median(),
            "f1_p05": group["f1"].quantile(0.05),
            "f1_p95": group["f1"].quantile(0.95),
        })
    return pd.DataFrame(rows)
