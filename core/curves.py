"""ROC / PR / F1 曲線と、閾値決定モード。

判定規則は metrics と同じ: **スコア >= 閾値 → NG判定**。
全曲線・全モードは単一の閾値スイープ (`sweep`) を唯一の真実とする。
AUC / AP のスカラーだけは順位ベースの sklearn を使う（閾値に依存しないため）。
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from sklearn.metrics import average_precision_score, roc_auc_score

from .metrics import confusion_at, metrics_at

# キー: 内部識別子, 値: UI 表示名
THRESHOLD_MODES: dict[str, str] = {
    "manual": "手動設定",
    "f1_max": "F1 最大",
    "zero_miss": "見逃しゼロ優先",
    "fpr_cap": "過検知率の上限",
    "cost_min": "現場コスト最小化",
}

# 曲線描画・最適化用の閾値グリッド。スライダーの 0.01 刻みを内包する。
# 末尾の 1.01 は「何も NG と判定しない」点まで曲線を閉じるための番兵。
GRID: np.ndarray = np.round(np.append(np.arange(0.0, 1.0 + 1e-9, 0.005), 1.01), 4)


@dataclass(frozen=True)
class Sweep:
    """各閾値での混同行列と指標を配列でまとめたもの。"""

    thresholds: np.ndarray
    fp: np.ndarray
    fn: np.ndarray
    tp: np.ndarray
    tn: np.ndarray
    recall: np.ndarray
    specificity: np.ndarray
    precision: np.ndarray
    f1: np.ndarray
    fpr: np.ndarray  # = 1 - specificity（過検知率）


def sweep(ok_scores, ng_scores, grid: np.ndarray = GRID) -> Sweep:
    """グリッド上の全閾値で混同行列と指標を計算する。"""
    cols: dict[str, list[float]] = {
        k: []
        for k in ("fp", "fn", "tp", "tn", "recall", "specificity", "precision", "f1", "fpr")
    }
    for t in grid:
        cm = confusion_at(ok_scores, ng_scores, t)
        m = metrics_at(cm)
        cols["fp"].append(cm.fp)
        cols["fn"].append(cm.fn)
        cols["tp"].append(cm.tp)
        cols["tn"].append(cm.tn)
        cols["recall"].append(m["recall"])
        cols["specificity"].append(m["specificity"])
        cols["precision"].append(m["precision"])
        cols["f1"].append(m["f1"])
        cols["fpr"].append(cm.fp / (cm.fp + cm.tn))
    arrays = {k: np.asarray(v, dtype=float) for k, v in cols.items()}
    return Sweep(thresholds=np.asarray(grid, dtype=float), **arrays)


def _labels_and_scores(ok_scores, ng_scores) -> tuple[np.ndarray, np.ndarray]:
    y = np.concatenate([np.zeros(len(ok_scores)), np.ones(len(ng_scores))])
    s = np.concatenate([np.asarray(ok_scores), np.asarray(ng_scores)])
    return y, s


def roc_auc(ok_scores, ng_scores) -> float:
    """ROC 曲線下の面積（順位ベース。閾値に依存しない）。"""
    y, s = _labels_and_scores(ok_scores, ng_scores)
    if y.sum() in (0, len(y)):
        return float("nan")
    return float(roc_auc_score(y, s))


def average_precision(ok_scores, ng_scores) -> float:
    """PR 曲線の平均適合率（PR-AUC の標準的な代替）。"""
    y, s = _labels_and_scores(ok_scores, ng_scores)
    if y.sum() == 0:
        return float("nan")
    return float(average_precision_score(y, s))


def choose_threshold(
    mode: str,
    sw: Sweep,
    *,
    manual: float,
    fpr_cap: float = 0.05,
    cost_fp: float = 1.0,
    cost_fn: float = 10.0,
) -> float:
    """決定モードに従って閾値を1つ選ぶ。

    - manual    : スライダーの値をそのまま使う
    - f1_max    : F1 が最大の閾値
    - zero_miss : 見逃し0（FN==0）を満たす中で最大の閾値（＝過検知が最小）
    - fpr_cap   : 過検知率(FPR) <= 上限 を満たす中で最小の閾値（＝Recall 最大）
    - cost_min  : cost_fp*FP + cost_fn*FN が最小の閾値
    """
    if mode == "manual":
        return manual
    if mode == "f1_max":
        if np.all(np.isnan(sw.f1)):
            return manual
        return float(sw.thresholds[np.nanargmax(sw.f1)])
    if mode == "zero_miss":
        ok = sw.fn == 0
        if not ok.any():  # どの閾値でも見逃しが残る（NGが極端に低スコア）
            return float(sw.thresholds[0])
        return float(sw.thresholds[np.where(ok)[0].max()])
    if mode == "fpr_cap":
        ok = sw.fpr <= fpr_cap + 1e-9
        if not ok.any():
            return float(sw.thresholds[-1])
        return float(sw.thresholds[np.where(ok)[0].min()])
    if mode == "cost_min":
        cost = cost_fp * sw.fp + cost_fn * sw.fn
        return float(sw.thresholds[int(np.argmin(cost))])
    raise ValueError(f"未知の閾値決定モード: {mode}")
