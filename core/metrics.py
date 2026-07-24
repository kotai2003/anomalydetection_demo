"""混同行列と評価指標。

判定規則は全体を通して一つ: **異常スコア >= 閾値 → NG判定**。

分母が 0 になる指標は 0 ではなく NaN を返す。少数サンプルの教材なので
「計算できない」と「0 だった」を混同させないため。
"""

from __future__ import annotations

import math
from dataclasses import dataclass


@dataclass(frozen=True)
class Confusion:
    """混同行列の4区分。"""

    tn: int  # 良品を正しく通過
    fp: int  # 良品を誤って停止（過検知）
    fn: int  # 欠陥を誤って通過（見逃し）
    tp: int  # 欠陥を正しく停止

    @property
    def total(self) -> int:
        return self.tn + self.fp + self.fn + self.tp


def confusion_at(ok_scores, ng_scores, threshold: float) -> Confusion:
    """OK/NG それぞれの異常スコア列と閾値から混同行列を作る。"""
    fp = sum(1 for s in ok_scores if s >= threshold)
    tp = sum(1 for s in ng_scores if s >= threshold)
    return Confusion(
        tn=len(ok_scores) - fp,
        fp=fp,
        fn=len(ng_scores) - tp,
        tp=tp,
    )


def _ratio(numerator: int, denominator: int) -> float:
    return numerator / denominator if denominator else math.nan


def metrics_at(cm: Confusion) -> dict[str, float]:
    """混同行列から各評価指標を計算する。"""
    recall = _ratio(cm.tp, cm.tp + cm.fn)
    specificity = _ratio(cm.tn, cm.tn + cm.fp)
    precision = _ratio(cm.tp, cm.tp + cm.fp)
    accuracy = _ratio(cm.tp + cm.tn, cm.total)

    if math.isnan(precision) or math.isnan(recall):
        f1 = math.nan
    elif precision + recall == 0:
        f1 = 0.0
    else:
        f1 = 2 * precision * recall / (precision + recall)

    return {
        "recall": recall,
        "specificity": specificity,
        "precision": precision,
        "accuracy": accuracy,
        "f1": f1,
    }
