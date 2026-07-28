"""混同行列と評価指標の仕様を固定する。

判定規則（スコア >= 閾値 → NG）と、分母 0 を 0 ではなく NaN にする規則は
アプリ全体の土台なので、ここで壊れたら他のテストの意味も無くなる。
"""

import math
import unittest

from core.metrics import confusion_at, metrics_at


class TestConfusionAt(unittest.TestCase):
    def test_score_equal_to_threshold_is_ng(self):
        """境界は「以上」で NG 側。ここを「より大きい」にすると全指標がずれる。"""
        cm = confusion_at([0.5], [0.5], 0.5)
        self.assertEqual((cm.fp, cm.tn), (1, 0))  # OK の 0.5 は NG と判定される
        self.assertEqual((cm.tp, cm.fn), (1, 0))  # NG の 0.5 も NG と判定される

    def test_known_counts(self):
        ok = [0.1, 0.2, 0.9]          # 0.9 だけが過検知
        ng = [0.05, 0.8, 0.95]        # 0.05 だけが見逃し
        cm = confusion_at(ok, ng, 0.5)
        self.assertEqual((cm.tn, cm.fp, cm.fn, cm.tp), (2, 1, 1, 2))
        self.assertEqual(cm.total, 6)

    def test_empty_inputs(self):
        cm = confusion_at([], [], 0.5)
        self.assertEqual((cm.tn, cm.fp, cm.fn, cm.tp), (0, 0, 0, 0))


class TestMetricsAt(unittest.TestCase):
    def test_known_values(self):
        m = metrics_at(confusion_at([0.1, 0.2, 0.9], [0.05, 0.8, 0.95], 0.5))
        self.assertAlmostEqual(m["recall"], 2 / 3)
        self.assertAlmostEqual(m["specificity"], 2 / 3)
        self.assertAlmostEqual(m["precision"], 2 / 3)
        self.assertAlmostEqual(m["accuracy"], 4 / 6)
        self.assertAlmostEqual(m["f1"], 2 / 3)

    def test_precision_is_nan_when_nothing_predicted_ng(self):
        """「NG と判定した件数が 0」は 0 点ではなく計算不能。"""
        m = metrics_at(confusion_at([0.1], [0.2], 0.9))
        self.assertTrue(math.isnan(m["precision"]))
        self.assertTrue(math.isnan(m["f1"]))
        self.assertEqual(m["recall"], 0.0)  # NG は 1 件あるので Recall は計算できる

    def test_recall_is_nan_when_no_ng_samples(self):
        m = metrics_at(confusion_at([0.1, 0.9], [], 0.5))
        self.assertTrue(math.isnan(m["recall"]))
        self.assertTrue(math.isnan(m["f1"]))

    def test_f1_is_zero_not_nan_when_tp_zero_but_fp_exists(self):
        """Precision も Recall も 0（計算はできる）なら F1 は 0。NaN ではない。"""
        m = metrics_at(confusion_at([0.9], [0.1], 0.5))
        self.assertEqual(m["precision"], 0.0)
        self.assertEqual(m["recall"], 0.0)
        self.assertEqual(m["f1"], 0.0)


if __name__ == "__main__":
    unittest.main()
