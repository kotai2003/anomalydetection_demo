"""閾値決定モードと、同点になったときの規則を固定する。

同点規則は「暗黙の argmax 任せ」にすると曲線が平らなときに結果が動く。
少数サンプルでは実際に平らになるので、ここをテストで縛っておく。
"""

import unittest

import numpy as np

from core.curves import GRID, choose_threshold, f1_argmax, roc_auc, sweep


class TestF1Argmax(unittest.TestCase):
    def test_unique_maximum(self):
        self.assertEqual(f1_argmax(np.array([0.1, 0.9, 0.3])), 1)

    def test_tie_takes_the_middle_not_the_left_edge(self):
        """同点は平坦区間の中央。左端（＝最も低い閾値）を採ると偏りが入る。"""
        self.assertEqual(f1_argmax(np.array([0.2, 0.9, 0.9, 0.9, 0.1])), 2)

    def test_tie_with_even_count_takes_upper_middle(self):
        self.assertEqual(f1_argmax(np.array([0.9, 0.9])), 1)

    def test_nan_is_ignored(self):
        self.assertEqual(f1_argmax(np.array([np.nan, 0.5, np.nan])), 1)


class TestChooseThreshold(unittest.TestCase):
    def setUp(self):
        rng = np.random.default_rng(0)
        self.ok = np.clip(rng.normal(0.31, 0.09, 120), 0, 1)
        self.ng = np.clip(rng.normal(0.57, 0.10, 100), 0, 1)
        self.sw = sweep(self.ok, self.ng)

    def test_manual_returns_slider_value(self):
        self.assertEqual(choose_threshold("manual", self.sw, manual=0.44), 0.44)

    def test_f1_max_matches_the_sweep(self):
        t = choose_threshold("f1_max", self.sw, manual=0.44)
        self.assertEqual(t, float(self.sw.thresholds[f1_argmax(self.sw.f1)]))
        self.assertIn(t, set(GRID.tolist()))

    def test_zero_miss_has_no_false_negative(self):
        t = choose_threshold("zero_miss", self.sw, manual=0.44)
        self.assertEqual(int(self.sw.fn[np.argmin(np.abs(self.sw.thresholds - t))]), 0)

    def test_zero_miss_takes_the_highest_such_threshold(self):
        """見逃し0を満たす中で最も高い閾値＝過検知が最小。"""
        feasible = self.sw.thresholds[self.sw.fn == 0]
        self.assertEqual(choose_threshold("zero_miss", self.sw, manual=0.44),
                         float(feasible.max()))

    def test_fpr_cap_is_never_exceeded(self):
        for cap in (0.01, 0.05, 0.2):
            t = choose_threshold("fpr_cap", self.sw, manual=0.44, fpr_cap=cap)
            i = int(np.argmin(np.abs(self.sw.thresholds - t)))
            self.assertLessEqual(self.sw.fpr[i], cap + 1e-9, f"cap={cap}")

    def test_cost_min_minimises_the_stated_cost(self):
        t = choose_threshold("cost_min", self.sw, manual=0.44, cost_fp=1.0, cost_fn=10.0)
        cost = 1.0 * self.sw.fp + 10.0 * self.sw.fn
        i = int(np.argmin(np.abs(self.sw.thresholds - t)))
        self.assertEqual(cost[i], cost.min())

    def test_cost_min_tie_takes_the_lowest_threshold(self):
        """同点なら見逃しが少ない側（低い閾値）。argmin の既定と同じだが仕様として固定する。"""
        flat = sweep(np.array([0.9]), np.array([0.9]))
        cost = 1.0 * flat.fp + 1.0 * flat.fn
        t = choose_threshold("cost_min", flat, manual=0.44, cost_fp=1.0, cost_fn=1.0)
        self.assertEqual(t, float(flat.thresholds[int(np.flatnonzero(cost == cost.min())[0])]))

    def test_unknown_mode_raises(self):
        with self.assertRaises(ValueError):
            choose_threshold("no_such_mode", self.sw, manual=0.44)

    def test_f1_max_falls_back_to_manual_when_all_nan(self):
        empty = sweep(np.array([0.5]), np.array([]))  # NG が無いと F1 は全て NaN
        self.assertEqual(choose_threshold("f1_max", empty, manual=0.44), 0.44)


class TestAuc(unittest.TestCase):
    def test_perfect_separation(self):
        self.assertEqual(roc_auc([0.0, 0.1], [0.9, 1.0]), 1.0)

    def test_single_class_is_nan(self):
        self.assertTrue(np.isnan(roc_auc([0.1, 0.2], [])))


if __name__ == "__main__":
    unittest.main()
