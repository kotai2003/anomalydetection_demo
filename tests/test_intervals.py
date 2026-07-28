"""信頼区間と必要サンプル数。画面に出している数字そのものを固定する。

タブ④は「Recall の下限」と「見逃し率の上限」を並べて表示する。両者が
同じ区間の両端（＝足すと 100%）であることが表の意味を支えているので、
その対称性をテストで縛る。3の法則は**別の推定法**なので、混同しないことも
テストしておく。
"""

import math
import unittest

from core.intervals import (
    required_n_for_margin,
    required_n_zero_events,
    rule_of_three,
    wilson,
)


class TestWilson(unittest.TestCase):
    def test_known_values(self):
        """タブ④とチュートリアルが載せている値。"""
        for n, expected_low in [(2, 0.342), (4, 0.510), (10, 0.722),
                                (20, 0.839), (60, 0.940)]:
            low, high = wilson(n, n)
            self.assertAlmostEqual(low, expected_low, places=3, msg=f"n={n}")
            self.assertAlmostEqual(high, 1.0, places=12, msg=f"n={n}")

    def test_zero_events_upper_bound(self):
        self.assertAlmostEqual(wilson(0, 2)[1], 0.658, places=3)
        self.assertAlmostEqual(wilson(0, 60)[1], 0.060, places=3)

    def test_is_symmetric(self):
        """表の 2 列が補数になることの根拠。1 - wilson(n,n)下限 == wilson(0,n)上限。"""
        for n in (1, 2, 5, 10, 60, 300):
            self.assertAlmostEqual(1.0 - wilson(n, n)[0], wilson(0, n)[1], places=12)

    def test_interval_stays_inside_unit_range(self):
        for successes, n in [(0, 1), (1, 1), (0, 3), (3, 3), (5, 10)]:
            low, high = wilson(successes, n)
            self.assertGreaterEqual(low, 0.0)
            self.assertLessEqual(high, 1.0)

    def test_zero_n_is_nan(self):
        low, high = wilson(0, 0)
        self.assertTrue(math.isnan(low) and math.isnan(high))


class TestRequiredNZeroEvents(unittest.TestCase):
    def test_published_values(self):
        """タブ④の必要枚数表。両側 95% Wilson 上限からの逆算。"""
        self.assertEqual(required_n_zero_events(0.20), 16)
        self.assertEqual(required_n_zero_events(0.10), 35)
        self.assertEqual(required_n_zero_events(0.05), 73)
        self.assertEqual(required_n_zero_events(0.03), 125)
        self.assertEqual(required_n_zero_events(0.01), 381)

    def test_result_actually_achieves_the_target(self):
        """境界の確認: n 枚なら届き、1 枚少ないと届かない。"""
        for rate in (0.20, 0.10, 0.05, 0.01):
            n = required_n_zero_events(rate)
            self.assertLessEqual(wilson(0, n)[1], rate, f"rate={rate}")
            self.assertGreater(wilson(0, n - 1)[1], rate, f"rate={rate}")

    def test_out_of_range_returns_zero(self):
        self.assertEqual(required_n_zero_events(0.0), 0)
        self.assertEqual(required_n_zero_events(1.0), 0)


class TestRuleOfThree(unittest.TestCase):
    def test_is_three_over_n(self):
        self.assertAlmostEqual(rule_of_three(60), 0.05)
        self.assertAlmostEqual(rule_of_three(300), 0.01)

    def test_differs_from_wilson_and_crosses_over(self):
        """片側の近似と両側 Wilson は別物。n=14 前後で大小が入れ替わる。"""
        self.assertGreater(rule_of_three(8), wilson(0, 8)[1])    # 37.5% > 32.4%
        self.assertLess(rule_of_three(60), wilson(0, 60)[1])     # 5.0% < 6.0%

    def test_zero_n_is_nan(self):
        self.assertTrue(math.isnan(rule_of_three(0)))


class TestRequiredNForMargin(unittest.TestCase):
    def test_worst_case_is_at_half(self):
        """事前情報が無いときに p=0.5 が最大枚数になること。"""
        worst = required_n_for_margin(0.5, 0.025)
        for p in (0.01, 0.1, 0.3, 0.7, 0.99):
            self.assertLessEqual(required_n_for_margin(p, 0.025), worst)
        self.assertEqual(worst, 1537)

    def test_tighter_margin_needs_more(self):
        self.assertGreater(required_n_for_margin(0.05, 0.01),
                           required_n_for_margin(0.05, 0.05))

    def test_non_positive_margin_returns_zero(self):
        self.assertEqual(required_n_for_margin(0.05, 0.0), 0)


if __name__ == "__main__":
    unittest.main()
