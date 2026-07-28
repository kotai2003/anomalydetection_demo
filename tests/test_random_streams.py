"""OK と NG の乱数系列が独立していること。

1 つの Generator を OK → NG の順に使い回すと、OK 側の**サンプル数・分布形状・
外れ値数**を変えただけで乱数の消費量が変わり、NG 側まで引き直されてしまう。
「OK だけ変えた比較」が成立しなくなるので、ここは回帰テストで縛る。
中心位置・ばらつきは消費量を変えないため、以前から影響していなかった。
"""

import unittest

import numpy as np

from core.distributions import DistributionSpec
from core.sampling import make_pool, pool_ng_size
from ui.sidebar import make_scores

OK = DistributionSpec(120, 0.31, 0.09, "normal", 0, 0.75)
NG = DistributionSpec(100, 0.57, 0.10, "normal", 0, 0.25)

# 乱数の消費量が変わる変更（＝以前 NG 側へ漏れていた操作）
OK_VARIANTS = {
    "サンプル数": DistributionSpec(121, 0.31, 0.09, "normal", 0, 0.75),
    "分布形状": DistributionSpec(120, 0.31, 0.09, "bimodal", 0, 0.75),
    "外れ値の数": DistributionSpec(120, 0.31, 0.09, "normal", 2, 0.75),
    "中心位置": DistributionSpec(120, 0.32, 0.09, "normal", 0, 0.75),
}
NG_VARIANTS = {
    "サンプル数": DistributionSpec(101, 0.57, 0.10, "normal", 0, 0.25),
    "分布形状": DistributionSpec(100, 0.57, 0.10, "uniform", 0, 0.25),
    "外れ値の数": DistributionSpec(100, 0.57, 0.10, "normal", 3, 0.25),
}


class TestSidebarScores(unittest.TestCase):
    def test_reproducible_for_the_same_seed(self):
        a_ok, a_ng = make_scores(OK, NG, 0)
        b_ok, b_ng = make_scores(OK, NG, 0)
        np.testing.assert_array_equal(a_ok, b_ok)
        np.testing.assert_array_equal(a_ng, b_ng)

    def test_new_seed_redraws_both(self):
        a_ok, a_ng = make_scores(OK, NG, 0)
        b_ok, b_ng = make_scores(OK, NG, 1)
        self.assertFalse(np.array_equal(a_ok, b_ok))
        self.assertFalse(np.array_equal(a_ng, b_ng))

    def test_changing_ok_never_changes_ng(self):
        _, base_ng = make_scores(OK, NG, 0)
        for name, variant in OK_VARIANTS.items():
            with self.subTest(name):
                _, ng = make_scores(variant, NG, 0)
                np.testing.assert_array_equal(ng, base_ng)

    def test_changing_ng_never_changes_ok(self):
        base_ok, _ = make_scores(OK, NG, 0)
        for name, variant in NG_VARIANTS.items():
            with self.subTest(name):
                ok, _ = make_scores(OK, variant, 0)
                np.testing.assert_array_equal(ok, base_ok)

    def test_ok_setting_still_changes_ok(self):
        """漏れを止めた副作用で、当の OK 側まで変わらなくなっていないこと。"""
        base_ok, _ = make_scores(OK, NG, 0)
        ok, _ = make_scores(OK_VARIANTS["中心位置"], NG, 0)
        self.assertFalse(np.array_equal(ok, base_ok))


class TestPoolStreams(unittest.TestCase):
    """タブ④の母集団も同じ設計であること（_build_pool と同じ手順を再現）。"""

    @staticmethod
    def _pools(ok_spec, ng_spec, seed=0, pool_ok=2000, ratio=5):
        ok_seed, ng_seed = np.random.SeedSequence(seed).spawn(2)
        return (
            make_pool(ok_spec, pool_ok, np.random.default_rng(ok_seed)),
            make_pool(ng_spec, pool_ng_size(ratio, pool_ok), np.random.default_rng(ng_seed)),
        )

    def test_changing_ok_never_changes_the_ng_pool(self):
        _, base = self._pools(OK, NG)
        for name, variant in OK_VARIANTS.items():
            with self.subTest(name):
                _, ng_pool = self._pools(variant, NG)
                np.testing.assert_array_equal(ng_pool, base)

    def test_pool_seed_redraws(self):
        _, a = self._pools(OK, NG, seed=0)
        _, b = self._pools(OK, NG, seed=1)
        self.assertFalse(np.array_equal(a, b))


if __name__ == "__main__":
    unittest.main()
