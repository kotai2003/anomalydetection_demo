"""反復実験まわり。再現性と、2 つある F1最大実装の一致を固定する。

`core.curves.choose_threshold("f1_max")` と `core.sampling.f1_max_threshold` は
別実装（後者は反復実験用に二分探索で高速化してある）。片方だけ直すと、
タブ③の「F1 最大」とタブ④の「正解の閾値」が静かに食い違う。
"""

import unittest

import numpy as np

from core.curves import choose_threshold, sweep
from core.distributions import DistributionSpec, generate_scores
from core.sampling import (
    conditions_for,
    draw_trial,
    f1_max_threshold,
    max_eval_ok,
    pool_ng_size,
    pool_reference,
    repeat_experiment,
    summarize,
)


def _pools(seed=0, n_ok=2000, n_ng=400):
    ok_seed, ng_seed = np.random.SeedSequence(seed).spawn(2)
    ok = generate_scores(DistributionSpec(n_ok, 0.31, 0.09), np.random.default_rng(ok_seed))
    ng = generate_scores(DistributionSpec(n_ng, 0.57, 0.10), np.random.default_rng(ng_seed))
    return ok, ng


class TestF1MaxAgreement(unittest.TestCase):
    def test_two_implementations_agree(self):
        rng = np.random.default_rng(7)
        for _ in range(200):
            n_ok, n_ng = int(rng.integers(2, 60)), int(rng.integers(1, 30))
            ok = np.clip(rng.normal(0.31, 0.09, n_ok), 0, 1)
            ng = np.clip(rng.normal(0.57, 0.10, n_ng), 0, 1)
            a = choose_threshold("f1_max", sweep(ok, ng), manual=0.44)
            b = f1_max_threshold(ok, ng)
            self.assertEqual(a, b)

    def test_all_nan_returns_nan(self):
        self.assertTrue(np.isnan(f1_max_threshold([0.1, 0.2], [])))


class TestRepeatExperiment(unittest.TestCase):
    def setUp(self):
        self.ok_pool, self.ng_pool = _pools()

    def test_is_reproducible(self):
        a = repeat_experiment(self.ok_pool, self.ng_pool, 10, 2, 20, seed=0)
        b = repeat_experiment(self.ok_pool, self.ng_pool, 10, 2, 20, seed=0)
        self.assertTrue(a.equals(b))

    def test_different_seed_gives_different_draws(self):
        a = repeat_experiment(self.ok_pool, self.ng_pool, 10, 2, 20, seed=0)
        b = repeat_experiment(self.ok_pool, self.ng_pool, 10, 2, 20, seed=1)
        self.assertFalse(a["threshold"].equals(b["threshold"]))

    def test_conditions_are_independent_of_each_other(self):
        """条件ごとに独立な乱数列。他の条件を足しても結果が変わらないこと。"""
        a = repeat_experiment(self.ok_pool, self.ng_pool, 10, 2, 10, seed=0)
        b = repeat_experiment(self.ok_pool, self.ng_pool, 10, 2, 10, seed=0)
        self.assertTrue(a.equals(b))

    def test_counts_are_consistent(self):
        df = repeat_experiment(self.ok_pool, self.ng_pool, 10, 2, 30, seed=0)
        self.assertTrue(((df["tn"] + df["fp"]) == df["n_ok"]).all())
        self.assertTrue(((df["tp"] + df["fn"]) == df["n_ng"]).all())
        self.assertTrue(((df["recall"] >= 0) & (df["recall"] <= 1)).all())

    def test_draw_trial_reproduces_the_same_draw(self):
        """箱ひげ図の 1 点を開いたとき、その試行と同じ閾値になること。"""
        df = repeat_experiment(self.ok_pool, self.ng_pool, 10, 2, 5, seed=0)
        for trial in df["trial"]:
            ok, ng = draw_trial(self.ok_pool, self.ng_pool, 10, 2, int(trial), seed=0)
            expected = df.loc[df["trial"] == trial, "threshold"].iloc[0]
            self.assertAlmostEqual(f1_max_threshold(ok, ng), expected, places=12)

    def test_sampling_is_without_replacement(self):
        ok, ng = draw_trial(self.ok_pool, self.ng_pool, 50, 10, 0, seed=0)
        self.assertEqual(len(np.unique(ok)), 50)
        self.assertEqual(len(np.unique(ng)), 10)

    def test_drawing_more_than_the_pool_raises(self):
        with self.assertRaises(ValueError):
            repeat_experiment(self.ok_pool[:5], self.ng_pool, 10, 2, 1, seed=0)


class TestConditions(unittest.TestCase):
    def test_conditions_are_sorted_and_deduplicated(self):
        self.assertEqual(conditions_for([50, 10, 10], 5), ((10, 2), (50, 10)))

    def test_ng_count_is_at_least_one(self):
        self.assertEqual(conditions_for([2], 20), ((2, 1),))

    def test_pool_ng_size_is_at_least_one(self):
        self.assertGreaterEqual(pool_ng_size(20, 5), 1)

    def test_eval_cap_is_half_the_pool(self):
        self.assertEqual(max_eval_ok(2000), 1000)


class TestSummarize(unittest.TestCase):
    def test_recall_100_rate(self):
        ok_pool, ng_pool = _pools()
        df = repeat_experiment(ok_pool, ng_pool, 10, 2, 50, seed=0)
        s = summarize(df)
        self.assertAlmostEqual(float(s["recall_100_rate"].iloc[0]),
                               float((df["recall"] >= 1.0).mean()))

    def test_empty_input(self):
        import pandas as pd
        self.assertTrue(summarize(pd.DataFrame()).empty)


class TestPoolReference(unittest.TestCase):
    def test_threshold_and_metrics_are_consistent(self):
        ok_pool, ng_pool = _pools()
        ref = pool_reference(ok_pool, ng_pool)
        self.assertEqual(ref["threshold"], f1_max_threshold(ok_pool, ng_pool))
        self.assertEqual(ref["tp"] + ref["fn"], len(ng_pool))
        self.assertEqual(ref["tn"] + ref["fp"], len(ok_pool))


if __name__ == "__main__":
    unittest.main()
