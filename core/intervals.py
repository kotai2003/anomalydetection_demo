"""割合の信頼区間。「何枚で測ったか」を評価値に添えるために使う。

Recall 90% という数字は、10枚中9枚なのか 300枚中270枚なのかで意味が違う。
点推定だけを報告すると、この違いが消えてしまう。
"""

from __future__ import annotations

import math

from scipy.stats import norm


def wilson(successes: int, n: int, confidence: float = 0.95) -> tuple[float, float]:
    """二項比率の Wilson 信頼区間。

    少数サンプルや 0%/100% の端でも区間が潰れないため、単純な正規近似
    （Wald 区間）より現場向けの説明に適する。n=0 のときは (nan, nan)。
    """
    if n <= 0:
        return math.nan, math.nan

    z = norm.isf((1.0 - confidence) / 2.0)
    p = successes / n
    denominator = 1.0 + z**2 / n
    center = (p + z**2 / (2 * n)) / denominator
    half_width = z * math.sqrt(p * (1 - p) / n + z**2 / (4 * n**2)) / denominator
    return max(0.0, center - half_width), min(1.0, center + half_width)


def required_n_zero_events(max_rate: float, confidence: float = 0.95) -> int:
    """見逃し 0 件で「見逃し率は max_rate 以下」と言い切るのに必要な評価件数。

    p = 0 のとき **両側** Wilson 上限は z² / (n + z²) に簡約されるので、
    閉じた式で解ける（両側 95% なら z = 1.96、z² ≈ 3.84）。

        z² / (n + z²) <= max_rate   ⟺   n >= z² (1 - max_rate) / max_rate

    このアプリは区間推定を**すべて両側 95% Wilson で通す**ので、逆算もそれに
    揃えてある（5% なら 73 件、1% なら 381 件）。

    `rule_of_three()` の「3の法則」はこの式の暗算版**ではない**。あちらは
    片側 95% 上限の近似で、係数が 3 と 3.84 で違うぶん、同じ 5% でも 60 件と
    出る（片側の厳密二項なら 59 件）。どれを採るかは方針の問題だが、
    **報告書では必ずどの方法かを明記し、混ぜて使わないこと。**
    """
    if not 0.0 < max_rate < 1.0:
        return 0
    z = norm.isf((1.0 - confidence) / 2.0)
    return math.ceil(z * z * (1.0 - max_rate) / max_rate)


def required_n_for_margin(
    rate: float, margin: float, confidence: float = 0.95
) -> int:
    """比率 rate を ±margin の精度で見積もるのに必要な件数（仮定付きの概算）。

    標準的な正規近似 n = z² p(1-p) / margin²。品質部門が見慣れた式でもある。
    この範囲では Wilson 区間の半幅ともほぼ一致する。

    **確定した必要数ではない。** 少なくとも次を織り込んでいない。

    - rate は「想定する過検知率」。現場では真の値が未知なので、事前情報が
      無ければ最大枚数になる rate = 0.5 を置くのが安全側になる（このアプリは
      疑似母集団の既知の値を渡しているが、実務ではパイロットからの仮定値）。
    - 独立な二項試行の仮定。有限母集団補正、ロット・設備・撮像条件による
      クラスタリング、小標本での正規近似誤差、複数条件を同時に保証する
      場合の多重性は、いずれも入っていない。
    """
    if margin <= 0.0:
        return 0
    z = norm.isf((1.0 - confidence) / 2.0)
    p = min(max(rate, 0.0), 1.0)
    return math.ceil(z * z * p * (1.0 - p) / (margin * margin))


def rule_of_three(n: int) -> float:
    """「3の法則」— n 件で見逃し 0 件だったときの、見逃し率の**片側**95%上限の暗算値。

        上限 ≈ 3 / n

    厳密には片側 95% 上限 1 - 0.05^(1/n) の近似（-ln 0.05 ≈ 3.00）であって、
    このアプリが表に出している**両側** 95% Wilson 上限 z² / (n + z²) とは
    別の推定法。n = 14 前後で大小が入れ替わるので、値も食い違う。

        n=8  → 3の法則 37.5% / Wilson 32.4%
        n=60 → 3の法則  5.0% / Wilson  6.0%

    暗算の目安としてのみ使い、**Wilson 由来の数字と同じ列に混ぜないこと**
    （下限を Wilson、上限を 3の法則にすると、足して 100% にならない数字が
    並ぶ）。

    n が小さいと 1.0 を超えた値が返る。呼び出し側でそのまま率として
    表示しないこと。n=0 のときは nan。
    """
    return 3.0 / n if n > 0 else math.nan
