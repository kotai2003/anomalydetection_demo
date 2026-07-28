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

    p = 0 のとき Wilson 上限は z² / (n + z²) に簡約されるので、閉じた式で解ける。

        z² / (n + z²) <= max_rate   ⟺   n >= z² (1 - max_rate) / max_rate

    「3の法則（n ≈ 3 / 率）」はこの式の暗算版にあたる（z² ≈ 3.84）。
    """
    if not 0.0 < max_rate < 1.0:
        return 0
    z = norm.isf((1.0 - confidence) / 2.0)
    return math.ceil(z * z * (1.0 - max_rate) / max_rate)


def required_n_for_margin(
    rate: float, margin: float, confidence: float = 0.95
) -> int:
    """比率 rate を ±margin の精度で見積もるのに必要な件数。

    標準的な正規近似 n = z² p(1-p) / margin²。品質部門が見慣れた式でもある。
    この範囲では Wilson 区間の半幅ともほぼ一致する。
    """
    if margin <= 0.0:
        return 0
    z = norm.isf((1.0 - confidence) / 2.0)
    p = min(max(rate, 0.0), 1.0)
    return math.ceil(z * z * p * (1.0 - p) / (margin * margin))


def rule_of_three(n: int) -> float:
    """「3の法則」— n 件で見逃し 0 件だったときの、見逃し率のおおよその95%上限。

        上限 ≈ 3 / n

    厳密な万能式ではなく、ゼロイベント時の 95% 上限を暗算で示すための近似
    （n が 30 以上あたりで wilson(0, n) の上限とよく一致する）。

    n が小さいと 1.0 を超えた値、つまり「上限が 100% を下回らない＝実質的に
    何も評価できていない」が返る。呼び出し側でそのまま率として表示しないこと。
    n=0 のときは nan。
    """
    return 3.0 / n if n > 0 else math.nan
