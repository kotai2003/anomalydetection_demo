"""配色・タイポグラフィ・表示ヘルパ（全タブ共通）。

dataviz 準拠。OK/NG は青/橙＝色覚多様性に安全なペア。
"""

from __future__ import annotations

import math

OK_COLOR = "#2a78d6"
NG_COLOR = "#eb6834"
# 曲線用の第3色（aqua）。Recall=青 / Precision=橙 / F1=aqua で色覚に安全な3色。
F1_COLOR = "#1baf7a"
# 採用中の閾値マーカー（ROC / PR 上の赤い点）
MARKER_RED = "#d03b3b"
# 「少数サンプル vs 多数サンプル」の対比。同じものを枚数だけ変えて見ている
# ので、別色ではなく同系色の濃淡にする（青/橙にすると OK/NG と誤読される）。
SMALL_N_COLOR = "#a8c8ea"
LARGE_N_COLOR = "#1f5c9e"
INK = "#0b0b0b"
INK_SECONDARY = "#52514e"
INK_MUTED = "#898781"
GRID = "#e1e0d9"
SURFACE = "#fcfcfb"

FONT = '"Meiryo", "Hiragino Sans", "Noto Sans JP", system-ui, sans-serif'


def pct(value: float) -> str:
    return "—" if math.isnan(value) else f"{value * 100:.1f}%"
