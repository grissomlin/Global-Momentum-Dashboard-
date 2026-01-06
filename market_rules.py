# -*- coding: utf-8 -*-
"""
market_rules.py
---------------
跨市場「漲停 / 強勢日 / 分箱」規則集中管理

設計目標：
1) processor.py 可用（提供 limit_up_pct/threshold/strength_intervals 等概念）
2) event_engine.py 可用（提供 is_limit_up / limit_price / 分箱區間）

重點：
- TW: 上市/上櫃 10%，興櫃無漲跌幅
- CN: 主板/中小 10%，創業板(300/301)與科創板(688) 20%
- JP: 依前收價格區間，使用「值幅金額(日圓)」判斷 Stop High/Low
- US/HK: 無漲跌幅限制 → 以 10% 當 pseudo-limit（你要的「10% 當漲停」）
"""

from __future__ import annotations
from dataclasses import dataclass
from typing import Optional, Tuple, List, Dict
import math


# -------------------------
# JPX 值幅（簡化但可用的常見表）
# （提示：若你之後想做「完全精準」，可以再把 JPX 官方完整表補齊/更新）
# -------------------------
_JPX_LIMIT_TABLE: List[Tuple[float, float, float]] = [
    # (min_price_inclusive, max_price_inclusive, limit_yen)
    (0, 99, 30),
    (100, 199, 50),
    (200, 499, 80),
    (500, 699, 100),
    (700, 999, 150),
    (1000, 1499, 300),
    (1500, 1999, 400),
    (2000, 2999, 500),
    (3000, 4999, 700),
    (5000, 6999, 1000),
    (7000, 9999, 1500),
    (10000, 14999, 3000),
    (15000, 19999, 4000),
    (20000, 29999, 5000),
    (30000, 49999, 7000),
    (50000, 69999, 10000),
    (70000, 99999, 15000),
    (100000, 149999, 30000),
    (150000, 199999, 40000),
    (200000, 299999, 50000),
    (300000, 499999, 70000),
    (500000, 699999, 100000),
    (700000, 999999, 150000),
    (1000000, 1499999, 300000),
    (1500000, 1999999, 400000),
    (2000000, 2999999, 500000),
    (3000000, 4999999, 700000),
    (5000000, 6999999, 1000000),
    (7000000, 9999999, 1500000),
    (10000000, 14999999, 3000000),
    (15000000, 19999999, 4000000),
    (20000000, 29999999, 5000000),
    (30000000, float("inf"), 7000000),
]


def jpx_limit_amount(prev_close: float) -> float:
    if prev_close is None or not (prev_close > 0):
        return 0.0
    for lo, hi, lim in _JPX_LIMIT_TABLE:
        if lo <= prev_close <= hi:
            return float(lim)
    return 0.0


def round_price_like_exchange(price: float) -> float:
    """
    交易所實務有 tick-size，但 yfinance 日K 多半已經是成交價。
    這裡只做保守 round(2) 避免浮點誤差。
    """
    if price is None:
        return price
    return round(float(price), 2)


# -------------------------
# Strength bins（你要的：10% 起、每 10% 一格，到 100% 以上）
# -------------------------
def strength_bins_10_to_100() -> List[Tuple[int, str]]:
    bins = []
    for v in range(10, 100, 10):
        bins.append((v, f"RANK_{v}_{v+10}"))
    bins.append((100, "RANK_100UP"))
    return bins


@dataclass(frozen=True)
class MarketRule:
    market: str                 # e.g. "TW","CN","JP","US","HK","KR"
    market_detail: str          # e.g. listed/otc/emerging/main/20pct/unknown...
    limit_up_pct: Optional[float]   # None 表示無固定百分比漲停（興櫃/美股/港股/日股）
    strong_threshold: float         # 強勢日 threshold（用於貢獻度/統計）
    strength_intervals: List[Tuple[int, str]]  # 分箱 label 規則（百分比）
    pseudo_limit_pct: Optional[float] = None   # 給 US/HK 這類：用 10% 當「漲停研究用」


class MarketConfig:
    """
    統一入口：給 processor/event_engine 查規則 & 計算漲停價
    """

    @staticmethod
    def get_rule(market: str, market_detail: str, symbol: str = "") -> MarketRule:
        m = (market or "").upper()
        d = (market_detail or "").lower()
        sym = (symbol or "").upper()

        # ----- Taiwan -----
        if m == "TW":
            if d == "emerging":
                return MarketRule("TW", "emerging", None, 0.20, strength_bins_10_to_100())
            if d in ("otc", "gtsm"):
                return MarketRule("TW", "otc", 0.10, 0.10, [(10, "RANK_10UP")])
            # listed/tse/default
            return MarketRule("TW", "listed", 0.10, 0.10, [(10, "RANK_10UP")])

        # ----- Korea (你 processor.py 既有：30%) -----
        if m == "KR" or "KOSPI" in (market or "") or "KOSDAQ" in (market or ""):
            # 韓國也可以用 10~100 bins，但你目前 processor 是 10/20/30UP
            return MarketRule("KR", d or "kospi", 0.30, 0.30, [(10, "RANK_10_20"), (20, "RANK_20_30"), (30, "RANK_30UP")])

        # ----- China A-share -----
        # downloader_cn.py 會存 market=SSE/SZSE，market_detail=main/20pct
        if m in ("CN", "SSE", "SZSE"):
            # 直接用 market_detail 判斷
            if d in ("20pct", "chinext", "star"):
                return MarketRule("CN", "20pct", 0.20, 0.10, strength_bins_10_to_100())
            return MarketRule("CN", "main", 0.10, 0.10, strength_bins_10_to_100())

        # 也支援從 symbol 推斷（防呆）
        if sym.endswith(".SZ") or sym.endswith(".SS"):
            # 300/301 創業板；688 科創板 → 20%
            code = sym.split(".")[0]
            if code.startswith(("300", "301", "688")):
                return MarketRule("CN", "20pct", 0.20, 0.10, strength_bins_10_to_100())
            return MarketRule("CN", "main", 0.10, 0.10, strength_bins_10_to_100())

        # ----- Japan -----
        # JP: 沒有固定百分比，改用值幅金額判斷 stop high
        if m == "JP" or sym.endswith(".T"):
            return MarketRule("JP", "tse", None, 0.10, strength_bins_10_to_100())

        # ----- Hong Kong -----
        if m == "HK" or "HKEX" in (market or "") or sym.endswith(".HK"):
            # 無漲跌幅限制 → 用 pseudo 10%
            return MarketRule("HK", "hkex", None, 0.10, strength_bins_10_to_100(), pseudo_limit_pct=0.10)

        # ----- US -----
        if m in ("US", "NASDAQ", "NYSE", "AMEX"):
            return MarketRule("US", m, None, 0.10, strength_bins_10_to_100(), pseudo_limit_pct=0.10)

        # default
        return MarketRule(market or "UNK", market_detail or "unknown", None, 0.10, strength_bins_10_to_100(), pseudo_limit_pct=0.10)

    @staticmethod
    def calc_limit_price(prev_close: float, rule: MarketRule) -> Tuple[Optional[float], Optional[float]]:
        """
        回傳 (limit_up_price, limit_down_price)
        - 若固定百分比：prev_close*(1±pct)
        - 若 JP：prev_close ± 值幅金額
        - 若 US/HK pseudo：prev_close*(1+0.10)（只做研究用）
        """
        if prev_close is None or not (prev_close > 0):
            return None, None

        if rule.limit_up_pct is not None:
            up = round_price_like_exchange(prev_close * (1.0 + rule.limit_up_pct))
            dn = round_price_like_exchange(prev_close * (1.0 - rule.limit_up_pct))
            return up, dn

        # JPX
        if rule.market == "JP":
            lim = jpx_limit_amount(prev_close)
            up = round_price_like_exchange(prev_close + lim)
            dn = round_price_like_exchange(prev_close - lim)
            return up, dn

        # pseudo-limit（US/HK）
        if rule.pseudo_limit_pct is not None:
            up = round_price_like_exchange(prev_close * (1.0 + rule.pseudo_limit_pct))
            dn = round_price_like_exchange(prev_close * (1.0 - rule.pseudo_limit_pct))
            return up, dn

        return None, None

    @staticmethod
    def is_limit_up(close: float, prev_close: float, rule: MarketRule, tol: float = 0.999) -> int:
        up, _ = MarketConfig.calc_limit_price(prev_close, rule)
        if up is None or close is None:
            return 0
        return int(float(close) >= float(up) * tol)

    @staticmethod
    def is_limit_down(close: float, prev_close: float, rule: MarketRule, tol: float = 1.001) -> int:
        _, dn = MarketConfig.calc_limit_price(prev_close, rule)
        if dn is None or close is None:
            return 0
        return int(float(close) <= float(dn) * tol)

    @staticmethod
    def strength_label(change_pct: float, rule: MarketRule) -> str:
        if change_pct is None or math.isnan(change_pct):
            return "NA"
        if change_pct <= 0:
            return "NEGATIVE"

        # 由大到小掃，符合最大門檻
        intervals = sorted(rule.strength_intervals, key=lambda x: x[0], reverse=True)
        for min_v, label in intervals:
            if change_pct >= min_v:
                return label

        return "POSITIVE"
