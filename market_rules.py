# -*- coding: utf-8 -*-
"""
market_rules.py
---------------
全球市場規則集中管理（給 Global-Momentum-Dashboard- / processor.py 使用）

✅ 支援市場：
- TW：上市/上櫃(±10%)、興櫃(無漲跌幅限制)
- CN：主板(±10%)、創業板ChiNext(±20%)、科創板STAR(±20%)  ※ ST(±5%)暫未細分
- JP：日股動態漲跌停（金額制 -> 動態百分比），支援 Stop High / Stop Low 計算
- KR：KOSPI/KOSDAQ(±30%)
- US/HK：無固定漲跌幅限制（但可用統一強勢分箱統計）

✅ 你額外要的功能：
- 統一強勢分箱：10%以上每 10% 一箱直到 100%+
- 漲停開盤型態分類（GAP_UP / FLOATING / HIGH_VOLUME_LOCK / NO_VOLUME_LOCK / OTHER）
- 隔日衝 / 衝板失敗 / 昨日漲停今日未漲 等 flags 計算工具
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Tuple, Optional, Any

import numpy as np
import pandas as pd


# =========================================================
# 0) 共用：強勢分箱（10% 起跳，每 10% 一區間，到 100%+）
# =========================================================
def build_strength_intervals_10_to_100() -> List[Tuple[int, str]]:
    """
    產生：
      10-20, 20-30, ... 90-100, 100UP
    注意：返回值是 (min_val, label)
      - label 為 RANK_10_20 / RANK_20_30 ... / RANK_100UP
      - 你 processor.py 目前的 label_detailed_strength 是由 intervals 推導 next_min
    """
    intervals: List[Tuple[int, str]] = []
    for x in range(10, 100, 10):
        intervals.append((x, f"RANK_{x}_{x+10}"))
    intervals.append((100, "RANK_100UP"))
    return intervals


# =========================================================
# 1) 日本：動態漲跌停金額制（簡化常見區間版本）
#    來源表格你之前貼的那份「常見區間整理」，可再補完整表也很容易。
# =========================================================
def jp_limit_amount(prev_close: float) -> Optional[float]:
    """
    回傳：當日允許最大漲跌「金額（日圓）」(±amount)

    注意：
    - 日本正式表格更完整（還有更多價位區間）
    - 這裡先用你貼的常見區間（夠用於大部分股票）
    - 若遇到極端高價股不在區間內，會回傳 None（你可自行擴表）
    """
    if prev_close is None or not np.isfinite(prev_close) or prev_close <= 0:
        return None

    p = float(prev_close)

    # 常見區間（可擴充）
    # [low, high] -> amount
    table = [
        (0, 100, 30),
        (100, 200, 50),
        (200, 500, 80),
        (500, 700, 100),
        (700, 1000, 150),
        (1000, 1500, 300),
        (1500, 2000, 400),
        (2000, 3000, 500),
        (3000, 5000, 700),
        (5000, 7000, 1000),
        (7000, 10000, 1500),
        (10000, 15000, 3000),
        (15000, 20000, 4000),
        (20000, 30000, 5000),
        (30000, 50000, 7000),
        (50000, 70000, 10000),
        (70000, 100000, 15000),
        (100000, 150000, 30000),
        (150000, 200000, 40000),
        (200000, 300000, 50000),
    ]

    for low, high, amt in table:
        # 依你表格的語意：100～199 是 [100,200)
        if low <= p < high:
            return float(amt)

    # 超出範圍 -> 先回 None（你可以自行擴表）
    return None


def jp_limit_up_price(prev_close: float) -> Optional[float]:
    amt = jp_limit_amount(prev_close)
    if amt is None:
        return None
    # 日本是「金額」限制；實務上報價最小跳動單位也會影響，但你做日線回測可先忽略 tick
    return float(prev_close) + float(amt)


def jp_limit_down_price(prev_close: float) -> Optional[float]:
    amt = jp_limit_amount(prev_close)
    if amt is None:
        return None
    return float(prev_close) - float(amt)


def jp_limit_up_pct(prev_close: float) -> Optional[float]:
    """
    回傳動態漲停「百分比」(例如 0.15)
    """
    amt = jp_limit_amount(prev_close)
    if amt is None or prev_close is None or prev_close <= 0:
        return None
    return float(amt) / float(prev_close)


# =========================================================
# 2) MarketConfig：各市場規則
# =========================================================
@dataclass(frozen=True)
class MarketRule:
    """
    統一規則結構

    limit_mode:
      - "pct": 固定百分比漲跌幅限制（TW/CN/KR）
      - "none": 無固定漲跌幅（US/HK/TW興櫃）
      - "jp_amount": 日本金額制（需要 prev_close 才能算）
    """
    limit_mode: str
    limit_up_pct: Optional[float]          # 若 limit_mode="pct" 才會用到
    threshold: float                       # 強勢日門檻（用於年度巔峰貢獻等）
    strength_intervals: List[Tuple[int, str]]
    max_strength: int


class MarketConfig:
    """
    市場配置類別，統一管理不同市場規則
    - 你 processor.py 只要讀這裡回傳的 dict 就能跑
    - 日本的 dynamic limit：用 limit_mode="jp_amount"
    """

    # 統一分箱（你說美/港要用、也想讓日本也有）
    INTERVALS_10_100 = build_strength_intervals_10_to_100()

    MARKET_RULES: Dict[str, MarketRule] = {
        # -----------------------------
        # Taiwan
        # -----------------------------
        "TW_LISTED": MarketRule(
            limit_mode="pct",
            limit_up_pct=0.10,
            threshold=0.10,
            strength_intervals=[(10, "RANK_10UP")],
            max_strength=10,
        ),
        "TW_OTC": MarketRule(
            limit_mode="pct",
            limit_up_pct=0.10,
            threshold=0.10,
            strength_intervals=[(10, "RANK_10UP")],
            max_strength=10,
        ),
        "TW_EMERGING": MarketRule(
            limit_mode="none",
            limit_up_pct=None,
            threshold=0.20,
            strength_intervals=INTERVALS_10_100,
            max_strength=100,
        ),

        # -----------------------------
        # Korea
        # -----------------------------
        "KR_KOSPI": MarketRule(
            limit_mode="pct",
            limit_up_pct=0.30,
            threshold=0.30,
            strength_intervals=[(10, "RANK_10_20"), (20, "RANK_20_30"), (30, "RANK_30UP")],
            max_strength=30,
        ),
        "KR_KOSDAQ": MarketRule(
            limit_mode="pct",
            limit_up_pct=0.30,
            threshold=0.30,
            strength_intervals=[(10, "RANK_10_20"), (20, "RANK_20_30"), (30, "RANK_30UP")],
            max_strength=30,
        ),

        # -----------------------------
        # China A-share
        # 主板：10%
        # 創業板 / 科創板：20%
        # -----------------------------
        "CN_MAIN": MarketRule(
            limit_mode="pct",
            limit_up_pct=0.10,
            threshold=0.10,
            strength_intervals=INTERVALS_10_100,
            max_strength=100,
        ),
        "CN_CHINEXT": MarketRule(
            limit_mode="pct",
            limit_up_pct=0.20,
            threshold=0.20,
            strength_intervals=INTERVALS_10_100,
            max_strength=100,
        ),
        "CN_STAR": MarketRule(
            limit_mode="pct",
            limit_up_pct=0.20,
            threshold=0.20,
            strength_intervals=INTERVALS_10_100,
            max_strength=100,
        ),

        # -----------------------------
        # Japan (JPX / TSE)
        # 動態金額制（以 prev_close 計算）
        # -----------------------------
        "JP_TSE": MarketRule(
            limit_mode="jp_amount",
            limit_up_pct=None,
            threshold=0.10,                 # 你要統一做 10%+ 分箱，threshold 也可先用 10%
            strength_intervals=INTERVALS_10_100,
            max_strength=100,
        ),

        # -----------------------------
        # US / HK (no fixed limit)
        # -----------------------------
        "US": MarketRule(
            limit_mode="none",
            limit_up_pct=None,
            threshold=0.10,
            strength_intervals=INTERVALS_10_100,
            max_strength=100,
        ),
        "HK": MarketRule(
            limit_mode="none",
            limit_up_pct=None,
            threshold=0.10,
            strength_intervals=INTERVALS_10_100,
            max_strength=100,
        ),
    }

    @classmethod
    def get_market_config(cls, market: str, market_detail: str) -> MarketRule:
        """
        market / market_detail 來自 stock_info：
          - TW: market="TW", market_detail in listed/otc/emerging
          - CN: market in SSE/SZSE, market_detail in main/chinext/star
          - JP: market="JP" or "TSE", market_detail="tse"
          - HK: market="HK" or "HKEX"
          - US: market="US" or exchange strings
          - KR: market contains KOSPI/KOSDAQ
        """
        m = (market or "").upper()
        d = (market_detail or "").lower()

        # ---- Taiwan ----
        if m == "TW":
            if d == "emerging":
                return cls.MARKET_RULES["TW_EMERGING"]
            if d in ("listed", "tse"):
                return cls.MARKET_RULES["TW_LISTED"]
            if d in ("otc", "gtsm"):
                return cls.MARKET_RULES["TW_OTC"]
            return cls.MARKET_RULES["TW_LISTED"]

        # ---- Korea ----
        if m == "KR" or ("KOSPI" in m) or ("KOSDAQ" in m) or ("KOSPI" in d.upper()) or ("KOSDAQ" in d.upper()):
            if "KOSDAQ" in m or "kosdaq" in d:
                return cls.MARKET_RULES["KR_KOSDAQ"]
            return cls.MARKET_RULES["KR_KOSPI"]

        # ---- China ----
        if m in ("SSE", "SZSE", "CN"):
            if d in ("chinext", "cyb", "创业板"):
                return cls.MARKET_RULES["CN_CHINEXT"]
            if d in ("star", "kcb", "科创板"):
                return cls.MARKET_RULES["CN_STAR"]
            return cls.MARKET_RULES["CN_MAIN"]

        # ---- Japan ----
        if m in ("JP", "TSE", "JPX") or d in ("tse", "jpx"):
            return cls.MARKET_RULES["JP_TSE"]

        # ---- Hong Kong ----
        if m in ("HK", "HKEX") or d in ("hkex", "hk"):
            return cls.MARKET_RULES["HK"]

        # ---- US ----
        if m in ("US", "NASDAQ", "NYSE", "AMEX"):
            return cls.MARKET_RULES["US"]

        # fallback：用 US 規則（無固定漲停，但保留分箱）
        return cls.MARKET_RULES["US"]


# =========================================================
# 3) 共用：計算漲停價 / 是否漲停 / 是否摸到漲停（含日本動態）
# =========================================================
def calc_limit_up_price(prev_close: float, rule: MarketRule) -> Optional[float]:
    if prev_close is None or not np.isfinite(prev_close) or prev_close <= 0:
        return None

    if rule.limit_mode == "pct":
        if rule.limit_up_pct is None:
            return None
        # 你 processor 目前 round 到 2 位；這裡不強制，交給 processor 或你自行決定
        return float(prev_close) * (1.0 + float(rule.limit_up_pct))

    if rule.limit_mode == "jp_amount":
        return jp_limit_up_price(prev_close)

    # none
    return None


def is_limit_up_day(close: float, prev_close: float, rule: MarketRule, tol: float = 0.999) -> int:
    """
    close 是否達到漲停價（可用 tol 放寬）
    """
    lp = calc_limit_up_price(prev_close, rule)
    if lp is None or close is None or not np.isfinite(close):
        return 0
    return int(float(close) >= float(lp) * tol)


def hit_limit_up_intraday(high: float, prev_close: float, rule: MarketRule, tol: float = 0.999) -> int:
    """
    high 是否盤中摸到漲停價（用於「衝板失敗」）
    """
    lp = calc_limit_up_price(prev_close, rule)
    if lp is None or high is None or not np.isfinite(high):
        return 0
    return int(float(high) >= float(lp) * tol)


# =========================================================
# 4) 漲停開盤型態分類（你的文章那套）
#    注意：日線 OHLC 沒有分鐘級資料，所以「盤中逐步推升」只能用 close/open 近似。
# =========================================================
def classify_limit_open_type(
    open_px: float,
    high_px: float,
    close_px: float,
    prev_close: float,
    volume: float,
    vol_ma5: float,
    is_limit_up: int,
    rule: MarketRule,
) -> str:
    """
    回傳：
      GAP_UP / FLOATING / HIGH_VOLUME_LOCK / NO_VOLUME_LOCK / OTHER

    依你定義的中間判斷：
      is_gap     = (open/prev_close - 1) >= 0.07
      is_highvol = (volume/vol_ma5) >= 3
      is_lowvol  = (volume/vol_ma5) <= 0.4
      is_float   = (not is_gap) and (close/open - 1 >= 0.05)

    最終優先序（你給的）：
      GAP_UP_LOCK   ：is_gap 且 is_low_vol
      GAP_UP        ：is_gap 且 非 is_low_vol
      FLOAT_HV      ：is_float 且 is_high_vol
      FLOAT         ：is_float 且 非 is_high_vol
      LOW_VOL_LOCK  ：is_low_vol
      HIGH_VOL_LOCK ：is_high_vol
      OTHER

    然後你要合併成四大類：
      FLOATING / GAP_UP / OTHER / HIGH_VOLUME_LOCK / NO_VOLUME_LOCK
    """
    # 基礎防呆
    if prev_close is None or not np.isfinite(prev_close) or prev_close <= 0:
        return "OTHER"
    if open_px is None or not np.isfinite(open_px) or open_px <= 0:
        return "OTHER"
    if close_px is None or not np.isfinite(close_px) or close_px <= 0:
        return "OTHER"

    vr = None
    if vol_ma5 is not None and np.isfinite(vol_ma5) and vol_ma5 > 0 and volume is not None and np.isfinite(volume):
        vr = float(volume) / float(vol_ma5)

    is_gap = (float(open_px) / float(prev_close) - 1.0) >= 0.07
    is_high_vol = (vr is not None) and (vr >= 3.0)
    is_low_vol = (vr is not None) and (vr <= 0.4)
    is_float = (not is_gap) and ((float(close_px) / float(open_px) - 1.0) >= 0.05)

    # 先走你那套 7 類，再合併
    if is_gap and is_low_vol:
        fine = "GAP_UP_LOCK"
    elif is_gap and (not is_low_vol):
        fine = "GAP_UP"
    elif is_float and is_high_vol:
        fine = "FLOAT_HV"
    elif is_float and (not is_high_vol):
        fine = "FLOAT"
    elif is_low_vol:
        fine = "LOW_VOL_LOCK"
    elif is_high_vol:
        fine = "HIGH_VOL_LOCK"
    else:
        fine = "OTHER"

    # 合併
    if fine in ("FLOAT", "FLOAT_HV"):
        return "FLOATING"
    if fine in ("GAP_UP", "GAP_UP_LOCK"):
        return "GAP_UP"
    if fine == "HIGH_VOL_LOCK":
        return "HIGH_VOLUME_LOCK"
    if fine == "LOW_VOL_LOCK":
        return "NO_VOLUME_LOCK"
    return "OTHER"


# =========================================================
# 5) 隔日衝 / 衝板失敗等 flags（給你做統計用）
# =========================================================
def add_daytrade_flags(df: pd.DataFrame, rule: MarketRule) -> pd.DataFrame:
    """
    需要 df 至少包含：
      open, high, close, volume
    並且要先有：
      prev_close (= close.shift(1))
      vol_ma5 (= volume.rolling(5).mean())
    建議在 processor.py 內 group-by symbol 後先補好 prev_close/vol_ma5，再呼叫此函式。

    會新增欄位：
      limit_up_price         : 當日漲停價（若 market 無固定漲停則為 NaN）
      hit_limit_up           : 盤中是否摸到漲停
      is_limit_up            : 收盤是否漲停（鎖住）
      fail_limit_up          : 盤中摸到漲停但收盤沒鎖（衝板失敗）
      yday_is_limit_up       : 昨天是否漲停
      yday_limit_today_not   : 昨天漲停、今天沒漲停（隔日衝常用觀察）
      yday_not_limit_today_fail : 昨天沒漲停、今天衝板失敗
      limit_open_type        : 漲停開盤型態（五類）
    """
    out = df.copy()

    # 先算當日漲停價（可能為 None）
    def _limit_price_row(r):
        pc = r.get("prev_close")
        return calc_limit_up_price(pc, rule) if pc is not None else None

    out["limit_up_price"] = out.apply(_limit_price_row, axis=1)

    # hit / close
    out["hit_limit_up"] = out.apply(
        lambda r: hit_limit_up_intraday(r.get("high"), r.get("prev_close"), rule),
        axis=1,
    )
    out["is_limit_up"] = out.apply(
        lambda r: is_limit_up_day(r.get("close"), r.get("prev_close"), rule),
        axis=1,
    )

    # 衝板失敗：摸到漲停但收盤沒鎖
    out["fail_limit_up"] = ((out["hit_limit_up"] == 1) & (out["is_limit_up"] == 0)).astype(int)

    # 昨日漲停 / 昨日漲停今日未漲停
    out["yday_is_limit_up"] = out["is_limit_up"].shift(1).fillna(0).astype(int)
    out["yday_limit_today_not"] = ((out["yday_is_limit_up"] == 1) & (out["is_limit_up"] == 0)).astype(int)

    # 昨天沒漲停，但今天衝板失敗
    out["yday_not_limit_today_fail"] = ((out["yday_is_limit_up"] == 0) & (out["fail_limit_up"] == 1)).astype(int)

    # 漲停開盤型態（就算不是漲停日，也可以給 OTHER）
    out["limit_open_type"] = out.apply(
        lambda r: classify_limit_open_type(
            open_px=r.get("open"),
            high_px=r.get("high"),
            close_px=r.get("close"),
            prev_close=r.get("prev_close"),
            volume=r.get("volume"),
            vol_ma5=r.get("vol_ma5"),
            is_limit_up=int(r.get("is_limit_up", 0)),
            rule=rule,
        ),
        axis=1,
    )

    return out
