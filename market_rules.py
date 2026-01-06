# market_rules.py
# -*- coding: utf-8 -*-
"""
market_rules.py
---------------
跨市場「制度漲跌幅 / 漲停(Stop High)」與「研究口徑(>=10%分箱)」規則集中管理

✅ 目的
1) 把各市場制度規則從 processor.py 抽離，避免 processor 越長越難維護
2) 同時支援兩種口徑：
   - regulatory / exchange rule：制度漲停(台灣10%、韓國30%、日本值幅制限；美/港無固定漲跌幅)
   - research / custom rule：你要的「>=10% 當類漲停」+「10%為步長分箱到>=100%」

✅ 注意
- US: 無固定漲跌幅(有 LULD/熔斷，但不是固定每日±X%) → 在此以 limit_up_pct=None 表示
- HK: 無固定漲跌幅(有 VCM 冷靜期，但不是固定每日±X%) → limit_up_pct=None
- JP: 値幅制限是「依前日收盤價區間，限制日圓幅度」→ 用 jp_limit_yen(prev_close) 計算
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, List, Tuple, Dict, Any

import math


# =========================================================
# 1) Rule dataclass
# =========================================================
@dataclass(frozen=True)
class MarketRule:
    """
    market: 建議用 'TW','KR','JP','US','HK','CN' 等大寫
    market_detail: 例如 'listed','otc','emerging','kosdaq','kospi' 等
    limit_up_pct:   固定百分比漲停（TW=0.10, KR=0.30；US/HK=None；JP=None 走日圓表）
    strong_threshold: processor 用的「強勢日」門檻（你想要 TW=10%、KR=30%、TW emerging=20% 等）
    custom_limit_up_pct: 你研究口徑的統一「>=10% 當類漲停」
    """
    market: str
    market_detail: str = ""
    limit_up_pct: Optional[float] = None
    limit_down_pct: Optional[float] = None
    strong_threshold: float = 0.10
    custom_limit_up_pct: float = 0.10

    # 你要的「分箱」設定：>=10% 起，每 10% 一格，直到 >=100%
    # 這是研究口徑，不等於交易所制度
    bin_step_pct: int = 10
    bin_start_pct: int = 10
    bin_max_pct: int = 100


# =========================================================
# 2) Market rule resolver
# =========================================================
def normalize_market(market: str) -> str:
    if market is None:
        return ""
    s = str(market).strip().upper()
    # downloader 有時會寫 KOSPI/KOSDAQ 到 market 欄位
    if s in ("KOSPI", "KOSDAQ"):
        return "KR"
    if s in ("HKEX",):
        return "HK"
    return s


def normalize_detail(detail: str) -> str:
    if detail is None:
        return ""
    return str(detail).strip().lower()


def get_rule(market: str, market_detail: str = "") -> MarketRule:
    """
    統一入口：根據 stock_info.market / stock_info.market_detail 決定規則
    """
    m = normalize_market(market)
    d = normalize_detail(market_detail)

    # -----------------------------
    # Taiwan
    # -----------------------------
    if m == "TW":
        # 興櫃：無固定漲跌幅限制（實務上有撮合限制等，但不等同固定±10%），你的 processor 也把它當 unrestricted
        if d in ("emerging", "emg", "emerging_board"):
            return MarketRule(
                market="TW",
                market_detail="emerging",
                limit_up_pct=None,
                limit_down_pct=None,
                strong_threshold=0.20,   # 你原本的設定
                custom_limit_up_pct=0.10
            )

        # 上市/上櫃：固定 10%
        if d in ("listed", "tse", "twse"):
            return MarketRule(
                market="TW",
                market_detail=d,
                limit_up_pct=0.10,
                limit_down_pct=0.10,
                strong_threshold=0.10,
                custom_limit_up_pct=0.10
            )

        if d in ("otc", "gtsm", "tpex"):
            return MarketRule(
                market="TW",
                market_detail=d,
                limit_up_pct=0.10,
                limit_down_pct=0.10,
                strong_threshold=0.10,
                custom_limit_up_pct=0.10
            )

        # fallback: 當上市處理
        return MarketRule(
            market="TW",
            market_detail=d,
            limit_up_pct=0.10,
            limit_down_pct=0.10,
            strong_threshold=0.10,
            custom_limit_up_pct=0.10
        )

    # -----------------------------
    # Korea (KOSPI/KOSDAQ): 30%
    # -----------------------------
    if m == "KR":
        # 你 downloader_kr 裡 market_detail 有 main/kosdaq/konex
        # processor 目前只特化 kospi/kosdaq；konex 先當 kosdaq 或 kospi 都行
        if "kosdaq" in d:
            md = "kosdaq"
        elif "kospi" in d or d == "main":
            md = "kospi"
        elif "konex" in d:
            md = "konex"
        else:
            md = d or "kospi"

        return MarketRule(
            market="KR",
            market_detail=md,
            limit_up_pct=0.30,
            limit_down_pct=0.30,
            strong_threshold=0.30,  # 你原本的設定
            custom_limit_up_pct=0.10
        )

    # -----------------------------
    # Japan: 値幅制限(以日圓計算)
    # -----------------------------
    if m == "JP":
        return MarketRule(
            market="JP",
            market_detail=d or "tse",
            limit_up_pct=None,      # 用日圓表
            limit_down_pct=None,    # 用日圓表
            strong_threshold=0.10,  # 研究用強勢日；你也可改成更高
            custom_limit_up_pct=0.10
        )

    # -----------------------------
    # US / HK: no fixed daily pct limit
    # -----------------------------
    if m == "US":
        return MarketRule(
            market="US",
            market_detail=d,
            limit_up_pct=None,
            limit_down_pct=None,
            strong_threshold=0.10,
            custom_limit_up_pct=0.10
        )

    if m == "HK":
        return MarketRule(
            market="HK",
            market_detail=d,
            limit_up_pct=None,
            limit_down_pct=None,
            strong_threshold=0.10,
            custom_limit_up_pct=0.10
        )

    # -----------------------------
    # CN: 一般 A 股有 10%/20% 等限制但牽涉 ST/創業板/科創板等細節
    # 先保守：不在這裡硬編（避免錯），等你要做再補齊
    # -----------------------------
    if m == "CN":
        return MarketRule(
            market="CN",
            market_detail=d,
            limit_up_pct=None,
            limit_down_pct=None,
            strong_threshold=0.10,
            custom_limit_up_pct=0.10
        )

    # fallback
    return MarketRule(
        market=m or "UNKNOWN",
        market_detail=d,
        limit_up_pct=None,
        limit_down_pct=None,
        strong_threshold=0.10,
        custom_limit_up_pct=0.10
    )


# =========================================================
# 3) Japan (JPX/TSE) price limit table (値幅制限)
# =========================================================
# 値幅制限：依「前日收盤價」決定當日允許的「日圓」漲跌幅度
# 這裡給的是「常用」表格（已涵蓋大多數價格帶），足夠做制度漲停/跌停判斷。
#
# 格式：(upper_bound_inclusive, limit_yen)
# 例如 prev_close <= 100 → ±30 yen
#
# 你之後若要「完全對齊官方最新表」，只要換這張表即可，不用動 processor。
_JP_LIMIT_TABLE: List[Tuple[float, int]] = [
    (100, 30),
    (200, 50),
    (500, 80),
    (700, 100),
    (1000, 150),
    (1500, 300),
    (2000, 400),
    (3000, 500),
    (5000, 700),
    (7000, 1000),
    (10000, 1500),
    (15000, 3000),
    (20000, 4000),
    (30000, 5000),
    (50000, 7000),
    (70000, 10000),
    (100000, 15000),
    (150000, 30000),
    (200000, 40000),
    (300000, 50000),
    (500000, 70000),
    (700000, 100000),
    (1000000, 150000),
    (1500000, 300000),
    (2000000, 400000),
    (3000000, 500000),
    (5000000, 700000),
    (7000000, 1000000),
    (10000000, 1500000),
    (15000000, 3000000),
    (20000000, 4000000),
    (30000000, 5000000),
    (50000000, 7000000),
]


def jp_limit_yen(prev_close: float) -> int:
    """
    回傳日本市場該檔在 prev_close 下的「制度」漲跌幅限制（日圓）
    """
    try:
        if prev_close is None:
            return 0
        p = float(prev_close)
        if not math.isfinite(p) or p <= 0:
            return 0
    except Exception:
        return 0

    for upper, lim in _JP_LIMIT_TABLE:
        if p <= upper:
            return lim

    # 超出最高區間時給一個很大的值（保底）
    return _JP_LIMIT_TABLE[-1][1]


def jp_limit_prices(prev_close: float) -> Tuple[float, float]:
    """
    回傳 (limit_up_price, limit_down_price)
    """
    lim = jp_limit_yen(prev_close)
    try:
        p = float(prev_close)
        return p + lim, p - lim
    except Exception:
        return (0.0, 0.0)


# =========================================================
# 4) Research bins (>=10%, step=10%, to >=100%)
# =========================================================
def label_change_bin_10pct(
    daily_change: float,
    bin_start_pct: int = 10,
    bin_step_pct: int = 10,
    bin_max_pct: int = 100,
) -> str:
    """
    研究口徑：把 daily_change(小數) 依「10%步長」做分箱：
    - chg < 0         → NEGATIVE
    - 0 ~ <10         → POS_0_10
    - 10 ~ <20        → POS_10_20
    ...
    - 90 ~ <100       → POS_90_100
    - >=100           → POS_100UP
    """
    try:
        if daily_change is None:
            return "NA"
        chg = float(daily_change) * 100.0
        if not math.isfinite(chg):
            return "NA"
    except Exception:
        return "NA"

    if chg < 0:
        return "NEGATIVE"
    if chg < bin_start_pct:
        return f"POS_0_{bin_start_pct}"

    if chg >= bin_max_pct:
        return f"POS_{bin_max_pct}UP"

    lo = int(chg // bin_step_pct) * bin_step_pct
    hi = lo + bin_step_pct
    # 確保下界至少從 bin_start_pct 開始
    if lo < bin_start_pct:
        lo = bin_start_pct
        hi = lo + bin_step_pct
    return f"POS_{lo}_{hi}"


def is_custom_limit_up(daily_change: float, custom_limit_up_pct: float = 0.10) -> int:
    """
    研究口徑：>=10% 當類漲停
    """
    try:
        if daily_change is None:
            return 0
        return int(float(daily_change) >= float(custom_limit_up_pct))
    except Exception:
        return 0


# =========================================================
# 5) Regulatory limit-up / limit-down evaluator
# =========================================================
def calc_reg_limit_flags(
    market: str,
    market_detail: str,
    prev_close: float,
    close: float,
    price_round: Optional[int] = 2,
    eps: float = 0.001,
) -> Dict[str, Any]:
    """
    計算「制度」漲停/跌停旗標（regulatory）
    回傳 dict：
      - is_limit_up_reg
      - is_limit_down_reg
      - limit_up_price_reg
      - limit_down_price_reg
      - reg_rule_type: 'PCT' | 'JPY' | 'NONE'
    """
    rule = get_rule(market, market_detail)

    # 基礎容錯
    try:
        pc = float(prev_close) if prev_close is not None else float("nan")
        cc = float(close) if close is not None else float("nan")
        if not (math.isfinite(pc) and math.isfinite(cc)) or pc <= 0:
            return {
                "is_limit_up_reg": 0,
                "is_limit_down_reg": 0,
                "limit_up_price_reg": None,
                "limit_down_price_reg": None,
                "reg_rule_type": "NONE",
            }
    except Exception:
        return {
            "is_limit_up_reg": 0,
            "is_limit_down_reg": 0,
            "limit_up_price_reg": None,
            "limit_down_price_reg": None,
            "reg_rule_type": "NONE",
        }

    # JP: yen table
    if rule.market == "JP":
        up, down = jp_limit_prices(pc)
        if price_round is not None:
            up = round(up, price_round)
            down = round(down, price_round)

        return {
            "is_limit_up_reg": int(cc >= up * (1 - eps)),
            "is_limit_down_reg": int(cc <= down * (1 + eps)),
            "limit_up_price_reg": up,
            "limit_down_price_reg": down,
            "reg_rule_type": "JPY",
        }

    # TW/KR fixed pct
    if rule.limit_up_pct is not None:
        up = pc * (1 + rule.limit_up_pct)
        down = pc * (1 - (rule.limit_down_pct if rule.limit_down_pct is not None else rule.limit_up_pct))
        if price_round is not None:
            up = round(up, price_round)
            down = round(down, price_round)

        return {
            "is_limit_up_reg": int(cc >= up * (1 - eps)),
            "is_limit_down_reg": int(cc <= down * (1 + eps)),
            "limit_up_price_reg": up,
            "limit_down_price_reg": down,
            "reg_rule_type": "PCT",
        }

    # US/HK etc: none
    return {
        "is_limit_up_reg": 0,
        "is_limit_down_reg": 0,
        "limit_up_price_reg": None,
        "limit_down_price_reg": None,
        "reg_rule_type": "NONE",
    }
