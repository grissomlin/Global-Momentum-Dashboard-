# event_engine.py
# -*- coding: utf-8 -*-
"""
event_engine.py
---------------
事件表引擎（漲停型態 + 隔日沖 + 未來報酬）
只做「事件表」，不做 feature layer（feature layer 在 processor.py -> stock_analysis）

依你的分工：
- market_rules.py：市場規則（TW/CN/JP 漲停判定 + tick + bins）
- processor.py：寫 stock_analysis（is_limit_up / lu_type / consecutive_limits / ...）
- event_engine.py：從 stock_analysis / stock_prices 產生事件表（limit-up events + daytrade events）

✅ 事件表內容：
A) limit_up_events（漲停日事件）
- 漲停型態 lu_type（你文章規則）
- 一字鎖 is_one_tick_lock
- 連板 consecutive_limits
- 隔日關鍵欄位：next_open_ret / next_open_gap / next_intraday_drawdown / next_close_ret / next_high_ret
- 未來報酬：ret_1d / ret_5d / max_up_5d / max_dd_5d

B) daytrade_events（隔日沖研究事件）
- 昨天漲停今天沒漲：prev_limit_up_today_not
- 昨天漲停今天「再衝漲停失敗」：prev_limit_up_today_fail
- 昨天沒漲停但今天「衝漲停失敗」：today_limit_up_fail_no_prev
- 昨天沒漲停但今天「收漲停」：today_limit_up_yes_no_prev
- 以及隔日開盤、盤中回撤、未來報酬等欄位（同樣結構）

⚠️ 前提：
- DB 需已有 stock_prices 與 stock_info
- processor.py 跑過會產生 stock_analysis（建議使用 stock_analysis 的 is_limit_up / lu_type / consecutive_limits / is_one_tick_lock）
- 若 stock_analysis 不存在，event_engine 會 fallback 用 stock_prices + market_rules 自算 is_limit_up（但你會少掉 processor 算好的 lu_type/consecutive_limits 等）

⚠️ 只支援 TW/CN/JP 的「精準漲停」：
- 若 market_rules.py 存在且提供 calc_limit_up_price() / tick_size()，會用精準判定
- 否則 fallback：TW=10%、CN=10/20%、JP=不判定（當作無漲停）

✅ 本版本修正（重要）：
- prev_close / daily_change 一律用 per-symbol groupby 計算或補缺（避免跨股票污染）
- only_markets 真的生效（可指定只跑 tw/cn/jp 子集合）
- 市場過濾不再用 apply(axis=1)（大幅加速）
- tick buffer 更 robust（NaN/例外時不炸）
- 完全不需要、也不會呼叫 data_cleaning.py
"""

import sqlite3
import pandas as pd
import numpy as np
from datetime import datetime
import warnings
from typing import Optional, Dict, Tuple, Iterable

warnings.filterwarnings("ignore", category=FutureWarning)

# -----------------------------
# market_rules: 精準漲停判定（建議你一定要有）
# -----------------------------
try:
    import market_rules  # type: ignore
    HAS_MARKET_RULES = True
except Exception:
    market_rules = None
    HAS_MARKET_RULES = False


# =============================================================================
# 0) 市場 key 判斷（向量化用）
# =============================================================================
def _market_to_key(market: str, symbol: str) -> str:
    m = (market or "").upper().strip()
    sym = (symbol or "").upper().strip()

    # TW
    if m in ("TW", "TSE", "GTSM") or sym.endswith(".TW") or sym.endswith(".TWO"):
        return "tw"
    # CN
    if m in ("SSE", "SZSE", "CN", "CHINA") or sym.endswith(".SS") or sym.endswith(".SZ"):
        return "cn"
    # JP
    if m in ("JP", "JPX", "TSEJ") or sym.endswith(".T"):
        return "jp"
    return "other"


# =============================================================================
# 1) Fallback 規則（只在 market_rules 不存在/不可用時）
# =============================================================================
def _fallback_cn_limit_pct(symbol: str) -> float:
    sym = (symbol or "").upper()
    code = "".join([c for c in sym if c.isdigit()])
    # CN: 創業板/科創板 20%
    if code.startswith(("300", "301", "688")):
        return 0.20
    return 0.10


def _fallback_calc_limit_up_price(prev_close: pd.Series, market: str, symbol: str) -> Optional[pd.Series]:
    m = (market or "").upper().strip()
    sym = (symbol or "").upper().strip()

    # TW
    if m in ("TW", "TSE", "GTSM") or sym.endswith(".TW") or sym.endswith(".TWO"):
        return (prev_close.astype(float) * 1.10).round(2)

    # CN
    if m in ("SSE", "SZSE", "CN", "CHINA") or sym.endswith(".SS") or sym.endswith(".SZ"):
        pct = _fallback_cn_limit_pct(sym)
        return (prev_close.astype(float) * (1 + pct)).round(2)

    # JP fallback：不做漲停
    return None


# =============================================================================
# 2) 讀取資料：stock_analysis（優先）/ fallback stock_prices
# =============================================================================
def _read_base_df(conn: sqlite3.Connection) -> pd.DataFrame:
    # 優先讀 stock_analysis（processor 產物）
    try:
        df = pd.read_sql(
            """
            SELECT
                a.*,
                i.sector AS info_sector
            FROM stock_analysis a
            LEFT JOIN stock_info i ON a.symbol = i.symbol
            """,
            conn,
        )
        if not df.empty:
            return df
    except Exception:
        pass

    # fallback：用 stock_prices + stock_info（但會少很多 feature）
    df = pd.read_sql(
        """
        SELECT
            p.*,
            i.market,
            i.market_detail,
            i.sector
        FROM stock_prices p
        LEFT JOIN stock_info i ON p.symbol = i.symbol
        """,
        conn,
    )
    return df


# =============================================================================
# 3) 精準判定：今日是否「收漲停」、是否「盤中到漲停但收不住」（fail）
# =============================================================================
def _calc_limit_flags(df_sym: pd.DataFrame) -> pd.DataFrame:
    """
    input df_sym：單一股票、按 date 升序，至少有 close/high/open/prev_close/market/market_detail/symbol

    output columns：
    - limit_up_price
    - hit_limit_up_close (close >= limit)
    - hit_limit_up_intraday (high >= limit)
    - limit_up_fail (high >= limit AND close < limit)
    """
    if df_sym.empty:
        return df_sym

    symbol = str(df_sym["symbol"].iloc[0])
    market = str(df_sym["market"].iloc[0]) if "market" in df_sym.columns else ""
    market_detail = str(df_sym["market_detail"].iloc[0]) if "market_detail" in df_sym.columns else ""

    prev_close = pd.to_numeric(df_sym.get("prev_close", np.nan), errors="coerce").astype(float)

    limit_price = None
    if HAS_MARKET_RULES and hasattr(market_rules, "calc_limit_up_price"):
        try:
            limit_price = market_rules.calc_limit_up_price(
                prev_close=prev_close,
                market=market,
                market_detail=market_detail,
                symbol=symbol,
            )
        except Exception:
            limit_price = None

    if limit_price is None:
        limit_price = _fallback_calc_limit_up_price(prev_close, market, symbol)

    if limit_price is None:
        df_sym["limit_up_price"] = np.nan
        df_sym["hit_limit_up_close"] = 0
        df_sym["hit_limit_up_intraday"] = 0
        df_sym["limit_up_fail"] = 0
        return df_sym

    df_sym["limit_up_price"] = pd.to_numeric(limit_price, errors="coerce").astype(float)

    # buffer：若有 tick_size，用 tick*1.0 當容忍；否則 0
    buffer = 0.0
    if HAS_MARKET_RULES and hasattr(market_rules, "tick_size"):
        try:
            # 對每列 prev_close 算 tick；NaN/<=0 -> 0
            def _tick(x: float) -> float:
                if not np.isfinite(x) or x <= 0:
                    return 0.0
                try:
                    return float(market_rules.tick_size(float(x), market=market, symbol=symbol))
                except Exception:
                    return 0.0

            ticks = prev_close.apply(_tick).fillna(0.0)
            buffer = (ticks * 1.0).astype(float)  # 1 tick 容忍
        except Exception:
            buffer = 0.0

    lim = pd.to_numeric(df_sym["limit_up_price"], errors="coerce").astype(float)
    hi = pd.to_numeric(df_sym.get("high", np.nan), errors="coerce").astype(float)
    cl = pd.to_numeric(df_sym.get("close", np.nan), errors="coerce").astype(float)

    # 若 buffer 是 scalar
    if isinstance(buffer, (int, float)):
        buf = float(buffer)
        df_sym["hit_limit_up_close"] = (cl >= (lim - buf)).astype(int)
        df_sym["hit_limit_up_intraday"] = (hi >= (lim - buf)).astype(int)
    else:
        # buffer 是 series（同 index）
        df_sym["hit_limit_up_close"] = (cl >= (lim - buffer)).astype(int)
        df_sym["hit_limit_up_intraday"] = (hi >= (lim - buffer)).astype(int)

    df_sym["limit_up_fail"] = ((df_sym["hit_limit_up_intraday"] == 1) & (df_sym["hit_limit_up_close"] == 0)).astype(int)
    return df_sym


# =============================================================================
# 4) 計算隔日欄位 + 未來報酬
# =============================================================================
def _add_forward_metrics(df_sym: pd.DataFrame) -> pd.DataFrame:
    """
    需要欄位：open/high/low/close/date
    會新增：
    - next_open/next_high/next_low/next_close
    - next_open_ret, next_open_gap, next_intraday_drawdown, next_close_ret, next_high_ret
    - ret_1d, ret_5d, max_up_5d, max_dd_5d
    """
    df_sym = df_sym.sort_values("date").copy()

    for col in ["open", "high", "low", "close"]:
        if col in df_sym.columns:
            df_sym[col] = pd.to_numeric(df_sym[col], errors="coerce")

    df_sym["next_open"] = df_sym["open"].shift(-1)
    df_sym["next_high"] = df_sym["high"].shift(-1)
    df_sym["next_low"] = df_sym["low"].shift(-1)
    df_sym["next_close"] = df_sym["close"].shift(-1)

    # next day (close-based)
    df_sym["next_open_ret"] = (df_sym["next_open"] / df_sym["close"] - 1)
    df_sym["next_open_gap"] = (df_sym["next_open"] / df_sym["close"] - 1)  # 同 next_open_ret（命名給你文章好讀）
    df_sym["next_intraday_drawdown"] = (df_sym["next_low"] / df_sym["next_open"] - 1)
    df_sym["next_close_ret"] = (df_sym["next_close"] / df_sym["close"] - 1)
    df_sym["next_high_ret"] = (df_sym["next_high"] / df_sym["close"] - 1)

    # future returns: close based
    df_sym["close_t1"] = df_sym["close"].shift(-1)
    df_sym["close_t5"] = df_sym["close"].shift(-5)

    df_sym["ret_1d"] = (df_sym["close_t1"] / df_sym["close"] - 1)
    df_sym["ret_5d"] = (df_sym["close_t5"] / df_sym["close"] - 1)

    # max up / max dd in next 5 days (using high/low) window: t+1..t+5
    highs_fwd = pd.concat([df_sym["high"].shift(-i) for i in range(1, 6)], axis=1)
    lows_fwd = pd.concat([df_sym["low"].shift(-i) for i in range(1, 6)], axis=1)

    df_sym["max_up_5d"] = (highs_fwd.max(axis=1) / df_sym["close"] - 1)
    df_sym["max_dd_5d"] = (lows_fwd.min(axis=1) / df_sym["close"] - 1)

    return df_sym


# =============================================================================
# 5) 生成事件表
# =============================================================================
def build_event_tables(db_path: str, only_markets: Iterable[str] = ("tw", "cn", "jp")) -> dict:
    """
    產生兩張表：
    - limit_up_events
    - daytrade_events

    only_markets: ('tw','cn','jp') 用於 safety gate（main.py 也會控制）
    """
    t0 = datetime.now()
    conn = sqlite3.connect(db_path)

    try:
        base = _read_base_df(conn)
        if base is None or base.empty:
            print("❌ event_engine: 找不到可用資料（stock_analysis/stock_prices 皆空）")
            return {"ok": False, "reason": "empty"}

        # ====== 基礎清理（不依賴 data_cleaning.py）======
        base["date"] = pd.to_datetime(base.get("date", None), errors="coerce")
        base = base.dropna(subset=["date"])
        base = base.sort_values(["symbol", "date"]).reset_index(drop=True)

        # sector 欄位名可能不同（stock_analysis join 的 info_sector）
        if "sector" not in base.columns and "info_sector" in base.columns:
            base["sector"] = base["info_sector"]

        # numeric cast
        for c in ["open", "high", "low", "close", "volume"]:
            if c in base.columns:
                base[c] = pd.to_numeric(base[c], errors="coerce")

        # prev_close / daily_change 必須 per-symbol（避免跨股票污染）
        base["prev_close"] = pd.to_numeric(base.get("prev_close", np.nan), errors="coerce")
        base["prev_close"] = base["prev_close"].fillna(base.groupby("symbol")["close"].shift(1))

        base["daily_change"] = pd.to_numeric(base.get("daily_change", np.nan), errors="coerce")
        base["daily_change"] = base["daily_change"].fillna(base.groupby("symbol")["close"].pct_change())

        # is_limit_up 若不存在：先全部 0，後面會用精準判定回填
        if "is_limit_up" not in base.columns:
            base["is_limit_up"] = 0
        base["is_limit_up"] = pd.to_numeric(base["is_limit_up"], errors="coerce").fillna(0).astype(int)

        # lu_type / consecutive_limits / is_one_tick_lock 若不存在：先補空（事件仍可跑）
        if "lu_type" not in base.columns:
            base["lu_type"] = None
        if "consecutive_limits" not in base.columns:
            base["consecutive_limits"] = 0
        else:
            base["consecutive_limits"] = pd.to_numeric(base["consecutive_limits"], errors="coerce").fillna(0).astype(int)

        if "is_one_tick_lock" not in base.columns:
            base["is_one_tick_lock"] = (
                (base["open"] == base["close"]) &
                (base["high"] == base["low"]) &
                (base["high"] == base["close"])
            ).astype(int)
        else:
            base["is_one_tick_lock"] = pd.to_numeric(base["is_one_tick_lock"], errors="coerce").fillna(0).astype(int)

        # ====== 市場 gate（only_markets 真正生效）======
        only = set([str(m).lower().strip() for m in (only_markets or ())])
        if not only:
            only = {"tw", "cn", "jp"}

        # 缺 market 欄位也能用 suffix 推斷（market_to_key 有處理）
        mcol = base["market"] if "market" in base.columns else ""
        base["_mkey"] = [_market_to_key(mk, sym) for mk, sym in zip(mcol, base["symbol"].astype(str))]
        base = base[base["_mkey"].isin(only)].drop(columns=["_mkey"])

        if base.empty:
            print("⏭️ event_engine: 本 DB 無指定市場資料，跳過")
            return {"ok": True, "skipped": True, "reason": "no_target_markets"}

        # ====== per-symbol: limit flags + forward metrics ======
        out_list = []
        for sym, g in base.groupby("symbol", sort=False):
            g = g.sort_values("date").copy()

            # prev_close：只補缺失（不要把 processor 的 prev_close 全洗掉）
            g["prev_close"] = pd.to_numeric(g.get("prev_close", np.nan), errors="coerce")
            g["prev_close"] = g["prev_close"].fillna(pd.to_numeric(g["close"], errors="coerce").shift(1))

            # daily_change：只補缺失（per-symbol）
            g["daily_change"] = pd.to_numeric(g.get("daily_change", np.nan), errors="coerce")
            g["daily_change"] = g["daily_change"].fillna(pd.to_numeric(g["close"], errors="coerce").pct_change())

            # 精準判定：今日是否 hit close / intraday / fail
            g = _calc_limit_flags(g)

            # 若 processor 有 is_limit_up，就以 processor 為主；
            # 但仍保留 hit flags 供「衝漲停失敗」事件使用
            # ⚠️ 你原本會覆蓋 processor 的 is_limit_up（只要算得出 limit_up_price）
            # 這裡維持你原邏輯，但更安全：僅在 hit 欄位有效時覆蓋
            if "is_limit_up" in g.columns and g["is_limit_up"].notna().any():
                if "limit_up_price" in g.columns and g["limit_up_price"].notna().any():
                    g["is_limit_up"] = np.where(g["limit_up_price"].notna(), g["hit_limit_up_close"], g["is_limit_up"])
            else:
                g["is_limit_up"] = g.get("hit_limit_up_close", 0)

            g["is_limit_up"] = pd.to_numeric(g["is_limit_up"], errors="coerce").fillna(0).astype(int)

            # forward metrics
            g = _add_forward_metrics(g)

            # prev day flags
            g["prev_is_limit_up"] = g["is_limit_up"].shift(1).fillna(0).astype(int)
            g["prev_hit_intraday"] = g["hit_limit_up_intraday"].shift(1).fillna(0).astype(int)
            g["prev_limit_up_fail"] = g["limit_up_fail"].shift(1).fillna(0).astype(int)

            out_list.append(g)

        df = pd.concat(out_list, ignore_index=True)

        # =========================
        # A) limit_up_events
        # =========================
        lu = df[df["is_limit_up"] == 1].copy()

        keep_cols = [
            "symbol", "date", "market", "market_detail", "sector",
            "open", "high", "low", "close", "volume",
            "prev_close", "daily_change",
            "limit_up_price",
            "is_limit_up", "hit_limit_up_intraday", "limit_up_fail",
            "is_one_tick_lock", "lu_type", "consecutive_limits",
            "next_open_ret", "next_open_gap", "next_intraday_drawdown", "next_close_ret", "next_high_ret",
            "ret_1d", "ret_5d", "max_up_5d", "max_dd_5d",
        ]
        keep_cols = [c for c in keep_cols if c in lu.columns]
        lu_events = lu[keep_cols].copy()
        lu_events["date"] = pd.to_datetime(lu_events["date"]).dt.strftime("%Y-%m-%d")

        # =========================
        # B) daytrade_events（隔日沖研究）
        # =========================
        dt = df.copy()

        dt["prev_limit_up_today_not"] = ((dt["prev_is_limit_up"] == 1) & (dt["is_limit_up"] == 0)).astype(int)
        dt["prev_limit_up_today_fail"] = ((dt["prev_is_limit_up"] == 1) & (dt["limit_up_fail"] == 1)).astype(int)
        dt["today_limit_up_fail_no_prev"] = ((dt["prev_is_limit_up"] == 0) & (dt["limit_up_fail"] == 1)).astype(int)
        dt["today_limit_up_yes_no_prev"] = ((dt["prev_is_limit_up"] == 0) & (dt["is_limit_up"] == 1)).astype(int)

        dt_cols = [
            "symbol", "date", "market", "market_detail", "sector",
            "open", "high", "low", "close", "volume",
            "prev_close", "daily_change",
            "limit_up_price",
            "is_limit_up", "hit_limit_up_intraday", "limit_up_fail",
            "prev_is_limit_up", "prev_limit_up_today_not", "prev_limit_up_today_fail",
            "today_limit_up_fail_no_prev", "today_limit_up_yes_no_prev",
            "is_one_tick_lock", "lu_type", "consecutive_limits",
            "next_open_ret", "next_open_gap", "next_intraday_drawdown", "next_close_ret", "next_high_ret",
            "ret_1d", "ret_5d", "max_up_5d", "max_dd_5d",
        ]
        dt_cols = [c for c in dt_cols if c in dt.columns]
        daytrade_events = dt[dt_cols].copy()
        daytrade_events["date"] = pd.to_datetime(daytrade_events["date"]).dt.strftime("%Y-%m-%d")

        # =========================
        # 寫回 DB：replace tables
        # =========================
        conn.execute("DROP TABLE IF EXISTS limit_up_events")
        conn.execute("DROP TABLE IF EXISTS daytrade_events")

        lu_events.to_sql("limit_up_events", conn, if_exists="replace", index=False)
        daytrade_events.to_sql("daytrade_events", conn, if_exists="replace", index=False)

        # index（加速查詢）
        try:
            conn.execute("CREATE INDEX IF NOT EXISTS idx_lu_symbol_date ON limit_up_events(symbol, date)")
            conn.execute("CREATE INDEX IF NOT EXISTS idx_lu_market ON limit_up_events(market)")
            conn.execute("CREATE INDEX IF NOT EXISTS idx_lu_lu_type ON limit_up_events(lu_type)")
            conn.execute("CREATE INDEX IF NOT EXISTS idx_dt_symbol_date ON daytrade_events(symbol, date)")
            conn.execute("CREATE INDEX IF NOT EXISTS idx_dt_flags ON daytrade_events(prev_limit_up_today_not, today_limit_up_fail_no_prev)")
        except Exception:
            pass

        conn.commit()

        dt_sec = (datetime.now() - t0).total_seconds()
        print(
            f"✅ event_engine: 事件表已產生 | "
            f"limit_up_events={len(lu_events):,} | daytrade_events={len(daytrade_events):,} | {dt_sec:.1f}s"
        )

        return {
            "ok": True,
            "limit_up_events": int(len(lu_events)),
            "daytrade_events": int(len(daytrade_events)),
        }

    finally:
        conn.close()


# =============================================================================
# CLI
# =============================================================================
if __name__ == "__main__":
    # 範例：python event_engine.py tw_stock_warehouse.db
    import sys

    if len(sys.argv) >= 2:
        db = sys.argv[1]
    else:
        db = "tw_stock_warehouse.db"

    build_event_tables(db)
