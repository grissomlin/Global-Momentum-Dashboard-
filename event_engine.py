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
"""

import sqlite3
import pandas as pd
import numpy as np
from datetime import datetime
import warnings

warnings.filterwarnings("ignore", category=FutureWarning)

# -----------------------------
# market_rules: 精準漲停判定（建議你一定要有）
# -----------------------------
try:
    import market_rules
    HAS_MARKET_RULES = True
except Exception:
    market_rules = None
    HAS_MARKET_RULES = False


# =============================================================================
# 0) 小工具：判斷是否 TW/CN/JP
# =============================================================================
def _is_target_market(market: str, symbol: str) -> bool:
    m = (market or "").upper().strip()
    sym = (symbol or "").upper().strip()

    if m in ("TW", "TSE", "GTSM"):
        return True
    if m in ("SSE", "SZSE", "CN", "CHINA"):
        return True
    if m in ("JP", "JPX", "TSEJ"):  # 你可以依你的 downloader_jp 寫入的 market 值調整
        return True

    # fallback by suffix
    if sym.endswith(".TW") or sym.endswith(".TWO"):
        return True
    if sym.endswith(".SS") or sym.endswith(".SZ"):
        return True
    if sym.endswith(".T"):
        return True

    return False


# =============================================================================
# 1) Fallback 規則（只在 market_rules 不存在/不可用時）
# =============================================================================
def _fallback_cn_limit_pct(symbol: str) -> float:
    sym = (symbol or "").upper()
    code = "".join([c for c in sym if c.isdigit()])
    if code.startswith(("300", "301", "688")):
        return 0.20
    return 0.10


def _fallback_calc_limit_up_price(prev_close: pd.Series, market: str, symbol: str) -> pd.Series | None:
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
    symbol = str(df_sym["symbol"].iloc[0])
    market = str(df_sym["market"].iloc[0]) if "market" in df_sym.columns else ""

    prev_close = df_sym["prev_close"].astype(float)

    limit_price = None
    if HAS_MARKET_RULES and hasattr(market_rules, "calc_limit_up_price"):
        try:
            limit_price = market_rules.calc_limit_up_price(
                prev_close=prev_close,
                market=market,
                market_detail=str(df_sym["market_detail"].iloc[0]) if "market_detail" in df_sym.columns else "",
                symbol=symbol,
            )
        except Exception:
            limit_price = None

    if limit_price is None:
        limit_price = _fallback_calc_limit_up_price(prev_close, market, symbol)

    df_sym["limit_up_price"] = limit_price.astype(float) if limit_price is not None else np.nan

    # buffer：若有 tick_size 用 tick*0.5 當容忍；否則用 0
    if limit_price is not None and HAS_MARKET_RULES and hasattr(market_rules, "tick_size"):
        try:
            ticks = prev_close.apply(lambda x: market_rules.tick_size(float(x), market=market, symbol=symbol))
            buffer = ticks.fillna(0) * 0.5
        except Exception:
            buffer = 0.0
    else:
        buffer = 0.0

    if limit_price is None:
        df_sym["hit_limit_up_close"] = 0
        df_sym["hit_limit_up_intraday"] = 0
        df_sym["limit_up_fail"] = 0
        return df_sym

    lim = df_sym["limit_up_price"].astype(float)
    hi = df_sym["high"].astype(float)
    cl = df_sym["close"].astype(float)

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
        df_sym[col] = pd.to_numeric(df_sym[col], errors="coerce")

    df_sym["next_open"] = df_sym["open"].shift(-1)
    df_sym["next_high"] = df_sym["high"].shift(-1)
    df_sym["next_low"] = df_sym["low"].shift(-1)
    df_sym["next_close"] = df_sym["close"].shift(-1)

    # next day
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

    # max up / max dd in next 5 days (using high/low)
    # window: t+1..t+5
    highs_fwd = pd.concat([df_sym["high"].shift(-i) for i in range(1, 6)], axis=1)
    lows_fwd = pd.concat([df_sym["low"].shift(-i) for i in range(1, 6)], axis=1)

    df_sym["max_up_5d"] = (highs_fwd.max(axis=1) / df_sym["close"] - 1)
    df_sym["max_dd_5d"] = (lows_fwd.min(axis=1) / df_sym["close"] - 1)

    return df_sym


# =============================================================================
# 5) 生成事件表
# =============================================================================
def build_event_tables(db_path: str, only_markets=("tw", "cn", "jp")) -> dict:
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
        if base.empty:
            print("❌ event_engine: 找不到可用資料（stock_analysis/stock_prices 皆空）")
            return {"ok": False, "reason": "empty"}

        # 統一欄位
        base["date"] = pd.to_datetime(base["date"], errors="coerce")
        base = base.dropna(subset=["date"])
        base = base.sort_values(["symbol", "date"]).reset_index(drop=True)

        # sector 欄位名可能不同（stock_analysis join 的 info_sector）
        if "sector" not in base.columns and "info_sector" in base.columns:
            base["sector"] = base["info_sector"]

        # prev_close 若不存在就補
        if "prev_close" not in base.columns:
            base["prev_close"] = pd.to_numeric(base["close"], errors="coerce").shift(1)

        # daily_change 若不存在就補
        if "daily_change" not in base.columns:
            base["daily_change"] = pd.to_numeric(base["close"], errors="coerce").pct_change()

        # is_limit_up 若不存在：先全部 0，後面會用精準判定回填
        if "is_limit_up" not in base.columns:
            base["is_limit_up"] = 0

        # lu_type / consecutive_limits / is_one_tick_lock 若不存在：先補空（事件仍可跑）
        if "lu_type" not in base.columns:
            base["lu_type"] = None
        if "consecutive_limits" not in base.columns:
            base["consecutive_limits"] = 0
        if "is_one_tick_lock" not in base.columns:
            base["is_one_tick_lock"] = (
                (base["open"] == base["close"]) &
                (base["high"] == base["low"]) &
                (base["high"] == base["close"])
            ).astype(int)

        # 只處理 tw/cn/jp（其他市場直接跳過）
        base["_target_market"] = base.apply(
            lambda r: _is_target_market(str(r.get("market", "")), str(r.get("symbol", ""))),
            axis=1,
        )
        base = base[base["_target_market"] == True].drop(columns=["_target_market"])
        if base.empty:
            print("⏭️ event_engine: 本 DB 無 TW/CN/JP 資料，跳過")
            return {"ok": True, "skipped": True, "reason": "no_target_markets"}

        # per-symbol: limit flags + forward metrics
        out_list = []
        for sym, g in base.groupby("symbol", sort=False):
            g = g.sort_values("date").copy()

            # prev_close 確保正確（每檔內）
            g["prev_close"] = pd.to_numeric(g["close"], errors="coerce").shift(1)

            # 精準判定：今日是否 hit close / intraday / fail
            g = _calc_limit_flags(g)

            # 若 processor 有 is_limit_up，就以 processor 為主；否則用 hit_limit_up_close
            # 但你要做事件「衝漲停失敗」需要 limit_up_fail，所以仍保留 hit flags
            if "is_limit_up" in g.columns and g["is_limit_up"].notna().any():
                # 有時 processor 對 JP 可能先沒有精準，這裡用 market_rules 精準結果覆蓋（只覆蓋 target）
                # 你若不想覆蓋，把下面兩行註解掉
                g["is_limit_up"] = np.where(g["limit_up_price"].notna(), g["hit_limit_up_close"], g["is_limit_up"])
            else:
                g["is_limit_up"] = g["hit_limit_up_close"]

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

        # 事件日基本欄位
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

        # 格式化 date（SQLite）
        lu_events["date"] = pd.to_datetime(lu_events["date"]).dt.strftime("%Y-%m-%d")

        # =========================
        # B) daytrade_events（隔日沖研究）
        # =========================
        dt = df.copy()

        # (1) 昨天漲停 今天沒漲停（隔日沖最常見的釋放）
        dt["prev_limit_up_today_not"] = ((dt["prev_is_limit_up"] == 1) & (dt["is_limit_up"] == 0)).astype(int)

        # (2) 昨天漲停 今天盤中再摸到漲停但收不住（衝高回落）
        dt["prev_limit_up_today_fail"] = ((dt["prev_is_limit_up"] == 1) & (dt["limit_up_fail"] == 1)).astype(int)

        # (3) 昨天沒漲停 今天衝漲停失敗
        dt["today_limit_up_fail_no_prev"] = ((dt["prev_is_limit_up"] == 0) & (dt["limit_up_fail"] == 1)).astype(int)

        # (4) 昨天沒漲停 今天收漲停（首板）
        dt["today_limit_up_yes_no_prev"] = ((dt["prev_is_limit_up"] == 0) & (dt["is_limit_up"] == 1)).astype(int)

        # 你要的「昨天漲停今天沒漲」 +「昨天沒漲停今天衝漲停失敗」都在上面

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
        print(f"✅ event_engine: 事件表已產生 | limit_up_events={len(lu_events):,} | daytrade_events={len(daytrade_events):,} | {dt_sec:.1f}s")

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
