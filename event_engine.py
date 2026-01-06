# -*- coding: utf-8 -*-
"""
event_engine.py
---------------
獨立事件表：專門給「漲停型態」與「隔日沖/衝漲停」研究使用（乾淨、可擴充）

輸入：同一個 db (stock_prices + stock_info)
輸出：兩張表
1) limitup_events：每一筆「當日漲停(或 pseudo-limit)事件」+ 型態 + 未來報酬
2) daytrade_events：更廣義：昨日漲停/今日漲停/今日衝漲停失敗 等事件標記 + 未來報酬

依賴：market_rules.py
"""

from __future__ import annotations
import sqlite3
from dataclasses import dataclass
from typing import Optional, Dict, Any, Tuple, List
from datetime import datetime

import numpy as np
import pandas as pd

from market_rules import MarketConfig


# -----------------------
# Helpers
# -----------------------
def log(msg: str):
    print(f"{pd.Timestamp.now():%H:%M:%S}: {msg}", flush=True)


def ensure_tables(conn: sqlite3.Connection):
    conn.execute("""
        CREATE TABLE IF NOT EXISTS limitup_events (
            symbol TEXT,
            date TEXT,
            market TEXT,
            market_detail TEXT,
            name TEXT,
            sector TEXT,

            prev_close REAL,
            open REAL,
            high REAL,
            low REAL,
            close REAL,
            volume INTEGER,

            daily_change REAL,
            daily_change_pct REAL,

            limit_up_price REAL,
            is_limit_up INTEGER,
            hit_limit INTEGER,
            is_limit_down INTEGER,

            vol_ma5 REAL,
            vol_ratio_ma5 REAL,

            lu_type_raw TEXT,
            lu_type_4 TEXT,

            next1d_ret_close REAL,
            next1d_ret_high REAL,
            fwd_max_up_1_5d REAL,
            fwd_max_down_1_5d REAL,

            created_at TEXT,
            PRIMARY KEY (symbol, date)
        )
    """)

    conn.execute("""
        CREATE TABLE IF NOT EXISTS daytrade_events (
            symbol TEXT,
            date TEXT,
            market TEXT,
            market_detail TEXT,
            name TEXT,
            sector TEXT,

            prev_close REAL,
            open REAL,
            high REAL,
            low REAL,
            close REAL,
            volume INTEGER,

            daily_change REAL,
            daily_change_pct REAL,

            limit_up_price REAL,
            is_limit_up INTEGER,
            hit_limit INTEGER,

            prev_is_limit_up INTEGER,
            prev_hit_limit INTEGER,

            y_limit_today_not_limit INTEGER,
            y_not_limit_today_fail_limit INTEGER,
            y_limit_today_gapdown INTEGER,
            y_limit_today_red INTEGER,

            next1d_ret_close REAL,
            next1d_ret_high REAL,
            next5d_ret_close REAL,
            fwd_max_up_1_5d REAL,
            fwd_max_down_1_5d REAL,

            created_at TEXT,
            PRIMARY KEY (symbol, date)
        )
    """)

    conn.execute("CREATE INDEX IF NOT EXISTS idx_limitup_events_market ON limitup_events(market)")
    conn.execute("CREATE INDEX IF NOT EXISTS idx_daytrade_events_market ON daytrade_events(market)")
    conn.execute("CREATE INDEX IF NOT EXISTS idx_daytrade_events_flags ON daytrade_events(prev_is_limit_up, is_limit_up, hit_limit)")
    conn.commit()


def load_price_data(conn: sqlite3.Connection) -> pd.DataFrame:
    """
    從 stock_prices + stock_info 拉出完整日K
    DB schema 以你現在 processor.py 相容版為準：
    - stock_prices: symbol,date,open,high,low,close,volume
    - stock_info  : symbol,name,sector,market,market_detail
    """
    q = """
    SELECT
        p.symbol, p.date, p.open, p.high, p.low, p.close, p.volume,
        i.name, i.sector, i.market, i.market_detail
    FROM stock_prices p
    LEFT JOIN stock_info i ON p.symbol = i.symbol
    """
    df = pd.read_sql(q, conn)
    if df.empty:
        return df
    df["date"] = pd.to_datetime(df["date"])
    df = df.sort_values(["symbol", "date"]).reset_index(drop=True)
    return df


def calc_forward_metrics(group: pd.DataFrame) -> pd.DataFrame:
    """
    針對單一 symbol：算隔日/5日/1~5日最大上漲下跌（用 close / high / low）
    """
    g = group.copy()
    g["close_next1"] = g["close"].shift(-1)
    g["high_next1"] = g["high"].shift(-1)
    g["close_next5"] = g["close"].shift(-5)

    g["next1d_ret_close"] = (g["close_next1"] / g["close"] - 1.0) * 100
    g["next1d_ret_high"] = (g["high_next1"] / g["close"] - 1.0) * 100
    g["next5d_ret_close"] = (g["close_next5"] / g["close"] - 1.0) * 100

    # 1~5 日最大上漲：max(high[t+1..t+5]) / close[t] - 1
    # 1~5 日最大下跌：min(low[t+1..t+5]) / close[t] - 1
    fwd_high_1_5 = []
    fwd_low_1_5 = []
    highs = g["high"].to_numpy()
    lows = g["low"].to_numpy()
    closes = g["close"].to_numpy()
    n = len(g)

    for i in range(n):
        j1 = i + 1
        j2 = min(i + 5, n - 1)
        if j1 > n - 1:
            fwd_high_1_5.append(np.nan)
            fwd_low_1_5.append(np.nan)
            continue
        mx = np.nanmax(highs[j1 : j2 + 1])
        mn = np.nanmin(lows[j1 : j2 + 1])
        base = closes[i]
        if base and base > 0:
            fwd_high_1_5.append((mx / base - 1.0) * 100)
            fwd_low_1_5.append((mn / base - 1.0) * 100)
        else:
            fwd_high_1_5.append(np.nan)
            fwd_low_1_5.append(np.nan)

    g["fwd_max_up_1_5d"] = fwd_high_1_5
    g["fwd_max_down_1_5d"] = fwd_low_1_5
    return g


def classify_limitup_type(row: pd.Series) -> Tuple[str, str]:
    """
    依你文章的規則：先做 7 類 raw，再合併成 4 大類（+OTHER）
    規則（以日K近似）：
    1) is_gap : (open/prev_close - 1) >= 0.07
    2) is_high_vol: volume/vol_ma5 >= 3
    3) is_low_vol : volume/vol_ma5 <= 0.4
    4) is_float   : (not is_gap) and (close/open - 1 >= 0.05)
    5) 優先序：
       GAP_UP_LOCK, GAP_UP, FLOAT_HV, FLOAT, LOW_VOL_LOCK, HIGH_VOL_LOCK, OTHER
    最後合併：
       FLOATING / GAP_UP / HIGH_VOLUME_LOCK / NO_VOLUME_LOCK / OTHER
    """
    prev_close = row.get("prev_close")
    o = row.get("open")
    c = row.get("close")
    vol = row.get("volume")
    vma5 = row.get("vol_ma5")

    if prev_close is None or not (prev_close > 0) or o is None or c is None or vol is None or vma5 is None or vma5 == 0:
        return "OTHER", "OTHER"

    gap = (o / prev_close - 1.0) >= 0.07
    vol_ratio = (vol / vma5) if vma5 else np.nan
    high_vol = (vol_ratio >= 3.0) if np.isfinite(vol_ratio) else False
    low_vol = (vol_ratio <= 0.4) if np.isfinite(vol_ratio) else False
    is_float = (not gap) and ((c / o - 1.0) >= 0.05) if o > 0 else False

    # raw 7-class
    if gap and low_vol:
        raw = "GAP_UP_LOCK"
    elif gap:
        raw = "GAP_UP"
    elif is_float and high_vol:
        raw = "FLOAT_HV"
    elif is_float:
        raw = "FLOAT"
    elif low_vol:
        raw = "LOW_VOL_LOCK"
    elif high_vol:
        raw = "HIGH_VOL_LOCK"
    else:
        raw = "OTHER"

    # merge to 4 (+OTHER)
    if raw in ("FLOAT", "FLOAT_HV"):
        merged = "FLOATING"
    elif raw in ("GAP_UP", "GAP_UP_LOCK"):
        merged = "GAP_UP"
    elif raw == "HIGH_VOL_LOCK":
        merged = "HIGH_VOLUME_LOCK"
    elif raw == "LOW_VOL_LOCK":
        merged = "NO_VOLUME_LOCK"
    else:
        merged = "OTHER"

    return raw, merged


def build_events(db_path: str):
    conn = sqlite3.connect(db_path, timeout=120)
    try:
        ensure_tables(conn)
        df = load_price_data(conn)
        if df.empty:
            log("❌ stock_prices 為空，無法建立事件表")
            return

        # 基礎欄位（逐股計算）
        df["prev_close"] = df.groupby("symbol")["close"].shift(1)
        df["daily_change"] = df.groupby("symbol")["close"].pct_change()
        df["daily_change_pct"] = df["daily_change"] * 100

        # vol_ma5 / ratio
        df["vol_ma5"] = df.groupby("symbol")["volume"].transform(lambda s: s.rolling(5, min_periods=1).mean())
        df["vol_ratio_ma5"] = df["volume"] / df["vol_ma5"]

        # forward metrics
        df = df.groupby("symbol", group_keys=False).apply(calc_forward_metrics)

        # 計算漲停 / 跌停（依市場規則）
        limit_up_prices = []
        is_limit_ups = []
        hit_limits = []
        is_limit_downs = []

        # 先把市場資訊缺失補 default
        df["market"] = df["market"].fillna("")
        df["market_detail"] = df["market_detail"].fillna("unknown")

        for r in df.itertuples(index=False):
            symbol = r.symbol
            market = r.market
            market_detail = r.market_detail
            prev_close = r.prev_close
            close = r.close
            high = r.high

            rule = MarketConfig.get_rule(market, market_detail, symbol=symbol)
            up, dn = MarketConfig.calc_limit_price(prev_close, rule)

            limit_up_prices.append(up)
            is_lu = MarketConfig.is_limit_up(close, prev_close, rule) if (prev_close is not None and close is not None) else 0
            is_ld = MarketConfig.is_limit_down(close, prev_close, rule) if (prev_close is not None and close is not None) else 0

            # hit_limit：盤中有碰到（用 high 判斷）
            hit = 0
            if up is not None and high is not None:
                hit = int(float(high) >= float(up) * 0.999)

            is_limit_ups.append(is_lu)
            is_limit_downs.append(is_ld)
            hit_limits.append(hit)

        df["limit_up_price"] = limit_up_prices
        df["is_limit_up"] = is_limit_ups
        df["hit_limit"] = hit_limits
        df["is_limit_down"] = is_limit_downs

        # 型態分類
        raw_types = []
        merged_types = []
        for _, row in df.iterrows():
            raw, merged = classify_limitup_type(row)
            raw_types.append(raw)
            merged_types.append(merged)
        df["lu_type_raw"] = raw_types
        df["lu_type_4"] = merged_types

        # 昨日資訊（隔日沖研究用）
        df["prev_is_limit_up"] = df.groupby("symbol")["is_limit_up"].shift(1).fillna(0).astype(int)
        df["prev_hit_limit"] = df.groupby("symbol")["hit_limit"].shift(1).fillna(0).astype(int)

        # 隔日沖常用事件旗標
        # 1) 昨天漲停，今天沒漲停
        df["y_limit_today_not_limit"] = ((df["prev_is_limit_up"] == 1) & (df["is_limit_up"] == 0)).astype(int)

        # 2) 昨天沒漲停，但今天「衝漲停失敗」（盤中碰到上限，但收盤沒鎖）
        df["y_not_limit_today_fail_limit"] = ((df["prev_is_limit_up"] == 0) & (df["hit_limit"] == 1) & (df["is_limit_up"] == 0)).astype(int)

        # 3) 昨天漲停，今天開低（常見隔日沖反向壓力）
        df["y_limit_today_gapdown"] = ((df["prev_is_limit_up"] == 1) & (df["open"] < df["prev_close"])).astype(int)

        # 4) 昨天漲停，今天收黑（隔日沖出貨可能）
        df["y_limit_today_red"] = ((df["prev_is_limit_up"] == 1) & (df["close"] < df["open"])).astype(int)

        # -----------------------
        # 寫入兩張事件表
        # -----------------------
        now = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        df_out = df.copy()
        df_out["date"] = df_out["date"].dt.strftime("%Y-%m-%d")
        df_out["created_at"] = now

        # limitup_events：只保留「當日漲停(或 pseudo-limit)」事件
        # 你要 US/HK 用 10% 當漲停 → 這裡 is_limit_up 已含 pseudo
        limitup_df = df_out[df_out["is_limit_up"] == 1].copy()

        limitup_keep = [
            "symbol","date","market","market_detail","name","sector",
            "prev_close","open","high","low","close","volume",
            "daily_change","daily_change_pct",
            "limit_up_price","is_limit_up","hit_limit","is_limit_down",
            "vol_ma5","vol_ratio_ma5",
            "lu_type_raw","lu_type_4",
            "next1d_ret_close","next1d_ret_high","fwd_max_up_1_5d","fwd_max_down_1_5d",
            "created_at"
        ]
        limitup_df = limitup_df[limitup_keep]

        # daytrade_events：保留「所有交易日」但帶事件旗標（隔日沖研究最乾淨）
        daytrade_keep = [
            "symbol","date","market","market_detail","name","sector",
            "prev_close","open","high","low","close","volume",
            "daily_change","daily_change_pct",
            "limit_up_price","is_limit_up","hit_limit",
            "prev_is_limit_up","prev_hit_limit",
            "y_limit_today_not_limit",
            "y_not_limit_today_fail_limit",
            "y_limit_today_gapdown",
            "y_limit_today_red",
            "next1d_ret_close","next1d_ret_high","next5d_ret_close",
            "fwd_max_up_1_5d","fwd_max_down_1_5d",
            "created_at"
        ]
        daytrade_df = df_out[daytrade_keep].copy()

        # 清掉舊資料（可改成增量 upsert，但先用 replace 最直覺）
        log("🧹 重新建立事件表（DROP + REPLACE）...")
        conn.execute("DROP TABLE IF EXISTS limitup_events")
        conn.execute("DROP TABLE IF EXISTS daytrade_events")
        conn.commit()
        ensure_tables(conn)

        log(f"✍️ 寫入 limitup_events: {len(limitup_df):,} 筆")
        limitup_df.to_sql("limitup_events", conn, if_exists="append", index=False)

        log(f"✍️ 寫入 daytrade_events: {len(daytrade_df):,} 筆")
        daytrade_df.to_sql("daytrade_events", conn, if_exists="append", index=False)

        log("🧹 VACUUM...")
        conn.execute("VACUUM")
        conn.commit()

        # 簡單摘要
        log("✅ 完成 event_engine 建表")
        log(f"   - limitup_events: {len(limitup_df):,}")
        log(f"   - daytrade_events: {len(daytrade_df):,}")

    finally:
        conn.close()


if __name__ == "__main__":
    # 例：build_events("tw_stock_warehouse.db")
    build_events("tw_stock_warehouse.db")
