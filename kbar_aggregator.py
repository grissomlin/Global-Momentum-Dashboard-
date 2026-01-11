# kbar_aggregator.py
# -*- coding: utf-8 -*-
"""
kbar_aggregator.py
------------------
從 SQLite 的 stock_prices 聚合出：
- kbar_weekly
- kbar_monthly
- kbar_yearly

設計重點：
- 只負責「聚合 + 寫回 DB」，不做 UI、不做事件研究。
- 在聚合前可選擇套用 data_cleaning.clean_pingpong_daily()，避免單根錯價污染週K/月K/年K。
- 週K以 W-FRI（週五收）為週期；若遇到市場不同週期需求，可改 build_weekly() 內的 freq。

需求（至少要存在）：
- stock_prices(symbol, date, open, high, low, close, volume)
可選：
- stock_info(symbol, market, market_detail, sector)  (目前聚合表不強依賴，但可擴充寫入 market/sector)

輸出表欄位（最小集合，供 kbar_contribution.py 使用）：
- kbar_weekly : symbol, year, week_id, period_start, period_end, open, high, low, close, volume
- kbar_monthly: symbol, year, month_id, period_start, period_end, open, high, low, close, volume
- kbar_yearly : symbol, year, period_start, period_end, open, high, low, close, volume, year_peak_date, year_peak_high
"""

from __future__ import annotations

import sqlite3
from typing import Optional, Tuple

import numpy as np
import pandas as pd

# 清洗模組（可用/不可用都不阻塞）
try:
    from data_cleaning import clean_pingpong_daily  # type: ignore
    CLEANING_AVAILABLE = True
except Exception:
    clean_pingpong_daily = None  # type: ignore
    CLEANING_AVAILABLE = False


# -----------------------------------------------------------------------------
# Helpers
# -----------------------------------------------------------------------------
def _connect(db_path: str) -> sqlite3.Connection:
    conn = sqlite3.connect(db_path)
    conn.execute("PRAGMA journal_mode=WAL;")
    conn.execute("PRAGMA synchronous=NORMAL;")
    conn.execute("PRAGMA temp_store=MEMORY;")
    conn.execute("PRAGMA cache_size=-200000;")  # ~200MB
    return conn


def _ensure_indices(conn: sqlite3.Connection) -> None:
    # stock_prices key
    conn.execute(
        "CREATE INDEX IF NOT EXISTS idx_stock_prices_symbol_date ON stock_prices(symbol, date);"
    )
    # kbar indices
    conn.execute(
        "CREATE INDEX IF NOT EXISTS idx_kbar_weekly_symbol_year_week ON kbar_weekly(symbol, year, week_id);"
    )
    conn.execute(
        "CREATE INDEX IF NOT EXISTS idx_kbar_monthly_symbol_year_month ON kbar_monthly(symbol, year, month_id);"
    )
    conn.execute(
        "CREATE INDEX IF NOT EXISTS idx_kbar_yearly_symbol_year ON kbar_yearly(symbol, year);"
    )
    conn.commit()


def _drop_non_trading_rows(df: pd.DataFrame) -> pd.DataFrame:
    """移除顯著非交易列：volume==0 且 OHLC 全相等（常見於停牌/抓價缺失補值）。"""
    if df.empty:
        return df
    cols = ["open", "high", "low", "close", "volume"]
    for c in cols:
        if c not in df.columns:
            return df
    mask_non_trade = (df["volume"].fillna(0) == 0) & (
        (df["open"] == df["high"]) & (df["high"] == df["low"]) & (df["low"] == df["close"])
    )
    return df.loc[~mask_non_trade].copy()


def _safe_first(s: pd.Series):
    s = s.dropna()
    return s.iloc[0] if len(s) else np.nan


def _safe_last(s: pd.Series):
    s = s.dropna()
    return s.iloc[-1] if len(s) else np.nan


# -----------------------------------------------------------------------------
# Build weekly/monthly/yearly
# -----------------------------------------------------------------------------
def build_weekly(df: pd.DataFrame, week_freq: str = "W-FRI") -> pd.DataFrame:
    """df: columns = [symbol,date,open,high,low,close,volume]"""
    if df.empty:
        return pd.DataFrame()

    df = df.sort_values(["symbol", "date"])
    out = []

    for sym, g in df.groupby("symbol", sort=False):
        g = g.sort_values("date").copy()
        g = _drop_non_trading_rows(g)

        # 可選：錯價清洗（不重算 prev_close；聚合只看 OHLCV）
        if CLEANING_AVAILABLE and clean_pingpong_daily is not None:
            try:
                g = clean_pingpong_daily(
                    g,
                    threshold=0.40,
                    abs_cap=0.80,
                    recompute_prev_close=False,
                    recompute_daily_change=False,
                )
            except Exception:
                pass

        if g.empty:
            continue

        g = g.set_index("date")

        agg = g.resample(week_freq).agg(
            open=("open", _safe_first),
            high=("high", "max"),
            low=("low", "min"),
            close=("close", _safe_last),
            volume=("volume", "sum"),
            period_start=("open", lambda x: x.index.min() if len(x) else pd.NaT),
            period_end=("open", lambda x: x.index.max() if len(x) else pd.NaT),
        )

        agg = agg.dropna(subset=["open", "close"]).reset_index(drop=True)
        if agg.empty:
            continue

        agg["symbol"] = sym
        agg["year"] = pd.to_datetime(agg["period_end"]).dt.year.astype("Int64")

        # week_id：同一年度內的週序（以 period_end 年為準），避免跨年週混淆
        agg = agg.sort_values(["year", "period_end"])
        agg["week_seq_in_year"] = agg.groupby("year").cumcount() + 1
        agg["week_id"] = (agg["year"].astype(int) * 100 + agg["week_seq_in_year"]).astype("Int64")
        agg = agg.drop(columns=["week_seq_in_year"])

        # date strings
        agg["period_start"] = pd.to_datetime(agg["period_start"]).dt.strftime("%Y-%m-%d")
        agg["period_end"] = pd.to_datetime(agg["period_end"]).dt.strftime("%Y-%m-%d")

        out.append(agg)

    if not out:
        return pd.DataFrame()

    wk = pd.concat(out, ignore_index=True)
    return wk[
        ["symbol", "year", "week_id", "period_start", "period_end", "open", "high", "low", "close", "volume"]
    ]


def build_monthly(df: pd.DataFrame) -> pd.DataFrame:
    if df.empty:
        return pd.DataFrame()

    df = df.sort_values(["symbol", "date"])
    out = []

    for sym, g in df.groupby("symbol", sort=False):
        g = g.sort_values("date").copy()
        g = _drop_non_trading_rows(g)

        if CLEANING_AVAILABLE and clean_pingpong_daily is not None:
            try:
                g = clean_pingpong_daily(
                    g,
                    threshold=0.40,
                    abs_cap=0.80,
                    recompute_prev_close=False,
                    recompute_daily_change=False,
                )
            except Exception:
                pass

        if g.empty:
            continue

        g = g.set_index("date")

        agg = g.resample("M").agg(
            open=("open", _safe_first),
            high=("high", "max"),
            low=("low", "min"),
            close=("close", _safe_last),
            volume=("volume", "sum"),
            period_start=("open", lambda x: x.index.min() if len(x) else pd.NaT),
            period_end=("open", lambda x: x.index.max() if len(x) else pd.NaT),
        )
        agg = agg.dropna(subset=["open", "close"]).reset_index(drop=True)
        if agg.empty:
            continue

        agg["symbol"] = sym
        pe = pd.to_datetime(agg["period_end"])
        agg["year"] = pe.dt.year.astype("Int64")
        agg["month_id"] = (pe.dt.year * 100 + pe.dt.month).astype("Int64")

        agg["period_start"] = pd.to_datetime(agg["period_start"]).dt.strftime("%Y-%m-%d")
        agg["period_end"] = pe.dt.strftime("%Y-%m-%d")

        out.append(agg)

    if not out:
        return pd.DataFrame()

    mk = pd.concat(out, ignore_index=True)
    return mk[
        ["symbol", "year", "month_id", "period_start", "period_end", "open", "high", "low", "close", "volume"]
    ]


def build_yearly(df: pd.DataFrame) -> pd.DataFrame:
    if df.empty:
        return pd.DataFrame()

    df = df.sort_values(["symbol", "date"])
    out = []

    for sym, g in df.groupby("symbol", sort=False):
        g = g.sort_values("date").copy()
        g = _drop_non_trading_rows(g)

        if CLEANING_AVAILABLE and clean_pingpong_daily is not None:
            try:
                g = clean_pingpong_daily(
                    g,
                    threshold=0.40,
                    abs_cap=0.80,
                    recompute_prev_close=False,
                    recompute_daily_change=False,
                )
            except Exception:
                pass

        if g.empty:
            continue

        g["year"] = g["date"].dt.year.astype("Int64")

        for y, gy in g.groupby("year", sort=False):
            if gy.empty:
                continue
            gy = gy.sort_values("date")

            period_start = gy["date"].iloc[0]
            period_end = gy["date"].iloc[-1]

            y_open = _safe_first(gy["open"])
            y_close = _safe_last(gy["close"])
            y_high = gy["high"].max()
            y_low = gy["low"].min()
            y_vol = gy["volume"].sum()

            # 年內 peak：以 daily high 的最大值為準
            idx_peak = gy["high"].idxmax()
            peak_date = gy.loc[idx_peak, "date"] if pd.notna(idx_peak) else period_end
            peak_high = float(gy.loc[idx_peak, "high"]) if pd.notna(idx_peak) else float(y_high)

            out.append(
                {
                    "symbol": sym,
                    "year": int(y),
                    "period_start": pd.to_datetime(period_start).strftime("%Y-%m-%d"),
                    "period_end": pd.to_datetime(period_end).strftime("%Y-%m-%d"),
                    "open": float(y_open) if pd.notna(y_open) else np.nan,
                    "high": float(y_high) if pd.notna(y_high) else np.nan,
                    "low": float(y_low) if pd.notna(y_low) else np.nan,
                    "close": float(y_close) if pd.notna(y_close) else np.nan,
                    "volume": float(y_vol) if pd.notna(y_vol) else np.nan,
                    "year_peak_date": pd.to_datetime(peak_date).strftime("%Y-%m-%d"),
                    "year_peak_high": peak_high,
                }
            )

    if not out:
        return pd.DataFrame()

    yk = pd.DataFrame(out)
    return yk[
        ["symbol", "year", "period_start", "period_end", "open", "high", "low", "close", "volume", "year_peak_date", "year_peak_high"]
    ]


# -----------------------------------------------------------------------------
# Public API
# -----------------------------------------------------------------------------
def build_kbar_tables(
    db_path: str,
    date_min: Optional[str] = None,
    date_max: Optional[str] = None,
    week_freq: str = "W-FRI",
    verbose: bool = True,
) -> Tuple[int, int, int]:
    """Build and write kbar_weekly / kbar_monthly / kbar_yearly.

    Returns:
        (n_weekly_rows, n_monthly_rows, n_yearly_rows)
    """
    conn = _connect(db_path)

    where = []
    params = []
    if date_min:
        where.append("date >= ?")
        params.append(date_min)
    if date_max:
        where.append("date <= ?")
        params.append(date_max)
    where_sql = ("WHERE " + " AND ".join(where)) if where else ""

    df = pd.read_sql(
        f"""
        SELECT symbol, date, open, high, low, close, volume
        FROM stock_prices
        {where_sql}
        """,
        conn,
        params=params,
        parse_dates=["date"],
    )

    if df.empty:
        if verbose:
            print("❌ stock_prices 無資料，無法聚合 K 線")
        conn.close()
        return (0, 0, 0)

    # 基本清理：確保數值型
    for c in ["open", "high", "low", "close", "volume"]:
        df[c] = pd.to_numeric(df[c], errors="coerce")

    wk = build_weekly(df, week_freq=week_freq)
    mk = build_monthly(df)
    yk = build_yearly(df)

    # write back
    wk.to_sql("kbar_weekly", conn, if_exists="replace", index=False)
    mk.to_sql("kbar_monthly", conn, if_exists="replace", index=False)
    yk.to_sql("kbar_yearly", conn, if_exists="replace", index=False)

    _ensure_indices(conn)

    if verbose:
        print(f"✅ kbar_weekly rows: {len(wk):,}")
        print(f"✅ kbar_monthly rows: {len(mk):,}")
        print(f"✅ kbar_yearly rows: {len(yk):,}")
        print(f"🧼 data_cleaning available: {CLEANING_AVAILABLE}")

    conn.close()
    return (len(wk), len(mk), len(yk))
