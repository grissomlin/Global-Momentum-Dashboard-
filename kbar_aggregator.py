# -*- coding: utf-8 -*-
"""
kbar_aggregator.py
------------------
把日K stock_prices 聚合成：
- kbar_weekly  (周K)
- kbar_monthly (月K)
- kbar_yearly  (年K + 年內最高點 peak_date/peak_high)

✅ 只依賴 SQLite + pandas
✅ 不改動原始 stock_prices
✅ 對齊儀表板需求：週/月K 可對到年K的 peak_date（後續 event_engine 做貢獻度更乾淨）

DB 依賴表：
- stock_prices(symbol,date,open,high,low,close,volume)
- stock_info(symbol, market, market_detail, ...)  (可無；沒有也能跑)

產出表：
- kbar_weekly
- kbar_monthly
- kbar_yearly

使用方式：
1) 在 main.py 下載完成後呼叫：
   from kbar_aggregator import build_kbars
   build_kbars(db_file)

2) CLI：
   python kbar_aggregator.py tw_stock_warehouse.db
"""

import os
import sqlite3
import warnings
from dataclasses import dataclass
from typing import Optional, Tuple

import numpy as np
import pandas as pd

warnings.filterwarnings("ignore", category=FutureWarning)


# --------------------------
# Utilities
# --------------------------
def log(msg: str):
    print(f"{pd.Timestamp.now():%H:%M:%S} | {msg}", flush=True)


def _safe_to_datetime(s: pd.Series) -> pd.Series:
    dt = pd.to_datetime(s, errors="coerce")
    # yfinance 可能有 tz，統一去掉
    try:
        return dt.dt.tz_localize(None)
    except Exception:
        return dt


def _ensure_numeric(df: pd.DataFrame, cols):
    for c in cols:
        if c in df.columns:
            df[c] = pd.to_numeric(df[c], errors="coerce")
    return df


def _read_stock_prices(conn: sqlite3.Connection) -> pd.DataFrame:
    df = pd.read_sql(
        """
        SELECT symbol, date, open, high, low, close, volume
        FROM stock_prices
        """,
        conn,
    )
    if df.empty:
        return df

    df["date"] = _safe_to_datetime(df["date"])
    df = df.dropna(subset=["symbol", "date"]).copy()
    df = _ensure_numeric(df, ["open", "high", "low", "close", "volume"])
    df["symbol"] = df["symbol"].astype(str)
    df = df.sort_values(["symbol", "date"])
    return df


def _read_stock_info(conn: sqlite3.Connection) -> pd.DataFrame:
    # 某些 DB 可能沒有 stock_info 或沒有 market_detail
    try:
        df = pd.read_sql("SELECT * FROM stock_info", conn)
        if df.empty:
            return df
        df["symbol"] = df["symbol"].astype(str)
        for col in ["market", "market_detail", "sector", "name"]:
            if col not in df.columns:
                df[col] = None
        return df[["symbol", "market", "market_detail", "sector", "name"]].copy()
    except Exception:
        return pd.DataFrame(columns=["symbol", "market", "market_detail", "sector", "name"])


def _attach_info(df: pd.DataFrame, info: pd.DataFrame) -> pd.DataFrame:
    if df.empty or info.empty:
        # 確保欄位存在
        for col in ["market", "market_detail", "sector", "name"]:
            if col not in df.columns:
                df[col] = None
        return df
    return df.merge(info, on="symbol", how="left")


def _ohlcv_agg(group: pd.DataFrame) -> pd.Series:
    """對某個 period 的日K做 OHLCV 聚合（假設 group 已按 date 排序）"""
    open_ = group["open"].iloc[0]
    close_ = group["close"].iloc[-1]
    high_ = group["high"].max()
    low_ = group["low"].min()
    vol_ = group["volume"].sum(min_count=1)

    return pd.Series(
        {
            "open": open_,
            "high": high_,
            "low": low_,
            "close": close_,
            "volume": vol_,
            "start_date": group["date"].iloc[0],
            "end_date": group["date"].iloc[-1],
            "n_bars": int(len(group)),
        }
    )


def _add_prev_ret(df: pd.DataFrame, key_cols: Tuple[str, ...], close_col="close") -> pd.DataFrame:
    """加上 prev_close / ret / logret（按 key_cols 的第一欄通常是 symbol 分組）"""
    df = df.sort_values(list(key_cols)).copy()
    sym_col = key_cols[0]
    df["prev_close"] = df.groupby(sym_col)[close_col].shift(1)
    df["ret"] = (df[close_col] / df["prev_close"]) - 1
    df["logret"] = np.log(df[close_col] / df["prev_close"])
    df.loc[df["prev_close"].isna(), ["ret", "logret"]] = np.nan
    return df


# --------------------------
# Core builder
# --------------------------
@dataclass
class KbarBuildResult:
    weekly_rows: int
    monthly_rows: int
    yearly_rows: int


def build_kbars(db_path: str) -> KbarBuildResult:
    """
    讀取 stock_prices -> 產生 kbar_weekly / kbar_monthly / kbar_yearly
    """
    if not os.path.exists(db_path):
        raise FileNotFoundError(f"DB not found: {db_path}")

    conn = sqlite3.connect(db_path, timeout=120)
    try:
        log(f"📥 讀取日K: {db_path}")
        df = _read_stock_prices(conn)
        if df.empty:
            log("❌ stock_prices 為空，跳過 kbar 聚合")
            return KbarBuildResult(0, 0, 0)

        info = _read_stock_info(conn)
        df = _attach_info(df, info)

        # ==========================
        # Weekly
        # ==========================
        log("🧱 建立 kbar_weekly ...")
        d = df.copy()
        # 以「週一」作為 week_start（常見定義；你儀表板也容易對齊）
        d["week_start"] = d["date"].dt.to_period("W-MON").apply(lambda p: p.start_time)
        d = d.sort_values(["symbol", "date"])

        wk = (
            d.groupby(["symbol", "week_start"], as_index=False, sort=False)
            .apply(lambda g: _ohlcv_agg(g.sort_values("date")))
            .reset_index(drop=True)
        )
        # 補 info（每檔固定）
        wk = wk.merge(
            d.groupby("symbol")[["market", "market_detail", "sector", "name"]].first().reset_index(),
            on="symbol",
            how="left",
        )

        wk["week_start"] = pd.to_datetime(wk["week_start"]).dt.strftime("%Y-%m-%d")
        wk["start_date"] = pd.to_datetime(wk["start_date"]).dt.strftime("%Y-%m-%d")
        wk["end_date"] = pd.to_datetime(wk["end_date"]).dt.strftime("%Y-%m-%d")

        wk = _add_prev_ret(wk, ("symbol", "week_start"), close_col="close")
        wk["period"] = "W"

        # ==========================
        # Monthly
        # ==========================
        log("🧱 建立 kbar_monthly ...")
        d2 = df.copy()
        d2["month_start"] = d2["date"].dt.to_period("M").dt.start_time
        d2 = d2.sort_values(["symbol", "date"])

        mo = (
            d2.groupby(["symbol", "month_start"], as_index=False, sort=False)
            .apply(lambda g: _ohlcv_agg(g.sort_values("date")))
            .reset_index(drop=True)
        )
        mo = mo.merge(
            d2.groupby("symbol")[["market", "market_detail", "sector", "name"]].first().reset_index(),
            on="symbol",
            how="left",
        )

        mo["month_start"] = pd.to_datetime(mo["month_start"]).dt.strftime("%Y-%m-%d")
        mo["start_date"] = pd.to_datetime(mo["start_date"]).dt.strftime("%Y-%m-%d")
        mo["end_date"] = pd.to_datetime(mo["end_date"]).dt.strftime("%Y-%m-%d")

        mo = _add_prev_ret(mo, ("symbol", "month_start"), close_col="close")
        mo["period"] = "M"

        # ==========================
        # Yearly + Peak
        # ==========================
        log("🧱 建立 kbar_yearly (含年內最高點 peak_date) ...")
        d3 = df.copy()
        d3["year"] = d3["date"].dt.year.astype(int)
        d3 = d3.sort_values(["symbol", "date"])

        # 年K OHLCV
        yr = (
            d3.groupby(["symbol", "year"], as_index=False, sort=False)
            .apply(lambda g: _ohlcv_agg(g.sort_values("date")))
            .reset_index(drop=True)
        )

        # 年內最高點（用 high）
        # 找每個 symbol-year 的最大 high 的那天（若有同高，取最早出現）
        peak_idx = d3.groupby(["symbol", "year"])["high"].idxmax()
        peak_df = d3.loc[peak_idx, ["symbol", "year", "date", "high"]].copy()
        peak_df = peak_df.rename(columns={"date": "peak_date", "high": "peak_high"})

        # 若 high 全 NaN，idxmax 會爆；做保護
        if peak_df.empty:
            yr["peak_date"] = None
            yr["peak_high"] = np.nan
        else:
            yr = yr.merge(peak_df, on=["symbol", "year"], how="left")

        # 補 info
        yr = yr.merge(
            d3.groupby("symbol")[["market", "market_detail", "sector", "name"]].first().reset_index(),
            on="symbol",
            how="left",
        )

        # 年K額外指標：年報酬、peak_ret
        yr = yr.sort_values(["symbol", "year"])
        yr["prev_close"] = yr.groupby("symbol")["close"].shift(1)
        yr["year_ret"] = (yr["close"] / yr["prev_close"]) - 1
        yr["year_logret"] = np.log(yr["close"] / yr["prev_close"])
        yr.loc[yr["prev_close"].isna(), ["year_ret", "year_logret"]] = np.nan

        # 年內 peak 相對年初 open 的 peak_ret（百分比）
        yr["peak_ret"] = np.where(
            (yr["open"].notna()) & (yr["open"] > 0) & (yr["peak_high"].notna()),
            (yr["peak_high"] / yr["open"] - 1) * 100.0,
            np.nan,
        )

        # 轉字串
        yr["peak_date"] = pd.to_datetime(yr["peak_date"], errors="coerce").dt.strftime("%Y-%m-%d")
        yr["start_date"] = pd.to_datetime(yr["start_date"]).dt.strftime("%Y-%m-%d")
        yr["end_date"] = pd.to_datetime(yr["end_date"]).dt.strftime("%Y-%m-%d")
        yr["period"] = "Y"

        # ==========================
        # Write back to DB
        # ==========================
        log("🧾 寫回資料庫（replace）: kbar_weekly / kbar_monthly / kbar_yearly")

        conn.execute("DROP TABLE IF EXISTS kbar_weekly")
        conn.execute("DROP TABLE IF EXISTS kbar_monthly")
        conn.execute("DROP TABLE IF EXISTS kbar_yearly")

        wk.to_sql("kbar_weekly", conn, if_exists="replace", index=False)
        mo.to_sql("kbar_monthly", conn, if_exists="replace", index=False)
        yr.to_sql("kbar_yearly", conn, if_exists="replace", index=False)

        # Indexes（查詢速度會差很多）
        conn.execute("CREATE INDEX IF NOT EXISTS idx_wk_symbol_week ON kbar_weekly(symbol, week_start)")
        conn.execute("CREATE INDEX IF NOT EXISTS idx_mo_symbol_month ON kbar_monthly(symbol, month_start)")
        conn.execute("CREATE INDEX IF NOT EXISTS idx_yr_symbol_year ON kbar_yearly(symbol, year)")
        conn.execute("CREATE INDEX IF NOT EXISTS idx_yr_peak_date ON kbar_yearly(peak_date)")
        conn.commit()

        # 小優化
        log("🧹 VACUUM ...")
        conn.execute("VACUUM")
        conn.commit()

        log(
            f"✅ 完成 kbars | weekly={len(wk):,} monthly={len(mo):,} yearly={len(yr):,}"
        )
        return KbarBuildResult(len(wk), len(mo), len(yr))

    finally:
        conn.close()


# --------------------------
# CLI
# --------------------------
if __name__ == "__main__":
    import sys

    if len(sys.argv) < 2:
        print("Usage: python kbar_aggregator.py <db_file>")
        print("Example: python kbar_aggregator.py tw_stock_warehouse.db")
        raise SystemExit(1)

    build_kbars(sys.argv[1])
