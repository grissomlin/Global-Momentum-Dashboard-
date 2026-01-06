# -*- coding: utf-8 -*-
"""
kbar_aggregator.py
------------------
將日K(stock_prices) 聚合成

✅ kbar_weekly   (週K)
✅ kbar_monthly  (月K)
✅ kbar_yearly   (年K, 含 peak_date / peak_high / peak_high_ret)

設計目標：
- 與你的 DB schema 相容：stock_prices(symbol,date,open,high,low,close,volume)
- 能用於後續「對齊年K最高點」的貢獻度研究（peak_date 是關鍵）
- 可直接在 main.py / pipeline 呼叫：build_kbars(db_path)

注意：
- 這裡使用「自然週」W-MON（週一開始，週日結束）你可以改成 W-FRI 等
- 月K：每月自然月
- 年K：每年自然年
"""

import sqlite3
from datetime import datetime
import pandas as pd


# ======================
# 工具
# ======================
def log(msg: str):
    print(f"{pd.Timestamp.now():%H:%M:%S}: {msg}", flush=True)


def _ensure_tables(conn: sqlite3.Connection):
    """建立 kbar_* 表（若不存在），並建立索引"""
    conn.execute(
        """
        CREATE TABLE IF NOT EXISTS kbar_weekly (
            symbol TEXT,
            week_start TEXT,
            week_end TEXT,
            open REAL,
            high REAL,
            low REAL,
            close REAL,
            volume INTEGER,
            prev_close REAL,
            ret_pct REAL,
            logret REAL,
            PRIMARY KEY (symbol, week_start)
        )
        """
    )
    conn.execute(
        """
        CREATE TABLE IF NOT EXISTS kbar_monthly (
            symbol TEXT,
            month TEXT,           -- YYYY-MM
            period_start TEXT,
            period_end TEXT,
            open REAL,
            high REAL,
            low REAL,
            close REAL,
            volume INTEGER,
            prev_close REAL,
            ret_pct REAL,
            logret REAL,
            PRIMARY KEY (symbol, month)
        )
        """
    )
    conn.execute(
        """
        CREATE TABLE IF NOT EXISTS kbar_yearly (
            symbol TEXT,
            year INTEGER,
            period_start TEXT,
            period_end TEXT,
            open REAL,
            high REAL,
            low REAL,
            close REAL,
            volume INTEGER,

            peak_date TEXT,       -- 年內最高價當日日期
            peak_high REAL,       -- 年內最高價
            peak_high_ret REAL,   -- (peak_high / year_open - 1) * 100

            prev_close REAL,
            ret_pct REAL,
            logret REAL,

            PRIMARY KEY (symbol, year)
        )
        """
    )

    conn.execute("CREATE INDEX IF NOT EXISTS idx_kbar_w_symbol ON kbar_weekly(symbol)")
    conn.execute("CREATE INDEX IF NOT EXISTS idx_kbar_m_symbol ON kbar_monthly(symbol)")
    conn.execute("CREATE INDEX IF NOT EXISTS idx_kbar_y_symbol ON kbar_yearly(symbol)")


def _read_stock_prices(conn: sqlite3.Connection) -> pd.DataFrame:
    """讀取 stock_prices + 基本清洗"""
    df = pd.read_sql(
        """
        SELECT symbol, date, open, high, low, close, volume
        FROM stock_prices
        """,
        conn,
    )
    if df.empty:
        return df

    df["date"] = pd.to_datetime(df["date"])
    df = df.sort_values(["symbol", "date"])
    # 基本防呆：缺欄位補 0
    if "volume" in df.columns:
        df["volume"] = pd.to_numeric(df["volume"], errors="coerce").fillna(0).astype("int64")
    for c in ["open", "high", "low", "close"]:
        df[c] = pd.to_numeric(df[c], errors="coerce")
    return df


def _calc_prev_ret_log(df: pd.DataFrame, key_cols):
    """
    給定 df（已經是週/月/年聚合結果），依 key_cols（symbol + period key）排序後
    加上 prev_close / ret_pct / logret
    """
    df = df.sort_values(["symbol"] + key_cols)

    df["prev_close"] = df.groupby("symbol")["close"].shift(1)

    df["ret_pct"] = None
    mask = df["prev_close"].notna() & (df["prev_close"] > 0) & df["close"].notna()
    df.loc[mask, "ret_pct"] = (df.loc[mask, "close"] / df.loc[mask, "prev_close"] - 1.0) * 100.0

    df["logret"] = None
    mask2 = df["prev_close"].notna() & (df["prev_close"] > 0) & df["close"].notna() & (df["close"] > 0)
    df.loc[mask2, "logret"] = (df.loc[mask2, "close"] / df.loc[mask2, "prev_close"]).map(
        lambda x: None if pd.isna(x) else float(pd.np.log(x))  # type: ignore
    )
    return df


# ======================
# 聚合：週K
# ======================
def _build_weekly(df_daily: pd.DataFrame) -> pd.DataFrame:
    """
    週K：W-MON（週一為週期起點）
    週區間：week_start / week_end
    """
    if df_daily.empty:
        return df_daily

    d = df_daily.copy()
    # week_start：週一
    d["week_start"] = d["date"].dt.to_period("W-MON").apply(lambda p: p.start_time)
    d["week_end"] = d["week_start"] + pd.Timedelta(days=6)

    g = d.groupby(["symbol", "week_start"], as_index=False)

    out = g.agg(
        week_end=("week_end", "max"),
        open=("open", "first"),
        high=("high", "max"),
        low=("low", "min"),
        close=("close", "last"),
        volume=("volume", "sum"),
    )

    # 字串化
    out["week_start"] = pd.to_datetime(out["week_start"]).dt.strftime("%Y-%m-%d")
    out["week_end"] = pd.to_datetime(out["week_end"]).dt.strftime("%Y-%m-%d")

    # prev_close / ret / logret（依 week_start 排）
    out = out.sort_values(["symbol", "week_start"])
    out["prev_close"] = out.groupby("symbol")["close"].shift(1)

    mask = out["prev_close"].notna() & (out["prev_close"] > 0) & out["close"].notna()
    out["ret_pct"] = None
    out.loc[mask, "ret_pct"] = (out.loc[mask, "close"] / out.loc[mask, "prev_close"] - 1.0) * 100.0

    out["logret"] = None
    mask2 = out["prev_close"].notna() & (out["prev_close"] > 0) & out["close"].notna() & (out["close"] > 0)
    out.loc[mask2, "logret"] = (out.loc[mask2, "close"] / out.loc[mask2, "prev_close"]).map(
        lambda x: None if pd.isna(x) else float(pd.np.log(x))  # type: ignore
    )

    return out


# ======================
# 聚合：月K
# ======================
def _build_monthly(df_daily: pd.DataFrame) -> pd.DataFrame:
    if df_daily.empty:
        return df_daily

    d = df_daily.copy()
    d["month"] = d["date"].dt.to_period("M").astype(str)  # YYYY-MM
    d["period_start"] = d["date"].dt.to_period("M").apply(lambda p: p.start_time)
    d["period_end"] = d["date"].dt.to_period("M").apply(lambda p: p.end_time)

    g = d.groupby(["symbol", "month"], as_index=False)

    out = g.agg(
        period_start=("period_start", "min"),
        period_end=("period_end", "max"),
        open=("open", "first"),
        high=("high", "max"),
        low=("low", "min"),
        close=("close", "last"),
        volume=("volume", "sum"),
    )

    out["period_start"] = pd.to_datetime(out["period_start"]).dt.strftime("%Y-%m-%d")
    out["period_end"] = pd.to_datetime(out["period_end"]).dt.strftime("%Y-%m-%d")

    out = out.sort_values(["symbol", "month"])
    out["prev_close"] = out.groupby("symbol")["close"].shift(1)

    mask = out["prev_close"].notna() & (out["prev_close"] > 0) & out["close"].notna()
    out["ret_pct"] = None
    out.loc[mask, "ret_pct"] = (out.loc[mask, "close"] / out.loc[mask, "prev_close"] - 1.0) * 100.0

    out["logret"] = None
    mask2 = out["prev_close"].notna() & (out["prev_close"] > 0) & out["close"].notna() & (out["close"] > 0)
    out.loc[mask2, "logret"] = (out.loc[mask2, "close"] / out.loc[mask2, "prev_close"]).map(
        lambda x: None if pd.isna(x) else float(pd.np.log(x))  # type: ignore
    )

    return out


# ======================
# 聚合：年K（含 peak_date）
# ======================
def _build_yearly(df_daily: pd.DataFrame) -> pd.DataFrame:
    if df_daily.empty:
        return df_daily

    d = df_daily.copy()
    d["year"] = d["date"].dt.year
    d["period_start"] = d["date"].dt.to_period("Y").apply(lambda p: p.start_time)
    d["period_end"] = d["date"].dt.to_period("Y").apply(lambda p: p.end_time)

    # 先做年K OHLCV
    g = d.groupby(["symbol", "year"], as_index=False)
    y = g.agg(
        period_start=("period_start", "min"),
        period_end=("period_end", "max"),
        open=("open", "first"),
        high=("high", "max"),
        low=("low", "min"),
        close=("close", "last"),
        volume=("volume", "sum"),
    )

    y["period_start"] = pd.to_datetime(y["period_start"]).dt.strftime("%Y-%m-%d")
    y["period_end"] = pd.to_datetime(y["period_end"]).dt.strftime("%Y-%m-%d")

    # peak_date / peak_high：由日K找年內最高 high 的那一天
    # 做法：先找每個 (symbol, year) 的 max_high，再回貼第一個命中的日期
    d_valid = d.dropna(subset=["high"]).copy()
    max_high = (
        d_valid.groupby(["symbol", "year"], as_index=False)["high"]
        .max()
        .rename(columns={"high": "peak_high"})
    )

    d2 = d_valid.merge(max_high, on=["symbol", "year"], how="left")
    d2 = d2[d2["high"] == d2["peak_high"]].sort_values(["symbol", "year", "date"])
    peak = d2.groupby(["symbol", "year"], as_index=False).first()[["symbol", "year", "date", "peak_high"]]
    peak = peak.rename(columns={"date": "peak_date"})
    peak["peak_date"] = pd.to_datetime(peak["peak_date"]).dt.strftime("%Y-%m-%d")

    y = y.merge(peak, on=["symbol", "year"], how="left")

    # peak_high_ret = (peak_high / year_open - 1) * 100
    y["peak_high_ret"] = None
    maskp = y["open"].notna() & (y["open"] > 0) & y["peak_high"].notna()
    y.loc[maskp, "peak_high_ret"] = (y.loc[maskp, "peak_high"] / y.loc[maskp, "open"] - 1.0) * 100.0

    # prev_close / ret / logret（依 year 排）
    y = y.sort_values(["symbol", "year"])
    y["prev_close"] = y.groupby("symbol")["close"].shift(1)

    mask = y["prev_close"].notna() & (y["prev_close"] > 0) & y["close"].notna()
    y["ret_pct"] = None
    y.loc[mask, "ret_pct"] = (y.loc[mask, "close"] / y.loc[mask, "prev_close"] - 1.0) * 100.0

    y["logret"] = None
    mask2 = y["prev_close"].notna() & (y["prev_close"] > 0) & y["close"].notna() & (y["close"] > 0)
    y.loc[mask2, "logret"] = (y.loc[mask2, "close"] / y.loc[mask2, "prev_close"]).map(
        lambda x: None if pd.isna(x) else float(pd.np.log(x))  # type: ignore
    )

    return y


# ======================
# 寫入 DB（replace or upsert）
# ======================
def _write_table(conn: sqlite3.Connection, df: pd.DataFrame, table_name: str, pk_cols: list):
    """
    以 INSERT OR REPLACE 寫入，避免重複
    """
    if df.empty:
        return

    cols = list(df.columns)
    placeholders = ", ".join(["?"] * len(cols))
    col_list = ", ".join(cols)

    sql = f"INSERT OR REPLACE INTO {table_name} ({col_list}) VALUES ({placeholders})"
    data = df[cols].where(pd.notna(df[cols]), None).values.tolist()
    conn.executemany(sql, data)


# ======================
# 對外 API
# ======================
def build_kbars(db_path: str, rebuild: bool = True) -> dict:
    """
    入口：從 stock_prices 建立 kbar_weekly/monthly/yearly

    rebuild=True：會先 DROP 舊表再重建（最乾淨）
    rebuild=False：保留表結構，只做 INSERT OR REPLACE 更新（較快）
    """
    t0 = time.time() if "time" in globals() else None  # 防呆

    log(f"🧱 開始建立 KBar 聚合表: {db_path}")

    conn = sqlite3.connect(db_path)
    try:
        if rebuild:
            conn.execute("DROP TABLE IF EXISTS kbar_weekly")
            conn.execute("DROP TABLE IF EXISTS kbar_monthly")
            conn.execute("DROP TABLE IF EXISTS kbar_yearly")
            conn.commit()

        _ensure_tables(conn)

        df_daily = _read_stock_prices(conn)
        if df_daily.empty:
            log("⚠️ stock_prices 沒資料，跳過 kbar 聚合")
            return {"ok": False, "weekly": 0, "monthly": 0, "yearly": 0}

        log(f"📥 讀取日K完成: {len(df_daily):,} 筆 | symbols={df_daily['symbol'].nunique():,}")

        # Build
        log("🧩 生成週K...")
        w = _build_weekly(df_daily)
        log(f"✅ 週K完成: {len(w):,}")

        log("🧩 生成月K...")
        m = _build_monthly(df_daily)
        log(f"✅ 月K完成: {len(m):,}")

        log("🧩 生成年K（含 peak_date）...")
        y = _build_yearly(df_daily)
        log(f"✅ 年K完成: {len(y):,}")

        # Write
        log("💾 寫入資料庫...")
        _write_table(conn, w, "kbar_weekly", ["symbol", "week_start"])
        _write_table(conn, m, "kbar_monthly", ["symbol", "month"])
        _write_table(conn, y, "kbar_yearly", ["symbol", "year"])
        conn.commit()

        # 索引與統計
        weekly_cnt = conn.execute("SELECT COUNT(*) FROM kbar_weekly").fetchone()[0]
        monthly_cnt = conn.execute("SELECT COUNT(*) FROM kbar_monthly").fetchone()[0]
        yearly_cnt = conn.execute("SELECT COUNT(*) FROM kbar_yearly").fetchone()[0]

        log(f"📊 KBar 聚合完成 | weekly={weekly_cnt:,} monthly={monthly_cnt:,} yearly={yearly_cnt:,}")

        return {"ok": True, "weekly": weekly_cnt, "monthly": monthly_cnt, "yearly": yearly_cnt}

    finally:
        conn.close()


# ======================
# CLI
# ======================
if __name__ == "__main__":
    import sys
    import time as _time

    if len(sys.argv) < 2:
        print("用法: python kbar_aggregator.py <db_path> [--no-rebuild]")
        raise SystemExit(1)

    db_path = sys.argv[1]
    rebuild = True
    if len(sys.argv) >= 3 and sys.argv[2] == "--no-rebuild":
        rebuild = False

    t0 = _time.time()
    res = build_kbars(db_path, rebuild=rebuild)
    print(res)
    print(f"耗時: {_time.time()-t0:.1f} 秒")
