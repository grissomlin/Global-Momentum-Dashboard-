# kbar_aggregator.py
# -*- coding: utf-8 -*-
"""
kbar_aggregator.py
------------------
從 stock_analysis（日頻特徵層）聚合出：
- k_weekly   : 週K（週開高低收/週報酬/週logret/週漲停天數等）
- k_monthly  : 月K（月開高低收/月報酬/月logret/月漲停天數等）
- k_annual   : 年K（年開高低收/年報酬/年logret/peak_date/peak_high 等）
並寫回同一個 SQLite DB。

✅ 特點
- 只依賴 stock_analysis（processor 產物），不依賴 yfinance Adj Close
- 內建「乒乓極端震盪」清洗（保守版）：避免減資/併股等造成的反常跳動污染聚合
- 產生可對齊用的 period id：
  - week_id  : YYYY-WW（ISO week）
  - month_id : YYYY-MM
  - year_id  : YYYY

⚠️ 注意
- 若你未來要 “交易所週定義”（台股週K用交易所週界），ISO week 大多可用。
  若你要完全對齊交易所定義，可再加自訂週切法（之後我也能幫你加）。
"""

import sqlite3
import pandas as pd
import numpy as np
from typing import Optional

# 乒乓清洗門檻（保守版）
PINGPONG_THRESHOLD = 0.40  # 40%

# 異常日報酬的硬上限（保守護欄）
ABS_DAILY_RET_CAP = 0.80   # 80%（超過視為高度可疑）

# 對於「極少量/無量」市場或資料品質差的 symbol，可選擇關閉清洗
ENABLE_CLEANING_DEFAULT = True


def _ensure_datetime(s: pd.Series) -> pd.Series:
    return pd.to_datetime(s, errors="coerce").dt.tz_localize(None)


def _pingpong_clean_daily(df: pd.DataFrame,
                          threshold: float = PINGPONG_THRESHOLD,
                          abs_cap: float = ABS_DAILY_RET_CAP) -> pd.DataFrame:
    """
    參考你那段 Adj Close 清洗邏輯，但因為 SQLite 沒 Adj Close，
    這裡用 close 的 pct_change 做「乒乓」偵測 + 超大跳動保護。

    規則：
    - 若連續兩日 |ret| > threshold 且方向相反 => 兩天都標記為異常剔除
    - 若單日 |ret| > abs_cap => 剔除（避免極端錯價）
    """
    if df.empty or len(df) < 5:
        return df

    df = df.sort_values("date").copy()
    close = df["close"].astype(float)
    ret = close.pct_change()

    # 硬上限
    mask_abs = ret.abs() > abs_cap

    # 乒乓：i 與 i+1 同時極端且反向
    mask_pingpong = pd.Series(False, index=df.index)
    for i in range(1, len(df) - 1):
        prev = ret.iloc[i]
        nxt = ret.iloc[i + 1]
        if pd.notna(prev) and pd.notna(nxt):
            if (abs(prev) > threshold) and (abs(nxt) > threshold) and (prev * nxt < 0):
                mask_pingpong.iloc[i] = True
                mask_pingpong.iloc[i + 1] = True

    mask = mask_abs | mask_pingpong
    out = df.loc[~mask].copy()

    # 剔除後可能造成 prev_close/daily_change 不一致，這裡重算最安全
    out["prev_close"] = out["close"].shift(1)
    out["daily_change"] = out["close"].pct_change()
    return out


def _make_period_ids(df: pd.DataFrame) -> pd.DataFrame:
    """
    增加 week_id / month_id / year_id / week_end_date（月末/週末對齊）
    week_id 使用 ISO week：YYYY-WW
    """
    d = df["date"]
    iso = d.dt.isocalendar()
    df["year_id"] = d.dt.year.astype(int)
    df["month_id"] = d.dt.strftime("%Y-%m")
    df["week_id"] = iso["year"].astype(str) + "-" + iso["week"].astype(str).str.zfill(2)

    # period end
    df["week_end_date"] = (d + pd.to_timedelta(6 - d.dt.weekday, unit="D")).dt.normalize()
    df["month_end_date"] = d.dt.to_period("M").dt.to_timestamp("M")
    df["year_end_date"] = d.dt.to_period("Y").dt.to_timestamp("Y")
    return df


def _agg_ohlc(group: pd.DataFrame, prefix: str) -> pd.Series:
    """
    對日K聚合成週/月/年 OHLC（用 open/high/low/close 原始欄位）
    """
    g = group.sort_values("date")
    o = g["open"].iloc[0]
    h = g["high"].max()
    l = g["low"].min()
    c = g["close"].iloc[-1]
    v = g["volume"].sum() if "volume" in g.columns else np.nan
    return pd.Series({
        f"{prefix}_open": float(o) if pd.notna(o) else np.nan,
        f"{prefix}_high": float(h) if pd.notna(h) else np.nan,
        f"{prefix}_low": float(l) if pd.notna(l) else np.nan,
        f"{prefix}_close": float(c) if pd.notna(c) else np.nan,
        f"{prefix}_volume": float(v) if pd.notna(v) else np.nan,
    })


def _calc_period_returns(df: pd.DataFrame, prefix: str) -> pd.DataFrame:
    """
    計算 period_return / period_logret_net
    - return: close/open - 1
    - logret_net: sum(log(1+daily_change))，更能做貢獻度相加
    """
    open_col = f"{prefix}_open"
    close_col = f"{prefix}_close"
    df[f"{prefix}_ret"] = (df[close_col] / df[open_col] - 1.0).replace([np.inf, -np.inf], np.nan)
    df[f"{prefix}_logret_net"] = df[f"{prefix}_logret_net"].astype(float)  # already computed
    return df


def build_kbars(db_path: str,
                enable_cleaning: bool = ENABLE_CLEANING_DEFAULT,
                verbose: bool = True):
    """
    主入口：生成週/月/年K表
    依賴：stock_analysis
    輸出：k_weekly, k_monthly, k_annual
    """
    conn = sqlite3.connect(db_path)
    try:
        df = pd.read_sql(
            """
            SELECT
                symbol, date, open, high, low, close, volume,
                market, market_detail,
                daily_change, prev_close,
                is_limit_up
            FROM stock_analysis
            """,
            conn,
        )
    except Exception as e:
        conn.close()
        raise RuntimeError(f"讀取 stock_analysis 失敗：{e}")

    if df.empty:
        conn.close()
        if verbose:
            print("❌ stock_analysis 無資料，跳過 kbar 聚合")
        return

    df["date"] = _ensure_datetime(df["date"])
    df = df.dropna(subset=["date", "symbol"]).sort_values(["symbol", "date"]).reset_index(drop=True)

    # 每檔做清洗（保守版）再聚合
    out_weekly = []
    out_monthly = []
    out_annual = []

    for sym, g in df.groupby("symbol", sort=False):
        g = g.copy().sort_values("date").reset_index(drop=True)

        if enable_cleaning:
            g = _pingpong_clean_daily(g)

        if len(g) < 30:
            continue

        g = _make_period_ids(g)

        # ========== Weekly ==========
        wk = g.groupby(["symbol", "week_id"], as_index=False).apply(lambda x: _agg_ohlc(x, "w"))
        wk = wk.reset_index(drop=True)

        # 週期間 net logret（用日 logret 可加總）
        g["daily_logret"] = np.log1p(g["daily_change"].astype(float).fillna(0.0))
        wk_net = g.groupby(["symbol", "week_id"], as_index=False)["daily_logret"].sum()
        wk_net = wk_net.rename(columns={"daily_logret": "w_logret_net"})
        wk = wk.merge(wk_net, on=["symbol", "week_id"], how="left")

        # 週漲停天數（dense）
        if "is_limit_up" in g.columns:
            wk_lu = g.groupby(["symbol", "week_id"], as_index=False)["is_limit_up"].sum()
            wk_lu = wk_lu.rename(columns={"is_limit_up": "w_is_limitup_dense"})
            wk = wk.merge(wk_lu, on=["symbol", "week_id"], how="left")
        else:
            wk["w_is_limitup_dense"] = 0

        # 加入週起迄日期
        wk_dates = g.groupby(["symbol", "week_id"], as_index=False).agg(
            w_start_date=("date", "min"),
            w_end_date=("date", "max"),
            market=("market", "last"),
            market_detail=("market_detail", "last"),
        )
        wk = wk.merge(wk_dates, on=["symbol", "week_id"], how="left")
        wk["w_ret"] = (wk["w_close"] / wk["w_open"] - 1.0).replace([np.inf, -np.inf], np.nan)

        out_weekly.append(wk)

        # ========== Monthly ==========
        mo = g.groupby(["symbol", "month_id"], as_index=False).apply(lambda x: _agg_ohlc(x, "m"))
        mo = mo.reset_index(drop=True)

        mo_net = g.groupby(["symbol", "month_id"], as_index=False)["daily_logret"].sum()
        mo_net = mo_net.rename(columns={"daily_logret": "m_logret_net"})
        mo = mo.merge(mo_net, on=["symbol", "month_id"], how="left")

        mo_lu = g.groupby(["symbol", "month_id"], as_index=False)["is_limit_up"].sum()
        mo_lu = mo_lu.rename(columns={"is_limit_up": "m_is_limitup_dense"})
        mo = mo.merge(mo_lu, on=["symbol", "month_id"], how="left")

        mo_dates = g.groupby(["symbol", "month_id"], as_index=False).agg(
            m_start_date=("date", "min"),
            m_end_date=("date", "max"),
            year_id=("year_id", "last"),
            market=("market", "last"),
            market_detail=("market_detail", "last"),
        )
        mo = mo.merge(mo_dates, on=["symbol", "month_id"], how="left")
        mo["m_ret"] = (mo["m_close"] / mo["m_open"] - 1.0).replace([np.inf, -np.inf], np.nan)

        out_monthly.append(mo)

        # ========== Annual ==========
        yr = g.groupby(["symbol", "year_id"], as_index=False).apply(lambda x: _agg_ohlc(x, "y"))
        yr = yr.reset_index(drop=True)

        yr_net = g.groupby(["symbol", "year_id"], as_index=False)["daily_logret"].sum()
        yr_net = yr_net.rename(columns={"daily_logret": "y_logret_net"})
        yr = yr.merge(yr_net, on=["symbol", "year_id"], how="left")

        yr_lu = g.groupby(["symbol", "year_id"], as_index=False)["is_limit_up"].sum()
        yr_lu = yr_lu.rename(columns={"is_limit_up": "y_is_limitup_dense"})
        yr = yr.merge(yr_lu, on=["symbol", "year_id"], how="left")

        # 年內 peak（用 high 最大值）
        g["year_id"] = g["year_id"].astype(int)
        peak = g.loc[g.groupby("year_id")["high"].idxmax(), ["year_id", "date", "high"]].copy()
        peak = peak.rename(columns={"date": "peak_date", "high": "peak_high"})
        peak["symbol"] = sym
        yr = yr.merge(peak, on=["symbol", "year_id"], how="left")

        yr_dates = g.groupby(["symbol", "year_id"], as_index=False).agg(
            y_start_date=("date", "min"),
            y_end_date=("date", "max"),
            market=("market", "last"),
            market_detail=("market_detail", "last"),
        )
        yr = yr.merge(yr_dates, on=["symbol", "year_id"], how="left")
        yr["y_ret"] = (yr["y_close"] / yr["y_open"] - 1.0).replace([np.inf, -np.inf], np.nan)

        out_annual.append(yr)

    if not out_weekly:
        conn.close()
        if verbose:
            print("❌ 無可聚合資料（可能 stock_analysis 太短或欄位缺失）")
        return

    k_weekly = pd.concat(out_weekly, ignore_index=True)
    k_monthly = pd.concat(out_monthly, ignore_index=True)
    k_annual = pd.concat(out_annual, ignore_index=True)

    # 日期轉字串，SQLite 穩定
    for c in ["w_start_date", "w_end_date", "m_start_date", "m_end_date", "y_start_date", "y_end_date", "peak_date"]:
        if c in k_weekly.columns:
            k_weekly[c] = pd.to_datetime(k_weekly[c], errors="coerce").dt.strftime("%Y-%m-%d")
        if c in k_monthly.columns:
            k_monthly[c] = pd.to_datetime(k_monthly[c], errors="coerce").dt.strftime("%Y-%m-%d")
        if c in k_annual.columns:
            k_annual[c] = pd.to_datetime(k_annual[c], errors="coerce").dt.strftime("%Y-%m-%d")

    # 寫回 DB（覆蓋）
    cur = conn.cursor()
    cur.execute("DROP TABLE IF EXISTS k_weekly")
    cur.execute("DROP TABLE IF EXISTS k_monthly")
    cur.execute("DROP TABLE IF EXISTS k_annual")
    conn.commit()

    k_weekly.to_sql("k_weekly", conn, if_exists="replace", index=False)
    k_monthly.to_sql("k_monthly", conn, if_exists="replace", index=False)
    k_annual.to_sql("k_annual", conn, if_exists="replace", index=False)

    # 索引
    try:
        cur.execute("CREATE INDEX IF NOT EXISTS idx_k_weekly_symbol_week ON k_weekly(symbol, week_id)")
        cur.execute("CREATE INDEX IF NOT EXISTS idx_k_monthly_symbol_month ON k_monthly(symbol, month_id)")
        cur.execute("CREATE INDEX IF NOT EXISTS idx_k_annual_symbol_year ON k_annual(symbol, year_id)")
        cur.execute("CREATE INDEX IF NOT EXISTS idx_k_annual_peak ON k_annual(symbol, peak_date)")
        conn.commit()
    except Exception:
        pass

    conn.close()

    if verbose:
        print("✅ kbar_aggregator 完成：已產生 k_weekly / k_monthly / k_annual")
        print(f"   k_weekly rows : {len(k_weekly):,}")
        print(f"   k_monthly rows: {len(k_monthly):,}")
        print(f"   k_annual rows : {len(k_annual):,}")


if __name__ == "__main__":
    # 測試：自行替換 DB
    build_kbars("tw_stock_warehouse.db", enable_cleaning=True, verbose=True)
