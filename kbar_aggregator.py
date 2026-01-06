# kbar_aggregator.py
# -*- coding: utf-8 -*-
"""
kbar_aggregator.py
------------------
從日K（stock_analysis 或 stock_prices）聚合週K/月K/年K：

輸出表：
- kbar_weekly : symbol, year, week_id, period_start, period_end, open, high, low, close, volume
- kbar_monthly: symbol, year, month_id, period_start, period_end, open, high, low, close, volume
- kbar_yearly : symbol, year, period_start, period_end, open, high, low, close, volume,
                year_peak_date, year_peak_high

✅ 特點（加法，不會破壞你現有功能）
- 使用「clean_close」做聚合：保留原 close，不改 stock_analysis
- 內建異常報酬清洗（可關）
  1) 超限值平滑：若市場有漲跌幅限制（TW/CN/JP 可用 market_rules），abs(daily_ret) > limit*1.5 視為異常
  2) pingpong：連續兩日 |ret| > 0.40 且方向相反，視為異常震盪（減資/併購/資料錯）
- 可優先讀 stock_analysis（有 prev_close / market 等），沒有再退回 stock_prices

用法：
    python kbar_aggregator.py tw_stock_warehouse.db
或：
    from kbar_aggregator import build_kbar_tables
    build_kbar_tables("tw_stock_warehouse.db")

依賴（可選）：
- market_rules.py（若存在，會用它拿 limit_up_pct / tick 等；不存在就 fallback）
"""

import sys
import sqlite3
import numpy as np
import pandas as pd
from typing import Optional, Dict

SQLITE_TIMEOUT = 120

# -------------------------
# optional market_rules
# -------------------------
try:
    import market_rules
    HAS_MARKET_RULES = True
except Exception:
    market_rules = None
    HAS_MARKET_RULES = False


def _fallback_limit_up_pct(market: str, market_detail: str, symbol: str) -> Optional[float]:
    """fallback：只做常見 TW=10%、CN=10/20、JP=None"""
    m = (market or "").upper().strip()
    sym = (symbol or "").upper().strip()
    md = (market_detail or "").lower().strip()

    if m in ["TW", "TSE", "GTSM"] or sym.endswith(".TW") or sym.endswith(".TWO"):
        if md == "emerging":
            return None
        return 0.10

    if m in ["CN", "SSE", "SZSE", "CHINA"] or sym.endswith(".SS") or sym.endswith(".SZ"):
        code = "".join([c for c in sym if c.isdigit()])
        if code.startswith(("300", "301", "688")):
            return 0.20
        return 0.10

    if m in ["JP", "JPX", "TSE"] or sym.endswith(".T"):
        return None

    return None


def _get_limit_up_pct(market: str, market_detail: str, symbol: str) -> Optional[float]:
    if HAS_MARKET_RULES and hasattr(market_rules, "get_rule"):
        try:
            rule = market_rules.get_rule(market=market, market_detail=market_detail, symbol=symbol)
            v = rule.get("limit_up_pct", None)
            if isinstance(v, (int, float)):
                return float(v)
            return None
        except Exception:
            return _fallback_limit_up_pct(market, market_detail, symbol)
    return _fallback_limit_up_pct(market, market_detail, symbol)


# -------------------------
# anomaly cleaning (ADD-ON)
# -------------------------
def _apply_anomaly_cleaning(
    g: pd.DataFrame,
    limit_up_pct: Optional[float],
    enable_pingpong: bool = True,
    pingpong_threshold: float = 0.40,
    enable_overlimit_smoothing: bool = True,
) -> pd.DataFrame:
    """
    以「clean_close」生成乾淨價格序列，用於聚合，不破壞原 close。

    策略（不 drop row，避免破壞週/月切段）：
    - overlimit：把 OHLC 設 NaN -> 以 close ffill -> 其餘用 close 補
    - pingpong：把 (i, i+1) 兩天 OHLC 設 NaN -> ffill
    """
    g = g.sort_values("date").copy()

    # clean_ohlc 初始 = 原始
    for c in ["open", "high", "low", "close"]:
        g[f"clean_{c}"] = pd.to_numeric(g[c], errors="coerce")

    # 先算 daily_ret（用 close）
    g["clean_ret"] = g["clean_close"].pct_change()

    # 1) overlimit smoothing
    if enable_overlimit_smoothing and isinstance(limit_up_pct, (int, float)) and limit_up_pct > 0:
        max_allowed = float(limit_up_pct) * 1.5
        mask_over = g["clean_ret"].abs() > max_allowed
        if mask_over.any():
            for c in ["clean_open", "clean_high", "clean_low", "clean_close"]:
                g.loc[mask_over, c] = np.nan

    # 2) pingpong
    if enable_pingpong:
        r = g["clean_ret"].values
        mask_pp = np.zeros(len(g), dtype=bool)
        for i in range(0, len(g) - 2):
            a = r[i + 0]
            b = r[i + 1]
            if np.isfinite(a) and np.isfinite(b):
                if abs(a) > pingpong_threshold and abs(b) > pingpong_threshold and (a * b) < 0:
                    mask_pp[i] = True
                    mask_pp[i + 1] = True
        if mask_pp.any():
            for c in ["clean_open", "clean_high", "clean_low", "clean_close"]:
                g.loc[mask_pp, c] = np.nan

    # ffill clean_close（核心）
    g["clean_close"] = g["clean_close"].ffill()

    # 其餘 OHLC 若 NaN，用 clean_close 補（保守）
    for c in ["clean_open", "clean_high", "clean_low"]:
        g[c] = g[c].fillna(g["clean_close"])

    # high/low 邏輯修正
    g["clean_high"] = np.maximum.reduce([g["clean_high"], g["clean_open"], g["clean_close"]])
    g["clean_low"] = np.minimum.reduce([g["clean_low"], g["clean_open"], g["clean_close"]])

    return g


# -------------------------
# aggregation helpers
# -------------------------
def _agg_period(g: pd.DataFrame) -> Dict[str, float]:
    """
    g：該 period 的日K（已含 clean_*）
    回傳 period OHLCV（用 clean_OHLC + 原 volume）
    """
    if g.empty:
        return dict(open=np.nan, high=np.nan, low=np.nan, close=np.nan, volume=0.0)

    open_ = float(g["clean_open"].iloc[0]) if np.isfinite(g["clean_open"].iloc[0]) else np.nan
    close_ = float(g["clean_close"].iloc[-1]) if np.isfinite(g["clean_close"].iloc[-1]) else np.nan
    high_ = float(np.nanmax(g["clean_high"].values)) if np.isfinite(np.nanmax(g["clean_high"].values)) else np.nan
    low_ = float(np.nanmin(g["clean_low"].values)) if np.isfinite(np.nanmin(g["clean_low"].values)) else np.nan
    vol_ = float(np.nansum(pd.to_numeric(g["volume"], errors="coerce").fillna(0).values))
    return dict(open=open_, high=high_, low=low_, close=close_, volume=vol_)


def build_kbar_tables(
    db_path: str,
    source_table_prefer: str = "stock_analysis",
    enable_anomaly_cleaning: bool = True,
    enable_pingpong: bool = True,
    pingpong_threshold: float = 0.40,
    enable_overlimit_smoothing: bool = True,
) -> Dict[str, int]:
    conn = sqlite3.connect(db_path, timeout=SQLITE_TIMEOUT)

    try:
        existing = set(pd.read_sql("SELECT name FROM sqlite_master WHERE type='table'", conn)["name"].tolist())

        # 選來源
        source = None
        if source_table_prefer in existing:
            source = source_table_prefer
        elif "stock_prices" in existing:
            source = "stock_prices"
        else:
            raise RuntimeError("找不到 stock_analysis 或 stock_prices，請先跑 downloader + processor")

        # 讀資料（若是 stock_prices 可能沒有 market / market_detail，盡量 join stock_info）
        if source == "stock_analysis":
            df = pd.read_sql(
                """
                SELECT symbol, date, open, high, low, close, volume,
                       market, market_detail
                FROM stock_analysis
                """,
                conn,
            )
        else:
            if "stock_info" in existing:
                df = pd.read_sql(
                    """
                    SELECT p.symbol, p.date, p.open, p.high, p.low, p.close, p.volume,
                           i.market, i.market_detail
                    FROM stock_prices p
                    LEFT JOIN stock_info i ON p.symbol = i.symbol
                    """,
                    conn,
                )
            else:
                df = pd.read_sql(
                    "SELECT symbol, date, open, high, low, close, volume FROM stock_prices",
                    conn,
                )
                df["market"] = ""
                df["market_detail"] = ""

        if df.empty:
            print("❌ 無日K資料")
            return {"weekly": 0, "monthly": 0, "yearly": 0}

        df["date"] = pd.to_datetime(df["date"], errors="coerce")
        df = df.dropna(subset=["date"]).sort_values(["symbol", "date"]).reset_index(drop=True)

        weekly_rows = []
        monthly_rows = []
        yearly_rows = []

        for sym, g in df.groupby("symbol", sort=False):
            g = g.sort_values("date").copy()
            if len(g) < 10:
                continue

            market = g["market"].iloc[0] if "market" in g.columns else ""
            market_detail = g["market_detail"].iloc[0] if "market_detail" in g.columns else ""
            limit_up_pct = _get_limit_up_pct(market, market_detail, sym)

            # 異常清洗（產生 clean_*）
            if enable_anomaly_cleaning:
                g = _apply_anomaly_cleaning(
                    g,
                    limit_up_pct=limit_up_pct,
                    enable_pingpong=enable_pingpong,
                    pingpong_threshold=pingpong_threshold,
                    enable_overlimit_smoothing=enable_overlimit_smoothing,
                )
            else:
                for c in ["open", "high", "low", "close"]:
                    g[f"clean_{c}"] = pd.to_numeric(g[c], errors="coerce")

            g["year"] = g["date"].dt.year.astype(int)

            # ========== weekly ==========
            # 週定義：Mon~Sun（pandas 'W-SUN'）
            # period_end = 該週週日，period_start = 週一
            g["week_end"] = g["date"].dt.to_period("W-SUN").dt.end_time.dt.normalize()
            wk_groups = g.groupby(["year", "week_end"], sort=False)

            for (yr, week_end), wg in wk_groups:
                if wg.empty:
                    continue
                week_end = pd.Timestamp(week_end).normalize()
                week_start = (week_end - pd.Timedelta(days=6)).normalize()
                ohlcv = _agg_period(wg)
                # week_id 用 ISO year-week（以 week_end 計）
                iso = week_end.isocalendar()
                week_id = f"{int(iso.year)}-W{int(iso.week):02d}"
                weekly_rows.append(
                    {
                        "symbol": sym,
                        "year": int(yr),
                        "week_id": week_id,
                        "period_start": week_start.strftime("%Y-%m-%d"),
                        "period_end": week_end.strftime("%Y-%m-%d"),
                        **ohlcv,
                    }
                )

            # ========== monthly ==========
            g["month_end"] = g["date"].dt.to_period("M").dt.end_time.dt.normalize()
            mo_groups = g.groupby(["year", "month_end"], sort=False)

            for (yr, month_end), mg in mo_groups:
                if mg.empty:
                    continue
                month_end = pd.Timestamp(month_end).normalize()
                month_start = pd.Timestamp(month_end.replace(day=1)).normalize()
                ohlcv = _agg_period(mg)
                month_id = month_start.strftime("%Y-%m")
                monthly_rows.append(
                    {
                        "symbol": sym,
                        "year": int(yr),
                        "month_id": month_id,
                        "period_start": month_start.strftime("%Y-%m-%d"),
                        "period_end": month_end.strftime("%Y-%m-%d"),
                        **ohlcv,
                    }
                )

            # ========== yearly ==========
            yr_groups = g.groupby("year", sort=False)
            for yr, yg in yr_groups:
                if yg.empty:
                    continue
                period_start = pd.Timestamp(f"{int(yr)}-01-01")
                period_end = pd.Timestamp(f"{int(yr)}-12-31")
                # 實際以該年資料 min/max date 當 period_start/end（更合理）
                period_start = pd.Timestamp(yg["date"].min()).normalize()
                period_end = pd.Timestamp(yg["date"].max()).normalize()

                ohlcv = _agg_period(yg)

                # 年高點：用 clean_high 找 peak date
                idx = yg["clean_high"].astype(float).idxmax()
                year_peak_date = None
                year_peak_high = np.nan
                if pd.notna(idx) and idx in yg.index:
                    year_peak_date = pd.Timestamp(yg.loc[idx, "date"]).normalize()
                    year_peak_high = float(yg.loc[idx, "clean_high"]) if np.isfinite(yg.loc[idx, "clean_high"]) else np.nan

                yearly_rows.append(
                    {
                        "symbol": sym,
                        "year": int(yr),
                        "period_start": period_start.strftime("%Y-%m-%d"),
                        "period_end": period_end.strftime("%Y-%m-%d"),
                        **ohlcv,
                        "year_peak_date": year_peak_date.strftime("%Y-%m-%d") if year_peak_date is not None else None,
                        "year_peak_high": year_peak_high,
                    }
                )

        wk_df = pd.DataFrame(weekly_rows)
        mo_df = pd.DataFrame(monthly_rows)
        yr_df = pd.DataFrame(yearly_rows)

        # 寫回（重建）
        conn.execute("DROP TABLE IF EXISTS kbar_weekly")
        conn.execute("DROP TABLE IF EXISTS kbar_monthly")
        conn.execute("DROP TABLE IF EXISTS kbar_yearly")

        wk_df.to_sql("kbar_weekly", conn, if_exists="replace", index=False)
        mo_df.to_sql("kbar_monthly", conn, if_exists="replace", index=False)
        yr_df.to_sql("kbar_yearly", conn, if_exists="replace", index=False)

        # 索引
        try:
            conn.execute("CREATE INDEX IF NOT EXISTS idx_kbar_w_sym_year ON kbar_weekly(symbol, year)")
            conn.execute("CREATE INDEX IF NOT EXISTS idx_kbar_m_sym_year ON kbar_monthly(symbol, year)")
            conn.execute("CREATE INDEX IF NOT EXISTS idx_kbar_y_sym_year ON kbar_yearly(symbol, year)")
        except Exception:
            pass

        conn.commit()

        print("\n✅ kbar_aggregator 完成（已產生週/月/年 K）")
        print(f"📌 kbar_weekly: {len(wk_df):,} rows")
        print(f"📌 kbar_monthly: {len(mo_df):,} rows")
        print(f"📌 kbar_yearly: {len(yr_df):,} rows")
        print(f"📌 異常清洗: {'ON' if enable_anomaly_cleaning else 'OFF'} (pingpong={enable_pingpong}, thr={pingpong_threshold})")

        return {"weekly": int(len(wk_df)), "monthly": int(len(mo_df)), "yearly": int(len(yr_df))}

    finally:
        conn.close()


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python kbar_aggregator.py <db_path>")
        sys.exit(1)

    build_kbar_tables(sys.argv[1])
