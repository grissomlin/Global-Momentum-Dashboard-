# kbar_contribution.py  (精準切段版)
# -*- coding: utf-8 -*-
"""
kbar_contribution.py
--------------------
精準切段版：用「日K逐日 log-return」去切週 / 月區段（依 kbar_weekly / kbar_monthly 的 period_start/period_end）
來算：

- 年K分箱（0~1000 每100 + 1000UP；0~100 再 10% 細分；0-50/50-100）
- 週/月集中度（top1 week / top4 weeks / top1 month / top3 months）
- 高點對齊（peak_date 對到 <=peak_date 的最後交易日）
- peak 前完成度（到 peak 為止的 logret / 年logret）
- peak 前漲停貢獻（漲停日 logret 加總 / 年logret；也提供 / peak_logret）
- peak 前週/月貢獻（用「日K切段」精準加總，不再用週/月收盤近似）

依賴：
- processor.py -> stock_analysis（需含 open/high/low/close/prev_close/is_limit_up）
- kbar_aggregator.py -> kbar_weekly / kbar_monthly / kbar_yearly

用法：
    python kbar_contribution.py tw_stock_warehouse.db
或：
    from kbar_contribution import build_contribution_tables
    build_contribution_tables("tw_stock_warehouse.db")
"""

import sys
import sqlite3
import numpy as np
import pandas as pd
from typing import Optional, Dict

SQLITE_TIMEOUT = 120


# -----------------------------------------------------------------------------
# bins
# -----------------------------------------------------------------------------
def _bin_year_ret_100(ret_pct: float) -> str:
    if not np.isfinite(ret_pct):
        return "NA"
    if ret_pct < 0:
        return "NEGATIVE"
    if ret_pct >= 1000:
        return "1000UP"
    lo = int(ret_pct // 100) * 100
    hi = lo + 100
    return f"{lo:04d}-{hi:04d}"


def _bin_year_ret_10_under100(ret_pct: float) -> str:
    if not np.isfinite(ret_pct):
        return "NA"
    if ret_pct < 0:
        return "NEGATIVE"
    if ret_pct >= 100:
        return "GE_100"
    lo = int(ret_pct // 10) * 10
    hi = lo + 10
    return f"{lo:02d}-{hi:02d}"


def _bin_year_ret_50_under100(ret_pct: float) -> str:
    if not np.isfinite(ret_pct):
        return "NA"
    if ret_pct < 0:
        return "NEGATIVE"
    if ret_pct < 50:
        return "00-50"
    if ret_pct < 100:
        return "50-100"
    return "100UP"


# -----------------------------------------------------------------------------
# math helpers
# -----------------------------------------------------------------------------
def _safe_log_ratio(a: float, b: float) -> float:
    if not np.isfinite(a) or not np.isfinite(b):
        return 0.0
    if a <= 0 or b <= 0:
        return 0.0
    return float(np.log(a / b))


def _topk_share(period_logrets: np.ndarray, denom: float, k: int) -> float:
    """
    只用「正的 period logret」做 top-k 集中度（避免下跌週/月干擾）
    denom <=0 回 0
    """
    if denom <= 0:
        return 0.0
    if period_logrets.size == 0:
        return 0.0
    pos = period_logrets[np.isfinite(period_logrets) & (period_logrets > 0)]
    if pos.size == 0:
        return 0.0
    pos_sorted = np.sort(pos)[::-1]
    return float(np.sum(pos_sorted[:k]) / denom)


def _sum_pos_share(period_logrets: np.ndarray, denom: float) -> float:
    if denom <= 0:
        return 0.0
    if period_logrets.size == 0:
        return 0.0
    s = float(np.nansum(period_logrets[np.isfinite(period_logrets) & (period_logrets > 0)]))
    return float(s / denom)


# -----------------------------------------------------------------------------
# period segmentation (精準切段：依 kbar_* 的 period_start/period_end)
# -----------------------------------------------------------------------------
def _sum_logret_by_period(
    daily: pd.DataFrame,
    periods: pd.DataFrame,
    cutoff_date: Optional[pd.Timestamp] = None,
) -> np.ndarray:
    """
    daily: columns [date, d_logret]
    periods: columns [period_start, period_end] (datetime)
    cutoff_date: 若提供，只計算 <= cutoff_date 的日K（用於 peak 前）
    回傳每個 period 的 logret 加總 array（長度 = len(periods)）
    """
    if daily.empty or periods.empty:
        return np.array([], dtype=float)

    d = daily[["date", "d_logret"]].copy()
    d["date"] = pd.to_datetime(d["date"], errors="coerce")
    d = d.dropna(subset=["date"]).sort_values("date")

    if cutoff_date is not None and pd.notna(cutoff_date):
        d = d[d["date"] <= cutoff_date]

    out = []
    for _, p in periods.iterrows():
        ps = p["period_start"]
        pe = p["period_end"]
        if pd.isna(ps) or pd.isna(pe):
            out.append(0.0)
            continue
        mask = (d["date"] >= ps) & (d["date"] <= pe)
        out.append(float(d.loc[mask, "d_logret"].sum()))
    return np.array(out, dtype=float)


def _align_peak_trade_date(daily_dates: pd.Series, peak_date: pd.Timestamp) -> Optional[pd.Timestamp]:
    """
    把 peak_date 對齊到 <= peak_date 的最後交易日
    """
    if peak_date is None or pd.isna(peak_date):
        return None
    dd = pd.to_datetime(daily_dates, errors="coerce").dropna().sort_values()
    if dd.empty:
        return None
    dd2 = dd[dd <= peak_date]
    if dd2.empty:
        return None
    return pd.Timestamp(dd2.iloc[-1])


# -----------------------------------------------------------------------------
# core
# -----------------------------------------------------------------------------
def build_contribution_tables(db_path: str, only_markets: Optional[set] = None) -> Dict[str, int]:
    conn = sqlite3.connect(db_path, timeout=SQLITE_TIMEOUT)

    try:
        existing = set(pd.read_sql("SELECT name FROM sqlite_master WHERE type='table'", conn)["name"].tolist())
        required = ["kbar_yearly", "kbar_monthly", "kbar_weekly", "stock_analysis"]
        missing = [t for t in required if t not in existing]
        if missing:
            raise RuntimeError(f"缺少必要表：{missing}\n請先跑 processor.py 與 kbar_aggregator.py")

        # 年K（kbar_yearly 建議已經把 year_peak_date/year_peak_high 算好）
        y = pd.read_sql(
            """
            SELECT symbol, year,
                   period_start, period_end,
                   open AS y_open, close AS y_close,
                   year_peak_date, year_peak_high
            FROM kbar_yearly
            """,
            conn,
        )
        if y.empty:
            print("❌ kbar_yearly 無資料")
            return {"year_rows": 0, "bin_rows": 0}

        # 市場過濾（可選）
        if only_markets:
            if "stock_info" not in existing:
                raise RuntimeError("你傳了 only_markets 但 DB 沒有 stock_info 表，無法過濾市場")
            info = pd.read_sql("SELECT symbol, market FROM stock_info", conn)
            y = y.merge(info, on="symbol", how="left")
            y = y[y["market"].str.lower().isin(set([m.lower() for m in only_markets]))].copy()
            y = y.drop(columns=["market"], errors="ignore")
            if y.empty:
                print("❌ only_markets 過濾後 kbar_yearly 無資料")
                return {"year_rows": 0, "bin_rows": 0}

        y["period_start"] = pd.to_datetime(y["period_start"], errors="coerce")
        y["period_end"] = pd.to_datetime(y["period_end"], errors="coerce")
        y["year_peak_date"] = pd.to_datetime(y["year_peak_date"], errors="coerce")

        # 年報酬 / 年log
        y["year_ret_pct"] = (y["y_close"].astype(float) / y["y_open"].astype(float) - 1.0) * 100.0
        y["year_logret"] = np.log(y["y_close"].astype(float) / y["y_open"].astype(float)).replace(
            [np.inf, -np.inf], np.nan
        ).fillna(0.0)

        # 年K分箱
        y["year_ret_bin_100"] = y["year_ret_pct"].apply(_bin_year_ret_100)
        y["year_ret_bin_10_under100"] = y["year_ret_pct"].apply(_bin_year_ret_10_under100)
        y["year_ret_bin_50_under100"] = y["year_ret_pct"].apply(_bin_year_ret_50_under100)

        # 週/月 periods（用 period_start/period_end 切段）
        wk = pd.read_sql(
            """
            SELECT symbol, year, week_id, period_start, period_end
            FROM kbar_weekly
            """,
            conn,
        )
        mk = pd.read_sql(
            """
            SELECT symbol, year, month_id, period_start, period_end
            FROM kbar_monthly
            """,
            conn,
        )
        wk["period_start"] = pd.to_datetime(wk["period_start"], errors="coerce")
        wk["period_end"] = pd.to_datetime(wk["period_end"], errors="coerce")
        mk["period_start"] = pd.to_datetime(mk["period_start"], errors="coerce")
        mk["period_end"] = pd.to_datetime(mk["period_end"], errors="coerce")

        # 日K（用 stock_analysis，因為你後面會做異常清洗/補值，最好沿用同一層）
        sa = pd.read_sql(
            """
            SELECT symbol, date, open, close, prev_close, is_limit_up
            FROM stock_analysis
            """,
            conn,
        )
        sa["date"] = pd.to_datetime(sa["date"], errors="coerce")
        sa = sa.dropna(subset=["date"]).sort_values(["symbol", "date"])
        sa["year"] = sa["date"].dt.year.astype(int)
        sa["is_limit_up"] = pd.to_numeric(sa["is_limit_up"], errors="coerce").fillna(0).astype(int)

        # 建「逐日 logret」：
        # - 第一天用 log(close/open)（避免跨年 prev_close 影響）
        # - 其餘用 log(close / prev_close)
        sa["d_logret"] = 0.0
        # close/prev_close
        mask_cp = (sa["close"].astype(float) > 0) & (sa["prev_close"].astype(float) > 0)
        sa.loc[mask_cp, "d_logret"] = np.log(sa.loc[mask_cp, "close"].astype(float) / sa.loc[mask_cp, "prev_close"].astype(float))

        # 把每個 symbol-year 的第一筆替換為 log(close/open)
        sa["rank_in_year"] = sa.groupby(["symbol", "year"]).cumcount()
        mask_first = sa["rank_in_year"] == 0
        mask_oc = mask_first & (sa["close"].astype(float) > 0) & (sa["open"].astype(float) > 0)
        sa.loc[mask_oc, "d_logret"] = np.log(sa.loc[mask_oc, "close"].astype(float) / sa.loc[mask_oc, "open"].astype(float))

        # 主迴圈：每個 symbol-year 一筆
        rows = []
        for _, r in y.iterrows():
            sym = r["symbol"]
            yr = int(r["year"])
            year_open = float(r["y_open"])
            year_close = float(r["y_close"])
            year_logret = float(r["year_logret"])
            denom = year_logret if year_logret > 0 else 0.0

            # 取該年日K
            d = sa[(sa["symbol"] == sym) & (sa["year"] == yr)].copy()
            if d.empty:
                continue
            d = d.sort_values("date")

            # peak_date 對齊到最後交易日
            peak_date_raw = r["year_peak_date"]
            peak_trade_date = _align_peak_trade_date(d["date"], peak_date_raw) if pd.notna(peak_date_raw) else None

            # peak_logret_from_open（用「年開盤」→「peak_trade_date 當天 close」）
            peak_close = np.nan
            if peak_trade_date is not None:
                d_peak = d[d["date"] == peak_trade_date]
                if not d_peak.empty:
                    peak_close = float(d_peak.iloc[-1]["close"])
            peak_logret = _safe_log_ratio(peak_close, year_open) if np.isfinite(peak_close) else 0.0

            # 週/月 periods
            wps = wk[(wk["symbol"] == sym) & (wk["year"] == yr)].sort_values("period_end")
            mps = mk[(mk["symbol"] == sym) & (mk["year"] == yr)].sort_values("period_end")

            # 精準週/月：用日K切段加總（全年度）
            w_logrets = _sum_logret_by_period(daily=d, periods=wps, cutoff_date=None)
            m_logrets = _sum_logret_by_period(daily=d, periods=mps, cutoff_date=None)

            top1_week_share = _topk_share(w_logrets, denom, 1)
            top4_weeks_share = _topk_share(w_logrets, denom, 4)
            sum_pos_week_share = _sum_pos_share(w_logrets, denom)

            top1_month_share = _topk_share(m_logrets, denom, 1)
            top3_months_share = _topk_share(m_logrets, denom, 3)
            sum_pos_month_share = _sum_pos_share(m_logrets, denom)

            # peak 前（精準）：直接用日K到 peak_trade_date 的 logret
            logret_to_peak = 0.0
            if peak_trade_date is not None:
                logret_to_peak = float(d.loc[d["date"] <= peak_trade_date, "d_logret"].sum())

            share_year_to_peak = float(logret_to_peak / denom) if denom > 0 else 0.0
            # 也給一個「相對 peak」的版本（更貼近你想問：高點前漲停/週月貢獻到高點）
            denom_peak = peak_logret if peak_logret > 0 else 0.0
            share_peak_to_peak = float(logret_to_peak / denom_peak) if denom_peak > 0 else 0.0  # 理論上接近 1（但資料/對齊可能有誤差）

            # peak 前週/月分段 logret 加總（到 peak 為止）
            w_logrets_to_peak = _sum_logret_by_period(daily=d, periods=wps, cutoff_date=peak_trade_date)
            m_logrets_to_peak = _sum_logret_by_period(daily=d, periods=mps, cutoff_date=peak_trade_date)
            week_log_share_to_peak_vs_year = float(np.nansum(w_logrets_to_peak[w_logrets_to_peak > 0]) / denom) if denom > 0 else 0.0
            month_log_share_to_peak_vs_year = float(np.nansum(m_logrets_to_peak[m_logrets_to_peak > 0]) / denom) if denom > 0 else 0.0
            week_log_share_to_peak_vs_peak = float(np.nansum(w_logrets_to_peak[w_logrets_to_peak > 0]) / denom_peak) if denom_peak > 0 else 0.0
            month_log_share_to_peak_vs_peak = float(np.nansum(m_logrets_to_peak[m_logrets_to_peak > 0]) / denom_peak) if denom_peak > 0 else 0.0

            # peak 前漲停貢獻
            if peak_trade_date is not None:
                d_to_peak = d[d["date"] <= peak_trade_date]
            else:
                d_to_peak = d

            limitup_count_to_peak = int((d_to_peak["is_limit_up"] == 1).sum())
            limitup_log_sum_to_peak = float(d_to_peak.loc[d_to_peak["is_limit_up"] == 1, "d_logret"].sum())
            limitup_log_share_to_peak_vs_year = float(limitup_log_sum_to_peak / denom) if denom > 0 else 0.0
            limitup_log_share_to_peak_vs_peak = float(limitup_log_sum_to_peak / denom_peak) if denom_peak > 0 else 0.0

            rows.append(
                {
                    "symbol": sym,
                    "year": yr,

                    # 年K
                    "y_open": year_open,
                    "y_close": year_close,
                    "year_ret_pct": float(r["year_ret_pct"]),
                    "year_logret": year_logret,

                    # 分箱
                    "year_ret_bin_100": r["year_ret_bin_100"],
                    "year_ret_bin_10_under100": r["year_ret_bin_10_under100"],
                    "year_ret_bin_50_under100": r["year_ret_bin_50_under100"],

                    # peak
                    "year_peak_date_raw": peak_date_raw.strftime("%Y-%m-%d") if pd.notna(peak_date_raw) else None,
                    "year_peak_trade_date": peak_trade_date.strftime("%Y-%m-%d") if peak_trade_date is not None else None,
                    "year_peak_high": float(r["year_peak_high"]) if np.isfinite(r["year_peak_high"]) else np.nan,
                    "peak_close_aligned": peak_close if np.isfinite(peak_close) else np.nan,
                    "peak_logret_from_open": peak_logret,

                    # 週/月集中度（全年度，精準切段）
                    "top1_week_share": top1_week_share,
                    "top4_weeks_share": top4_weeks_share,
                    "sum_pos_week_share": sum_pos_week_share,

                    "top1_month_share": top1_month_share,
                    "top3_months_share": top3_months_share,
                    "sum_pos_month_share": sum_pos_month_share,

                    # peak 前完成度（用日K精準）
                    "logret_to_peak": logret_to_peak,
                    "share_year_to_peak": share_year_to_peak,
                    "share_peak_to_peak": share_peak_to_peak,

                    # peak 前週/月貢獻（精準切段：加總到 peak）
                    "week_pos_log_share_to_peak_vs_year": week_log_share_to_peak_vs_year,
                    "month_pos_log_share_to_peak_vs_year": month_log_share_to_peak_vs_year,
                    "week_pos_log_share_to_peak_vs_peak": week_log_share_to_peak_vs_peak,
                    "month_pos_log_share_to_peak_vs_peak": month_log_share_to_peak_vs_peak,

                    # 漲停貢獻（peak 前）
                    "limitup_count_to_peak": limitup_count_to_peak,
                    "limitup_log_sum_to_peak": limitup_log_sum_to_peak,
                    "limitup_log_share_to_peak_vs_year": limitup_log_share_to_peak_vs_year,
                    "limitup_log_share_to_peak_vs_peak": limitup_log_share_to_peak_vs_peak,
                }
            )

        out = pd.DataFrame(rows)
        if out.empty:
            print("❌ year_contribution 無資料（可能 year/kbar 對不起來或 stock_analysis 缺日K）")
            return {"year_rows": 0, "bin_rows": 0}

        # 寫回 year_contribution（重建）
        conn.execute("DROP TABLE IF EXISTS year_contribution")
        out.to_sql("year_contribution", conn, if_exists="replace", index=False)

        try:
            conn.execute("CREATE INDEX IF NOT EXISTS idx_year_contrib_symbol_year ON year_contribution(symbol, year)")
            conn.execute("CREATE INDEX IF NOT EXISTS idx_year_contrib_bin100 ON year_contribution(year_ret_bin_100)")
        except Exception:
            pass

        # bins summary（以 100% 分箱為主）
        def _agg(df: pd.DataFrame) -> pd.Series:
            return pd.Series(
                {
                    "n": int(len(df)),
                    "avg_year_ret_pct": float(df["year_ret_pct"].mean()),
                    "median_year_ret_pct": float(df["year_ret_pct"].median()),

                    "avg_top1_week_share": float(df["top1_week_share"].mean()),
                    "avg_top1_month_share": float(df["top1_month_share"].mean()),
                    "avg_share_year_to_peak": float(df["share_year_to_peak"].mean()),

                    "avg_limitup_count_to_peak": float(df["limitup_count_to_peak"].mean()),
                    "avg_limitup_log_share_to_peak_vs_year": float(df["limitup_log_share_to_peak_vs_year"].mean()),
                    "avg_limitup_log_share_to_peak_vs_peak": float(df["limitup_log_share_to_peak_vs_peak"].mean()),

                    # 你儀表板常會想看的門檻比例（可自行調）
                    "pct_top1_week_ge_0_4": float((df["top1_week_share"] >= 0.4).mean() * 100),
                    "pct_top1_month_ge_0_4": float((df["top1_month_share"] >= 0.4).mean() * 100),
                    "pct_limitup_share_year_ge_0_4": float((df["limitup_log_share_to_peak_vs_year"] >= 0.4).mean() * 100),
                }
            )

        bins = out.groupby("year_ret_bin_100", sort=False).apply(_agg).reset_index()
        conn.execute("DROP TABLE IF EXISTS year_contribution_bins")
        bins.to_sql("year_contribution_bins", conn, if_exists="replace", index=False)

        try:
            conn.execute("CREATE INDEX IF NOT EXISTS idx_year_contrib_bins_bin ON year_contribution_bins(year_ret_bin_100)")
        except Exception:
            pass

        conn.commit()

        print("\n✅ kbar_contribution（精準切段版）完成：")
        print(f"📌 year_contribution rows: {len(out):,}")
        print(f"📌 year_contribution_bins rows: {len(bins):,}")
        print("📌 週/月集中度：top1_week_share / top1_month_share")
        print("📌 peak 前完成度：share_year_to_peak（到年高點已完成全年漲幅比例）")
        print("📌 漲停貢獻：limitup_log_share_to_peak_vs_year / vs_peak")

        return {"year_rows": int(len(out)), "bin_rows": int(len(bins))}

    finally:
        conn.close()


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python kbar_contribution.py <db_path>")
        sys.exit(1)

    db = sys.argv[1]
    build_contribution_tables(db_path=db)
