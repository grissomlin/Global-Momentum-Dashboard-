# kbar_contribution.py
# -*- coding: utf-8 -*-
"""
kbar_contribution.py
--------------------
目的：做你要的「年K分箱 + 週/月/漲停對年K的貢獻度（含集中/緩漲判斷）」的資料層，
寫回 SQLite 供儀表板直接查。

✅ 依賴（建議先跑完）：
1) processor.py  -> stock_analysis（含 is_limit_up, lu_type, consecutive_limits...）
2) kbar_aggregator.py -> kbar_weekly / kbar_monthly / kbar_yearly
3) (可選) event_engine.py -> limit_up_events / daytrade_events（若你之後要用事件表做更細的統計）

✅ 這支會產生（寫回 DB）：
- year_contribution：每一檔每一年一筆（最重要）
  包含：
  - year_ret% / year_logret
  - 年K分箱：100%間隔(0~1000+)、0~100%內再10%細分、以及0-50/50-100
  - 週/月集中度：top1 week / top4 weeks / top1 month / top3 months 的 log-return 佔年log-return比例
  - 漲停貢獻：peak_date 前漲停根數、peak_date 前漲停log貢獻占比（用 log-return 加總）
  - peak_date 前「週/月」貢獻（可用來回答：飆股是不是在 peak 前就漲完）
- year_contribution_bins：依年K分箱彙總（平均/中位數/樣本數）

📌 你要研究的主題（這張表直接支援）：
- 飆股是否「集中在一週/一個月就漲完」：
  看 top1_week_share / top1_month_share 是否很高（例如 >0.4、>0.6）
- 漲停板是否是主要貢獻來源：
  看 limitup_log_share_to_peak、limitup_count_to_peak
- 週/月對年K高點的貢獻：
  看 week_log_share_to_peak / month_log_share_to_peak

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
# helpers
# -----------------------------------------------------------------------------
def _safe_logret(p0: float, p1: float) -> float:
    if p0 is None or p1 is None:
        return 0.0
    if not np.isfinite(p0) or not np.isfinite(p1):
        return 0.0
    if p0 <= 0 or p1 <= 0:
        return 0.0
    return float(np.log(p1 / p0))


def _bin_year_ret_100(ret_pct: float) -> str:
    """
    年K分箱：0~1000 每100 + >1000
    注意：只對「正報酬」做 0~1000；負報酬另外獨立
    """
    if not np.isfinite(ret_pct):
        return "NA"
    if ret_pct < 0:
        return "NEGATIVE"
    if ret_pct >= 1000:
        return "1000UP"
    # 0~999.999
    lo = int(ret_pct // 100) * 100
    hi = lo + 100
    return f"{lo:04d}-{hi:04d}"


def _bin_year_ret_10_under100(ret_pct: float) -> str:
    """
    0~100% 內再細分 10% 一格，其餘回傳 OTHER
    """
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
    """
    0-50 / 50-100 / >=100 / negative
    """
    if not np.isfinite(ret_pct):
        return "NA"
    if ret_pct < 0:
        return "NEGATIVE"
    if ret_pct < 50:
        return "00-50"
    if ret_pct < 100:
        return "50-100"
    return "100UP"


def _topk_share(logrets: np.ndarray, denom: float, k: int) -> float:
    """
    top-k 正logret 佔比（用 log-return 計算，避免百分比加總偏誤）
    denom <=0 時回 0
    """
    if denom <= 0:
        return 0.0
    if logrets.size == 0:
        return 0.0
    pos = logrets[np.isfinite(logrets) & (logrets > 0)]
    if pos.size == 0:
        return 0.0
    pos_sorted = np.sort(pos)[::-1]
    return float(np.sum(pos_sorted[:k]) / denom)


def _sum_share(logrets: np.ndarray, denom: float) -> float:
    if denom <= 0:
        return 0.0
    if logrets.size == 0:
        return 0.0
    s = float(np.nansum(logrets[np.isfinite(logrets) & (logrets > 0)]))
    return float(s / denom)


# -----------------------------------------------------------------------------
# core
# -----------------------------------------------------------------------------
def build_contribution_tables(db_path: str, only_markets: Optional[set] = None) -> Dict[str, int]:
    """
    讀取：
      - kbar_yearly / kbar_monthly / kbar_weekly
      - stock_analysis（用 is_limit_up + daily logret 做漲停貢獻）
      - stock_prices（用來抓 peak_date 的 close，避免 peak_date 落在非交易日/或缺漏）
    產出：
      - year_contribution
      - year_contribution_bins
    """
    conn = sqlite3.connect(db_path, timeout=SQLITE_TIMEOUT)

    try:
        # ---- 檢查必要表 ----
        required = ["kbar_yearly", "kbar_monthly", "kbar_weekly", "stock_analysis", "stock_prices"]
        existing = set(pd.read_sql("SELECT name FROM sqlite_master WHERE type='table'", conn)["name"].tolist())
        missing = [t for t in required if t not in existing]
        if missing:
            raise RuntimeError(
                f"缺少必要表：{missing}\n"
                f"請先跑：processor.py（stock_analysis）與 kbar_aggregator.py（kbar_*）"
            )

        # ---- 讀取年K ----
        y = pd.read_sql(
            """
            SELECT symbol, year, period_start, period_end,
                   open AS y_open, close AS y_close, high AS y_high,
                   year_peak_date, year_peak_high
            FROM kbar_yearly
            """,
            conn,
        )
        if y.empty:
            print("❌ kbar_yearly 無資料")
            return {"year_rows": 0, "bin_rows": 0}

        # 若你只想做 tw/cn/jp，可傳 only_markets 並靠 stock_info 過濾
        if only_markets:
            info = pd.read_sql("SELECT symbol, market FROM stock_info", conn)
            y = y.merge(info, on="symbol", how="left")
            y = y[y["market"].str.lower().isin(set([m.lower() for m in only_markets]))].copy()
            y = y.drop(columns=["market"], errors="ignore")
            if y.empty:
                print("❌ 依 only_markets 過濾後 kbar_yearly 無資料")
                return {"year_rows": 0, "bin_rows": 0}

        # 日期
        y["period_start"] = pd.to_datetime(y["period_start"], errors="coerce")
        y["period_end"] = pd.to_datetime(y["period_end"], errors="coerce")
        y["year_peak_date"] = pd.to_datetime(y["year_peak_date"], errors="coerce")

        # 年報酬
        y["year_ret_pct"] = (y["y_close"].astype(float) / y["y_open"].astype(float) - 1.0) * 100.0
        y["year_logret"] = np.log(y["y_close"].astype(float) / y["y_open"].astype(float)).replace([np.inf, -np.inf], np.nan).fillna(0.0)

        # 年K分箱
        y["year_ret_bin_100"] = y["year_ret_pct"].apply(_bin_year_ret_100)
        y["year_ret_bin_10_under100"] = y["year_ret_pct"].apply(_bin_year_ret_10_under100)
        y["year_ret_bin_50_under100"] = y["year_ret_pct"].apply(_bin_year_ret_50_under100)

        # ---- 讀 weekly/monthly ----
        w = pd.read_sql(
            """
            SELECT symbol, year, week_id, period_start, period_end, close AS w_close
            FROM kbar_weekly
            """,
            conn,
        )
        w["period_end"] = pd.to_datetime(w["period_end"], errors="coerce")
        w["year"] = w["year"].astype(int)

        m = pd.read_sql(
            """
            SELECT symbol, year, month_id, period_start, period_end, close AS m_close
            FROM kbar_monthly
            """,
            conn,
        )
        m["period_end"] = pd.to_datetime(m["period_end"], errors="coerce")
        m["year"] = m["year"].astype(int)

        # ---- stock_analysis：用來算「漲停貢獻」與 peak 前漲停根數 ----
        sa = pd.read_sql(
            """
            SELECT symbol, date, close, prev_close, is_limit_up
            FROM stock_analysis
            """,
            conn,
        )
        sa["date"] = pd.to_datetime(sa["date"], errors="coerce")
        sa = sa.dropna(subset=["date"]).sort_values(["symbol", "date"])
        sa["year"] = sa["date"].dt.year.astype(int)

        # daily logret（用 close/prev_close）
        sa["d_logret"] = 0.0
        mask = (sa["close"].astype(float) > 0) & (sa["prev_close"].astype(float) > 0)
        sa.loc[mask, "d_logret"] = np.log(sa.loc[mask, "close"].astype(float) / sa.loc[mask, "prev_close"].astype(float))
        sa["is_limit_up"] = pd.to_numeric(sa["is_limit_up"], errors="coerce").fillna(0).astype(int)

        # ---- stock_prices：為了抓 peak_date 當天 close（若 peak_date 缺或落在缺口） ----
        sp = pd.read_sql(
            """
            SELECT symbol, date, close
            FROM stock_prices
            """,
            conn,
        )
        sp["date"] = pd.to_datetime(sp["date"], errors="coerce")
        sp = sp.dropna(subset=["date"]).sort_values(["symbol", "date"])
        sp["year"] = sp["date"].dt.year.astype(int)

        # ---- per symbol-year 計算 ----
        rows = []
        # 用 merge keys 迭代年表
        for _, r in y.iterrows():
            sym = r["symbol"]
            yr = int(r["year"])
            y_open = float(r["y_open"])
            y_close = float(r["y_close"])
            peak_date = r["year_peak_date"]

            # 年log denom
            denom = float(r["year_logret"])
            denom_pos = denom if denom > 0 else 0.0

            # --- weekly logrets within year ---
            wg = w[(w["symbol"] == sym) & (w["year"] == yr)].sort_values("period_end")
            w_close = wg["w_close"].astype(float).values
            w_log = np.array([])
            if w_close.size >= 2:
                w_log = np.log(w_close[1:] / w_close[:-1])
                w_log = w_log[np.isfinite(w_log)]
            top1_week_share = _topk_share(w_log, denom_pos, 1)
            top4_weeks_share = _topk_share(w_log, denom_pos, 4)
            sum_pos_week_share = _sum_share(w_log, denom_pos)

            # --- monthly logrets within year ---
            mg = m[(m["symbol"] == sym) & (m["year"] == yr)].sort_values("period_end")
            m_close = mg["m_close"].astype(float).values
            m_log = np.array([])
            if m_close.size >= 2:
                m_log = np.log(m_close[1:] / m_close[:-1])
                m_log = m_log[np.isfinite(m_log)]
            top1_month_share = _topk_share(m_log, denom_pos, 1)
            top3_months_share = _topk_share(m_log, denom_pos, 3)
            sum_pos_month_share = _sum_share(m_log, denom_pos)

            # --- peak_date 對齊：找 <= peak_date 最近交易日 close ---
            peak_close = np.nan
            if pd.notna(peak_date):
                spt = sp[(sp["symbol"] == sym) & (sp["year"] == yr) & (sp["date"] <= peak_date)]
                if not spt.empty:
                    peak_close = float(spt.iloc[-1]["close"])
            # peak_logret（year_open → peak_close）
            peak_logret = _safe_logret(y_open, peak_close) if np.isfinite(peak_close) else 0.0

            # --- peak 前 week/month log share ---
            week_log_to_peak = 0.0
            if denom_pos > 0 and pd.notna(peak_date) and not wg.empty:
                # 取 period_end <= peak_date 的週，算「年初→該週收盤」logret
                # 用週收盤序列近似：log( last_w_close / first_w_close ) 再加上年初->第一週 close 的誤差
                # 更精準可改用日K，但這裡先用可跑版本
                w_end = wg[wg["period_end"] <= peak_date]
                if len(w_end) >= 1:
                    w_last = float(w_end.iloc[-1]["w_close"])
                    week_log_to_peak = _safe_logret(y_open, w_last)
            week_log_share_to_peak = float(week_log_to_peak / denom_pos) if denom_pos > 0 else 0.0

            month_log_to_peak = 0.0
            if denom_pos > 0 and pd.notna(peak_date) and not mg.empty:
                m_end = mg[mg["period_end"] <= peak_date]
                if len(m_end) >= 1:
                    m_last = float(m_end.iloc[-1]["m_close"])
                    month_log_to_peak = _safe_logret(y_open, m_last)
            month_log_share_to_peak = float(month_log_to_peak / denom_pos) if denom_pos > 0 else 0.0

            # --- limit up count/log contribution to peak ---
            lug = sa[(sa["symbol"] == sym) & (sa["year"] == yr)]
            if pd.notna(peak_date):
                lug_to_peak = lug[lug["date"] <= peak_date]
            else:
                lug_to_peak = lug

            limitup_count_to_peak = int((lug_to_peak["is_limit_up"] == 1).sum())
            limitup_log_sum_to_peak = float(lug_to_peak.loc[lug_to_peak["is_limit_up"] == 1, "d_logret"].sum())
            limitup_log_share_to_peak = float(limitup_log_sum_to_peak / denom_pos) if denom_pos > 0 else 0.0

            rows.append(
                {
                    "symbol": sym,
                    "year": yr,

                    # 年K
                    "y_open": y_open,
                    "y_close": y_close,
                    "year_ret_pct": float(r["year_ret_pct"]),
                    "year_logret": denom,

                    # 年分箱
                    "year_ret_bin_100": r["year_ret_bin_100"],
                    "year_ret_bin_10_under100": r["year_ret_bin_10_under100"],
                    "year_ret_bin_50_under100": r["year_ret_bin_50_under100"],

                    # peak
                    "year_peak_date": peak_date.strftime("%Y-%m-%d") if pd.notna(peak_date) else None,
                    "year_peak_high": float(r["year_peak_high"]) if np.isfinite(r["year_peak_high"]) else np.nan,
                    "peak_close_aligned": peak_close if np.isfinite(peak_close) else np.nan,
                    "peak_logret_from_open": peak_logret,

                    # 週/月集中度（回答：集中一週/一月 vs 緩漲）
                    "top1_week_share": top1_week_share,
                    "top4_weeks_share": top4_weeks_share,
                    "sum_pos_week_share": sum_pos_week_share,

                    "top1_month_share": top1_month_share,
                    "top3_months_share": top3_months_share,
                    "sum_pos_month_share": sum_pos_month_share,

                    # peak 前週/月已完成度（回答：是否 peak 前就漲完）
                    "week_log_share_to_peak": week_log_share_to_peak,
                    "month_log_share_to_peak": month_log_share_to_peak,

                    # 漲停貢獻（回答：飆股是否靠漲停堆出來）
                    "limitup_count_to_peak": limitup_count_to_peak,
                    "limitup_log_sum_to_peak": limitup_log_sum_to_peak,
                    "limitup_log_share_to_peak": limitup_log_share_to_peak,
                }
            )

        out = pd.DataFrame(rows)
        if out.empty:
            print("❌ year_contribution 無資料（可能 year/kbar 對不起來）")
            return {"year_rows": 0, "bin_rows": 0}

        # ---- 寫回 year_contribution ----
        conn.execute("DROP TABLE IF EXISTS year_contribution")
        out.to_sql("year_contribution", conn, if_exists="replace", index=False)

        try:
            conn.execute("CREATE INDEX IF NOT EXISTS idx_year_contrib_symbol_year ON year_contribution(symbol, year)")
            conn.execute("CREATE INDEX IF NOT EXISTS idx_year_contrib_bin100 ON year_contribution(year_ret_bin_100)")
        except Exception:
            pass

        # ---- bins summary（依 100% 分箱為主；你也可以另外做 under100 的細分彙總） ----
        def _agg_summary(df: pd.DataFrame) -> pd.Series:
            return pd.Series(
                {
                    "n": int(len(df)),
                    "avg_year_ret_pct": float(df["year_ret_pct"].mean()),
                    "median_year_ret_pct": float(df["year_ret_pct"].median()),

                    "avg_top1_week_share": float(df["top1_week_share"].mean()),
                    "avg_top1_month_share": float(df["top1_month_share"].mean()),
                    "avg_limitup_count_to_peak": float(df["limitup_count_to_peak"].mean()),
                    "avg_limitup_log_share_to_peak": float(df["limitup_log_share_to_peak"].mean()),

                    # “集中度門檻” 你儀表板常會想看的比例
                    "pct_top1_week_share_ge_0_4": float((df["top1_week_share"] >= 0.4).mean() * 100),
                    "pct_top1_month_share_ge_0_4": float((df["top1_month_share"] >= 0.4).mean() * 100),
                    "pct_limitup_log_share_ge_0_4": float((df["limitup_log_share_to_peak"] >= 0.4).mean() * 100),
                }
            )

        bins = out.groupby("year_ret_bin_100", sort=False).apply(_agg_summary).reset_index()
        conn.execute("DROP TABLE IF EXISTS year_contribution_bins")
        bins.to_sql("year_contribution_bins", conn, if_exists="replace", index=False)

        try:
            conn.execute("CREATE INDEX IF NOT EXISTS idx_year_contrib_bins_bin ON year_contribution_bins(year_ret_bin_100)")
        except Exception:
            pass

        conn.commit()

        print("\n✅ kbar_contribution 完成：")
        print(f"📌 year_contribution rows: {len(out):,}")
        print(f"📌 year_contribution_bins rows: {len(bins):,}")
        print("📌 你可以直接用 year_contribution.top1_week_share / top1_month_share 判斷『集中 vs 緩漲』")
        print("📌 用 limitup_* 欄位判斷『漲停是否主貢獻』")

        return {"year_rows": int(len(out)), "bin_rows": int(len(bins))}

    finally:
        conn.close()


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python kbar_contribution.py <db_path>")
        sys.exit(1)

    db = sys.argv[1]
    build_contribution_tables(db_path=db)
