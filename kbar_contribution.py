# kbar_contribution.py
# -*- coding: utf-8 -*-
"""
kbar_contribution.py  (精準切段版｜都來｜不刪舊功能，只加)
-----------------------------------------------
依賴：
- kbar_aggregator.py -> kbar_weekly / kbar_monthly / kbar_yearly
- processor.py -> stock_analysis（需 open/close/prev_close/is_limit_up）

輸出：
- year_contribution
- year_contribution_bins

新增（你說「都來」）：
- burst_style_week / burst_style_month：
    * ONE_WEEK_BURST  : top1_week_share_net >= 0.5
    * ONE_MONTH_BURST : top1_month_share_net >= 0.5

本次新增（你要的）：
- 周K/月K 回撤（period 內 close 的 max drawdown log）
- 周K/月K 波動率（該年週/月 logret 的 std）
- 上週 vs 本週（最後兩週）漲跌幅「幅度變化率」
- denom_year
- top_week_id / top_week_logret_net / top_week_is_limitup_dense
- top_week_id_net_to_peak（peak 前 top1 week）
- limitup_count_in_top1week_to_peak
- 月K對應欄位：top_month_id / ... / to_peak / limitup_count_in_top1month_to_peak
"""

import sys
import sqlite3
import numpy as np
import pandas as pd
from typing import Optional, Dict

SQLITE_TIMEOUT = 120
EPS = 1e-12


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


def _topk_share_pos(period_logrets: np.ndarray, denom: float, k: int) -> float:
    if denom <= 0 or period_logrets.size == 0:
        return 0.0
    pos = period_logrets[np.isfinite(period_logrets) & (period_logrets > 0)]
    if pos.size == 0:
        return 0.0
    pos_sorted = np.sort(pos)[::-1]
    return float(np.sum(pos_sorted[:k]) / denom)


def _topk_share_net(period_logrets: np.ndarray, denom: float, k: int) -> float:
    if denom <= 0 or period_logrets.size == 0:
        return 0.0
    v = period_logrets[np.isfinite(period_logrets)]
    if v.size == 0:
        return 0.0
    v_sorted = np.sort(v)[::-1]
    return float(np.sum(v_sorted[:k]) / denom)


def _sum_pos_share(period_logrets: np.ndarray, denom: float) -> float:
    if denom <= 0 or period_logrets.size == 0:
        return 0.0
    s = float(np.nansum(period_logrets[np.isfinite(period_logrets) & (period_logrets > 0)]))
    return float(s / denom)


def _sum_net_share(period_logrets: np.ndarray, denom: float) -> float:
    if denom <= 0 or period_logrets.size == 0:
        return 0.0
    s = float(np.nansum(period_logrets[np.isfinite(period_logrets)]))
    return float(s / denom)


def _max_drawdown_log_from_close(close: pd.Series) -> float:
    c = pd.to_numeric(close, errors="coerce")
    c = c[(c > 0) & np.isfinite(c)]
    if c.empty:
        return 0.0
    logc = np.log(c.values.astype(float))
    run_max = np.maximum.accumulate(logc)
    dd = logc - run_max
    return float(np.min(dd))


def _max_drawdown_log_from_logrets(logrets: np.ndarray) -> float:
    """Max drawdown computed on cumulative log-return series (period-level)."""
    if logrets is None:
        return 0.0
    v = np.array(logrets, dtype=float)
    v = v[np.isfinite(v)]
    if v.size == 0:
        return 0.0
    cum = np.cumsum(v)
    run_max = np.maximum.accumulate(cum)
    dd = cum - run_max
    return float(np.min(dd)) if dd.size else 0.0



def _logret_to_ret_pct(logret: float) -> float:
    if not np.isfinite(logret):
        return np.nan
    return float((np.expm1(logret)) * 100.0)


# -----------------------------------------------------------------------------
# period segmentation (精準切段)
# -----------------------------------------------------------------------------
def _sum_logret_by_period(
    daily: pd.DataFrame,
    periods: pd.DataFrame,
    cutoff_date: Optional[pd.Timestamp] = None,
) -> np.ndarray:
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
# NEW: build per-period stats (week/month) from daily + periods
# -----------------------------------------------------------------------------
def _build_period_stats(
    daily: pd.DataFrame,
    periods: pd.DataFrame,
    *,
    id_col: str,
    cutoff_date: Optional[pd.Timestamp] = None,
) -> pd.DataFrame:
    """
    回傳每個 period 的：
    - id_col (week_id/month_id)
    - period_start/period_end
    - logret_sum (依 d_logret 加總；若 cutoff_date，會截到 cutoff_date)
    - ret_pct
    - max_drawdown_log (period 內 close 的 max drawdown log)
    - limitup_count_dense (period 內 is_limit_up=1 數量；若 cutoff_date，會截到 cutoff_date)
    """
    if daily.empty or periods.empty:
        return pd.DataFrame(columns=[id_col, "period_start", "period_end", "logret_sum", "ret_pct",
                                     "max_drawdown_log", "limitup_count_dense"])

    d = daily[["date", "close", "d_logret", "is_limit_up"]].copy()
    d["date"] = pd.to_datetime(d["date"], errors="coerce")
    d = d.dropna(subset=["date"]).sort_values("date")

    if cutoff_date is not None and pd.notna(cutoff_date):
        d = d[d["date"] <= cutoff_date]

    out_rows = []
    for _, p in periods.iterrows():
        pid = p.get(id_col, None)
        ps = p.get("period_start", None)
        pe = p.get("period_end", None)
        if pd.isna(ps) or pd.isna(pe):
            out_rows.append(
                {id_col: pid, "period_start": ps, "period_end": pe,
                 "logret_sum": 0.0, "ret_pct": 0.0, "max_drawdown_log": 0.0, "limitup_count_dense": 0}
            )
            continue

        mask = (d["date"] >= ps) & (d["date"] <= pe)
        seg = d.loc[mask].copy()
        if seg.empty:
            out_rows.append(
                {id_col: pid, "period_start": ps, "period_end": pe,
                 "logret_sum": 0.0, "ret_pct": 0.0, "max_drawdown_log": 0.0, "limitup_count_dense": 0}
            )
            continue

        logret_sum = float(np.nansum(seg["d_logret"].values.astype(float)))
        out_rows.append(
            {
                id_col: pid,
                "period_start": ps,
                "period_end": pe,
                "logret_sum": logret_sum,
                "ret_pct": _logret_to_ret_pct(logret_sum),
                "max_drawdown_log": _max_drawdown_log_from_close(seg["close"]),
                "limitup_count_dense": int((pd.to_numeric(seg["is_limit_up"], errors="coerce").fillna(0).astype(int) == 1).sum()),
            }
        )

    return pd.DataFrame(out_rows)


def _last_two_abs_change_rate(a: float, b: float) -> float:
    """
    你要的「上期 vs 本期 漲跌幅幅度變化率」：
    (|b| - |a|) / (|a| + eps)
    """
    if not np.isfinite(a) or not np.isfinite(b):
        return 0.0
    denom = abs(a) + EPS
    return float((abs(b) - abs(a)) / denom)


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

        # 年K
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

        # 年報酬
        y["year_ret_pct"] = (y["y_close"].astype(float) / y["y_open"].astype(float) - 1.0) * 100.0
        y["year_logret"] = np.log(y["y_close"].astype(float) / y["y_open"].astype(float)).replace(
            [np.inf, -np.inf], np.nan
        ).fillna(0.0)

        # 年K分箱
        y["year_ret_bin_100"] = y["year_ret_pct"].apply(_bin_year_ret_100)
        y["year_ret_bin_10_under100"] = y["year_ret_pct"].apply(_bin_year_ret_10_under100)
        y["year_ret_bin_50_under100"] = y["year_ret_pct"].apply(_bin_year_ret_50_under100)

        # 週/月 periods
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

        # 日K（stock_analysis）
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

        # 逐日 logret：
        sa["d_logret"] = 0.0
        mask_cp = (sa["close"].astype(float) > 0) & (sa["prev_close"].astype(float) > 0)
        sa.loc[mask_cp, "d_logret"] = np.log(
            sa.loc[mask_cp, "close"].astype(float) / sa.loc[mask_cp, "prev_close"].astype(float)
        )

        sa["rank_in_year"] = sa.groupby(["symbol", "year"]).cumcount()
        mask_first = sa["rank_in_year"] == 0
        mask_oc = mask_first & (sa["close"].astype(float) > 0) & (sa["open"].astype(float) > 0)
        sa.loc[mask_oc, "d_logret"] = np.log(sa.loc[mask_oc, "close"].astype(float) / sa.loc[mask_oc, "open"].astype(float))

        rows = []
        for _, r in y.iterrows():
            sym = r["symbol"]
            yr = int(r["year"])
            year_open = float(r["y_open"])
            year_close = float(r["y_close"])
            year_logret = float(r["year_logret"])
            denom_year = year_logret if year_logret > 0 else 0.0

            d = sa[(sa["symbol"] == sym) & (sa["year"] == yr)].copy()
            if d.empty:
                continue
            d = d.sort_values("date")

            # 年內最大回撤
            year_max_dd_log = _max_drawdown_log_from_close(d["close"])

            # peak date 對齊
            peak_date_raw = r["year_peak_date"]
            peak_trade_date = _align_peak_trade_date(d["date"], peak_date_raw) if pd.notna(peak_date_raw) else None

            peak_close = np.nan
            if peak_trade_date is not None:
                d_peak = d[d["date"] == peak_trade_date]
                if not d_peak.empty:
                    peak_close = float(d_peak.iloc[-1]["close"])

            peak_logret = _safe_log_ratio(peak_close, year_open) if np.isfinite(peak_close) else 0.0
            denom_peak = peak_logret if peak_logret > 0 else 0.0

            # 從 peak 到年末回撤
            peak_to_year_end_dd_log = 0.0
            if peak_trade_date is not None and np.isfinite(peak_close) and peak_close > 0:
                year_end_close = float(d.iloc[-1]["close"]) if np.isfinite(d.iloc[-1]["close"]) else np.nan
                peak_to_year_end_dd_log = _safe_log_ratio(year_end_close, peak_close) if np.isfinite(year_end_close) else 0.0

            # 週/月 periods
            wps = wk[(wk["symbol"] == sym) & (wk["year"] == yr)].sort_values("period_end")
            mps = mk[(mk["symbol"] == sym) & (mk["year"] == yr)].sort_values("period_end")

            # 精準切段：日K切週/月（原本 array）
            w_logrets = _sum_logret_by_period(daily=d, periods=wps, cutoff_date=None)
            m_logrets = _sum_logret_by_period(daily=d, periods=mps, cutoff_date=None)

            worst_week_logret = float(np.nanmin(w_logrets)) if w_logrets.size > 0 else 0.0
            worst_month_logret = float(np.nanmin(m_logrets)) if m_logrets.size > 0 else 0.0

            # ========== NEW: 週/月 period stats（含回撤、limitup 密度） ==========
            w_stats = _build_period_stats(daily=d, periods=wps, id_col="week_id", cutoff_date=None)
            m_stats = _build_period_stats(daily=d, periods=mps, id_col="month_id", cutoff_date=None)

            # 週/月 波動率（logret std）
            week_vol_logret_std = float(np.nanstd(w_stats["logret_sum"].values.astype(float))) if len(w_stats) > 0 else 0.0
            month_vol_logret_std = float(np.nanstd(m_stats["logret_sum"].values.astype(float))) if len(m_stats) > 0 else 0.0

            # 上週 vs 本週（最後兩週）幅度變化率（用 ret_pct）
            last_week_ret_pct = float(w_stats["ret_pct"].iloc[-1]) if len(w_stats) >= 1 else np.nan
            prev_week_ret_pct = float(w_stats["ret_pct"].iloc[-2]) if len(w_stats) >= 2 else np.nan
            week_abs_ret_change_rate = _last_two_abs_change_rate(prev_week_ret_pct, last_week_ret_pct) if len(w_stats) >= 2 else 0.0

            # 上月 vs 本月（最後兩月）
            last_month_ret_pct = float(m_stats["ret_pct"].iloc[-1]) if len(m_stats) >= 1 else np.nan
            prev_month_ret_pct = float(m_stats["ret_pct"].iloc[-2]) if len(m_stats) >= 2 else np.nan
            month_abs_ret_change_rate = _last_two_abs_change_rate(prev_month_ret_pct, last_month_ret_pct) if len(m_stats) >= 2 else 0.0

            # top1 week/month（全年 net 最大 logret）
            top_week_id = None
            top_week_logret_net = 0.0
            top_week_is_limitup_dense = 0
            top_week_max_drawdown_log = 0.0

            if len(w_stats) > 0:
                idx = int(np.nanargmax(w_stats["logret_sum"].values.astype(float)))
                top_week_id = w_stats["week_id"].iloc[idx]
                top_week_logret_net = float(w_stats["logret_sum"].iloc[idx])
                top_week_is_limitup_dense = int(w_stats["limitup_count_dense"].iloc[idx])
                top_week_max_drawdown_log = float(w_stats["max_drawdown_log"].iloc[idx])

            top_month_id = None
            top_month_logret_net = 0.0
            top_month_is_limitup_dense = 0
            top_month_max_drawdown_log = 0.0

            if len(m_stats) > 0:
                idxm = int(np.nanargmax(m_stats["logret_sum"].values.astype(float)))
                top_month_id = m_stats["month_id"].iloc[idxm]
                top_month_logret_net = float(m_stats["logret_sum"].iloc[idxm])
                top_month_is_limitup_dense = int(m_stats["limitup_count_dense"].iloc[idxm])
                top_month_max_drawdown_log = float(m_stats["max_drawdown_log"].iloc[idxm])

            # 集中度（POS/NET）
            top1_week_share_pos = _topk_share_pos(w_logrets, denom_year, 1)
            top4_weeks_share_pos = _topk_share_pos(w_logrets, denom_year, 4)
            top1_week_share_net = _topk_share_net(w_logrets, denom_year, 1)
            top4_weeks_share_net = _topk_share_net(w_logrets, denom_year, 4)

            top1_month_share_pos = _topk_share_pos(m_logrets, denom_year, 1)
            top3_months_share_pos = _topk_share_pos(m_logrets, denom_year, 3)
            top1_month_share_net = _topk_share_net(m_logrets, denom_year, 1)
            top3_months_share_net = _topk_share_net(m_logrets, denom_year, 3)

            sum_pos_week_share = _sum_pos_share(w_logrets, denom_year)
            sum_net_week_share = _sum_net_share(w_logrets, denom_year)
            sum_pos_month_share = _sum_pos_share(m_logrets, denom_year)
            sum_net_month_share = _sum_net_share(m_logrets, denom_year)

            # ✅ burst label（你說都來）
            burst_style_week = "ONE_WEEK_BURST" if top1_week_share_net >= 0.5 else "NON_CONCENTRATED"
            burst_style_month = "ONE_MONTH_BURST" if top1_month_share_net >= 0.5 else "NON_CONCENTRATED"

            # peak 前：日K到 peak
            logret_to_peak = 0.0
            if peak_trade_date is not None:
                logret_to_peak = float(d.loc[d["date"] <= peak_trade_date, "d_logret"].sum())

            share_year_to_peak = float(logret_to_peak / denom_year) if denom_year > 0 else 0.0
            share_peak_to_peak = float(logret_to_peak / denom_peak) if denom_peak > 0 else 0.0

            # peak 前：週/月切段到 peak（array）
            w_logrets_to_peak = _sum_logret_by_period(daily=d, periods=wps, cutoff_date=peak_trade_date)
            m_logrets_to_peak = _sum_logret_by_period(daily=d, periods=mps, cutoff_date=peak_trade_date)

            week_pos_log_share_to_peak_vs_year = float(np.nansum(w_logrets_to_peak[w_logrets_to_peak > 0]) / denom_year) if denom_year > 0 else 0.0
            week_net_log_share_to_peak_vs_year = float(np.nansum(w_logrets_to_peak[np.isfinite(w_logrets_to_peak)]) / denom_year) if denom_year > 0 else 0.0
            month_pos_log_share_to_peak_vs_year = float(np.nansum(m_logrets_to_peak[m_logrets_to_peak > 0]) / denom_year) if denom_year > 0 else 0.0
            month_net_log_share_to_peak_vs_year = float(np.nansum(m_logrets_to_peak[np.isfinite(m_logrets_to_peak)]) / denom_year) if denom_year > 0 else 0.0

            week_pos_log_share_to_peak_vs_peak = float(np.nansum(w_logrets_to_peak[w_logrets_to_peak > 0]) / denom_peak) if denom_peak > 0 else 0.0
            week_net_log_share_to_peak_vs_peak = float(np.nansum(w_logrets_to_peak[np.isfinite(w_logrets_to_peak)]) / denom_peak) if denom_peak > 0 else 0.0
            month_pos_log_share_to_peak_vs_peak = float(np.nansum(m_logrets_to_peak[m_logrets_to_peak > 0]) / denom_peak) if denom_peak > 0 else 0.0
            month_net_log_share_to_peak_vs_peak = float(np.nansum(m_logrets_to_peak[np.isfinite(m_logrets_to_peak)]) / denom_peak) if denom_peak > 0 else 0.0

            # peak 前漲停貢獻（日K logret）
            if peak_trade_date is not None:
                d_to_peak = d[d["date"] <= peak_trade_date]
            else:
                d_to_peak = d

            limitup_count_to_peak = int((d_to_peak["is_limit_up"] == 1).sum())
            limitup_log_sum_to_peak = float(d_to_peak.loc[d_to_peak["is_limit_up"] == 1, "d_logret"].sum())
            limitup_log_share_to_peak_vs_year = float(limitup_log_sum_to_peak / denom_year) if denom_year > 0 else 0.0
            limitup_log_share_to_peak_vs_peak = float(limitup_log_sum_to_peak / denom_peak) if denom_peak > 0 else 0.0

            # ========== NEW: peak 前 top1 week/month ==========
            top_week_id_net_to_peak = None
            limitup_count_in_top1week_to_peak = 0

            top_month_id_net_to_peak = None
            limitup_count_in_top1month_to_peak = 0

            if peak_trade_date is not None and len(wps) > 0:
                # 依「截到 peak」的 period logret 選 top1
                if w_logrets_to_peak.size == len(wps):
                    idxp = int(np.nanargmax(w_logrets_to_peak.astype(float)))
                    top_week_id_net_to_peak = wps.iloc[idxp]["week_id"]

                    # 計算該週內（但只到 peak）漲停數
                    ps = wps.iloc[idxp]["period_start"]
                    pe = wps.iloc[idxp]["period_end"]
                    pe2 = min(pd.Timestamp(pe), pd.Timestamp(peak_trade_date))
                    seg = d[(d["date"] >= ps) & (d["date"] <= pe2)]
                    limitup_count_in_top1week_to_peak = int((seg["is_limit_up"] == 1).sum())

            if peak_trade_date is not None and len(mps) > 0:
                if m_logrets_to_peak.size == len(mps):
                    idxmp = int(np.nanargmax(m_logrets_to_peak.astype(float)))
                    top_month_id_net_to_peak = mps.iloc[idxmp]["month_id"]

                    ps = mps.iloc[idxmp]["period_start"]
                    pe = mps.iloc[idxmp]["period_end"]
                    pe2 = min(pd.Timestamp(pe), pd.Timestamp(peak_trade_date))
                    seg = d[(d["date"] >= ps) & (d["date"] <= pe2)]
                    limitup_count_in_top1month_to_peak = int((seg["is_limit_up"] == 1).sum())

            rows.append(
                {
                    "symbol": sym,
                    "year": yr,

                    # ✅ denom
                    "denom_year": denom_year,

                    # 年K
                    "y_open": year_open,
                    "y_close": year_close,
                    "year_ret_pct": float(r["year_ret_pct"]),
                    "year_logret": year_logret,

                    # bins
                    "year_ret_bin_100": r["year_ret_bin_100"],
                    "year_ret_bin_10_under100": r["year_ret_bin_10_under100"],
                    "year_ret_bin_50_under100": r["year_ret_bin_50_under100"],

                    # ✅ burst style
                    "burst_style_week": burst_style_week,
                    "burst_style_month": burst_style_month,

                    # peak
                    "year_peak_date_raw": peak_date_raw.strftime("%Y-%m-%d") if pd.notna(peak_date_raw) else None,
                    "year_peak_trade_date": peak_trade_date.strftime("%Y-%m-%d") if peak_trade_date is not None else None,
                    "year_peak_high": float(r["year_peak_high"]) if np.isfinite(r["year_peak_high"]) else np.nan,
                    "peak_close_aligned": peak_close if np.isfinite(peak_close) else np.nan,
                    "peak_logret_from_open": peak_logret,

                    # 回撤
                    "year_max_drawdown_log": year_max_dd_log,
                    "peak_to_year_end_drawdown_log": peak_to_year_end_dd_log,
                    "worst_week_logret": worst_week_logret,
                    "worst_month_logret": worst_month_logret,

                    # ✅ NEW: 周/月波動率（logret std）
                    "week_vol_logret_std": week_vol_logret_std,
                    "month_vol_logret_std": month_vol_logret_std,
                    "year_weekly_max_drawdown_log": year_weekly_max_drawdown_log,
                    "year_monthly_max_drawdown_log": year_monthly_max_drawdown_log,

                    # ✅ NEW: 最後兩週/兩月幅度變化率
                    "prev_week_ret_pct": prev_week_ret_pct if np.isfinite(prev_week_ret_pct) else np.nan,
                    "last_week_ret_pct": last_week_ret_pct if np.isfinite(last_week_ret_pct) else np.nan,
                    "week_abs_ret_change_rate": week_abs_ret_change_rate,

                    "prev_month_ret_pct": prev_month_ret_pct if np.isfinite(prev_month_ret_pct) else np.nan,
                    "last_month_ret_pct": last_month_ret_pct if np.isfinite(last_month_ret_pct) else np.nan,
                    "month_abs_ret_change_rate": month_abs_ret_change_rate,

                    # ✅ NEW: top week/month info（全年）
                    "top_week_id": top_week_id,
                    "top_week_logret_net": top_week_logret_net,
                    "top_week_is_limitup_dense": top_week_is_limitup_dense,
                    "top_week_max_drawdown_log": top_week_max_drawdown_log,

                    "top_month_id": top_month_id,
                    "top_month_logret_net": top_month_logret_net,
                    "top_month_is_limitup_dense": top_month_is_limitup_dense,
                    "top_month_max_drawdown_log": top_month_max_drawdown_log,

                    # ✅ NEW: peak 前 top1 week/month
                    "top_week_id_net_to_peak": top_week_id_net_to_peak,
                    "limitup_count_in_top1week_to_peak": int(limitup_count_in_top1week_to_peak),

                    "top_month_id_net_to_peak": top_month_id_net_to_peak,
                    "limitup_count_in_top1month_to_peak": int(limitup_count_in_top1month_to_peak),

                    # 週/月集中度（全年）
                    "top1_week_share_pos": top1_week_share_pos,
                    "top4_weeks_share_pos": top4_weeks_share_pos,
                    "top1_week_share_net": top1_week_share_net,
                    "top4_weeks_share_net": top4_weeks_share_net,

                    "top1_month_share_pos": top1_month_share_pos,
                    "top3_months_share_pos": top3_months_share_pos,
                    "top1_month_share_net": top1_month_share_net,
                    "top3_months_share_net": top3_months_share_net,

                    "sum_pos_week_share": sum_pos_week_share,
                    "sum_net_week_share": sum_net_week_share,
                    "sum_pos_month_share": sum_pos_month_share,
                    "sum_net_month_share": sum_net_month_share,

                    # peak 前完成度
                    "logret_to_peak": logret_to_peak,
                    "share_year_to_peak": share_year_to_peak,
                    "share_peak_to_peak": share_peak_to_peak,

                    # peak 前週/月貢獻（POS/NET）
                    "week_pos_log_share_to_peak_vs_year": week_pos_log_share_to_peak_vs_year,
                    "week_net_log_share_to_peak_vs_year": week_net_log_share_to_peak_vs_year,
                    "month_pos_log_share_to_peak_vs_year": month_pos_log_share_to_peak_vs_year,
                    "month_net_log_share_to_peak_vs_year": month_net_log_share_to_peak_vs_year,

                    "week_pos_log_share_to_peak_vs_peak": week_pos_log_share_to_peak_vs_peak,
                    "week_net_log_share_to_peak_vs_peak": week_net_log_share_to_peak_vs_peak,
                    "month_pos_log_share_to_peak_vs_peak": month_pos_log_share_to_peak_vs_peak,
                    "month_net_log_share_to_peak_vs_peak": month_net_log_share_to_peak_vs_peak,

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

        # 寫回 year_contribution
        conn.execute("DROP TABLE IF EXISTS year_contribution")
        out.to_sql("year_contribution", conn, if_exists="replace", index=False)

        try:
            conn.execute("CREATE INDEX IF NOT EXISTS idx_year_contrib_symbol_year ON year_contribution(symbol, year)")
            conn.execute("CREATE INDEX IF NOT EXISTS idx_year_contrib_bin100 ON year_contribution(year_ret_bin_100)")
            conn.execute("CREATE INDEX IF NOT EXISTS idx_year_contrib_burst_week ON year_contribution(burst_style_week)")
            conn.execute("CREATE INDEX IF NOT EXISTS idx_year_contrib_burst_month ON year_contribution(burst_style_month)")
        except Exception:
            pass

        # bins summary（以 100% 分箱為主）
        def _agg(df: pd.DataFrame) -> pd.Series:
            return pd.Series(
                {
                    "n": int(len(df)),
                    "avg_year_ret_pct": float(df["year_ret_pct"].mean()),
                    "median_year_ret_pct": float(df["year_ret_pct"].median()),

                    # 集中度（pos/net）
                    "avg_top1_week_share_pos": float(df["top1_week_share_pos"].mean()),
                    "avg_top1_week_share_net": float(df["top1_week_share_net"].mean()),
                    "avg_top1_month_share_pos": float(df["top1_month_share_pos"].mean()),
                    "avg_top1_month_share_net": float(df["top1_month_share_net"].mean()),

                    # burst 比例
                    "pct_one_week_burst": float((df["burst_style_week"] == "ONE_WEEK_BURST").mean() * 100),
                    "pct_one_month_burst": float((df["burst_style_month"] == "ONE_MONTH_BURST").mean() * 100),

                    # peak 完成度
                    "avg_share_year_to_peak": float(df["share_year_to_peak"].mean()),

                    # 漲停貢獻
                    "avg_limitup_count_to_peak": float(df["limitup_count_to_peak"].mean()),
                    "avg_limitup_log_share_to_peak_vs_year": float(df["limitup_log_share_to_peak_vs_year"].mean()),
                    "avg_limitup_log_share_to_peak_vs_peak": float(df["limitup_log_share_to_peak_vs_peak"].mean()),

                    # 回撤
                    "avg_year_max_drawdown_log": float(df["year_max_drawdown_log"].mean()),
                    "avg_peak_to_year_end_drawdown_log": float(df["peak_to_year_end_drawdown_log"].mean()),

                    # ✅ NEW: 週/月波動率均值
                    "avg_week_vol_logret_std": float(df["week_vol_logret_std"].mean()) if "week_vol_logret_std" in df.columns else 0.0,
                    "avg_month_vol_logret_std": float(df["month_vol_logret_std"].mean()) if "month_vol_logret_std" in df.columns else 0.0,

                    # 直覺門檻
                    "pct_top1_week_net_ge_0_4": float((df["top1_week_share_net"] >= 0.4).mean() * 100),
                    "pct_top1_month_net_ge_0_4": float((df["top1_month_share_net"] >= 0.4).mean() * 100),
                    "pct_limitup_share_year_ge_0_4": float((df["limitup_log_share_to_peak_vs_year"] >= 0.4).mean() * 100),
                    "pct_peak_to_year_end_dd_le_m0_2": float((df["peak_to_year_end_drawdown_log"] <= -0.2).mean() * 100),
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

        print("\n✅ kbar_contribution（精準切段｜都來｜加周月波動/回撤/peak前top1）完成：")
        print(f"📌 year_contribution rows: {len(out):,}")
        print(f"📌 year_contribution_bins rows: {len(bins):,}")
        print("📌 新增：week/month vol, top_week/month info, peak 前 top1 week/month + 該期漲停數")

        return {"year_rows": int(len(out)), "bin_rows": int(len(bins))}

    finally:
        conn.close()


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python kbar_contribution.py <db_path>")
        sys.exit(1)

    build_contribution_tables(sys.argv[1])
