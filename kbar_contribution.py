# kbar_contribution.py
# -*- coding: utf-8 -*-
"""
kbar_contribution.py  (進階整合版｜精準切段｜都來｜不刪舊功能，只加)
---------------------------------------------------------------
依賴：
- kbar_aggregator.py -> kbar_weekly / kbar_monthly / kbar_yearly
- processor.py       -> stock_analysis（需 open/close/prev_close/is_limit_up）

輸出：
- year_contribution
- year_contribution_bins

新增（整合需求）：
1) denom_year / denom_peak：
   - denom_year: max(year_logret, 0)
   - denom_peak: max(peak_logret_from_open, 0)

2) top week / top month 相關（NET）：
   - top_week_id_net, top_week_logret_net, top_week_share_net
   - top_week_is_limitup_dense (= top_week_limitup_count / top_week_n_days)
   - top_week_max_dd_log, top_week_vol_dlogret_std, top_week_abs_ret_change_rate

   - peak 前版本：
     top_week_id_net_to_peak, top_week_logret_net_to_peak, top_week_share_net_to_peak
     limitup_count_in_top1week_to_peak, top_week_is_limitup_dense_to_peak

3) 週K/月K「回撤 + 波動率 + 幅度變化率」：
   - per-week:  max_dd_log, vol_dlogret_std, abs_ret_change_rate
   - per-month: max_dd_log, vol_dlogret_std, abs_ret_change_rate

定義：
- period_net_logret = sum(d_logret) in period
- period_ret_pct    = (exp(net_logret)-1)*100
- period_abs_ret_pct = abs(period_ret_pct)
- abs_ret_change_rate = (abs_ret_pct - prev_abs_ret_pct) / max(prev_abs_ret_pct, eps)
- max_dd_log：period 內用 daily close 算 log drawdown（peak-to-trough, log space）
- vol_dlogret_std：period 內 daily logret 的標準差（不年化；你寫文章最直覺）
"""

import sys
import sqlite3
import numpy as np
import pandas as pd
from typing import Optional, Dict, Tuple

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


# -----------------------------------------------------------------------------
# period segmentation (精準切段)
# -----------------------------------------------------------------------------
def _sum_logret_by_period(
    daily: pd.DataFrame,
    periods: pd.DataFrame,
    cutoff_date: Optional[pd.Timestamp] = None,
) -> np.ndarray:
    """回傳 array: 每個 period 的 sum(d_logret)，精準切段（可 cutoff）"""
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
    """把 peak_date 對齊到 <= peak_date 的最後一個交易日"""
    if peak_date is None or pd.isna(peak_date):
        return None
    dd = pd.to_datetime(daily_dates, errors="coerce").dropna().sort_values()
    if dd.empty:
        return None
    dd2 = dd[dd <= peak_date]
    if dd2.empty:
        return None
    return pd.Timestamp(dd2.iloc[-1])


def _period_features(
    daily: pd.DataFrame,
    periods: pd.DataFrame,
    *,
    id_col: str,
    cutoff_date: Optional[pd.Timestamp] = None,
) -> pd.DataFrame:
    """
    對每個 period 計算：
    - net_logret (sum d_logret)
    - ret_pct / abs_ret_pct
    - abs_ret_change_rate（和上一期 abs_ret_pct 的變化率）
    - max_dd_log（period 內 close 的最大回撤）
    - vol_dlogret_std（period 內 d_logret 標準差）
    - limitup_count / limitup_log_sum / n_days
    - active_to_cutoff（peak 前用）
    """
    if daily.empty or periods.empty:
        return pd.DataFrame()

    d = daily.copy()
    d["date"] = pd.to_datetime(d["date"], errors="coerce")
    d = d.dropna(subset=["date"]).sort_values("date")

    p = periods.copy()
    p["period_start"] = pd.to_datetime(p["period_start"], errors="coerce")
    p["period_end"] = pd.to_datetime(p["period_end"], errors="coerce")
    p = p.dropna(subset=["period_start", "period_end"]).sort_values("period_end")

    rows = []
    prev_abs = None

    for _, r in p.iterrows():
        pid = r[id_col]
        ps = r["period_start"]
        pe = r["period_end"]

        # peak 前：只允許算到 cutoff_date（但 period 若完全在 cutoff 後，視為 inactive）
        active = True
        pe_eff = pe
        if cutoff_date is not None and pd.notna(cutoff_date):
            if ps > cutoff_date:
                active = False
            pe_eff = min(pe, cutoff_date)

        mask = (d["date"] >= ps) & (d["date"] <= pe_eff)
        dd = d.loc[mask].copy()

        if (not active) or dd.empty:
            net_logret = 0.0
            ret_pct = 0.0
            abs_ret = 0.0
            max_dd_log = 0.0
            vol_std = 0.0
            lu_cnt = 0
            lu_log_sum = 0.0
            n_days = 0
        else:
            net_logret = float(dd["d_logret"].sum())
            ret_pct = float((np.exp(net_logret) - 1.0) * 100.0)
            abs_ret = float(abs(ret_pct))
            max_dd_log = _max_drawdown_log_from_close(dd["close"]) if "close" in dd.columns else 0.0
            vol_std = float(pd.to_numeric(dd["d_logret"], errors="coerce").std(ddof=0) or 0.0)
            lu_cnt = int((dd["is_limit_up"] == 1).sum()) if "is_limit_up" in dd.columns else 0
            lu_log_sum = float(dd.loc[dd["is_limit_up"] == 1, "d_logret"].sum()) if "is_limit_up" in dd.columns else 0.0
            n_days = int(len(dd))

        if prev_abs is None:
            abs_chg_rate = 0.0
        else:
            abs_chg_rate = float((abs_ret - prev_abs) / max(prev_abs, EPS))

        if active:
            prev_abs = abs_ret

        rows.append(
            {
                id_col: pid,
                "period_start": ps,
                "period_end": pe,
                "active_to_cutoff": int(active),
                "net_logret": net_logret,
                "ret_pct": ret_pct,
                "abs_ret_pct": abs_ret,
                "abs_ret_change_rate": abs_chg_rate,
                "max_dd_log": max_dd_log,
                "vol_dlogret_std": vol_std,
                "limitup_count": lu_cnt,
                "limitup_log_sum": lu_log_sum,
                "n_days": n_days,
            }
        )

    return pd.DataFrame(rows)


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
        y["year_logret"] = (
            np.log(y["y_close"].astype(float) / y["y_open"].astype(float))
            .replace([np.inf, -np.inf], np.nan)
            .fillna(0.0)
        )

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

        # 逐日 logret（第一天用 open->close，其餘用 prev_close->close）
        sa["d_logret"] = 0.0
        mask_cp = (sa["close"].astype(float) > 0) & (sa["prev_close"].astype(float) > 0)
        sa.loc[mask_cp, "d_logret"] = np.log(
            sa.loc[mask_cp, "close"].astype(float) / sa.loc[mask_cp, "prev_close"].astype(float)
        )

        sa["rank_in_year"] = sa.groupby(["symbol", "year"]).cumcount()
        mask_first = sa["rank_in_year"] == 0
        mask_oc = mask_first & (sa["close"].astype(float) > 0) & (sa["open"].astype(float) > 0)
        sa.loc[mask_oc, "d_logret"] = np.log(
            sa.loc[mask_oc, "close"].astype(float) / sa.loc[mask_oc, "open"].astype(float)
        )

        rows = []
        for _, r in y.iterrows():
            sym = r["symbol"]
            yr = int(r["year"])
            year_open = float(r["y_open"])
            year_close = float(r["y_close"])
            year_logret = float(r["year_logret"])

            denom_year = float(year_logret) if year_logret > 0 else 0.0  # ✅ 你要的 denom_year

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
            denom_peak = float(peak_logret) if peak_logret > 0 else 0.0  # ✅ 你要的 denom_peak/denom_peak概念

            # 從 peak 到年末回撤（log ratio）
            peak_to_year_end_dd_log = 0.0
            if peak_trade_date is not None and np.isfinite(peak_close) and peak_close > 0:
                year_end_close = float(d.iloc[-1]["close"]) if np.isfinite(d.iloc[-1]["close"]) else np.nan
                peak_to_year_end_dd_log = _safe_log_ratio(year_end_close, peak_close) if np.isfinite(year_end_close) else 0.0

            # 週/月 periods（當年）
            wps = wk[(wk["symbol"] == sym) & (wk["year"] == yr)].sort_values("period_end")
            mps = mk[(mk["symbol"] == sym) & (mk["year"] == yr)].sort_values("period_end")

            # 精準切段：sum logret（全年）
            w_logrets = _sum_logret_by_period(daily=d, periods=wps, cutoff_date=None)
            m_logrets = _sum_logret_by_period(daily=d, periods=mps, cutoff_date=None)

            worst_week_logret = float(np.nanmin(w_logrets)) if w_logrets.size > 0 else 0.0
            worst_month_logret = float(np.nanmin(m_logrets)) if m_logrets.size > 0 else 0.0

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

            # peak 前：週/月切段到 peak（用 sum logret）
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

            # -----------------------------------------------------------------
            # ✅ 新增：週/月 period 內的「回撤 + 波動率 + 幅度變化率」
            # -----------------------------------------------------------------
            wfeat = _period_features(d, wps, id_col="week_id", cutoff_date=None)
            mfeat = _period_features(d, mps, id_col="month_id", cutoff_date=None)

            # top1 week（NET）詳細資訊（全年）
            top_week_id_net = None
            top_week_logret_net = 0.0
            top_week_share_net = 0.0
            top_week_is_limitup_dense = 0.0
            top_week_max_dd_log = 0.0
            top_week_vol_dlogret_std = 0.0
            top_week_abs_ret_change_rate = 0.0
            top_week_limitup_count = 0
            top_week_n_days = 0

            if not wfeat.empty:
                wfeat2 = wfeat.copy()
                wfeat2 = wfeat2.sort_values("net_logret", ascending=False)
                top = wfeat2.iloc[0]
                top_week_id_net = int(top["week_id"])
                top_week_logret_net = float(top["net_logret"])
                top_week_share_net = float(top_week_logret_net / denom_year) if denom_year > 0 else 0.0

                top_week_n_days = int(top.get("n_days", 0) or 0)
                top_week_limitup_count = int(top.get("limitup_count", 0) or 0)
                top_week_is_limitup_dense = float(top_week_limitup_count / max(top_week_n_days, 1))

                top_week_max_dd_log = float(top.get("max_dd_log", 0.0) or 0.0)
                top_week_vol_dlogret_std = float(top.get("vol_dlogret_std", 0.0) or 0.0)
                top_week_abs_ret_change_rate = float(top.get("abs_ret_change_rate", 0.0) or 0.0)

            # top1 month（NET）可選也補一份（你之後想寫月K文章會很爽）
            top_month_id_net = None
            top_month_logret_net = 0.0
            top_month_share_net = 0.0
            top_month_max_dd_log = 0.0
            top_month_vol_dlogret_std = 0.0
            top_month_abs_ret_change_rate = 0.0

            if not mfeat.empty:
                mfeat2 = mfeat.sort_values("net_logret", ascending=False)
                topm = mfeat2.iloc[0]
                top_month_id_net = int(topm["month_id"])
                top_month_logret_net = float(topm["net_logret"])
                top_month_share_net = float(top_month_logret_net / denom_year) if denom_year > 0 else 0.0
                top_month_max_dd_log = float(topm.get("max_dd_log", 0.0) or 0.0)
                top_month_vol_dlogret_std = float(topm.get("vol_dlogret_std", 0.0) or 0.0)
                top_month_abs_ret_change_rate = float(topm.get("abs_ret_change_rate", 0.0) or 0.0)

            # peak 前 top1 week（NET）詳細資訊
            top_week_id_net_to_peak = None
            top_week_logret_net_to_peak = 0.0
            top_week_share_net_to_peak = 0.0
            limitup_count_in_top1week_to_peak = 0
            top_week_is_limitup_dense_to_peak = 0.0

            if peak_trade_date is not None and not wps.empty:
                wfeat_peak = _period_features(d, wps, id_col="week_id", cutoff_date=peak_trade_date)
                wfeat_peak = wfeat_peak[wfeat_peak["active_to_cutoff"] == 1].copy()
                if not wfeat_peak.empty:
                    wfeat_peak = wfeat_peak.sort_values("net_logret", ascending=False)
                    top_p = wfeat_peak.iloc[0]
                    top_week_id_net_to_peak = int(top_p["week_id"])
                    top_week_logret_net_to_peak = float(top_p["net_logret"])
                    top_week_share_net_to_peak = float(top_week_logret_net_to_peak / denom_year) if denom_year > 0 else 0.0

                    n_days_p = int(top_p.get("n_days", 0) or 0)
                    lu_cnt_p = int(top_p.get("limitup_count", 0) or 0)
                    limitup_count_in_top1week_to_peak = lu_cnt_p
                    top_week_is_limitup_dense_to_peak = float(lu_cnt_p / max(n_days_p, 1))

            # -----------------------------------------------------------------
            # 寫 row
            # -----------------------------------------------------------------
            rows.append(
                {
                    "symbol": sym,
                    "year": yr,

                    # 年K
                    "y_open": year_open,
                    "y_close": year_close,
                    "year_ret_pct": float(r["year_ret_pct"]),
                    "year_logret": year_logret,

                    # ✅ denom
                    "denom_year": denom_year,
                    "denom_peak": denom_peak,

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

                    # 回撤（年）
                    "year_max_drawdown_log": year_max_dd_log,
                    "peak_to_year_end_drawdown_log": peak_to_year_end_dd_log,
                    "worst_week_logret": worst_week_logret,
                    "worst_month_logret": worst_month_logret,

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

                    # ✅ 你之前要的：top week（全年）
                    "top_week_id_net": top_week_id_net,
                    "top_week_logret_net": top_week_logret_net,
                    "top_week_share_net": top_week_share_net,
                    "top_week_n_days": top_week_n_days,
                    "top_week_limitup_count": top_week_limitup_count,
                    "top_week_is_limitup_dense": top_week_is_limitup_dense,
                    "top_week_max_dd_log": top_week_max_dd_log,
                    "top_week_vol_dlogret_std": top_week_vol_dlogret_std,
                    "top_week_abs_ret_change_rate": top_week_abs_ret_change_rate,

                    # ✅ peak 前 top1 week（只在 peak 前挑 top1 週）
                    "top_week_id_net_to_peak": top_week_id_net_to_peak,
                    "top_week_logret_net_to_peak": top_week_logret_net_to_peak,
                    "top_week_share_net_to_peak": top_week_share_net_to_peak,
                    "limitup_count_in_top1week_to_peak": limitup_count_in_top1week_to_peak,
                    "top_week_is_limitup_dense_to_peak": top_week_is_limitup_dense_to_peak,

                    # ✅ top month（全年，順手補齊）
                    "top_month_id_net": top_month_id_net,
                    "top_month_logret_net": top_month_logret_net,
                    "top_month_share_net": top_month_share_net,
                    "top_month_max_dd_log": top_month_max_dd_log,
                    "top_month_vol_dlogret_std": top_month_vol_dlogret_std,
                    "top_month_abs_ret_change_rate": top_month_abs_ret_change_rate,
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

                    # 直覺門檻
                    "pct_top1_week_net_ge_0_4": float((df["top1_week_share_net"] >= 0.4).mean() * 100),
                    "pct_top1_month_net_ge_0_4": float((df["top1_month_share_net"] >= 0.4).mean() * 100),
                    "pct_limitup_share_year_ge_0_4": float((df["limitup_log_share_to_peak_vs_year"] >= 0.4).mean() * 100),
                    "pct_peak_to_year_end_dd_le_m0_2": float((df["peak_to_year_end_drawdown_log"] <= -0.2).mean() * 100),

                    # ✅ 新增：top week 波動/回撤摘要（幫你寫周K文章用）
                    "avg_top_week_vol_dlogret_std": float(df["top_week_vol_dlogret_std"].mean()),
                    "avg_top_week_max_dd_log": float(df["top_week_max_dd_log"].mean()),
                    "avg_top_week_is_limitup_dense": float(df["top_week_is_limitup_dense"].mean()),
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

        print("\n✅ kbar_contribution（進階整合｜精準切段｜都來）完成：")
        print(f"📌 year_contribution rows: {len(out):,}")
        print(f"📌 year_contribution_bins rows: {len(bins):,}")
        print("📌 新增：denom_year/denom_peak、top_week*、peak前top_week*、週/月回撤&波動&幅度變化率")

        return {"year_rows": int(len(out)), "bin_rows": int(len(bins))}

    finally:
        conn.close()


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python kbar_contribution.py <db_path>")
        sys.exit(1)

    build_contribution_tables(sys.argv[1])
