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

本次加強（你要的）：
- 週K/月K：period 內最大回撤（max_dd_log）
- 週K/月K：period 內波動率（vol_dlogret_std：週內/月內 daily d_logret 的 std，不年化）
- 週K/月K：「幅度變化率」（abs_ret_change_rate：abs(本期報酬%) 相對 abs(上期報酬%) 的變化率）
- 也把「最強週/最強月」的上述指標帶出（寫文章最好用）
- peak 前 top1 week（net 最大）也帶出 + 漲停數（你點名的）
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


def _logret_to_pct(logret: float) -> float:
    """logret -> pct return"""
    if not np.isfinite(logret):
        return 0.0
    return float((np.exp(logret) - 1.0) * 100.0)


def _period_features(
    daily: pd.DataFrame,
    periods: pd.DataFrame,
    *,
    id_col: str,
    cutoff_date: Optional[pd.Timestamp] = None,
) -> pd.DataFrame:
    """
    對每個 period 計算：
      - net_logret: sum(d_logret)
      - ret_pct / abs_ret_pct
      - abs_ret_change_rate: abs_ret_pct 對上一期 abs_ret_pct 的變化率
      - max_dd_log: period 內 close 的最大回撤（log）
      - vol_dlogret_std: period 內 d_logret 的 std（不年化）
      - limitup_count / n_days（方便算密度）

    cutoff_date 用於 peak 前切段：
      - period_end 會被截到 cutoff
      - 若 period_start > cutoff，視為 inactive（該期 features=0）
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
            n_days = 0
        else:
            net_logret = float(dd["d_logret"].sum())
            ret_pct = _logret_to_pct(net_logret)
            abs_ret = float(abs(ret_pct))
            max_dd_log = _max_drawdown_log_from_close(dd["close"])
            vol_std = float(pd.to_numeric(dd["d_logret"], errors="coerce").std(ddof=0) or 0.0)
            lu_cnt = int((dd["is_limit_up"] == 1).sum())
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
                "n_days": n_days,
            }
        )

    return pd.DataFrame(rows)


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

        # 逐日 logret
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

            # 從 peak 到年末回撤（log）
            peak_to_year_end_dd_log = 0.0
            if peak_trade_date is not None and np.isfinite(peak_close) and peak_close > 0:
                year_end_close = float(d.iloc[-1]["close"]) if np.isfinite(d.iloc[-1]["close"]) else np.nan
                peak_to_year_end_dd_log = _safe_log_ratio(year_end_close, peak_close) if np.isfinite(year_end_close) else 0.0

            # 週/月 periods
            wps = wk[(wk["symbol"] == sym) & (wk["year"] == yr)].sort_values("period_end")
            mps = mk[(mk["symbol"] == sym) & (mk["year"] == yr)].sort_values("period_end")

            # 精準切段：日K切週/月（logret array，保持你原本算法）
            w_logrets = _sum_logret_by_period(daily=d, periods=wps, cutoff_date=None)
            m_logrets = _sum_logret_by_period(daily=d, periods=mps, cutoff_date=None)

            worst_week_logret = float(np.nanmin(w_logrets)) if w_logrets.size > 0 else 0.0
            worst_month_logret = float(np.nanmin(m_logrets)) if m_logrets.size > 0 else 0.0

            # ✅ 週/月 period 內：回撤 / 波動率 / 幅度變化率（abs週報酬變化）
            wfeat = _period_features(d, wps, id_col="week_id", cutoff_date=None)
            mfeat = _period_features(d, mps, id_col="month_id", cutoff_date=None)

            # 取「週/月」的平均特徵（全年度）
            avg_week_max_dd_log = float(wfeat["max_dd_log"].mean()) if not wfeat.empty else 0.0
            avg_week_vol_dlogret_std = float(wfeat["vol_dlogret_std"].mean()) if not wfeat.empty else 0.0
            avg_week_abs_ret_change_rate = float(wfeat["abs_ret_change_rate"].mean()) if not wfeat.empty else 0.0

            avg_month_max_dd_log = float(mfeat["max_dd_log"].mean()) if not mfeat.empty else 0.0
            avg_month_vol_dlogret_std = float(mfeat["vol_dlogret_std"].mean()) if not mfeat.empty else 0.0
            avg_month_abs_ret_change_rate = float(mfeat["abs_ret_change_rate"].mean()) if not mfeat.empty else 0.0

            # 取「最強週 / 最強月」（net_logret 最大）
            top_week_id_net = None
            top_week_logret_net = 0.0
            top_week_max_dd_log = 0.0
            top_week_vol_dlogret_std = 0.0
            top_week_abs_ret_change_rate = 0.0
            top_week_is_limitup_dense = 0.0  # 漲停密度 = 漲停天數/該週交易日數

            if not wfeat.empty:
                ww = wfeat.sort_values("net_logret", ascending=False).iloc[0]
                top_week_id_net = int(ww["week_id"])
                top_week_logret_net = float(ww["net_logret"])
                top_week_max_dd_log = float(ww["max_dd_log"])
                top_week_vol_dlogret_std = float(ww["vol_dlogret_std"])
                top_week_abs_ret_change_rate = float(ww["abs_ret_change_rate"])
                n_days = int(ww["n_days"]) if np.isfinite(ww["n_days"]) else 0
                lu_cnt = int(ww["limitup_count"]) if np.isfinite(ww["limitup_count"]) else 0
                top_week_is_limitup_dense = float(lu_cnt / max(n_days, 1))

            top_month_id_net = None
            top_month_logret_net = 0.0
            top_month_max_dd_log = 0.0
            top_month_vol_dlogret_std = 0.0
            top_month_abs_ret_change_rate = 0.0
            top_month_is_limitup_dense = 0.0

            if not mfeat.empty:
                mm = mfeat.sort_values("net_logret", ascending=False).iloc[0]
                top_month_id_net = int(mm["month_id"])
                top_month_logret_net = float(mm["net_logret"])
                top_month_max_dd_log = float(mm["max_dd_log"])
                top_month_vol_dlogret_std = float(mm["vol_dlogret_std"])
                top_month_abs_ret_change_rate = float(mm["abs_ret_change_rate"])
                n_days = int(mm["n_days"]) if np.isfinite(mm["n_days"]) else 0
                lu_cnt = int(mm["limitup_count"]) if np.isfinite(mm["limitup_count"]) else 0
                top_month_is_limitup_dense = float(lu_cnt / max(n_days, 1))

            # ✅ peak 前 top1 週（net 最大）+ 漲停數
            top_week_id_net_to_peak = None
            top_week_logret_net_to_peak = 0.0
            top_week_max_dd_log_to_peak = 0.0
            top_week_vol_dlogret_std_to_peak = 0.0
            top_week_abs_ret_change_rate_to_peak = 0.0
            limitup_count_in_top1week_to_peak = 0
            top_week_is_limitup_dense_to_peak = 0.0

            if peak_trade_date is not None and not wps.empty:
                wfeat_peak = _period_features(d, wps, id_col="week_id", cutoff_date=peak_trade_date)
                wfeat_peak = wfeat_peak[wfeat_peak["active_to_cutoff"] == 1].copy()
                if not wfeat_peak.empty:
                    ww2 = wfeat_peak.sort_values("net_logret", ascending=False).iloc[0]
                    top_week_id_net_to_peak = int(ww2["week_id"])
                    top_week_logret_net_to_peak = float(ww2["net_logret"])
                    top_week_max_dd_log_to_peak = float(ww2["max_dd_log"])
                    top_week_vol_dlogret_std_to_peak = float(ww2["vol_dlogret_std"])
                    top_week_abs_ret_change_rate_to_peak = float(ww2["abs_ret_change_rate"])
                    n_days2 = int(ww2["n_days"]) if np.isfinite(ww2["n_days"]) else 0
                    lu_cnt2 = int(ww2["limitup_count"]) if np.isfinite(ww2["limitup_count"]) else 0
                    limitup_count_in_top1week_to_peak = lu_cnt2
                    top_week_is_limitup_dense_to_peak = float(lu_cnt2 / max(n_days2, 1))

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

            # peak 前：週/月切段到 peak（維持你原本算法）
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

            rows.append(
                {
                    "symbol": sym,
                    "year": yr,

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

                    # ✅ 週K/月K（年度平均）回撤/波動/幅度變化率
                    "avg_week_max_dd_log": avg_week_max_dd_log,
                    "avg_week_vol_dlogret_std": avg_week_vol_dlogret_std,
                    "avg_week_abs_ret_change_rate": avg_week_abs_ret_change_rate,

                    "avg_month_max_dd_log": avg_month_max_dd_log,
                    "avg_month_vol_dlogret_std": avg_month_vol_dlogret_std,
                    "avg_month_abs_ret_change_rate": avg_month_abs_ret_change_rate,

                    # ✅ 最強週 / 最強月（net 最大）回撤/波動/幅度變化率 + 漲停密度
                    "top_week_id_net": top_week_id_net,
                    "top_week_logret_net": top_week_logret_net,
                    "top_week_max_dd_log": top_week_max_dd_log,
                    "top_week_vol_dlogret_std": top_week_vol_dlogret_std,
                    "top_week_abs_ret_change_rate": top_week_abs_ret_change_rate,
                    "top_week_is_limitup_dense": top_week_is_limitup_dense,

                    "top_month_id_net": top_month_id_net,
                    "top_month_logret_net": top_month_logret_net,
                    "top_month_max_dd_log": top_month_max_dd_log,
                    "top_month_vol_dlogret_std": top_month_vol_dlogret_std,
                    "top_month_abs_ret_change_rate": top_month_abs_ret_change_rate,
                    "top_month_is_limitup_dense": top_month_is_limitup_dense,

                    # ✅ peak 前 top1 週（net 最大）+ 漲停數（你點名的）
                    "top_week_id_net_to_peak": top_week_id_net_to_peak,
                    "top_week_logret_net_to_peak": top_week_logret_net_to_peak,
                    "top_week_max_dd_log_to_peak": top_week_max_dd_log_to_peak,
                    "top_week_vol_dlogret_std_to_peak": top_week_vol_dlogret_std_to_peak,
                    "top_week_abs_ret_change_rate_to_peak": top_week_abs_ret_change_rate_to_peak,
                    "limitup_count_in_top1week_to_peak": limitup_count_in_top1week_to_peak,
                    "top_week_is_limitup_dense_to_peak": top_week_is_limitup_dense_to_peak,
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

                    # ✅ 週/月平均（回撤/波動/幅度變化率）
                    "avg_week_max_dd_log": float(df["avg_week_max_dd_log"].mean()) if "avg_week_max_dd_log" in df.columns else 0.0,
                    "avg_week_vol_dlogret_std": float(df["avg_week_vol_dlogret_std"].mean()) if "avg_week_vol_dlogret_std" in df.columns else 0.0,
                    "avg_week_abs_ret_change_rate": float(df["avg_week_abs_ret_change_rate"].mean()) if "avg_week_abs_ret_change_rate" in df.columns else 0.0,

                    "avg_month_max_dd_log": float(df["avg_month_max_dd_log"].mean()) if "avg_month_max_dd_log" in df.columns else 0.0,
                    "avg_month_vol_dlogret_std": float(df["avg_month_vol_dlogret_std"].mean()) if "avg_month_vol_dlogret_std" in df.columns else 0.0,
                    "avg_month_abs_ret_change_rate": float(df["avg_month_abs_ret_change_rate"].mean()) if "avg_month_abs_ret_change_rate" in df.columns else 0.0,

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

        print("\n✅ kbar_contribution（精準切段｜都來）完成：")
        print(f"📌 year_contribution rows: {len(out):,}")
        print(f"📌 year_contribution_bins rows: {len(bins):,}")
        print("📌 burst labels：burst_style_week / burst_style_month 已加入")
        print("📌 NEW：週/月回撤(max_dd_log)、波動(vol_dlogret_std)、幅度變化率(abs_ret_change_rate) 已加入（含 top1 週/月 與 peak 前 top1 週）")

        return {"year_rows": int(len(out)), "bin_rows": int(len(bins))}

    finally:
        conn.close()


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python kbar_contribution.py <db_path>")
        sys.exit(1)

    build_contribution_tables(sys.argv[1])
