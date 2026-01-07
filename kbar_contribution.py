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

本次修改（重點）：
- 使用 data_cleaning.py 來清洗 stock_analysis（日K），並在清洗後重算 prev_close + d_logret
  => 避免減資/併股等極端跳動污染貢獻度（與你的 kbar_aggregator 口徑一致化）
"""

import sys
import sqlite3
import numpy as np
import pandas as pd
from typing import Optional, Dict

SQLITE_TIMEOUT = 120


# -----------------------------------------------------------------------------
# ✅ optional import: data_cleaning
# -----------------------------------------------------------------------------
DATA_CLEANING_AVAILABLE = False
try:
    # 同 repo 直接 import
    from data_cleaning import preset_close_cleaning, clean_pingpong  # type: ignore
    DATA_CLEANING_AVAILABLE = True
except Exception:
    DATA_CLEANING_AVAILABLE = False


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
# ✅ daily cleaning + recompute d_logret
# -----------------------------------------------------------------------------
def _recompute_d_logret_in_year(d: pd.DataFrame) -> pd.DataFrame:
    """
    d: 單一 symbol + 單一年份，且已依 date 排序。
    - 先確保 prev_close 連續（用 close shift）
    - 再計算 d_logret：
        * 非第一天：log(close/prev_close)
        * 第一天：   log(close/open)（與你原本邏輯一致）
    """
    out = d.copy()
    out["date"] = pd.to_datetime(out["date"], errors="coerce")
    out = out.dropna(subset=["date"]).sort_values("date").reset_index(drop=True)

    # prev_close 用 close shift 重算（避免剔除後斷裂）
    out["prev_close"] = pd.to_numeric(out["close"], errors="coerce").shift(1)

    # rank in year
    out["rank_in_year"] = np.arange(len(out), dtype=int)

    # d_logret
    out["d_logret"] = 0.0

    close = pd.to_numeric(out["close"], errors="coerce")
    prev = pd.to_numeric(out["prev_close"], errors="coerce")
    open_ = pd.to_numeric(out["open"], errors="coerce")

    mask_cp = (close > 0) & (prev > 0)
    out.loc[mask_cp, "d_logret"] = np.log(close[mask_cp] / prev[mask_cp])

    # first day uses close/open
    mask_first = out["rank_in_year"] == 0
    mask_oc = mask_first & (close > 0) & (open_ > 0)
    out.loc[mask_oc, "d_logret"] = np.log(close[mask_oc] / open_[mask_oc])

    return out


def _maybe_clean_daily(d: pd.DataFrame, enable_cleaning: bool, pingpong_threshold: float, abs_ret_cap: float) -> pd.DataFrame:
    """
    清洗單一 symbol+year 的日K，並重算 prev_close + d_logret
    """
    if d.empty:
        return d

    d = d.copy()
    d["date"] = pd.to_datetime(d["date"], errors="coerce")
    d = d.dropna(subset=["date"]).sort_values("date")

    if (not enable_cleaning) or (not DATA_CLEANING_AVAILABLE):
        # 沒清洗也要確保 d_logret 正確（避免 stock_analysis 欄位不一致）
        return _recompute_d_logret_in_year(d)

    # 使用 data_cleaning 的 close preset
    cfg = preset_close_cleaning(date_col="date", pingpong_threshold=pingpong_threshold, abs_ret_cap=abs_ret_cap)

    # clean_pingpong 建議以 close 做清洗，然後重算 prev_close
    cleaned = clean_pingpong(
        d,
        cfg,
        recompute_prev_close=True,
        recompute_daily_change=False,  # 這支貢獻度不吃 daily_change
        prev_close_col="prev_close",
        daily_change_col="daily_change",
        return_reasons=False,
    )

    # 清洗後再重算 d_logret（用重算後的 prev_close）
    cleaned = _recompute_d_logret_in_year(cleaned)

    # is_limit_up 保留（clean_pingpong 不會刪欄）
    if "is_limit_up" in d.columns and "is_limit_up" in cleaned.columns:
        cleaned["is_limit_up"] = pd.to_numeric(cleaned["is_limit_up"], errors="coerce").fillna(0).astype(int)

    return cleaned


# -----------------------------------------------------------------------------
# core
# -----------------------------------------------------------------------------
def build_contribution_tables(
    db_path: str,
    only_markets: Optional[set] = None,
    *,
    enable_cleaning: bool = True,
    pingpong_threshold: float = 0.40,
    abs_ret_cap: float = 0.80,
    verbose: bool = True,
) -> Dict[str, int]:
    """
    enable_cleaning:
      - True 且 data_cleaning.py 可用：會對每個 symbol-year 的日K做清洗 + 重算 d_logret
      - False：不清洗，但仍會重算 d_logret 以保持口徑一致
    """
    conn = sqlite3.connect(db_path, timeout=SQLITE_TIMEOUT)

    try:
        existing = set(pd.read_sql("SELECT name FROM sqlite_master WHERE type='table'", conn)["name"].tolist())
        required = ["kbar_yearly", "kbar_monthly", "kbar_weekly", "stock_analysis"]
        missing = [t for t in required if t not in existing]
        if missing:
            raise RuntimeError(f"缺少必要表：{missing}\n請先跑 processor.py 與 kbar_aggregator.py")

        if enable_cleaning and not DATA_CLEANING_AVAILABLE:
            if verbose:
                print("⚠️ enable_cleaning=True 但找不到 data_cleaning.py（或 import 失敗），將改為不清洗（但仍重算 d_logret）。")

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
            if verbose:
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
                if verbose:
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

        rows = []
        for _, r in y.iterrows():
            sym = r["symbol"]
            yr = int(r["year"])

            year_open = float(r["y_open"])
            year_close = float(r["y_close"])
            year_logret = float(r["year_logret"])
            denom_year = year_logret if year_logret > 0 else 0.0

            d0 = sa[(sa["symbol"] == sym) & (sa["year"] == yr)].copy()
            if d0.empty:
                continue

            # ✅ 清洗 + 重算 d_logret（或至少重算 d_logret）
            d = _maybe_clean_daily(
                d0,
                enable_cleaning=enable_cleaning,
                pingpong_threshold=pingpong_threshold,
                abs_ret_cap=abs_ret_cap,
            )
            if d.empty:
                continue

            # 年內最大回撤（用 close）
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

            # 週/月 periods（該股該年）=> 建議 reset_index，避免日後你擴充要拿 iloc 對齊時踩坑
            wps = wk[(wk["symbol"] == sym) & (wk["year"] == yr)].sort_values("period_end").reset_index(drop=True)
            mps = mk[(mk["symbol"] == sym) & (mk["year"] == yr)].sort_values("period_end").reset_index(drop=True)

            # 精準切段：日K切週/月
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

            # peak 前：週/月切段到 peak
            w_logrets_to_peak = _sum_logret_by_period(daily=d, periods=wps, cutoff_date=peak_trade_date)
            m_logrets_to_peak = _sum_logret_by_period(daily=d, periods=mps, cutoff_date=peak_trade_date)

            week_pos_log_share_to_peak_vs_year = (
                float(np.nansum(w_logrets_to_peak[w_logrets_to_peak > 0]) / denom_year) if denom_year > 0 else 0.0
            )
            week_net_log_share_to_peak_vs_year = (
                float(np.nansum(w_logrets_to_peak[np.isfinite(w_logrets_to_peak)]) / denom_year) if denom_year > 0 else 0.0
            )
            month_pos_log_share_to_peak_vs_year = (
                float(np.nansum(m_logrets_to_peak[m_logrets_to_peak > 0]) / denom_year) if denom_year > 0 else 0.0
            )
            month_net_log_share_to_peak_vs_year = (
                float(np.nansum(m_logrets_to_peak[np.isfinite(m_logrets_to_peak)]) / denom_year) if denom_year > 0 else 0.0
            )

            week_pos_log_share_to_peak_vs_peak = (
                float(np.nansum(w_logrets_to_peak[w_logrets_to_peak > 0]) / denom_peak) if denom_peak > 0 else 0.0
            )
            week_net_log_share_to_peak_vs_peak = (
                float(np.nansum(w_logrets_to_peak[np.isfinite(w_logrets_to_peak)]) / denom_peak) if denom_peak > 0 else 0.0
            )
            month_pos_log_share_to_peak_vs_peak = (
                float(np.nansum(m_logrets_to_peak[m_logrets_to_peak > 0]) / denom_peak) if denom_peak > 0 else 0.0
            )
            month_net_log_share_to_peak_vs_peak = (
                float(np.nansum(m_logrets_to_peak[np.isfinite(m_logrets_to_peak)]) / denom_peak) if denom_peak > 0 else 0.0
            )

            # peak 前漲停貢獻（日K logret）
            d_to_peak = d[d["date"] <= peak_trade_date] if peak_trade_date is not None else d

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
                }
            )

        out = pd.DataFrame(rows)
        if out.empty:
            if verbose:
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

        if verbose:
            print("\n✅ kbar_contribution（精準切段｜都來｜已接 data_cleaning）完成：")
            print(f"📌 year_contribution rows: {len(out):,}")
            print(f"📌 year_contribution_bins rows: {len(bins):,}")
            if enable_cleaning and DATA_CLEANING_AVAILABLE:
                print(f"📌 清洗：pingpong_threshold={pingpong_threshold}, abs_ret_cap={abs_ret_cap}")
            else:
                print("📌 清洗：停用（但仍重算 d_logret 以保持一致）")
            print("📌 burst labels：burst_style_week / burst_style_month 已加入")

        return {"year_rows": int(len(out)), "bin_rows": int(len(bins))}

    finally:
        conn.close()


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python kbar_contribution.py <db_path>")
        sys.exit(1)

    build_contribution_tables(sys.argv[1])
