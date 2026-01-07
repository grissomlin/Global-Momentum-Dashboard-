# kbar_contribution.py
# -*- coding: utf-8 -*-
"""
kbar_contribution.py  (精準切段版｜進階整合版｜不刪舊功能，只加)
-----------------------------------------------------------------
依賴：
- kbar_aggregator.py -> kbar_yearly / kbar_monthly / kbar_weekly
- processor.py       -> stock_analysis（需 open/close/prev_close/is_limit_up）

新增（進階整合，本版）：
1) ✅ 統一清洗口徑：可呼叫 data_cleaning.py（若不存在則 fallback 內建）
   - 乒乓極端震盪剔除（pingpong）
   - 單日超大跳動 cap（abs_ret_cap）
   - 剔除後自動重算 prev_close + d_logret（避免斷裂污染）
2) ✅ denom_year 可切換：
   - POS_ONLY : year_logret <= 0 時 denom=0（原版行為）
   - ABS      : denom=abs(year_logret)（下跌年也能算集中度/比例）
3) ✅ top week 相關欄位（全年 + peak 前）：
   - top_week_id_net
   - top_week_logret_net
   - top_week_is_limitup_dense
   - limitup_count_in_top1week_to_peak
   - top_week_id_net_to_peak
   - top_week_logret_net_to_peak
   - top_week_is_limitup_dense_to_peak
4) ✅ burst labels（保留舊功能）：
   - burst_style_week / burst_style_month

輸出：
- year_contribution
- year_contribution_bins
"""

import sys
import sqlite3
import numpy as np
import pandas as pd
from typing import Optional, Dict, Tuple

SQLITE_TIMEOUT = 120

# -----------------------------
# Cleaning defaults (align with your aggregator idea)
# -----------------------------
PINGPONG_THRESHOLD_DEFAULT = 0.40
ABS_DAILY_RET_CAP_DEFAULT = 0.80
ENABLE_CLEANING_DEFAULT = True

# -----------------------------
# denom modes
# -----------------------------
DENOM_POS_ONLY = "POS_ONLY"  # year_logret>0 才算比例（原本邏輯）
DENOM_ABS = "ABS"            # denom=abs(year_logret)，下跌年也可算集中度


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
# Cleaning integration (prefer data_cleaning.py)
# -----------------------------------------------------------------------------
def _fallback_pingpong_clean_daily(
    df: pd.DataFrame,
    threshold: float,
    abs_cap: float,
) -> pd.DataFrame:
    """
    Fallback cleaning:
    - use close pct_change to detect pingpong + abs_cap
    - then recompute prev_close and leave other cols untouched
    """
    if df.empty or len(df) < 5:
        return df

    g = df.sort_values("date").copy()
    close = pd.to_numeric(g["close"], errors="coerce").astype(float)
    ret = close.pct_change()

    mask_abs = ret.abs() > abs_cap

    mask_pingpong = pd.Series(False, index=g.index)
    for i in range(1, len(g) - 1):
        prev = ret.iloc[i]
        nxt = ret.iloc[i + 1]
        if pd.notna(prev) and pd.notna(nxt):
            if (abs(prev) > threshold) and (abs(nxt) > threshold) and (prev * nxt < 0):
                mask_pingpong.iloc[i] = True
                mask_pingpong.iloc[i + 1] = True

    mask = mask_abs | mask_pingpong
    out = g.loc[~mask].copy()
    return out


def _clean_daily_with_data_cleaning(
    d: pd.DataFrame,
    enable_cleaning: bool,
    pingpong_threshold: float,
    abs_ret_cap: float,
) -> pd.DataFrame:
    """
    Try to call repo's data_cleaning.py.
    If unavailable, fallback to internal conservative cleaning.

    Important:
    - After cleaning, we ALWAYS recompute prev_close and d_logret safely.
    """
    if d.empty:
        return d

    d = d.copy()
    d["date"] = pd.to_datetime(d["date"], errors="coerce")
    d = d.dropna(subset=["date"]).sort_values("date").reset_index(drop=True)

    if enable_cleaning:
        try:
            # Expect your repo has something like:
            # from data_cleaning import clean_pingpong_daily
            # (naming may differ; we support multiple possible function names)
            import data_cleaning as dc  # type: ignore

            # pick a usable function if exists
            fn = None
            for cand in ["clean_pingpong_daily", "pingpong_clean_daily", "clean_daily_pingpong", "clean_k_data_daily"]:
                if hasattr(dc, cand):
                    fn = getattr(dc, cand)
                    break

            if fn is not None:
                d = fn(d, threshold=pingpong_threshold, abs_cap=abs_ret_cap)  # your data_cleaning should accept these
            else:
                # fallback
                d = _fallback_pingpong_clean_daily(d, threshold=pingpong_threshold, abs_cap=abs_ret_cap)

        except Exception:
            d = _fallback_pingpong_clean_daily(d, threshold=pingpong_threshold, abs_cap=abs_ret_cap)

    # ✅ recompute prev_close after potential deletions (critical!)
    d["prev_close"] = pd.to_numeric(d["close"], errors="coerce").shift(1)

    # ✅ recompute d_logret safely (first day uses open->close)
    d["d_logret"] = 0.0
    close = pd.to_numeric(d["close"], errors="coerce").astype(float)
    prevc = pd.to_numeric(d["prev_close"], errors="coerce").astype(float)
    openp = pd.to_numeric(d["open"], errors="coerce").astype(float)

    mask_cp = (close > 0) & (prevc > 0)
    d.loc[mask_cp, "d_logret"] = np.log(close[mask_cp] / prevc[mask_cp])

    # first row in this (symbol-year) segment: log(close/open)
    if len(d) > 0 and np.isfinite(close.iloc[0]) and np.isfinite(openp.iloc[0]) and close.iloc[0] > 0 and openp.iloc[0] > 0:
        d.loc[d.index[0], "d_logret"] = float(np.log(close.iloc[0] / openp.iloc[0]))

    # ensure is_limit_up int
    d["is_limit_up"] = pd.to_numeric(d["is_limit_up"], errors="coerce").fillna(0).astype(int)

    return d


# -----------------------------------------------------------------------------
# top week helpers
# -----------------------------------------------------------------------------
def _find_top_week_net(
    wps: pd.DataFrame,
    w_logrets: np.ndarray,
) -> Tuple[Optional[str], float, Optional[pd.Timestamp], Optional[pd.Timestamp]]:
    """
    回傳：
    - top_week_id_net
    - top_week_logret_net
    - top_week_start
    - top_week_end
    """
    if wps.empty or w_logrets.size == 0:
        return None, 0.0, None, None

    # nan-safe
    if not np.isfinite(w_logrets).any():
        return None, 0.0, None, None

    idx = int(np.nanargmax(w_logrets))
    top_row = wps.iloc[idx]
    top_id = str(top_row["week_id"]) if "week_id" in top_row else None
    top_lr = float(w_logrets[idx]) if np.isfinite(w_logrets[idx]) else 0.0

    ts = top_row.get("period_start", pd.NaT)
    te = top_row.get("period_end", pd.NaT)
    ts = pd.Timestamp(ts) if pd.notna(ts) else None
    te = pd.Timestamp(te) if pd.notna(te) else None
    return top_id, top_lr, ts, te


def _count_limitups_in_range(daily: pd.DataFrame, start: Optional[pd.Timestamp], end: Optional[pd.Timestamp]) -> int:
    if daily.empty or start is None or end is None:
        return 0
    d = daily.copy()
    d["date"] = pd.to_datetime(d["date"], errors="coerce")
    mask = (d["date"] >= start) & (d["date"] <= end)
    return int((d.loc[mask, "is_limit_up"] == 1).sum())


def _denom_year_from_mode(year_logret: float, mode: str) -> float:
    if not np.isfinite(year_logret):
        return 0.0
    if mode == DENOM_ABS:
        return float(abs(year_logret))
    # POS_ONLY
    return float(year_logret) if year_logret > 0 else 0.0


# -----------------------------------------------------------------------------
# core
# -----------------------------------------------------------------------------
def build_contribution_tables(
    db_path: str,
    only_markets: Optional[set] = None,
    enable_cleaning: bool = ENABLE_CLEANING_DEFAULT,
    pingpong_threshold: float = PINGPONG_THRESHOLD_DEFAULT,
    abs_ret_cap: float = ABS_DAILY_RET_CAP_DEFAULT,
    denom_mode: str = DENOM_POS_ONLY,  # POS_ONLY | ABS
) -> Dict[str, int]:
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
        y_open = y["y_open"].astype(float)
        y_close = y["y_close"].astype(float)
        y["year_ret_pct"] = (y_close / y_open - 1.0) * 100.0

        y["year_logret"] = np.log(y_close / y_open).replace([np.inf, -np.inf], np.nan).fillna(0.0)

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
        sa = sa.dropna(subset=["date"]).sort_values(["symbol", "date"]).reset_index(drop=True)
        sa["year"] = sa["date"].dt.year.astype(int)
        sa["is_limit_up"] = pd.to_numeric(sa["is_limit_up"], errors="coerce").fillna(0).astype(int)

        # （先不算 d_logret；我們會在 symbol-year 清洗後重算）
        rows = []

        for _, r in y.iterrows():
            sym = r["symbol"]
            yr = int(r["year"])

            year_open = float(r["y_open"])
            year_close = float(r["y_close"])
            year_logret = float(r["year_logret"])

            denom_year = _denom_year_from_mode(year_logret, denom_mode)

            # 取出該股該年的日K
            d0 = sa[(sa["symbol"] == sym) & (sa["year"] == yr)].copy()
            if d0.empty:
                continue

            # ✅ 清洗 + 重算 prev_close + d_logret（關鍵）
            d = _clean_daily_with_data_cleaning(
                d0,
                enable_cleaning=enable_cleaning,
                pingpong_threshold=pingpong_threshold,
                abs_ret_cap=abs_ret_cap,
            )
            if d.empty:
                continue

            # 年內最大回撤（用清洗後 close）
            year_max_dd_log = _max_drawdown_log_from_close(d["close"])

            # peak date 對齊（用清洗後交易日）
            peak_date_raw = r["year_peak_date"]
            peak_trade_date = _align_peak_trade_date(d["date"], peak_date_raw) if pd.notna(peak_date_raw) else None

            peak_close = np.nan
            if peak_trade_date is not None:
                d_peak = d[d["date"] == peak_trade_date]
                if not d_peak.empty:
                    peak_close = float(pd.to_numeric(d_peak.iloc[-1]["close"], errors="coerce"))

            peak_logret = _safe_log_ratio(peak_close, year_open) if np.isfinite(peak_close) else 0.0
            denom_peak = abs(peak_logret) if (np.isfinite(peak_logret) and peak_logret != 0) else 0.0

            # 從 peak 到年末回撤（log）
            peak_to_year_end_dd_log = 0.0
            if peak_trade_date is not None and np.isfinite(peak_close) and peak_close > 0:
                year_end_close = float(pd.to_numeric(d.iloc[-1]["close"], errors="coerce"))
                peak_to_year_end_dd_log = _safe_log_ratio(year_end_close, peak_close) if np.isfinite(year_end_close) else 0.0

            # 週/月 periods（該股該年）
            wps = wk[(wk["symbol"] == sym) & (wk["year"] == yr)].sort_values("period_end").reset_index(drop=True)
            mps = mk[(mk["symbol"] == sym) & (mk["year"] == yr)].sort_values("period_end").reset_index(drop=True)

            # 精準切段：日K切週/月（全年）
            w_logrets = _sum_logret_by_period(daily=d, periods=wps, cutoff_date=None)
            m_logrets = _sum_logret_by_period(daily=d, periods=mps, cutoff_date=None)

            worst_week_logret = float(np.nanmin(w_logrets)) if w_logrets.size > 0 and np.isfinite(w_logrets).any() else 0.0
            worst_month_logret = float(np.nanmin(m_logrets)) if m_logrets.size > 0 and np.isfinite(m_logrets).any() else 0.0

            # 週/月集中度（全年）
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

            # ✅ burst label（保留舊功能）
            burst_style_week = "ONE_WEEK_BURST" if top1_week_share_net >= 0.5 else "NON_CONCENTRATED"
            burst_style_month = "ONE_MONTH_BURST" if top1_month_share_net >= 0.5 else "NON_CONCENTRATED"

            # peak 前：日K到 peak
            logret_to_peak = 0.0
            if peak_trade_date is not None:
                logret_to_peak = float(d.loc[d["date"] <= peak_trade_date, "d_logret"].sum())
            else:
                # 沒 peak：用全年（避免除以 0）
                logret_to_peak = float(d["d_logret"].sum())

            share_year_to_peak = float(logret_to_peak / denom_year) if denom_year > 0 else 0.0
            share_peak_to_peak = float(logret_to_peak / denom_peak) if denom_peak > 0 else 0.0

            # peak 前：週/月切段到 peak
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

            # 漲停貢獻（peak 前）
            d_to_peak = d[d["date"] <= peak_trade_date] if peak_trade_date is not None else d
            limitup_count_to_peak = int((d_to_peak["is_limit_up"] == 1).sum())
            limitup_log_sum_to_peak = float(d_to_peak.loc[d_to_peak["is_limit_up"] == 1, "d_logret"].sum())
            limitup_log_share_to_peak_vs_year = float(limitup_log_sum_to_peak / denom_year) if denom_year > 0 else 0.0
            limitup_log_share_to_peak_vs_peak = float(limitup_log_sum_to_peak / denom_peak) if denom_peak > 0 else 0.0

            # -----------------------------------------------------------------
            # ✅ 新增：全年 top1 week (net) + 漲停密度
            # -----------------------------------------------------------------
            top_week_id_net, top_week_logret_net, top_week_start, top_week_end = _find_top_week_net(wps, w_logrets)
            top_week_is_limitup_dense = _count_limitups_in_range(d, top_week_start, top_week_end)

            # ✅ 新增：peak 前 top1 week(net)（只在 peak 前挑 top1 週）
            # 作法：用 w_logrets_to_peak（cutoff=peak_trade_date）再找 argmax
            top_week_id_net_to_peak, top_week_logret_net_to_peak, top_week_start_to_peak, top_week_end_to_peak = _find_top_week_net(
                wps, w_logrets_to_peak
            )
            top_week_is_limitup_dense_to_peak = _count_limitups_in_range(d, top_week_start_to_peak, top_week_end_to_peak)

            # ✅ peak 前「最大貢獻週」裡的漲停數
            # 定義：全年 top1 week(net) 若落在 peak_trade_date 之後 → 0（你要 peak 前）
            limitup_count_in_top1week_to_peak = 0
            if peak_trade_date is not None and top_week_end is not None:
                if top_week_end <= peak_trade_date:
                    limitup_count_in_top1week_to_peak = top_week_is_limitup_dense
                else:
                    limitup_count_in_top1week_to_peak = 0
            elif peak_trade_date is None:
                # 沒 peak：等同全年 top1 week
                limitup_count_in_top1week_to_peak = top_week_is_limitup_dense

            rows.append(
                {
                    "symbol": sym,
                    "year": yr,

                    # 年K
                    "y_open": year_open,
                    "y_close": year_close,
                    "year_ret_pct": float(r["year_ret_pct"]),
                    "year_logret": year_logret,

                    # denom (新增：讓你回頭檢查)
                    "denom_mode": denom_mode,
                    "denom_year_used": float(denom_year),

                    # bins
                    "year_ret_bin_100": r["year_ret_bin_100"],
                    "year_ret_bin_10_under100": r["year_ret_bin_10_under100"],
                    "year_ret_bin_50_under100": r["year_ret_bin_50_under100"],

                    # burst
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

                    # ✅ 新增欄位（全年 top week）
                    "top_week_id_net": top_week_id_net,
                    "top_week_logret_net": top_week_logret_net,
                    "top_week_is_limitup_dense": top_week_is_limitup_dense,

                    # ✅ 新增欄位（peak 前 top week）
                    "top_week_id_net_to_peak": top_week_id_net_to_peak,
                    "top_week_logret_net_to_peak": top_week_logret_net_to_peak,
                    "top_week_is_limitup_dense_to_peak": top_week_is_limitup_dense_to_peak,

                    # ✅ 新增欄位（你要的 peak 前最大貢獻週的漲停數）
                    "limitup_count_in_top1week_to_peak": int(limitup_count_in_top1week_to_peak),

                    # cleaning audit（方便 debug）
                    "cleaning_enabled": int(bool(enable_cleaning)),
                    "pingpong_threshold": float(pingpong_threshold),
                    "abs_daily_ret_cap": float(abs_ret_cap),
                    "n_daily_rows_after_clean": int(len(d)),
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
            conn.execute("CREATE INDEX IF NOT EXISTS idx_year_contrib_topweek ON year_contribution(top_week_id_net)")
            conn.execute("CREATE INDEX IF NOT EXISTS idx_year_contrib_topweek_peak ON year_contribution(top_week_id_net_to_peak)")
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

                    # ✅ 新增統計：top week net 與漲停密度（全年 + peak 前）
                    "avg_top_week_logret_net": float(df["top_week_logret_net"].mean()),
                    "avg_top_week_is_limitup_dense": float(df["top_week_is_limitup_dense"].mean()),
                    "avg_top_week_logret_net_to_peak": float(df["top_week_logret_net_to_peak"].mean()),
                    "avg_top_week_is_limitup_dense_to_peak": float(df["top_week_is_limitup_dense_to_peak"].mean()),
                    "avg_limitup_count_in_top1week_to_peak": float(df["limitup_count_in_top1week_to_peak"].mean()),

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

        print("\n✅ kbar_contribution（精準切段｜進階整合｜含清洗+denom+topweek）完成：")
        print(f"📌 year_contribution rows: {len(out):,}")
        print(f"📌 year_contribution_bins rows: {len(bins):,}")
        print("📌 新增：denom_mode/denom_year_used + top_week_* + top_week_*_to_peak + limitup_count_in_top1week_to_peak")
        print(f"📌 cleaning: enable={enable_cleaning}, pingpong={pingpong_threshold}, abs_cap={abs_ret_cap}")
        print(f"📌 denom_mode: {denom_mode}")

        return {"year_rows": int(len(out)), "bin_rows": int(len(bins))}

    finally:
        conn.close()


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python kbar_contribution.py <db_path> [denom_mode=POS_ONLY|ABS]")
        sys.exit(1)

    denom_mode = sys.argv[2].strip().upper() if len(sys.argv) >= 3 else DENOM_POS_ONLY
    if denom_mode not in (DENOM_POS_ONLY, DENOM_ABS):
        denom_mode = DENOM_POS_ONLY

    build_contribution_tables(
        sys.argv[1],
        enable_cleaning=True,
        pingpong_threshold=PINGPONG_THRESHOLD_DEFAULT,
        abs_ret_cap=ABS_DAILY_RET_CAP_DEFAULT,
        denom_mode=denom_mode,
    )
