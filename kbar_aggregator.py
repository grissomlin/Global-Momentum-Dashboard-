# kbar_aggregator.py
# -*- coding: utf-8 -*-
"""
kbar_aggregator.py
------------------
從 SQLite(stock_prices) 日K → 清洗 → 聚合產生 週K / 月K / 年K（寫回同一個 DB）

✅ 目標（支援你的儀表板/研究）：
- 產出 kbar_weekly / kbar_monthly / kbar_yearly 三張表
- 週/月/年K「同源一致」：全部由日K聚合（避免 yfinance 1wk 定義不一致）
- 內建「異常報酬清洗」：參考你貼的 pingpong 概念 + limit sanity check
- 額外提供 year_peak_date / year_peak_high：讓你能快速算
  「年K高點前有幾根漲停」、「週/月對年K高點貢獻度」等

📌 依賴：
- pandas, numpy（都在你的環境中）
- SQLite 表：
  - stock_prices(symbol,date,open,high,low,close,volume)
  - stock_info(symbol,market,market_detail,sector,name...)  (可選，但建議有)

⚠️ 注意：
- 這支腳本不依賴 yfinance，不會額外抓資料
- 若你 downloader 已使用 auto_adjust=True，close 已接近還原價，這裡的清洗會更可靠

用法：
    python kbar_aggregator.py tw_stock_warehouse.db
或在程式裡呼叫：
    from kbar_aggregator import build_kbars
    build_kbars("tw_stock_warehouse.db")

"""

import sys
import sqlite3
import warnings
from dataclasses import dataclass
from typing import Optional, Dict, Tuple

import numpy as np
import pandas as pd

warnings.filterwarnings("ignore", category=FutureWarning)


# =============================================================================
# 可選：接 market_rules（若存在就用它的 limit/tick 規則做更精準清洗）
# =============================================================================
try:
    import market_rules  # 你的規則檔（若已完成）
    HAS_MARKET_RULES = True
except Exception:
    market_rules = None
    HAS_MARKET_RULES = False


# =============================================================================
# 設定
# =============================================================================
PINGPONG_THRESHOLD = 0.40     # 你貼的：連續兩日 abs(ret)>0.4 且反向 → 異常
LIMIT_SANITY_MULT = 1.50      # 有漲跌幅限制市場：abs(ret) > limit*1.5 視為異常
MIN_DAYS_PER_SYMBOL = 40      # 太短的不做
SQLITE_TIMEOUT = 120


@dataclass
class LimitRule:
    kind: str                 # 'pct' / 'none'
    up_pct: Optional[float]   # 0.10 / 0.20 / None


def _fallback_limit_rule(market: confirm str, market_detail: str, symbol: str) -> LimitRule:
    """
    若 market_rules 不存在時的保底判斷：
    - TW: listed/otc 10%, emerging none
    - CN: 300/301/688 20% else 10%
    - JP: none (你之後用 market_rules 補精準値幅)
    """
    m = (market or "").upper().strip()
    md = (market_detail or "").lower().strip()
    sym = (symbol or "").upper().strip()

    # TW
    if m in ["TW", "TSE", "GTSM"] or sym.endswith(".TW") or sym.endswith(".TWO"):
        if md == "emerging":
            return LimitRule(kind="none", up_pct=None)
        return LimitRule(kind="pct", up_pct=0.10)

    # CN
    if m in ["SSE", "SZSE", "CN", "CHINA"] or sym.endswith(".SS") or sym.endswith(".SZ"):
        code = "".join([c for c in sym if c.isdigit()])
        if code.startswith(("300", "301", "688")):
            return LimitRule(kind="pct", up_pct=0.20)
        return LimitRule(kind="pct", up_pct=0.10)

    # JP
    if m in ["JP", "JPX", "TSE"] or sym.endswith(".T"):
        return LimitRule(kind="none", up_pct=None)

    return LimitRule(kind="none", up_pct=None)


def _get_limit_rule(market: str, market_detail: str, symbol: str) -> LimitRule:
    """
    盡量用 market_rules.get_rule()，否則 fallback。
    只拿「是否 pct limit」與「上限」用於 sanity check。
    """
    if HAS_MARKET_RULES and hasattr(market_rules, "get_rule"):
        try:
            r = market_rules.get_rule(market=market, market_detail=market_detail, symbol=symbol)
            kind = r.get("limit_kind", "none")
            up = r.get("limit_up_pct", None)
            if kind == "pct" and isinstance(up, (int, float)):
                return LimitRule(kind="pct", up_pct=float(up))
            return LimitRule(kind="none", up_pct=None)
        except Exception:
            pass
    return _fallback_limit_rule(market, market_detail, symbol)


# =============================================================================
# 清洗：pingpong + limit sanity + 基礎修補
# =============================================================================
def _clean_daily(df: pd.DataFrame, limit_rule: LimitRule) -> pd.DataFrame:
    """
    df: 單一 symbol 的日K，需包含 date/open/high/low/close/volume
    回傳清洗後 df（仍保持日頻），並新增 clean_ret
    """

    if df.empty:
        return df

    df = df.sort_values("date").reset_index(drop=True).copy()

    # 基礎：價量無效
    for c in ["open", "high", "low", "close"]:
        df.loc[df[c].astype(float) <= 0, c] = np.nan

    # 修補 close（因為 ret 依賴 close）
    df["close"] = df["close"].astype(float).ffill()

    # 用 close 算報酬
    df["clean_ret"] = df["close"].pct_change().astype(float)

    # (1) limit sanity check（有漲跌幅限制市場）
    if limit_rule.kind == "pct" and limit_rule.up_pct is not None:
        max_abs = float(limit_rule.up_pct) * LIMIT_SANITY_MULT
        bad = df["clean_ret"].abs() > max_abs
        # 這些日子視為異常：把 OHLC 全設 NaN，再用 close ffill 讓聚合不炸
        if bad.any():
            for c in ["open", "high", "low", "close"]:
                df.loc[bad, c] = np.nan
            df["close"] = df["close"].ffill()
            df["clean_ret"] = df["close"].pct_change().astype(float)

    # (2) pingpong filter（你貼的精神）
    # 若 i 與 i+1 連續兩日 abs(ret)>threshold 且 ret 方向相反 → i, i+1 標記異常
    r = df["clean_ret"].values
    mask = np.zeros(len(df), dtype=bool)
    for i in range(1, len(df) - 1):
        prev = r[i]
        nxt = r[i + 1]
        if np.isfinite(prev) and np.isfinite(nxt):
            if (abs(prev) > PINGPONG_THRESHOLD) and (abs(nxt) > PINGPONG_THRESHOLD) and (prev * nxt < 0):
                mask[i] = True
                mask[i + 1] = True

    if mask.any():
        for c in ["open", "high", "low", "close"]:
            df.loc[mask, c] = np.nan
        df["close"] = df["close"].ffill()
        df["clean_ret"] = df["close"].pct_change().astype(float)

    # 重新補 open/high/low（保守：用 close 近似補洞，確保聚合不中斷）
    # 你若更想嚴格，可以改成：只 ffill close，不補 open/high/low，但聚合可能缺資料
    df["open"] = df["open"].astype(float)
    df["high"] = df["high"].astype(float)
    df["low"] = df["low"].astype(float)

    df["open"] = df["open"].fillna(df["close"])
    df["high"] = df["high"].fillna(df[["open", "close"]].max(axis=1))
    df["low"] = df["low"].fillna(df[["open", "close"]].min(axis=1))

    # volume
    if "volume" in df.columns:
        df["volume"] = pd.to_numeric(df["volume"], errors="coerce").fillna(0).astype(float)
    else:
        df["volume"] = 0.0

    return df


# =============================================================================
# 聚合：由日K生成 週/月/年
# =============================================================================
def _agg_ohlcv(g: pd.DataFrame) -> pd.Series:
    """對單一 period 的 OHLCV 聚合"""
    if g.empty:
        return pd.Series({"open": np.nan, "high": np.nan, "low": np.nan, "close": np.nan, "volume": 0.0})

    return pd.Series(
        {
            "open": float(g["open"].iloc[0]),
            "high": float(np.nanmax(g["high"].values)),
            "low": float(np.nanmin(g["low"].values)),
            "close": float(g["close"].iloc[-1]),
            "volume": float(np.nansum(g["volume"].values)),
        }
    )


def _build_weekly(df: pd.DataFrame) -> pd.DataFrame:
    """
    週K：以 ISO week 做週期鍵（跨年週會歸到 ISO year）
    period_end：週最後一個交易日（實際存在的日K最後一天）
    """
    x = df.copy()
    iso = x["date"].dt.isocalendar()
    x["iso_year"] = iso["year"].astype(int)
    x["iso_week"] = iso["week"].astype(int)
    x["period_key"] = x["iso_year"].astype(str) + "-W" + x["iso_week"].astype(str).str.zfill(2)

    out = (
        x.groupby("period_key", sort=True)
        .apply(_agg_ohlcv)
        .reset_index()
        .rename(columns={"period_key": "week_id"})
    )

    # start/end date
    se = x.groupby("period_key")["date"].agg(["min", "max"]).reset_index()
    se.columns = ["week_id", "period_start", "period_end"]
    out = out.merge(se, on="week_id", how="left")

    # year/week
    out["year"] = out["week_id"].str.slice(0, 4).astype(int)
    out["week"] = out["week_id"].str.split("-W").str[1].astype(int)

    return out[["week_id", "year", "week", "period_start", "period_end", "open", "high", "low", "close", "volume"]]


def _build_monthly(df: pd.DataFrame) -> pd.DataFrame:
    x = df.copy()
    x["year"] = x["date"].dt.year.astype(int)
    x["month"] = x["date"].dt.month.astype(int)
    x["month_id"] = x["year"].astype(str) + "-" + x["month"].astype(str).str.zfill(2)

    out = x.groupby("month_id", sort=True).apply(_agg_ohlcv).reset_index()
    se = x.groupby("month_id")["date"].agg(["min", "max"]).reset_index()
    se.columns = ["month_id", "period_start", "period_end"]
    out = out.merge(se, on="month_id", how="left")

    out["year"] = out["month_id"].str.slice(0, 4).astype(int)
    out["month"] = out["month_id"].str.slice(5, 7).astype(int)

    return out[["month_id", "year", "month", "period_start", "period_end", "open", "high", "low", "close", "volume"]]


def _build_yearly(df: pd.DataFrame) -> pd.DataFrame:
    x = df.copy()
    x["year"] = x["date"].dt.year.astype(int)
    x["year_id"] = x["year"].astype(str)

    out = x.groupby("year_id", sort=True).apply(_agg_ohlcv).reset_index()
    se = x.groupby("year_id")["date"].agg(["min", "max"]).reset_index()
    se.columns = ["year_id", "period_start", "period_end"]
    out = out.merge(se, on="year_id", how="left")

    out["year"] = out["year_id"].astype(int)

    # 年內高點（用 high）
    peak = x.groupby("year_id").apply(lambda g: pd.Series({
        "year_peak_date": g.loc[g["high"].astype(float).idxmax(), "date"] if len(g) else pd.NaT,
        "year_peak_high": float(np.nanmax(g["high"].astype(float).values)) if len(g) else np.nan
    })).reset_index().rename(columns={"year_id": "year_id"})

    out = out.merge(peak, left_on="year_id", right_on="year_id", how="left")
    return out[["year_id", "year", "period_start", "period_end", "open", "high", "low", "close", "volume", "year_peak_date", "year_peak_high"]]


# =============================================================================
# DB IO
# =============================================================================
def _ensure_indexes(conn: sqlite3.Connection):
    # 日K索引（如果沒有）
    try:
        conn.execute("CREATE INDEX IF NOT EXISTS idx_prices_symbol_date ON stock_prices(symbol, date)")
    except Exception:
        pass


def _write_table(conn: sqlite3.Connection, name: str, df: pd.DataFrame):
    conn.execute(f"DROP TABLE IF EXISTS {name}")
    df.to_sql(name, conn, if_exists="replace", index=False)

    # 常用索引
    try:
        if name == "kbar_weekly":
            conn.execute("CREATE INDEX IF NOT EXISTS idx_kbar_weekly_symbol_end ON kbar_weekly(symbol, period_end)")
            conn.execute("CREATE INDEX IF NOT EXISTS idx_kbar_weekly_symbol_week ON kbar_weekly(symbol, week_id)")
        elif name == "kbar_monthly":
            conn.execute("CREATE INDEX IF NOT EXISTS idx_kbar_monthly_symbol_end ON kbar_monthly(symbol, period_end)")
            conn.execute("CREATE INDEX IF NOT EXISTS idx_kbar_monthly_symbol_month ON kbar_monthly(symbol, month_id)")
        elif name == "kbar_yearly":
            conn.execute("CREATE INDEX IF NOT EXISTS idx_kbar_yearly_symbol_year ON kbar_yearly(symbol, year)")
            conn.execute("CREATE INDEX IF NOT EXISTS idx_kbar_yearly_symbol_end ON kbar_yearly(symbol, period_end)")
    except Exception:
        pass


# =============================================================================
# 主函數
# =============================================================================
def build_kbars(db_path: str, symbols: Optional[list] = None) -> Dict[str, int]:
    """
    讀 stock_prices → 清洗 → 聚合 → 寫回 kbar_weekly/monthly/yearly
    回傳統計 dict
    """

    conn = sqlite3.connect(db_path, timeout=SQLITE_TIMEOUT)
    try:
        _ensure_indexes(conn)

        # 讀 stock_prices + stock_info（market判定用）
        query = """
        SELECT p.symbol, p.date, p.open, p.high, p.low, p.close, p.volume,
               i.market, i.market_detail
        FROM stock_prices p
        LEFT JOIN stock_info i ON p.symbol = i.symbol
        """
        df = pd.read_sql(query, conn)

        if df.empty:
            print("❌ stock_prices 無資料")
            return {"symbols": 0, "weekly_rows": 0, "monthly_rows": 0, "yearly_rows": 0}

        df["date"] = pd.to_datetime(df["date"], errors="coerce")
        df = df.dropna(subset=["date"]).sort_values(["symbol", "date"]).reset_index(drop=True)

        if symbols:
            sset = set(symbols)
            df = df[df["symbol"].isin(sset)].copy()
            if df.empty:
                print("❌ 指定 symbols 在 DB 找不到資料")
                return {"symbols": 0, "weekly_rows": 0, "monthly_rows": 0, "yearly_rows": 0}

        wk_list, mo_list, yr_list = [], [], []
        symbol_count = 0

        for sym, g in df.groupby("symbol", sort=False):
            g = g.sort_values("date").reset_index(drop=True)
            if len(g) < MIN_DAYS_PER_SYMBOL:
                continue

            market = g["market"].iloc[0] if "market" in g.columns else ""
            market_detail = g["market_detail"].iloc[0] if "market_detail" in g.columns else ""

            rule = _get_limit_rule(market, market_detail, sym)

            gd = _clean_daily(g, rule)
            if gd.empty or gd["close"].isna().all():
                continue

            # 聚合
            w = _build_weekly(gd)
            w.insert(0, "symbol", sym)
            m = _build_monthly(gd)
            m.insert(0, "symbol", sym)
            y = _build_yearly(gd)
            y.insert(0, "symbol", sym)

            wk_list.append(w)
            mo_list.append(m)
            yr_list.append(y)

            symbol_count += 1

        if symbol_count == 0:
            print("❌ 沒有足夠資料可聚合（可能都不足 MIN_DAYS_PER_SYMBOL）")
            return {"symbols": 0, "weekly_rows": 0, "monthly_rows": 0, "yearly_rows": 0}

        df_wk = pd.concat(wk_list, ignore_index=True)
        df_mo = pd.concat(mo_list, ignore_index=True)
        df_yr = pd.concat(yr_list, ignore_index=True)

        # 日期欄位轉字串（SQLite穩）
        for col in ["period_start", "period_end"]:
            df_wk[col] = pd.to_datetime(df_wk[col]).dt.strftime("%Y-%m-%d")
            df_mo[col] = pd.to_datetime(df_mo[col]).dt.strftime("%Y-%m-%d")
            df_yr[col] = pd.to_datetime(df_yr[col]).dt.strftime("%Y-%m-%d")

        df_yr["year_peak_date"] = pd.to_datetime(df_yr["year_peak_date"], errors="coerce").dt.strftime("%Y-%m-%d")

        # 寫回
        _write_table(conn, "kbar_weekly", df_wk)
        _write_table(conn, "kbar_monthly", df_mo)
        _write_table(conn, "kbar_yearly", df_yr)

        conn.commit()

        print("\n✅ kbar 聚合完成（由日K聚合，已清洗）")
        print(f"📌 symbols: {symbol_count}")
        print(f"📌 kbar_weekly rows:  {len(df_wk):,}")
        print(f"📌 kbar_monthly rows: {len(df_mo):,}")
        print(f"📌 kbar_yearly rows:  {len(df_yr):,}")

        return {
            "symbols": int(symbol_count),
            "weekly_rows": int(len(df_wk)),
            "monthly_rows": int(len(df_mo)),
            "yearly_rows": int(len(df_yr)),
        }

    finally:
        conn.close()


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python kbar_aggregator.py <db_path> [symbol1 symbol2 ...]")
        sys.exit(1)

    db = sys.argv[1]
    syms = sys.argv[2:] if len(sys.argv) > 2 else None
    build_kbars(db, symbols=syms)
