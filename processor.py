# processor.py
# -*- coding: utf-8 -*-
"""
processor.py
------------
Feature Layer（寫回 stock_analysis）

✅ 目標：
- 只負責把 is_limit_up / strength_rank(10%起每10%分箱到100%+) / lu_type / consecutive_limits
  + 技術指標 + 年度巔峰貢獻度 等 features 寫回 stock_analysis

✅ 設計原則：
- 市場規則集中在 market_rules.py（TW/CN/JP 的 limit 判定 + tick + 分箱 intervals）
- processor.py 不硬寫各市場規則：盡量透過 market_rules.get_rule(...) / market_rules.calc_limit_up_price(...)
- 若 market_rules.py 尚未完成，processor 仍有 fallback（可跑，但 TW/JP 漲停精準度會較差）

⚠️ 注意：
- 事件表（漲停型態 / 隔日沖 / 未來報酬）請放 event_engine.py
  processor.py 不做事件表、不做 future returns。
"""

import sqlite3
import pandas as pd
import numpy as np
import warnings

warnings.filterwarnings("ignore", category=FutureWarning)

SQLITE_TIMEOUT = 120


# =============================================================================
# 0) 匯入市場規則（主路徑），若沒有則 fallback
# =============================================================================
try:
    import market_rules  # 你會另外提供：TW/CN/JP limit 判定 + tick + bins
    HAS_MARKET_RULES = True
except Exception:
    market_rules = None
    HAS_MARKET_RULES = False


# =============================================================================
# 1) Fallback 規則（只有當 market_rules.py 不存在時才用）
# =============================================================================
def _fallback_get_rule(market: str, market_detail: str, symbol: str) -> dict:
    """
    回傳 dict：
    - limit_kind: 'pct' / 'none'
    - limit_up_pct: float or None
    - threshold: float（強勢日門檻，給 peak_contribution 用）
    - strength_edges: list of edges in % for pd.cut
    - strength_labels: labels for bins
    - max_strength: int
    """
    m = (market or "").upper().strip()
    md = (market_detail or "").lower().strip()
    sym = (symbol or "").upper().strip()

    edges = [0, 10, 20, 30, 40, 50, 60, 70, 80, 90, 100, np.inf]
    labels = [
        "RANK_0_10", "RANK_10_20", "RANK_20_30", "RANK_30_40", "RANK_40_50",
        "RANK_50_60", "RANK_60_70", "RANK_70_80", "RANK_80_90", "RANK_90_100", "RANK_100UP",
    ]

    # --- TW ---
    if m in ["TW", "TSE", "GTSM"] or sym.endswith(".TW") or sym.endswith(".TWO"):
        if md == "emerging":
            # 興櫃：無漲跌幅限制（fallback：不做漲停判定）
            return dict(limit_kind="none", limit_up_pct=None, threshold=0.20,
                        strength_edges=edges, strength_labels=labels, max_strength=100)
        else:
            # 上市/上櫃：10% 漲停（fallback：不做 tick 對齊）
            return dict(limit_kind="pct", limit_up_pct=0.10, threshold=0.10,
                        strength_edges=edges, strength_labels=labels, max_strength=100)

    # --- CN ---
    if m in ["SSE", "SZSE", "CN", "CHINA"] or sym.endswith(".SS") or sym.endswith(".SZ"):
        # 300/301/688 => 20%（創業板/科創板），其他 10%
        code = "".join([c for c in sym if c.isdigit()])
        up = 0.20 if code.startswith(("300", "301", "688")) else 0.10
        return dict(limit_kind="pct", limit_up_pct=up, threshold=up,
                    strength_edges=edges, strength_labels=labels, max_strength=100)

    # --- JP (fallback：當作無漲跌幅限制，不算漲停) ---
    if m in ["JP", "TSE", "JPX"] or sym.endswith(".T"):
        return dict(limit_kind="none", limit_up_pct=None, threshold=0.10,
                    strength_edges=edges, strength_labels=labels, max_strength=100)

    # --- Default ---
    return dict(limit_kind="none", limit_up_pct=None, threshold=0.10,
                strength_edges=edges, strength_labels=labels, max_strength=100)


def _fallback_calc_limit_up_price(prev_close: pd.Series, limit_up_pct: float) -> pd.Series:
    # fallback：不對齊 tick
    return (prev_close * (1 + limit_up_pct)).round(2)


# =============================================================================
# 2) 共用工具：分箱、連板、LU 型態（你文章規則）
# =============================================================================
def _make_strength_bins(change_pct: pd.Series, edges, labels) -> pd.Series:
    """
    change_pct：百分比（例如 12.3 表示 +12.3%）
    edges：例如 [0,10,20,...,100,inf]
    labels：對應 bins
    """
    out = pd.cut(change_pct, bins=edges, labels=labels, right=False, include_lowest=True)
    out = out.astype("object")
    out = np.where(change_pct <= 0, "NEGATIVE", out)
    out = np.where((change_pct > 0) & (change_pct < edges[1]), "POSITIVE", out)  # 0~10% 正值
    return pd.Series(out, index=change_pct.index)


def _strength_value_from_rank(rank: pd.Series) -> pd.Series:
    """
    取分箱的數值表示（例如 RANK_30_40 => 30, RANK_100UP => 100, POSITIVE=>1, NEGATIVE=>0）
    """
    def _v(x):
        if x in ("NEGATIVE", None) or (isinstance(x, float) and np.isnan(x)):
            return 0
        if x == "POSITIVE":
            return 1
        if isinstance(x, str) and x.startswith("RANK_"):
            if x.endswith("UP"):
                digits = "".join([c for c in x if c.isdigit()])
                return int(digits) if digits else 0
            parts = x.replace("RANK_", "").split("_")
            try:
                return int(parts[0])
            except Exception:
                return 0
        return 0

    return rank.apply(_v)


def _compute_consecutive_limits(is_limit_up: pd.Series) -> pd.Series:
    """
    連板：只在 is_limit_up==1 時計算連續天數，其他為 0
    """
    grp = (is_limit_up != is_limit_up.shift(1)).cumsum()
    streak = is_limit_up.groupby(grp).cumsum()
    return np.where(is_limit_up == 1, streak, 0)


def _compute_lu_type_article_style(
    is_limit_up: pd.Series,
    open_: pd.Series,
    high: pd.Series,
    low: pd.Series,
    close: pd.Series,
    prev_close: pd.Series,
    volume: pd.Series,
    vol_ma5: pd.Series,
) -> pd.Series:
    """
    依你文章的「自動化分類」：
    1) is_gap  : (Open/Prev_Close - 1) >= 0.07
    2) is_high : Volume/Vol_MA5 >= 3
    3) is_low  : Volume/Vol_MA5 <= 0.4
    4) is_float: 非 gap 且 (Close/Open - 1) >= 0.05
    優先序：
    1. GAP_UP_LOCK   ：is_gap 且 is_low_vol
    2. GAP_UP        ：is_gap 且 非 is_low_vol
    3. FLOAT_HV      ：is_float 且 is_high_vol
    4. FLOAT         ：is_float 且 非 is_high_vol
    5. LOW_VOL_LOCK  ：is_low_vol
    6. HIGH_VOL_LOCK ：is_high_vol
    7. OTHER
    合併成五類：
      FLOATING / GAP_UP / OTHER / HIGH_VOLUME_LOCK / NO_VOLUME_LOCK
    """
    safe_prev = prev_close.replace(0, np.nan)
    safe_vol_ma5 = vol_ma5.replace(0, np.nan)
    safe_open = open_.replace(0, np.nan)

    gap = (open_ / safe_prev - 1) >= 0.07
    vol_ratio = volume / safe_vol_ma5
    high_vol = vol_ratio >= 3
    low_vol = vol_ratio <= 0.4
    floating = (~gap) & ((close / safe_open - 1) >= 0.05)

    cat = pd.Series("OTHER", index=is_limit_up.index, dtype="object")
    cat = np.where(gap & low_vol, "GAP_UP_LOCK", cat)
    cat = np.where(gap & (~low_vol), "GAP_UP", cat)
    cat = np.where(floating & high_vol, "FLOAT_HV", cat)
    cat = np.where(floating & (~high_vol), "FLOAT", cat)
    cat = np.where((~gap) & (~floating) & low_vol, "LOW_VOL_LOCK", cat)
    cat = np.where((~gap) & (~floating) & high_vol, "HIGH_VOL_LOCK", cat)

    merged = pd.Series("OTHER", index=is_limit_up.index, dtype="object")
    merged = np.where(np.isin(cat, ["FLOAT", "FLOAT_HV"]), "FLOATING", merged)
    merged = np.where(np.isin(cat, ["GAP_UP", "GAP_UP_LOCK"]), "GAP_UP", merged)
    merged = np.where(cat == "HIGH_VOL_LOCK", "HIGH_VOLUME_LOCK", merged)
    merged = np.where(cat == "LOW_VOL_LOCK", "NO_VOLUME_LOCK", merged)

    merged = np.where(is_limit_up == 1, merged, None)
    return pd.Series(merged, index=is_limit_up.index, dtype="object")


# =============================================================================
# 3) 主流程
# =============================================================================
def process_market_data(db_path: str):
    conn = sqlite3.connect(db_path, timeout=SQLITE_TIMEOUT)

    # 1) 讀取數據並關聯 stock_info 取得市場與產業別
    query = """
    SELECT p.*, i.market, i.sector, i.market_detail
    FROM stock_prices p
    LEFT JOIN stock_info i ON p.symbol = i.symbol
    """
    df = pd.read_sql(query, conn)

    if df.empty:
        print("❌ 沒有找到股票數據")
        conn.close()
        return

    # 必要欄位保底（避免某些市場缺 low/high/volume）
    for c in ["open", "high", "low", "close", "volume"]:
        if c not in df.columns:
            df[c] = np.nan

    if "market" not in df.columns:
        df["market"] = ""
    if "market_detail" not in df.columns:
        df["market_detail"] = ""
    if "sector" not in df.columns:
        df["sector"] = None

    # 型別清理
    df["date"] = pd.to_datetime(df["date"], errors="coerce")
    df = df.dropna(subset=["date"])
    df = df.sort_values(["symbol", "date"]).reset_index(drop=True)

    for c in ["open", "high", "low", "close", "volume"]:
        df[c] = pd.to_numeric(df[c], errors="coerce")

    processed_list = []

    for symbol, group in df.groupby("symbol", sort=False):
        group = group.copy().sort_values("date").reset_index(drop=True)

        # 太短不做（避免指標噪聲很大）
        if len(group) < 40:
            continue

        market = str(group["market"].iloc[0]) if "market" in group.columns else ""
        market_detail = str(group["market_detail"].iloc[0]) if "market_detail" in group.columns else ""

        # 取得市場規則（以 market_rules.py 為主）
        if HAS_MARKET_RULES and hasattr(market_rules, "get_rule"):
            rule = market_rules.get_rule(market=market, market_detail=market_detail, symbol=symbol)
        else:
            rule = _fallback_get_rule(market, market_detail, symbol)

        # --- 基礎欄位 ---
        group["prev_close"] = group["close"].shift(1)
        group["daily_change"] = group["close"].pct_change()
        group["avg_vol_20"] = group["volume"].rolling(window=20, min_periods=1).mean()
        group["vol_ma5"] = group["volume"].rolling(window=5, min_periods=1).mean()
        group["year"] = group["date"].dt.year

        # --- 漲幅百分比 ---
        change_pct = (group["daily_change"] * 100).astype(float)

        # --- strength_rank / strength_value（10%起每10%到100%+） ---
        edges = rule.get("strength_edges")
        labels = rule.get("strength_labels")
        if edges is None or labels is None:
            edges = [0, 10, 20, 30, 40, 50, 60, 70, 80, 90, 100, np.inf]
            labels = [
                "RANK_0_10", "RANK_10_20", "RANK_20_30", "RANK_30_40", "RANK_40_50",
                "RANK_50_60", "RANK_60_70", "RANK_70_80", "RANK_80_90", "RANK_90_100", "RANK_100UP",
            ]

        group["strength_rank"] = _make_strength_bins(change_pct, edges, labels)
        group["strength_value"] = _strength_value_from_rank(group["strength_rank"]).astype(int)

        # --- 漲停判定 is_limit_up ---
        group["is_limit_up"] = 0
        limit_kind = rule.get("limit_kind", "none")
        limit_up_pct = rule.get("limit_up_pct", None)

        # market_rules 精準版（TW tick / JP 値幅制限）
        used_precise = False
        if HAS_MARKET_RULES and hasattr(market_rules, "calc_limit_up_price"):
            try:
                limit_price = market_rules.calc_limit_up_price(
                    prev_close=group["prev_close"].astype(float),
                    market=market,
                    market_detail=market_detail,
                    symbol=symbol,
                )
                if limit_price is not None:
                    if hasattr(market_rules, "tick_size"):
                        tick = group["prev_close"].astype(float).apply(
                            lambda x: market_rules.tick_size(float(x), market=market, symbol=symbol)
                        )
                        buffer = tick.fillna(0) * 0.5
                    else:
                        buffer = 0.0
                    group["is_limit_up"] = (group["close"].astype(float) >= (limit_price.astype(float) - buffer)).astype(int)
                    used_precise = True
            except Exception:
                used_precise = False

        # fallback：固定百分比（僅在精準判定沒生效時）
        if (not used_precise) and limit_kind == "pct" and isinstance(limit_up_pct, (int, float)):
            limit_price = _fallback_calc_limit_up_price(group["prev_close"].astype(float), float(limit_up_pct))
            group["is_limit_up"] = (group["close"].astype(float) >= limit_price * 0.999).astype(int)

        # 其餘：無漲跌幅限制（processor 不把 10% 當事件漲停；事件在 event_engine）
        synth = rule.get("synthetic_limit_up_pct", None)
        if synth is not None and group["is_limit_up"].sum() == 0:
            group["is_limit_up"] = (group["daily_change"].astype(float) >= float(synth)).astype(int)

        # --- 一字鎖 ---
        group["is_one_tick_lock"] = (
            (group["open"] == group["close"]) &
            (group["high"] == group["low"]) &
            (group["high"] == group["close"])
        ).astype(int)

        # --- LU 型態（只在漲停日給類型） ---
        group["lu_type"] = _compute_lu_type_article_style(
            is_limit_up=group["is_limit_up"],
            open_=group["open"].astype(float),
            high=group["high"].astype(float),
            low=group["low"].astype(float),
            close=group["close"].astype(float),
            prev_close=group["prev_close"].astype(float),
            volume=group["volume"].astype(float),
            vol_ma5=group["vol_ma5"].astype(float),
        )

        # --- 連板次數 ---
        group["consecutive_limits"] = _compute_consecutive_limits(group["is_limit_up"]).astype(int)

        # --- 年度巔峰貢獻度（用 rule['threshold']） ---
        threshold = float(rule.get("threshold", 0.10))

        def calc_peak_contribution(df_year: pd.DataFrame) -> pd.DataFrame:
            if df_year.empty:
                df_year["peak_date"] = None
                df_year["peak_high_ret"] = np.nan
                df_year["strong_day_contribution"] = 0.0
                return df_year

            valid_high = pd.to_numeric(df_year["high"], errors="coerce").dropna()
            if valid_high.empty:
                df_year["peak_date"] = None
                df_year["peak_high_ret"] = np.nan
                df_year["strong_day_contribution"] = 0.0
                return df_year

            peak_idx = valid_high.idxmax()
            peak_date = df_year.loc[peak_idx, "date"] if peak_idx in df_year.index else None
            peak_price = df_year.loc[peak_idx, "high"] if peak_idx in df_year.index else np.nan

            year_open = df_year.iloc[0]["open"] if len(df_year) > 0 else np.nan

            if pd.notna(peak_price) and pd.notna(year_open) and float(year_open) > 0:
                total_peak_log = float(np.log(float(peak_price) / float(year_open)))
            else:
                total_peak_log = 0.0

            if peak_date is not None:
                mask_before = (df_year["date"] <= peak_date)
            else:
                mask_before = pd.Series(False, index=df_year.index)

            # logret：close/prev_close（避免 prev_close=0）
            safe_prev = pd.to_numeric(df_year["prev_close"], errors="coerce").replace(0, np.nan)
            safe_close = pd.to_numeric(df_year["close"], errors="coerce").replace(0, np.nan)
            daily_logs = np.log(safe_close / safe_prev).replace([np.inf, -np.inf], np.nan).fillna(0.0)

            strong_day_mask = (pd.to_numeric(df_year["daily_change"], errors="coerce").fillna(0.0) >= threshold) & mask_before

            if strong_day_mask.any() and total_peak_log > 0:
                strong_contribution = float(daily_logs[strong_day_mask].sum())
                strong_day_contribution = float(strong_contribution / total_peak_log * 100)
            else:
                strong_day_contribution = 0.0

            df_year["peak_date"] = peak_date
            df_year["peak_high_ret"] = (
                (float(peak_price) - float(year_open)) / float(year_open) * 100
                if pd.notna(peak_price) and pd.notna(year_open) and float(year_open) > 0
                else np.nan
            )
            df_year["strong_day_contribution"] = strong_day_contribution
            return df_year

        year_values = group["year"].copy()
        try:
            group = group.groupby("year", group_keys=False).apply(calc_peak_contribution, include_groups=False)
        except TypeError:
            group = group.groupby("year", group_keys=False).apply(calc_peak_contribution)

        if "year" not in group.columns:
            group["year"] = year_values

        # --- 技術指標 ---
        group["ma20"] = group["close"].rolling(window=20, min_periods=1).mean()
        group["ma60"] = group["close"].rolling(window=60, min_periods=1).mean()

        ema12 = group["close"].ewm(span=12, adjust=False).mean()
        ema26 = group["close"].ewm(span=26, adjust=False).mean()
        group["macd"] = ema12 - ema26
        group["macds"] = group["macd"].ewm(span=9, adjust=False).mean()
        group["macdh"] = group["macd"] - group["macds"]

        # 年化波動率（20D）
        group["volatility_20"] = group["daily_change"].rolling(window=20, min_periods=1).std() * np.sqrt(252)

        # RSI 14
        delta = group["close"].diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=14, min_periods=1).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=14, min_periods=1).mean()
        rs = gain / loss.replace(0, np.nan)
        group["rsi"] = 100 - (100 / (1 + rs))

        group["volume_ratio"] = group["volume"] / group["avg_vol_20"].replace(0, np.nan)

        rolling_20_high = group["high"].rolling(window=20, min_periods=1).max()
        rolling_20_low = group["low"].rolling(window=20, min_periods=1).min()
        denom = (rolling_20_high - rolling_20_low).replace(0, np.nan)
        group["price_position_20"] = (group["close"] - rolling_20_low) / denom

        # YTD Ret（用收盤）
        year_start_prices = group.groupby("year")["close"].first()
        year_to_start = year_start_prices.to_dict()
        group["year_start_price"] = group["year"].map(year_to_start)
        group["ytd_ret"] = ((group["close"] - group["year_start_price"]) / group["year_start_price"] * 100).round(2)

        processed_list.append(group)

    if not processed_list:
        print("❌ 沒有處理後的數據（可能是資料太少或欄位缺失）")
        conn.close()
        return

    df_final = pd.concat(processed_list, ignore_index=True)

    # 日期轉文字（SQLite 寫入穩定）
    df_final["date"] = pd.to_datetime(df_final["date"]).dt.strftime("%Y-%m-%d")
    if "peak_date" in df_final.columns:
        df_final["peak_date"] = pd.to_datetime(df_final["peak_date"], errors="coerce").dt.strftime("%Y-%m-%d")

    # 重建 stock_analysis（✅ 這一步會讓 DB 真的「增加欄位」：因為 schema 會跟著 df_final 變）
    conn.execute("DROP TABLE IF EXISTS stock_analysis")
    df_final.to_sql("stock_analysis", conn, if_exists="replace", index=False)

    # 索引
    try:
        conn.execute("CREATE INDEX IF NOT EXISTS idx_symbol_date ON stock_analysis (symbol, date)")
        conn.execute("CREATE INDEX IF NOT EXISTS idx_strength_rank ON stock_analysis (strength_rank)")
        conn.execute("CREATE INDEX IF NOT EXISTS idx_market ON stock_analysis (market)")
        conn.execute("CREATE INDEX IF NOT EXISTS idx_market_detail ON stock_analysis (market_detail)")
        conn.execute("CREATE INDEX IF NOT EXISTS idx_is_limit_up ON stock_analysis (is_limit_up)")
        conn.execute("CREATE INDEX IF NOT EXISTS idx_lu_type ON stock_analysis (lu_type)")
    except Exception:
        pass

    conn.commit()

    # 統計輸出
    total_symbols = df_final["symbol"].nunique()
    date_range = f"{df_final['date'].min()} ~ {df_final['date'].max()}"

    print("\n✅ Feature Layer 完成（stock_analysis 已重建）")
    print(f"📌 股票數量: {total_symbols}")
    print(f"📌 期間: {date_range}")
    print(f"📌 總行數: {len(df_final):,}")
    print("📌 新增/確認欄位包含：is_limit_up, is_one_tick_lock, lu_type(文章規則), consecutive_limits, strength_rank, volatility_20")

    conn.close()


if __name__ == "__main__":
    # 測試：改成你的 DB 檔名
    process_market_data("tw_stock_warehouse.db")
