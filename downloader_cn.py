# -*- coding: utf-8 -*-
"""
downloader_cn.py
----------------
A 股資料下載器（與 Global-Momentum-Dashboard- main.py / processor.py 相容版）

✅ 介面相容：run_sync(start_date, end_date)
✅ DB schema 相容 processor.py：
   - stock_prices(symbol,date,open,high,low,close,volume) 主鍵(symbol,date)
   - stock_info(symbol,name,sector,market,market_detail,updated_at)
✅ 增量下載：依 stock_prices 的 MAX(date) 決定每檔實際開始日
✅ akshare 取 A 股名單（若環境無 akshare，會 fallback 不讓流程直接炸）
✅ 下載失敗寫入 download_errors（不洗版）
✅ ✅ 新增：market_detail 精準分類（main / chinext / star）
   - main: 主板 ±10%
   - chinext: 創業板(300/301) ±20%
   - star: 科創板(688) ±20%
"""

import os
import time
import sqlite3
from datetime import datetime, timedelta

import pandas as pd
import yfinance as yf
from tqdm import tqdm


# ========== 1) 環境設定 ==========
MARKET_CODE = "cn-share"
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DB_PATH = os.path.join(BASE_DIR, "cn_stock_warehouse.db")


def log(msg: str):
    print(f"{pd.Timestamp.now():%H:%M:%S}: {msg}", flush=True)


# ========== 2) DB 初始化（對齊 processor） ==========
def init_db():
    conn = sqlite3.connect(DB_PATH)
    try:
        conn.execute(
            """
            CREATE TABLE IF NOT EXISTS stock_prices (
                symbol TEXT,
                date TEXT,
                open REAL,
                high REAL,
                low REAL,
                close REAL,
                volume INTEGER,
                PRIMARY KEY (symbol, date)
            )
            """
        )

        conn.execute(
            """
            CREATE TABLE IF NOT EXISTS stock_info (
                symbol TEXT PRIMARY KEY,
                name TEXT,
                sector TEXT,
                market TEXT,
                market_detail TEXT,
                updated_at TEXT
            )
            """
        )

        conn.execute(
            """
            CREATE TABLE IF NOT EXISTS download_errors (
                symbol TEXT,
                name TEXT,
                start_date TEXT,
                end_date TEXT,
                error TEXT,
                created_at TEXT
            )
            """
        )

        conn.execute("CREATE INDEX IF NOT EXISTS idx_prices_symbol ON stock_prices(symbol)")
        conn.execute("CREATE INDEX IF NOT EXISTS idx_prices_date ON stock_prices(date)")
        conn.execute("CREATE INDEX IF NOT EXISTS idx_info_market ON stock_info(market)")
        conn.execute("CREATE INDEX IF NOT EXISTS idx_info_market_detail ON stock_info(market_detail)")
        conn.execute("CREATE INDEX IF NOT EXISTS idx_err_symbol ON download_errors(symbol)")
        conn.commit()
    finally:
        conn.close()


def load_last_dates_map() -> dict:
    """一次性載入每個 symbol 的最後日期，避免每檔查 DB"""
    if not os.path.exists(DB_PATH):
        return {}
    conn = sqlite3.connect(DB_PATH)
    try:
        rows = conn.execute(
            "SELECT symbol, MAX(date) AS last_date FROM stock_prices GROUP BY symbol"
        ).fetchall()
        return {sym: d for sym, d in rows if sym and d}
    except Exception:
        return {}
    finally:
        conn.close()


# ========== 3) A 股板塊分類（關鍵：給 market_detail） ==========
def classify_cn_by_code(code6: str):
    """
    回傳 (market, market_detail)
    market: SSE / SZSE
    market_detail:
      - 'main'    主板/中小板/其他先歸主板制度（±10%）
      - 'chinext' 創業板（300/301）±20%
      - 'star'    科創板（688）±20%
    """
    code6 = str(code6).zfill(6)

    # 科創板
    if code6.startswith("688"):
        return "SSE", "star"

    # 創業板
    if code6.startswith(("300", "301")):
        return "SZSE", "chinext"

    # 其他：主板/中小板
    if code6.startswith("6"):
        return "SSE", "main"
    return "SZSE", "main"


# ========== 4) 取得 A 股清單 ==========
def get_cn_stock_list():
    """
    回傳 [(symbol, name), ...]
    symbol 使用 Yahoo Finance 格式：.SS / .SZ
    """
    log("📡 正在獲取 A 股清單...")

    # 1) 優先使用 akshare（名單最完整）
    try:
        import akshare as ak

        df_spot = ak.stock_zh_a_spot_em()

        valid_prefixes = (
            "000", "001", "002", "003",  # 深市主板/中小板
            "300", "301",                # 創業板
            "600", "601", "603", "605",  # 滬市主板
            "688",                       # 科創板
        )

        conn = sqlite3.connect(DB_PATH)
        stock_list = []
        try:
            for _, row in df_spot.iterrows():
                code = str(row.get("代码", "")).zfill(6)
                if not code.startswith(valid_prefixes):
                    continue

                market, market_detail = classify_cn_by_code(code)
                symbol = f"{code}.SS" if market == "SSE" else f"{code}.SZ"

                name = str(row.get("名称", "Unknown")).strip() or "Unknown"

                # sector：你原本寫 A-Share，維持一致
                sector = "A-Share"

                conn.execute(
                    """
                    INSERT OR REPLACE INTO stock_info
                    (symbol, name, sector, market, market_detail, updated_at)
                    VALUES (?, ?, ?, ?, ?, ?)
                    """,
                    (
                        symbol,
                        name,
                        sector,
                        market,
                        market_detail,
                        datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                    ),
                )
                stock_list.append((symbol, name))

            conn.commit()
        finally:
            conn.close()

        log(f"✅ A 股清單導入成功: {len(stock_list)} 檔")
        return stock_list

    except Exception as e:
        log(f"⚠️ akshare 名單取得失敗（將 fallback）: {e}")

    # 2) fallback：從 DB stock_info 既有清單跑（若你之前曾成功導入過）
    conn = sqlite3.connect(DB_PATH)
    try:
        rows = conn.execute("SELECT symbol, name FROM stock_info").fetchall()
        items = [(s, n or "Unknown") for s, n in rows if s]
        if items:
            log(f"✅ 使用 stock_info 既有清單: {len(items)} 檔")
            return items
    finally:
        conn.close()

    log("❌ 無可用 A 股清單（akshare 失敗且 DB 無既有名單）")
    return []


# ========== 5) 單檔下載 ==========
def download_one_cn(symbol: str, actual_start: str, end_date: str):
    """
    回傳 (df, err)
    df 欄位：symbol,date,open,high,low,close,volume
    """
    max_retries = 2
    last_err = None

    for attempt in range(max_retries + 1):
        try:
            df = yf.download(
                symbol,
                start=actual_start,
                end=end_date,
                progress=False,
                timeout=25,
                auto_adjust=True,
                threads=False,
            )

            if df is None or df.empty:
                last_err = "empty"
                if attempt < max_retries:
                    time.sleep(1.5)
                    continue
                return None, last_err

            if isinstance(df.columns, pd.MultiIndex):
                df.columns = df.columns.get_level_values(0)

            df = df.reset_index()
            df.columns = [c.lower() for c in df.columns]

            # 日期欄位
            if "date" not in df.columns:
                if "index" in df.columns:
                    df["date"] = df["index"]
                else:
                    return None, "no_date_col"

            df["date"] = pd.to_datetime(df["date"]).dt.tz_localize(None).dt.strftime("%Y-%m-%d")

            for col in ["open", "high", "low", "close", "volume"]:
                if col not in df.columns:
                    df[col] = None

            out = df[["date", "open", "high", "low", "close", "volume"]].copy()
            out["symbol"] = symbol
            out = out[["symbol", "date", "open", "high", "low", "close", "volume"]]
            return out, None

        except Exception as e:
            msg = str(e)
            last_err = f"exception: {msg}"
            if attempt < max_retries:
                time.sleep(2.0)
                continue
            return None, last_err

    return None, last_err or "unknown"


# ========== 6) 主流程（main.py 相容） ==========
def run_sync(start_date=None, end_date=None):
    """
    main.py 會呼叫：run_sync(start_date=..., end_date=...)
    """
    t0 = time.time()
    init_db()

    if not start_date:
        start_date = "2024-01-01"
    if not end_date:
        end_date = datetime.now().strftime("%Y-%m-%d")

    log(f"🚀 啟動 A 股同步 | 期間: {start_date} ~ {end_date}")

    items = get_cn_stock_list()
    if not items:
        return {"success": 0, "total": 0, "has_changed": False}

    last_date_map = load_last_dates_map()

    success_count = 0
    fail_count = 0
    skip_count = 0

    conn = sqlite3.connect(DB_PATH, timeout=120)
    try:
        pbar = tqdm(items, desc="CN同步", unit="檔")
        for symbol, name in pbar:
            # 增量計算
            actual_start = start_date
            last_date = last_date_map.get(symbol)
            if last_date:
                try:
                    next_day = (pd.to_datetime(last_date) + timedelta(days=1)).strftime("%Y-%m-%d")
                    actual_start = next_day
                    if pd.to_datetime(actual_start) > pd.to_datetime(end_date):
