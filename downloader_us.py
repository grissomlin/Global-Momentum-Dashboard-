# -*- coding: utf-8 -*-
"""
downloader_us.py
----------------
美股資料下載器（與 Global-Momentum-Dashboard- main.py 相容版）

✅ 介面相容：run_sync(start_date, end_date)
✅ DB schema 相容 processor.py：
   - stock_prices(symbol,date,open,high,low,close,volume) 主鍵(symbol,date)
   - stock_info(symbol,name,sector,market,market_detail,updated_at)
✅ 名單來源：Nasdaq 官方 API（被擋則 fallback：NASDAQ/NYSE/AMEX CSV）
✅ 單執行緒 threads=False，降低 Yahoo 風控/錯亂
✅ 下載失敗：寫入 download_errors 表，不洗版
"""

import os
import io
import re
import time
import sqlite3
from datetime import datetime

import pandas as pd
import yfinance as yf
import requests
from tqdm import tqdm

# =====================================================
# 1) 基本設定
# =====================================================
MARKET_CODE = "us-share"
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DB_PATH = os.path.join(BASE_DIR, "us_stock_warehouse.db")

NASDAQ_API = "https://api.nasdaq.com/api/screener/stocks?tableonly=true&limit=20000&download=true"
NASDAQ_REFERER = "https://www.nasdaq.com/market-activity/stocks/screener"


def log(msg: str):
    print(f"{pd.Timestamp.now():%H:%M:%S}: {msg}", flush=True)


# =====================================================
# 2) DB 初始化（對齊 processor.py）
# =====================================================
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
        conn.execute("CREATE INDEX IF NOT EXISTS idx_prices_symbol_date ON stock_prices(symbol, date)")
        conn.execute("CREATE INDEX IF NOT EXISTS idx_info_market ON stock_info(market)")
        conn.execute("CREATE INDEX IF NOT EXISTS idx_err_symbol ON download_errors(symbol)")
        conn.commit()
    finally:
        conn.close()


# =====================================================
# 3) 名單抓取（優先 Nasdaq API，失敗則 fallback）
# =====================================================
def _fetch_us_list_from_nasdaq_api():
    log("📡 正在從 Nasdaq 官方 API 同步美股名單...")
    headers = {
        "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36",
        "Accept": "application/json, text/plain, */*",
        "Referer": NASDAQ_REFERER,
    }
    r = requests.get(NASDAQ_API, headers=headers, timeout=30)
    r.raise_for_status()
    j = r.json()
    rows = (j.get("data") or {}).get("rows") or []
    return rows


def _fetch_us_list_fallback_csv():
    """
    fallback：Stooq 的 symbols（通常穩，但可能含 ETF/基金/權證，需要過濾）
    你不想用 stooq 也行，這只是備援。
    """
    log("📡 Nasdaq API 失敗，改用 fallback CSV 名單...")
    url = "https://stooq.com/q/l/?s=us&i=1"  # CSV list
    r = requests.get(url, timeout=30)
    r.raise_for_status()
    # stooq 這份通常是簡單 list；我們只抽 symbols
    df = pd.read_csv(io.StringIO(r.text))
    # 常見欄位是 Symbol 或 s
    sym_col = None
    for c in df.columns:
        if str(c).lower() in ["symbol", "s"]:
            sym_col = c
            break
    if not sym_col:
        return []
    return [{"symbol": str(x).strip().upper(), "name": "Unknown", "sector": "Unknown", "exchange": "Unknown"} for x in df[sym_col].dropna().tolist()]


def get_us_stock_list():
    """
    回傳 [(symbol, name), ...]，並寫入 stock_info
    """
    rows = []
    try:
        rows = _fetch_us_list_from_nasdaq_api()
        source = "NASDAQ_API"
    except Exception as e:
        log(f"⚠️ Nasdaq API 名單取得失敗: {e}")
        try:
            rows = _fetch_us_list_fallback_csv()
            source = "FALLBACK_CSV"
        except Exception as e2:
            log(f"❌ fallback 名單也失敗: {e2}")
            return []

    conn = sqlite3.connect(DB_PATH)
    stock_list = []

    # 排除：Warrant / Right / Preferred / Unit / ETF / Index...
    exclude_kw = re.compile(r"Warrant|Right|Preferred|Unit|ETF|Index|Index-linked|Trust|Fund|Notes", re.I)

    try:
        for row in rows:
            symbol = str(row.get("symbol", "")).strip().upper()

            # 基本格式過濾：只保留常見股票代碼（允許 . 例如 BRK.B、BF.B）
            if not symbol:
                continue
            if len(symbol) > 8:
                continue
            if not re.match(r"^[A-Z0-9.\-]+$", symbol):
                continue

            name = str(row.get("name", "Unknown")).strip()
            if exclude_kw.search(name or ""):
                continue

            # Nasdaq API 有 exchange/sector；fallback 可能沒有
            sector = str(row.get("sector", "Unknown")).strip() or "Unknown"
            exchange = str(row.get("exchange", "Unknown")).strip() or "Unknown"

            # 你 processor 會用 market + market_detail
            # market: US, market_detail: exchange（NASDAQ/NYSE/AMEX...）
            conn.execute(
                """
                INSERT OR REPLACE INTO stock_info
                (symbol, name, sector, market, market_detail, updated_at)
                VALUES (?, ?, ?, ?, ?, ?)
                """,
                (
                    symbol,
                    name if name else symbol,
                    sector,
                    "US",
                    exchange,
                    datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                ),
            )
            stock_list.append((symbol, name if name else symbol))

        conn.commit()
    finally:
        conn.close()

    log(f"✅ 美股名單導入成功: {len(stock_list)} 檔（來源={source}）")
    return stock_list


# =====================================================
# 4) 下載單檔（穩定單執行緒）
# =====================================================
def download_one_us(symbol: str, start_date: str, end_date: str):
    """
    回傳 (df, err)
    df 欄位：symbol,date,open,high,low,close,volume
    """
    max_retries = 2

    for attempt in range(max_retries):
        try:
            df = yf.download(
                symbol,
                start=start_date,
                end=end_date,
                progress=False,
                auto_adjust=True,
                threads=False,
                timeout=20,
            )

            if df is None or df.empty:
                return None, "empty"

            if isinstance(df.columns, pd.MultiIndex):
                df.columns = df.columns.get_level_values(0)

            df = df.reset_index()
            df.columns = [str(c).lower() for c in df.columns]

            if "date" not in df.columns:
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
            if "possibly delisted" in msg or "no timezone found" in msg:
                return None, "delisted_or_no_timezone"

            if attempt < max_retries - 1:
                time.sleep(1.5 * (attempt + 1))
                continue
            return None, f"exception: {msg}"

    return None, "unknown"


# =====================================================
# 5) 主流程（相容 main.py：run_sync(start_date, end_date)）
# =====================================================
def run_sync(start_date=None, end_date=None):
    start_time = time.time()
    init_db()

    if not start_date:
        start_date = "2023-01-01"
    if not end_date:
        end_date = datetime.now().strftime("%Y-%m-%d")

    items = get_us_stock_list()
    if not items:
        return {"success": 0, "total": 0, "has_changed": False}

    log(f"🚀 啟動美股同步 | 期間: {start_date} ~ {end_date} | 目標: {len(items)} 檔")

    conn = sqlite3.connect(DB_PATH, timeout=60)
    success_count = 0
    fail_count = 0

    pbar = tqdm(items, desc="美國下載", unit="檔")
    try:
        for symbol, name in pbar:
            pbar.set_postfix({"股票": symbol})

            df_res, err = download_one_us(symbol, start_date, end_date)

            if df_res is not None and not df_res.empty:
                try:
                    df_res.to_sql(
                        "stock_prices",
                        conn,
                        if_exists="append",
                        index=False,
                        method=lambda table, conn2, keys, data_iter: conn2.executemany(
                            f"INSERT OR REPLACE INTO {table.name} ({', '.join(keys)}) VALUES ({', '.join(['?']*len(keys))})",
                            data_iter,
                        ),
                    )
                    success_count += 1
                except Exception as e:
                    err = f"db_insert_failed: {e}"

            if err:
                fail_count += 1
                try:
                    conn.execute(
                        "INSERT INTO download_errors (symbol, name, start_date, end_date, error, created_at) VALUES (?, ?, ?, ?, ?, ?)",
                        (symbol, name, start_date, end_date, err, datetime.now().strftime("%Y-%m-%d %H:%M:%S")),
                    )
                except Exception:
                    pass

            time.sleep(0.02)  # 避免 Yahoo 限流

        conn.commit()
        log("🧹 VACUUM...")
        conn.execute("VACUUM")
        conn.commit()

        duration = (time.time() - start_time) / 60
        log(f"📊 US 同步完成 | 成功: {success_count}/{len(items)} | 失敗記錄: {fail_count} | {duration:.1f} 分")

        return {"success": success_count, "total": len(items), "has_changed": success_count > 0}

    finally:
        conn.close()


if __name__ == "__main__":
    # python downloader_us.py --start=2024-01-01 --end=2025-12-31
    s, e = None, None
    for arg in sys.argv[1:]:
        if arg.startswith("--start="):
            s = arg.split("=", 1)[1]
        elif arg.startswith("--end="):
            e = arg.split("=", 1)[1]
    run_sync(s, e)
