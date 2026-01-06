# -*- coding: utf-8 -*-
"""
downloader_hk.py
----------------
港股資料下載器（與 Global-Momentum-Dashboard- main.py / processor.py 相容版）

✅ 介面相容：run_sync(start_date, end_date)
✅ DB schema 相容 processor.py：
   - stock_prices(symbol,date,open,high,low,close,volume) 主鍵(symbol,date)
   - stock_info(symbol,name,sector,market,market_detail,updated_at)
✅ 增量下載：依 stock_prices 的 MAX(date) 決定每檔實際開始日
✅ HKEX 名單：解析 xls，自動處理表頭位置
✅ yfinance ticker 嘗試：00001.HK / 1.HK
✅ 下載失敗寫入 download_errors（不洗版）
"""

import os
import io
import re
import time
import sqlite3
from datetime import datetime, timedelta

import pandas as pd
import yfinance as yf
import requests
from tqdm import tqdm

# ========== 1) 環境設定 ==========
MARKET_CODE = "hk-share"
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DB_PATH = os.path.join(BASE_DIR, "hk_stock_warehouse.db")

HKEX_LIST_URL = (
    "https://www.hkex.com.hk/-/media/HKEX-Market/Services/Trading/"
    "Securities/Securities-Lists/"
    "Securities-Using-Standard-Transfer-Form-(including-GEM)-"
    "By-Stock-Code-Order/secstkorder.xls"
)

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

        # processor.py 會 LEFT JOIN stock_info 取 market/sector/market_detail
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


# ========== 3) HKEX 清單解析 ==========
def normalize_code_5d(val) -> str:
    digits = re.sub(r"\D", "", str(val))
    if digits.isdigit() and 1 <= int(digits) <= 99999:
        return digits.zfill(5)
    return ""


def get_hk_stock_list():
    """
    回傳 [(code_5d, name), ...]
    並寫入 stock_info（symbol=5位數字，不含.HK）
    """
    log("📡 正在從港交所下載最新股票清單...")

    headers = {
        "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36",
        "Referer": "https://www.hkex.com.hk/Market-Data/Securities-Prices/Equities",
        "Accept": "*/*",
    }

    try:
        r = requests.get(HKEX_LIST_URL, headers=headers, timeout=40)
        r.raise_for_status()
        # header=None：因為 HKEX xls 的表頭常不是第一列
        df_raw = pd.read_excel(io.BytesIO(r.content), header=None)
    except Exception as e:
        log(f"❌ 無法獲取 HKEX 清單: {e}")
        return []

    # 找表頭所在列（包含 Stock Code / Short Name）
    header_row = None
    for i in range(min(30, len(df_raw))):
        row_vals = [str(x).replace("\xa0", " ").strip() for x in df_raw.iloc[i].values]
        if any("Stock Code" in v for v in row_vals) and any("Short Name" in v for v in row_vals):
            header_row = i
            break

    if header_row is None:
        log("❌ 無法辨識 HKEX Excel 結構（找不到表頭）")
        return []

    df = df_raw.iloc[header_row + 1:].copy()
    df.columns = [str(x).replace("\xa0", " ").strip() for x in df_raw.iloc[header_row].values]

    # 欄位定位
    code_col = next((c for c in df.columns if "Stock Code" in c), None)
    name_col = next((c for c in df.columns if "Short Name" in c), None)
    if not code_col or not name_col:
        log("❌ 無法定位 Stock Code / Short Name 欄位")
        return []

    conn = sqlite3.connect(DB_PATH)
    stock_list = []

    try:
        for _, row in df.iterrows():
            code_5d = normalize_code_5d(row.get(code_col))
            if not code_5d:
                continue

            name = str(row.get(name_col, "")).strip()
            if not name:
                name = "Unknown"

            # HK market_detail：這裡先給個穩定值，之後你想細分 GEM/MAIN 再加欄位解析
            conn.execute(
                """
                INSERT OR REPLACE INTO stock_info
                (symbol, name, sector, market, market_detail, updated_at)
                VALUES (?, ?, ?, ?, ?, ?)
                """,
                (
                    code_5d,
                    name,
                    "HK-Share",
                    "HKEX",
                    "hk",
                    datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                ),
            )
            stock_list.append((code_5d, name))

        conn.commit()
    finally:
        conn.close()

    log(f"✅ 港股名單同步完成：共 {len(stock_list)} 檔")
    return stock_list


# ========== 4) 單檔下載（增量、ticker fallback） ==========
def build_possible_tickers(code_5d: str):
    # yfinance 常見：00001.HK 或 1.HK
    tickers = [f"{code_5d}.HK"]
    if code_5d.startswith("0"):
        tickers.append(f"{code_5d.lstrip('0')}.HK")
    return tickers


def download_one_hk(code_5d: str, actual_start: str, end_date: str):
    """
    回傳 (df, err)
    df 欄位：symbol,date,open,high,low,close,volume
    symbol: 存 5 位 code（不含.HK）
    """
    tickers = build_possible_tickers(code_5d)
    last_err = None

    for sym in tickers:
        try:
            df = yf.download(
                sym,
                start=actual_start,
                end=end_date,
                progress=False,
                auto_adjust=True,
                threads=False,
                timeout=25,
            )

            if df is None or df.empty:
                last_err = "empty"
                continue

            if isinstance(df.columns, pd.MultiIndex):
                df.columns = df.columns.get_level_values(0)

            df = df.reset_index()
            df.columns = [c.lower() for c in df.columns]

            if "date" not in df.columns:
                if "index" in df.columns:
                    df["date"] = df["index"]
                else:
                    last_err = "no_date_col"
                    continue

            df["date"] = pd.to_datetime(df["date"]).dt.tz_localize(None).dt.strftime("%Y-%m-%d")

            for col in ["open", "high", "low", "close", "volume"]:
                if col not in df.columns:
                    df[col] = None

            out = df[["date", "open", "high", "low", "close", "volume"]].copy()
            out["symbol"] = code_5d
            out = out[["symbol", "date", "open", "high", "low", "close", "volume"]]
            return out, None

        except Exception as e:
            msg = str(e)
            if "possibly delisted" in msg or "no timezone found" in msg:
                return None, "delisted_or_no_timezone"
            last_err = f"exception: {msg}"
            continue

    return None, last_err or "unknown"


# ========== 5) 主流程（main.py 相容） ==========
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

    log(f"🚀 啟動港股同步 | 期間: {start_date} ~ {end_date}")

    stocks = get_hk_stock_list()
    if not stocks:
        log("❌ 沒有可下載的港股清單")
        return {"success": 0, "total": 0, "has_changed": False}

    last_date_map = load_last_dates_map()

    success_count = 0
    fail_count = 0
    skip_count = 0

    conn = sqlite3.connect(DB_PATH, timeout=120)
    try:
        pbar = tqdm(stocks, desc="HK同步", unit="檔")
        for code_5d, name in pbar:
            # 增量計算
            actual_start = start_date
            last_date = last_date_map.get(code_5d)
            if last_date:
                try:
                    next_day = (pd.to_datetime(last_date) + timedelta(days=1)).strftime("%Y-%m-%d")
                    actual_start = next_day
                    if pd.to_datetime(actual_start) > pd.to_datetime(end_date):
                        skip_count += 1
                        continue
                except Exception:
                    actual_start = start_date

            df_res, err = download_one_hk(code_5d, actual_start, end_date)

            if df_res is not None and not df_res.empty:
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
            else:
                fail_count += 1
                if err:
                    try:
                        conn.execute(
                            "INSERT INTO download_errors (symbol, name, start_date, end_date, error, created_at) VALUES (?, ?, ?, ?, ?, ?)",
                            (code_5d, name, start_date, end_date, err, datetime.now().strftime("%Y-%m-%d %H:%M:%S")),
                        )
                    except Exception:
                        pass

            # 控速（港股比美股更容易被判定過快）
            time.sleep(0.05)

        conn.commit()

        log("🧹 執行資料庫 VACUUM...")
        conn.execute("VACUUM")
        conn.commit()

    finally:
        conn.close()

    mins = (time.time() - t0) / 60
    log(f"📊 港股同步完成 | 成功:{success_count} 跳過:{skip_count} 失敗:{fail_count} | {mins:.1f} 分鐘")

    return {
        "success": success_count,
        "total": len(stocks),
        "skipped": skip_count,
        "failed": fail_count,
        "has_changed": success_count > 0,
    }


if __name__ == "__main__":
    run_sync(start_date="2024-01-01", end_date="2025-12-31")
