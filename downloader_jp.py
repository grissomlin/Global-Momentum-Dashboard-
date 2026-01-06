# -*- coding: utf-8 -*-
"""
downloader_jp.py
----------------
日股資料下載器（與 Global-Momentum-Dashboard- main.py 相容版）

✅ 介面相容：run_sync(start_date, end_date)
✅ DB schema 相容 processor.py：
   - stock_prices(symbol,date,open,high,low,close,volume) 主鍵(symbol,date)
   - stock_info(symbol,name,sector,market,market_detail,updated_at)
✅ JPX 清單：優先從 JPX 下載 data_e.xls；失敗可 fallback 本地檔（可選）
✅ 單執行緒 threads=False，避免 Yahoo 資料錯亂
✅ 下載失敗/下市/無timezone：不洗版、記到 download_errors 表
"""

import os
import sys
import io
import time
import sqlite3
import json
from datetime import datetime

import pandas as pd
import yfinance as yf
import requests
from tqdm import tqdm


# =====================================================
# 1) 基本設定
# =====================================================
MARKET_CODE = "jp-share"
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DB_PATH = os.path.join(BASE_DIR, "jp_stock_warehouse.db")

# JPX 官方清單（你 repo 原本用的）
JPX_XLS_URL = "https://www.jpx.co.jp/english/markets/statistics-equities/misc/tvdivq0000001vg2-att/data_e.xls"
JPX_REFERER = "https://www.jpx.co.jp/english/markets/statistics-equities/misc/01.html"

# 可選：如果 JPX 下載失敗，你可以放一份本地 fallback
# （自己手動放在 repo 或 Actions 產物裡）
FALLBACK_CSV = os.path.join(BASE_DIR, "jpx_list_fallback.csv")


def log(msg: str):
    print(f"{pd.Timestamp.now():%H:%M:%S}: {msg}", flush=True)


# =====================================================
# 2) DB 初始化（相容 processor.py）
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
# 3) 取得 JPX 股票清單
# =====================================================
def _read_jpx_excel_from_web() -> pd.DataFrame | None:
    headers = {
        "User-Agent": "Mozilla/5.0",
        "Referer": JPX_REFERER,
    }
    try:
        r = requests.get(JPX_XLS_URL, headers=headers, timeout=30)
        r.raise_for_status()

        # 這份是 .xls（舊格式），pandas 需要 xlrd
        # 請在 requirements.txt 加：xlrd==2.0.1（或 >=2.0.1）
        df = pd.read_excel(io.BytesIO(r.content))
        return df
    except Exception as e:
        log(f"⚠️ JPX 名單下載/解析失敗: {e}")
        return None


def _read_jpx_fallback_csv() -> pd.DataFrame | None:
    if not os.path.exists(FALLBACK_CSV):
        return None
    try:
        df = pd.read_csv(FALLBACK_CSV, encoding="utf-8-sig")
        return df
    except Exception as e:
        log(f"⚠️ fallback CSV 讀取失敗: {e}")
        return None


def get_jp_stock_list():
    """
    回傳 [(symbol, name), ...]
    並寫入 stock_info（market='JP'，market_detail=Section/Products）
    """
    log("📡 正在同步日股名單（JPX）...")

    df = _read_jpx_excel_from_web()
    source = "JPX_WEB"

    if df is None:
        df = _read_jpx_fallback_csv()
        source = "FALLBACK_CSV"

    if df is None or df.empty:
        log("❌ 無法取得 JPX 股票名單（web 失敗且無 fallback）")
        return []

    # JPX Excel 標準欄位（你原本 repo 用的）
    C_CODE = "Local Code"
    C_NAME = "Name (English)"
    C_PROD = "Section/Products"
    C_SECTOR = "33 Sector(name)"

    # fallback CSV 若欄位不同，你可以自行在這裡做 mapping
    # 這裡先做「能跑就跑」的兼容處理
    cols = set(df.columns.astype(str))
    if C_CODE not in cols:
        # 嘗試常見替代欄位
        for alt in ["Code", "code", "LocalCode", "local_code"]:
            if alt in cols:
                df = df.rename(columns={alt: C_CODE})
                break
    if C_NAME not in cols:
        for alt in ["Name", "name", "Company", "company_name"]:
            if alt in cols:
                df = df.rename(columns={alt: C_NAME})
                break
    if C_PROD not in cols:
        for alt in ["Section", "section", "Products", "products"]:
            if alt in cols:
                df = df.rename(columns={alt: C_PROD})
                break
    if C_SECTOR not in cols:
        for alt in ["Sector", "sector", "33 Sector", "sector_name"]:
            if alt in cols:
                df = df.rename(columns={alt: C_SECTOR})
                break

    conn = sqlite3.connect(DB_PATH)
    stock_list = []
    try:
        for _, row in df.iterrows():
            raw_code = row.get(C_CODE)
            if pd.isna(raw_code):
                continue

            code = str(raw_code).split(".")[0].strip()

            # 只保留 4 位數普通股
            if not (len(code) == 4 and code.isdigit()):
                continue

            product = str(row.get(C_PROD, "")).strip()
            # 排除 ETF
            if product.lower().startswith("etfs") or "ETF" in product:
                continue

            symbol = f"{code}.T"
            name = str(row.get(C_NAME, "")).strip() or symbol
            sector = str(row.get(C_SECTOR, "Unknown")).strip() or "Unknown"

            # processor 的 MarketConfig 目前沒 JP 規則也沒關係（會走預設）
            # 但為了未來擴展，先把 market='JP'，market_detail=product
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
                    "JP",
                    product if product else "unknown",
                    datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                ),
            )
            stock_list.append((symbol, name))

        conn.commit()
    finally:
        conn.close()

    log(f"✅ 日股名單同步完成：{len(stock_list)} 檔（來源={source}）")
    return stock_list


# =====================================================
# 4) 下載單一股票（避免洗版 + 記錄錯誤）
# =====================================================
def download_one_jp(symbol: str, start_date: str, end_date: str):
    """
    回傳 (df, err)
    df 欄位：symbol,date,open,high,low,close,volume（與 processor 相容）
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

            if "date" in df.columns:
                df["date"] = pd.to_datetime(df["date"]).dt.tz_localize(None).dt.strftime("%Y-%m-%d")
            else:
                return None, "no_date_col"

            for col in ["open", "high", "low", "close", "volume"]:
                if col not in df.columns:
                    df[col] = None

            out = df[["date", "open", "high", "low", "close", "volume"]].copy()
            out["symbol"] = symbol
            out = out[["symbol", "date", "open", "high", "low", "close", "volume"]]
            return out, None

        except Exception as e:
            msg = str(e)

            # 常見錯誤：下市 / 無timezone → 直接停止重試
            if "possibly delisted" in msg or "no timezone found" in msg:
                return None, "delisted_or_no_timezone"

            if attempt < max_retries - 1:
                time.sleep(1.5 * (attempt + 1))
                continue

            return None, f"exception: {msg}"

    return None, "unknown"


# =====================================================
# 5) 主流程（介面相容 main.py）
# =====================================================
def run_sync(start_date=None, end_date=None):
    """
    main.py 會呼叫：
      downloader_jp.run_sync(start_date=..., end_date=...)
    """
    start_time = time.time()
    init_db()

    if not start_date:
        start_date = "2023-01-01"
    if not end_date:
        end_date = datetime.now().strftime("%Y-%m-%d")

    items = get_jp_stock_list()
    if not items:
        return {"success": 0, "total": 0, "has_changed": False}

    log(f"🚀 開始日股同步 | 期間: {start_date} ~ {end_date} | 目標: {len(items)} 檔")

    conn = sqlite3.connect(DB_PATH, timeout=60)
    success_count = 0
    fail_count = 0

    pbar = tqdm(items, desc="日本下載", unit="檔")
    try:
        for symbol, name in pbar:
            pbar.set_postfix({"股票": name[:12]})
            df_res, err = download_one_jp(symbol, start_date, end_date)

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
                        (
                            symbol,
                            name,
                            start_date,
                            end_date,
                            err,
                            datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                        ),
                    )
                except Exception:
                    pass

            # 避免被 Yahoo 限流
            time.sleep(0.05)

        conn.commit()
        log("🧹 VACUUM...")
        conn.execute("VACUUM")
        conn.commit()

        duration = (time.time() - start_time) / 60
        log(f"📊 JP 同步完成 | 成功: {success_count}/{len(items)} | 失敗記錄: {fail_count} | {duration:.1f} 分")

        return {
            "success": success_count,
            "total": len(items),
            "has_changed": success_count > 0,
        }
    finally:
        conn.close()


if __name__ == "__main__":
    # CLI 測試
    # python downloader_jp.py --start=2024-01-01 --end=2025-12-31
    s = None
    e = None
    for arg in sys.argv[1:]:
        if arg.startswith("--start="):
            s = arg.split("=", 1)[1]
        elif arg.startswith("--end="):
            e = arg.split("=", 1)[1]
    run_sync(s, e)
