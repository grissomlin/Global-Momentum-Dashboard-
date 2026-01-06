# -*- coding: utf-8 -*-
"""
downloader_tw.py
----------------
台股資料下載器（與 Global-Momentum-Dashboard- main.py 相容版）

✅ 介面相容：run_sync(start_date, end_date)
✅ DB schema 相容 processor.py：
   - stock_prices(symbol,date,open,high,low,close,volume) 主鍵(symbol,date)
   - stock_info(symbol,name,sector,market,market_detail,updated_at)
✅ 多執行緒批次下載（可控 max_workers + batch_size）
✅ 增量下載：用一次性 max(date) map，避免每檔開 DB
✅ 寫入用 INSERT OR REPLACE（避免重跑被 IGNORE）
✅ 下載失敗寫入 download_errors（不洗版）
"""

import os
import io
import time
import sqlite3
import random
from datetime import datetime, timedelta
from concurrent.futures import ThreadPoolExecutor, as_completed

import pandas as pd
import yfinance as yf
import requests
from io import StringIO
from tqdm import tqdm

# ========== 1) 環境設定 ==========
MARKET_CODE = "tw-share"
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DB_PATH = os.path.join(BASE_DIR, "tw_stock_warehouse.db")


def log(msg: str):
    print(f"{pd.Timestamp.now():%H:%M:%S}: {msg}", flush=True)


# ========== 2) 資料庫初始化（對齊 processor.py） ==========
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

        # 讓錯誤可追蹤，不要洗 log
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

        conn.execute("CREATE INDEX IF NOT EXISTS idx_prices_date ON stock_prices(date)")
        conn.execute("CREATE INDEX IF NOT EXISTS idx_prices_symbol ON stock_prices(symbol)")
        conn.execute("CREATE INDEX IF NOT EXISTS idx_err_symbol ON download_errors(symbol)")
        conn.commit()
    finally:
        conn.close()


def load_last_dates_map() -> dict:
    """
    一次性載入每個 symbol 的最後日期，避免每檔開 DB（速度/穩定性大幅提升）
    """
    if not os.path.exists(DB_PATH):
        return {}
    conn = sqlite3.connect(DB_PATH)
    try:
        rows = conn.execute("SELECT symbol, MAX(date) AS last_date FROM stock_prices GROUP BY symbol").fetchall()
        return {sym: d for sym, d in rows if sym and d}
    except Exception:
        return {}
    finally:
        conn.close()


# ========== 3) 台股清單 ==========
def get_tw_stock_list():
    url_configs = [
        {
            "name": "上市",
            "market": "上市",
            "market_detail": "listed",
            "url": "https://isin.twse.com.tw/isin/class_main.jsp?market=1&issuetype=1&Page=1&chklike=Y",
            "suffix": ".TW",
        },
        {
            "name": "上櫃",
            "market": "上櫃",
            "market_detail": "otc",
            "url": "https://isin.twse.com.tw/isin/class_main.jsp?market=2&issuetype=4&Page=1&chklike=Y",
            "suffix": ".TWO",
        },
        {
            "name": "興櫃",
            "market": "興櫃",
            "market_detail": "emerging",
            "url": "https://isin.twse.com.tw/isin/class_main.jsp?market=E&issuetype=R&Page=1&chklike=Y",
            "suffix": ".TWO",
        },
        {
            "name": "ETF",
            "market": "上市",
            "market_detail": "etf",
            "url": "https://isin.twse.com.tw/isin/class_main.jsp?market=1&issuetype=I&Page=1&chklike=Y",
            "suffix": ".TW",
        },
        {
            "name": "臺灣創新板",
            "market": "上市",
            "market_detail": "innovation",
            "url": "https://isin.twse.com.tw/isin/class_main.jsp?market=C&issuetype=C&Page=1&chklike=Y",
            "suffix": ".TW",
        },
    ]

    log("📡 獲取台股清單...")
    conn = sqlite3.connect(DB_PATH)
    stock_list = []

    try:
        for cfg in url_configs:
            try:
                resp = requests.get(cfg["url"], timeout=20)
                dfs = pd.read_html(StringIO(resp.text), header=0)
                if not dfs:
                    continue

                df = dfs[0]
                df.columns = [str(col).strip() for col in df.columns]

                code_field = next((f for f in ["有價證券代號", "代號", "證券代號"] if f in df.columns), None)
                name_field = next((f for f in ["有價證券名稱", "名稱", "證券名稱"] if f in df.columns), None)
                if not code_field or not name_field:
                    log(f"⚠️ {cfg['name']} 欄位解析失敗，略過")
                    continue

                for _, row in df.iterrows():
                    code = str(row.get(code_field, "")).strip()
                    name = str(row.get(name_field, "")).strip()

                    if not code or not code.isalnum():
                        continue
                    if not (4 <= len(code) <= 6):
                        continue

                    symbol = f"{code}{cfg['suffix']}"

                    sector = ""
                    for f in ["產業別", "產業分類", "類別"]:
                        if f in df.columns:
                            sector = str(row.get(f, "")).strip()
                            break

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
                            cfg["market"],
                            cfg["market_detail"],
                            datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                        ),
                    )

                    stock_list.append((symbol, name, cfg["market"], cfg["market_detail"]))

                # 顯示該分類數量
                cnt = sum(1 for x in stock_list if x[3] == cfg["market_detail"])
                log(f"✅ {cfg['name']} 載入完成，共 {cnt} 檔")

            except Exception as e:
                log(f"❌ {cfg['name']} 抓取失敗: {e}")
                continue

        conn.commit()

    finally:
        conn.close()

    # 去重：以 (symbol,name) 為 key
    final = list({(s[0], s[1]): s for s in stock_list}.values())
    log(f"📊 全市場掃描完畢，總計 {len(final)} 檔有效標的")
    return final


# ========== 4) 單檔下載（給 ThreadPoolExecutor 用） ==========
def download_one_tw(symbol: str, actual_start: str, end_date: str):
    max_retries = 2
    for attempt in range(max_retries):
        try:
            df = yf.download(
                symbol,
                start=actual_start,
                end=end_date,
                progress=False,
                auto_adjust=True,
                threads=False,
                timeout=25,
            )
            if df is None or df.empty:
                return None, "empty"

            if isinstance(df.columns, pd.MultiIndex):
                df.columns = df.columns.get_level_values(0)

            df = df.reset_index()
            df.columns = [str(c).lower() for c in df.columns]

            if "date" not in df.columns:
                # 有些情況會叫 index
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
            if "possibly delisted" in msg or "no timezone found" in msg:
                return None, "delisted_or_no_timezone"

            if attempt < max_retries - 1:
                time.sleep(1.5 * (attempt + 1))
                continue
            return None, f"exception: {msg}"

    return None, "unknown"


def process_single_stock(item, start_date, end_date, last_date_map: dict):
    """
    回傳: (status, df, err)
    status: success / skipped / no_data / error
    """
    symbol, name, market, market_detail = item

    # 計算實際開始日期（增量）
    actual_start = start_date
    last_date = last_date_map.get(symbol)
    if last_date:
        try:
            last_dt = pd.to_datetime(last_date)
            end_dt = pd.to_datetime(end_date)
            if last_dt >= end_dt:
                return "skipped", None, None
            actual_start = (last_dt + timedelta(days=1)).strftime("%Y-%m-%d")
        except Exception:
            actual_start = start_date

    if pd.to_datetime(actual_start) > pd.to_datetime(end_date):
        return "skipped", None, None

    df, err = download_one_tw(symbol, actual_start, end_date)
    if df is None:
        if err == "empty":
            return "no_data", None, err
        return "error", None, err
    return "success", df, None


# ========== 5) 主流程 ==========
def run_sync(start_date="2024-01-01", end_date="2025-12-31", max_workers=8, batch_size=500):
    start_time = time.time()

    init_db()

    log("📋 開始獲取股票清單...")
    items = get_tw_stock_list()
    if not items:
        log("❌ 未獲取到任何股票清單")
        return {"success": 0, "total": 0, "skipped": 0, "failed": 0, "has_changed": False}

    # 一次性載入每檔最後日期（超重要）
    last_date_map = load_last_dates_map()

    log(f"🚀 多執行緒同步啟動 | workers={max_workers} | batch={batch_size} | 目標={len(items)} 檔")

    success_count, skip_count, fail_count = 0, 0, 0

    total_batches = (len(items) + batch_size - 1) // batch_size

    for batch_num in range(total_batches):
        batch_items = items[batch_num * batch_size : min((batch_num + 1) * batch_size, len(items))]
        log(f"📦 批次 {batch_num+1}/{total_batches} | {len(batch_items)} 檔")

        conn = sqlite3.connect(DB_PATH, timeout=120)
        try:
            with ThreadPoolExecutor(max_workers=max_workers) as ex:
                futures = {
                    ex.submit(process_single_stock, item, start_date, end_date, last_date_map): item
                    for item in batch_items
                }

                for future in tqdm(as_completed(futures), total=len(batch_items), desc=f"TW批次{batch_num+1}", leave=False):
                    item = futures[future]
                    symbol, name, *_ = item

                    try:
                        status, df_res, err = future.result()
                    except Exception as e:
                        status, df_res, err = "error", None, f"future_exception: {e}"

                    if status == "skipped":
                        skip_count += 1
                        continue

                    if status == "success" and df_res is not None and not df_res.empty:
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
                            fail_count += 1
                            err = f"db_insert_failed: {e}"

                    else:
                        fail_count += 1

                    if err:
                        # 記錄錯誤但不洗 log
                        try:
                            conn.execute(
                                "INSERT INTO download_errors (symbol, name, start_date, end_date, error, created_at) VALUES (?, ?, ?, ?, ?, ?)",
                                (symbol, name, start_date, end_date, err, datetime.now().strftime("%Y-%m-%d %H:%M:%S")),
                            )
                        except Exception:
                            pass

            conn.commit()

        finally:
            conn.close()

        # 批次間隔，避免過度請求（比你原本 2 秒更溫和）
        if batch_num < total_batches - 1:
            time.sleep(1.0)

    # VACUUM
    log("🧹 優化資料庫 VACUUM...")
    conn = sqlite3.connect(DB_PATH, timeout=120)
    try:
        conn.execute("VACUUM")
        conn.commit()
    finally:
        conn.close()

    duration = (time.time() - start_time) / 60
    log(f"📊 TW 同步完成 | 成功:{success_count} 跳過:{skip_count} 失敗:{fail_count} | {duration:.1f} 分鐘")

    return {
        "success": success_count,
        "total": len(items),
        "skipped": skip_count,
        "failed": fail_count,
        "has_changed": success_count > 0,
    }


if __name__ == "__main__":
    r = run_sync(start_date="2024-01-01", end_date="2025-12-31", max_workers=6, batch_size=500)
    print(r)
