# -*- coding: utf-8 -*-
"""
downloader_kr.py
----------------
韓國股市資料下載器 (與主系統兼容版)

✔ 優先嘗試從 KRX KIND 下載公司清單（失敗才 fallback 本地 CSV）
✔ 支持增量下載 (start_date, end_date 參數)
✔ 與 main.py / processor.py 兼容（stock_prices + stock_info）
✔ 保留雲端同步功能（但會避免「下載舊 DB 沒表」造成 stock_info 不存在）
"""

import os
import sys
import time
import sqlite3
import csv
import json
import io
from datetime import datetime

import pandas as pd
import yfinance as yf
import requests
from tqdm import tqdm
from dotenv import load_dotenv

from google.oauth2 import service_account
from googleapiclient.discovery import build
from googleapiclient.http import MediaIoBaseDownload, MediaFileUpload


# ========== 配置 ==========
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DB_PATH = os.path.join(BASE_DIR, "kr_stock_warehouse.db")
CSV_PATH = os.path.join(BASE_DIR, "krx_corp_list.csv")

# KRX corp list 下載網址（你提供的）
KRX_CORP_LIST_URL = "http://kind.krx.co.kr/corpgeneral/corpList.do?method=download&searchType=13"
KRX_CORP_LIST_REFERER = "http://kind.krx.co.kr/corpgeneral/corpList.do"

# Env
load_dotenv()
GDRIVE_FOLDER_ID = os.environ.get("GDRIVE_FOLDER_ID")


def log(msg: str):
    print(f"{datetime.now().strftime('%H:%M:%S')}: {msg}", flush=True)


# ========== 雲端服務函數 ==========
def get_drive_service():
    """獲取 Google Drive 服務實例"""
    env_json = os.environ.get("GDRIVE_SERVICE_ACCOUNT")
    try:
        if env_json:
            info = json.loads(env_json)
            creds = service_account.Credentials.from_service_account_info(
                info, scopes=["https://www.googleapis.com/auth/drive"]
            )
            return build("drive", "v3", credentials=creds, cache_discovery=False)
        return None
    except Exception as e:
        log(f"❌ Drive 服務初始化失敗: {e}")
        return None


def download_db_from_drive(service, file_name: str, local_path: str):
    """從 Google Drive 下載資料庫到指定路徑"""
    if not GDRIVE_FOLDER_ID or not service:
        return False

    query = f"name = '{file_name}' and '{GDRIVE_FOLDER_ID}' in parents and trashed = false"
    try:
        results = service.files().list(q=query, fields="files(id)").execute()
        items = results.get("files", [])
        if not items:
            log(f"ℹ️ 雲端沒有 {file_name}（將使用本地新建/既有 DB）")
            return False

        file_id = items[0]["id"]
        log(f"📡 從雲端同步韓國資料庫: {file_name} -> {local_path}")

        request = service.files().get_media(fileId=file_id)
        os.makedirs(os.path.dirname(local_path), exist_ok=True)

        with io.FileIO(local_path, "wb") as fh:
            downloader = MediaIoBaseDownload(fh, request, chunksize=5 * 1024 * 1024)
            done = False
            while not done:
                _, done = downloader.next_chunk()

        log("✅ 雲端下載完成")
        return True

    except Exception as e:
        log(f"⚠️ 雲端下載失敗: {e}")
        return False


def upload_db_to_drive(service, file_path: str):
    """上傳資料庫到 Google Drive"""
    if not GDRIVE_FOLDER_ID or not service or not os.path.exists(file_path):
        return False

    file_name = os.path.basename(file_path)
    file_size = os.path.getsize(file_path)

    chunk_size = 5 * 1024 * 1024
    if file_size > 100 * 1024 * 1024:
        chunk_size = 10 * 1024 * 1024

    try:
        media = MediaFileUpload(
            file_path, mimetype="application/x-sqlite3", resumable=True, chunksize=chunk_size
        )

        query = f"name = '{file_name}' and '{GDRIVE_FOLDER_ID}' in parents and trashed = false"
        results = service.files().list(q=query, fields="files(id)").execute()
        items = results.get("files", [])

        if items:
            request = service.files().update(fileId=items[0]["id"], media_body=media, fields="id")
            log("🔄 更新雲端韓國資料庫")
        else:
            meta = {"name": file_name, "parents": [GDRIVE_FOLDER_ID]}
            request = service.files().create(body=meta, media_body=media, fields="id")
            log("🆕 創建雲端韓國資料庫")

        response = None
        while response is None:
            status, response = request.next_chunk()
            if status:
                log(f"  上傳進度: {int(status.progress() * 100)}%")

        log("✅ 韓國資料庫上傳成功")
        return True

    except Exception as e:
        log(f"❌ 上傳失敗: {e}")
        return False


# ========== DB schema ==========
def init_db():
    """初始化資料庫表格（一定要能補齊 stock_info）"""
    conn = sqlite3.connect(DB_PATH)
    try:
        # stock_prices（與 processor.py 查詢兼容）
        conn.execute(
            """CREATE TABLE IF NOT EXISTS stock_prices (
                date TEXT,
                symbol TEXT,
                open REAL,
                high REAL,
                low REAL,
                close REAL,
                volume INTEGER,
                PRIMARY KEY (date, symbol)
            )"""
        )

        # stock_info（processor 需要 market/sector/market_detail）
        conn.execute(
            """CREATE TABLE IF NOT EXISTS stock_info (
                symbol TEXT PRIMARY KEY,
                name TEXT,
                sector TEXT,
                market TEXT,
                market_detail TEXT,
                updated_at TEXT
            )"""
        )

        # index
        conn.execute(
            """CREATE INDEX IF NOT EXISTS idx_symbol_date
               ON stock_prices (symbol, date)"""
        )
    finally:
        conn.close()

    log("✅ 韓國資料庫初始化完成（含 stock_info/stock_prices）")


# ========== KRX corp list：先抓 URL，失敗才用本地 ==========
def try_download_krx_corp_list_csv(save_path: str = CSV_PATH) -> bool:
    """
    嘗試從 KRX KIND 下載公司清單並存成 CSV（UTF-8-SIG）
    成功回 True；失敗回 False（後續可 fallback 本地檔）
    """
    headers = {
        "User-Agent": (
            "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
            "AppleWebKit/537.36 (KHTML, like Gecko) "
            "Chrome/119.0.0.0 Safari/537.36"
        ),
        "Referer": KRX_CORP_LIST_REFERER,
    }

    log("📡 嘗試從 KRX KIND 下載公司清單（若被擋會 fallback 本地 CSV）")

    try:
        resp = requests.get(KRX_CORP_LIST_URL, headers=headers, timeout=30)
        resp.raise_for_status()

        # KIND 這個 endpoint 通常回傳 HTML table
        dfs = pd.read_html(io.BytesIO(resp.content))
        if not dfs:
            log("⚠️ 已取得回應，但找不到表格（read_html 解析不到）")
            return False

        df = dfs[0]
        if df is None or df.empty:
            log("⚠️ 表格為空（可能被導向/阻擋/內容變更）")
            return False

        # 存成 csv
        df.to_csv(save_path, index=False, encoding="utf-8-sig")
        log(f"✅ KRX 公司清單下載成功，已存成: {save_path}")
        return True

    except Exception as e:
        log(f"⚠️ KRX 下載失敗：{e}")
        return False


# ========== 股票清單處理 ==========
def get_kr_stock_list():
    """
    從 CSV 文件獲取韓國股票清單，並同步寫入 stock_info。
    - market 固定寫 'KR'
    - market_detail 寫 'KOSPI' / 'KOSDAQ' / 'KONEX'（讓 processor 判斷更穩）
    """
    log("📡 讀取韓國股票清單...")

    # 1) 若本地 CSV 不存在，先嘗試抓 KRX
    if not os.path.exists(CSV_PATH):
        ok = try_download_krx_corp_list_csv(CSV_PATH)
        if not ok and not os.path.exists(CSV_PATH):
            log(f"❌ 找不到股票清單文件且無法從網路取得: {CSV_PATH}")
            return []

    stocks = []
    conn = sqlite3.connect(DB_PATH)

    try:
        with open(CSV_PATH, "r", encoding="utf-8-sig") as f:
            # 處理 BOM
            first = f.read(1)
            if first != "\ufeff":
                f.seek(0)

            reader = csv.DictReader(f)
            for row in reader:
                try:
                    company_name = (row.get("회사명", "") or "").strip()
                    market_kor = (row.get("시장구분", "") or "").strip()
                    code = (row.get("종목코드", "") or "").strip().zfill(6)
                    sector = (row.get("업종", "") or "").strip()

                    if not company_name or not code:
                        continue

                    # 市場對應
                    if market_kor == "유가":
                        suffix = ".KS"
                        market_detail = "KOSPI"
                    elif market_kor == "코스닥":
                        suffix = ".KQ"
                        market_detail = "KOSDAQ"
                    elif market_kor == "코넥스":
                        suffix = ".KN"
                        market_detail = "KONEX"
                    else:
                        continue

                    symbol = f"{code}{suffix}"

                    conn.execute(
                        """
                        INSERT OR REPLACE INTO stock_info
                        (symbol, name, sector, market, market_detail, updated_at)
                        VALUES (?, ?, ?, ?, ?, ?)
                        """,
                        (
                            symbol,
                            company_name,
                            sector,
                            "KR",
                            market_detail,
                            datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                        ),
                    )

                    stocks.append((symbol, company_name))

                except Exception as e:
                    log(f"⚠️ 處理股票行時出錯: {e}")
                    continue

        conn.commit()
        log(f"✅ 股票清單載入完成: {len(stocks)} 檔")
        return stocks

    except Exception as e:
        log(f"❌ 讀取 CSV 失敗: {e}")
        return []
    finally:
        conn.close()


# ========== 單一股票下載 ==========
def download_one_stock(symbol: str, start_date: str, end_date: str):
    """下載單一股票歷史數據"""
    max_retries = 3

    for attempt in range(max_retries):
        try:
            df = yf.download(
                symbol,
                start=start_date,
                end=end_date,
                progress=False,
                auto_adjust=True,
                threads=False,
                timeout=30,
            )

            if df is None or df.empty:
                if attempt < max_retries - 1:
                    time.sleep(2)
                    continue
                return None

            if isinstance(df.columns, pd.MultiIndex):
                df.columns = df.columns.get_level_values(0)

            df.reset_index(inplace=True)
            df.columns = [str(c).lower() for c in df.columns]

            date_col = "date" if "date" in df.columns else df.columns[0]
            df["date"] = pd.to_datetime(df[date_col]).dt.strftime("%Y-%m-%d")

            # 確保欄位存在
            for col in ["open", "high", "low", "close", "volume"]:
                if col not in df.columns:
                    df[col] = None

            df_final = df[["date", "open", "high", "low", "close", "volume"]].copy()
            df_final["symbol"] = symbol
            return df_final

        except Exception as e:
            if attempt < max_retries - 1:
                time.sleep(3)
                continue
            log(f"⚠️ 下載失敗 {symbol}: {e}")
            return None


# ========== 主下載函數 ==========
def run_sync(start_date=None, end_date=None):
    """
    韓國股市同步主函數

    參數:
        start_date: 開始日期 (YYYY-MM-DD)
        end_date: 結束日期 (YYYY-MM-DD)
    """
    start_time = time.time()

    if not start_date:
        start_date = "2023-01-01"
    if not end_date:
        end_date = datetime.now().strftime("%Y-%m-%d")

    log(f"🚀 啟動韓國股市同步 | 期間: {start_date} ~ {end_date}")

    # 1) 雲端同步：先下載舊 DB（如果有）
    service = get_drive_service()
    if service:
        download_db_from_drive(service, "kr_stock_warehouse.db", local_path=DB_PATH)

    # ✅ 2) 關鍵：下載後再補 schema（舊 DB 沒 stock_info 也能補）
    init_db()

    # 3) 股票清單（先抓 URL，不行才用本地）
    stocks = get_kr_stock_list()
    if not stocks:
        log("❌ 沒有可下載的股票（corp list 空或抓取失敗）")
        return {"success": 0, "total": 0, "has_changed": False}

    log(f"📊 開始下載 {len(stocks)} 檔韓國股票")

    conn = sqlite3.connect(DB_PATH, timeout=60)
    success_count = 0

    pbar = tqdm(stocks, desc="韓國下載", unit="檔")
    for symbol, name in pbar:
        pbar.set_postfix({"股票": name[:10]})

        df = download_one_stock(symbol, start_date, end_date)

        if df is not None and not df.empty:
            try:
                df.to_sql(
                    "stock_prices",
                    conn,
                    if_exists="append",
                    index=False,
                    method=lambda table, _conn, keys, data_iter: _conn.executemany(
                        f"INSERT OR REPLACE INTO {table.name} ({', '.join(keys)}) "
                        f"VALUES ({', '.join(['?']*len(keys))})",
                        data_iter,
                    ),
                )
                success_count += 1
            except Exception as e:
                log(f"⚠️ 存入資料庫失敗 {symbol}: {e}")

        time.sleep(0.1)

    conn.commit()

    log("🧹 優化資料庫...")
    try:
        conn.execute("VACUUM")
    except Exception as e:
        log(f"⚠️ VACUUM 失敗（可忽略）: {e}")
    conn.close()

    # 4) 上傳到雲端
    if service and success_count > 0:
        upload_db_to_drive(service, DB_PATH)

    duration = (time.time() - start_time) / 60

    log(
        f"""
✅ 韓國股市同步完成！
📊 統計:
   - 成功下載: {success_count}/{len(stocks)} 檔
   - 資料期間: {start_date} ~ {end_date}
   - 執行時間: {duration:.1f} 分鐘
"""
    )

    return {"success": success_count, "total": len(stocks), "has_changed": success_count > 0}


if __name__ == "__main__":
    start_date = None
    end_date = None

    if len(sys.argv) > 1:
        for arg in sys.argv[1:]:
            if arg.startswith("--start="):
                start_date = arg.split("=", 1)[1]
            elif arg.startswith("--end="):
                end_date = arg.split("=", 1)[1]

    run_sync(start_date, end_date)
