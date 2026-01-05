# -*- coding: utf-8 -*-
"""
downloader_kr.py
----------------
韓國股市資料下載器 (與主系統兼容版)

✔ 使用本地 CSV 文件獲取股票清單 (krx_corp_list.csv)
✔ 支持增量下載 (start_date, end_date 參數)
✔ 與主系統的 main.py 和 processor.py 兼容
✔ 保留雲端同步功能
"""

import os, sys, time, sqlite3, csv, json, io
import pandas as pd
import yfinance as yf
from datetime import datetime, timedelta
from tqdm import tqdm
from google.oauth2 import service_account
from googleapiclient.discovery import build
from googleapiclient.http import MediaIoBaseDownload, MediaFileUpload

# ========== 配置 ==========
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DB_PATH = os.path.join(BASE_DIR, "kr_stock_warehouse.db")
CSV_PATH = os.path.join(BASE_DIR, "krx_corp_list.csv")

# 從環境變數獲取 Google Drive 配置
import os
from dotenv import load_dotenv
load_dotenv()

GDRIVE_FOLDER_ID = os.environ.get('GDRIVE_FOLDER_ID')

def log(msg: str):
    print(f"{datetime.now().strftime('%H:%M:%S')}: {msg}", flush=True)

# ========== 雲端服務函數 ==========
def get_drive_service():
    """獲取 Google Drive 服務實例"""
    env_json = os.environ.get('GDRIVE_SERVICE_ACCOUNT')
    try:
        if env_json:
            info = json.loads(env_json)
            creds = service_account.Credentials.from_service_account_info(
                info, scopes=['https://www.googleapis.com/auth/drive'])
            return build('drive', 'v3', credentials=creds, cache_discovery=False)
        return None
    except Exception as e:
        log(f"❌ Drive 服務初始化失敗: {e}")
        return None

def download_db_from_drive(service, file_name):
    """從 Google Drive 下載資料庫"""
    if not GDRIVE_FOLDER_ID or not service:
        return False
    
    query = f"name = '{file_name}' and '{GDRIVE_FOLDER_ID}' in parents and trashed = false"
    try:
        results = service.files().list(q=query, fields="files(id)").execute()
        items = results.get('files', [])
        if not items:
            return False
        
        file_id = items[0]['id']
        log(f"📡 從雲端同步韓國資料庫: {file_name}")
        request = service.files().get_media(fileId=file_id)
        with io.FileIO(file_name, 'wb') as fh:
            downloader = MediaIoBaseDownload(fh, request, chunksize=5*1024*1024)
            done = False
            while not done:
                _, done = downloader.next_chunk()
        return True
    except Exception as e:
        log(f"⚠️ 雲端下載失敗: {e}")
        return False

def upload_db_to_drive(service, file_path):
    """上傳資料庫到 Google Drive"""
    if not GDRIVE_FOLDER_ID or not service or not os.path.exists(file_path):
        return False
    
    file_name = os.path.basename(file_path)
    file_size = os.path.getsize(file_path)
    
    # 根據文件大小調整分片大小
    chunk_size = 5 * 1024 * 1024
    if file_size > 100 * 1024 * 1024:
        chunk_size = 10 * 1024 * 1024
    
    try:
        media = MediaFileUpload(file_path, mimetype='application/x-sqlite3', 
                               resumable=True, chunksize=chunk_size)
        
        # 檢查是否已存在
        query = f"name = '{file_name}' and '{GDRIVE_FOLDER_ID}' in parents and trashed = false"
        results = service.files().list(q=query, fields="files(id)").execute()
        items = results.get('files', [])
        
        if items:
            # 更新現有文件
            request = service.files().update(fileId=items[0]['id'], media_body=media, fields='id')
            log("🔄 更新雲端韓國資料庫")
        else:
            # 創建新文件
            meta = {'name': file_name, 'parents': [GDRIVE_FOLDER_ID]}
            request = service.files().create(body=meta, media_body=media, fields='id')
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

# ========== 資料庫初始化 ==========
def init_db():
    """初始化資料庫表格"""
    conn = sqlite3.connect(DB_PATH)
    try:
        # 股價資料表
        conn.execute('''CREATE TABLE IF NOT EXISTS stock_prices (
                            date TEXT, 
                            symbol TEXT, 
                            open REAL, 
                            high REAL, 
                            low REAL, 
                            close REAL, 
                            volume INTEGER,
                            PRIMARY KEY (date, symbol))''')
        
        # 股票資訊表 (兼容 processor.py 需要的字段)
        conn.execute('''CREATE TABLE IF NOT EXISTS stock_info (
                            symbol TEXT PRIMARY KEY, 
                            name TEXT, 
                            sector TEXT, 
                            market TEXT,
                            market_detail TEXT,
                            updated_at TEXT)''')
        
        # 創建索引
        conn.execute('''CREATE INDEX IF NOT EXISTS idx_symbol_date 
                       ON stock_prices (symbol, date)''')
        
    finally:
        conn.close()
    log("✅ 韓國資料庫初始化完成")

# ========== 股票清單處理 ==========
def get_kr_stock_list():
    """從 CSV 文件獲取韓國股票清單"""
    log("📡 讀取韓國股票清單...")
    
    if not os.path.exists(CSV_PATH):
        log(f"❌ 找不到股票清單文件: {CSV_PATH}")
        return []
    
    stocks = []
    conn = sqlite3.connect(DB_PATH)
    
    try:
        with open(CSV_PATH, 'r', encoding='utf-8-sig') as f:
            # 跳過可能的 BOM 字元
            if f.read(1) == '\ufeff':
                f.seek(1)
            else:
                f.seek(0)
            
            reader = csv.DictReader(f)
            for row in reader:
                try:
                    # 提取股票資訊
                    company_name = row.get('회사명', '').strip()
                    market = row.get('시장구분', '').strip()
                    code = row.get('종목코드', '').strip().zfill(6)
                    sector = row.get('업종', '').strip()
                    region = row.get('지역', '').strip()
                    listing_date = row.get('상장일', '').strip()
                    
                    # 決定市場後綴
                    if market == '유가':
                        suffix = '.KS'  # KOSPI
                        market_detail = 'main'
                    elif market == '코스닥':
                        suffix = '.KQ'  # KOSDAQ
                        market_detail = 'kosdaq'
                    elif market == '코넥스':
                        suffix = '.KN'  # KONEX
                        market_detail = 'konex'
                    else:
                        continue  # 忽略其他市場
                    
                    symbol = f"{code}{suffix}"
                    
                    # 存入 stock_info 表
                    conn.execute("""
                        INSERT OR REPLACE INTO stock_info 
                        (symbol, name, sector, market, market_detail, updated_at) 
                        VALUES (?, ?, ?, ?, ?, ?)
                    """, (symbol, company_name, sector, market, market_detail, 
                          datetime.now().strftime("%Y-%m-%d")))
                    
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
def download_one_stock(symbol, start_date, end_date):
    """下載單一股票歷史數據"""
    max_retries = 3
    
    for attempt in range(max_retries):
        try:
            # 使用 yfinance 下載，禁用多執行緒避免問題
            df = yf.download(
                symbol, 
                start=start_date, 
                end=end_date,
                progress=False,
                auto_adjust=True,
                threads=False,
                timeout=30
            )
            
            if df is None or df.empty:
                if attempt < max_retries - 1:
                    time.sleep(2)
                    continue
                return None
            
            # 清理數據格式
            if isinstance(df.columns, pd.MultiIndex):
                df.columns = df.columns.get_level_values(0)
            
            df.reset_index(inplace=True)
            df.columns = [col.lower() for col in df.columns]
            
            # 標準化日期格式
            date_col = 'date' if 'date' in df.columns else df.columns[0]
            df['date_str'] = pd.to_datetime(df[date_col]).dt.strftime('%Y-%m-%d')
            
            # 選擇需要的欄位
            required_cols = ['date_str', 'open', 'high', 'low', 'close', 'volume']
            available_cols = [col for col in required_cols if col in df.columns]
            
            if 'date_str' not in available_cols:
                return None
            
            df_final = df[available_cols].copy()
            df_final.columns = ['date', 'open', 'high', 'low', 'close', 'volume']
            df_final['symbol'] = symbol
            
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
    
    # 設置日期範圍
    if not start_date:
        start_date = "2023-01-01"
    if not end_date:
        end_date = datetime.now().strftime("%Y-%m-%d")
    
    log(f"🚀 啟動韓國股市同步 | 期間: {start_date} ~ {end_date}")
    
    # 初始化資料庫
    init_db()
    
    # 雲端同步
    service = get_drive_service()
    if service:
        download_db_from_drive(service, "kr_stock_warehouse.db")
    
    # 獲取股票清單
    stocks = get_kr_stock_list()
    if not stocks:
        log("❌ 沒有可下載的股票")
        return {"success": 0, "total": 0, "has_changed": False}
    
    log(f"📊 開始下載 {len(stocks)} 檔韓國股票")
    
    # 連接資料庫
    conn = sqlite3.connect(DB_PATH, timeout=60)
    success_count = 0
    
    # 進度條
    pbar = tqdm(stocks, desc="韓國下載", unit="檔")
    for symbol, name in pbar:
        pbar.set_postfix({"股票": name[:10]})
        
        df = download_one_stock(symbol, start_date, end_date)
        
        if df is not None and not df.empty:
            try:
                # 存入資料庫
                df.to_sql(
                    'stock_prices', 
                    conn, 
                    if_exists='append', 
                    index=False,
                    method=lambda table, conn, keys, data_iter: 
                    conn.executemany(
                        f"INSERT OR REPLACE INTO {table.name} ({', '.join(keys)}) VALUES ({', '.join(['?']*len(keys))})", 
                        data_iter
                    )
                )
                success_count += 1
            except Exception as e:
                log(f"⚠️ 存入資料庫失敗 {symbol}: {e}")
        
        # 控制下載速度
        time.sleep(0.1)
    
    conn.commit()
    
    # 執行資料庫優化
    log("🧹 優化資料庫...")
    conn.execute("VACUUM")
    conn.close()
    
    # 上傳到雲端
    if service and success_count > 0:
        upload_db_to_drive(service, DB_PATH)
    
    # 計算執行時間
    duration = (time.time() - start_time) / 60
    
    log(f"""
✅ 韓國股市同步完成！
📊 統計:
   - 成功下載: {success_count}/{len(stocks)} 檔
   - 資料期間: {start_date} ~ {end_date}
   - 執行時間: {duration:.1f} 分鐘
    """)
    
    return {
        "success": success_count, 
        "total": len(stocks), 
        "has_changed": success_count > 0
    }

# ========== 命令行直接執行 ==========
if __name__ == "__main__":
    # 解析命令行參數
    start_date = None
    end_date = None
    
    if len(sys.argv) > 1:
        for arg in sys.argv[1:]:
            if arg.startswith("--start="):
                start_date = arg.split("=")[1]
            elif arg.startswith("--end="):
                end_date = arg.split("=")[1]
    
    run_sync(start_date, end_date)
