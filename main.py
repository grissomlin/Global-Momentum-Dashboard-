# -*- coding: utf-8 -*-
import os, sys, sqlite3, json, time, socket, io
import pandas as pd
from datetime import datetime, timedelta
from google.oauth2 import service_account
from googleapiclient.discovery import build
from googleapiclient.http import MediaFileUpload, MediaIoBaseDownload
from dotenv import load_dotenv

# 💡 1. 載入環境變數與環境設定
load_dotenv() 
socket.setdefaulttimeout(600)

# 💡 2. 強制日期限制 (依需求鎖定)
FORCE_START_DATE = "2024-01-01"
FORCE_END_DATE = "2025-12-31"

GDRIVE_FOLDER_ID = os.environ.get('GDRIVE_FOLDER_ID')
SERVICE_ACCOUNT_FILE = 'citric-biplane-319514-75fead53b0f5.json'

# 💡 3. 導入特徵加工模組 (保留 processor)
try:
    from processor import process_market_data
except ImportError:
    print("⚠️ 系統提示：找不到 processor.py")
    process_market_data = None

# 💡 4. 【關鍵修正】動態導入模組，避免因缺少檔案崩潰，但保留所有市場功能
def dynamic_import(name):
    try:
        return __import__(name)
    except ImportError:
        return None

# 這裡依然保留所有國家的接口，不會因為你現在只有台灣檔案就報錯
downloader_tw = dynamic_import('downloader_tw')
downloader_us = dynamic_import('downloader_us')
downloader_cn = dynamic_import('downloader_cn')
downloader_hk = dynamic_import('downloader_hk')
downloader_jp = dynamic_import('downloader_jp')
downloader_kr = dynamic_import('downloader_kr')

module_map = {
    'tw': downloader_tw, 'us': downloader_us, 'cn': downloader_cn, 
    'hk': downloader_hk, 'jp': downloader_jp, 'kr': downloader_kr
}

# ========== 💡 輔助函式 (完全保留原始邏輯) ==========

def get_db_last_date(db_path):
    if not os.path.exists(db_path): return None
    try:
        conn = sqlite3.connect(db_path)
        res = conn.execute("SELECT MAX(date) FROM stock_prices").fetchone()
        conn.close()
        return res[0] if res[0] else None
    except: return None

# ========== ☁️ Google Drive 服務函式 (保留完整 100+ 行穩定性代碼) ==========

def get_drive_service():
    env_json = os.environ.get('GDRIVE_SERVICE_ACCOUNT')
    try:
        if env_json:
            info = json.loads(env_json)
            creds = service_account.Credentials.from_service_account_info(
                info, scopes=['https://www.googleapis.com/auth/drive'])
            return build('drive', 'v3', credentials=creds, cache_discovery=False)
        return None
    except Exception as e:
        print(f"❌ Drive 服務初始化失敗: {e}")
        return None

def download_db_from_drive(service, file_name):
    if not GDRIVE_FOLDER_ID: return False
    query = f"name = '{file_name}' and '{GDRIVE_FOLDER_ID}' in parents and trashed = false"
    try:
        results = service.files().list(q=query, fields="files(id)").execute()
        items = results.get('files', [])
        if not items: return False
        
        file_id = items[0]['id']
        print(f"📡 從雲端同步快取檔案: {file_name}")
        request = service.files().get_media(fileId=file_id)
        with io.FileIO(file_name, 'wb') as fh:
            downloader = MediaIoBaseDownload(fh, request, chunksize=5*1024*1024)
            done = False
            while not done: _, done = downloader.next_chunk()
        return True
    except: return False

def upload_db_to_drive(service, file_path, max_retries=3):
    """【完整保留】您原始代碼中的分片上傳、進度顯示與 SSL 重試機制，一行都不刪"""
    if not GDRIVE_FOLDER_ID or not os.path.exists(file_path): 
        print(f"⚠️ 無法上傳 {file_path}")
        return False
    
    file_name = os.path.basename(file_path)
    file_size = os.path.getsize(file_path)
    chunk_size = 5 * 1024 * 1024
    if file_size > 100 * 1024 * 1024: chunk_size = 10 * 1024 * 1024
    
    for attempt in range(max_retries):
        try:
            media = MediaFileUpload(file_path, mimetype='application/x-sqlite3', resumable=True, chunksize=chunk_size)
            query = f"name = '{file_name}' and '{GDRIVE_FOLDER_ID}' in parents and trashed = false"
            results = service.files().list(q=query, fields="files(id)").execute()
            items = results.get('files', [])
            
            if items:
                print(f"🔄 更新現有檔案 (第 {attempt+1} 次重試)")
                request = service.files().update(fileId=items[0]['id'], media_body=media, fields='id')
            else:
                print(f"🆕 創建新檔案 (第 {attempt+1} 次重試)")
                meta = {'name': file_name, 'parents': [GDRIVE_FOLDER_ID]}
                request = service.files().create(body=meta, media_body=media, fields='id')
            
            response = None
            while response is None:
                status, response = request.next_chunk()
                if status: print(f"  上傳進度: {int(status.progress() * 100)}%")
            print(f"✅ {file_name} 上傳成功!")
            return True
            
        except Exception as e:
            error_msg = str(e)
            print(f"⚠️ 上傳失敗: {error_msg}")
            if "SSL" in error_msg or "EOF" in error_msg:
                time.sleep(5 * (attempt + 1))
                service = get_drive_service() # 重連
            else:
                time.sleep(2 * (attempt + 1))
    return False

# ========== 🏁 主程式 ==========

def main():
    target_market = sys.argv[1].lower() if len(sys.argv) > 1 else 'all'
    service = get_drive_service()

    # 只針對有定義的市場跑
    markets_to_run = [target_market] if target_market in module_map else list(module_map.keys())

    for m in markets_to_run:
        target_module = module_map.get(m)
        if not target_module: # 💡 如果沒檔案就自動跳過，不會再噴報錯中止了！
            print(f"⏭️ 市場 {m.upper()} 缺少下載器檔案，跳過。")
            continue
            
        db_file = f"{m}_stock_warehouse.db"
        print(f"\n--- 🚀 市場啟動: {m.upper()} ---")

        # 1. 抓取快取
        if service:
            download_db_from_drive(service, db_file)

        # 2. 增量日期計算
        last_date = get_db_last_date(db_file)
        actual_start = FORCE_START_DATE
        if last_date:
            actual_start = (pd.to_datetime(last_date) + timedelta(days=1)).strftime("%Y-%m-%d")

        # 3. 💡 執行下載與加工 (強制鎖定在 2024-2025)
        if actual_start and actual_start <= FORCE_END_DATE:
            print(f"📡 同步區間: {actual_start} ~ {FORCE_END_DATE}")
            target_module.run_sync(start_date=actual_start, end_date=FORCE_END_DATE)
            
            # 特徵加工
            if process_market_data:
                process_market_data(db_file)

            # 優化與回傳
            if service:
                try:
                    conn = sqlite3.connect(db_file)
                    conn.execute("VACUUM")
                    conn.close()
                    upload_db_to_drive(service, db_file)
                except Exception as e:
                    print(f"❌ 優化上傳失敗: {e}")

    print("\n✅ 所有選定市場處理完畢。")

if __name__ == "__main__":
    main()
