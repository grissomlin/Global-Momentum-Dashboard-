# -*- coding: utf-8 -*-
import os, sys, sqlite3, json, time, socket, io
import pandas as pd
from datetime import datetime, timedelta
from google.oauth2 import service_account
from googleapiclient.discovery import build
from googleapiclient.http import MediaFileUpload, MediaIoBaseDownload
from dotenv import load_dotenv

# 💡 載入環境變數
load_dotenv() 
socket.setdefaulttimeout(600)

# 💡 環境變數讀取
GDRIVE_FOLDER_ID = os.environ.get('GDRIVE_FOLDER_ID')

# 💡 導入特徵加工模組
try:
    from processor import process_market_data
except ImportError:
    print("⚠️ 系統提示：找不到 processor.py，將跳過特徵工程。")
    process_market_data = None

import downloader_tw, downloader_us, downloader_cn, downloader_hk, downloader_jp, downloader_kr

# ========== 💡 快取輔助函式 ==========

def get_db_last_date(db_path):
    """檢查資料庫中所有標的最新的日期，作為全域快取參考"""
    if not os.path.exists(db_path):
        return None
    try:
        conn = sqlite3.connect(db_path)
        # 抓取資料庫中最後一筆日期
        res = conn.execute("SELECT MAX(date) FROM stock_prices").fetchone()
        conn.close()
        return res[0] if res[0] else None
    except:
        return None

# ========== Google Drive 服務函式 ==========

def get_drive_service():
    env_json = os.environ.get('GDRIVE_SERVICE_ACCOUNT')
    try:
        if env_json:
            info = json.loads(env_json)
            creds = service_account.Credentials.from_service_account_info(
                info, scopes=['https://www.googleapis.com/auth/drive']
            )
            return build('drive', 'v3', credentials=creds, cache_discovery=False)
        else:
            print("❌ 錯誤：找不到環境變數 GDRIVE_SERVICE_ACCOUNT")
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
    """
    上傳資料庫到 Google Drive，加入重試機制
    """
    if not GDRIVE_FOLDER_ID or not os.path.exists(file_path): 
        print(f"⚠️ 無法上傳 {file_path}: 缺少 GDRIVE_FOLDER_ID 或檔案不存在")
        return False
    
    file_name = os.path.basename(file_path)
    file_size = os.path.getsize(file_path)
    
    print(f"📤 準備上傳 {file_name} (大小: {file_size/1024/1024:.2f} MB)")
    
    # 檢查檔案大小，如果太大需要調整 chunksize
    chunk_size = 5 * 1024 * 1024  # 5MB
    if file_size > 100 * 1024 * 1024:  # 如果大於 100MB
        chunk_size = 10 * 1024 * 1024  # 使用 10MB chunks
    
    for attempt in range(max_retries):
        try:
            # 每次重試都重新創建 media
            media = MediaFileUpload(
                file_path, 
                mimetype='application/x-sqlite3', 
                resumable=True,
                chunksize=chunk_size
            )
            
            query = f"name = '{file_name}' and '{GDRIVE_FOLDER_ID}' in parents and trashed = false"
            results = service.files().list(q=query, fields="files(id)").execute()
            items = results.get('files', [])
            
            if items:
                print(f"🔄 嘗試更新現有檔案 (第 {attempt+1}/{max_retries} 次嘗試)")
                file_id = items[0]['id']
                
                # 更新檔案
                request = service.files().update(
                    fileId=file_id,
                    media_body=media,
                    fields='id'
                )
                
                # 執行更新請求
                response = None
                while response is None:
                    status, response = request.next_chunk()
                    if status:
                        print(f"  上傳進度: {int(status.progress() * 100)}%")
                
            else:
                print(f"🔄 嘗試創建新檔案 (第 {attempt+1}/{max_retries} 次嘗試)")
                meta = {'name': file_name, 'parents': [GDRIVE_FOLDER_ID]}
                
                # 創建新檔案
                request = service.files().create(
                    body=meta,
                    media_body=media,
                    fields='id'
                )
                
                # 執行創建請求
                response = None
                while response is None:
                    status, response = request.next_chunk()
                    if status:
                        print(f"  上傳進度: {int(status.progress() * 100)}%")
            
            print(f"✅ {file_name} 上傳成功!")
            return True
            
        except Exception as e:
            error_msg = str(e)
            print(f"⚠️ {file_name} 上傳失敗 (第 {attempt+1}/{max_retries} 次): {error_msg}")
            
            # 檢查是否為 SSL 相關錯誤
            if "SSL" in error_msg or "EOF" in error_msg or "SSL23" in error_msg:
                print("  偵測到 SSL 連線問題，等待後重試...")
                time.sleep(5 * (attempt + 1))  # 指數退避等待
                
                # 嘗試重新獲取服務
                try:
                    service = get_drive_service()
                    if not service:
                        print("  ❌ 無法重新建立 Google Drive 服務")
                        continue
                except Exception as reconnect_error:
                    print(f"  ❌ 重新建立服務失敗: {reconnect_error}")
            else:
                # 其他錯誤，等待較短時間
                time.sleep(2 * (attempt + 1))
    
    print(f"❌ {file_name} 上傳失敗，已嘗試 {max_retries} 次")
    return False

# ========== 主程式邏輯 ==========

def main():
    target_market = sys.argv[1].lower() if len(sys.argv) > 1 else 'all'
    module_map = {
        'tw': downloader_tw, 'us': downloader_us, 'cn': downloader_cn, 
        'hk': downloader_hk, 'jp': downloader_jp, 'kr': downloader_kr
    }
    
    markets_to_run = [target_market] if target_market in module_map else list(module_map.keys())
    service = get_drive_service()

    # 設定預設下載區間
    DEFAULT_START = "2024-01-01"
    DEFAULT_END = (datetime.now() + timedelta(days=1)).strftime("%Y-%m-%d")

    for m in markets_to_run:
        db_file = f"{m}_stock_warehouse.db"
        print(f"\n--- 🚀 市場啟動: {m.upper()} ---")

        # 1. 下載雲端快取
        has_cache = False
        if service:
            has_cache = download_db_from_drive(service, db_file)
            if m == 'kr':
                download_db_from_drive(service, "kr_list_all.csv")

        # 2. 💡 計算增量更新日期 (快取核心邏輯)
        last_date = get_db_last_date(db_file)
        if last_date:
            # 如果快取存在，從最後一天的隔天開始抓
            actual_start = (pd.to_datetime(last_date) + timedelta(days=1)).strftime("%Y-%m-%d")
            print(f"📦 偵測到快取數據，最後日期: {last_date}。將從 {actual_start} 開始增量下載。")
            
            # 如果計算出的開始日期已經大於等於明天，則無需重複下載
            if actual_start >= DEFAULT_END:
                print(f"✨ 數據已是最新，跳過 {m.upper()} 下載步驟。")
                actual_start = None # 標記為不執行
        else:
            actual_start = DEFAULT_START
            print(f"🆕 無可用快取，將執行完整下載 (起始日: {actual_start})")

        # 3. 執行下載 (只有在需要更新時執行)
        if actual_start:
            target_module = module_map.get(m)
            if target_module:
                print(f"📡 正在抓取 {actual_start} ~ {DEFAULT_END} 的數據...")
                target_module.run_sync(start_date=actual_start, end_date=DEFAULT_END)
        
        # 4. 執行特徵工程加工
        if process_market_data and os.path.exists(db_file):
            print(f"🧪 執行特徵工程加工...")
            process_market_data(db_file)
        
        # 5. 優化資料庫並回傳雲端
        if service and os.path.exists(db_file):
            print(f"🧹 優化資料庫並同步至雲端快取...")
            try:
                conn = sqlite3.connect(db_file)
                conn.execute("VACUUM")
                conn.close()
                
                # 使用改進後的上傳函數
                if upload_db_to_drive(service, db_file):
                    print(f"✅ {db_file} 雲端快取更新成功!")
                else:
                    print(f"⚠️ {db_file} 雲端快取更新失敗，但本地檔案已儲存")
                    
            except Exception as e:
                print(f"❌ 資料庫優化或上傳失敗: {e}")
                
                # 嘗試簡單備份
                try:
                    backup_file = f"{db_file}.backup"
                    import shutil
                    shutil.copy2(db_file, backup_file)
                    print(f"📋 已建立本地備份: {backup_file}")
                except:
                    print("⚠️ 無法建立本地備份")

    print("\n✅ 所有選定市場處理完畢。")

if __name__ == "__main__":
    main()
