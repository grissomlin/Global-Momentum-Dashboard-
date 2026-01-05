-*- coding: utf-8 -*-
import os, sys, sqlite3, json, time, socket, io
import pandas as pd
from datetime import datetime, timedelta
from google.oauth2 import service_account
from googleapiclient.discovery import build
from googleapiclient.http import MediaFileUpload, MediaIoBaseDownload
from dotenv import load_dotenv

💡 1. 載入環境變數與環境設定
load_dotenv()
socket.setdefaulttimeout(600)

💡 2. 強制日期限制 (依需求鎖定)
FORCE_START_DATE = "2024-01-01"
FORCE_END_DATE = "2025-12-31"

GDRIVE_FOLDER_ID = os.environ.get('GDRIVE_FOLDER_ID')

💡 3. 導入特徵加工模組 (保留 processor)
try:
from processor import process_market_data
except ImportError:
print("⚠️ 系統提示：找不到 processor.py")
process_market_data = None

💡 5. 動態導入下載器模組
def load_downloader(module_name):
"""動態載入下載器模組，包含錯誤處理"""
try:
module = import(module_name)

text
    # 檢查模組是否有必要的 run_sync 函數
    if hasattr(module, 'run_sync'):
        return module
    else:
        print(f"⚠️ {module_name} 模組缺少 run_sync 函數")
        return None
except ImportError as e:
    print(f"⚠️ 無法載入 {module_name} 模組: {e}")
    return None
except Exception as e:
    print(f"⚠️ 載入 {module_name} 時發生錯誤: {e}")
    return None
載入所有下載器
downloader_tw = load_downloader('downloader_tw')
downloader_us = load_downloader('downloader_us')
downloader_cn = load_downloader('downloader_cn')
downloader_hk = load_downloader('downloader_hk')
downloader_jp = load_downloader('downloader_jp')
downloader_kr = load_downloader('downloader_kr')

建立市場映射
module_map = {
'tw': downloader_tw,
'us': downloader_us,
'cn': downloader_cn,
'hk': downloader_hk,
'jp': downloader_jp,
'kr': downloader_kr
}

========== 💡 輔助函式 ==========
def get_db_last_date(db_path):
"""取得資料庫最後更新日期"""
if not os.path.exists(db_path):
return None
try:
conn = sqlite3.connect(db_path)
res = conn.execute("SELECT MAX(date) FROM stock_prices").fetchone()
conn.close()
return res[0] if res[0] else None
except Exception:
return None

def check_market_requirements(market):
"""檢查市場特定需求"""
if market == 'kr':
# 檢查韓國市場需要的 CSV 文件
csv_files = ['krx_corp_list.csv']
missing_files = []

text
    for csv_file in csv_files:
        if not os.path.exists(csv_file):
            missing_files.append(csv_file)
    
    if missing_files:
        print(f"❌ 韓國市場需要以下文件: {', '.join(missing_files)}")
        print(f"   請將文件放置於當前目錄: {os.getcwd()}")
        return False
    
    print(f"✅ 找到韓國股票清單文件: {csv_files[0]}")

return True
def get_market_display_name(market_code):
"""取得市場顯示名稱"""
market_names = {
'tw': '台灣',
'us': '美國',
'cn': '中國',
'hk': '香港',
'jp': '日本',
'kr': '韓國'
}
return market_names.get(market_code, market_code.upper())

========== ☁️ Google Drive 服務函式 ==========
def get_drive_service():
"""取得 Google Drive 服務"""
env_json = os.environ.get('GDRIVE_SERVICE_ACCOUNT')
try:
if env_json:
info = json.loads(env_json)
creds = service_account.Credentials.from_service_account_info(
info, scopes=['https://www.googleapis.com/auth/drive'])
return build('drive', 'v3', credentials=creds, cache_discovery=False)
else:
print("⚠️ 未找到 GDRIVE_SERVICE_ACCOUNT 環境變數，跳過雲端同步")
return None
except Exception as e:
print(f"❌ Drive 服務初始化失敗: {e}")
return None

def download_db_from_drive(service, file_name):
"""從 Google Drive 下載資料庫"""
if not GDRIVE_FOLDER_ID:
print("⚠️ 未設定 GDRIVE_FOLDER_ID，跳過雲端下載")
return False

text
query = f"name = '{file_name}' and '{GDRIVE_FOLDER_ID}' in parents and trashed = false"
try:
    results = service.files().list(q=query, fields="files(id)").execute()
    items = results.get('files', [])
    if not items: 
        print(f"ℹ️ 雲端無 {file_name} 檔案，將創建新檔")
        return False
    
    file_id = items[0]['id']
    print(f"📡 從雲端同步: {file_name}")
    request = service.files().get_media(fileId=file_id)
    
    with io.FileIO(file_name, 'wb') as fh:
        downloader = MediaIoBaseDownload(fh, request, chunksize=5*1024*1024)
        done = False
        while not done: 
            _, done = downloader.next_chunk()
    
    print(f"✅ 雲端下載完成: {file_name}")
    return True
except Exception as e:
    print(f"⚠️ 雲端下載失敗 {file_name}: {e}")
    return False
def upload_db_to_drive(service, file_path, max_retries=3):
"""上傳資料庫到 Google Drive"""
if not GDRIVE_FOLDER_ID or not os.path.exists(file_path):
return False

text
file_name = os.path.basename(file_path)
file_size = os.path.getsize(file_path)
chunk_size = 5 * 1024 * 1024

if file_size > 100 * 1024 * 1024: 
    chunk_size = 10 * 1024 * 1024

for attempt in range(max_retries):
    try:
        media = MediaFileUpload(file_path, mimetype='application/x-sqlite3', 
                               resumable=True, chunksize=chunk_size)
        
        # 檢查是否已存在
        query = f"name = '{file_name}' and '{GDRIVE_FOLDER_ID}' in parents and trashed = false"
        results = service.files().list(q=query, fields="files(id)").execute()
        items = results.get('files', [])
        
        if items:
            print(f"🔄 更新雲端檔案 (第 {attempt+1} 次重試)")
            request = service.files().update(fileId=items[0]['id'], media_body=media, fields='id')
        else:
            print(f"🆕 創建雲端檔案 (第 {attempt+1} 次重試)")
            meta = {'name': file_name, 'parents': [GDRIVE_FOLDER_ID]}
            request = service.files().create(body=meta, media_body=media, fields='id')
        
        response = None
        while response is None:
            status, response = request.next_chunk()
            if status: 
                print(f"  上傳進度: {int(status.progress() * 100)}%")
        
        print(f"✅ {file_name} 上傳成功!")
        return True
        
    except Exception as e:
        error_msg = str(e)
        print(f"⚠️ 上傳失敗 {file_name}: {error_msg}")
        
        if "SSL" in error_msg or "EOF" in error_msg:
            time.sleep(5 * (attempt + 1))
            # 重新建立服務
            service = get_drive_service()
            if not service:
                print("❌ 無法重新建立 Drive 服務")
                return False
        else:
            time.sleep(2 * (attempt + 1))

print(f"❌ {file_name} 上傳失敗，已達最大重試次數")
return False
def optimize_database(db_file):
"""優化資料庫"""
try:
conn = sqlite3.connect(db_file)

text
    # 檢查表結構
    cursor = conn.cursor()
    cursor.execute("SELECT name FROM sqlite_master WHERE type='table'")
    tables = cursor.fetchall()
    
    print(f"🔧 優化資料庫: {db_file}")
    print(f"   發現 {len(tables)} 個表格")
    
    # 執行 VACUUM
    conn.execute("VACUUM")
    conn.close()
    
    print(f"✅ 資料庫優化完成: {db_file}")
    return True
    
except Exception as e:
    print(f"❌ 資料庫優化失敗 {db_file}: {e}")
    return False
========== 🏁 市場處理函式 ==========
def process_market(market_code, service):
"""處理單一市場的下載與處理流程"""
print(f"\n{'='*50}")
print(f"🚀 開始處理: {get_market_display_name(market_code)}市場 ({market_code.upper()})")
print(f"{'='*50}")

text
# 檢查下載器模組
downloader_module = module_map.get(market_code)
if not downloader_module:
    print(f"❌ {get_market_display_name(market_code)}市場下載器未載入，跳過")
    return False

# 檢查市場特定需求
if not check_market_requirements(market_code):
    return False

# 設定資料庫檔案名稱
db_file = f"{market_code}_stock_warehouse.db"

# 從雲端下載現有資料庫
if service:
    download_db_from_drive(service, db_file)

# 計算增量下載日期
last_date = get_db_last_date(db_file)
actual_start = FORCE_START_DATE

if last_date:
    try:
        last_date_dt = pd.to_datetime(last_date)
        next_day = last_date_dt + timedelta(days=1)
        actual_start = next_day.strftime("%Y-%m-%d")
        print(f"📅 最後更新日期: {last_date}，增量下載從: {actual_start}")
    except Exception:
        print(f"⚠️ 無法解析最後更新日期，從頭下載")

# 檢查是否需要下載
if actual_start and actual_start <= FORCE_END_DATE:
    print(f"📡 同步區間: {actual_start} ~ {FORCE_END_DATE}")
    
    # 執行下載
    try:
        download_start_time = time.time()
        
        # 執行下載器
        result = downloader_module.run_sync(
            start_date=actual_start, 
            end_date=FORCE_END_DATE
        )
        
        download_duration = time.time() - download_start_time
        
        if result and result.get('success', 0) > 0:
            success_count = result.get('success', 0)
            total_count = result.get('total', 0)
            
            print(f"✅ {get_market_display_name(market_code)}下載完成")
            print(f"   成功: {success_count}/{total_count}")
            print(f"   耗時: {download_duration:.1f}秒")
            
            # 執行特徵處理
            if process_market_data:
                print(f"🔧 開始特徵處理...")
                process_start_time = time.time()
                
                try:
                    process_market_data(db_file)
                    process_duration = time.time() - process_start_time
                    print(f"✅ 特徵處理完成，耗時: {process_duration:.1f}秒")
                except Exception as e:
                    print(f"❌ 特徵處理失敗: {e}")
            else:
                print(f"⚠️ 跳過特徵處理 (未載入 processor)")
            
            # 優化並上傳到雲端
            if service:
                print(f"☁️ 開始雲端同步...")
                upload_start_time = time.time()
                
                # 先優化資料庫
                if optimize_database(db_file):
                    # 上傳到雲端
                    if upload_db_to_drive(service, db_file):
                        upload_duration = time.time() - upload_start_time
                        print(f"✅ 雲端同步完成，耗時: {upload_duration:.1f}秒")
                    else:
                        print(f"⚠️ 雲端同步失敗")
                else:
                    print(f"⚠️ 跳過雲端同步 (資料庫優化失敗)")
            
            return True
        else:
            print(f"⚠️ {get_market_display_name(market_code)}下載未成功")
            if result:
                print(f"   成功: {result.get('success', 0)}/{result.get('total', 0)}")
            return False
            
    except Exception as e:
        print(f"❌ {get_market_display_name(market_code)}下載錯誤: {e}")
        import traceback
        traceback.print_exc()
        return False
else:
    print(f"⏭️ 無需更新，最後日期: {last_date}")
    return True
========== 🏁 主程式 ==========
def main():
"""主程式入口"""
print("🌍 全球股票數據同步系統")
print("="*50)

text
# 解析命令行參數
target_market = sys.argv[1].lower() if len(sys.argv) > 1 else 'all'

# 顯示系統資訊
print(f"📅 強制日期範圍: {FORCE_START_DATE} ~ {FORCE_END_DATE}")
print(f"🎯 目標市場: {get_market_display_name(target_market) if target_market != 'all' else '全部市場'}")

# 初始化雲端服務
service = get_drive_service()
if service and GDRIVE_FOLDER_ID:
    print("☁️ 雲端同步: 啟用")
else:
    print("☁️ 雲端同步: 停用")

# 確定要執行的市場
if target_market == 'all':
    markets_to_run = list(module_map.keys())
elif target_market in module_map:
    markets_to_run = [target_market]
else:
    print(f"❌ 未知的市場代碼: {target_market}")
    print(f"   可用的市場: {', '.join([f'{k}({get_market_display_name(k)})' for k in module_map.keys()])}")
    return

print(f"📊 將處理 {len(markets_to_run)} 個市場")

# 開始處理
start_time = time.time()
successful_markets = []
failed_markets = []

for market_code in markets_to_run:
    market_start_time = time.time()
    
    if process_market(market_code, service):
        successful_markets.append(market_code)
    else:
        failed_markets.append(market_code)
    
    market_duration = time.time() - market_start_time
    print(f"⏱️  {get_market_display_name(market_code)}處理時間: {market_duration:.1f}秒\n")

# 總結報告
total_duration = time.time() - start_time

print("="*50)
print("📊 處理總結報告")
print("="*50)

if successful_markets:
    print(f"✅ 成功處理: {len(successful_markets)} 個市場")
    for market in successful_markets:
        print(f"   - {get_market_display_name(market)}")

if failed_markets:
    print(f"❌ 處理失敗: {len(failed_markets)} 個市場")
    for market in failed_markets:
        print(f"   - {get_market_display_name(market)}")

print(f"\n⏱️  總處理時間: {total_duration:.1f}秒 ({total_duration/60:.1f}分鐘)")
print(f"✅ 同步完成!")
if name == "main":
main()
