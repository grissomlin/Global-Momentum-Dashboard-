# -*- coding: utf-8 -*-
import sys, os
from processor import process_market_data

# 💡 從 main.py 借用雲端連線與傳輸功能
try:
    from main import get_drive_service, download_db_from_drive, upload_db_to_drive
except ImportError as e:
    print(f"⚠️ 導入 main.py 時發生錯誤（可能是依賴套件未安裝）: {e}")

def run_remote_process(market):
    db_file = f"{market}_stock_warehouse.db"
    
    # 💡 修正函式名稱：確保與 main.py 內的名稱一致
    service = get_drive_service() 

    if service:
        # 1. 如果本地沒檔案(快取失效)，去 Google Drive 抓原始檔
        if not os.path.exists(db_file):
            print(f"📡 本地無快取，嘗試從雲端下載 {db_file}...")
            download_db_from_drive(service, db_file)
        
        # 2. 執行特徵工程 (processor.py)
        if os.path.exists(db_file):
            print(f"🧪 開始對 {market.upper()} 執行資料清洗與特徵加工...")
            process_market_data(db_file)
            
            # 3. 加工完後，傳回雲端覆蓋舊檔
            print(f"📤 將加工後的數據庫同步回雲端...")
            upload_db_to_drive(service, db_file)
            print(f"✨ {market.upper()} 加工任務成功完成！")
        else:
            print(f"❌ 錯誤：無法從雲端取得 {db_file}，請確認 Folder ID 是否正確。")
    else:
        print("❌ 錯誤：無法建立 Google Drive 連線，請檢查 Secrets。")

if __name__ == "__main__":
    if len(sys.argv) > 1:
        target_market = sys.argv[1].lower()
        run_remote_process(target_market)
    else:
        print("請帶入參數，例如: python only_feature.py tw")
