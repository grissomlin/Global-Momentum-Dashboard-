# -*- coding: utf-8 -*-
"""
only_feature.py
---------------
GitHub Actions 入口：下載(或使用快取) <market>_stock_warehouse.db，
跑 processor.py 產生/更新 stock_analysis（新增欄位、重建 feature layer），
然後把 DB 上傳回 Google Drive 覆蓋雲端版本。

✅ 不依賴 data_cleaning.py
✅ Google Drive 相關功能統一走 gdrive_utils.py
"""

import os
import sys

from processor import process_market_data

# Google Drive helpers
try:
    from gdrive_utils import (
        get_drive_service,
        download_file_from_drive,
        upload_file_to_drive_stable,
    )
    HAS_GDRIVE = True
except Exception as e:
    print(f"⚠️ 無法導入 gdrive_utils（Google Drive 功能將停用）: {e}")
    HAS_GDRIVE = False


def download_db_from_drive(service, db_file: str) -> bool:
    """相容舊命名：下載 db_file（會從 GDRIVE_FOLDER_ID 指定的資料夾找同名檔）"""
    folder_id = os.getenv("GDRIVE_FOLDER_ID")
    if not folder_id:
        print("❌ 缺少環境變數 GDRIVE_FOLDER_ID，無法從 Drive 下載")
        return False

    ok = download_file_from_drive(
        service=service,
        file_name=db_file,
        output_path=db_file,
        folder_id=folder_id,
    )
    return bool(ok)


def upload_db_to_drive(service, db_file: str) -> bool:
    """相容舊命名：上傳/覆蓋 db_file 到 GDRIVE_FOLDER_ID"""
    folder_id = os.getenv("GDRIVE_FOLDER_ID")
    if not folder_id:
        print("❌ 缺少環境變數 GDRIVE_FOLDER_ID，無法上傳到 Drive")
        return False

    ok = upload_file_to_drive_stable(
        service=service,
        local_path=db_file,
        drive_folder_id=folder_id,
        drive_filename=db_file,
    )
    return bool(ok)


def run_remote_process(market: str):
    market = (market or "tw").lower().strip()
    db_file = f"{market}_stock_warehouse.db"

    service = None
    if HAS_GDRIVE:
        service = get_drive_service()

    # 1) 取得 DB：優先使用快取；沒有再去雲端下載
    if not os.path.exists(db_file):
        if service:
            print(f"📡 本地無快取，嘗試從雲端下載 {db_file}...")
            ok = download_db_from_drive(service, db_file)
            if not ok:
                print(f"❌ 下載失敗：{db_file}（請確認 Drive 裡有同名檔案，或 Folder ID/憑證正確）")
        else:
            print("❌ 本地無 DB 且 Drive 功能不可用（缺少 Secrets 或套件）。")
    else:
        print(f"💾 使用本地快取 DB：{db_file}")

    # 2) 跑 feature layer
    if os.path.exists(db_file):
        print(f"🧪 開始對 {market.upper()} 執行 Feature Layer（processor.py -> stock_analysis）...")
        process_market_data(db_file)

        # 3) 上傳回雲端
        if service:
            print("📤 將加工後的數據庫同步回雲端...")
            ok = upload_db_to_drive(service, db_file)
            if ok:
                print(f"✨ {market.upper()} 加工任務成功完成並已同步！")
            else:
                print("⚠️ 加工完成，但上傳 Drive 失敗（請檢查 Folder ID/權限）。")
        else:
            print("⚠️ 加工完成，但 Drive 功能不可用，因此未同步。")
    else:
        print(f"❌ 錯誤：找不到 {db_file}，無法執行加工。")


if __name__ == "__main__":
    if len(sys.argv) > 1:
        target_market = sys.argv[1]
    else:
        target_market = "tw"
    run_remote_process(target_market)
