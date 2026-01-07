# -*- coding: utf-8 -*-
"""
only_feature.py
---------------
GitHub Actions / 手動執行用：只做 Feature Engineering（重建 stock_analysis），並將 DB 同步回 Google Drive。

需求：
- 環境變數
  - GDRIVE_SERVICE_ACCOUNT : JSON 字串（Service Account）
  - GDRIVE_FOLDER_ID      : Drive 資料夾 ID
- 同專案內：
  - processor.py -> process_market_data(db_path)
  - gdrive_utils.py -> get_drive_service / download_file_from_drive / upload_file_to_drive_stable

用法：
  python -u only_feature.py tw
  python -u only_feature.py us
"""

import os
import sys
from typing import Optional

from processor import process_market_data

# ✅ 直接用 gdrive_utils（不要再從 main.py 轉 import，避免名稱不一致）
from gdrive_utils import (
    get_drive_service,
    download_file_from_drive,
    upload_file_to_drive_stable,
)


def download_db_from_drive(service, db_file: str, folder_id: str) -> bool:
    """
    從 Drive folder 下載 db_file 到本地同名檔案。
    gdrive_utils.download_file_from_drive 的參數是 local_path（不是 output_path）。
    """
    try:
        ok = download_file_from_drive(
            service=service,
            folder_id=folder_id,
            file_name=db_file,
            file_path=db_file,
        )
        return bool(ok)
    except Exception as e:
        print(f"❌ 下載失敗：{e}")
        return False


def upload_db_to_drive(service, db_file: str, folder_id: str) -> bool:
    """把本地 db_file 上傳回 Drive folder（同名覆蓋/更新）。"""
    try:
        ok = upload_file_to_drive_stable(
            service=service,
            folder_id=folder_id,
            file_path=db_file,
            file_name=db_file,
        )
        return bool(ok)
    except Exception as e:
        print(f"❌ 上傳失敗：{e}")
        return False


def run_remote_process(market: str, db_file: Optional[str] = None) -> None:
    market = (market or "").lower().strip()
    if not market:
        print("❌ market 參數不可為空（例：tw/us/cn/hk/jp/kr）")
        return

    if db_file is None:
        db_file = f"{market}_stock_warehouse.db"

    folder_id = os.getenv("GDRIVE_FOLDER_ID", "").strip()
    if not folder_id:
        print("❌ 缺少環境變數 GDRIVE_FOLDER_ID（GitHub Secrets / Actions env）")
        return

    # 1) 建立 Drive service
    service = get_drive_service()
    if not service:
        print("❌ 無法建立 Google Drive 連線，請檢查 GDRIVE_SERVICE_ACCOUNT")
        return

    # 2) 若本地沒有 DB，先下載
    if not os.path.exists(db_file):
        print(f"📡 本地無快取，嘗試從雲端下載 {db_file}...")
        ok = download_db_from_drive(service, db_file, folder_id)
        if not ok:
            print(f"❌ 無法從雲端取得 {db_file}，請確認 Folder ID 與檔名是否正確")
            return

    # 3) Feature Engineering
    print(f"🧪 開始對 {market.upper()} 執行 Feature Layer（processor.py -> stock_analysis）...")
    process_market_data(db_file)

    # 4) 上傳回雲端
    if os.path.exists(db_file):
        print("📤 將加工後的數據庫同步回雲端...")
        ok = upload_db_to_drive(service, db_file, folder_id)
        if ok:
            print(f"✨ {market.upper()} 加工任務成功完成！")
        else:
            print("⚠️ 加工完成但上傳失敗（請檢查 Drive 權限/配額/網路）")
    else:
        print(f"❌ 錯誤：本地找不到 {db_file}（可能 processor 過程中被刪除/寫入失敗）")


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("用法：python -u only_feature.py <market>  (例：tw/us/cn/hk/jp/kr)")
        sys.exit(1)

    run_remote_process(sys.argv[1])
