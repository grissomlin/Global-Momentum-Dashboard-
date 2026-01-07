# -*- coding: utf-8 -*-
"""
only_feature.py
---------------
GitHub Actions / local CLI 用的「只跑 Feature Layer」入口。

功能：
1) 從 Google Drive 下載 <market>_stock_warehouse.db（若本機沒有）
2) 執行 processor.py -> process_market_data(db)
3) 上傳回 Google Drive 覆蓋同名 DB

依賴：
- processor.py : process_market_data
- gdrive_utils.py : get_drive_service / download_file_from_drive / upload_file_to_drive_stable

環境變數（Actions 由 secrets 注入）：
- GDRIVE_SERVICE_ACCOUNT : Service Account JSON（字串）
- GDRIVE_FOLDER_ID       : Drive folder id

用法：
  python -u only_feature.py tw
  python -u only_feature.py all
"""

import os
import sys
from typing import List

from processor import process_market_data

# Drive utils（以 gdrive_utils 為準，避免 main.py 介面變動）
try:
    from gdrive_utils import (
        get_drive_service,
        download_file_from_drive,
        upload_file_to_drive_stable,
    )
except Exception as e:
    get_drive_service = None
    download_file_from_drive = None
    upload_file_to_drive_stable = None
    print(f"❌ 無法匯入 gdrive_utils.py：{e}")

SUPPORTED_MARKETS: List[str] = ["tw", "us", "cn", "hk", "jp", "kr"]


def _folder_id() -> str:
    fid = os.getenv("GDRIVE_FOLDER_ID", "").strip()
    if not fid:
        raise RuntimeError("缺少環境變數 GDRIVE_FOLDER_ID（GitHub Secrets）")
    return fid


def download_db_from_drive(service, db_file: str) -> bool:
    """
    下載 DB（同名覆蓋到本機檔案 db_file）
    gdrive_utils.download_file_from_drive(service, folder_id, file_name, local_path)
    """
    if download_file_from_drive is None:
        raise RuntimeError("download_file_from_drive 不可用（gdrive_utils 匯入失敗）")

    folder_id = _folder_id()
    print(f"📡 從雲端同步: {db_file}")
    ok = download_file_from_drive(
        service=service,
        folder_id=folder_id,
        file_name=db_file,
        local_path=db_file,
    )
    if ok:
        print(f"✅ 雲端下載完成: {db_file}")
    else:
        print(f"❌ 雲端下載失敗: {db_file}（請確認 Folder ID / 檔名 / 權限）")
    return bool(ok)


def upload_db_to_drive(service, db_file: str) -> bool:
    """
    上傳 DB（本機 db_file -> Drive folder）
    gdrive_utils.upload_file_to_drive_stable(service, folder_id, file_path)
    """
    if upload_file_to_drive_stable is None:
        raise RuntimeError("upload_file_to_drive_stable 不可用（gdrive_utils 匯入失敗）")

    folder_id = _folder_id()
    print(f"📤 上傳回雲端: {db_file}")
    ok = upload_file_to_drive_stable(
        service=service,
        folder_id=folder_id,
        file_path=db_file,
    )
    if ok:
        print(f"✅ 上傳完成: {db_file}")
    else:
        print(f"❌ 上傳失敗: {db_file}（請檢查 Drive 權限/配額/網路/Service Account 權限）")
    return bool(ok)


def run_one_market(market: str) -> None:
    market = (market or "").lower().strip()
    if market not in SUPPORTED_MARKETS:
        raise ValueError(f"不支援 market={market}，可用：{SUPPORTED_MARKETS} 或 all")

    db_file = f"{market}_stock_warehouse.db"

    if get_drive_service is None:
        raise RuntimeError("get_drive_service 不可用（gdrive_utils 匯入失敗）")

    service = get_drive_service()
    if service is None:
        raise RuntimeError("無法建立 Google Drive 連線（請檢查 GDRIVE_SERVICE_ACCOUNT / 套件）")

    # 1) 下載（本機不存在才抓）
    if not os.path.exists(db_file):
        print(f"📡 本地無快取，嘗試從雲端下載 {db_file}...")
        ok = download_db_from_drive(service, db_file)
        if not ok or not os.path.exists(db_file):
            print(f"❌ 無法從雲端取得 {db_file}，請確認 Folder ID 與檔名是否正確")
            return
    else:
        print(f"💾 使用本地快取: {db_file}")

    # 2) Feature layer
    print(f"🧪 開始對 {market.upper()} 執行 Feature Layer（processor.py -> stock_analysis）...")
    process_market_data(db_file)

    # 3) 上傳
    print("📤 將加工後的數據庫同步回雲端...")
    ok = upload_db_to_drive(service, db_file)
    if not ok:
        print("⚠️ 加工完成但上傳失敗（請檢查 Drive 權限/配額/網路）")
    else:
        print(f"✨ {market.upper()} 加工任務成功完成！")


def run_all_markets() -> None:
    for m in SUPPORTED_MARKETS:
        print("=" * 80)
        print(f"🧪 Running market: {m}")
        try:
            run_one_market(m)
        except Exception as e:
            # all 模式：不中斷，讓其他市場繼續跑
            print(f"❌ 市場 {m} 失敗：{e}")


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("用法：python -u only_feature.py <market|all>")
        sys.exit(1)

    target = sys.argv[1].lower().strip()
    if target == "all":
        run_all_markets()
    else:
        run_one_market(target)
