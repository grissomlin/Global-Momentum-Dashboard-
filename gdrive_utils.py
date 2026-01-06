# gdrive_utils.py
# -*- coding: utf-8 -*-

import io
import json
import os
import time
from typing import Optional, Callable

from google.oauth2 import service_account
from googleapiclient.discovery import build
from googleapiclient.http import MediaFileUpload, MediaIoBaseDownload

from config import DRIVE_SCOPES, ENV_GDRIVE_SERVICE_ACCOUNT


def get_drive_service() -> Optional[object]:
    """
    建立 Google Drive service。
    讀取環境變數 ENV_GDRIVE_SERVICE_ACCOUNT（JSON 字串）。
    """
    env_json = os.environ.get(ENV_GDRIVE_SERVICE_ACCOUNT)
    if not env_json:
        return None

    try:
        info = json.loads(env_json)
        creds = service_account.Credentials.from_service_account_info(info, scopes=DRIVE_SCOPES)
        return build("drive", "v3", credentials=creds, cache_discovery=False)
    except Exception as e:
        print(f"❌ Drive 服務初始化失敗: {e}")
        return None


def download_file_from_drive(
    service: object,
    folder_id: str,
    file_name: str,
    local_path: Optional[str] = None,
) -> bool:
    """
    從指定 folder 下載指定檔名到 local_path（預設同名）。
    """
    if not service or not folder_id:
        return False

    local_path = local_path or file_name

    query = f"name = '{file_name}' and '{folder_id}' in parents and trashed = false"
    try:
        results = service.files().list(q=query, fields="files(id)").execute()
        items = results.get("files", [])
        if not items:
            print(f"ℹ️ 雲端無 {file_name} 檔案，將使用本地新檔/空檔開始")
            return False

        file_id = items[0]["id"]
        print(f"📡 從雲端同步: {file_name}")

        request = service.files().get_media(fileId=file_id)
        with io.FileIO(local_path, "wb") as fh:
            downloader = MediaIoBaseDownload(fh, request, chunksize=5 * 1024 * 1024)
            done = False
            while not done:
                _, done = downloader.next_chunk()

        print(f"✅ 雲端下載完成: {local_path}")
        return True

    except Exception as e:
        print(f"⚠️ 雲端下載失敗 {file_name}: {e}")
        return False


def upload_file_to_drive_stable(
    service: object,
    folder_id: str,
    file_path: str,
    *,
    max_retries: int = 3,
    rebuild_service_fn: Optional[Callable[[], Optional[object]]] = None,
) -> bool:
    """
    穩定性上傳（resumable + retry + SSL/EOF 重新建 service）
    - 會先找 folder 內是否有同名檔，有就 update，沒有就 create
    """
    if not folder_id or not service:
        return False
    if not os.path.exists(file_path):
        return False

    file_name = os.path.basename(file_path)
    file_size = os.path.getsize(file_path)

    # 根據檔案大小調 chunk
    chunk_size = 5 * 1024 * 1024
    if file_size > 100 * 1024 * 1024:
        chunk_size = 10 * 1024 * 1024

    for attempt in range(1, max_retries + 1):
        try:
            media = MediaFileUpload(
                file_path,
                mimetype="application/x-sqlite3",
                resumable=True,
                chunksize=chunk_size,
            )

            query = f"name = '{file_name}' and '{folder_id}' in parents and trashed = false"
            results = service.files().list(q=query, fields="files(id)").execute()
            items = results.get("files", [])

            if items:
                print(f"🔄 更新雲端檔案 (attempt {attempt}/{max_retries})")
                request = service.files().update(fileId=items[0]["id"], media_body=media, fields="id")
            else:
                print(f"🆕 創建雲端檔案 (attempt {attempt}/{max_retries})")
                meta = {"name": file_name, "parents": [folder_id]}
                request = service.files().create(body=meta, media_body=media, fields="id")

            response = None
            while response is None:
                status, response = request.next_chunk()
                if status:
                    print(f"  上傳進度: {int(status.progress() * 100)}%")

            print(f"✅ {file_name} 上傳成功!")
            return True

        except Exception as e:
            msg = str(e)
            print(f"⚠️ 上傳失敗 {file_name} (attempt {attempt}/{max_retries}): {msg}")

            # 常見不穩定錯誤：SSL / EOF / connection reset
            is_network_flaky = any(k in msg for k in ["SSL", "EOF", "Connection reset", "Remote end closed"])
            if is_network_flaky and rebuild_service_fn:
                time.sleep(5 * attempt)
                new_service = rebuild_service_fn()
                if not new_service:
                    print("❌ 無法重新建立 Drive 服務")
                    return False
                service = new_service
            else:
                time.sleep(2 * attempt)

    print(f"❌ {file_name} 上傳失敗，已達最大重試次數")
    return False
