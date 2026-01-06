# config.py
# -*- coding: utf-8 -*-

# ====== 強制日期限制（鎖死 2024-2025）======
FORCE_START_DATE = "2024-01-01"
FORCE_END_DATE = "2025-12-31"

# ====== 系統設定 ======
SOCKET_TIMEOUT = 600

# ====== Google Drive 環境變數名稱 ======
ENV_GDRIVE_FOLDER_ID = "GDRIVE_FOLDER_ID"
ENV_GDRIVE_SERVICE_ACCOUNT = "GDRIVE_SERVICE_ACCOUNT"

# ====== Google Drive scopes ======
DRIVE_SCOPES = ["https://www.googleapis.com/auth/drive"]
