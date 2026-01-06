# main.py
# -*- coding: utf-8 -*-

import os
import sys
import sqlite3
import time
import socket
import importlib
from datetime import timedelta

import pandas as pd
from dotenv import load_dotenv

from config import (
    FORCE_START_DATE,
    FORCE_END_DATE,
    SOCKET_TIMEOUT,
    ENV_GDRIVE_FOLDER_ID,
)
from gdrive_utils import (
    get_drive_service,
    download_file_from_drive,
    upload_file_to_drive_stable,
)

# 1) 載入環境變數與環境設定
load_dotenv()
socket.setdefaulttimeout(SOCKET_TIMEOUT)

GDRIVE_FOLDER_ID = os.environ.get(ENV_GDRIVE_FOLDER_ID)

# 2) 導入特徵加工模組（保留 processor）
try:
    from processor import process_market_data
except ImportError:
    print("⚠️ 系統提示：找不到 processor.py，將跳過特徵處理")
    process_market_data = None

# 2.5) 導入事件表引擎（limitup_events / daytrade_events）
try:
    from event_engine import build_events
except ImportError:
    print("⚠️ 系統提示：找不到 event_engine.py，將跳過事件表生成")
    build_events = None

# ✅ 只針對這些市場跑事件表（其他市場跳過）
EVENT_ENGINE_MARKETS = {"tw", "cn", "jp"}


def load_downloader(module_name: str):
    """動態載入下載器模組，並檢查是否有 run_sync。"""
    try:
        module = importlib.import_module(module_name)
        if hasattr(module, "run_sync"):
            return module
        print(f"⚠️ {module_name} 模組缺少 run_sync 函數")
        return None
    except ImportError as e:
        print(f"⚠️ 無法載入 {module_name} 模組: {e}")
        return None
    except Exception as e:
        print(f"⚠️ 載入 {module_name} 時發生錯誤: {e}")
        return None


# 3) 載入各市場下載器
module_map = {
    "tw": load_downloader("downloader_tw"),
    "us": load_downloader("downloader_us"),
    "cn": load_downloader("downloader_cn"),
    "hk": load_downloader("downloader_hk"),
    "jp": load_downloader("downloader_jp"),
    "kr": load_downloader("downloader_kr"),
}


def get_market_display_name(market_code: str) -> str:
    market_names = {"tw": "台灣", "us": "美國", "cn": "中國", "hk": "香港", "jp": "日本", "kr": "韓國"}
    return market_names.get(market_code, market_code.upper())


def get_db_last_date(db_path: str):
    """取得資料庫最後更新日期"""
    if not os.path.exists(db_path):
        return None
    try:
        conn = sqlite3.connect(db_path)
        res = conn.execute("SELECT MAX(date) FROM stock_prices").fetchone()
        conn.close()
        return res[0] if res and res[0] else None
    except Exception:
        return None


def optimize_database(db_file: str) -> bool:
    """可選：簡單 VACUUM，讓上傳檔案更乾淨"""
    try:
        conn = sqlite3.connect(db_file)
        conn.execute("VACUUM")
        conn.close()
        return True
    except Exception as e:
        print(f"⚠️ 資料庫優化失敗 {db_file}: {e}")
        return False


def process_market(market_code: str, drive_service):
    print(f"\n{'='*50}")
    print(f"🚀 開始處理: {get_market_display_name(market_code)}市場 ({market_code.upper()})")
    print(f"{'='*50}")

    downloader = module_map.get(market_code)
    if not downloader:
        print(f"❌ {get_market_display_name(market_code)}市場下載器未載入，跳過")
        return False

    db_file = f"{market_code}_stock_warehouse.db"

    # (A) 雲端下載（如果有啟用）
    if drive_service and GDRIVE_FOLDER_ID:
        download_file_from_drive(drive_service, GDRIVE_FOLDER_ID, db_file, local_path=db_file)

    # (B) 增量起始日
    last_date = get_db_last_date(db_file)
    actual_start = FORCE_START_DATE
    if last_date:
        try:
            next_day = pd.to_datetime(last_date) + timedelta(days=1)
            actual_start = next_day.strftime("%Y-%m-%d")
            print(f"📅 最後更新日期: {last_date}，增量下載從: {actual_start}")
        except Exception:
            print("⚠️ 無法解析最後更新日期，改為從頭下載")

    # (C) 檢查是否需要更新
    if actual_start and actual_start <= FORCE_END_DATE:
        print(f"📡 同步區間: {actual_start} ~ {FORCE_END_DATE}")

        try:
            t0 = time.time()
            result = downloader.run_sync(start_date=actual_start, end_date=FORCE_END_DATE)
            dt = time.time() - t0

            if not result or result.get("success", 0) <= 0:
                print(f"⚠️ {get_market_display_name(market_code)}下載未成功")
                if result:
                    print(f"   成功: {result.get('success', 0)}/{result.get('total', 0)}")
                return False

            print(f"✅ {get_market_display_name(market_code)}下載完成")
            print(f"   成功: {result.get('success', 0)}/{result.get('total', 0)}")
            print(f"   耗時: {dt:.1f}秒")

            # (D) 特徵處理
            if process_market_data:
                try:
                    print("🔧 開始特徵處理...")
                    t1 = time.time()
                    process_market_data(db_file)
                    print(f"✅ 特徵處理完成，耗時: {time.time()-t1:.1f}秒")
                except Exception as e:
                    print(f"❌ 特徵處理失敗: {e}")
            else:
                print("⚠️ 跳過特徵處理 (未載入 processor)")

            # (D2) 事件表生成：只針對 tw/cn/jp
            if market_code in EVENT_ENGINE_MARKETS:
                if build_events:
                    try:
                        print("🧩 開始生成事件表 (limitup_events / daytrade_events)...")
                        t2 = time.time()
                        build_events(db_file)
                        print(f"✅ 事件表生成完成，耗時: {time.time()-t2:.1f}秒")
                    except Exception as e:
                        print(f"❌ 事件表生成失敗: {e}")
                else:
                    print("⚠️ 跳過事件表生成 (未載入 event_engine)")
            else:
                print(f"⏭️ 跳過事件表生成（{market_code.upper()} 不在事件表目標市場 {sorted(EVENT_ENGINE_MARKETS)}）")

            # (E) 雲端上傳（穩定性上傳）
            if drive_service and GDRIVE_FOLDER_ID:
                print("☁️ 開始雲端同步...")
                optimize_database(db_file)
                ok = upload_file_to_drive_stable(
                    drive_service,
                    GDRIVE_FOLDER_ID,
                    db_file,
                    max_retries=3,
                    rebuild_service_fn=get_drive_service,
                )
                if not ok:
                    print("⚠️ 雲端同步失敗")

            return True

        except Exception as e:
            print(f"❌ {get_market_display_name(market_code)}下載錯誤: {e}")
            import traceback
            traceback.print_exc()
            return False

    print(f"⏭️ 無需更新，最後日期: {last_date}")
    return True


def main():
    print("🌍 全球股票數據同步系統")
    print("=" * 50)

    target_market = sys.argv[1].lower() if len(sys.argv) > 1 else "all"

    print(f"📅 強制日期範圍: {FORCE_START_DATE} ~ {FORCE_END_DATE}")
    print(f"🎯 目標市場: {get_market_display_name(target_market) if target_market != 'all' else '全部市場'}")

    drive_service = get_drive_service()
    cloud_enabled = bool(drive_service and GDRIVE_FOLDER_ID)
    print(f"☁️ 雲端同步: {'啟用' if cloud_enabled else '停用'}")

    if target_market == "all":
        markets_to_run = list(module_map.keys())
    elif target_market in module_map:
        markets_to_run = [target_market]
    else:
        print(f"❌ 未知的市場代碼: {target_market}")
        print("   可用的市場:", ", ".join([f"{k}({get_market_display_name(k)})" for k in module_map.keys()]))
        return

    print(f"📊 將處理 {len(markets_to_run)} 個市場")

    start_time = time.time()
    ok_list, fail_list = [], []

    for m in markets_to_run:
        t0 = time.time()
        (ok_list if process_market(m, drive_service) else fail_list).append(m)
        print(f"⏱️  {get_market_display_name(m)}處理時間: {time.time()-t0:.1f}秒\n")

    total = time.time() - start_time
    print("=" * 50)
    print("📊 處理總結報告")
    print("=" * 50)

    if ok_list:
        print(f"✅ 成功處理: {len(ok_list)} 個市場")
        for m in ok_list:
            print(f"   - {get_market_display_name(m)}")

    if fail_list:
        print(f"❌ 處理失敗: {len(fail_list)} 個市場")
        for m in fail_list:
            print(f"   - {get_market_display_name(m)}")

    print(f"\n⏱️  總處理時間: {total:.1f}秒 ({total/60:.1f}分鐘)")
    print("✅ 同步完成!")


if __name__ == "__main__":
    main()
