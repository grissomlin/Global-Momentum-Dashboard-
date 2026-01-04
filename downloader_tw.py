# -*- coding: utf-8 -*-
import os, io, time, random, sqlite3, requests
import pandas as pd
import yfinance as yf
from io import StringIO
from datetime import datetime, timedelta
from tqdm import tqdm
from concurrent.futures import ThreadPoolExecutor, as_completed

# ========== 1. 環境設定 ==========
MARKET_CODE = "tw-share"
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DB_PATH = os.path.join(BASE_DIR, "tw_stock_warehouse.db")

def log(msg: str):
    print(f"{pd.Timestamp.now():%H:%M:%S}: {msg}", flush=True)

# ========== 2. 資料庫初始化 ==========
def init_db():
    conn = sqlite3.connect(DB_PATH)
    try:
        conn.execute('''CREATE TABLE IF NOT EXISTS stock_prices (
                            date TEXT, symbol TEXT, open REAL, high REAL, 
                            low REAL, close REAL, volume INTEGER,
                            PRIMARY KEY (date, symbol))''')
        conn.execute('''CREATE TABLE IF NOT EXISTS stock_info (
                            symbol TEXT PRIMARY KEY, name TEXT, sector TEXT, market TEXT, updated_at TEXT)''')
    finally:
        conn.close()

def get_last_date(symbol, conn):
    try:
        query = "SELECT MAX(date) FROM stock_prices WHERE symbol = ?"
        res = conn.execute(query, (symbol,)).fetchone()
        return res[0] if res[0] else None
    except:
        return None

# ========== 3. 獲取台股清單 (維持原樣) ==========
def get_tw_stock_list():
    url_configs = [
        {'name': 'listed', 'url': 'https://isin.twse.com.tw/isin/class_main.jsp?market=1&issuetype=1&Page=1&chklike=Y', 'suffix': '.TW'},
        {'name': 'otc', 'url': 'https://isin.twse.com.tw/isin/class_main.jsp?market=2&issuetype=4&Page=1&chklike=Y', 'suffix': '.TWO'},
        {'name': 'etf', 'url': 'https://isin.twse.com.tw/isin/class_main.jsp?owncode=&stockname=&isincode=&market=1&issuetype=I&industry_code=&Page=1&chklike=Y', 'suffix': '.TW'}
    ]
    log(f"📡 獲取台股清單...")
    conn = sqlite3.connect(DB_PATH)
    stock_list = []
    for cfg in url_configs:
        try:
            resp = requests.get(cfg['url'], timeout=15)
            dfs = pd.read_html(StringIO(resp.text), header=0)
            if not dfs: continue
            df = dfs[0]
            for _, row in df.iterrows():
                code = str(row['有價證券代號']).strip()
                name = str(row['有價證券名稱']).strip()
                if code.isalnum() and len(code) >= 4:
                    symbol = f"{code}{cfg['suffix']}"
                    conn.execute("INSERT OR REPLACE INTO stock_info VALUES (?, ?, ?, ?, ?)", 
                                 (symbol, name, str(row.get('產業別','')), cfg['name'], datetime.now().strftime("%Y-%m-%d")))
                    stock_list.append((symbol, name))
        except: continue
    conn.commit()
    conn.close()
    return list(set(stock_list))

# ========== 4. 多執行緒下載單元 ==========
def process_single_stock(item, start_date, end_date):
    """執行單一股票的檢查與下載邏輯"""
    symbol, name = item
    
    # 這裡重新建立連線，因為 SQLite 在多執行緒下寫入需要小心
    # 我們這裡先唯讀檢查日期
    conn = sqlite3.connect(DB_PATH, timeout=30)
    last_date = get_last_date(symbol, conn)
    conn.close()
    
    actual_start = start_date
    if last_date:
        if last_date >= end_date:
            return "skipped", None
        actual_start = (pd.to_datetime(last_date) + timedelta(days=1)).strftime('%Y-%m-%d')

    try:
        df = yf.download(symbol, start=actual_start, end=end_date, progress=False, 
                         auto_adjust=True, threads=False, timeout=15)
        if df is None or df.empty:
            return "no_data", None
        
        if isinstance(df.columns, pd.MultiIndex):
            df.columns = df.columns.get_level_values(0)
        df.reset_index(inplace=True)
        df.columns = [c.lower() for c in df.columns]
        df['date'] = pd.to_datetime(df['date']).dt.tz_localize(None).dt.strftime('%Y-%m-%d')
        
        df_final = df[['date', 'open', 'high', 'low', 'close', 'volume']].copy()
        df_final['symbol'] = symbol
        return "success", df_final
    except:
        return "error", None

# ========== 5. 主流程 (Multi-threading) ==========
def run_sync(start_date="2024-01-01", end_date="2025-12-31", max_workers=5):
    start_time = time.time()
    init_db()
    
    items = get_tw_stock_list()
    if not items: return {"success": 0, "total": 0}

    log(f"🚀 多執行緒同步啟動 | 線程數: {max_workers} | 目標: {len(items)} 檔")

    success_count = 0
    skip_count = 0
    
    # 使用 ThreadPoolExecutor
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        # 建立任務列表
        futures = {executor.submit(process_single_stock, item, start_date, end_date): item for item in items}
        
        conn = sqlite3.connect(DB_PATH, timeout=60)
        
        for future in tqdm(as_completed(futures), total=len(items), desc="TW併發下載"):
            status, df_res = future.result()
            
            if status == "skipped":
                skip_count += 1
            elif status == "success" and df_res is not None:
                # 寫入資料庫 (SQLite 寫入建議回到主線程處理以避免 lock)
                df_res.to_sql('stock_prices', conn, if_exists='append', index=False, 
                              method=lambda table, conn, keys, data_iter: 
                              conn.executemany(f"INSERT OR REPLACE INTO {table.name} ({', '.join(keys)}) VALUES ({', '.join(['?']*len(keys))})", data_iter))
                success_count += 1
            
            # 每 100 筆 commit 一次
            if (success_count + skip_count) % 100 == 0:
                conn.commit()

        conn.commit()
        log(f"🧹 優化資料庫...")
        conn.execute("VACUUM")
        conn.close()

    duration = (time.time() - start_time) / 60
    log(f"📊 同步完成！更新: {success_count} | 跳過: {skip_count} | 耗時: {duration:.1f} 分鐘")
    return {"success": success_count, "total": len(items)}

if __name__ == "__main__":
    # 建議 max_workers 設定在 5~10 之間，太高會被 Yahoo 封鎖 IP
    run_sync(max_workers=8)
