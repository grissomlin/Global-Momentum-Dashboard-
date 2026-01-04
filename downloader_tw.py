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
    """初始化資料庫表結構，與 processor.py 兼容"""
    conn = sqlite3.connect(DB_PATH)
    try:
        # stock_prices 表 - 欄位順序與 processor.py 查詢兼容
        conn.execute('''CREATE TABLE IF NOT EXISTS stock_prices (
                            symbol TEXT,
                            date TEXT,
                            open REAL,
                            high REAL,
                            low REAL,
                            close REAL,
                            volume INTEGER,
                            PRIMARY KEY (symbol, date))''')
        
        # stock_info 表 - 擴展 market 欄位以便 processor.py 正確識別
        conn.execute('''CREATE TABLE IF NOT EXISTS stock_info (
                            symbol TEXT PRIMARY KEY,
                            name TEXT,
                            sector TEXT,
                            market TEXT,
                            market_detail TEXT,
                            updated_at TEXT)''')
        
        # 確保 stock_analysis 表存在（processor.py 會重建，但先建立以備不時之需）
        conn.execute('''CREATE TABLE IF NOT EXISTS stock_analysis (
                            symbol TEXT,
                            date TEXT,
                            open REAL,
                            high REAL,
                            low REAL,
                            close REAL,
                            volume INTEGER,
                            market TEXT,
                            sector TEXT,
                            daily_change REAL,
                            prev_close REAL,
                            avg_vol_20 REAL,
                            year INTEGER,
                            is_limit_up INTEGER,
                            strength_rank TEXT,
                            lu_type TEXT,
                            consecutive_limits INTEGER,
                            peak_date TEXT,
                            peak_high_ret REAL,
                            strong_day_contribution REAL,
                            ma20 REAL,
                            ma60 REAL,
                            macd REAL,
                            macds REAL,
                            macdh REAL,
                            year_start_price REAL,
                            ytd_ret REAL,
                            PRIMARY KEY (symbol, date))''')
        
        # 創建索引提高查詢效率
        conn.execute("CREATE INDEX IF NOT EXISTS idx_stock_prices_date ON stock_prices (date)")
        conn.execute("CREATE INDEX IF NOT EXISTS idx_stock_prices_symbol ON stock_prices (symbol)")
        
    finally:
        conn.close()

def get_last_date(symbol, conn):
    """獲取某股票最後更新日期"""
    try:
        query = "SELECT MAX(date) FROM stock_prices WHERE symbol = ?"
        res = conn.execute(query, (symbol,)).fetchone()
        return res[0] if res[0] else None
    except:
        return None

# ========== 3. 獲取台股清單（改進版） ==========
def get_tw_stock_list():
    """獲取台灣股票清單，改進市場分類以配合 processor.py"""
    url_configs = [
        {
            'name': '上市', 
            'market': '上市',
            'market_detail': 'listed',
            'url': 'https://isin.twse.com.tw/isin/class_main.jsp?market=1&issuetype=1&Page=1&chklike=Y', 
            'suffix': '.TW',
            'category': 'stock'
        },
        {
            'name': '上櫃', 
            'market': '上櫃',
            'market_detail': 'otc',
            'url': 'https://isin.twse.com.tw/isin/class_main.jsp?market=2&issuetype=4&Page=1&chklike=Y', 
            'suffix': '.TWO',
            'category': 'stock'
        },
        {
            'name': '興櫃', 
            'market': '興櫃',
            'market_detail': 'emerging',
            'url': 'https://isin.twse.com.tw/isin/class_main.jsp?market=E&issuetype=R&Page=1&chklike=Y', 
            'suffix': '.TWO',
            'category': 'stock'
        },
        {
            'name': 'ETF', 
            'market': '上市',  # ETF 也歸類為上市市場
            'market_detail': 'etf',
            'url': 'https://isin.twse.com.tw/isin/class_main.jsp?market=1&issuetype=I&Page=1&chklike=Y', 
            'suffix': '.TW',
            'category': 'etf'
        },
        {
            'name': '臺灣創新板', 
            'market': '上市',  # 創新板歸類為上市
            'market_detail': 'innovation',
            'url': 'https://isin.twse.com.tw/isin/class_main.jsp?market=C&issuetype=C&Page=1&chklike=Y', 
            'suffix': '.TW',
            'category': 'stock'
        }
    ]
    
    log(f"📡 獲取台股清單...")
    conn = sqlite3.connect(DB_PATH)
    stock_list = []
    
    for cfg in url_configs:
        try:
            resp = requests.get(cfg['url'], timeout=15)
            dfs = pd.read_html(StringIO(resp.text), header=0)
            if not dfs: 
                continue
                
            df = dfs[0]
            # 處理欄位名稱不一致問題
            df.columns = [str(col).strip() for col in df.columns]
            
            for _, row in df.iterrows():
                # 嘗試不同的欄位名稱
                code_field = None
                for field in ['有價證券代號', '代號', '證券代號']:
                    if field in df.columns:
                        code_field = field
                        break
                
                name_field = None
                for field in ['有價證券名稱', '名稱', '證券名稱']:
                    if field in df.columns:
                        name_field = field
                        break
                
                if not code_field or not name_field:
                    continue
                    
                code = str(row[code_field]).strip()
                name = str(row[name_field]).strip()
                
                # 過濾無效代碼
                if code and code.isalnum() and 4 <= len(code) <= 6:
                    symbol = f"{code}{cfg['suffix']}"
                    
                    # 獲取產業別
                    sector = ''
                    for field in ['產業別', '產業分類', '類別']:
                        if field in df.columns:
                            sector = str(row[field]).strip()
                            break
                    
                    # 存入資料庫
                    conn.execute("""
                        INSERT OR REPLACE INTO stock_info 
                        (symbol, name, sector, market, market_detail, updated_at) 
                        VALUES (?, ?, ?, ?, ?, ?)
                    """, (
                        symbol, 
                        name, 
                        sector, 
                        cfg['market'], 
                        cfg['market_detail'], 
                        datetime.now().strftime("%Y-%m-%d %H:%M:%S")
                    ))
                    
                    stock_list.append((symbol, name, cfg['market'], cfg['market_detail']))
                    
            log(f"✅ {cfg['name']} 載入完成，共 {len([s for s in stock_list if s[3]==cfg['market_detail']])} 檔")
            
        except Exception as e:
            log(f"❌ {cfg['name']} 抓取失敗: {str(e)}")
            continue
    
    conn.commit()
    conn.close()
    
    # 去重
    final_list = list({(s[0], s[1]): s for s in stock_list}.values())
    log(f"📊 全市場掃描完畢，總計 {len(final_list)} 檔有效標的")
    
    return final_list

# ========== 4. 多執行緒下載單元（改進版） ==========
def process_single_stock(item, start_date, end_date, retry_count=0):
    """執行單一股票的檢查與下載邏輯（含重試機制）"""
    symbol, name, market, market_detail = item
    
    # 檢查最後更新日期
    conn = sqlite3.connect(DB_PATH, timeout=30)
    last_date = get_last_date(symbol, conn)
    conn.close()
    
    # 計算實際開始日期
    actual_start = start_date
    if last_date:
        last_dt = pd.to_datetime(last_date)
        end_dt = pd.to_datetime(end_date)
        if last_dt >= end_dt:
            return "skipped", None
        actual_start = (last_dt + timedelta(days=1)).strftime('%Y-%m-%d')
    
    # 檢查是否需要下載
    if pd.to_datetime(actual_start) > pd.to_datetime(end_date):
        return "skipped", None
    
    try:
        # 下載股票數據
        df = yf.download(
            symbol, 
            start=actual_start, 
            end=end_date, 
            progress=False, 
            auto_adjust=True, 
            threads=False, 
            timeout=30
        )
        
        if df is None or df.empty:
            return "no_data", None
        
        # 處理多層索引
        if isinstance(df.columns, pd.MultiIndex):
            df.columns = df.columns.get_level_values(0)
        
        # 重設索引並重命名欄位
        df.reset_index(inplace=True)
        df.columns = [str(c).lower() for c in df.columns]
        
        # 處理日期格式
        if 'date' in df.columns:
            df['date'] = pd.to_datetime(df['date']).dt.tz_localize(None).dt.strftime('%Y-%m-%d')
        elif 'index' in df.columns:
            df['date'] = pd.to_datetime(df['index']).dt.tz_localize(None).dt.strftime('%Y-%m-%d')
            df = df.drop('index', axis=1)
        
        # 確保所有必要欄位都存在
        required_cols = ['open', 'high', 'low', 'close', 'volume']
        for col in required_cols:
            if col not in df.columns:
                df[col] = None
        
        # 選擇需要的欄位並添加symbol
        df = df[['date'] + required_cols].copy()
        df['symbol'] = symbol
        
        # 確保欄位順序一致
        df = df[['symbol', 'date', 'open', 'high', 'low', 'close', 'volume']]
        
        return "success", df
        
    except Exception as e:
        # 重試邏輯
        if retry_count < 2:
            time.sleep(2 * (retry_count + 1))
            return process_single_stock(item, start_date, end_date, retry_count + 1)
        else:
            log(f"⚠️ {symbol} 下載失敗（重試{retry_count+1}次）: {str(e)}")
            return "error", None

# ========== 5. 主流程（Multi-threading） ==========
def run_sync(start_date="2024-01-01", end_date="2025-12-31", max_workers=8):
    """主同步流程"""
    start_time = time.time()
    
    # 初始化資料庫
    init_db()
    
    # 獲取股票清單
    log("📋 開始獲取股票清單...")
    items = get_tw_stock_list()
    
    if not items:
        log("❌ 未獲取到任何股票清單")
        return {"success": 0, "total": 0, "skipped": 0, "failed": 0}
    
    log(f"🚀 多執行緒同步啟動 | 線程數: {max_workers} | 目標: {len(items)} 檔")
    
    # 統計變數
    success_count = 0
    skip_count = 0
    error_count = 0
    
    # 分批處理，避免記憶體不足
    batch_size = 500
    total_batches = (len(items) + batch_size - 1) // batch_size
    
    for batch_num in range(total_batches):
        batch_start = batch_num * batch_size
        batch_end = min((batch_num + 1) * batch_size, len(items))
        batch_items = items[batch_start:batch_end]
        
        log(f"📦 處理批次 {batch_num+1}/{total_batches} ({batch_start+1}-{batch_end})")
        
        # 使用 ThreadPoolExecutor
        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            # 建立任務列表
            futures = {
                executor.submit(process_single_stock, item, start_date, end_date): item 
                for item in batch_items
            }
            
            # 批次資料庫連接
            conn = sqlite3.connect(DB_PATH, timeout=120)
            
            # 處理結果
            for future in tqdm(
                as_completed(futures), 
                total=len(batch_items), 
                desc=f"批次 {batch_num+1}",
                leave=False
            ):
                status, df_res = future.result()
                
                if status == "skipped":
                    skip_count += 1
                elif status == "success" and df_res is not None:
                    try:
                        # 批次寫入資料庫
                        df_res.to_sql(
                            'stock_prices', 
                            conn, 
                            if_exists='append', 
                            index=False,
                            method=lambda table, conn, keys, data_iter: 
                                conn.executemany(
                                    f"INSERT OR IGNORE INTO {table.name} ({', '.join(keys)}) VALUES ({', '.join(['?']*len(keys))})", 
                                    data_iter
                                )
                        )
                        success_count += 1
                    except Exception as e:
                        error_count += 1
                        log(f"⚠️ 寫入資料庫失敗: {str(e)}")
                else:
                    error_count += 1
            
            # 提交批次
            conn.commit()
            conn.close()
        
        # 批次間隔，避免過度請求
        if batch_num < total_batches - 1:
            time.sleep(2)
    
    # 最終資料庫優化
    log(f"🧹 優化資料庫...")
    conn = sqlite3.connect(DB_PATH, timeout=120)
    conn.execute("VACUUM")
    conn.close()
    
    # 計算統計資訊
    duration = (time.time() - start_time) / 60
    
    log(f"""
📊 同步完成！
✅ 成功更新: {success_count} 檔
⏭️ 已跳過: {skip_count} 檔
❌ 失敗: {error_count} 檔
⏱️ 總耗時: {duration:.1f} 分鐘
    """)
    
    return {
        "success": success_count, 
        "total": len(items), 
        "skipped": skip_count, 
        "failed": error_count
    }

if __name__ == "__main__":
    # 測試執行
    result = run_sync(
        start_date="2024-01-01", 
        end_date="2025-12-31", 
        max_workers=6
    )
    print(f"執行結果: {result}")
