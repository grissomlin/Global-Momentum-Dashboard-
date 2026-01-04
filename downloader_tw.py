# -*- coding: utf-8 -*-
import os, io, time, sqlite3, requests
import pandas as pd
from io import StringIO
from datetime import datetime

# 假設 DB_PATH 已定義
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DB_PATH = os.path.join(BASE_DIR, "tw_stock_warehouse.db")

def log(msg: str):
    print(f"{pd.Timestamp.now():%H:%M:%S}: {msg}", flush=True)

def get_tw_stock_list():
    # ✅ 完整定義清單，包含創新板 (Market C) 與 戰略新板 (Market A)
    url_configs = [
        {'name': '上市', 'url': 'https://isin.twse.com.tw/isin/class_main.jsp?market=1&issuetype=1&Page=1&chklike=Y', 'suffix': '.TW'},
        {'name': '上櫃', 'url': 'https://isin.twse.com.tw/isin/class_main.jsp?market=2&issuetype=4&Page=1&chklike=Y', 'suffix': '.TWO'},
        {'name': '興櫃', 'url': 'https://isin.twse.com.tw/isin/class_main.jsp?owncode=&stockname=&isincode=&market=E&issuetype=R&industry_code=&Page=1&chklike=Y', 'suffix': '.TWO'},
        {'name': 'ETF', 'url': 'https://isin.twse.com.tw/isin/class_main.jsp?owncode=&stockname=&isincode=&market=1&issuetype=I&industry_code=&Page=1&chklike=Y', 'suffix': '.TW'},
        {'name': '臺灣創新板', 'url': 'https://isin.twse.com.tw/isin/class_main.jsp?market=C&issuetype=C&Page=1&chklike=Y', 'suffix': '.TW'},
        {'name': '戰略新板', 'url': 'https://isin.twse.com.tw/isin/class_main.jsp?market=A&issuetype=C&Page=1&chklike=Y', 'suffix': '.TWO'}
    ]
    
    log(f"📡 正在獲取全市場清單 (含創新板/戰略新板)...")
    conn = sqlite3.connect(DB_PATH)
    stock_list = []
    
    for cfg in url_configs:
        try:
            # 增加 User-Agent 避免被擋
            headers = {'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36'}
            resp = requests.get(cfg['url'], headers=headers, timeout=15)
            dfs = pd.read_html(StringIO(resp.text), header=0)
            
            if not dfs:
                log(f"⚠️ {cfg['name']} 未抓取到表格資料")
                continue
                
            df = dfs[0]
            count = 0
            
            for _, row in df.iterrows():
                # 欄位名稱校正：ISIN 網頁有時代號欄位會有空格或名稱差異
                code = str(row.get('有價證券代號', '')).strip()
                name = str(row.get('有價證券名稱', '')).strip()
                industry = str(row.get('產業別', '')).strip()
                
                # 剔除權證（通常代號長度非 4-5 碼，或依據 issuetype 過濾）
                if code.isalnum() and 4 <= len(code) <= 6:
                    symbol = f"{code}{cfg['suffix']}"
                    
                    # 存入資料庫，這裡 market 欄位會標註 '臺灣創新板' 或 '戰略新板'
                    conn.execute("""
                        INSERT OR REPLACE INTO stock_info (symbol, name, sector, market, updated_at) 
                        VALUES (?, ?, ?, ?, ?)
                    """, (symbol, name, industry, cfg['name'], datetime.now().strftime("%Y-%m-%d")))
                    
                    stock_list.append((symbol, name))
                    count += 1
            log(f"✅ {cfg['name']} 載入完成，共 {count} 檔")
            
        except Exception as e:
            log(f"❌ {cfg['name']} 抓取失敗: {str(e)}")
            
    conn.commit()
    conn.close()
    
    final_list = list(set(stock_list))
    log(f"📊 全市場掃描完畢，總計 {len(final_list)} 檔有效標的")
    return final_list

if __name__ == "__main__":
    get_tw_stock_list()
