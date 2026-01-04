# -*- coding: utf-8 -*-
import sqlite3
import pandas as pd
import numpy as np

def process_market_data(db_path):
    conn = sqlite3.connect(db_path)
    # 1. 讀取數據 (建議 JOIN stock_info 取得市場類型以精確判斷漲停限制)
    query = """
    SELECT p.*, i.market 
    FROM stock_prices p
    LEFT JOIN stock_info i ON p.symbol = i.symbol
    """
    df = pd.read_sql(query, conn)
    df['date'] = pd.to_datetime(df['date'])
    df = df.sort_values(['symbol', 'date'])

    processed_list = []
    
    # 2. 分組計算指標
    for symbol, group in df.groupby('symbol'):
        group = group.copy().sort_values('date')
        
        # --- 🟢 第一步：資料清洗 (Data Cleaning) ---
        # 保留原有的異常值處理，避免錯誤價格干擾漲停判斷
        group['daily_change'] = group['close'].pct_change()
        group.loc[abs(group['daily_change']) > 0.6, 'close'] = np.nan
        group['close'] = group['close'].ffill() 
        
        if len(group) < 60: continue 

        # 基礎計算準備
        group['prev_close'] = group['close'].shift(1)
        group['avg_vol_20'] = group['volume'].rolling(window=20).mean()
        group['year'] = group['date'].dt.year

        # --- 🔴 第二步：漲停偵測與 LU_Type4 分類 ---
        # 台灣市場邏輯：10% 限制 (若為興櫃則不計算漲停)
        is_tw_limit = (group['market'] != '興櫃') & (group['market'] != 'ETF') # 簡易過濾
        group['is_limit_up'] = ((group['close'] >= (group['prev_close'] * 1.0945)) & is_tw_limit).astype(int)
        
        # 分類：一字板(NO_VOLUME_LOCK)、跳空鎖(GAP_UP)、爆量鎖(HIGH_VOLUME_LOCK)、爛板(FLOATING)
        conditions = [
            (group['is_limit_up'] == 1) & (group['open'] == group['close']) & (group['high'] == group['low']),
            (group['is_limit_up'] == 1) & (group['open'] > group['prev_close'] * 1.05),
            (group['is_limit_up'] == 1) & (group['volume'] > group['avg_vol_20'] * 2),
            (group['is_limit_up'] == 1)
        ]
        choices = ['NO_VOLUME_LOCK', 'GAP_UP', 'HIGH_VOLUME_LOCK', 'FLOATING']
        group['lu_type'] = np.select(conditions, choices, default=None)

        # 連板計數
        streak = group['is_limit_up'].groupby((group['is_limit_up'] != group['is_limit_up'].shift()).cumsum()).cumsum()
        group['consecutive_limits'] = np.where(group['is_limit_up'] == 1, streak, 0)

        # 隔日沖空間 (隔日開盤/最高漲幅)
        group['next_open_ret'] = ((group['open'].shift(-1) / group['close']) - 1) * 100
        group['next_high_ret'] = ((group['high'].shift(-1) / group['close']) - 1) * 100

        # --- 🟣 第三步：年度巔峰貢獻度計算 ---
        def calc_peak_metrics(df_year):
            if len(df_year) == 0: return df_year
            # 找到該年最高價日期 (第一次到達最高點)
            peak_idx = df_year['high'].idxmax()
            peak_date = df_year.loc[peak_idx, 'date']
            peak_high = df_year.loc[peak_idx, 'high']
            year_open = df_year.iloc[0]['open']
            
            # 總巔峰 Log 報酬
            total_peak_log = np.log(peak_high / year_open)
            
            # 最高點之前的數據
            mask_before = df_year['date'] <= peak_date
            # 計算最高點前「漲停日」的 Log 貢獻
            # Log 報酬具有相加性：ln(A/B) = ln(A) - ln(B)
            lu_logs = np.log(df_year['close'] / df_year['prev_close'])
            lu_contribution = lu_logs[(df_year['is_limit_up'] == 1) & mask_before].sum()
            
            df_year['peak_date'] = peak_date
            df_year['peak_high_ret'] = ((peak_high - year_open) / year_open * 100)
            df_year['lu_peak_contribution'] = (lu_contribution / total_peak_log * 100) if total_peak_log > 0 else 0
            return df_year

        group = group.groupby('year', group_keys=False).apply(calc_peak_metrics)

        # --- 🔵 第四步：原有技術指標 (MA, MACD, KD) ---
        group['ma20'] = group['close'].rolling(window=20).mean()
        group['ma60'] = group['close'].rolling(window=60).mean()
        group['ma20_slope'] = (group['ma20'].diff(3) / 3).round(4)
        
        ema12 = group['close'].ewm(span=12, adjust=False).mean()
        ema26 = group['close'].ewm(span=26, adjust=False).mean()
        group['macd'] = (ema12 - ema26)
        group['macds'] = group['macd'].ewm(span=9, adjust=False).mean()
        group['macdh'] = (group['macd'] - group['macds'])
        
        # 年度 YTD 報酬 (實測收盤)
        group['year_start_price'] = group.groupby('year')['close'].transform('first')
        group['ytd_ret'] = ((group['close'] - group['year_start_price']) / group['year_start_price'] * 100).round(2)

        # 未來報酬區間 (1-20日)
        windows = {'1-5': (1, 5), '6-10': (6, 10), '11-20': (11, 20)}
        for label, (s, e) in windows.items():
            f_high = group['high'].shift(-s).rolling(window=(e-s+1)).max()
            group[f'up_{label}'] = ((f_high / group['close'] - 1) * 100).round(2)
            f_low = group['low'].shift(-s).rolling(window=(e-s+1)).min()
            group[f'down_{label}'] = ((f_low / group['close'] - 1) * 100).round(2)

        processed_list.append(group)

    # 3. 寫回資料庫
    df_final = pd.concat(processed_list)
    df_final.to_sql('stock_analysis', conn, if_exists='replace', index=False)
    conn.execute("CREATE INDEX IF NOT EXISTS idx_analysis ON stock_analysis (symbol, date)")
    conn.close()
    print(f"✅ 特徵工程完成！包含：資料清洗、漲停分類、隔日沖與巔峰貢獻度分析。")

if __name__ == "__main__":
    process_market_data("tw_stock_warehouse.db")
