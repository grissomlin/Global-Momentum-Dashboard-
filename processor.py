# -*- coding: utf-8 -*-
import sqlite3
import pandas as pd
import numpy as np

def process_market_data(db_path):
    conn = sqlite3.connect(db_path)
    # 1. 讀取數據
    query = "SELECT * FROM stock_prices"
    df = pd.read_sql(query, conn)
    df['date'] = pd.to_datetime(df['date'])
    df = df.sort_values(['symbol', 'date'])

    processed_list = []
    
    # 2. 分組計算指標
    for symbol, group in df.groupby('symbol'):
        group = group.copy().sort_values('date')
        
        # --- 🟢 資料清洗 (Data Cleaning) ---
        # A. 計算單日漲跌幅，用來偵測異常值 (例如 8476 異常的 300% 漲幅)
        group['daily_change'] = group['close'].pct_change()
        
        # B. 剔除異常數據：如果單日漲幅或跌幅超過 50% 且成交量異常，
        # 在這裡我們可以選擇修正它或標記它。為了穩定性，我們將極端異常值平滑化
        # (這裡以超過 60% 為例，避免誤刪除權息後的真實波動)
        group.loc[abs(group['daily_change']) > 0.6, 'close'] = np.nan
        group['close'] = group['close'].ffill() # 用前一天價格填充異常值
        
        if len(group) < 60: continue 

        # --- A. 指標計算 (MA, MACD, KD) ---
        group['ma20'] = group['close'].rolling(window=20).mean()
        group['ma60'] = group['close'].rolling(window=60).mean()
        group['ma20_slope'] = (group['ma20'].diff(3) / 3).round(4) # 補上 round
        group['ma60_slope'] = (group['ma60'].diff(3) / 3).round(4)
        
        # --- 增加特徵斜率計算 ---
        group['ma60_slope'] = (group['ma60'].diff(3) / 3).round(4)
        ema12 = group['close'].ewm(span=12, adjust=False).mean()
        ema26 = group['close'].ewm(span=26, adjust=False).mean()
        group['macd'] = (ema12 - ema26)
        group['macds'] = group['macd'].ewm(span=9, adjust=False).mean()
        group['macdh'] = (group['macd'] - group['macds'])
        group['macdh_slope'] = (group['macdh'].diff(1)).round(4) # 柱狀體變化速度
        low_min = group['low'].rolling(window=9).min()
        high_max = group['high'].rolling(window=9).max()
        # 避免分母為 0
        denominator = high_max - low_min + 1e-9
        rsv = 100 * (group['close'] - low_min) / denominator
        group['k'] = rsv.ewm(com=2, adjust=False).mean()
        group['d'] = group['k'].ewm(com=2, adjust=False).mean()
        group['kd_gold'] = ((group['k'] > group['d']) & (group['k'].shift(1) <= group['d'].shift(1))).astype(int)

        # --- B. 底部背離 ---
        lookback = 10
        price_low_new = group['close'] < group['close'].shift(1).rolling(window=lookback).min()
        group['macd_bottom_div'] = ((price_low_new) & (group['macdh'] > group['macdh'].shift(1).rolling(window=lookback).min())).astype(int)
        group['kd_bottom_div'] = ((price_low_new) & (group['k'] > group['k'].shift(1).rolling(window=lookback).min())).astype(int)

        # --- 🔵 年度報酬對帳 (Annual Performance Logic) ---
        # 計算該日期相對於該年「第一筆交易日」的漲跌幅 (實測漲幅)
        group['year'] = group['date'].dt.year
        group['year_start_price'] = group.groupby('year')['close'].transform('first')
        group['ytd_ret'] = ((group['close'] - group['year_start_price']) / group['year_start_price'] * 100).round(2)

        # --- C. 未來報酬 (最大漲跌幅 % ) ---
        windows = {'1-5': (1, 5), '6-10': (6, 10), '11-20': (11, 20)}
        for label, (s, e) in windows.items():
            f_high = group['high'].shift(-s).rolling(window=(e-s+1)).max()
            group[f'up_{label}'] = ((f_high / group['close'] - 1) * 100).round(2)
            
            f_low = group['low'].shift(-s).rolling(window=(e-s+1)).min()
            group[f'down_{label}'] = ((f_low / group['close'] - 1) * 100).round(2)

        processed_list.append(group)

    # 3. 寫回資料庫
    df_final = pd.concat(processed_list)
    
    # 清除中間計算用的欄位以保持整潔
    cols_to_drop = ['daily_change', 'year_start_price']
    df_final = df_final.drop(columns=[c for c in cols_to_drop if c in df_final.columns])
    
    df_final.to_sql('stock_analysis', conn, if_exists='replace', index=False)
    conn.execute("CREATE INDEX IF NOT EXISTS idx_analysis ON stock_analysis (symbol, date)")
    conn.close()
    print(f"✅ {db_path} 特徵工程完成 (含資料清洗與 YTD 實測漲幅)")
