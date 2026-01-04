# -*- coding: utf-8 -*-
import sqlite3
import pandas as pd
import numpy as np

def process_market_data(db_path):
    conn = sqlite3.connect(db_path)
    
    # 1. 讀取數據並關聯 stock_info 取得市場與產業別
    # 確保你的 downloader 已經把 '興櫃', '上市', '上櫃' 存入 stock_info
    query = """
    SELECT p.*, i.market, i.sector
    FROM stock_prices p
    LEFT JOIN stock_info i ON p.symbol = i.symbol
    """
    df = pd.read_sql(query, conn)
    df['date'] = pd.to_datetime(df['date'])
    df = df.sort_values(['symbol', 'date'])

    processed_list = []
    
    # 2. 分組計算
    for symbol, group in df.groupby('symbol'):
        group = group.copy().sort_values('date')
        
        # --- 🟢 第一步：資料清洗 ---
        group['daily_change'] = group['close'].pct_change()
        # 平滑異常值 (>60% 且非興櫃則視為異常)
        is_emerging = group['market'].iloc[0] == '興櫃'
        if not is_emerging:
            group.loc[abs(group['daily_change']) > 0.6, 'close'] = np.nan
            group['close'] = group['close'].ffill()
        
        if len(group) < 40: continue 

        # 基礎欄位
        group['prev_close'] = group['close'].shift(1)
        group['avg_vol_20'] = group['volume'].rolling(window=20).mean()
        group['year'] = group['date'].dt.year
        
        # --- 🔴 第二步：漲停與長紅區間標記 (LU_Type & Brackets) ---
        # 漲幅百分比
        change_pct = group['daily_change'] * 100
        
        # 判定是否為「受限市場漲停」
        group['is_limit_up'] = 0
        if not is_emerging:
            # 上市櫃 10% 判定
            group['is_limit_up'] = (group['close'] >= (group['prev_close'] * 1.0945)).astype(int)
        
        # 針對「無限制」或「長紅棒」定義區間 (10% - 100%+)
        def label_strength(row):
            chg = row['daily_change'] * 100
            if chg >= 100: return "RANK_100UP"
            elif chg >= 50: return "RANK_50_100"
            elif chg >= 30: return "RANK_30_50"
            elif chg >= 20: return "RANK_20_30"
            elif chg >= 10: return "RANK_10_20"
            elif chg > 0:   return "POSITIVE"
            return "NEGATIVE"
        
        group['strength_rank'] = group.apply(label_strength, axis=1)

        # 漲停類型 (LU_Type4)
        conditions = [
            (group['is_limit_up'] == 1) & (group['open'] == group['close']) & (group['high'] == group['low']),
            (group['is_limit_up'] == 1) & (group['open'] > group['prev_close'] * 1.05),
            (group['is_limit_up'] == 1) & (group['volume'] > group['avg_vol_20'] * 2),
            (group['is_limit_up'] == 1)
        ]
        choices = ['NO_VOLUME_LOCK', 'GAP_UP', 'HIGH_VOLUME_LOCK', 'FLOATING']
        group['lu_type'] = np.select(conditions, choices, default=None)

        # 連板次數
        streak = group['is_limit_up'].groupby((group['is_limit_up'] != group['is_limit_up'].shift()).cumsum()).cumsum()
        group['consecutive_limits'] = np.where(group['is_limit_up'] == 1, streak, 0)

        # --- 🟣 第三步：年度巔峰貢獻度 (以最高價 Peak High 計算) ---
        def calc_peak_contribution(df_year):
            if df_year.empty: return df_year
            peak_idx = df_year['high'].idxmax()
            peak_date = df_year.loc[peak_idx, 'date']
            peak_price = df_year.loc[peak_idx, 'high']
            year_open = df_year.iloc[0]['open']
            
            # 年度總巔峰報酬 (對數)
            total_peak_log = np.log(peak_price / year_open)
            mask_before = df_year['date'] <= peak_date
            
            # 計算所有「漲幅 > 10%」日子的總貢獻
            daily_logs = np.log(df_year['close'] / df_year['prev_close'])
            strong_day_mask = (df_year['daily_change'] >= 0.095) & mask_before
            strong_contribution = daily_logs[strong_day_mask].sum()
            
            df_year['peak_date'] = peak_date
            df_year['peak_high_ret'] = ((peak_price - year_open) / year_open * 100)
            df_year['strong_day_contribution'] = (strong_contribution / total_peak_log * 100) if total_peak_log > 0 else 0
            return df_year

        group = group.groupby('year', group_keys=False).apply(calc_peak_contribution)

        # --- 🔵 第四步：原有技術指標 ---
        # MA
        group['ma20'] = group['close'].rolling(window=20).mean()
        group['ma60'] = group['close'].rolling(window=60).mean()
        
        # MACD
        ema12 = group['close'].ewm(span=12, adjust=False).mean()
        ema26 = group['close'].ewm(span=26, adjust=False).mean()
        group['macd'] = ema12 - ema26
        group['macds'] = group['macd'].ewm(span=9, adjust=False).mean()
        group['macdh'] = group['macd'] - group['macds']
        
        # YTD Ret (實測收盤)
        group['year_start_price'] = group.groupby('year')['close'].transform('first')
        group['ytd_ret'] = ((group['close'] - group['year_start_price']) / group['year_start_price'] * 100).round(2)

        processed_list.append(group)

    # 3. 寫回並優化
    df_final = pd.concat(processed_list)
    df_final.to_sql('stock_analysis', conn, if_exists='replace', index=False)
    conn.execute("CREATE INDEX IF NOT EXISTS idx_symbol_date ON stock_analysis (symbol, date)")
    conn.close()
    print("✅ 特徵工程完整版完成：含市場分類、長紅強度區間、巔峰貢獻度。")

if __name__ == "__main__":
    process_market_data("tw_stock_warehouse.db")
