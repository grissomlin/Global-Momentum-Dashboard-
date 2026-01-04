# -*- coding: utf-8 -*-
import sqlite3
import pandas as pd
import numpy as np
from datetime import datetime
import warnings

# 忽略特定警告
warnings.filterwarnings('ignore', category=FutureWarning)

def process_market_data(db_path):
    conn = sqlite3.connect(db_path)
    
    # 1. 讀取數據並關聯 stock_info 取得市場與產業別
    query = """
    SELECT p.*, i.market, i.sector, i.market_detail
    FROM stock_prices p
    LEFT JOIN stock_info i ON p.symbol = i.symbol
    """
    df = pd.read_sql(query, conn)
    df['date'] = pd.to_datetime(df['date'])
    df = df.sort_values(['symbol', 'date'])

    # 檢查是否有數據
    if df.empty:
        print("❌ 沒有找到股票數據")
        conn.close()
        return

    processed_list = []
    
    # 定義市場分類函數
    def is_unrestricted_market(market_detail):
        """判斷是否為無漲跌幅限制的市場"""
        if pd.isna(market_detail):
            return False
        unrestricted_markets = ['emerging', 'managed', 'strategic']  # 興櫃、管理股票、戰略新板
        return market_detail in unrestricted_markets
    
    # 2. 分組計算
    for symbol, group in df.groupby('symbol'):
        group = group.copy().sort_values('date')
        
        # 跳過數據太少的股票
        if len(group) < 40: 
            continue
        
        # 獲取市場信息
        market_detail = group['market_detail'].iloc[0] if not group['market_detail'].isna().all() else ''
        is_unrestricted = is_unrestricted_market(market_detail)
        
        # --- 🟢 第一步：資料清洗 ---
        group['daily_change'] = group['close'].pct_change()
        
        # 平滑異常值 (有漲跌幅限制的市場，>60%視為異常)
        if not is_unrestricted:
            group.loc[abs(group['daily_change']) > 0.6, 'close'] = np.nan
            group['close'] = group['close'].ffill()
        
        # 基礎欄位
        group['prev_close'] = group['close'].shift(1)
        group['avg_vol_20'] = group['volume'].rolling(window=20).mean()
        group['year'] = group['date'].dt.year
        
        # --- 🔴 第二步：漲停與長紅區間標記 (LU_Type & Brackets) ---
        # 漲幅百分比
        change_pct = group['daily_change'] * 100
        
        # 判定是否為「受限市場漲停」
        group['is_limit_up'] = 0
        if not is_unrestricted:
            # 上市櫃 10% 判定 (考慮四捨五入)
            limit_price = group['prev_close'] * 1.1
            limit_price = round(limit_price, 2)
            group['is_limit_up'] = (group['close'] >= limit_price * 0.999).astype(int)
        
        # --- 🟡 新增：詳細漲幅區間分類 ---
        def label_detailed_strength(row, is_unrestricted):
            """為無漲跌幅限制市場創建詳細區間分類"""
            chg = row['daily_change'] * 100
            
            if pd.isna(chg):
                return "NEGATIVE"
            
            if is_unrestricted:
                # 無漲跌幅限制市場：每10%一個區間，直到100%以上
                if chg >= 100:
                    return "RANK_100UP"
                elif chg >= 90:
                    return "RANK_90_100"
                elif chg >= 80:
                    return "RANK_80_90"
                elif chg >= 70:
                    return "RANK_70_80"
                elif chg >= 60:
                    return "RANK_60_70"
                elif chg >= 50:
                    return "RANK_50_60"
                elif chg >= 40:
                    return "RANK_40_50"
                elif chg >= 30:
                    return "RANK_30_40"
                elif chg >= 20:
                    return "RANK_20_30"
                elif chg >= 10:
                    return "RANK_10_20"
                elif chg > 0:
                    return "POSITIVE"
            else:
                # 有漲跌幅限制市場：簡化分類
                if chg >= 10:
                    return "RANK_10UP"
                elif chg > 0:
                    return "POSITIVE"
            
            return "NEGATIVE"
        
        # 應用詳細分類
        group['strength_rank'] = group.apply(
            lambda row: label_detailed_strength(row, is_unrestricted), 
            axis=1
        )
        
        # --- 🟠 新增：漲幅區間數值標記（用於統計）---
        def get_strength_value(row, is_unrestricted):
            """返回漲幅區間的數值表示"""
            chg = row['daily_change'] * 100
            
            if pd.isna(chg) or chg <= 0:
                return 0
            
            if is_unrestricted:
                # 無漲跌幅限制：返回區間下限
                if chg >= 100:
                    return 100
                elif chg >= 90:
                    return 90
                elif chg >= 80:
                    return 80
                elif chg >= 70:
                    return 70
                elif chg >= 60:
                    return 60
                elif chg >= 50:
                    return 50
                elif chg >= 40:
                    return 40
                elif chg >= 30:
                    return 30
                elif chg >= 20:
                    return 20
                elif chg >= 10:
                    return 10
                else:
                    return 1
            else:
                # 有漲跌幅限制：簡單分類
                if chg >= 10:
                    return 10
                else:
                    return 1
        
        group['strength_value'] = group.apply(
            lambda row: get_strength_value(row, is_unrestricted), 
            axis=1
        )
        
        # --- 🟤 新增：興櫃股票統計特徵 ---
        if is_unrestricted:
            # 計算每個區間的出現次數
            strength_counts = group['strength_rank'].value_counts().to_dict()
            
            # 創建特徵：過去20天內各強度區間的出現次數
            for rank in ['RANK_10_20', 'RANK_20_30', 'RANK_30_40', 'RANK_40_50', 
                        'RANK_50_60', 'RANK_60_70', 'RANK_70_80', 'RANK_80_90', 
                        'RANK_90_100', 'RANK_100UP']:
                col_name = f'count_{rank.lower()}'
                group[col_name] = (group['strength_rank'] == rank).rolling(window=20, min_periods=1).sum()
        
        # 漲停類型 (LU_Type4) - 僅限有漲跌幅限制的市場
        group['lu_type'] = None
        if not is_unrestricted:
            conditions = [
                (group['is_limit_up'] == 1) & (group['open'] == group['close']) & (group['high'] == group['low']),
                (group['is_limit_up'] == 1) & (group['open'] > group['prev_close'] * 1.05),
                (group['is_limit_up'] == 1) & (group['volume'] > group['avg_vol_20'] * 2),
                (group['is_limit_up'] == 1)
            ]
            choices = ['NO_VOLUME_LOCK', 'GAP_UP', 'HIGH_VOLUME_LOCK', 'FLOATING']
            group['lu_type'] = np.select(conditions, choices, default=None)

        # 連板次數
        if not is_unrestricted:
            streak = group['is_limit_up'].groupby((group['is_limit_up'] != group['is_limit_up'].shift()).cumsum()).cumsum()
            group['consecutive_limits'] = np.where(group['is_limit_up'] == 1, streak, 0)
        else:
            group['consecutive_limits'] = 0

        # --- 🟣 第三步：年度巔峰貢獻度 (以最高價 Peak High 計算) ---
        def calc_peak_contribution(df_year):
            if df_year.empty:
                df_year['peak_date'] = None
                df_year['peak_high_ret'] = np.nan
                df_year['strong_day_contribution'] = np.nan
                return df_year
            
            # 確保有有效的高價數據
            valid_high = df_year['high'].dropna()
            if valid_high.empty:
                df_year['peak_date'] = None
                df_year['peak_high_ret'] = np.nan
                df_year['strong_day_contribution'] = np.nan
                return df_year
            
            # 找到最高價的索引
            peak_idx = valid_high.idxmax()
            if pd.isna(peak_idx):
                df_year['peak_date'] = None
                df_year['peak_high_ret'] = np.nan
                df_year['strong_day_contribution'] = np.nan
                return df_year
            
            # 獲取峰值日期和價格
            peak_date = df_year.loc[peak_idx, 'date'] if peak_idx in df_year.index else None
            peak_price = df_year.loc[peak_idx, 'high'] if peak_idx in df_year.index else np.nan
            
            # 獲取年度開盤價
            if not df_year.empty:
                year_open = df_year.iloc[0]['open']
            else:
                year_open = np.nan
            
            # 計算年度總巔峰報酬 (對數)
            if pd.notna(peak_price) and pd.notna(year_open) and year_open > 0:
                total_peak_log = np.log(peak_price / year_open)
            else:
                total_peak_log = 0
            
            mask_before = df_year['date'] <= peak_date if peak_date else pd.Series(False, index=df_year.index)
            
            # 計算所有「漲幅 > 10%」日子的總貢獻
            daily_logs = np.log(df_year['close'] / df_year['prev_close'])
            
            # 調整閾值：有漲跌幅限制用10%，無限制用20%
            threshold = 0.10 if not is_unrestricted else 0.20
            strong_day_mask = (df_year['daily_change'] >= threshold) & mask_before
            
            if strong_day_mask.any() and total_peak_log > 0:
                strong_contribution = daily_logs[strong_day_mask].sum()
                strong_day_contribution = (strong_contribution / total_peak_log * 100)
            else:
                strong_day_contribution = 0
            
            df_year['peak_date'] = peak_date
            df_year['peak_high_ret'] = ((peak_price - year_open) / year_open * 100) if pd.notna(peak_price) and pd.notna(year_open) and year_open > 0 else np.nan
            df_year['strong_day_contribution'] = strong_day_contribution
            
            return df_year

        # 保存 year 欄位，然後進行分組計算
        year_values = group['year'].copy()
        
        # 進行分組計算
        try:
            group = group.groupby('year', group_keys=False).apply(calc_peak_contribution, include_groups=False)
        except TypeError:
            group = group.groupby('year', group_keys=False).apply(calc_peak_contribution)
        
        # 確保 year 欄位存在
        if 'year' not in group.columns:
            group['year'] = year_values

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
        
        # 波動率指標
        group['volatility_20'] = group['daily_change'].rolling(window=20).std() * np.sqrt(252)
        
        # RSI
        delta = group['close'].diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
        rs = gain / loss
        group['rsi'] = 100 - (100 / (1 + rs))
        
        # 成交量相關指標
        group['volume_ratio'] = group['volume'] / group['avg_vol_20']
        
        # 價格位置指標
        rolling_20_high = group['high'].rolling(window=20).max()
        rolling_20_low = group['low'].rolling(window=20).min()
        group['price_position_20'] = (group['close'] - rolling_20_low) / (rolling_20_high - rolling_20_low)
        
        # YTD Ret (實測收盤)
        year_start_prices = group.groupby('year')['close'].first()
        year_to_start_price = year_start_prices.to_dict()
        group['year_start_price'] = group['year'].map(year_to_start_price)
        group['ytd_ret'] = ((group['close'] - group['year_start_price']) / group['year_start_price'] * 100).round(2)

        processed_list.append(group)
    
    # 3. 檢查是否有處理後的數據
    if not processed_list:
        print("❌ 沒有處理後的數據")
        conn.close()
        return
    
    # 4. 寫回並優化
    df_final = pd.concat(processed_list)
    
    # 刪除舊表（如果存在）
    conn.execute("DROP TABLE IF EXISTS stock_analysis")
    
    # 創建新表
    df_final.to_sql('stock_analysis', conn, if_exists='replace', index=False)
    
    # 創建索引
    conn.execute("CREATE INDEX IF NOT EXISTS idx_symbol_date ON stock_analysis (symbol, date)")
    conn.execute("CREATE INDEX IF NOT EXISTS idx_strength_rank ON stock_analysis (strength_rank)")
    conn.execute("CREATE INDEX IF NOT EXISTS idx_market ON stock_analysis (market_detail)")
    
    # 5. 計算統計信息
    total_symbols = df_final['symbol'].nunique()
    date_range = f"{df_final['date'].min()} 到 {df_final['date'].max()}"
    
    # 統計興櫃股票的各區間漲幅數量
    emerging_stocks = df_final[df_final['market_detail'] == 'emerging']
    
    if not emerging_stocks.empty:
        print("\n📊 興櫃股票漲幅區間統計：")
        strength_distribution = emerging_stocks['strength_rank'].value_counts().sort_index()
        
        for rank, count in strength_distribution.items():
            if rank.startswith('RANK_'):
                # 提取區間
                if rank == 'RANK_100UP':
                    print(f"  {rank}: {count:,} 筆 (100%以上)")
                elif '_' in rank:
                    parts = rank.split('_')
                    if len(parts) >= 3:
                        lower = parts[1]
                        upper = parts[2] if parts[2] != 'UP' else '∞'
                        print(f"  {rank}: {count:,} 筆 ({lower}% ~ {upper}%)")
                else:
                    print(f"  {rank}: {count:,} 筆")
        
        # 計算平均每日漲幅大於10%的數量
        strong_days = (emerging_stocks['daily_change'] > 0.10).sum()
        total_days = len(emerging_stocks)
        strong_percentage = (strong_days / total_days * 100) if total_days > 0 else 0
        print(f"  📈 漲幅大於10%的天數: {strong_days:,} / {total_days:,} ({strong_percentage:.1f}%)")
    
    conn.close()
    
    print(f"""
✅ 特徵工程完成！
📊 處理統計：
   - 處理股票數量: {total_symbols}
   - 數據期間: {date_range}
   - 總數據行數: {len(df_final):,}
   - 新增特徵: 詳細漲幅區間、漲停標記、強度分級、年度巔峰貢獻度、技術指標等
   - 特別功能: 興櫃股票每10%漲幅區間統計
    """)

if __name__ == "__main__":
    process_market_data("tw_stock_warehouse.db")
