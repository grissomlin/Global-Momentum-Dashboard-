# -*- coding: utf-8 -*-
import sqlite3
import pandas as pd
import numpy as np
from datetime import datetime
import warnings

# 忽略特定警告
warnings.filterwarnings('ignore', category=FutureWarning)

class MarketConfig:
    """市場配置類別，統一管理不同市場的規則"""
    
    # 市場分類定義
    MARKET_RULES = {
        # 台灣市場
        'TW_LISTED': {  # 上市
            'limit_up_pct': 0.10,
            'threshold': 0.10,  # 強勢日閾值
            'strength_intervals': [(10, 'RANK_10UP')],  # 只用一個10%以上的區間
            'max_strength': 10
        },
        'TW_OTC': {  # 上櫃
            'limit_up_pct': 0.10,
            'threshold': 0.10,
            'strength_intervals': [(10, 'RANK_10UP')],
            'max_strength': 10
        },
        'TW_EMERGING': {  # 興櫃
            'limit_up_pct': None,  # 無漲跌幅限制
            'threshold': 0.20,
            'strength_intervals': [
                (10, 'RANK_10_20'), (20, 'RANK_20_30'), (30, 'RANK_30_40'),
                (40, 'RANK_40_50'), (50, 'RANK_50_60'), (60, 'RANK_60_70'),
                (70, 'RANK_70_80'), (80, 'RANK_80_90'), (90, 'RANK_90_100'),
                (100, 'RANK_100UP')
            ],
            'max_strength': 100
        },
        # 韓國市場
        'KR_KOSPI': {
            'limit_up_pct': 0.30,
            'threshold': 0.30,  # 韓國強勢日閾值用30%
            'strength_intervals': [
                (10, 'RANK_10_20'), (20, 'RANK_20_30'), (30, 'RANK_30UP')
            ],
            'max_strength': 30
        },
        'KR_KOSDAQ': {
            'limit_up_pct': 0.30,
            'threshold': 0.30,
            'strength_intervals': [
                (10, 'RANK_10_20'), (20, 'RANK_20_30'), (30, 'RANK_30UP')
            ],
            'max_strength': 30
        }
    }
    
    @classmethod
    def get_market_config(cls, market, market_detail):
        """根據市場和市場細分類獲取配置"""
        # 台灣市場判斷
        if market == 'TW':
            if market_detail == 'emerging':
                return cls.MARKET_RULES['TW_EMERGING']
            elif market_detail in ['listed', 'tse']:
                return cls.MARKET_RULES['TW_LISTED']
            elif market_detail in ['otc', 'gtsm']:
                return cls.MARKET_RULES['TW_OTC']
        
        # 韓國市場判斷（從下載器的資料判斷）
        elif market == 'KR' or 'KOSPI' in str(market) or 'KOSDAQ' in str(market):
            if 'KOSPI' in str(market_detail) or 'KOSPI' in str(market):
                return cls.MARKET_RULES['KR_KOSPI']
            elif 'KOSDAQ' in str(market_detail) or 'KOSDAQ' in str(market):
                return cls.MARKET_RULES['KR_KOSDAQ']
            else:
                return cls.MARKET_RULES['KR_KOSPI']  # 預設為KOSPI
        
        # 預設為台灣上市櫃規則（向下兼容）
        return cls.MARKET_RULES['TW_LISTED']

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
    
    # 2. 分組計算
    for symbol, group in df.groupby('symbol'):
        group = group.copy().sort_values('date')
        
        # 跳過數據太少的股票
        if len(group) < 40: 
            continue
        
        # 獲取市場信息
        market = group['market'].iloc[0] if not group['market'].isna().all() else ''
        market_detail = group['market_detail'].iloc[0] if not group['market_detail'].isna().all() else ''
        
        # 獲取市場配置
        config = MarketConfig.get_market_config(market, market_detail)
        limit_up_pct = config['limit_up_pct']
        is_unrestricted = (limit_up_pct is None)
        threshold = config['threshold']  # 強勢日閾值
        
        print(f"處理 {symbol}: 市場={market}, 細類={market_detail}, 漲停={limit_up_pct}, 強勢閾值={threshold}")
        
        # --- 🟢 第一步：資料清洗 ---
        group['daily_change'] = group['close'].pct_change()
        
        # 平滑異常值 (有漲跌幅限制的市場)
        if not is_unrestricted and limit_up_pct is not None:
            # 將超過漲停幅度的視為異常
            max_allowed_change = limit_up_pct * 1.5  # 允許一些誤差
            group.loc[abs(group['daily_change']) > max_allowed_change, 'close'] = np.nan
            group['close'] = group['close'].ffill()
        
        # 基礎欄位
        group['prev_close'] = group['close'].shift(1)
        group['avg_vol_20'] = group['volume'].rolling(window=20).mean()
        group['year'] = group['date'].dt.year
        
        # --- 🔴 第二步：漲停與長紅區間標記 ---
        # 漲幅百分比
        change_pct = group['daily_change'] * 100
        
        # 判定是否為「漲停」
        group['is_limit_up'] = 0
        if limit_up_pct is not None:
            # 根據市場配置的漲停幅度計算
            limit_price = group['prev_close'] * (1 + limit_up_pct)
            limit_price = round(limit_price, 2)
            group['is_limit_up'] = (group['close'] >= limit_price * 0.999).astype(int)
        
        # --- 🟡 新增：詳細漲幅區間分類（統一方法）---
        def label_detailed_strength(row):
            """為所有市場創建詳細區間分類"""
            chg = row['daily_change'] * 100
            
            if pd.isna(chg) or chg <= 0:
                return "NEGATIVE"
            
            # 使用市場配置的區間
            for min_val, rank_label in config['strength_intervals']:
                if chg >= min_val:
                    # 找到適合的區間
                    if rank_label == 'RANK_10UP' or rank_label == 'RANK_30UP':
                        # 這些是「以上」的區間
                        return rank_label
                    else:
                        # 檢查是否在區間內
                        next_min = next((m for m, _ in config['strength_intervals'] if m > min_val), None)
                        if next_min is None or chg < next_min:
                            return rank_label
            
            # 如果小於最小區間但是正值
            return "POSITIVE"
        
        # 應用詳細分類
        group['strength_rank'] = group.apply(
            lambda row: label_detailed_strength(row), 
            axis=1
        )
        
        # --- 🟠 新增：漲幅區間數值標記（用於統計）---
        def get_strength_value(row):
            """返回漲幅區間的數值表示"""
            chg = row['daily_change'] * 100
            
            if pd.isna(chg) or chg <= 0:
                return 0
            
            # 使用市場配置的區間
            for min_val, rank_label in config['strength_intervals']:
                if chg >= min_val:
                    # 如果是「以上」的區間，返回該值
                    if rank_label in ['RANK_10UP', 'RANK_30UP']:
                        return min_val
                    # 檢查是否在區間內
                    next_min = next((m for m, _ in config['strength_intervals'] if m > min_val), None)
                    if next_min is None or chg < next_min:
                        return min_val
            
            # 小於最小區間但是正值
            return 1
        
        group['strength_value'] = group.apply(
            lambda row: get_strength_value(row), 
            axis=1
        )
        
        # --- 🟤 新增：特殊市場統計特徵 ---
        if is_unrestricted or limit_up_pct == 0.30:
            # 對於無漲跌幅限制或韓國市場，計算各區間出現次數
            for min_val, rank_label in config['strength_intervals']:
                if rank_label not in ['RANK_10UP', 'RANK_30UP']:  # 排除「以上」的區間
                    col_name = f'count_{rank_label.lower()}'
                    group[col_name] = (group['strength_rank'] == rank_label).rolling(window=20, min_periods=1).sum()
        
        # 漲停類型 (LU_Type4) - 僅限有漲跌幅限制的市場
        group['lu_type'] = None
        if limit_up_pct is not None:
            conditions = [
                (group['is_limit_up'] == 1) & (group['open'] == group['close']) & (group['high'] == group['low']),
                (group['is_limit_up'] == 1) & (group['open'] > group['prev_close'] * (1 + limit_up_pct * 0.5)),
                (group['is_limit_up'] == 1) & (group['volume'] > group['avg_vol_20'] * 2),
                (group['is_limit_up'] == 1)
            ]
            choices = ['NO_VOLUME_LOCK', 'GAP_UP', 'HIGH_VOLUME_LOCK', 'FLOATING']
            group['lu_type'] = np.select(conditions, choices, default=None)

        # 連板次數
        if limit_up_pct is not None:
            streak = group['is_limit_up'].groupby((group['is_limit_up'] != group['is_limit_up'].shift()).cumsum()).cumsum()
            group['consecutive_limits'] = np.where(group['is_limit_up'] == 1, streak, 0)
        else:
            group['consecutive_limits'] = 0

        # --- 🟣 第三步：年度巔峰貢獻度 ---
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
            
            # 計算所有「漲幅 > threshold」日子的總貢獻
            daily_logs = np.log(df_year['close'] / df_year['prev_close'])
            
            # 使用市場配置的threshold
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

        # --- 🔵 第四步：技術指標 ---
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
    conn.execute("CREATE INDEX IF NOT EXISTS idx_market ON stock_analysis (market)")
    conn.execute("CREATE INDEX IF NOT EXISTS idx_market_detail ON stock_analysis (market_detail)")
    
    # 5. 計算統計信息
    total_symbols = df_final['symbol'].nunique()
    date_range = f"{df_final['date'].min()} 到 {df_final['date'].max()}"
    
    # 統計不同市場的漲幅分佈
    print("\n📊 全球市場漲幅區間統計：")
    
    for market in df_final['market'].unique():
        if pd.isna(market):
            continue
            
        market_data = df_final[df_final['market'] == market]
        if market_data.empty:
            continue
            
        print(f"\n🔹 市場: {market}")
        strength_distribution = market_data['strength_rank'].value_counts().sort_index()
        
        for rank, count in strength_distribution.items():
            if rank != "NEGATIVE" and rank != "POSITIVE":
                print(f"  {rank}: {count:,} 筆")
        
        # 計算各市場的強勢日比例
        if market in ['KR', 'KOSPI', 'KOSDAQ']:
            strong_threshold = 0.30  # 韓國30%
        elif market == 'TW':
            # 判斷是否為興櫃
            emerging_data = market_data[market_data['market_detail'] == 'emerging']
            if not emerging_data.empty:
                strong_threshold = 0.20  # 台灣興櫃20%
            else:
                strong_threshold = 0.10  # 台灣上市櫃10%
        else:
            strong_threshold = 0.10  # 預設10%
        
        strong_days = (market_data['daily_change'] > strong_threshold).sum()
        total_days = len(market_data)
        strong_percentage = (strong_days / total_days * 100) if total_days > 0 else 0
        print(f"  📈 漲幅大於{strong_threshold*100:.0f}%的天數: {strong_days:,} / {total_days:,} ({strong_percentage:.1f}%)")
        
        # 特別顯示韓國市場的統計
        if market in ['KR', 'KOSPI', 'KOSDAQ']:
            kr_10_20 = ((market_data['daily_change'] >= 0.10) & (market_data['daily_change'] < 0.20)).sum()
            kr_20_30 = ((market_data['daily_change'] >= 0.20) & (market_data['daily_change'] < 0.30)).sum()
            kr_30_up = (market_data['daily_change'] >= 0.30).sum()
            print(f"  🇰🇷 韓國專屬統計:")
            print(f"     10-20%: {kr_10_20:,} 筆")
            print(f"     20-30%: {kr_20_30:,} 筆")
            print(f"     30%以上: {kr_30_up:,} 筆")
    
    conn.close()
    
    print(f"""
✅ 全球市場特徵工程完成！
📊 處理統計：
   - 處理股票數量: {total_symbols}
   - 數據期間: {date_range}
   - 總數據行數: {len(df_final):,}
   - 新增特徵: 詳細漲幅區間、漲停標記、強度分級、年度巔峰貢獻度、技術指標等
   - 支援市場: 台灣上市/上櫃/興櫃、韓國KOSPI/KOSDAQ
   - 特別功能: 跨市場統一漲幅區間分析
    """)

if __name__ == "__main__":
    # 可以根據需要選擇處理哪個資料庫
    # process_market_data("tw_stock_warehouse.db")  # 台灣
    process_market_data("kr_stock_warehouse.db")  # 韓國
    # 未來可以擴展: process_market_data("us_stock_warehouse.db")  # 美國
