import streamlit as st
import os, json, sqlite3, io, urllib.parse
import pandas as pd
import numpy as np
import plotly.graph_objects as go 
from scipy.stats import skew, kurtosis
from google.oauth2 import service_account
from googleapiclient.discovery import build
from googleapiclient.http import MediaIoBaseDownload
from datetime import datetime

# --- 0. 頁面基本設定 ---
st.set_page_config(
    page_title="全球股市特徵引擎", 
    layout="wide",
    initial_sidebar_state="expanded"
)

# --- 1. 固定變數定義 ---
# 市場代碼映射
MARKET_MAP = {
    "台股 (TW)": "tw",
    "美股 (US)": "us", 
    "陸股 (CN)": "cn",
    "港股 (HK)": "hk",
    "日股 (JP)": "jp",
    "韓股 (KR)": "kr"
}

# --- 2. 輔助函數：獲取配置值 ---
def get_config_value(key, default=None):
    """獲取配置值，優先從環境變數，其次從 Streamlit Secrets"""
    # 先嘗試環境變數 (Render 部署用)
    env_value = os.environ.get(key)
    if env_value:
        return env_value
    
    # 再嘗試 Streamlit Secrets (本地開發用)
    try:
        if key in st.secrets:
            return st.secrets[key]
    except Exception:
        # 如果在 Render 上 st.secrets 不可用，會跳到這裡
        pass
    
    return default

# --- 3. Google Drive 服務初始化 ---
@st.cache_resource
def get_gdrive_service():
    """初始化 Google Drive 服務，同時支援環境變數和 Streamlit Secrets"""
    # 嘗試從環境變數或 Secrets 獲取服務帳戶資訊
    service_account_json = get_config_value("GDRIVE_SERVICE_ACCOUNT")
    
    if not service_account_json:
        st.error("❌ 找不到 GDRIVE_SERVICE_ACCOUNT 配置")
        st.info("請在 Render 環境變數或 Streamlit Secrets 中設定 GDRIVE_SERVICE_ACCOUNT")
        return None
    
    try:
        # 解析 JSON (無論來自環境變數或 Secrets)
        if isinstance(service_account_json, str):
            info = json.loads(service_account_json)
        else:
            info = service_account_json
        
        creds = service_account.Credentials.from_service_account_info(
            info, scopes=['https://www.googleapis.com/auth/drive.readonly']
        )
        return build('drive', 'v3', credentials=creds)
    except json.JSONDecodeError as e:
        st.error(f"❌ GDRIVE_SERVICE_ACCOUNT JSON 解析失敗: {e}")
        return None
    except Exception as e:
        st.error(f"❌ 服務初始化失敗: {e}")
        return None

def download_file(service, file_id, file_name):
    """下載檔案從 Google Drive"""
    request = service.files().get_media(fileId=file_id)
    fh = io.FileIO(file_name, 'wb')
    downloader = MediaIoBaseDownload(fh, request)
    done = False
    with st.spinner(f'🚀 正在同步 {file_name}...'):
        while done is False:
            _, done = downloader.next_chunk()
    return True

def get_database_stats(db_path, market_code):
    """獲取資料庫統計資訊"""
    stats = {
        "股票數量": 0,
        "數據天數": 0,
        "最早日期": None,
        "最晚日期": None,
        "分析表筆數": 0,
        "漲停天數": 0,
        "興櫃強勢天數": 0
    }
    
    if os.path.exists(db_path):
        try:
            conn = sqlite3.connect(db_path)
            
            # 股票數量
            query = "SELECT COUNT(DISTINCT symbol) FROM stock_prices"
            result = conn.execute(query).fetchone()
            stats["股票數量"] = result[0] if result else 0
            
            # 數據天數範圍
            query = "SELECT MIN(date), MAX(date), COUNT(*) FROM stock_prices"
            result = conn.execute(query).fetchone()
            if result:
                stats["最早日期"] = result[0]
                stats["最晚日期"] = result[1]
                stats["數據天數"] = result[2]
            
            # 分析表統計
            query = "SELECT COUNT(*) FROM stock_analysis"
            result = conn.execute(query).fetchone()
            stats["分析表筆數"] = result[0] if result else 0
            
            # 漲停統計 (僅限非興櫃)
            query = """
            SELECT COUNT(*) FROM stock_analysis 
            WHERE is_limit_up = 1 AND market_detail != 'emerging'
            """
            result = conn.execute(query).fetchone()
            stats["漲停天數"] = result[0] if result else 0
            
            # 興櫃強勢天數 (漲幅大於10%)
            query = """
            SELECT COUNT(*) FROM stock_analysis 
            WHERE strength_value >= 10 AND market_detail = 'emerging'
            """
            result = conn.execute(query).fetchone()
            stats["興櫃強勢天數"] = result[0] if result else 0
            
            conn.close()
            
        except Exception as e:
            st.error(f"統計資料庫時發生錯誤: {e}")
    
    return stats

# --- 4. 側邊欄配置 ---
st.sidebar.title("🌐 導航選單")

# 頁面選擇
page_options = {
    "🏠 首頁 - 策略篩選": "home",
    "📊 週K分析": "weekly",
    "📈 月K分析": "monthly", 
    "🎯 漲停板分析": "limit_up",
    "📉 年度貢獻度分析": "annual_contribution",
    "🔍 除錯工具": "debug"
}

selected_page = st.sidebar.radio("選擇頁面", list(page_options.keys()))

# 市場選擇 (大部分頁面都需要)
st.sidebar.header("📊 市場選擇")
selected_market_label = st.sidebar.selectbox("選擇市場", list(MARKET_MAP.keys()))
market_code = MARKET_MAP[selected_market_label]
TARGET_DB = f"{market_code}_stock_warehouse.db"

# 下載資料庫
service = get_gdrive_service()
db_stats = None

if service and st.sidebar.button("🔄 同步資料庫", type="secondary"):
    with st.spinner("正在從雲端同步資料庫..."):
        # 獲取資料夾 ID (從環境變數或 Secrets)
        folder_id = get_config_value("GDRIVE_FOLDER_ID", "")
        
        if folder_id:
            query = f"'{folder_id}' in parents and name = '{TARGET_DB}' and trashed = false"
            results = service.files().list(q=query, fields="files(id, name)").execute()
            files = results.get('files', [])
            if files: 
                download_file(service, files[0]['id'], TARGET_DB)
                st.sidebar.success("✅ 同步完成")
                st.rerun()  # 重新整理頁面以顯示最新數據
        else:
            st.sidebar.warning("⚠️ 未設定 GDRIVE_FOLDER_ID")

# 顯示資料庫統計 (如果存在)
if os.path.exists(TARGET_DB):
    db_stats = get_database_stats(TARGET_DB, market_code)
    st.sidebar.divider()
    st.sidebar.subheader("📊 資料庫統計")
    
    col1, col2 = st.sidebar.columns(2)
    with col1:
        st.metric("股票數量", f"{db_stats['股票數量']:,}")
        st.metric("總天數", f"{db_stats['數據天數']:,}")
    with col2:
        st.metric("分析筆數", f"{db_stats['分析表筆數']:,}")
        if db_stats['漲停天數'] > 0:
            st.metric("漲停天數", f"{db_stats['漲停天數']:,}")

# --- 5. 主頁面邏輯 ---
def render_home_page():
    """首頁 - 策略篩選"""
    st.title("🏠 策略篩選中心")
    
    if db_stats:
        # 顯示統計卡片
        st.subheader("📈 市場數據總覽")
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            st.metric("股票總數", f"{db_stats['股票數量']:,}", "支股票")
        with col2:
            st.metric("數據期間", 
                     f"{db_stats['最早日期']} 至 {db_stats['最晚日期']}", 
                     f"{db_stats['數據天數']:,} 天")
        with col3:
            st.metric("分析記錄", f"{db_stats['分析表筆數']:,}", "筆技術分析")
        with col4:
            if db_stats['漲停天數'] > 0:
                st.metric("漲停天數", f"{db_stats['漲停天數']:,}", "天")
            else:
                st.metric("興櫃強勢", f"{db_stats['興櫃強勢天數']:,}", "天")
    
    st.divider()
    
    # 策略篩選條件
    st.sidebar.header("🎯 策略篩選條件")
    
    year = st.sidebar.selectbox("選擇年份", [2024, 2025], index=1)
    month = st.sidebar.selectbox("選擇月份", list(range(1, 13)), index=0)
    
    # 技術指標策略
    strategy_type = st.sidebar.selectbox(
        "1. 技術指標策略", 
        ["無", "KD 黃金交叉", "MACD 柱狀圖轉正", "均線多頭排列(MA20>MA60)"]
    )
    
    # 背離選單
    divergence_type = st.sidebar.selectbox(
        "2. 疊加背離條件",
        ["不限", "MACD 底部背離", "KD 底部背離", "雙重背離 (MACD+KD)"]
    )
    
    # 評估期間
    period_options = {
        "1-5 天 (極短線展望)": "1-5",
        "6-10 天 (波段啟動期)": "6-10",
        "11-20 天 (中期趨勢驗證)": "11-20"
    }
    selected_period_label = st.sidebar.selectbox(
        "3. 評估未來報酬區間", 
        list(period_options.keys())
    )
    reward_period = period_options[selected_period_label]
    
    # 執行策略篩選
    if os.path.exists(TARGET_DB):
        try:
            conn = sqlite3.connect(TARGET_DB)
            start_date = f"{year}-{month:02d}-01"
            end_date = f"{year}-{month:02d}-31"
            query = f"SELECT * FROM stock_analysis WHERE date BETWEEN '{start_date}' AND '{end_date}'"
            df = pd.read_sql(query, conn)
            conn.close()
            
            if not df.empty:
                # 顯示篩選結果
                st.subheader(f"🎯 {year}年{month}月 符合訊號標的")
                
                # 基本篩選邏輯
                if strategy_type != "無":
                    if strategy_type == "KD 黃金交叉":
                        df = df[df['kd_golden_cross'] == 1]
                    elif strategy_type == "MACD 柱狀圖轉正":
                        df = df[df['macd_histogram_turn_positive'] == 1]
                    elif strategy_type == "均線多頭排列(MA20>MA60)":
                        df = df[df['ma20_ma60_cross'] == 1]
                
                # 背離條件篩選
                if divergence_type != "不限":
                    if divergence_type == "MACD 底部背離":
                        df = df[df['macd_divergence'] == 1]
                    elif divergence_type == "KD 底部背離":
                        df = df[df['kd_divergence'] == 1]
                    elif divergence_type == "雙重背離 (MACD+KD)":
                        df = df[(df['macd_divergence'] == 1) & (df['kd_divergence'] == 1)]
                
                # 顯示結果
                if not df.empty:
                    st.success(f"✅ 找到 {len(df)} 個符合條件的股票")
                    st.dataframe(df.head(50), use_container_width=True)
                    
                    # 顯示統計圖表
                    if 'ytd_ret' in df.columns:
                        st.subheader("📊 今年以來報酬分布")
                        fig = go.Figure(data=[go.Histogram(x=df['ytd_ret'], nbinsx=30)])
                        fig.update_layout(title="YTD 報酬分布", xaxis_title="報酬率(%)", yaxis_title="股票數量")
                        st.plotly_chart(fig, use_container_width=True)
                else:
                    st.warning("⚠️ 沒有找到符合條件的股票")
                    
            else:
                st.info("📭 該時段內無資料，請更換年份或月份。")
                
        except Exception as e:
            st.error(f"❌ 數據讀取失敗: {e}")
    else:
        st.warning("⚠️ 請先點擊側邊欄的『同步資料庫』按鈕下載數據")

def render_weekly_analysis():
    """週K分析頁面"""
    st.title("📊 週K分析")
    
    if not os.path.exists(TARGET_DB):
        st.warning("⚠️ 請先同步資料庫")
        return
    
    st.info("🏗️ 週K分析功能開發中...")
    
    # 週K統計卡片
    if db_stats:
        col1, col2, col3 = st.columns(3)
        with col1:
            st.metric("週K總數", "計算中...", "週")
        with col2:
            st.metric("週均漲幅", "計算中...", "%")
        with col3:
            st.metric("週漲停數", "計算中...", "次")
    
    # 週K圖表示例
    st.subheader("週K走勢圖")
    st.line_chart(pd.DataFrame({
        '週數': list(range(1, 21)),
        '平均漲幅': np.random.randn(20).cumsum()
    }).set_index('週數'))

def render_monthly_analysis():
    """月K分析頁面"""
    st.title("📈 月K分析")
    
    if not os.path.exists(TARGET_DB):
        st.warning("⚠️ 請先同步資料庫")
        return
    
    st.info("🏗️ 月K分析功能開發中...")
    
    # 月份選擇
    months = st.multiselect("選擇月份範圍", 
                          ["1月", "2月", "3月", "4月", "5月", "6月", 
                           "7月", "8月", "9月", "10月", "11月", "12月"],
                          default=["1月", "6月", "12月"])
    
    # 月K統計
    st.subheader("月K統計")
    monthly_data = pd.DataFrame({
        '月份': months,
        '平均漲幅': np.random.randn(len(months)) * 5 + 2,
        '漲停次數': np.random.randint(5, 20, len(months))
    })
    st.dataframe(monthly_data, use_container_width=True)

def render_limit_up_analysis():
    """漲停板分析頁面"""
    st.title("🎯 漲停板分析")
    
    if not os.path.exists(TARGET_DB):
        st.warning("⚠️ 請先同步資料庫")
        return
    
    try:
        conn = sqlite3.connect(TARGET_DB)
        
        # 漲停統計
        query = """
        SELECT 
            date,
            COUNT(*) as total_stocks,
            SUM(CASE WHEN is_limit_up = 1 THEN 1 ELSE 0 END) as limit_up_count,
            SUM(CASE WHEN is_limit_up = 1 THEN 1 ELSE 0 END) * 100.0 / COUNT(*) as limit_up_percentage
        FROM stock_analysis 
        WHERE market_detail != 'emerging'
        GROUP BY date
        ORDER BY date DESC
        LIMIT 100
        """
        
        limit_up_df = pd.read_sql(query, conn)
        
        if not limit_up_df.empty:
            # 統計卡片
            st.subheader("📈 漲停板統計總覽")
            
            col1, col2, col3, col4 = st.columns(4)
            total_days = len(limit_up_df)
            avg_limit_up = limit_up_df['limit_up_count'].mean()
            max_limit_up = limit_up_df['limit_up_count'].max()
            avg_percentage = limit_up_df['limit_up_percentage'].mean()
            
            with col1:
                st.metric("分析天數", f"{total_days:,}", "天")
            with col2:
                st.metric("日均漲停", f"{avg_limit_up:.1f}", "支")
            with col3:
                st.metric("單日最高", f"{max_limit_up:,}", "支")
            with col4:
                st.metric("漲停比率", f"{avg_percentage:.2f}", "%")
            
            # 漲停趨勢圖
            st.subheader("📊 每日漲停家數趨勢")
            fig = go.Figure()
            fig.add_trace(go.Scatter(
                x=limit_up_df['date'], 
                y=limit_up_df['limit_up_count'],
                mode='lines+markers',
                name='漲停家數',
                line=dict(color='red', width=2)
            ))
            fig.update_layout(
                title="漲停家數趨勢圖",
                xaxis_title="日期",
                yaxis_title="漲停家數",
                hovermode='x unified'
            )
            st.plotly_chart(fig, use_container_width=True)
            
            # 漲停類型分析
            st.subheader("🔍 漲停類型分析 (4U分析)")
            query_4u = """
            SELECT 
                lu_type,
                COUNT(*) as count,
                AVG(CASE WHEN strength_rank LIKE 'RANK_%' THEN 1 ELSE 0 END) as strong_ratio,
                AVG(volume_ratio) as avg_volume_ratio,
                AVG(daily_change) as avg_daily_change
            FROM stock_analysis 
            WHERE is_limit_up = 1 AND lu_type IS NOT NULL
            GROUP BY lu_type
            ORDER BY count DESC
            """
            
            type_df = pd.read_sql(query_4u, conn)
            if not type_df.empty:
                st.dataframe(type_df, use_container_width=True)
                
                # 4U分布圖
                fig2 = go.Figure(data=[go.Pie(
                    labels=type_df['lu_type'],
                    values=type_df['count'],
                    hole=.3
                )])
                fig2.update_layout(title="漲停類型分布 (4U分析)")
                st.plotly_chart(fig2, use_container_width=True)
        
        # 連板統計
        st.subheader("🏆 連板天數統計")
        query_streak = """
        SELECT 
            consecutive_limits,
            COUNT(*) as stock_count,
            AVG(daily_change) as avg_next_day_change,
            AVG(ytd_ret) as avg_ytd_ret
        FROM stock_analysis 
        WHERE consecutive_limits > 0
        GROUP BY consecutive_limits
        ORDER BY consecutive_limits
        """
        
        streak_df = pd.read_sql(query_streak, conn)
        if not streak_df.empty:
            st.dataframe(streak_df, use_container_width=True)
        
        conn.close()
        
    except Exception as e:
        st.error(f"❌ 分析失敗: {e}")

def render_annual_contribution():
    """年度貢獻度分析"""
    st.title("📉 年度貢獻度分析")
    
    if not os.path.exists(TARGET_DB):
        st.warning("⚠️ 請先同步資料庫")
        return
    
    try:
        conn = sqlite3.connect(TARGET_DB)
        
        # 年度貢獻度統計
        query = """
        SELECT 
            year,
            symbol,
            AVG(peak_high_ret) as avg_peak_return,
            AVG(strong_day_contribution) as avg_strong_contribution,
            SUM(CASE WHEN is_limit_up = 1 THEN 1 ELSE 0 END) as limit_up_days,
            SUM(CASE WHEN daily_change > 0.095 THEN 1 ELSE 0 END) as strong_up_days
        FROM stock_analysis 
        WHERE year IS NOT NULL
        GROUP BY year, symbol
        HAVING avg_peak_return IS NOT NULL
        ORDER BY year, avg_peak_return DESC
        """
        
        annual_df = pd.read_sql(query, conn)
        
        if not annual_df.empty:
            # 年度選擇
            years = sorted(annual_df['year'].unique())
            selected_year = st.selectbox("選擇年度", years, index=len(years)-1)
            
            # 年度統計
            year_data = annual_df[annual_df['year'] == selected_year]
            
            st.subheader(f"📊 {selected_year} 年度貢獻度分析")
            
            # 頂級貢獻股票
            st.subheader("🏆 年度貢獻度排行榜")
            top_contributors = year_data.nlargest(10, 'avg_peak_return')
            st.dataframe(top_contributors, use_container_width=True)
            
            # 散點圖：貢獻度 vs 漲停天數
            st.subheader("📈 貢獻度與漲停天數關係")
            fig = go.Figure(data=go.Scatter(
                x=year_data['limit_up_days'],
                y=year_data['avg_peak_return'],
                mode='markers',
                marker=dict(
                    size=year_data['strong_up_days'],
                    color=year_data['avg_strong_contribution'],
                    colorscale='Viridis',
                    showscale=True,
                    colorbar=dict(title="強勢日貢獻度%")
                ),
                text=year_data['symbol'],
                hovertemplate='<b>%{text}</b><br>' +
                            '漲停天數: %{x}<br>' +
                            '年度巔峰: %{y:.1f}%<br>' +
                            '強勢日貢獻: %{marker.color:.1f}%<br>'
            ))
            fig.update_layout(
                title=f"{selected_year}年 貢獻度分析",
                xaxis_title="漲停天數",
                yaxis_title="年度巔峰報酬率 (%)"
            )
            st.plotly_chart(fig, use_container_width=True)
        
        conn.close()
        
    except Exception as e:
        st.error(f"❌ 分析失敗: {e}")

def render_debug_tools():
    """除錯工具頁面"""
    st.title("🔍 資料庫除錯工具")
    
    # 顯示環境資訊
    st.subheader("環境資訊")
    col1, col2 = st.columns(2)
    
    with col1:
        st.write("**作業系統**:", os.name)
        st.write("**當前目錄**:", os.getcwd())
        st.write("**Python 版本**:", os.sys.version)
    
    with col2:
        st.write("**資料庫檔案**:")
        for market in MARKET_MAP.values():
            db_file = f"{market}_stock_warehouse.db"
            if os.path.exists(db_file):
                st.success(f"✅ {db_file} - {os.path.getsize(db_file):,} bytes")
            else:
                st.error(f"❌ {db_file} - 不存在")
    
    # 檢查配置
    st.subheader("配置檢查")
    gdrive_sa = get_config_value("GDRIVE_SERVICE_ACCOUNT")
    if gdrive_sa:
        st.success("✅ GDRIVE_SERVICE_ACCOUNT 已設定")
        # 顯示部分資訊 (保護敏感資料)
        if isinstance(gdrive_sa, str):
            try:
                sa_info = json.loads(gdrive_sa)
                st.write("**服務帳戶**:", sa_info.get("client_email", "未知"))
            except:
                st.write("**服務帳戶**: JSON 格式正確")
    else:
        st.error("❌ GDRIVE_SERVICE_ACCOUNT 未設定")
    
    folder_id = get_config_value("GDRIVE_FOLDER_ID")
    if folder_id:
        st.success(f"✅ GDRIVE_FOLDER_ID 已設定: {folder_id}")
    else:
        st.warning("⚠️ GDRIVE_FOLDER_ID 未設定")
    
    # 資料庫檢查
    if os.path.exists(TARGET_DB):
        st.subheader("資料庫檢查")
        try:
            conn = sqlite3.connect(TARGET_DB)
            cursor = conn.cursor()
            
            # 檢查表格
            cursor.execute("SELECT name FROM sqlite_master WHERE type='table';")
            tables = cursor.fetchall()
            
            st.write("**資料庫表格**:")
            for table in tables:
                cursor.execute(f"SELECT COUNT(*) FROM {table[0]};")
                count = cursor.fetchone()[0]
                st.write(f"- {table[0]}: {count:,} 筆記錄")
            
            conn.close()
        except Exception as e:
            st.error(f"資料庫檢查失敗: {e}")

# --- 6. 頁面路由 ---
page_mapping = {
    "home": render_home_page,
    "weekly": render_weekly_analysis,
    "monthly": render_monthly_analysis,
    "limit_up": render_limit_up_analysis,
    "annual_contribution": render_annual_contribution,
    "debug": render_debug_tools
}

# 根據選擇渲染頁面
selected_page_id = page_options[selected_page]
page_mapping[selected_page_id]()
