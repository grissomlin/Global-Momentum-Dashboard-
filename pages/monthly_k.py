# pages/monthly_k.py
# Streamlit page: Monthly K-line viewer (reads kbar_monthly from SQLite)

import os
import sqlite3
import pandas as pd
import streamlit as st

try:
    import plotly.graph_objects as go
    PLOTLY_OK = True
except Exception:
    PLOTLY_OK = False

st.set_page_config(page_title="Monthly K (月K)", layout="wide")
st.title("🗓️ 月K（kbar_monthly）檢視器")

def list_db_files():
    bases = [os.getcwd(), "/content", "/mount/src", os.path.expanduser("~")]
    out = []
    for b in bases:
        if os.path.isdir(b):
            for f in os.listdir(b):
                if f.endswith(".db"):
                    out.append(os.path.join(b, f))
    return sorted(set(out))

def connect(db_path):
    conn = sqlite3.connect(db_path, check_same_thread=False)
    conn.execute("PRAGMA journal_mode=WAL;")
    return conn

@st.cache_data(ttl=60)
def table_exists(db_path, table):
    conn = connect(db_path)
    try:
        cur = conn.execute(
            "SELECT 1 FROM sqlite_master WHERE type='table' AND name=?",
            (table,),
        )
        return cur.fetchone() is not None
    finally:
        conn.close()

@st.cache_data(ttl=60)
def load_meta(db_path):
    conn = connect(db_path)
    try:
        return pd.read_sql(
            """
            SELECT symbol, year, COUNT(*) AS n_months
            FROM kbar_monthly
            GROUP BY symbol, year
            ORDER BY year DESC, n_months DESC
            """,
            conn,
        )
    finally:
        conn.close()

@st.cache_data(ttl=60)
def load_monthly(db_path, symbol, year):
    conn = connect(db_path)
    try:
        df = pd.read_sql(
            """
            SELECT symbol, year, month_id, period_start, period_end,
                   open, high, low, close, volume
            FROM kbar_monthly
            WHERE symbol=? AND year=?
            ORDER BY period_start
            """,
            conn,
            params=(symbol, year),
        )
        if not df.empty:
            df["period_end"] = pd.to_datetime(df["period_end"])
            df["ret_pct"] = (df["close"] / df["open"] - 1) * 100
            df["cum_ret_pct"] = (df["close"] / df["open"].iloc[0] - 1) * 100
        return df
    finally:
        conn.close()

def plot_candle(df):
    fig = go.Figure(
        data=[
            go.Candlestick(
                x=df["period_end"],
                open=df["open"],
                high=df["high"],
                low=df["low"],
                close=df["close"],
            )
        ]
    )
    fig.update_layout(xaxis_rangeslider_visible=False, height=520)
    st.plotly_chart(fig, use_container_width=True)

db_files = list_db_files()
db_path = st.selectbox("選擇 SQLite DB", db_files)

if not db_path:
    st.stop()

if not table_exists(db_path, "kbar_monthly"):
    st.error("DB 中找不到 kbar_monthly，請先執行 kbar_aggregator.py")
    st.stop()

meta = load_meta(db_path)
symbols = sorted(meta["symbol"].unique())
years = sorted(meta["year"].unique(), reverse=True)

c1, c2 = st.columns([2, 1])
symbol = c1.selectbox("Symbol", symbols)
year = c2.selectbox("Year", years)

df = load_monthly(db_path, symbol, int(year))
if df.empty:
    st.warning("沒有資料")
    st.stop()

m1, m2, m3 = st.columns(3)
m1.metric("月份數", len(df))
m2.metric("年度累積報酬%", f"{df['cum_ret_pct'].iloc[-1]:.2f}")
m3.metric("最大單月漲幅%", f"{df['ret_pct'].max():.2f}")

st.subheader("📈 月K走勢")
if PLOTLY_OK:
    plot_candle(df)
else:
    st.line_chart(df.set_index("period_end")["close"])

st.subheader("📋 月K明細")
st.dataframe(df, use_container_width=True)

st.download_button(
    "⬇️ 下載 CSV",
    data=df.to_csv(index=False).encode("utf-8-sig"),
    file_name=f"monthly_k_{symbol}_{year}.csv",
)
