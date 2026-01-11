# pages/weekly_k.py
# Streamlit page: Weekly K-line viewer (reads kbar_weekly from SQLite)
# Place this file under your Streamlit app's `pages/` directory.

import os
import sqlite3
from datetime import datetime

import pandas as pd
import streamlit as st

# Optional: Plotly candlestick (preferred)
try:
    import plotly.graph_objects as go  # type: ignore
    PLOTLY_OK = True
except Exception:
    PLOTLY_OK = False

st.set_page_config(page_title="Weekly K (週K)", layout="wide")
st.title("📅 週K（kbar_weekly）檢視器")

# ----------------------------
# Helpers
# ----------------------------
def _list_db_candidates() -> list[str]:
    """Find SQLite db files in common working directories."""
    candidates = []
    for base in [os.getcwd(), "/content", "/mount/src", os.path.expanduser("~")]:
        if not base or not os.path.isdir(base):
            continue
        try:
            for fn in os.listdir(base):
                if fn.endswith(".db") and ("stock_warehouse" in fn or fn.endswith(".db")):
                    candidates.append(os.path.join(base, fn))
        except Exception:
            pass
    # de-dup while keeping order
    seen = set()
    out = []
    for p in candidates:
        if p not in seen and os.path.exists(p):
            seen.add(p)
            out.append(p)
    return out


def _connect(db_path: str) -> sqlite3.Connection:
    conn = sqlite3.connect(db_path, check_same_thread=False)
    conn.execute("PRAGMA journal_mode=WAL;")
    conn.execute("PRAGMA synchronous=NORMAL;")
    conn.execute("PRAGMA temp_store=MEMORY;")
    return conn


@st.cache_data(ttl=30)
def _table_exists(db_path: str, table: str) -> bool:
    conn = _connect(db_path)
    try:
        cur = conn.execute(
            "SELECT 1 FROM sqlite_master WHERE type='table' AND name=? LIMIT 1;",
            (table,),
        )
        return cur.fetchone() is not None
    finally:
        conn.close()


@st.cache_data(ttl=30)
def _get_symbol_years(db_path: str) -> pd.DataFrame:
    conn = _connect(db_path)
    try:
        df = pd.read_sql(
            """
            SELECT symbol, year, COUNT(*) AS n_weeks,
                   MIN(period_start) AS min_start,
                   MAX(period_end)   AS max_end
            FROM kbar_weekly
            GROUP BY symbol, year
            ORDER BY year DESC, n_weeks DESC;
            """,
            conn,
        )
        return df
    finally:
        conn.close()


@st.cache_data(ttl=30)
def _load_weekly(db_path: str, symbol: str, year: int) -> pd.DataFrame:
    conn = _connect(db_path)
    try:
        df = pd.read_sql(
            """
            SELECT symbol, year, week_id, period_start, period_end,
                   open, high, low, close, volume
            FROM kbar_weekly
            WHERE symbol = ? AND year = ?
            ORDER BY period_start ASC;
            """,
            conn,
            params=(symbol, year),
        )
        if not df.empty:
            # Normalize dates for plotting / filtering
            df["period_start"] = pd.to_datetime(df["period_start"], errors="coerce")
            df["period_end"] = pd.to_datetime(df["period_end"], errors="coerce")
            # Derived metrics
            df["ret_pct"] = (df["close"] / df["open"] - 1.0) * 100.0
            df["cum_ret_pct"] = (df["close"] / df["open"].iloc[0] - 1.0) * 100.0
        return df
    finally:
        conn.close()


def _candlestick_plot(df: pd.DataFrame):
    x = df["period_end"]
    fig = go.Figure(
        data=[
            go.Candlestick(
                x=x,
                open=df["open"],
                high=df["high"],
                low=df["low"],
                close=df["close"],
                name="週K",
            )
        ]
    )
    fig.update_layout(
        xaxis_title="Week End",
        yaxis_title="Price",
        height=520,
        margin=dict(l=10, r=10, t=30, b=10),
        xaxis_rangeslider_visible=False,
    )
    st.plotly_chart(fig, use_container_width=True)

    # Volume bar (separate simple chart)
    fig2 = go.Figure()
    fig2.add_trace(go.Bar(x=x, y=df["volume"], name="Volume"))
    fig2.update_layout(
        xaxis_title="Week End",
        yaxis_title="Volume",
        height=260,
        margin=dict(l=10, r=10, t=30, b=10),
    )
    st.plotly_chart(fig2, use_container_width=True)


def _fallback_line_plot(df: pd.DataFrame):
    import matplotlib.pyplot as plt  # local import per Streamlit best practice

    x = df["period_end"]
    fig = plt.figure()
    plt.plot(x, df["close"])
    plt.xticks(rotation=45)
    plt.title("Weekly Close")
    plt.tight_layout()
    st.pyplot(fig, clear_figure=True)


# ----------------------------
# UI: DB selection
# ----------------------------
db_candidates = _list_db_candidates()

left, right = st.columns([2, 1])
with left:
    db_path = st.selectbox("選擇 SQLite DB 檔", options=db_candidates, index=0 if db_candidates else None)
with right:
    st.caption("找不到 DB？把 `.db` 放到專案根目錄或 /content。")

if not db_path:
    st.warning("目前沒有找到任何 `.db` 檔。")
    st.stop()

if not os.path.exists(db_path):
    st.error(f"DB 不存在：{db_path}")
    st.stop()

if not _table_exists(db_path, "kbar_weekly"):
    st.error("DB 裡找不到 `kbar_weekly` 表。請先跑 kbar_aggregator.py 產生週K。")
    st.stop()

# ----------------------------
# UI: Symbol/year selection
# ----------------------------
meta = _get_symbol_years(db_path)
if meta.empty:
    st.warning("kbar_weekly 目前沒有任何資料。")
    st.stop()

with st.expander("📌 資料概況（symbol × year）", expanded=False):
    st.dataframe(meta, use_container_width=True, height=280)

symbols = sorted(meta["symbol"].unique().tolist())
years = sorted(meta["year"].unique().tolist(), reverse=True)

col1, col2, col3 = st.columns([2, 1, 1])
with col1:
    symbol = st.selectbox("Symbol", options=symbols, index=0)
with col2:
    year = st.selectbox("Year", options=years, index=0)
with col3:
    n_last = st.selectbox("顯示最近 N 週", options=[0, 26, 52, 104, 260], index=2, help="0 = 全年")

df = _load_weekly(db_path, symbol, int(year))

if df.empty:
    st.warning("沒有資料。可能該 symbol/year 不存在。")
    st.stop()

if n_last and n_last > 0 and len(df) > n_last:
    df_view = df.tail(n_last).copy()
else:
    df_view = df.copy()

# ----------------------------
# Quick metrics
# ----------------------------
m1, m2, m3, m4 = st.columns(4)
m1.metric("週數", f"{len(df)}")
m2.metric("年度累積報酬(%)", f"{df['cum_ret_pct'].iloc[-1]:.2f}")
m3.metric("最大單週漲幅(%)", f"{df['ret_pct'].max():.2f}")
m4.metric("最大單週跌幅(%)", f"{df['ret_pct'].min():.2f}")

# ----------------------------
# Charts
# ----------------------------
st.subheader("📈 週K走勢")
if PLOTLY_OK:
    _candlestick_plot(df_view)
else:
    st.info("Plotly 未安裝，改用簡易折線圖。你可安裝 plotly 取得 K 線圖：pip install plotly")
    _fallback_line_plot(df_view)

# ----------------------------
# Table + download
# ----------------------------
st.subheader("📋 週K明細")
show_cols = [
    "symbol", "year", "week_id", "period_start", "period_end",
    "open", "high", "low", "close", "volume", "ret_pct", "cum_ret_pct"
]
st.dataframe(df_view[show_cols], use_container_width=True, height=420)

csv_bytes = df_view[show_cols].to_csv(index=False).encode("utf-8-sig")
st.download_button(
    "⬇️ 下載 CSV",
    data=csv_bytes,
    file_name=f"weekly_k_{symbol}_{year}.csv",
    mime="text/csv",
)
