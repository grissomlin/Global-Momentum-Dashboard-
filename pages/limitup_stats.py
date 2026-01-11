# pages/limitup_stats.py
# Streamlit page: Limit-up statistics
# Reads: limit_up_daily_summary / limit_up_sector_daily
# Place under `pages/` directory.

import os
import sqlite3
from datetime import datetime

import pandas as pd
import streamlit as st

try:
    import plotly.express as px  # type: ignore
    PLOTLY_OK = True
except Exception:
    PLOTLY_OK = False

st.set_page_config(page_title="Limit-Up Stats (漲停統計)", layout="wide")
st.title("🚀 漲停統計（Summary Tables）")

# ----------------------------
# Helpers
# ----------------------------
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
    conn.execute("PRAGMA synchronous=NORMAL;")
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
def load_daily_summary(db_path):
    conn = connect(db_path)
    try:
        df = pd.read_sql(
            """
            SELECT date, market,
                   limitup_count, one_tick_count, fail_count, intraday_hit_count,
                   avg_consecutive, median_consecutive
            FROM limit_up_daily_summary
            ORDER BY date DESC, market ASC
            """,
            conn,
        )
        if not df.empty:
            df["date"] = pd.to_datetime(df["date"], errors="coerce")
            # Derived ratios
            df["one_tick_ratio"] = df.apply(
                lambda r: (r["one_tick_count"] / r["limitup_count"]) if r["limitup_count"] else 0.0,
                axis=1,
            )
        return df
    finally:
        conn.close()

@st.cache_data(ttl=60)
def load_sector_daily(db_path):
    conn = connect(db_path)
    try:
        df = pd.read_sql(
            """
            SELECT date, market, sector,
                   limitup_count, one_tick_count,
                   avg_consecutive, median_consecutive
            FROM limit_up_sector_daily
            ORDER BY date DESC, market ASC, limitup_count DESC
            """,
            conn,
        )
        if not df.empty:
            df["date"] = pd.to_datetime(df["date"], errors="coerce")
        return df
    finally:
        conn.close()

def _safe_bar(df, x, y, title):
    if not PLOTLY_OK:
        st.line_chart(df.set_index(x)[y] if x in df.columns else df[y])
        return
    fig = px.bar(df, x=x, y=y, title=title)
    fig.update_layout(height=320, margin=dict(l=10, r=10, t=40, b=10))
    st.plotly_chart(fig, use_container_width=True)

# ----------------------------
# UI: DB selection
# ----------------------------
db_files = list_db_files()
db_path = st.selectbox("選擇 SQLite DB", db_files)

if not db_path:
    st.stop()

need_tables = ["limit_up_daily_summary", "limit_up_sector_daily"]
missing = [t for t in need_tables if not table_exists(db_path, t)]
if missing:
    st.error(
        "DB 中缺少 summary 表："
        + ", ".join(missing)
        + "。請先執行 event_engine.py 產生 summary tables。"
    )
    st.stop()

daily = load_daily_summary(db_path)
sector = load_sector_daily(db_path)

if daily.empty:
    st.warning("limit_up_daily_summary 目前沒有資料。")
    st.stop()

# ----------------------------
# Filters
# ----------------------------
markets = sorted(daily["market"].dropna().unique().tolist())
date_min = daily["date"].min()
date_max = daily["date"].max()

c1, c2, c3 = st.columns([1.2, 1.2, 1])
market = c1.selectbox("Market", ["ALL"] + markets, index=0)
date_range = c2.slider(
    "日期範圍",
    min_value=date_min.to_pydatetime(),
    max_value=date_max.to_pydatetime(),
    value=(date_min.to_pydatetime(), date_max.to_pydatetime()),
    format="YYYY-MM-DD",
)
top_n = c3.selectbox("產業 Top N", [10, 15, 20, 30, 50], index=1)

start_dt, end_dt = date_range

daily_f = daily[(daily["date"] >= start_dt) & (daily["date"] <= end_dt)].copy()
sector_f = sector[(sector["date"] >= start_dt) & (sector["date"] <= end_dt)].copy()

if market != "ALL":
    daily_f = daily_f[daily_f["market"] == market]
    sector_f = sector_f[sector_f["market"] == market]

if daily_f.empty:
    st.warning("篩選後沒有資料。")
    st.stop()

# ----------------------------
# KPIs (latest date in filtered set)
# ----------------------------
latest_dt = daily_f["date"].max()
latest = daily_f[daily_f["date"] == latest_dt].copy()

# Aggregate latest across markets if market=ALL
if market == "ALL":
    latest_row = pd.Series({
        "limitup_count": latest["limitup_count"].sum(),
        "one_tick_count": latest["one_tick_count"].sum(),
        "fail_count": latest["fail_count"].sum(),
        "intraday_hit_count": latest["intraday_hit_count"].sum(),
        "avg_consecutive": (latest["avg_consecutive"] * latest["limitup_count"]).sum() / max(latest["limitup_count"].sum(), 1),
        "median_consecutive": latest["median_consecutive"].median(),
    })
else:
    latest_row = latest.iloc[0]

k1, k2, k3, k4, k5 = st.columns(5)
k1.metric("最新日期", latest_dt.strftime("%Y-%m-%d"))
k2.metric("漲停家數", int(latest_row["limitup_count"]))
k3.metric("一字鎖家數", int(latest_row["one_tick_count"]))
k4.metric("漲停失敗家數", int(latest_row["fail_count"]))
k5.metric("平均連板", f"{float(latest_row['avg_consecutive']):.2f}")

# ----------------------------
# Trend charts
# ----------------------------
st.subheader("📈 趨勢（每日）")

# Aggregate daily for chart if market=ALL
chart_df = daily_f.copy()
if market == "ALL":
    chart_df = (
        chart_df.groupby("date", as_index=False)[["limitup_count", "one_tick_count", "fail_count", "intraday_hit_count"]]
        .sum()
        .sort_values("date")
    )
else:
    chart_df = chart_df.sort_values("date")

cA, cB = st.columns(2)
with cA:
    _safe_bar(chart_df, "date", "limitup_count", "每日漲停家數")
with cB:
    _safe_bar(chart_df, "date", "one_tick_count", "每日一字鎖家數")

cC, cD = st.columns(2)
with cC:
    _safe_bar(chart_df, "date", "fail_count", "每日漲停失敗家數")
with cD:
    _safe_bar(chart_df, "date", "intraday_hit_count", "盤中觸及漲停（但未收漲停）")

# ----------------------------
# Sector leaderboard (latest date)
# ----------------------------
st.subheader("🏷️ 產業漲停榜（最新日）")

sector_latest = sector_f[sector_f["date"] == latest_dt].copy()
if market == "ALL":
    sector_latest = (
        sector_latest.groupby("sector", as_index=False)[["limitup_count", "one_tick_count"]]
        .sum()
        .sort_values("limitup_count", ascending=False)
    )
else:
    sector_latest = sector_latest.sort_values("limitup_count", ascending=False)

sector_latest_top = sector_latest.head(int(top_n)).copy()

c1, c2 = st.columns([1.2, 1])
with c1:
    st.dataframe(sector_latest_top, use_container_width=True, height=420)
with c2:
    if PLOTLY_OK and not sector_latest_top.empty:
        fig = px.bar(sector_latest_top, x="limitup_count", y=("sector" if "sector" in sector_latest_top.columns else sector_latest_top.index),
                     orientation="h", title=f"Top {top_n} 產業漲停家數")
        fig.update_layout(height=420, margin=dict(l=10, r=10, t=40, b=10))
        st.plotly_chart(fig, use_container_width=True)
    else:
        st.info("plotly 未安裝或無資料，略過圖表。")

# ----------------------------
# Raw tables (optional)
# ----------------------------
with st.expander("📋 原始資料表（daily summary）", expanded=False):
    st.dataframe(daily_f.sort_values(["date", "market"]).reset_index(drop=True), use_container_width=True, height=320)

with st.expander("📋 原始資料表（sector daily）", expanded=False):
    st.dataframe(sector_f.sort_values(["date", "market", "limitup_count"], ascending=[False, True, False]).reset_index(drop=True),
                 use_container_width=True, height=320)

# ----------------------------
# Download
# ----------------------------
st.subheader("⬇️ 下載")
c1, c2 = st.columns(2)
c1.download_button(
    "下載 daily summary CSV",
    data=daily_f.to_csv(index=False).encode("utf-8-sig"),
    file_name=f"limitup_daily_summary_{market}_{start_dt:%Y%m%d}_{end_dt:%Y%m%d}.csv",
    mime="text/csv",
)
c2.download_button(
    "下載 sector daily CSV",
    data=sector_f.to_csv(index=False).encode("utf-8-sig"),
    file_name=f"limitup_sector_daily_{market}_{start_dt:%Y%m%d}_{end_dt:%Y%m%d}.csv",
    mime="text/csv",
)
