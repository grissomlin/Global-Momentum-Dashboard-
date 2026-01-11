# pages/year_contribution.py
# Streamlit page: Year contribution by return bins
# Reads: year_contribution_bins (produced by kbar_contribution.py)
# Place under `pages/` directory.

import os
import sqlite3
import pandas as pd
import streamlit as st

try:
    import plotly.express as px  # type: ignore
    PLOTLY_OK = True
except Exception:
    PLOTLY_OK = False

st.set_page_config(page_title="Year Contribution (年度貢獻度)", layout="wide")
st.title("🏁 年度貢獻度（year_contribution_bins）")

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
def load_bins(db_path):
    conn = connect(db_path)
    try:
        # Intentionally select all columns; schema might evolve
        df = pd.read_sql("SELECT * FROM year_contribution_bins", conn)
        return df
    finally:
        conn.close()

def pick_col(df, candidates):
    for c in candidates:
        if c in df.columns:
            return c
    return None

# ----------------------------
# UI: DB selection
# ----------------------------
db_files = list_db_files()
db_path = st.selectbox("選擇 SQLite DB", db_files)

if not db_path:
    st.stop()

if not table_exists(db_path, "year_contribution_bins"):
    st.error("DB 中找不到 year_contribution_bins，請先執行 kbar_contribution.py 產生年度貢獻度分箱表。")
    st.stop()

df = load_bins(db_path)
if df.empty:
    st.warning("year_contribution_bins 目前沒有資料。")
    st.stop()

# Try to infer key columns with flexible naming
col_year = pick_col(df, ["year", "Year"])
col_bin = pick_col(df, ["ytd_bin", "ytd_ret_bin", "ret_bin", "bin", "YTD_Bin"])
col_market = pick_col(df, ["market", "Market", "market_detail", "MarketDetail"])
col_sector = pick_col(df, ["sector", "Sector"])

# Count / sample columns
col_count = pick_col(df, ["n", "count", "sample_count", "stocks", "num_stocks", "N"])
# Contribution columns (may differ by implementation)
col_avg_contrib = pick_col(df, ["avg_top_week_contrib", "avg_top_week_contribution", "top_week_contrib_avg", "avg_contribution"])
col_avg_lu = pick_col(df, ["avg_limitup_in_topweek", "avg_limitup_count_topweek", "limitup_count_avg", "avg_limitup_count"])
col_success = pick_col(df, ["win_rate", "success_rate", "prob_up", "winrate"])

# Sanity: required-ish
missing_key = [name for name, col in [("year", col_year), ("bin", col_bin)] if col is None]
if missing_key:
    st.warning(
        "⚠️ year_contribution_bins 欄位命名與頁面預期不同，仍可檢視原始資料，但篩選/圖表可能受限。\n\n"
        f"缺少可辨識欄位：{', '.join(missing_key)}"
    )

with st.expander("📌 表欄位預覽", expanded=False):
    st.write("Columns:", list(df.columns))
    st.dataframe(df.head(30), use_container_width=True)

# ----------------------------
# Filters
# ----------------------------
years = sorted(df[col_year].dropna().unique().tolist(), reverse=True) if col_year else []
bins = sorted(df[col_bin].dropna().unique().tolist()) if col_bin else []
markets = sorted(df[col_market].dropna().unique().tolist()) if col_market else []
sectors = sorted(df[col_sector].dropna().unique().tolist()) if col_sector else []

f1, f2, f3, f4 = st.columns([1, 1.5, 1.2, 1.2])

year_sel = f1.selectbox("Year", years if years else ["(unknown)"], index=0)
bin_sel = f2.multiselect("YTD 分箱", bins, default=bins[:8] if bins else [])
market_sel = f3.selectbox("Market", ["ALL"] + markets, index=0) if markets else "ALL"
sector_sel = f4.selectbox("Sector", ["ALL"] + sectors, index=0) if sectors else "ALL"

df_f = df.copy()
if col_year and year_sel != "(unknown)":
    df_f = df_f[df_f[col_year] == int(year_sel)]

if col_bin and bin_sel:
    df_f = df_f[df_f[col_bin].isin(bin_sel)]

if col_market and market_sel != "ALL":
    df_f = df_f[df_f[col_market] == market_sel]

if col_sector and sector_sel != "ALL":
    df_f = df_f[df_f[col_sector] == sector_sel]

if df_f.empty:
    st.warning("篩選後沒有資料。")
    st.stop()

# ----------------------------
# KPI cards (if columns exist)
# ----------------------------
kcols = st.columns(4)
if col_count:
    kcols[0].metric("樣本數合計", int(df_f[col_count].sum()))
else:
    kcols[0].metric("列數", int(len(df_f)))

if col_avg_contrib:
    kcols[1].metric("平均 TopWeek 貢獻(%)", f"{df_f[col_avg_contrib].mean():.2f}")
else:
    kcols[1].metric("欄位", "avg_contrib N/A")

if col_avg_lu:
    kcols[2].metric("平均 TopWeek 漲停數", f"{df_f[col_avg_lu].mean():.2f}")
else:
    kcols[2].metric("欄位", "avg_lu N/A")

if col_success:
    kcols[3].metric("平均勝率", f"{df_f[col_success].mean():.2f}")
else:
    kcols[3].metric("欄位", "win_rate N/A")

# ----------------------------
# Charts
# ----------------------------
st.subheader("📊 分箱比較")

# Choose a value for plotting
value_candidates = [c for c in [col_avg_contrib, col_avg_lu, col_success] if c is not None]
value_col = st.selectbox("圖表指標", value_candidates, index=0) if value_candidates else None

if value_col and col_bin:
    plot_df = df_f.copy()
    # If multiple rows per bin (e.g., by sector), aggregate
    agg_cols = {value_col: "mean"}
    if col_count:
        agg_cols[col_count] = "sum"

    plot_df = plot_df.groupby(col_bin, as_index=False).agg(agg_cols).sort_values(col_bin)

    if PLOTLY_OK:
        fig = px.bar(plot_df, x=col_bin, y=value_col, title=f"{value_col} by {col_bin}")
        fig.update_layout(height=360, margin=dict(l=10, r=10, t=40, b=10))
        st.plotly_chart(fig, use_container_width=True)
    else:
        st.bar_chart(plot_df.set_index(col_bin)[value_col])
else:
    st.info("目前缺少可用的分箱欄位或指標欄位，因此略過圖表。")

# ----------------------------
# Table + download
# ----------------------------
st.subheader("📋 year_contribution_bins 明細")
st.dataframe(df_f.reset_index(drop=True), use_container_width=True, height=520)

st.download_button(
    "⬇️ 下載 CSV（篩選後）",
    data=df_f.to_csv(index=False).encode("utf-8-sig"),
    file_name="year_contribution_bins_filtered.csv",
    mime="text/csv",
)

