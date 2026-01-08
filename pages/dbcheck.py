# pages/dbcheck.py
# -*- coding: utf-8 -*-
import os
import sqlite3
import pandas as pd
import streamlit as st
from datetime import datetime

# -----------------------------
# Helpers
# -----------------------------
def list_db_files(search_dirs):
    out = []
    for d in search_dirs:
        if d and os.path.isdir(d):
            for fn in os.listdir(d):
                if fn.lower().endswith(".db"):
                    out.append(os.path.join(d, fn))
    # 去重 + 排序
    out = sorted(list(dict.fromkeys(out)))
    return out

def get_tables(conn):
    q = "SELECT name FROM sqlite_master WHERE type='table' AND name NOT LIKE 'sqlite_%' ORDER BY name"
    return [r[0] for r in conn.execute(q).fetchall()]

def get_columns(conn, table):
    # PRAGMA table_info returns: cid, name, type, notnull, dflt_value, pk
    rows = conn.execute(f"PRAGMA table_info({table})").fetchall()
    return [{"name": r[1], "type": r[2], "notnull": r[3], "pk": r[5]} for r in rows]

def get_count(conn, table):
    return conn.execute(f"SELECT COUNT(*) FROM {table}").fetchone()[0]

def read_head(conn, table, n):
    return pd.read_sql(f"SELECT * FROM {table} LIMIT {int(n)}", conn)

def make_ai_prompt(table, cols, sample_df):
    col_lines = "\n".join([f"- {c['name']} ({c['type']})" for c in cols])
    sample_csv = sample_df.to_csv(index=False)
    return f"""你是一位資料工程/量化研究助理。請協助我理解 SQLite 資料表的欄位定義與用途，並檢查是否有缺欄、型別不一致或可疑資料品質問題。

【Table】{table}

【Schema】
{col_lines}

【Top rows (CSV)】
{sample_csv}

請輸出：
1) 你對每個欄位的用途推測（中英文）
2) 你覺得最可能需要補充/修正的欄位與理由
3) 建議我後續用哪些 SQL/查詢去驗證資料品質
"""

# -----------------------------
# UI
# -----------------------------
st.set_page_config(page_title="DB Check", layout="wide")
st.title("🧪 DB Check（資料庫讀取 + Schema + Sample + 欄位字典 + AI Prompt）")
st.caption("用途：確認 SQLite DB 可正常讀取、快速看到每個 table 的欄位與前 N 筆，並生成可直接貼去問 AI 的 prompt。")

# 搜尋 db 的目錄：專案根目錄 + /tmp（Render/GHA 常用）
search_dirs = [os.getcwd(), "/tmp"]
db_files = list_db_files(search_dirs)

with st.expander("🔍 環境資訊 / DB 掃描", expanded=False):
    st.write("當前目錄:", os.getcwd())
    st.write("可掃描目錄:", search_dirs)
    if db_files:
        for p in db_files:
            st.write(f"✅ {os.path.basename(p)} - {os.path.getsize(p):,} bytes - {p}")
    else:
        st.warning("找不到任何 .db 檔案。請先同步 DB 或確認 DB 落地路徑。")

if not db_files:
    st.stop()

# 預設挑 tw_stock_warehouse.db（如果存在）
default_idx = 0
for i, p in enumerate(db_files):
    if os.path.basename(p) == "tw_stock_warehouse.db":
        default_idx = i
        break

db_path = st.selectbox(
    "SQLite DB 路徑（自動掃描）",
    db_files,
    index=default_idx,
)

# 全站共用：寫入 session_state
st.session_state["db_path"] = db_path
st.session_state["db_name"] = os.path.basename(db_path)

n_head = st.number_input("每表顯示前 N 筆", min_value=1, max_value=200, value=10, step=1)

# 欄位字典（你可以慢慢擴充）
# key: (table, column) -> {"zh": "...", "en": "...", "note": "..."}
COLUMN_DICT = {
    ("stock_prices", "symbol"): {"zh": "股票代號", "en": "Symbol/Ticker"},
    ("stock_prices", "date"): {"zh": "交易日期", "en": "Trading date"},
    ("stock_prices", "open"): {"zh": "開盤價", "en": "Open price"},
    ("stock_prices", "high"): {"zh": "最高價", "en": "High price"},
    ("stock_prices", "low"): {"zh": "最低價", "en": "Low price"},
    ("stock_prices", "close"): {"zh": "收盤價", "en": "Close price"},
    ("stock_prices", "volume"): {"zh": "成交量", "en": "Volume"},
    ("stock_info", "market"): {"zh": "市場代碼", "en": "Market code"},
    ("stock_info", "sector"): {"zh": "產業別", "en": "Sector"},
    ("stock_analysis", "is_limit_up"): {"zh": "是否漲停(收盤)", "en": "Is limit-up at close"},
    ("stock_analysis", "lu_type"): {"zh": "漲停型態", "en": "Limit-up pattern type"},
    ("stock_analysis", "consecutive_limits"): {"zh": "連板天數", "en": "Consecutive limit-up days"},
    ("stock_analysis", "strength_rank"): {"zh": "強度分箱標籤", "en": "Strength rank bin label"},
    ("stock_analysis", "strength_value"): {"zh": "強度數值化", "en": "Strength numeric value"},
}

# 讀 DB
conn = sqlite3.connect(db_path, timeout=60)

try:
    tables = get_tables(conn)
    if not tables:
        st.warning("DB 內沒有任何 table。")
        st.stop()

    st.subheader("📚 Tables")
    cols = st.columns([2, 1, 1, 4])
    with cols[0]:
        table = st.selectbox("選擇 Table", tables, index=0)
    with cols[1]:
        show_schema = st.toggle("顯示欄位", value=True)
    with cols[2]:
        show_dict = st.toggle("顯示字典", value=True)
    with cols[3]:
        st.info("提示：你也可以用這頁生成 prompt，直接貼去問 AI 做資料品質檢查 / 欄位用途推測。")

    # 基本資訊
    total = get_count(conn, table)
    st.write(f"**{table}** | rows: **{total:,}**")

    # Schema
    cols_meta = get_columns(conn, table)
    if show_schema:
        schema_df = pd.DataFrame(cols_meta)
        st.markdown("### 🧱 Schema")
        st.dataframe(schema_df, use_container_width=True)

    # Sample rows
    st.markdown("### 🔟 Sample (Top N rows)")
    sample_df = read_head(conn, table, n_head)
    st.dataframe(sample_df, use_container_width=True)

    # Dictionary
    if show_dict:
        st.markdown("### 📖 欄位字典（中英文）")
        dict_rows = []
        for c in cols_meta:
            key = (table, c["name"])
            d = COLUMN_DICT.get(key, {})
            dict_rows.append({
                "column": c["name"],
                "type": c["type"],
                "zh": d.get("zh", ""),
                "en": d.get("en", ""),
                "note": d.get("note", ""),
            })
        dict_df = pd.DataFrame(dict_rows)
        st.dataframe(dict_df, use_container_width=True)

        st.caption("✅ 這份字典你可以逐步補齊（新增到 COLUMN_DICT）。")

    # AI Prompt
    st.markdown("### 🤖 一鍵生成「可貼去問 AI」的 Prompt")
    prompt = make_ai_prompt(table, cols_meta, sample_df)
    st.text_area("Prompt（可直接複製）", prompt, height=260)

finally:
    conn.close()
