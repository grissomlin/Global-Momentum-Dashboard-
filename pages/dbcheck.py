# pages/dbcheck.py
# -*- coding: utf-8 -*-
import os
import sqlite3
import pandas as pd
import streamlit as st

# -----------------------------
# Helpers
# -----------------------------
def walk_find_db_files(search_roots, max_files=200):
    found = []
    seen = set()

    for root in search_roots:
        if not root or not os.path.exists(root):
            continue
        # 如果 root 本身是檔案
        if os.path.isfile(root) and root.lower().endswith(".db"):
            p = os.path.abspath(root)
            if p not in seen:
                found.append(p); seen.add(p)
            continue

        # 遞迴掃資料夾
        for dirpath, _, filenames in os.walk(root):
            for fn in filenames:
                if fn.lower().endswith(".db"):
                    p = os.path.abspath(os.path.join(dirpath, fn))
                    if p not in seen:
                        found.append(p); seen.add(p)
                        if len(found) >= max_files:
                            return sorted(found)
    return sorted(found)

def get_tables(conn):
    q = "SELECT name FROM sqlite_master WHERE type='table' AND name NOT LIKE 'sqlite_%' ORDER BY name"
    return [r[0] for r in conn.execute(q).fetchall()]

def get_columns(conn, table):
    rows = conn.execute(f"PRAGMA table_info({table})").fetchall()
    return [{"name": r[1], "type": r[2], "notnull": r[3], "pk": r[5]} for r in rows]

def get_count(conn, table):
    return conn.execute(f"SELECT COUNT(*) FROM {table}").fetchone()[0]

def read_head(conn, table, n):
    return pd.read_sql(f"SELECT * FROM {table} LIMIT {int(n)}", conn)

def make_ai_prompt(table, cols, sample_df):
    col_lines = "\n".join([f"- {c['name']} ({c['type']})" for c in cols])
    sample_csv = sample_df.to_csv(index=False)
    return f"""你是一位資料工程/量化研究助理。請協助我理解 SQLite 資料表欄位定義與用途，並檢查是否有缺欄、型別不一致或可疑資料品質問題。

【Table】{table}

【Schema】
{col_lines}

【Top rows (CSV)】
{sample_csv}

請輸出：
1) 每個欄位用途推測（中英文）
2) 最可能需要補充/修正的欄位與理由
3) 建議我後續用哪些 SQL/查詢去驗證資料品質
"""

# -----------------------------
# UI
# -----------------------------
st.set_page_config(page_title="DB Check", layout="wide")
st.title("🧪 DB Check（資料庫讀取 + Schema + Sample + 欄位字典 + AI Prompt）")
st.caption("用途：確認 SQLite DB 可正常讀取、快速看到每個 table 的欄位與前 N 筆，並生成可直接貼去問 AI 的 prompt。")

cwd = os.getcwd()

# 你在 Render 常見路徑：/opt/render/project/src（程式）+ /opt/render/project（上層）+ /tmp（暫存）
search_roots = [
    st.session_state.get("db_path", ""),  # 若其他頁面已設 db_path，優先放進來
    cwd,
    "/opt/render/project/src",
    "/opt/render/project",
    "/tmp",
]

with st.expander("🔍 環境資訊 / DB 掃描", expanded=True):
    st.write("當前目錄:", cwd)
    st.write("掃描 roots:", search_roots)

    # 額外：列出 cwd 檔案，讓你一眼看到 DB 到底有沒有在這層
    try:
        cwd_files = sorted(os.listdir(cwd))
        st.write(f"cwd 檔案數: {len(cwd_files)}")
        st.code("\n".join(cwd_files[:200]) + ("\n...(truncated)" if len(cwd_files) > 200 else ""))
    except Exception as e:
        st.warning(f"無法列出 cwd 檔案：{e}")

db_files = walk_find_db_files(search_roots)

# 手動輸入路徑（超重要，救命用）
manual_path = st.text_input("（可選）手動輸入 DB 絕對路徑", value="")

# 如果手動輸入有效，直接用
picked_db = None
if manual_path and os.path.exists(manual_path) and manual_path.lower().endswith(".db"):
    picked_db = os.path.abspath(manual_path)

# 否則用掃描結果
if picked_db is None:
    if not db_files:
        st.error("找不到任何 .db 檔案。👉 這代表 DB 不在掃描路徑內或同步其實沒落地到這個 container。")
        st.info("建議：把你『同步 DB』那段程式下載的實際路徑 print 出來（或在同步後把 db_path 寫入 st.session_state['db_path']）。")
        st.stop()

    # 預設挑 tw_stock_warehouse.db
    default_idx = 0
    for i, p in enumerate(db_files):
        if os.path.basename(p) == "tw_stock_warehouse.db":
            default_idx = i
            break

    picked_db = st.selectbox("SQLite DB（遞迴掃描結果）", db_files, index=default_idx)

# 全站共用
st.session_state["db_path"] = picked_db
st.session_state["db_name"] = os.path.basename(picked_db)

st.success(f"✅ 使用 DB：{picked_db}")

n_head = st.number_input("每表顯示前 N 筆", min_value=1, max_value=200, value=10, step=1)

# 欄位字典（可慢慢擴充）
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

conn = sqlite3.connect(picked_db, timeout=60)
try:
    tables = get_tables(conn)
    if not tables:
        st.warning("DB 內沒有任何 table。")
        st.stop()

    st.subheader("📚 Tables")
    table = st.selectbox("選擇 Table", tables, index=0)

    total = get_count(conn, table)
    st.write(f"**{table}** | rows: **{total:,}**")

    cols_meta = get_columns(conn, table)

    st.markdown("### 🧱 Schema")
    st.dataframe(pd.DataFrame(cols_meta), use_container_width=True)

    st.markdown("### 🔟 Sample (Top N rows)")
    sample_df = read_head(conn, table, n_head)
    st.dataframe(sample_df, use_container_width=True)

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
    st.dataframe(pd.DataFrame(dict_rows), use_container_width=True)

    st.markdown("### 🤖 一鍵生成「可貼去問 AI」的 Prompt")
    st.text_area("Prompt（可直接複製）", make_ai_prompt(table, cols_meta, sample_df), height=260)

finally:
    conn.close()
