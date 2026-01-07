# pages/dbcheck.py
# -*- coding: utf-8 -*-
"""
DB Check / Schema Explorer (Streamlit Page)
-------------------------------------------
目的：
- 檢查 SQLite DB 是否可讀
- 列出每個 table 的欄位 (schema) 與前 10 筆資料
- 在下方提供「欄位中英文解釋（Data Dictionary）」與可直接複製貼上問 AI 的 prompt

✅ 不依賴 data_cleaning.py（純 DB 讀取檢查）
"""

from __future__ import annotations

import os
import sqlite3
from typing import Dict, Optional, Tuple, List

import pandas as pd
import streamlit as st


# =========================
# Helpers
# =========================
@st.cache_data(show_spinner=False)
def list_tables(db_path: str) -> List[str]:
    conn = sqlite3.connect(db_path)
    try:
        df = pd.read_sql(
            "SELECT name FROM sqlite_master WHERE type='table' AND name NOT LIKE 'sqlite_%' ORDER BY name;",
            conn,
        )
        return df["name"].tolist()
    finally:
        conn.close()


@st.cache_data(show_spinner=False)
def read_table_schema(db_path: str, table: str) -> pd.DataFrame:
    conn = sqlite3.connect(db_path)
    try:
        # PRAGMA table_info returns: cid, name, type, notnull, dflt_value, pk
        rows = conn.execute(f"PRAGMA table_info('{table}')").fetchall()
        return pd.DataFrame(rows, columns=["cid", "name", "type", "notnull", "dflt_value", "pk"])
    finally:
        conn.close()


@st.cache_data(show_spinner=False)
def read_table_head(db_path: str, table: str, n: int = 10) -> pd.DataFrame:
    conn = sqlite3.connect(db_path)
    try:
        return pd.read_sql(f"SELECT * FROM '{table}' LIMIT {int(n)};", conn)
    finally:
        conn.close()


def _safe_read_scalar(conn: sqlite3.Connection, sql: str) -> Optional[float]:
    try:
        r = conn.execute(sql).fetchone()
        return r[0] if r else None
    except Exception:
        return None


@st.cache_data(show_spinner=False)
def get_quick_stats(db_path: str) -> Dict[str, Optional[float]]:
    """盡量不假設 table 一定存在；有就顯示，沒有就略過。"""
    conn = sqlite3.connect(db_path)
    try:
        stats: Dict[str, Optional[float]] = {}
        stats["tables"] = _safe_read_scalar(conn, "SELECT COUNT(*) FROM sqlite_master WHERE type='table' AND name NOT LIKE 'sqlite_%';")
        stats["stock_prices_rows"] = _safe_read_scalar(conn, "SELECT COUNT(*) FROM stock_prices;")
        stats["stock_analysis_rows"] = _safe_read_scalar(conn, "SELECT COUNT(*) FROM stock_analysis;")
        stats["limit_up_events_rows"] = _safe_read_scalar(conn, "SELECT COUNT(*) FROM limit_up_events;")
        stats["daytrade_events_rows"] = _safe_read_scalar(conn, "SELECT COUNT(*) FROM daytrade_events;")
        stats["year_contribution_rows"] = _safe_read_scalar(conn, "SELECT COUNT(*) FROM year_contribution;")
        return stats
    finally:
        conn.close()


def _build_default_dictionary() -> Dict[str, Dict[str, Tuple[str, str]]]:
    """
    data_dictionary[table][column] = (中文, English)
    只放你 repo 常用表；遇到未知欄位會顯示空白，方便你補。
    """
    dd: Dict[str, Dict[str, Tuple[str, str]]] = {}

    # -------------------------
    # stock_info
    # -------------------------
    dd["stock_info"] = {
        "symbol": ("股票代號（含市場後綴）", "Ticker/Symbol (with market suffix)"),
        "name": ("公司名稱", "Company name"),
        "market": ("市場（TW/CN/JP/HK/US...）", "Market code (TW/CN/JP/HK/US...)"),
        "market_detail": ("市場細分（上市/上櫃/科創板等）", "Market detail (TSE/OTC/STAR/...)"),
        "sector": ("產業別", "Sector"),
        "industry": ("細產業別", "Industry"),
    }

    # -------------------------
    # stock_prices (raw or lightly cleaned)
    # -------------------------
    dd["stock_prices"] = {
        "symbol": ("股票代號", "Symbol"),
        "date": ("交易日期", "Trading date"),
        "open": ("開盤價", "Open"),
        "high": ("最高價", "High"),
        "low": ("最低價", "Low"),
        "close": ("收盤價", "Close"),
        "adj_close": ("還原收盤價（若有）", "Adjusted close (if available)"),
        "volume": ("成交量", "Volume"),
    }

    # -------------------------
    # stock_analysis (processor 產物)
    # -------------------------
    dd["stock_analysis"] = {
        "symbol": ("股票代號", "Symbol"),
        "date": ("交易日期", "Trading date"),
        "open": ("開盤價", "Open"),
        "high": ("最高價", "High"),
        "low": ("最低價", "Low"),
        "close": ("收盤價", "Close"),
        "volume": ("成交量", "Volume"),
        "prev_close": ("前一日收盤價", "Previous close"),
        "daily_change": ("日報酬率（close/prev_close-1）", "Daily return (close/prev_close - 1)"),
        "market": ("市場代碼", "Market code"),
        "market_detail": ("市場細分", "Market detail"),
        "sector": ("產業別", "Sector"),
        "is_limit_up": ("是否收漲停", "Is limit-up at close"),
        "lu_type": ("漲停型態（你文章規則）", "Limit-up pattern/type (your article rules)"),
        "consecutive_limits": ("連板天數", "Consecutive limit-up days"),
        "is_one_tick_lock": ("是否一字鎖", "One-tick lock (open=high=low=close)"),
        "limit_up_price": ("漲停價（精準）", "Limit-up price (exact)"),
        "hit_limit_up_intraday": ("盤中是否觸及漲停", "Hit limit-up intraday"),
        "limit_up_fail": ("盤中到漲停但收不住", "Hit limit-up but failed at close"),
    }

    # -------------------------
    # kbar_*
    # -------------------------
    dd["kbar_weekly"] = {
        "symbol": ("股票代號", "Symbol"),
        "year": ("年份（用於分群）", "Year (grouping key)"),
        "week_id": ("週序號（年內第幾週）", "Week id (within year)"),
        "period_start": ("週期起日（含）", "Period start (inclusive)"),
        "period_end": ("週期訖日（含）", "Period end (inclusive)"),
        "open": ("週開盤價", "Weekly open"),
        "high": ("週最高價", "Weekly high"),
        "low": ("週最低價", "Weekly low"),
        "close": ("週收盤價", "Weekly close"),
        "volume": ("週成交量", "Weekly volume"),
    }
    dd["kbar_monthly"] = {
        "symbol": ("股票代號", "Symbol"),
        "year": ("年份（用於分群）", "Year (grouping key)"),
        "month_id": ("月份（1-12）", "Month id (1-12)"),
        "period_start": ("週期起日（含）", "Period start (inclusive)"),
        "period_end": ("週期訖日（含）", "Period end (inclusive)"),
        "open": ("月開盤價", "Monthly open"),
        "high": ("月最高價", "Monthly high"),
        "low": ("月最低價", "Monthly low"),
        "close": ("月收盤價", "Monthly close"),
        "volume": ("月成交量", "Monthly volume"),
    }
    dd["kbar_yearly"] = {
        "symbol": ("股票代號", "Symbol"),
        "year": ("年份", "Year"),
        "period_start": ("年度起日（含）", "Period start (inclusive)"),
        "period_end": ("年度訖日（含）", "Period end (inclusive)"),
        "open": ("年開盤價", "Yearly open"),
        "high": ("年最高價", "Yearly high"),
        "low": ("年最低價", "Yearly low"),
        "close": ("年收盤價", "Yearly close"),
        "volume": ("年成交量", "Yearly volume"),
        "year_peak_date": ("年度最高點日期（raw）", "Peak date within year (raw)"),
        "year_peak_high": ("年度最高價", "Peak high within year"),
    }

    # -------------------------
    # year_contribution / events（你的研究表）
    # -------------------------
    dd["year_contribution"] = {
        "symbol": ("股票代號", "Symbol"),
        "year": ("年份", "Year"),
        "year_ret_pct": ("年度報酬率（%）", "Year return (%)"),
        "year_logret": ("年度 log return", "Year log return"),
        "burst_style_week": ("爆發型態（週）", "Burst style (week)"),
        "burst_style_month": ("爆發型態（月）", "Burst style (month)"),
        "year_peak_trade_date": ("峰值對齊到實際交易日", "Peak date aligned to trading day"),
        "share_year_to_peak": ("到 peak 的 logret 佔全年正報酬比例", "Share of year positive logret achieved by peak"),
        "limitup_count_to_peak": ("peak 前漲停天數", "Limit-up count before peak"),
    }

    dd["limit_up_events"] = {
        "symbol": ("股票代號", "Symbol"),
        "date": ("事件日期", "Event date"),
        "market": ("市場", "Market"),
        "sector": ("產業", "Sector"),
        "is_limit_up": ("收漲停", "Limit-up at close"),
        "lu_type": ("漲停型態", "Limit-up pattern/type"),
        "consecutive_limits": ("連板天數", "Consecutive limit-ups"),
        "is_one_tick_lock": ("一字鎖", "One-tick lock"),
        "next_open_ret": ("隔日開盤報酬（next_open/close-1）", "Next-day open return (next_open/close - 1)"),
        "next_intraday_drawdown": ("隔日盤中回撤（next_low/next_open-1）", "Next-day intraday drawdown (next_low/next_open - 1)"),
        "ret_1d": ("未來 1 日報酬（close-based）", "Forward 1D return (close-based)"),
        "ret_5d": ("未來 5 日報酬（close-based）", "Forward 5D return (close-based)"),
        "max_up_5d": ("未來 5 日最大上漲（high/close-1）", "Max up in next 5D (high/close - 1)"),
        "max_dd_5d": ("未來 5 日最大回撤（low/close-1）", "Max drawdown in next 5D (low/close - 1)"),
    }

    dd["daytrade_events"] = {
        "symbol": ("股票代號", "Symbol"),
        "date": ("事件日期", "Event date"),
        "prev_limit_up_today_not": ("昨天漲停、今天沒漲停", "Prev day limit-up, today not"),
        "prev_limit_up_today_fail": ("昨天漲停、今天衝漲停失敗", "Prev day limit-up, today fail"),
        "today_limit_up_fail_no_prev": ("昨天沒漲停、今天衝漲停失敗", "No prev limit-up, today fail"),
        "today_limit_up_yes_no_prev": ("昨天沒漲停、今天收漲停（首板）", "No prev limit-up, today limit-up"),
    }

    return dd


def render_data_dictionary(table: str, schema_df: pd.DataFrame, dd_map: Dict[str, Dict[str, Tuple[str, str]]]) -> pd.DataFrame:
    cols = schema_df["name"].tolist()
    tmap = dd_map.get(table, {})
    out_rows = []
    for c in cols:
        zh, en = ("", "")
        if c in tmap:
            zh, en = tmap[c]
        out_rows.append({"column": c, "中文說明": zh, "English": en})
    return pd.DataFrame(out_rows)


def build_ai_prompt(db_path: str, table: str, schema_df: pd.DataFrame, head_df: pd.DataFrame, dict_df: pd.DataFrame) -> str:
    # 盡量短且好複製：schema + sample + dictionary
    schema_lines = []
    for _, r in schema_df.iterrows():
        schema_lines.append(f"- {r['name']} ({r['type']}){' [PK]' if int(r['pk'])==1 else ''}")
    schema_txt = "\n".join(schema_lines)

    # sample：避免太長
    sample_txt = head_df.to_csv(index=False)

    # dictionary：只列出有填說明者（避免淹沒）
    dict_filled = dict_df.copy()
    dict_filled["has_desc"] = (dict_filled["中文說明"].astype(str).str.len() > 0) | (dict_filled["English"].astype(str).str.len() > 0)
    dict_filled = dict_filled[dict_filled["has_desc"]].drop(columns=["has_desc"], errors="ignore")

    dict_lines = []
    for _, r in dict_filled.iterrows():
        dict_lines.append(f"- {r['column']}: {r['中文說明']} | {r['English']}")
    dict_txt = "\n".join(dict_lines) if dict_lines else "(No dictionary entries yet. Fill them in this page.)"

    return f"""你是資深量化/資料工程顧問。請幫我檢查這個 SQLite 資料表的設計是否合理，並指出：
1) 欄位命名是否一致、缺哪些必要欄位、有哪些可疑欄位或型別
2) 依我提供的 sample 前 10 筆，推測是否有資料異常或欄位定義不清
3) 給我 3-5 個建議：如何讓這張表更適合研究/回測/儀表板查詢

DB Path: {db_path}
Table: {table}

[Schema]
{schema_txt}

[Sample head (CSV, first 10 rows)]
{sample_txt}

[Data Dictionary]
{dict_txt}
"""


# =========================
# UI
# =========================
st.set_page_config(page_title="DB Check", layout="wide")

st.title("🧪 DB Check（資料庫讀取 + Schema + Sample + 欄位字典）")
st.caption("用途：確認 SQLite DB 可正常讀取、快速看到每個 table 的欄位與前 10 筆，並生成可直接貼去問 AI 的 prompt。")

# 讓它跟 dashboard.py 一樣的 market_code 命名（有 session_state 就沿用）
MARKET_MAP = {
    "台灣 (TW)": "tw-share",
    "香港 (HK)": "hk-share",
    "美國 (US)": "us-share",
    "日本 (JP)": "jp-share",
    "韓國 (KR)": "kr-share",
    "中國 (CN)": "cn-share",
}

c1, c2, c3 = st.columns([1.2, 1.3, 1.5], vertical_alignment="bottom")

with c1:
    default_market_label = None
    if "market_selection" in st.session_state:
        # dashboard.py 常用 key：market_selection = "台灣 (TW)"...
        default_market_label = st.session_state.get("market_selection")
    market_label = st.selectbox("Market", list(MARKET_MAP.keys()), index=list(MARKET_MAP.keys()).index(default_market_label) if default_market_label in MARKET_MAP else 0)

market_code = MARKET_MAP[market_label]
default_db = f"{market_code}_stock_warehouse.db"

with c2:
    db_path = st.text_input("SQLite DB 路徑", value=st.session_state.get("db_path", default_db))

with c3:
    head_n = st.number_input("每表顯示前 N 筆", min_value=5, max_value=200, value=10, step=5)

# 你可能從 dashboard 同步 DB：在這頁也提供提示
if not os.path.exists(db_path):
    st.warning(f"找不到 DB 檔：{db_path}\n\n如果你是在 dashboard 首頁做「同步資料庫」，請先同步後再來 DB Check。")

# 快速 stats
if os.path.exists(db_path):
    stats = get_quick_stats(db_path)
    a, b, c, d, e, f = st.columns(6)
    a.metric("Tables", int(stats.get("tables") or 0))
    b.metric("stock_prices", int(stats.get("stock_prices_rows") or 0))
    c.metric("stock_analysis", int(stats.get("stock_analysis_rows") or 0))
    d.metric("limit_up_events", int(stats.get("limit_up_events_rows") or 0))
    e.metric("daytrade_events", int(stats.get("daytrade_events_rows") or 0))
    f.metric("year_contribution", int(stats.get("year_contribution_rows") or 0))

st.divider()

if os.path.exists(db_path):
    try:
        tables = list_tables(db_path)
    except Exception as e:
        st.error(f"無法讀取 DB tables：{e}")
        st.stop()

    if not tables:
        st.info("DB 內沒有任何 tables（或只有 sqlite 系統表）。")
        st.stop()

    dd_map = _build_default_dictionary()

    left, right = st.columns([1.0, 2.2], vertical_alignment="top")

    with left:
        table = st.selectbox("選擇 table", tables)
        show_all_tables = st.checkbox("一次展開所有 tables（可能很慢）", value=False)

        st.markdown("#### 操作小抄")
        st.markdown(
            "- **Schema**：PRAGMA table_info\n"
            "- **Sample**：SELECT * LIMIT N\n"
            "- **Dictionary**：可在程式碼中補齊，或你也可以把 dict 區塊搬去獨立檔\n"
            "- **AI Prompt**：底下會自動產生，直接複製貼上即可"
        )

    def render_one_table(tname: str):
        schema_df = read_table_schema(db_path, tname)
        head_df = read_table_head(db_path, tname, n=int(head_n))
        dict_df = render_data_dictionary(tname, schema_df, dd_map)

        st.subheader(f"📋 {tname}")

        t1, t2 = st.tabs(["Schema & Sample", "Data Dictionary & AI Prompt"])

        with t1:
            st.markdown("**Schema**")
            st.dataframe(schema_df, use_container_width=True)

            st.markdown(f"**Sample head (top {int(head_n)})**")
            st.dataframe(head_df, use_container_width=True)

        with t2:
            st.markdown("**欄位中英文解釋（可自行補齊）**")
            st.dataframe(dict_df, use_container_width=True)

            st.markdown("**可直接貼上問 AI 的 prompt**（`st.code` 右上角通常可一鍵複製）")
            prompt = build_ai_prompt(db_path, tname, schema_df, head_df, dict_df)
            st.code(prompt, language="markdown")

    with right:
        if show_all_tables:
            for tname in tables:
                with st.expander(tname, expanded=False):
                    render_one_table(tname)
        else:
            render_one_table(table)

else:
    st.info("請先準備/同步 SQLite DB 檔案，再來使用 DB Check。")
