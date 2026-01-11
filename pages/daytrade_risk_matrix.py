# pages/daytrade_risk_matrix.py
# Streamlit page: Daytrade risk matrix for post-limit-up behaviors
# Works with stock_analysis table produced by processor_OVERWRITE_v2.py
#
# Focus: 2024 and 2025 (tabs), plus flexible filters.
#
# Metrics computed (per event x volume level):
# - sample size
# - D1 close red rate (D1_close_ret > 0)
# - D1 close mean
# - Hit 10% in 1-5d (max_ret_1_5d >= 0.10)
# - Hit 10% in 6-20d (max_ret_6_20d >= 0.10)
# - Mean max/min returns (1-5d, 6-20d)

import os
import sqlite3
import pandas as pd
import streamlit as st

try:
    import plotly.express as px  # type: ignore
    PLOTLY_OK = True
except Exception:
    PLOTLY_OK = False

st.set_page_config(page_title="Daytrade Risk Matrix", layout="wide")
st.title("🧨 隔日沖風險矩陣（事件 × 爆量分層 × 後續報酬）")

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
        cur = conn.execute("SELECT 1 FROM sqlite_master WHERE type='table' AND name=?", (table,))
        return cur.fetchone() is not None
    finally:
        conn.close()

@st.cache_data(ttl=60)
def load_stock_analysis(db_path, year: int):
    """
    Load minimal columns needed for the matrix.
    If some columns are missing, they will be filled with NaN/0 safely.
    """
    needed = [
        "symbol", "date", "year",
        "is_limit_up", "prev_day_limit_up", "prev_consecutive_limits",
        "vol_level", "vol_ratio",
        "has_upper_shadow", "is_long_black",
        "hit_limit_up_intraday", "hit_but_failed_limit_up",
        "open_near_limit_up", "gap_up_and_long_black", "hit_limitup_intraday_and_long_black",
        "D1_close_ret",
        "max_ret_1_5d", "min_ret_1_5d",
        "max_ret_6_20d", "min_ret_6_20d",
    ]
    conn = connect(db_path)
    try:
        # Load all columns (SQLite doesn't support SELECT * EXCEPT) but keep it small via projection if possible
        # We'll attempt projection; if it fails due to missing cols, fallback to SELECT * and reindex.
        cols_sql = ", ".join([f'"{c}"' for c in needed])
        try:
            df = pd.read_sql(
                f"""
                SELECT {cols_sql}
                FROM stock_analysis
                WHERE year = ?
                """,
                conn,
                params=(year,),
            )
        except Exception:
            df = pd.read_sql("SELECT * FROM stock_analysis WHERE year = ?", conn, params=(year,))

        # Normalize / ensure columns exist
        for c in needed:
            if c not in df.columns:
                df[c] = pd.NA

        # Types / safety
        if "date" in df.columns:
            df["date"] = pd.to_datetime(df["date"], errors="coerce")
        for c in ["is_limit_up", "prev_day_limit_up", "has_upper_shadow", "is_long_black",
                  "hit_limit_up_intraday", "hit_but_failed_limit_up",
                  "open_near_limit_up", "gap_up_and_long_black", "hit_limitup_intraday_and_long_black"]:
            if c in df.columns:
                df[c] = pd.to_numeric(df[c], errors="coerce").fillna(0).astype(int)

        for c in ["prev_consecutive_limits"]:
            if c in df.columns:
                df[c] = pd.to_numeric(df[c], errors="coerce")

        for c in ["D1_close_ret", "max_ret_1_5d", "min_ret_1_5d", "max_ret_6_20d", "min_ret_6_20d", "vol_ratio"]:
            if c in df.columns:
                df[c] = pd.to_numeric(df[c], errors="coerce")

        if "vol_level" in df.columns:
            df["vol_level"] = df["vol_level"].fillna("NA").astype(str)

        return df
    finally:
        conn.close()

def build_events_df(df: pd.DataFrame, streak_min: int = 2):
    """
    Build boolean event columns for matrix.
    Only meaningful on days where prev_day_limit_up==1 (i.e., the day after a limit-up).
    """
    base = df.copy()

    # If prev_consecutive_limits exists, define "post_streak" as >= streak_min on prev day.
    post_streak = (base["prev_day_limit_up"] == 1)
    if "prev_consecutive_limits" in base.columns:
        post_streak = post_streak & (base["prev_consecutive_limits"].fillna(0) >= streak_min)

    base["E_post_lu_upper_shadow_fail"] = (
        (base["prev_day_limit_up"] == 1) &
        (base["hit_but_failed_limit_up"] == 1) &
        (base["has_upper_shadow"] == 1)
    ).astype(int)

    base["E_post_lu_long_black"] = (
        (base["prev_day_limit_up"] == 1) &
        (base["is_long_black"] == 1)
    ).astype(int)

    base["E_post_streak_upper_shadow_fail"] = (
        post_streak &
        (base["hit_but_failed_limit_up"] == 1) &
        (base["has_upper_shadow"] == 1)
    ).astype(int)

    base["E_post_streak_long_black"] = (
        post_streak &
        (base["is_long_black"] == 1)
    ).astype(int)

    base["E_open_near_limit_up"] = (
        (base["prev_day_limit_up"] == 1) &
        (base["open_near_limit_up"] == 1)
    ).astype(int)

    base["E_gap_up_and_long_black"] = (
        (base["prev_day_limit_up"] == 1) &
        (base["gap_up_and_long_black"] == 1)
    ).astype(int)

    base["E_hit_limitup_then_long_black"] = (
        (base["prev_day_limit_up"] == 1) &
        (base["hit_limitup_intraday_and_long_black"] == 1)
    ).astype(int)

    return base

def summarize_matrix(df: pd.DataFrame, event_col: str, hit_th_1_5: float, hit_th_6_20: float):
    """
    Return a matrix-like table grouped by vol_level for given event.
    """
    d = df[df[event_col] == 1].copy()
    if d.empty:
        return pd.DataFrame()

    def _safe_mean(s):
        s = pd.to_numeric(s, errors="coerce")
        return float(s.mean()) if s.notna().any() else float("nan")

    def _safe_rate(cond_series):
        cond_series = cond_series.fillna(False)
        return float(cond_series.mean()) if len(cond_series) else float("nan")

    out = []
    for vol_level, g in d.groupby("vol_level"):
        n = len(g)
        d1_red = _safe_rate(pd.to_numeric(g["D1_close_ret"], errors="coerce") > 0)
        d1_mean = _safe_mean(g["D1_close_ret"])

        p_hit_1_5 = _safe_rate(pd.to_numeric(g["max_ret_1_5d"], errors="coerce") >= hit_th_1_5)
        p_hit_6_20 = _safe_rate(pd.to_numeric(g["max_ret_6_20d"], errors="coerce") >= hit_th_6_20)

        out.append({
            "vol_level": vol_level,
            "n": n,
            "D1_red_rate": d1_red,
            "D1_close_mean": d1_mean,
            f"P_max_1_5>= {hit_th_1_5:.0%}": p_hit_1_5,
            f"P_max_6_20>= {hit_th_6_20:.0%}": p_hit_6_20,
            "max_1_5_mean": _safe_mean(g["max_ret_1_5d"]),
            "min_1_5_mean": _safe_mean(g["min_ret_1_5d"]),
            "max_6_20_mean": _safe_mean(g["max_ret_6_20d"]),
            "min_6_20_mean": _safe_mean(g["min_ret_6_20d"]),
        })

    res = pd.DataFrame(out).sort_values(["vol_level", "n"], ascending=[True, False])

    # Pretty formatting columns (keep numeric for download; format in display)
    return res

def format_display(df: pd.DataFrame):
    if df.empty:
        return df
    disp = df.copy()
    pct_cols = [c for c in disp.columns if c.startswith("P_") or c.endswith("_rate")]
    for c in pct_cols:
        disp[c] = (pd.to_numeric(disp[c], errors="coerce") * 100).round(2)
    num_cols = [c for c in disp.columns if c.endswith("_mean")]
    for c in num_cols:
        disp[c] = (pd.to_numeric(disp[c], errors="coerce") * 100).round(2)
    return disp

# ----------------------------
# UI: DB selection
# ----------------------------
db_files = list_db_files()
db_path = st.selectbox("選擇 SQLite DB", db_files)

if not db_path:
    st.stop()

if not table_exists(db_path, "stock_analysis"):
    st.error("DB 中找不到 stock_analysis。請先跑 processor / pipeline 產生。")
    st.stop()

# Controls
c1, c2, c3, c4 = st.columns([1.1, 1.1, 1.2, 1.3])
streak_min = c1.selectbox("連板定義（前一日 ≥ 幾連）", [2, 3, 4], index=0)
hit_th_1_5 = c2.selectbox("1–5日 續漲門檻", [0.05, 0.10, 0.15, 0.20], index=1, format_func=lambda x: f"{int(x*100)}%")
hit_th_6_20 = c3.selectbox("6–20日 續漲門檻", [0.05, 0.10, 0.15, 0.20], index=1, format_func=lambda x: f"{int(x*100)}%")
min_n = c4.selectbox("最小樣本數 n", [10, 20, 50, 100, 200], index=1)

EVENTS = [
    ("E_post_lu_upper_shadow_fail", "昨天漲停 → 今日碰板未鎖 + 上影線"),
    ("E_post_lu_long_black", "昨天漲停 → 今日長黑"),
    ("E_open_near_limit_up", "昨天漲停 → 今日開盤接近漲停(≤0.5%)"),
    ("E_gap_up_and_long_black", "昨天漲停 → 今日高開 + 長黑"),
    ("E_hit_limitup_then_long_black", "昨天漲停 → 今日碰板後收長黑"),
    ("E_post_streak_upper_shadow_fail", f"昨天連板(≥{streak_min}) → 今日碰板未鎖 + 上影線"),
    ("E_post_streak_long_black", f"昨天連板(≥{streak_min}) → 今日長黑"),
]

def render_year(year: int):
    df = load_stock_analysis(db_path, year)
    df = build_events_df(df, streak_min=streak_min)

    # Filter to "day after limit up" universe for interpretability
    universe = df[df["prev_day_limit_up"] == 1].copy()

    # Show universe size
    u1, u2, u3 = st.columns(3)
    u1.metric("Year", str(year))
    u2.metric("T+1樣本數（昨天漲停的隔天）", int(len(universe)))
    hv_rate = (universe["vol_level"].eq("HV").mean() * 100) if len(universe) else 0.0
    u3.metric("HV比例(隔天爆量)", f"{hv_rate:.2f}%")

    # Event selector
    event_col = st.selectbox(
        "選擇事件（Event）",
        options=[e[0] for e in EVENTS],
        format_func=lambda k: dict(EVENTS).get(k, k),
        key=f"event_{year}",
    )

    mat = summarize_matrix(df, event_col, hit_th_1_5, hit_th_6_20)
    if mat.empty:
        st.warning("此事件在該年度沒有樣本。")
        return

    mat = mat[mat["n"] >= int(min_n)].copy()
    if mat.empty:
        st.warning("有樣本，但在你設定的最小樣本數門檻下都被篩掉了。")
        return

    st.subheader("📌 風險矩陣（依爆量分層）")
    st.caption("顯示為百分比：D1_red_rate / P_hit / 各 mean（已轉成 %）。下載 CSV 會保留原始小數。")
    st.dataframe(format_display(mat), use_container_width=True, height=360)

    # Simple chart: n and probabilities
    if PLOTLY_OK:
        show_cols = ["vol_level", "n"] + [c for c in mat.columns if c.startswith("P_") or c.endswith("_rate")]
        chart = mat[show_cols].copy()
        chart_long = chart.melt(id_vars=["vol_level", "n"], var_name="metric", value_name="value")
        fig = px.bar(chart_long, x="vol_level", y="value", color="metric", barmode="group",
                     title="分層比較（機率/比例）")
        fig.update_layout(height=360, margin=dict(l=10, r=10, t=40, b=10))
        st.plotly_chart(fig, use_container_width=True)

    # Download
    st.download_button(
        "⬇️ 下載矩陣 CSV（原始值）",
        data=mat.to_csv(index=False).encode("utf-8-sig"),
        file_name=f"daytrade_risk_matrix_{year}_{event_col}.csv",
        mime="text/csv",
        key=f"dl_{year}_{event_col}",
    )

    # Also provide top symbols for quick inspection (optional)
    with st.expander("🔎 事件明細（Top 200 rows）", expanded=False):
        detail = df[df[event_col] == 1].copy()
        cols_show = [
            "date", "symbol", "prev_consecutive_limits", "vol_level", "vol_ratio",
            "D1_close_ret", "max_ret_1_5d", "min_ret_1_5d", "max_ret_6_20d", "min_ret_6_20d",
            "open_near_limit_up", "has_upper_shadow", "is_long_black",
            "hit_limit_up_intraday", "hit_but_failed_limit_up",
            "gap_up_and_long_black", "hit_limitup_intraday_and_long_black",
        ]
        cols_show = [c for c in cols_show if c in detail.columns]
        detail = detail.sort_values(["date"]).tail(200)[cols_show]
        st.dataframe(detail, use_container_width=True, height=420)

# Tabs for 2024/2025
tab1, tab2 = st.tabs(["2024", "2025"])
with tab1:
    render_year(2024)
with tab2:
    render_year(2025)
