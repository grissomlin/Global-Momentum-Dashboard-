# data_cleaning.py
# -*- coding: utf-8 -*-
"""
data_cleaning.py
----------------
通用價格資料清洗工具（獨立模組）
目標：把「乒乓極端震盪 / 極端錯價」清洗口徑集中管理，供各 pipeline 重用。

✅ 支援兩種常見口徑：
- price_col="Adj Close"：用於「還原股價 / 公司行為校正」研究（Supabase 那支）
- price_col="close"     ：用於「事件/漲停/貢獻度」的聚合清洗（SQLite pipeline）

✅ 提供：
- detect_pingpong_mask(): 只偵測（回傳 mask 與原因）
- clean_pingpong():      直接清洗（可選擇是否重算 prev_close / daily_change）

注意：
- 本檔案只做「資料品質清洗」，不依賴 market_rules，也不碰漲停判定。
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Dict, Tuple

import numpy as np
import pandas as pd


# -----------------------------
# Config
# -----------------------------
@dataclass(frozen=True)
class CleaningConfig:
    date_col: str = "date"             # 或 "Date"
    price_col: str = "close"           # 或 "Adj Close"
    pingpong_threshold: float = 0.40   # 40%
    abs_ret_cap: Optional[float] = 0.80  # 80%（None 表示不啟用）
    require_min_rows: int = 5
    keep_first_row: bool = True        # 避免把第一天 drop 掉（因 pct_change=NaN）


# -----------------------------
# Helpers
# -----------------------------
def _ensure_datetime_col(df: pd.DataFrame, date_col: str) -> pd.Series:
    s = pd.to_datetime(df[date_col], errors="coerce")
    # 去掉 timezone（若有）
    try:
        s = s.dt.tz_localize(None)
    except Exception:
        pass
    return s


def _safe_pct_change(price: pd.Series) -> pd.Series:
    p = pd.to_numeric(price, errors="coerce")
    # 避免 0 或負值造成怪異 pct
    p = p.where(p > 0, np.nan)
    return p.pct_change()


# -----------------------------
# Core: detect
# -----------------------------
def detect_pingpong_mask(
    df: pd.DataFrame,
    config: CleaningConfig = CleaningConfig(),
) -> Tuple[pd.Series, pd.DataFrame]:
    """
    回傳：
    - drop_mask: True 表示該列要被剔除
    - reasons  : 每列原因標記（pingpong/abs_cap），方便 debug/報表

    規則：
    1) abs_ret_cap：若 |ret| > abs_ret_cap，剔除該日
    2) pingpong：連續兩日 |ret| > threshold 且方向相反 => 兩日都剔除
    """
    if df is None or df.empty:
        empty_mask = pd.Series([], dtype=bool)
        empty_reasons = pd.DataFrame(columns=["is_abs_cap", "is_pingpong"], dtype=int)
        return empty_mask, empty_reasons

    date_col = config.date_col
    price_col = config.price_col

    if date_col not in df.columns:
        raise KeyError(f"[data_cleaning] 缺少 date_col='{date_col}' 欄位")
    if price_col not in df.columns:
        raise KeyError(f"[data_cleaning] 缺少 price_col='{price_col}' 欄位")

    if len(df) < config.require_min_rows:
        drop_mask = pd.Series(False, index=df.index)
        reasons = pd.DataFrame(
            {"is_abs_cap": 0, "is_pingpong": 0},
            index=df.index,
            dtype=int,
        )
        return drop_mask, reasons

    work = df.copy()
    work[date_col] = _ensure_datetime_col(work, date_col)
    work = work.dropna(subset=[date_col]).sort_values(date_col).reset_index(drop=False)  # keep original index
    orig_index = work["index"].copy()

    ret = _safe_pct_change(work[price_col])

    # abs cap
    is_abs_cap = pd.Series(False, index=work.index)
    if config.abs_ret_cap is not None:
        is_abs_cap = ret.abs() > float(config.abs_ret_cap)

    # pingpong
    is_pingpong = pd.Series(False, index=work.index)
    thr = float(config.pingpong_threshold)

    # i 與 i+1 都滿足 |ret|>thr 且反向
    # 注意：ret[0] 會是 NaN
    for i in range(1, len(work) - 1):
        prev = ret.iloc[i]
        nxt = ret.iloc[i + 1]
        if pd.notna(prev) and pd.notna(nxt):
            if (abs(prev) > thr) and (abs(nxt) > thr) and (prev * nxt < 0):
                is_pingpong.iloc[i] = True
                is_pingpong.iloc[i + 1] = True

    # 可選：保留第一列（避免因 NaN ret 被誤判）
    if config.keep_first_row and len(work) > 0:
        is_abs_cap.iloc[0] = False
        is_pingpong.iloc[0] = False

    drop_local = (is_abs_cap | is_pingpong).fillna(False)

    # 映射回原 df.index
    drop_mask = pd.Series(False, index=df.index)
    drop_mask.loc[orig_index.values] = drop_local.values

    reasons = pd.DataFrame(
        {
            "is_abs_cap": 0,
            "is_pingpong": 0,
        },
        index=df.index,
        dtype=int,
    )
    reasons.loc[orig_index.values, "is_abs_cap"] = is_abs_cap.astype(int).values
    reasons.loc[orig_index.values, "is_pingpong"] = is_pingpong.astype(int).values

    return drop_mask, reasons


# -----------------------------
# Core: clean
# -----------------------------
def clean_pingpong(
    df: pd.DataFrame,
    config: CleaningConfig = CleaningConfig(),
    *,
    # 若你是 SQLite pipeline 日K聚合：通常想重算這兩個
    recompute_prev_close: bool = False,
    recompute_daily_change: bool = False,
    prev_close_col: str = "prev_close",
    daily_change_col: str = "daily_change",
    # 回傳原因方便你 debug
    return_reasons: bool = False,
) -> pd.DataFrame | Tuple[pd.DataFrame, pd.DataFrame]:
    """
    直接清洗 df，剔除 pingpong/abs-cap 日。

    - recompute_prev_close / recompute_daily_change：清洗後重算（同一 symbol 內）
      ⚠️ 本函數不分組，你若是多檔股票，請在外面 groupby(symbol) 後再呼叫。
    """
    if df is None or df.empty:
        return (df.copy(), pd.DataFrame()) if return_reasons else df.copy()

    drop_mask, reasons = detect_pingpong_mask(df, config=config)

    out = df.loc[~drop_mask].copy()

    # 重新排序（保持時間序）
    date_col = config.date_col
    out[date_col] = _ensure_datetime_col(out, date_col)
    out = out.dropna(subset=[date_col]).sort_values(date_col).reset_index(drop=True)

    # 清洗後可選重算 prev_close / daily_change
    if recompute_prev_close:
        if config.price_col in out.columns:
            out[prev_close_col] = pd.to_numeric(out[config.price_col], errors="coerce").shift(1)
        elif "close" in out.columns:
            out[prev_close_col] = pd.to_numeric(out["close"], errors="coerce").shift(1)

    if recompute_daily_change:
        # 以 price_col 為主；否則 fallback close
        base_col = config.price_col if config.price_col in out.columns else "close"
        out[daily_change_col] = pd.to_numeric(out[base_col], errors="coerce").pct_change()

    if return_reasons:
        # reasons 是針對原 df 的 index；這裡回傳原 reasons，讓你可以對照被刪掉哪些
        return out, reasons

    return out


# -----------------------------
# Convenience presets
# -----------------------------
def preset_adjclose_cleaning(
    *,
    date_col: str = "Date",
    pingpong_threshold: float = 0.40,
    abs_ret_cap: Optional[float] = None,  # 你原本 Supabase 版本沒有 abs cap
) -> CleaningConfig:
    return CleaningConfig(
        date_col=date_col,
        price_col="Adj Close",
        pingpong_threshold=pingpong_threshold,
        abs_ret_cap=abs_ret_cap,
    )


def preset_close_cleaning(
    *,
    date_col: str = "date",
    pingpong_threshold: float = 0.40,
    abs_ret_cap: Optional[float] = 0.80,
) -> CleaningConfig:
    return CleaningConfig(
        date_col=date_col,
        price_col="close",
        pingpong_threshold=pingpong_threshold,
        abs_ret_cap=abs_ret_cap,
    )


# -----------------------------
# Example usage (do not run in production import)
# -----------------------------
if __name__ == "__main__":
    # quick self-test (toy)
    toy = pd.DataFrame(
        {
            "date": pd.date_range("2025-01-01", periods=7, freq="D"),
            "close": [100, 160, 96, 97, 98, 99, 100],  # 100->160 (+60%), 160->96 (-40%) pingpong-ish
        }
    )
    cfg = preset_close_cleaning()
    cleaned, rs = clean_pingpong(
        toy, cfg, recompute_prev_close=True, recompute_daily_change=True, return_reasons=True
    )
    print("raw:\n", toy)
    print("reasons:\n", rs)
    print("cleaned:\n", cleaned)
