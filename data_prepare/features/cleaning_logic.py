import pandas as pd
from scipy.stats import skew


def use_missing_strategy(df: pd.DataFrame, col: str, strategy: str) -> pd.DataFrame:
    """แก้ Missing Values ในคอลัมน์ด้วย strategy ที่กำหนด"""
    df = df.copy()
    if strategy == "mean":
        df[col] = df[col].fillna(df[col].mean())
    elif strategy == "median":
        df[col] = df[col].fillna(df[col].median())
    elif strategy == "median (rounded)":
        df[col] = df[col].fillna(round(df[col].median()))
    elif strategy == "most frequent":
        mode_vals = df[col].mode()
        if len(mode_vals) > 0:
            df[col] = df[col].fillna(mode_vals[0])
    elif strategy == "drop rows":
        df = df.dropna(subset=[col]).reset_index(drop=True)
    return df


def _recalc_bounds(series: pd.Series) -> tuple[float, float]:
    """คำนวณ outlier bounds ใหม่บน series ปัจจุบัน (ใช้ method เดียวกับ data_distribute)"""
    s = series.dropna()
    if s.nunique() <= 1 or s.std() == 0:
        return float(s.min()), float(s.max())
    col_skew = skew(s)
    if abs(col_skew) < 0.5:
        mean, std = s.mean(), s.std()
        return mean - 3 * std, mean + 3 * std
    else:
        q1, q3 = s.quantile(0.25), s.quantile(0.75)
        iqr = q3 - q1
        if iqr == 0:
            return float(q1), float(q3)
        return q1 - 1.5 * iqr, q3 + 1.5 * iqr


def use_outlier_strategy(
    df: pd.DataFrame, col: str, strategy: str, lower: float, upper: float
) -> pd.DataFrame:
    """จัดการ Outliers ในคอลัมน์ด้วย strategy ที่กำหนด

    clip: วนซ้ำสูงสุด 5 รอบด้วย bounds ที่คำนวณใหม่แต่ละรอบ
    เพื่อแก้ปัญหาที่ clip ไปแล้ว statistics เปลี่ยน ทำให้ค่ายังโผล่เป็น outlier
    """
    df = df.copy()
    series = df[col]
    if strategy == "clip":
        # รอบแรกใช้ bounds จาก UI
        df[col] = series.clip(lower=lower, upper=upper)
        # iterate ด้วย bounds ใหม่จนกว่าจะไม่มี outlier หรือครบ 5 รอบ
        for _ in range(4):
            new_lower, new_upper = _recalc_bounds(df[col])
            s = df[col].dropna()
            still_out = ((s < new_lower) | (s > new_upper)).sum()
            if still_out == 0:
                break
            df[col] = df[col].clip(lower=new_lower, upper=new_upper)
    elif strategy == "drop rows":
        mask = (series >= lower) & (series <= upper)
        df = df[mask].reset_index(drop=True)
    return df
