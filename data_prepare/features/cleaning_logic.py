import pandas as pd
from data_prepare.features.data_distribute import data_distribution


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
    elif strategy == "forward fill":
        df[col] = df[col].ffill()
    elif strategy == "backward fill":
        df[col] = df[col].bfill()
    elif strategy == "drop rows":
        df = df.dropna(subset=[col]).reset_index(drop=True)
    return df


def use_outlier_strategy(
    df: pd.DataFrame, col: str, strategy: str, lower: float, upper: float
) -> pd.DataFrame:
    """จัดการ Outliers ในคอลัมน์ด้วย strategy ที่กำหนด

    loop จนกว่า data_distribution() จะรายงาน 0 outlier (สูงสุด 10 รอบ)
    ใช้ data_distribution() เป็น oracle โดยตรง — รับประกัน consistent กับ UI เสมอ
    """
    df = df.copy()

    for _ in range(10):
        series = df[col]
        is_out = series.notna() & ((series < lower) | (series > upper))
        if not is_out.any():
            # floating-point boundary: values clipped to exact bound may not satisfy > upper
            # check oracle first before declaring done
            _, check = data_distribution(df[[col]])
            check_match = next((d for d in check if d["Column"] == col), None)
            if check_match is None or check_match["Outliers"] == 0:
                break
            eps = max(abs(upper - lower), 1.0) * 1e-10
            is_out = series.notna() & ((series < lower + eps) | (series > upper - eps))
            if not is_out.any():
                break
        if strategy == "clip":
            df[col] = series.clip(lower=lower, upper=upper)
        else:  # drop rows — NaN ไม่ถือเป็น outlier
            df = df[~is_out].reset_index(drop=True)
        _, details = data_distribution(df[[col]])
        match = next((d for d in details if d["Column"] == col), None)
        if match is None or match["Outliers"] == 0:
            break
        new_lower, new_upper = match["Lower"], match["Upper"]
        # bounds หยุดเปลี่ยน → clip converged แล้ว (floating point precision)
        if abs(new_lower - lower) < 1e-9 and abs(new_upper - upper) < 1e-9:
            break
        lower, upper = new_lower, new_upper

    return df
