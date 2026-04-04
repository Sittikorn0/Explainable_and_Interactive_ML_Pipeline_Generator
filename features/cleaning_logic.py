import pandas as pd

def use_missing_strategy(df: pd.DataFrame, col: str, strategy: str) -> pd.DataFrame:
    """แก้ Missing Values ในคอลัมน์ด้วย strategy ที่กำหนด"""
    if strategy == "mean":
        df[col] = df[col].fillna(df[col].mean())
    elif strategy == "median":
        df[col] = df[col].fillna(df[col].median())
    elif strategy == "most frequent":
        df[col] = df[col].fillna(df[col].mode()[0])
    elif strategy == "drop rows":
        df = df.dropna(subset=[col]).reset_index(drop=True)
    return df


def use_outlier_strategy(
    df: pd.DataFrame, col: str, strategy: str, lower: float, upper: float
) -> pd.DataFrame:
    """จัดการ Outliers ในคอลัมน์ด้วย strategy ที่กำหนด"""
    series = df[col]
    if strategy == "clip":
        df[col] = series.clip(lower=lower, upper=upper)
    elif strategy == "drop rows":
        mask = (series >= lower) & (series <= upper)
        df = df[mask].reset_index(drop=True)
    return df
