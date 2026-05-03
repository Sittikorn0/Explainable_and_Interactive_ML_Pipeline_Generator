import pandas as pd
from data_prepare.features.data_distribute import data_distribution


def use_missing_strategy(df: pd.DataFrame, col: str, strategy: str, inplace: bool = False) -> pd.DataFrame:
    """แก้ Missing Values ในคอลัมน์เดียว"""
    if not inplace:
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


def use_missing_strategy_bulk(df: pd.DataFrame, strategies: dict) -> pd.DataFrame:
    """แก้ Missing Values หลายคอลัมน์ในครั้งเดียวเพื่อลดการสร้าง DataFrame ซ้ำซ้อน (Performance optimization)"""
    df = df.copy()
    drop_cols = []
    
    for col, strategy in strategies.items():
        if strategy == "drop rows":
            drop_cols.append(col)
        else:
            use_missing_strategy(df, col, strategy, inplace=True)
            
    if drop_cols:
        df = df.dropna(subset=drop_cols).reset_index(drop=True)
        
    return df


def use_outlier_strategy(
    df: pd.DataFrame, col: str, strategy: str, lower: float, upper: float, inplace: bool = False
) -> pd.DataFrame:
    """จัดการ Outliers ในคอลัมน์ด้วย Vectorized Operations (ทำงานรอบเดียว ไม่วนลูปซ้ำ)"""
    if not inplace:
        df = df.copy()

    series = df[col]
    if strategy == "clip":
        df[col] = series.clip(lower=lower, upper=upper)
    else:  # drop rows
        is_out = series.notna() & ((series < lower) | (series > upper))
        df = df[~is_out].reset_index(drop=True)

    return df


def use_outlier_strategy_bulk(
    df: pd.DataFrame, strategies: dict
) -> pd.DataFrame:
    """จัดการ Outliers หลายคอลัมน์ในครั้งเดียว (Performance optimization)
    strategies = { "col_name": {"strategy": "clip", "lower": -1.0, "upper": 1.0} }
    """
    df = df.copy()
    outlier_mask = None
    
    for col, params in strategies.items():
        strategy = params["strategy"]
        lower = params["lower"]
        upper = params["upper"]
        
        if strategy == "clip":
            df[col] = df[col].clip(lower=lower, upper=upper)
        else:  # drop rows
            series = df[col]
            is_out = series.notna() & ((series < lower) | (series > upper))
            if outlier_mask is None:
                outlier_mask = is_out
            else:
                outlier_mask = outlier_mask | is_out
                
    if outlier_mask is not None:
        df = df[~outlier_mask].reset_index(drop=True)
        
    return df

