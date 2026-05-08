import pandas as pd
from data_prepare.logic.data_distribute import data_distribution


def use_missing_strategy(dataset: pd.DataFrame, column_name: str, missing_strategy: str, modify_inplace: bool = False) -> pd.DataFrame:
    """แก้ Missing Values ในคอลัมน์เดียว"""
    if not modify_inplace:
        dataset = dataset.copy()
        
    if missing_strategy == "mean":
        dataset[column_name] = dataset[column_name].fillna(dataset[column_name].mean())
    elif missing_strategy == "median":
        dataset[column_name] = dataset[column_name].fillna(dataset[column_name].median())
    elif missing_strategy == "median (rounded)":
        dataset[column_name] = dataset[column_name].fillna(round(dataset[column_name].median()))
    elif missing_strategy == "most frequent":
        frequent_values = dataset[column_name].mode()
        if len(frequent_values) > 0:
            dataset[column_name] = dataset[column_name].fillna(frequent_values[0])
    elif missing_strategy == "forward fill":
        dataset[column_name] = dataset[column_name].ffill()
    elif missing_strategy == "backward fill":
        dataset[column_name] = dataset[column_name].bfill()
    elif missing_strategy == "drop rows":
        dataset = dataset.dropna(subset=[column_name]).reset_index(drop=True)
        
    return dataset


def use_missing_strategy_bulk(dataset: pd.DataFrame, column_strategies: dict) -> pd.DataFrame:
    """แก้ Missing Values หลายคอลัมน์ในครั้งเดียวเพื่อลดการสร้าง DataFrame ซ้ำซ้อน (Performance optimization)"""
    dataset = dataset.copy()
    columns_to_drop = []
    
    for column_name, missing_strategy in column_strategies.items():
        if missing_strategy == "drop rows":
            columns_to_drop.append(column_name)
        else:
            use_missing_strategy(dataset, column_name, missing_strategy, modify_inplace=True)
            
    if columns_to_drop:
        dataset = dataset.dropna(subset=columns_to_drop).reset_index(drop=True)
        
    return dataset


def use_outlier_strategy(
    dataset: pd.DataFrame, column_name: str, outlier_strategy: str, lower_bound: float, upper_bound: float, modify_inplace: bool = False
) -> pd.DataFrame:
    """จัดการ Outliers ในคอลัมน์ด้วย Vectorized Operations (ทำงานรอบเดียว ไม่วนลูปซ้ำ)"""
    if not modify_inplace:
        dataset = dataset.copy()

    data_series = dataset[column_name]
    if outlier_strategy == "clip":
        dataset[column_name] = data_series.clip(lower=lower_bound, upper=upper_bound)
    else:  # drop rows
        is_outlier = data_series.notna() & ((data_series < lower_bound) | (data_series > upper_bound))
        dataset = dataset[~is_outlier].reset_index(drop=True)

    return dataset


def use_outlier_strategy_bulk(
    dataset: pd.DataFrame, column_strategies: dict
) -> pd.DataFrame:
    """จัดการ Outliers หลายคอลัมน์ในครั้งเดียว (Performance optimization)
    column_strategies = { "col_name": {"strategy": "clip", "lower": -1.0, "upper": 1.0} }
    """
    dataset = dataset.copy()
    global_outlier_mask = None
    
    for column_name, strategy_params in column_strategies.items():
        outlier_strategy = strategy_params["strategy"]
        lower_bound = strategy_params["lower"]
        upper_bound = strategy_params["upper"]
        
        if outlier_strategy == "clip":
            dataset[column_name] = dataset[column_name].clip(lower=lower_bound, upper=upper_bound)
        else:  # drop rows
            data_series = dataset[column_name]
            is_outlier = data_series.notna() & ((data_series < lower_bound) | (data_series > upper_bound))
            
            if global_outlier_mask is None:
                global_outlier_mask = is_outlier
            else:
                global_outlier_mask = global_outlier_mask | is_outlier
                
    if global_outlier_mask is not None:
        dataset = dataset[~global_outlier_mask].reset_index(drop=True)
        
    return dataset
