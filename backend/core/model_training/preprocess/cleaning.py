# Libraries
import numpy as np
import pandas as pd

# Missing Value Handling (Cleaning)
def clean_fit_transform(features_train: pd.DataFrame, features_test: pd.DataFrame, missing_rules: dict = None) -> tuple:
    """
    เติมค่าว่าง (Missing Value) ป้องกัน Data Leakage โดยคำนวณค่าสถิติจาก features_train เท่านั้น
    และใช้กฎจากหน้า UI Data Cleaning (missing_rules)
    """
    if missing_rules is None:
        missing_rules = {}
        
    # จัดการข้อมูลประเภท Boolean และ Infinity
    for dataset_split in [features_train, features_test]:
        boolean_columns = dataset_split.select_dtypes(include="bool").columns
        for col in boolean_columns:
            dataset_split[col] = dataset_split[col].astype(int)
        dataset_split.replace([np.inf, -np.inf], 0, inplace=True)

    # คำนวณค่าจาก features_train ตาม rules
    fill_values_dict = {}
    for column_name in features_train.columns:
        if features_train[column_name].isna().any() or features_test[column_name].isna().any():
            strategy = missing_rules.get(column_name, "median" if pd.api.types.is_numeric_dtype(features_train[column_name]) else "most frequent")
            
            if strategy == "mean":
                fill_values_dict[column_name] = features_train[column_name].mean()
            elif strategy == "median":
                fill_values_dict[column_name] = features_train[column_name].median()
            elif strategy == "median (rounded)":
                fill_values_dict[column_name] = round(features_train[column_name].median())
            elif strategy == "most frequent":
                modes = features_train[column_name].mode()
                fill_values_dict[column_name] = modes.iloc[0] if not modes.empty else 0
            elif strategy in ("forward fill", "backward fill"):
                # ffill/bfill ไม่มีความหมายหลัง random split — fallback ใช้ median/mode (fit on train)
                if pd.api.types.is_numeric_dtype(features_train[column_name]):
                    fill_values_dict[column_name] = features_train[column_name].median()
                else:
                    modes = features_train[column_name].mode()
                    fill_values_dict[column_name] = modes.iloc[0] if not modes.empty else 0
            elif strategy == "drop rows":
                # Drop rows happens before split. If it reaches here, fallback to median/mode
                if pd.api.types.is_numeric_dtype(features_train[column_name]):
                    fill_values_dict[column_name] = features_train[column_name].median()
                else:
                    modes = features_train[column_name].mode()
                    fill_values_dict[column_name] = modes.iloc[0] if not modes.empty else 0
                
            if pd.isna(fill_values_dict.get(column_name, 0)):
                fill_values_dict[column_name] = 0

    # เติมค่าว่างให้ทั้ง Train และ Test
    if fill_values_dict:
        features_train = features_train.fillna(fill_values_dict)
        features_test  = features_test.fillna(fill_values_dict)

    return features_train, features_test

# Outlier Handling (Clipping)
def outlier_fit_transform(features_train: pd.DataFrame, features_test: pd.DataFrame, outlier_rules: dict = None) -> tuple:
    """
    คลิป Outliers โดยใช้ขอบเขตที่ได้จากกฎหน้า UI (ซึ่งตั้งอยู่บนโครงสร้างข้อมูลดั้งเดิม แต่ตัดที่ Train/Test แบบแยกกัน)
    """
    if outlier_rules is None:
        outlier_rules = {}
        
    for column_name, rule in outlier_rules.items():
        if column_name in features_train.columns:
            strategy = rule.get("strategy")
            lower = rule.get("lower")
            upper = rule.get("upper")
            
            if strategy == "clip":
                features_train[column_name] = features_train[column_name].clip(lower=lower, upper=upper)
                features_test[column_name] = features_test[column_name].clip(lower=lower, upper=upper)
                
    return features_train, features_test