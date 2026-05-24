# Libraries
import numpy as np
import pandas as pd

# เติม missing values fit บน train เท่านั้น แปลง bool→int และล้าง inf ใช้ใน preprocess
def clean_fit_transform(features_train: pd.DataFrame, features_test: pd.DataFrame, missing_rules: dict = None) -> tuple:
    if missing_rules is None:
        missing_rules = {}

    for dataset_split in [features_train, features_test]:
        boolean_columns = dataset_split.select_dtypes(include="bool").columns
        for col in boolean_columns:
            dataset_split[col] = dataset_split[col].astype(int)
        dataset_split.replace([np.inf, -np.inf], 0, inplace=True)

    fill_values_dict = {}
    for column_name in features_train.columns:
        if features_train[column_name].isna().any() or features_test[column_name].isna().any():
            strategy = missing_rules.get(column_name, "median" if pd.api.types.is_numeric_dtype(features_train[column_name]) else "most frequent")
            
            if strategy == "mean":
                fill_values_dict[column_name] = features_train[column_name].mean()
            elif strategy == "median":
                median_val = features_train[column_name].median()
                if features_train[column_name].dropna().mod(1).eq(0).all():
                    median_val = round(median_val)
                fill_values_dict[column_name] = median_val
            elif strategy == "most frequent":
                modes = features_train[column_name].mode()
                fill_values_dict[column_name] = modes.iloc[0] if not modes.empty else 0
            elif strategy in ("forward fill", "backward fill"):
                # ffill/bfill ไม่มีความหมายหลัง random split  fallback ใช้ median/mode (fit on train)
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

    if fill_values_dict:
        features_train = features_train.fillna(fill_values_dict)
        features_test = features_test.fillna(fill_values_dict)

    return features_train, features_test

# clip outliers ตาม rules ที่ user ตั้งไว้ใน UI fit บน train เท่านั้น ใช้ใน preprocess
def outlier_fit_transform(features_train: pd.DataFrame, features_test: pd.DataFrame, outlier_rules: dict = None) -> tuple:
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