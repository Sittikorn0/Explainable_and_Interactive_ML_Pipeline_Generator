# Libraries
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler, MinMaxScaler, RobustScaler

def _get_scaler(method: str):
    return {
        "standard_scaler": StandardScaler(),
        "minmax_scaler":   MinMaxScaler(),
        "robust_scaler":   RobustScaler(),
    }.get(method)


def scale_data(features_train: pd.DataFrame, features_test: pd.DataFrame,
               scaling_decisions: dict | str) -> tuple:
    """
    ปรับขนาดข้อมูล (Scaling) ป้องกัน Data Leakage โดย Fit บน features_train เท่านั้น

    scaling_decisions:
        dict  → {col_name: method}  วิเคราะห์ต่อคอลัมน์
        str   → method เดียวสำหรับทุก numeric column (backward compat)
    """
    # ── backward compat: ถ้าเป็น string ให้แปลงเป็น dict ก่อน ──
    if isinstance(scaling_decisions, str):
        method = scaling_decisions
        if method == "no_scaling":
            return features_train, features_test
        numeric_cols = features_train.select_dtypes(include="number").columns.tolist()
        scaling_decisions = {col: method for col in numeric_cols}

    if not scaling_decisions:
        return features_train, features_test

    features_train = features_train.copy()
    features_test  = features_test.copy()

    # กรณี log_transform: ทำ log1p ก่อน แล้วตาม standard_scaler
    log_cols = [col for col, m in scaling_decisions.items()
                if m == "log_transform" and col in features_train.columns]
    if log_cols:
        for col in log_cols:
            if features_train[col].min() >= 0:
                features_train[col] = np.log1p(features_train[col])
                features_test[col]  = np.log1p(np.maximum(features_test[col], 0))
        # ตาม standard_scaler สำหรับ log cols
        std_scaler = StandardScaler()
        features_train[log_cols] = std_scaler.fit_transform(features_train[log_cols])
        features_test[log_cols]  = std_scaler.transform(features_test[log_cols])

    # จัดกลุ่ม columns ตาม method (ไม่รวม log_transform และ no_scaling)
    from collections import defaultdict
    by_method: dict[str, list[str]] = defaultdict(list)
    for col, method in scaling_decisions.items():
        if method in ("log_transform", "no_scaling"):
            continue
        if col in features_train.columns:
            by_method[method].append(col)

    for method, cols in by_method.items():
        scaler = _get_scaler(method)
        if scaler is None or not cols:
            continue
        features_train[cols] = scaler.fit_transform(features_train[cols])
        features_test[cols]  = scaler.transform(features_test[cols])

    return features_train, features_test
