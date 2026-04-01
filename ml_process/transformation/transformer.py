"""
ml_process/transformation/transformer.py
Apply transformations ตาม decisions ที่ user เลือก
ไม่มี Streamlit → test ได้อิสระ
"""
import sys
from pathlib import Path
_ROOT = Path(__file__).resolve().parent.parent.parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

import numpy as np
import pandas as pd
from sklearn.preprocessing import LabelEncoder, StandardScaler, MinMaxScaler, RobustScaler


def apply_encoding(df: pd.DataFrame, decisions: dict, target_col: str) -> pd.DataFrame:
    """
    Apply encoding ตาม decisions dict
    decisions format: {"col_name": "one_hot_encoding" | "label_encoding" | "drop_column"}
    """
    result = df.copy()

    for col, method in decisions.items():
        if col not in result.columns or col == target_col:
            continue

        if method == "drop_column":
            result = result.drop(columns=[col])

        elif method == "one_hot_encoding":
            dummies = pd.get_dummies(result[col], prefix=col, drop_first=True, dtype=int)
            result  = pd.concat([result.drop(columns=[col]), dummies], axis=1)

        elif method == "label_encoding":
            le = LabelEncoder()
            result[col] = le.fit_transform(result[col].astype(str))

    return result


def apply_scaling(df: pd.DataFrame, method: str, target_col: str) -> pd.DataFrame:
    """
    Apply scaling ตาม method ที่เลือก
    เฉพาะ numeric columns ที่ไม่ใช่ target
    """
    if method == "no_scaling":
        return df

    result   = df.copy()
    num_cols = [
        c for c in result.columns
        if c != target_col and pd.api.types.is_numeric_dtype(result[c])
    ]

    if not num_cols:
        return result

    scaler_map = {
        "standard_scaler": StandardScaler(),
        "minmax_scaler":   MinMaxScaler(),
        "robust_scaler":   RobustScaler(),
    }
    scaler = scaler_map.get(method)
    if scaler:
        result[num_cols] = scaler.fit_transform(result[num_cols])

    return result


def apply_feature_selection(df: pd.DataFrame,
                             drop_cols: list[str],
                             target_col: str) -> pd.DataFrame:
    """
    Drop columns ที่ user เลือกให้ตัดออก
    ป้องกันการตัด target column โดยไม่ตั้งใจ
    """
    safe_drop = [c for c in drop_cols if c != target_col and c in df.columns]
    return df.drop(columns=safe_drop)


def apply_all(df: pd.DataFrame,
              encoding_decisions: dict,
              scaling_method: str,
              drop_cols: list[str],
              target_col: str) -> tuple[pd.DataFrame, dict]:
    """
    Apply ทุก transformation ตามลำดับที่ถูกต้อง:
    1. Feature Selection (ตัดคอลัมน์ก่อน)
    2. Encoding
    3. Scaling

    Returns: (transformed_df, summary)
    """
    original_shape = df.shape
    result = df.copy()

    # 1. Feature Selection
    result = apply_feature_selection(result, drop_cols, target_col)
    after_fs_shape = result.shape

    # 2. Encoding
    result = apply_encoding(result, encoding_decisions, target_col)
    after_enc_shape = result.shape

    # 3. Scaling
    result = apply_scaling(result, scaling_method, target_col)

    summary = {
        "original_rows":    original_shape[0],
        "original_cols":    original_shape[1],
        "dropped_cols":     len(drop_cols),
        "encoded_cols":     len(encoding_decisions),
        "final_cols":       result.shape[1],
        "scaling_method":   scaling_method,
    }

    return result, summary