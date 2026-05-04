"""
ml_process/transformation/transformer.py
Apply transformations ตาม decisions ที่ user เลือก

หลักการป้องกัน Data Leakage:
- Encoding ทำได้ที่นี่ (ไม่ขึ้นกับ distribution ของ test set)
- Scaling ไม่ทำที่นี่ — บันทึกแค่ method ไว้
  แล้วให้ preprocess.py ทำหลัง train/test split เท่านั้น
"""

import numpy as np
import pandas as pd
from sklearn.preprocessing import LabelEncoder
from ml_process.features.data_analyzer import detect_task


def apply_encoding(df: pd.DataFrame, decisions: dict, target_col: str) -> pd.DataFrame:
    """
    Apply encoding ตาม decisions dict
    decisions format: {"col_name": "one_hot_encoding" | "label_encoding" | "ordinal_encoding" | "drop_column"}
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
        elif method == "ordinal_encoding":
            sorted_cats = sorted(result[col].dropna().astype(str).unique())
            cat_to_int  = {cat: i for i, cat in enumerate(sorted_cats)}
            result[col] = result[col].astype(str).map(cat_to_int)

    return result


def apply_feature_selection(df: pd.DataFrame,
                             drop_cols: list,
                             target_col: str) -> pd.DataFrame:
    """ตัด columns ที่ user เลือก ป้องกันตัด target โดยไม่ตั้งใจ"""
    safe_drop = [c for c in drop_cols if c != target_col and c in df.columns]
    return df.drop(columns=safe_drop)


def apply_all(df: pd.DataFrame,
              encoding_decisions: dict,
              scaling_method: str,
              drop_cols: list,
              target_col: str) -> tuple:
    """
    Apply transformations ตามลำดับ:
      1. Feature Selection
      2. Target Sanitization
      3. Encoding Features
      4. Target Encoding (Classification)
      ❌ Scaling — ไม่ทำที่นี่ เพื่อป้องกัน Data Leakage
    """
    result = df.copy()

    # 1. Feature Selection
    result = apply_feature_selection(result, drop_cols, target_col)

    # ตรวจสอบว่ายังมี feature เหลืออยู่อย่างน้อย 1 column (นอกจาก target)
    non_target_cols = [c for c in result.columns if c != target_col]
    if len(non_target_cols) == 0:
        raise ValueError(
            "ไม่สามารถ apply transformation ได้ เพราะ Feature Selection ตัด feature ออกจนหมด "
            "— ต้องเหลือ feature อย่างน้อย 1 column (นอกจาก target)"
        )

    # 2. Target Sanitization (แก้ชนิดข้อมูลถ้าเป็นตัวเลขที่เก็บเป็น String)
    if result[target_col].dtype == object:
        converted = pd.to_numeric(result[target_col], errors="coerce")
        if converted.notna().all():
            result[target_col] = converted

    # 3. Task Detection (ใช้กลางสำหรับระบุประเภทงาน)
    task_type = detect_task(result, target_col)

    # 4. Encoding features
    result = apply_encoding(result, encoding_decisions, target_col)

    # 5. Target Encoding (เฉพาะ Classification ถ้ายังเป็น categorical)
    if task_type == "classification" and (
        result[target_col].dtype == object or result[target_col].dtype.name == "category"
    ):
        le = LabelEncoder()
        result[target_col] = le.fit_transform(result[target_col].astype(str))

    # ❌ ไม่ scale ที่นี่ — preprocess.py จะทำให้หลัง split
    
    summary = {
        "original_rows":  df.shape[0],
        "original_cols":  df.shape[1],
        "dropped_cols":   len(drop_cols),
        "encoded_cols":   len(encoding_decisions),
        "final_cols":     result.shape[1],
        "scaling_method": scaling_method,
        "task_type":      task_type,
    }

    return result, summary