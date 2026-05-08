"""
ml_process/features/data_transformer.py
Apply transformations ตาม decisions ที่ user เลือก

หลักการป้องกัน Data Leakage:
- Encoding ทำได้ที่นี่ (ไม่ขึ้นกับ distribution ของ test set)
- Scaling ไม่ทำที่นี่ — บันทึกแค่ method ไว้
  แล้วให้ preprocess.py ทำหลัง train/test split เท่านั้น
"""

import numpy as np
import pandas as pd
from sklearn.preprocessing import LabelEncoder
from ml_process.logic.data_analyzer import detect_task

def apply_encoding(dataset: pd.DataFrame, encoding_decisions: dict, target_column: str) -> pd.DataFrame:
    """
    Apply encoding ตาม decisions dict
    decisions format: {"col_name": "one_hot_encoding" | "label_encoding" | "ordinal_encoding" | "drop_column"}
    """
    transformed_dataset = dataset.copy()

    for column_name, method in encoding_decisions.items():
        if column_name not in transformed_dataset.columns or column_name == target_column:
            continue

        if method == "drop_column":
            transformed_dataset = transformed_dataset.drop(columns=[column_name])
            
        elif method == "one_hot_encoding":
            dummies = pd.get_dummies(transformed_dataset[column_name], prefix=column_name, drop_first=True, dtype=int)
            transformed_dataset = pd.concat([transformed_dataset.drop(columns=[column_name]), dummies], axis=1)
            
        elif method == "label_encoding":
            label_encoder = LabelEncoder()
            transformed_dataset[column_name] = label_encoder.fit_transform(transformed_dataset[column_name].astype(str))
            
        elif method == "ordinal_encoding":
            sorted_categories = sorted(transformed_dataset[column_name].dropna().astype(str).unique())
            category_to_int = {category: index for index, category in enumerate(sorted_categories)}
            transformed_dataset[column_name] = transformed_dataset[column_name].astype(str).map(category_to_int)

    return transformed_dataset


def apply_feature_selection(dataset: pd.DataFrame, columns_to_drop: list, target_column: str) -> pd.DataFrame:
    """ตัด columns ที่ user เลือก ป้องกันตัด target โดยไม่ตั้งใจ"""
    safe_columns_to_drop = [column for column in columns_to_drop if column != target_column and column in dataset.columns]
    return dataset.drop(columns=safe_columns_to_drop)


def apply_all(dataset: pd.DataFrame, encoding_decisions: dict, scaling_method: str, columns_to_drop: list, target_column: str) -> tuple:
    """
    Apply transformations ตามลำดับ:
      - Feature Selection
      - Target Sanitization
      - Encoding Features
      - Target Encoding (Classification)
      ❌ Scaling — ไม่ทำที่นี่ เพื่อป้องกัน Data Leakage
    """
    transformed_dataset = dataset.copy()

    # Feature Selection
    transformed_dataset = apply_feature_selection(transformed_dataset, columns_to_drop, target_column)

    # ตรวจสอบว่ายังมี feature เหลืออยู่อย่างน้อย 1 column (นอกจาก target)
    non_target_columns = [column for column in transformed_dataset.columns if column != target_column]
    if len(non_target_columns) == 0:
        raise ValueError(
            "ไม่สามารถ apply transformation ได้ เพราะ Feature Selection ตัด feature ออกจนหมด "
            "— ต้องเหลือ feature อย่างน้อย 1 column (นอกจาก target)"
        )

    # Target Sanitization (แก้ชนิดข้อมูลถ้าเป็นตัวเลขที่เก็บเป็น String)
    if transformed_dataset[target_column].dtype == object:
        converted_target = pd.to_numeric(transformed_dataset[target_column], errors="coerce")
        if converted_target.notna().all():
            transformed_dataset[target_column] = converted_target

    # Task Detection (ใช้กลางสำหรับระบุประเภทงาน)
    task_type = detect_task(transformed_dataset, target_column)

    # Encoding features
    transformed_dataset = apply_encoding(transformed_dataset, encoding_decisions, target_column)

    # Target Encoding (เฉพาะ Classification ถ้ายังเป็น categorical)
    if task_type == "classification" and (
        transformed_dataset[target_column].dtype == object or transformed_dataset[target_column].dtype.name == "category"
    ):
        label_encoder = LabelEncoder()
        transformed_dataset[target_column] = label_encoder.fit_transform(transformed_dataset[target_column].astype(str))

    # ❌ ไม่ scale ที่นี่ — preprocess.py จะทำให้หลัง split
    
    transformation_summary = {
        "original_rows":  dataset.shape[0],
        "original_cols":  dataset.shape[1],
        "dropped_cols":   len(columns_to_drop),
        "encoded_cols":   len(encoding_decisions),
        "final_cols":     transformed_dataset.shape[1],
        "scaling_method": scaling_method,
        "task_type":      task_type,
    }

    return transformed_dataset, transformation_summary