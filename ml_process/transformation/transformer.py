"""
ml_process/transformation/transformer.py
Apply transformations ตาม decisions ที่ user เลือก

หลักการป้องกัน Data Leakage:
- One-hot encoding: ทำได้ที่นี่ (pd.get_dummies ไม่ขึ้นกับ statistics)
- Label encoding: ไม่ทำที่นี่ — บันทึก decisions ไว้ใน summary
  แล้วให้ preprocess.py fit บน X_train เท่านั้นหลัง split
- Scaling: ไม่ทำที่นี่ — บันทึกแค่ method ไว้
  แล้วให้ preprocess.py ทำหลัง train/test split เท่านั้น
"""
import sys
from pathlib import Path
_ROOT = Path(__file__).resolve().parent.parent.parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

import numpy as np
import pandas as pd


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
            # ไม่ fit ที่นี่ — บันทึกไว้ใน label_encoding_cols
            # preprocess.py จะ fit บน X_train เท่านั้นหลัง split
            pass

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
      2. Encoding
      ❌ Scaling — ไม่ทำที่นี่ เพื่อป้องกัน Data Leakage
         scaling_method ถูกบันทึกไว้ใน summary แล้วส่งให้ preprocess.py
         ทำหลัง train/test split แทน

    Returns: (transformed_df, summary)
    """
    result = df.copy()

    # 1. Feature Selection
    result = apply_feature_selection(result, drop_cols, target_col)

    # 2. Encoding
    result = apply_encoding(result, encoding_decisions, target_col)

    # ❌ ไม่ scale ที่นี่ — preprocess.py จะทำให้หลัง split

    # columns ที่ user เลือก label_encoding — preprocess.py จะ fit หลัง split
    label_encoding_cols = [
        col for col, method in encoding_decisions.items()
        if method == "label_encoding" and col in result.columns and col != target_col
    ]

    summary = {
        "original_rows":      df.shape[0],
        "original_cols":      df.shape[1],
        "dropped_cols":       len(drop_cols),
        "encoded_cols":       len(encoding_decisions),
        "final_cols":         result.shape[1],
        "scaling_method":     scaling_method,
        "label_encoding_cols": label_encoding_cols,  # ส่งให้ preprocess.py ทำหลัง split
    }

    return result, summary