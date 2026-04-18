"""
explainable/domains/modeling.py
High-level API สำหรับ model selection decisions
"""
from __future__ import annotations
import pandas as pd
from explainable.knowledge_base.engine import suggest, explain_all


def suggest_model_strategy(
    df: pd.DataFrame,
    target_col: str,
    task_type: str,
) -> list[dict]:
    """
    คืน list ของ suggestions ที่เกี่ยวข้องกับ dataset นี้
    (อาจมีหลาย rule ที่ match พร้อมกัน เช่น เล็ก + imbalanced)
    """
    n_samples = len(df)
    n_features = df.shape[1] - 1  # ไม่นับ target

    facts_base = {
        "task_type":   task_type,
        "n_samples":   n_samples,
        "n_features":  n_features,
    }

    # คำนวณ class imbalance สำหรับ classification
    if task_type == "classification":
        counts = df[target_col].value_counts()
        if len(counts) >= 2:
            ratio = float(counts.iloc[0] / counts.iloc[-1])
        else:
            ratio = 1.0
        facts_base["class_imbalance_ratio"] = ratio
        facts_base["n_classes"] = int(df[target_col].nunique())
    else:
        facts_base["class_imbalance_ratio"] = 1.0
        facts_base["n_classes"] = 0

    return explain_all("model_selection", facts_base)


def get_dataset_profile(df: pd.DataFrame, target_col: str, task_type: str) -> dict:
    """
    คืน profile summary ของ dataset สำหรับแสดงใน trace log และ explainable page
    """
    n_samples  = len(df)
    n_features = df.shape[1] - 1
    n_numeric  = df.drop(columns=[target_col]).select_dtypes(include="number").shape[1]
    n_categ    = df.drop(columns=[target_col]).select_dtypes(include=["object", "category"]).shape[1]
    n_missing  = int(df.isnull().sum().sum())

    profile = {
        "n_samples":   n_samples,
        "n_features":  n_features,
        "n_numeric":   n_numeric,
        "n_categ":     n_categ,
        "n_missing":   n_missing,
        "task_type":   task_type,
    }

    if task_type == "classification":
        counts = df[target_col].value_counts()
        profile["n_classes"]             = int(df[target_col].nunique())
        profile["class_imbalance_ratio"] = round(float(counts.iloc[0] / counts.iloc[-1]), 2) if len(counts) >= 2 else 1.0

    return profile
