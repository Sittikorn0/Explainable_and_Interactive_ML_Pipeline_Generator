"""
explainable/domains/transformation.py
High-level API สำหรับ transformation decisions
"""
from __future__ import annotations
import pandas as pd
from scipy.stats import skew as _skew
from explainable.knowledge_base.engine import suggest, explain_all


# ── Encoding ──────────────────────────────────────────────────────────────────

def suggest_encoding(series: pd.Series, n_rows: int) -> dict:
    """
    คืน suggestion dict สำหรับ encoding method ของ categorical column นี้
    """
    cardinality       = int(series.nunique())
    cardinality_ratio = cardinality / n_rows if n_rows > 0 else 0.0

    facts = {"cardinality": cardinality, "cardinality_ratio": cardinality_ratio}
    result = suggest("encoding", facts)
    if result is None:
        result = {
            "action":      "label_encoding",
            "rule_id":     "ENC_FALLBACK",
            "explanation": "ไม่มี rule ที่ match — ใช้ Label Encoding เป็น default",
            "reference":   "Topic 9 — Data Transformation",
            "confidence":  0.6,
        }
    result["facts"] = facts
    return result


# ── Scaling ───────────────────────────────────────────────────────────────────

def suggest_scaling(df: pd.DataFrame, target_col: str) -> dict:
    """
    วิเคราะห์ numeric columns ทั้งหมดแล้วแนะนำ scaling method
    """
    num_cols = [
        c for c in df.columns
        if c != target_col and pd.api.types.is_numeric_dtype(df[c])
    ]

    if not num_cols:
        return {
            "action":      "no_scaling",
            "rule_id":     "SCL_001",
            "explanation": "ไม่มี numeric feature — ไม่จำเป็นต้องทำ scaling",
            "reference":   "Topic 9 — Data Transformation",
            "confidence":  1.0,
            "facts":       {"no_numeric": True},
        }

    has_outliers   = False
    is_skewed      = False
    has_heavy_skew = False

    for col in num_cols:
        clean = df[col].dropna()
        if len(clean) < 3:
            continue
        col_skew = abs(float(_skew(clean)))
        q1, q3  = clean.quantile(0.25), clean.quantile(0.75)
        iqr     = q3 - q1
        n_out   = int(((clean < q1 - 1.5 * iqr) | (clean > q3 + 1.5 * iqr)).sum())
        if n_out / len(clean) > 0.05:
            has_outliers = True
        if col_skew > 1:
            is_skewed = True
        if col_skew > 2:
            has_heavy_skew = True

    facts = {
        "no_numeric":     False,
        "has_outliers":   has_outliers,
        "is_skewed":      is_skewed,
        "has_heavy_skew": has_heavy_skew,
    }
    result = suggest("scaling", facts)
    if result is None:
        result = {
            "action":      "standard_scaler",
            "rule_id":     "SCL_FALLBACK",
            "explanation": "ใช้ Standard Scaler เป็น default",
            "reference":   "Topic 9 — Data Transformation",
            "confidence":  0.7,
        }
    result["facts"] = facts
    return result
