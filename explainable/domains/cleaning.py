"""
explainable/domains/cleaning.py
High-level API สำหรับ cleaning decisions — ใช้ rule engine ข้างใน
"""
from __future__ import annotations
import pandas as pd
from scipy.stats import skew as _skew
from explainable.knowledge_base.engine import suggest, explain_all


# ── Dtype mapping ─────────────────────────────────────────────────────────────

def _map_dtype(series: pd.Series) -> str:
    if pd.api.types.is_bool_dtype(series):
        return "bool"
    if pd.api.types.is_datetime64_any_dtype(series):
        return "datetime"
    if pd.api.types.is_integer_dtype(series):
        return "int"
    if pd.api.types.is_float_dtype(series):
        return "float"
    return "string"


# ── Missing Value ─────────────────────────────────────────────────────────────

def suggest_missing(series: pd.Series) -> dict:
    """
    คืน suggestion dict สำหรับ missing value strategy ของ column นี้
    {action, rule_id, explanation, reference, confidence}
    """
    n_total   = len(series)
    n_missing = int(series.isnull().sum())
    miss_pct  = n_missing / n_total if n_total > 0 else 0.0
    dtype     = _map_dtype(series)

    skewness_abs = 0.0
    if dtype in ("float", "int"):
        clean = series.dropna()
        if len(clean) >= 3:
            skewness_abs = abs(float(_skew(clean)))

    facts = {
        "dtype":        dtype,
        "missing_pct":  miss_pct,
        "skewness_abs": skewness_abs,
    }
    result = suggest("missing_value", facts)
    if result is None:
        result = {
            "action":      "most_frequent",
            "rule_id":     "MISS_FALLBACK",
            "explanation": "ไม่มี rule ที่ match — ใช้ Most Frequent เป็น default",
            "reference":   "Topic 7 — Missing Data",
            "confidence":  0.5,
        }
    result["facts"] = facts
    return result


def explain_missing(series: pd.Series) -> list[dict]:
    """คืน list ของทุก rule ที่ match สำหรับ educational display"""
    n_total   = len(series)
    n_missing = int(series.isnull().sum())
    dtype     = _map_dtype(series)
    miss_pct  = n_missing / n_total if n_total > 0 else 0.0
    skewness_abs = 0.0
    if dtype in ("float", "int"):
        clean = series.dropna()
        if len(clean) >= 3:
            skewness_abs = abs(float(_skew(clean)))
    return explain_all("missing_value", {
        "dtype": dtype, "missing_pct": miss_pct, "skewness_abs": skewness_abs,
    })


# ── Outlier ───────────────────────────────────────────────────────────────────

def suggest_outlier(series: pd.Series, outlier_count: int) -> dict:
    """
    คืน suggestion dict สำหรับ outlier strategy
    {action, rule_id, explanation, reference, confidence}
    """
    clean = series.dropna()
    n     = len(clean)

    skewness_abs = abs(float(_skew(clean))) if n >= 3 else 0.0
    outlier_pct  = outlier_count / n * 100 if n > 0 else 0.0

    facts = {"skewness_abs": skewness_abs, "outlier_pct": outlier_pct}
    result = suggest("outlier", facts)
    if result is None:
        result = {
            "action":      "clip",
            "rule_id":     "OUT_FALLBACK",
            "explanation": "ใช้ Clip เป็น default — ปลอดภัยกว่าการ Drop Rows",
            "reference":   "Topic 8 — Outlier Detection",
            "confidence":  0.7,
        }
    result["facts"] = facts
    return result
