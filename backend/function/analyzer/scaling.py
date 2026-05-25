# Libraries
import pandas as pd

# Logic Import
from backend.core.cleaning.statistic import get_outlier_bounds
from backend.core.insight.reasoning_engine.engine import suggest

_SCL_ACTION_MAP = {
    "robust_scaler":   "robust_scaler",
    "log_transform":   "log_transform",
    "minmax_scaler":   "minmax_scaler",
    "standard_scaler": "standard_scaler",
    "no_scaling":      "no_scaling",
}

def _suggest_scaler_for_column(has_outliers: bool, is_skewed: bool, has_heavy_skew: bool) -> tuple[str, str, str]:
    """คืน (method, rule_id, reason) สำหรับ 1 column"""
    facts = {
        "no_numeric":     False,
        "has_outliers":   has_outliers,
        "is_skewed":      is_skewed,
        "has_heavy_skew": has_heavy_skew,
    }
    rule_result = suggest("scaling", facts)
    if rule_result:
        method  = _SCL_ACTION_MAP.get(rule_result["action"], "standard_scaler")
        rule_id = rule_result["rule_id"]
        reason  = rule_result["explanation"]
    else:
        method, rule_id, reason = "standard_scaler", "SCL_FALLBACK", "ใช้ Standard Scaler เป็น default"
    return method, rule_id, reason


# Functions
def analyze_scaling(dataset: pd.DataFrame, target_column: str) -> dict:
    """
    วิเคราะห์คอลัมน์ที่เป็นตัวเลข (Numeric) เพื่อหาวิธี Scaling ที่เหมาะสม
    แต่ละคอลัมน์ได้รับการวิเคราะห์แยกกันและมี method ของตัวเอง

    Returns: dictionary ที่ประกอบด้วย column_decisions (per-column) และ column_stats
    """
    numeric_columns = [
        column_name for column_name in dataset.columns
        if column_name != target_column and pd.api.types.is_numeric_dtype(dataset[column_name])
    ]

    if not numeric_columns:
        return {
            "column_decisions": [],
            "options":          ["no_scaling"],
            "column_stats":     [],
            "has_outliers":     False,
            "is_skewed":        False,
            "has_heavy_skew":   False,
            "outlier_cols":     [],
            "skewed_cols":      [],
            "heavy_skew_cols":  [],
        }

    column_decisions   = []
    column_statistics  = []
    columns_with_outliers = []
    columns_skewed        = []
    columns_heavy_skewed  = []

    for column_name in numeric_columns:
        feature_series = dataset[column_name].dropna()
        if len(feature_series) == 0:
            continue

        outlier_bounds   = get_outlier_bounds(feature_series)
        skewness_value   = outlier_bounds["skewness"]

        if outlier_bounds["method"] == "N/A":
            num_outliers = 0
        else:
            lower_bound  = outlier_bounds["lower"]
            upper_bound  = outlier_bounds["upper"]
            num_outliers = int(((feature_series < lower_bound) | (feature_series > upper_bound)).sum())

        outlier_pct  = (num_outliers / len(feature_series)) * 100
        has_outliers = outlier_pct > 5
        is_skewed    = abs(skewness_value) > 1
        has_heavy_skew = abs(skewness_value) > 2

        method, rule_id, reason = _suggest_scaler_for_column(has_outliers, is_skewed, has_heavy_skew)

        stats = {
            "col":         column_name,
            "min":         round(float(feature_series.min()), 3),
            "max":         round(float(feature_series.max()), 3),
            "mean":        round(float(feature_series.mean()), 3),
            "std":         round(float(feature_series.std()), 3),
            "skew":        round(skewness_value, 3),
            "outlier_pct": round(outlier_pct, 1),
        }
        column_statistics.append(stats)

        column_decisions.append({
            **stats,
            "recommended": method,
            "rule_id":     rule_id,
            "reason":      reason,
        })

        if has_outliers:     columns_with_outliers.append(column_name)
        if is_skewed:        columns_skewed.append(column_name)
        if has_heavy_skew:   columns_heavy_skewed.append(column_name)

    return {
        "column_decisions": column_decisions,
        "options":          ["log_transform", "standard_scaler", "minmax_scaler", "robust_scaler", "no_scaling"],
        "column_stats":     column_statistics,
        "has_outliers":     len(columns_with_outliers) > 0,
        "is_skewed":        len(columns_skewed) > 0,
        "has_heavy_skew":   len(columns_heavy_skewed) > 0,
        "outlier_cols":     columns_with_outliers,
        "skewed_cols":      columns_skewed,
        "heavy_skew_cols":  columns_heavy_skewed,
    }
