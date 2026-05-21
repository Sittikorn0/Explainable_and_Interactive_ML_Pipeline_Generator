# Libraries
import pandas as pd

# Logic Import
from backend.core.cleaning.statistic import get_outlier_bounds
from backend.core.insight.reasoning_engine.engine import suggest

# Functions
def analyze_scaling(dataset: pd.DataFrame, target_column: str) -> dict:
    """
    วิเคราะห์คอลัมน์ที่เป็นตัวเลข (Numeric) เพื่อหาวิธี Scaling ที่เหมาะสม

    Returns: dictionary ที่ประกอบด้วยวิธีแนะนำ เหตุผล สถิติของแต่ละคอลัมน์
    """
    numeric_columns = [
        column_name for column_name in dataset.columns
        if column_name != target_column and pd.api.types.is_numeric_dtype(dataset[column_name])
    ]

    if not numeric_columns:
        return {
            "recommended": "no_scaling",
            "options":     ["no_scaling"],
            "reason":      "ไม่มีคอลัมน์ประเภทตัวเลขที่จำเป็นต้องทำ Scaling",
            "column_stats": [],
            "has_outliers": False,
            "is_skewed":    False,
            "has_heavy_skew": False,
            "outlier_cols":   [],
            "skewed_cols":    [],
            "heavy_skew_cols": [],
        }

    column_statistics = []
    columns_with_outliers = []
    columns_skewed = []
    columns_heavy_skewed = []

    for column_name in numeric_columns:
        feature_series = dataset[column_name].dropna()
        if len(feature_series) == 0:
            continue

        outlier_bounds = get_outlier_bounds(feature_series)
        skewness_value = outlier_bounds["skewness"]
        
        if outlier_bounds["method"] == "N/A":
            num_outliers = 0
        else:
            lower_bound = outlier_bounds["lower"]
            upper_bound = outlier_bounds["upper"]
            num_outliers = int(((feature_series < lower_bound) | (feature_series > upper_bound)).sum())
            
        outlier_percentage = (num_outliers / len(feature_series)) * 100

        column_statistics.append({
            "col":     column_name,
            "min":     round(float(feature_series.min()), 3),
            "max":     round(float(feature_series.max()), 3),
            "mean":    round(float(feature_series.mean()), 3),
            "std":     round(float(feature_series.std()), 3),
            "skew":    round(skewness_value, 3),
            "outlier_pct": round(outlier_percentage, 1),
        })

        if outlier_percentage > 5:
            columns_with_outliers.append(column_name)
        if abs(skewness_value) > 1:
            columns_skewed.append(column_name)
        if abs(skewness_value) > 2:
            columns_heavy_skewed.append(column_name)

    has_outliers    = len(columns_with_outliers) > 0
    is_skewed       = len(columns_skewed) > 0
    has_heavy_skew  = len(columns_heavy_skewed) > 0

    # ── ใช้ Rule Engine แนะนำ Scaling Method ──
    facts = {
        "no_numeric":     False,
        "has_outliers":   has_outliers,
        "is_skewed":      is_skewed,
        "has_heavy_skew": has_heavy_skew,
    }
    rule_result = suggest("scaling", facts)

    _scl_action_map = {
        "robust_scaler":   "robust_scaler",
        "log_transform":   "log_transform",
        "minmax_scaler":   "minmax_scaler",
        "standard_scaler": "standard_scaler",
        "no_scaling":      "no_scaling",
    }
    if rule_result:
        recommended_method = _scl_action_map.get(rule_result["action"], "standard_scaler")
        reason     = rule_result["explanation"]
        rule_id    = rule_result["rule_id"]
    else:
        recommended_method = "standard_scaler"
        reason     = "ใช้ Standard Scaler เป็น default"
        rule_id    = "SCL_FALLBACK"

    return {
        "recommended":    recommended_method,
        "options":        ["log_transform", "standard_scaler", "minmax_scaler", "robust_scaler", "no_scaling"],
        "reason":         reason,
        "rule_id":        rule_id,
        "column_stats":   column_statistics,
        "has_outliers":   has_outliers,
        "is_skewed":      is_skewed,
        "has_heavy_skew": has_heavy_skew,
        "outlier_cols":   columns_with_outliers,
        "skewed_cols":    columns_skewed,
        "heavy_skew_cols": columns_heavy_skewed,
    }