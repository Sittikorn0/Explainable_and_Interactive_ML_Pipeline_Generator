import numpy as np
import pandas as pd
from scipy.stats import skew
from data_prepare.logic.statistics import get_outlier_bounds
from explainable.knowledge_base.engine import suggest

# Core Analyzer (Main Entry Point)
def analyze_all(dataset: pd.DataFrame, target_column: str) -> dict:
    """
    รัน Analysis ทั้งหมด (Encoding, Scaling, Feature Selection, Leakage)
    """
    from ml_process.logic.logic import analyze_leakage
    return {
        "encoding":          analyze_encoding(dataset, target_column),
        "scaling":           analyze_scaling(dataset, target_column),
        "feature_selection": analyze_feature_selection(dataset, target_column),
        "leakage":           analyze_leakage(dataset, target_column),
    }

# Task Detection
def detect_task(dataset: pd.DataFrame, target_column: str) -> str:
    """
    ตรวจสอบและตัดสินใจว่าข้อมูลนี้เป็นปัญหา Classification หรือ Regression

    เกณฑ์การตัดสินใจ:
    - ถ้าเป็น string/category จะเป็น Classification เสมอ
    - ถ้าเป็นตัวเลข และมีค่าไม่ซ้ำ ≤ 15 ค่า จะเป็น Classification (มองเป็นคลาสแยกส่วน)
    - ถ้าเป็นตัวเลข และมีค่าไม่ซ้ำ > 100 ค่า จะเป็น Regression เสมอ (มองเป็นค่าต่อเนื่อง)
    - ถ้าเป็นตัวเลข มีค่าไม่ซ้ำระหว่าง 16-100 ค่า จะตรวจสอบสัดส่วน (Ratio):
        - ถ้า Ratio ≥ 5% ของจำนวนข้อมูลทั้งหมด จะเป็น Regression
        - ถ้า Ratio < 5% จะเป็น Classification
    """
    target_series = dataset[target_column]
    
    if not pd.api.types.is_numeric_dtype(target_series):
        return "classification"
        
    num_unique_values = target_series.nunique()
    
    if num_unique_values <= 15:
        return "classification"
    if num_unique_values > 100:
        return "regression"
    
    # กรณี 16-100 unique values: ดูสัดส่วนเมื่อเทียบกับจำนวนแถวทั้งหมด
    unique_to_row_ratio = num_unique_values / len(target_series)
    
    if unique_to_row_ratio >= 0.05:
        return "regression"
    else:
        return "classification"

# Encoding Analysis
def analyze_encoding(dataset: pd.DataFrame, target_column: str) -> list[dict]:
    """
    วิเคราะห์คอลัมน์ที่เป็น Categorical (หมวดหมู่) เพื่อหา Encoding Method ที่เหมาะสมที่สุด

    Returns: list of dictionary ที่บรรจุรายละเอียดและคำแนะนำสำหรับแต่ละคอลัมน์
    """
    analysis_results = []
    
    # หาคอลัมน์ที่เป็นข้อความและไม่ใช่ target
    categorical_columns = [
        column_name for column_name in dataset.columns
        if column_name != target_column and dataset[column_name].dtype == object
    ]

    total_rows = len(dataset)

    for column_name in categorical_columns:
        num_unique_values = dataset[column_name].nunique()
        unique_to_row_ratio = num_unique_values / total_rows

        # ── ใช้ Rule Engine แนะนำ Encoding Method ──
        facts = {
            "cardinality":       num_unique_values,
            "cardinality_ratio": unique_to_row_ratio,
        }
        rule_result = suggest("encoding", facts)

        # Map rule action → encoding key
        _enc_action_map = {
            "one_hot_encoding": "one_hot_encoding",
            "label_encoding":   "label_encoding",
            "drop_column":      "drop_column",
        }
        if rule_result:
            recommended_method = _enc_action_map.get(rule_result["action"], "label_encoding")
            reason      = rule_result["explanation"]
            reference   = rule_result["reference"]
            confidence  = rule_result["confidence"]
            rule_id     = rule_result["rule_id"]
        else:
            recommended_method = "label_encoding"
            reason      = f"มี {num_unique_values} categories — ใช้ Label Encoding เป็น default"
            reference   = "Topic 9 — Data Transformation"
            confidence  = 0.6
            rule_id     = "ENC_FALLBACK"

        warning_message = None
        if recommended_method == "drop_column":
            warning_message = "คอลัมน์นี้อาจเป็นข้อมูลระบุตัวตน (ID) หรือข้อความอิสระจึงควรตัดออก"
        elif num_unique_values > 10 and recommended_method == "one_hot_encoding":
            warning_message = f"การใช้ One-hot จะทำให้โครงสร้างข้อมูลกว้างขึ้น {num_unique_values-1} คอลัมน์"
        elif num_unique_values > 20 and recommended_method == "label_encoding":
            warning_message = f"ข้อควรระวัง: High cardinality มีถึง {num_unique_values} ค่าที่ไม่ซ้ำกัน"

        analysis_results.append({
            "col":         column_name,
            "cardinality": num_unique_values,
            "recommended": recommended_method,
            "options":     ["one_hot_encoding", "label_encoding", "ordinal_encoding", "drop_column"],
            "reason":      reason,
            "reference":   reference,
            "confidence":  confidence,
            "rule_id":     rule_id,
            "warning":     warning_message,
            "sample_values": list(dataset[column_name].dropna().unique()[:5]),
        })

    return analysis_results

# Scaling Analysis
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
        reference  = rule_result["reference"]
        confidence = rule_result["confidence"]
        rule_id    = rule_result["rule_id"]
    else:
        recommended_method = "standard_scaler"
        reason     = "ใช้ Standard Scaler เป็น default"
        reference  = "Topic 9 — Data Transformation"
        confidence = 0.7
        rule_id    = "SCL_FALLBACK"

    return {
        "recommended":    recommended_method,
        "options":        ["log_transform", "standard_scaler", "minmax_scaler", "robust_scaler", "no_scaling"],
        "reason":         reason,
        "reference":      reference,
        "confidence":     confidence,
        "rule_id":        rule_id,
        "column_stats":   column_statistics,
        "has_outliers":   has_outliers,
        "is_skewed":      is_skewed,
        "has_heavy_skew": has_heavy_skew,
        "outlier_cols":   columns_with_outliers,
        "skewed_cols":    columns_skewed,
        "heavy_skew_cols": columns_heavy_skewed,
    }

# Feature Selection Analysis
def analyze_feature_selection(dataset: pd.DataFrame, target_column: str) -> dict:
    """
    วิเคราะห์ความซ้ำซ้อนและการกระจายตัวของข้อมูล เพื่อแนะนำ Feature ที่ควรตัดออก
    เช่น ข้อมูลที่แปรปรวนต่ำมาก (Low Variance) หรือมีความสัมพันธ์กันเองสูงมาก (High Correlation)
    """
    numeric_columns = [
        column_name for column_name in dataset.columns
        if column_name != target_column and pd.api.types.is_numeric_dtype(dataset[column_name])
    ]

    columns_to_drop_corr = []
    columns_to_drop_var = []

    # ตรวจสอบ High Correlation (Multicollinearity)
    if len(numeric_columns) >= 2:
        correlation_matrix = dataset[numeric_columns].corr().abs()
        processed_pairs = set()
        
        for i, column_a in enumerate(numeric_columns):
            for column_b in numeric_columns[i+1:]:
                pair = tuple(sorted([column_a, column_b]))
                if pair in processed_pairs:
                    continue
                processed_pairs.add(pair)
                
                corr_value = correlation_matrix.loc[column_a, column_b]
                if corr_value >= 0.85:
                    columns_to_drop_corr.append({
                        "col_a": column_a,
                        "col_b": column_b,
                        "corr":  round(float(corr_value), 3),
                        "drop":  column_b,   # แนะนำให้ตัดคอลัมน์ที่สองทิ้ง
                    })

    # ตรวจสอบ Low Variance (ไม่มีการเปลี่ยนแปลงข้อมูล)
    for column_name in numeric_columns:
        feature_series = dataset[column_name].dropna()
        if len(feature_series) == 0:
            continue
            
        coefficient_of_variation = feature_series.std() / (feature_series.mean() + 1e-9)
        if abs(coefficient_of_variation) < 0.01:
            columns_to_drop_var.append({
                "col": column_name,
                "std": round(float(feature_series.std()), 6),
                "cv":  round(float(coefficient_of_variation), 6),
            })

    reason_for_corr_drop = (
        "คอลัมน์ที่มี Correlation ≥ 0.85 เมื่อเทียบกับคอลัมน์อื่น "
        "ถือว่ามีข้อมูลซ้ำซ้อนกันมาก (Multicollinearity) การเก็บทั้งคู่ไว้จะทำให้โมเดล "
        "ให้น้ำหนักเกินความเป็นจริง และแปลผลได้ยาก"
    )
    reason_for_var_drop = (
        "คอลัมน์ที่มี Variance ต่ำมาก (Coefficient of Variation < 1%) ถือว่าไม่มีข้อมูลที่เป็นประโยชน์ "
        "เพราะค่าเกือบทั้งหมดเหมือนกัน โมเดลไม่สามารถเรียนรู้ Pattern จากข้อมูลที่ไม่มีความแตกต่างกันได้"
    )

    return {
        "drop_high_corr": columns_to_drop_corr,
        "drop_low_var":   columns_to_drop_var,
        "reason_corr":    reason_for_corr_drop,
        "reason_var":     reason_for_var_drop,
    }