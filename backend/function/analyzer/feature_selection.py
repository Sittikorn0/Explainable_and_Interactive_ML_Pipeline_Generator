# Libraries
import pandas as pd

# Logic Import
from backend.core.insight.reasoning_engine.engine import suggest

# Functions
# วิเคราะห์ high-correlation และ low-variance features ผ่าน Rule Engine คืน dict ใช้ใน analyze_all และ transformation_page
def analyze_feature_selection(dataset: pd.DataFrame, target_column: str) -> dict:
    numeric_columns = [
        column_name for column_name in dataset.columns
        if column_name != target_column and pd.api.types.is_numeric_dtype(dataset[column_name])
    ]

    columns_to_drop_corr = []
    columns_to_drop_var = []
    corr_rule_id = ""
    var_rule_id = ""

    if len(numeric_columns) >= 2:
        correlation_matrix = dataset[numeric_columns].corr().abs()
        processed_pairs = set()

        for i, column_a in enumerate(numeric_columns):
            for column_b in numeric_columns[i+1:]:
                pair = tuple(sorted([column_a, column_b]))
                if pair in processed_pairs:
                    continue
                processed_pairs.add(pair)

                corr_value = float(correlation_matrix.loc[column_a, column_b])
                rule_result = suggest("feature_selection", {"corr_value": corr_value})
                if rule_result and rule_result["action"] == "drop_high_correlation":
                    columns_to_drop_corr.append({
                        "col_a": column_a,
                        "col_b": column_b,
                        "corr": round(corr_value, 3),
                        "drop": column_b,
                        "rule_id": rule_result["rule_id"],
                    })
                    corr_rule_id = rule_result["rule_id"]

    for column_name in numeric_columns:
        feature_series = dataset[column_name].dropna()
        if len(feature_series) == 0:
            continue

        coefficient_of_variation = feature_series.std() / (feature_series.mean() + 1e-9)
        cv_abs = abs(float(coefficient_of_variation))
        rule_result = suggest("feature_selection", {"cv_abs": cv_abs})
        if rule_result and rule_result["action"] == "drop_low_variance":
            columns_to_drop_var.append({
                "col": column_name,
                "std": round(float(feature_series.std()), 6),
                "cv": round(float(coefficient_of_variation), 6),
                "rule_id": rule_result["rule_id"],
            })
            var_rule_id = rule_result["rule_id"]

    corr_rule = suggest("feature_selection", {"corr_value": 1.0})
    var_rule = suggest("feature_selection", {"cv_abs": 0.0})

    reason_for_corr_drop = corr_rule["explanation"] if corr_rule else (
        "คอลัมน์ที่มี Correlation สูงถือว่ามีข้อมูลซ้ำซ้อนกัน (Multicollinearity)"
    )
    reason_for_var_drop = var_rule["explanation"] if var_rule else (
        "คอลัมน์ที่มี Variance ต่ำมากไม่มีข้อมูลที่เป็นประโยชน์ต่อโมเดล"
    )

    return {
        "drop_high_corr": columns_to_drop_corr,
        "drop_low_var": columns_to_drop_var,
        "reason_corr": reason_for_corr_drop,
        "reason_var": reason_for_var_drop,
        "corr_rule_id": corr_rule_id,
        "var_rule_id": var_rule_id,
    }
