# Libraries
import pandas as pd

# Logic Import
from backend.core.insight.reasoning_engine.engine import *

# Known ordinal pattern sets (lowercase) — if ALL unique values of a column fall in one set,
# the column is considered ordinal
_ORDINAL_PATTERNS: list[set[str]] = [
    {"low", "medium", "high"},
    {"low", "mid", "high"},
    {"low", "medium", "high", "very high"},
    {"very low", "low", "medium", "high", "very high"},
    {"small", "medium", "large"},
    {"small", "medium", "large", "extra large"},
    {"xs", "s", "m", "l", "xl"},
    {"xs", "s", "m", "l", "xl", "xxl"},
    {"none", "low", "medium", "high"},
    {"none", "mild", "moderate", "severe"},
    {"poor", "fair", "good", "very good", "excellent"},
    {"strongly disagree", "disagree", "neutral", "agree", "strongly agree"},
    {"never", "rarely", "sometimes", "often", "always"},
    {"beginner", "intermediate", "advanced", "expert"},
    {"junior", "mid", "senior"},
    {"junior", "middle", "senior"},
    {"bronze", "silver", "gold", "platinum"},
]

# ตรวจว่า unique values ตรงกับ pattern ordinal ที่รู้จัก ใช้ภายใน analyze_encoding
def _looks_ordinal(unique_values: list) -> bool:
    lower_values = {str(v).strip().lower() for v in unique_values}
    return any(lower_values == pattern for pattern in _ORDINAL_PATTERNS)

# วิเคราะห์ categorical columns ทั้งหมด แนะนำวิธี encoding ผ่าน Rule Engine คืน list[dict] ใช้ใน analyze_all และ transformation_page
def analyze_encoding(dataset: pd.DataFrame, target_column: str) -> list[dict]:
    analysis_results = []
    categorical_columns = [
        column_name for column_name in dataset.columns
        if column_name != target_column and dataset[column_name].dtype == object
    ]

    total_rows = len(dataset)

    for column_name in categorical_columns:
        num_unique_values = dataset[column_name].nunique()
        unique_to_row_ratio = num_unique_values / total_rows
        all_unique_values = list(dataset[column_name].dropna().unique())
        is_ordinal = _looks_ordinal(all_unique_values)

        facts = {
            "cardinality":       num_unique_values,
            "cardinality_ratio": unique_to_row_ratio,
            "looks_ordinal":     is_ordinal,
        }
        rule_result = suggest("encoding", facts)

        _enc_action_map = {
            "one_hot_encoding":  "one_hot_encoding",
            "label_encoding":    "label_encoding",
            "ordinal_encoding":  "ordinal_encoding",
            "drop_column":       "drop_column",
        }
        if rule_result:
            recommended_method = _enc_action_map.get(rule_result["action"], "label_encoding")
            reason      = rule_result["explanation"]
            rule_id     = rule_result["rule_id"]
        else:
            recommended_method = "label_encoding"
            reason      = f"มี {num_unique_values} categories  ใช้ Label Encoding เป็น default"
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
            "rule_id":     rule_id,
            "warning":     warning_message,
            "sample_values": list(dataset[column_name].dropna().unique()[:5]),
        })

    return analysis_results