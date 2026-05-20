# Libraries
import pandas as pd

# Logic Import
from backend.core.insight.reasoning_engine.engine import *

# Functions
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
            confidence  = rule_result.get("confidence", 0.8)
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