import pandas as pd
from data_prepare.logic.data_type_detection import actual_type


def score_column(dataset: pd.DataFrame, column_name: str, column_index: int) -> tuple[float, list[str]]:
    """
    คำนวณ score ว่า column นี้น่าจะเป็น target แค่ไหน
    คืนค่า (score, [เหตุผล])
    """
    data_series = dataset[column_name]
    total_rows = len(data_series)
    unique_count = data_series.nunique()
    missing_count = int(data_series.isnull().sum())
    actual_data_type = actual_type(data_series)
    
    column_score = 0.0
    reason_list = []

    # ถ้าเป็น column สุดท้าย
    if column_index == len(dataset.columns) - 1:
        column_score += 1.0
        reason_list.append("เป็น column สุดท้าย (convention ทั่วไปของ ML datasets)")

    # ถ้า Binary (unique = 2)
    if unique_count == 2:
        column_score += 3.0
        reason_list.append(f"มีแค่ 2 ค่า Unique — ลักษณะของ Binary Classification target")

    # ถ้า Cardinality ต่ำ (unique ≤ 5% ของแถว และ ≤ 20)
    elif unique_count <= max(10, total_rows * 0.05) and actual_data_type in ("int", "string", "bool"):
        column_score += 2.0
        unique_percentage = unique_count / total_rows * 100
        reason_list.append(f"Unique เพียง {unique_count} ค่า ({unique_percentage:.1f}% ของข้อมูล) — เหมาะเป็น Classification target")

    # ถ้า unique สูงมาก (น่าจะเป็น ID หรือ continuous feature)
    if unique_count > total_rows * 0.9:
        column_score -= 3.0
        reason_list.append(f"Unique สูงมาก ({unique_count:,} ค่า) — อาจเป็น ID หรือ key ไม่ใช่ target")

    # ถ้ามี missing มาก
    if missing_count > total_rows * 0.1:
        column_score -= 1.5
        reason_list.append(f"Missing {missing_count:,} ค่า ({missing_count/total_rows*100:.1f}%) — target มักสมบูรณ์")
    elif missing_count == 0:
        column_score += 0.5
        reason_list.append("ไม่มี Missing Values — สัญญาณที่ดีของ target column")

    # datetime ไม่ควรเป็น target
    if actual_data_type == "datetime":
        column_score -= 5.0
        reason_list.append("เป็น Datetime — ไม่เหมาะเป็น target")

    return column_score, reason_list


def get_column_reasons(dataset: pd.DataFrame, column_name: str) -> list[str]:
    """คืน reasons ของ column ที่ระบุ (ใช้แสดงผลเมื่อผู้ใช้เลือก column เอง)"""
    column_index = list(dataset.columns).index(column_name)
    _, reason_list = score_column(dataset, column_name, column_index)
    return reason_list


def suggest_target(dataset: pd.DataFrame) -> tuple[str, list[str]]:
    """
    แนะนำ target column โดยใช้ scoring heuristic จากลักษณะข้อมูล
    คืนค่า (column_name, [เหตุผล])
    """
    best_column_name = dataset.columns[-1]
    best_column_score = float("-inf")
    best_reason_list: list[str] = []

    for index, column_name in enumerate(dataset.columns):
        score, reason_list = score_column(dataset, column_name, index)
        if score > best_column_score:
            best_column_score = score
            best_column_name = column_name
            best_reason_list = reason_list

    return best_column_name, best_reason_list


def describe_target(dataset: pd.DataFrame, column_name: str) -> str:
    """อธิบาย column ที่ผู้ใช้เลือกเป็น target"""
    data_series = dataset[column_name]
    actual_data_type = actual_type(data_series)
    unique_count = data_series.nunique()
    missing_count = int(data_series.isnull().sum())
    missing_percentage = missing_count / len(data_series) * 100 if len(data_series) > 0 else 0

    if actual_data_type == "bool" or unique_count == 2:
        task_name = "Binary Classification"
    elif actual_data_type == "string" or (actual_data_type == "int" and unique_count <= 20):
        task_name = "Classification"
    elif actual_data_type in ("int", "float"):
        task_name = "Regression"
    else:
        task_name = "ไม่สามารถระบุได้ชัดเจน"

    unique_values = data_series.dropna().unique()[:5]
    unique_preview_text = ", ".join(str(v) for v in unique_values)
    
    if unique_count > 5:
        unique_preview_text += f", … (+{unique_count - 5} อื่นๆ)"

    description_lines = [
        f"**ประเภทข้อมูล:** {actual_data_type}  |  **Unique:** {unique_count:,} ค่า ({unique_preview_text})",
        f"**Task ที่คาดว่าเหมาะสม:** {task_name}",
    ]
    if missing_count > 0:
        description_lines.append(f"**Missing:** {missing_count:,} ค่า ({missing_percentage:.1f}%) — ควรจัดการใน Cleaning")

    return "\n\n".join(description_lines)
