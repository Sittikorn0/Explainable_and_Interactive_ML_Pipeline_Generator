# Libraries
import pandas as pd

# Logic Import
from backend.function.data_type.dtype_detection import actual_type


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
        reason_list.append("เป็น column สุดท้ายซึ่งเป็น convention ทั่วไปของ ML datasets")

    # ถ้า Binary (unique = 2)
    if unique_count == 2:
        column_score += 3.0
        reason_list.append(f"มีแค่ 2 ค่า Unique เป็นลักษณะของ Binary Classification target")

    # ถ้า Cardinality ต่ำ (unique ≤ 5% ของแถว และ ≤ 20)
    elif unique_count <= max(10, total_rows * 0.05) and actual_data_type in ("int", "string", "bool"):
        column_score += 2.0
        unique_percentage = unique_count / total_rows * 100
        reason_list.append(f"มีค่า Unique เพียงแค่ {unique_count} จากจำนวนทั้งหมด {total_rows} ค่า ({unique_percentage:.1f}% ของข้อมูล) เหมาะเป็น Classification target")

    # ถ้า unique สูงมาก (น่าจะเป็น ID หรือ continuous feature)
    if unique_count > total_rows * 0.9:
        column_score -= 3.0
        reason_list.append(f"มีค่า Unique สูงมากถึง ({unique_count:,} ค่า) อาจเป็น ID หรือ key ไม่ใช่ target")

    # ถ้ามี missing มาก
    if missing_count > total_rows * 0.1:
        column_score -= 1.5
        reason_list.append(f"มีค่า Missing {missing_count:,} ค่า ({missing_count/total_rows*100:.1f}%) target มักสมบูรณ์")
    elif missing_count == 0:
        column_score += 0.5
        reason_list.append("ไม่มี Missing Values เป็นสัญญาณที่ดีสำหรับใช้เป็น target column")

    # datetime ไม่ควรเป็น target
    if actual_data_type == "datetime":
        column_score -= 5.0
        reason_list.append("เป็น Datetime ไม่เหมาะสำหรับใช้เป็น target column")

    return column_score, reason_list


def get_column_reasons(dataset: pd.DataFrame, column_name: str) -> list[str]:
    """คืน reasons ของ column ที่ระบุ (ใช้แสดงผลเมื่อผู้ใช้เลือก column เอง)"""
    column_index = list(dataset.columns).index(column_name)
    _, reason_list = score_column(dataset, column_name, column_index)
    return reason_list


