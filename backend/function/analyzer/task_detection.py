# Libraries
import pandas as pd
import numpy as np
from scipy.stats import skew

# Logic Import
from backend.core.cleaning.statistic import get_outlier_bounds

# Functions
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