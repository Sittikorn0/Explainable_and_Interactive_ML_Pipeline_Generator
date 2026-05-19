import pandas as pd

# Type Detection
def is_pseudo_int(data_series: pd.Series) -> bool:
    """float64 ที่ pandas บังคับเพราะมี NaN แต่ค่าจริงเป็นจำนวนเต็มทั้งหมด"""
    non_null_values = data_series.dropna()
    if len(non_null_values) == 0:
        return False
    return bool((non_null_values % 1 == 0).all())


def actual_type(data_series: pd.Series) -> str:
    """คืน type จริง ไม่ใช่ pandas dtype"""
    pandas_dtype = str(data_series.dtype)
    
    if pandas_dtype.startswith("int"):
        return "int"
    if pandas_dtype.startswith("float"):
        return "int" if is_pseudo_int(data_series) else "float"
    if pandas_dtype == "bool":
        return "bool"
    if pandas_dtype == "object":
        try:
            pd.to_datetime(data_series.dropna().head(20), format="mixed", dayfirst=False)
            return "datetime"
        except (ValueError, TypeError):
            pass
        return "string"
    return pandas_dtype


def ml_category(actual_data_type: str, is_target: bool) -> str:
    """จัดประเภทตาม ML Category"""
    if actual_data_type == "datetime":
        category_name = "Datetime"
    elif actual_data_type == "int":
        category_name = "Numeric/Discrete"
    elif actual_data_type == "float":
        category_name = "Numeric/Continuous"
    else:
        category_name = "Categorical/Nominal"
        
    return f"{category_name} (Target)" if is_target else category_name
