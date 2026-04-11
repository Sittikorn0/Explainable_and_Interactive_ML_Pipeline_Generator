import pandas as pd

# ── Type Detection ────────────────────────────────────────────────────────────

def _is_pseudo_int(s: pd.Series) -> bool:
    """float64 ที่ pandas บังคับเพราะมี NaN แต่ค่าจริงเป็นจำนวนเต็มทั้งหมด"""
    non_null = s.dropna()
    if len(non_null) == 0:
        return False
    return bool((non_null % 1 == 0).all())


def actual_type(series: pd.Series) -> str:
    """คืน type จริง ไม่ใช่ pandas dtype"""
    dtype = str(series.dtype)
    if dtype.startswith("int"):
        return "int"
    if dtype.startswith("float"):
        return "int" if _is_pseudo_int(series) else "float"
    if dtype == "bool":
        return "bool"
    if dtype == "object":
        try:
            pd.to_datetime(series.dropna().head(20), format="mixed", dayfirst=False)
            return "datetime"
        except (ValueError, TypeError):
            pass
        return "string"
    return dtype


def ml_category(actual: str, is_target: bool) -> str:
    """จัดประเภทตาม ML Category (อ้างอิง Topic 2 - Attribute Types)"""
    if actual == "datetime":
        cat = "Datetime"
    elif actual == "int":
        cat = "Numeric/Discrete"
    elif actual == "float":
        cat = "Numeric/Continuous"
    else:
        cat = "Categorical/Nominal"
    return f"{cat} (Target)" if is_target else cat