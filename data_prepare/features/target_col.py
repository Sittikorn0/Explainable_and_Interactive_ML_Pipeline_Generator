import pandas as pd
from data_prepare.features.data_type_detection import actual_type


def _score_column(df: pd.DataFrame, col: str, col_idx: int) -> tuple[float, list[str]]:
    """
    คำนวณ score ว่า column นี้น่าจะเป็น target แค่ไหน
    คืนค่า (score, [เหตุผล])
    """
    series = df[col]
    n_rows = len(series)
    n_unique = series.nunique()
    n_missing = int(series.isnull().sum())
    actual = actual_type(series)
    score = 0.0
    reasons = []

    # + ถ้าเป็น column สุดท้าย
    if col_idx == len(df.columns) - 1:
        score += 1.0
        reasons.append("เป็น column สุดท้าย (convention ทั่วไปของ ML datasets)")

    # + ถ้า Binary (unique = 2)
    if n_unique == 2:
        score += 3.0
        reasons.append(f"มีแค่ 2 ค่า Unique — ลักษณะของ Binary Classification target")

    # + ถ้า Cardinality ต่ำ (unique ≤ 5% ของแถว และ ≤ 20)
    elif n_unique <= max(10, n_rows * 0.05) and actual in ("int", "string", "bool"):
        score += 2.0
        pct = n_unique / n_rows * 100
        reasons.append(f"Unique เพียง {n_unique} ค่า ({pct:.1f}% ของข้อมูล) — เหมาะเป็น Classification target")

    # - ถ้า unique สูงมาก (น่าจะเป็น ID หรือ continuous feature)
    if n_unique > n_rows * 0.9:
        score -= 3.0
        reasons.append(f"Unique สูงมาก ({n_unique:,} ค่า) — อาจเป็น ID หรือ key ไม่ใช่ target")

    # - ถ้ามี missing มาก
    if n_missing > n_rows * 0.1:
        score -= 1.5
        reasons.append(f"Missing {n_missing:,} ค่า ({n_missing/n_rows*100:.1f}%) — target มักสมบูรณ์")
    elif n_missing == 0:
        score += 0.5
        reasons.append("ไม่มี Missing Values — สัญญาณที่ดีของ target column")

    # - datetime ไม่ควรเป็น target
    if actual == "datetime":
        score -= 5.0
        reasons.append("เป็น Datetime — ไม่เหมาะเป็น target")

    return score, reasons


def get_column_reasons(df: pd.DataFrame, col: str) -> list[str]:
    """คืน reasons ของ column ที่ระบุ (ใช้แสดงผลเมื่อผู้ใช้เลือก column เอง)"""
    col_idx = list(df.columns).index(col)
    _, reasons = _score_column(df, col, col_idx)
    return reasons


def suggest_target(df: pd.DataFrame) -> tuple[str, list[str]]:
    """
    แนะนำ target column โดยใช้ scoring heuristic จากลักษณะข้อมูล
    คืนค่า (column_name, [เหตุผล])
    """
    best_col = df.columns[-1]
    best_score = float("-inf")
    best_reasons: list[str] = []

    for idx, col in enumerate(df.columns):
        score, reasons = _score_column(df, col, idx)
        if score > best_score:
            best_score = score
            best_col = col
            best_reasons = reasons

    return best_col, best_reasons


def describe_target(df: pd.DataFrame, col: str) -> str:
    """อธิบาย column ที่ผู้ใช้เลือกเป็น target"""
    series = df[col]
    actual = actual_type(series)
    n_unique = series.nunique()
    n_missing = int(series.isnull().sum())
    missing_pct = n_missing / len(series) * 100 if len(series) > 0 else 0

    if actual == "bool" or n_unique == 2:
        task = "Binary Classification"
    elif actual == "string" or (actual == "int" and n_unique <= 20):
        task = "Classification"
    elif actual in ("int", "float"):
        task = "Regression"
    else:
        task = "ไม่สามารถระบุได้ชัดเจน"

    unique_vals = series.dropna().unique()[:5]
    unique_preview = ", ".join(str(v) for v in unique_vals)
    if n_unique > 5:
        unique_preview += f", … (+{n_unique - 5} อื่นๆ)"

    lines = [
        f"**ประเภทข้อมูล:** {actual}  |  **Unique:** {n_unique:,} ค่า ({unique_preview})",
        f"**Task ที่คาดว่าเหมาะสม:** {task}",
    ]
    if n_missing > 0:
        lines.append(f"**Missing:** {n_missing:,} ค่า ({missing_pct:.1f}%) — ควรจัดการใน Cleaning")

    return "\n\n".join(lines)
