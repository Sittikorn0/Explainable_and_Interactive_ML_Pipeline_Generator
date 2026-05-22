# Libraries
import pandas as pd

# Logic Import
from backend.function.data_type.dtype_detection import actual_type
from backend.core.upload.column import score_column

# แนะนำ target column โดยใช้ scoring heuristic คืน (column_name, [เหตุผล]) ใช้ใน upload_page
def suggest_target(dataset: pd.DataFrame) -> tuple[str, list[str]]:
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


# อธิบาย target column ที่ user เลือก (dtype/task/missing) เป็น markdown string ใช้ใน upload_page
def describe_target(dataset: pd.DataFrame, column_name: str) -> str:
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
        description_lines.append(f"**Missing:** {missing_count:,} ค่า ({missing_percentage:.1f}%) ควรจัดการต่อในขั้นตอน Cleaning")

    return "\n\n".join(description_lines)
