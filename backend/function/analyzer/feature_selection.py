# Libraries
import pandas as pd

# Logic Import

# Functions
def analyze_feature_selection(dataset: pd.DataFrame, target_column: str) -> dict:
    """
    วิเคราะห์ความซ้ำซ้อนและการกระจายตัวของข้อมูล เพื่อแนะนำ Feature ที่ควรตัดออก
    เช่น ข้อมูลที่แปรปรวนต่ำมาก (Low Variance) หรือมีความสัมพันธ์กันเองสูงมาก (High Correlation)
    """
    numeric_columns = [
        column_name for column_name in dataset.columns
        if column_name != target_column and pd.api.types.is_numeric_dtype(dataset[column_name])
    ]

    columns_to_drop_corr = []
    columns_to_drop_var = []

    # ตรวจสอบ High Correlation (Multicollinearity)
    if len(numeric_columns) >= 2:
        correlation_matrix = dataset[numeric_columns].corr().abs()
        processed_pairs = set()
        
        for i, column_a in enumerate(numeric_columns):
            for column_b in numeric_columns[i+1:]:
                pair = tuple(sorted([column_a, column_b]))
                if pair in processed_pairs:
                    continue
                processed_pairs.add(pair)
                
                corr_value = correlation_matrix.loc[column_a, column_b]
                if corr_value >= 0.85:
                    columns_to_drop_corr.append({
                        "col_a": column_a,
                        "col_b": column_b,
                        "corr":  round(float(corr_value), 3),
                        "drop":  column_b,   # แนะนำให้ตัดคอลัมน์ที่สองทิ้ง
                    })

    # ตรวจสอบ Low Variance (ไม่มีการเปลี่ยนแปลงข้อมูล)
    for column_name in numeric_columns:
        feature_series = dataset[column_name].dropna()
        if len(feature_series) == 0:
            continue
            
        coefficient_of_variation = feature_series.std() / (feature_series.mean() + 1e-9)
        if abs(coefficient_of_variation) < 0.01:
            columns_to_drop_var.append({
                "col": column_name,
                "std": round(float(feature_series.std()), 6),
                "cv":  round(float(coefficient_of_variation), 6),
            })

    reason_for_corr_drop = (
        "คอลัมน์ที่มี Correlation ≥ 0.85 เมื่อเทียบกับคอลัมน์อื่น "
        "ถือว่ามีข้อมูลซ้ำซ้อนกันมาก (Multicollinearity) การเก็บทั้งคู่ไว้จะทำให้โมเดล "
        "ให้น้ำหนักเกินความเป็นจริง และแปลผลได้ยาก"
    )
    reason_for_var_drop = (
        "คอลัมน์ที่มี Variance ต่ำมาก (Coefficient of Variation < 1%) ถือว่าไม่มีข้อมูลที่เป็นประโยชน์ "
        "เพราะค่าเกือบทั้งหมดเหมือนกัน โมเดลไม่สามารถเรียนรู้ Pattern จากข้อมูลที่ไม่มีความแตกต่างกันได้"
    )

    return {
        "drop_high_corr": columns_to_drop_corr,
        "drop_low_var":   columns_to_drop_var,
        "reason_corr":    reason_for_corr_drop,
        "reason_var":     reason_for_var_drop,
    }