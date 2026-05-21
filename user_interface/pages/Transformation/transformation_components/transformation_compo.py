import pandas as pd
import streamlit as st
from sklearn.preprocessing import LabelEncoder

# Logic Import
from backend.function.analyzer.task_detection import detect_task
from backend.core.model_training.scaling import SCALING_LABELS

# Function
def render_summary_view(dataframe: pd.DataFrame, transformed_dataframe: pd.DataFrame,
                    summary_dict: dict, target_column: str):
    st.markdown('<div class="section-header">รายงานสรุปการแปลงข้อมูล (SUMMARY REPORT)</div>', unsafe_allow_html=True)

    method_label = SCALING_LABELS.get(summary_dict["scaling_method"], summary_dict["scaling_method"]).upper()
    
    st.markdown(f"""
<div style="display: flex; justify-content: space-around; background-color: rgba(255,255,255,0.02); border: 1px solid rgba(255,255,255,0.1); border-radius: 8px; padding: 20px; margin-bottom: 24px;">
    <div style="text-align: center;">
        <div style="color: #94A3B8; font-size: 0.85rem; font-family: monospace; margin-bottom: 4px;">คอลัมน์ตั้งต้น</div>
        <div style="color: #7AA2F7; font-size: 1.8rem; font-weight: bold;">{summary_dict["original_cols"]}</div>
    </div>
    <div style="text-align: center;">
        <div style="color: #94A3B8; font-size: 0.85rem; font-family: monospace; margin-bottom: 4px;">คัดออก (Drop)</div>
        <div style="color: #f85149; font-size: 1.8rem; font-weight: bold;">{summary_dict.get('dropped_cols', 0)}</div>
    </div>
    <div style="text-align: center;">
        <div style="color: #94A3B8; font-size: 0.85rem; font-family: monospace; margin-bottom: 4px;">คอลัมน์ที่เหลือ</div>
        <div style="color: #BB9AF7; font-size: 1.8rem; font-weight: bold;">{summary_dict["final_cols"]}</div>
    </div>
    <div style="text-align: center;">
        <div style="color: #94A3B8; font-size: 0.85rem; font-family: monospace; margin-bottom: 4px;">เทคนิคปรับสเกล</div>
        <div style="color: #F59E0B; font-size: 1.25rem; font-weight: bold; line-height: 1.5; margin-top: 6px;">{method_label}</div>
    </div>
</div>
""", unsafe_allow_html=True)

    with st.expander("ตัวอย่างข้อมูลหลังการแปลง (DATA PREVIEW)"):
        st.info(f"**ข้อมูล:** การปรับสเกล ({method_label}) จะถูกคำนวณและใช้งานจริงในขั้นตอน ML เพื่อความแม่นยำสูงสุดและป้องกันข้อมูลรั่วไหล")
        st.dataframe(transformed_dataframe.head(5), width="stretch")


    # Success message minimal style
    st.success(f"**[ READY ]** การแปลงข้อมูลเสร็จสมบูรณ์ ระบบพร้อมนำไปสอนโมเดลในขั้นตอนถัดไปโดยใช้เทคนิค **{method_label}**")

def apply_encoding(dataset: pd.DataFrame, encoding_decisions: dict, target_column: str) -> pd.DataFrame:
    """
    Apply encoding ตาม decisions dict
    decisions format: {"col_name": "one_hot_encoding" | "label_encoding" | "ordinal_encoding" | "drop_column"}
    """
    transformed_dataset = dataset.copy()

    for column_name, method in encoding_decisions.items():
        if column_name not in transformed_dataset.columns or column_name == target_column:
            continue

        if method == "drop_column":
            transformed_dataset = transformed_dataset.drop(columns=[column_name])
            
        elif method == "one_hot_encoding":
            dummies = pd.get_dummies(transformed_dataset[column_name], prefix=column_name, drop_first=True, dtype=int)
            transformed_dataset = pd.concat([transformed_dataset.drop(columns=[column_name]), dummies], axis=1)
            
        elif method == "label_encoding":
            label_encoder = LabelEncoder()
            transformed_dataset[column_name] = label_encoder.fit_transform(transformed_dataset[column_name].astype(str))
            
        elif method == "ordinal_encoding":
            sorted_categories = sorted(transformed_dataset[column_name].dropna().astype(str).unique())
            category_to_int = {category: index for index, category in enumerate(sorted_categories)}
            transformed_dataset[column_name] = transformed_dataset[column_name].astype(str).map(category_to_int)

    return transformed_dataset


def apply_feature_selection(dataset: pd.DataFrame, columns_to_drop: list, target_column: str) -> pd.DataFrame:
    """ตัด columns ที่ user เลือก ป้องกันตัด target โดยไม่ตั้งใจ"""
    safe_columns_to_drop = [column for column in columns_to_drop if column != target_column and column in dataset.columns]
    return dataset.drop(columns=safe_columns_to_drop)


def apply_all(dataset: pd.DataFrame, encoding_decisions: dict, scaling_method: str,
              columns_to_drop: list, target_column: str,
              scaling_analysis: dict | None = None,
              encoding_analysis: list | None = None) -> tuple:
    """
    Apply transformations ตามลำดับ:
      - Feature Selection (รวม drop_column จาก encoding)
      - Target Sanitization
      ❌ Encoding  ไม่ทำที่นี่ เพื่อป้องกัน Data Leakage
                    preprocess.py จะ encode หลัง train/test split
      ❌ Scaling   ไม่ทำที่นี่ เพื่อป้องกัน Data Leakage
    """
    transformed_dataset = dataset.copy()

    # Feature Selection: รวม columns_to_drop + คอลัมน์ที่ user เลือก drop จาก encoding
    encoding_drops = [
        col for col, method in encoding_decisions.items()
        if method == "drop_column" and col in transformed_dataset.columns and col != target_column
    ]
    all_drops = list(set(columns_to_drop + encoding_drops))
    transformed_dataset = apply_feature_selection(transformed_dataset, all_drops, target_column)

    # ตรวจสอบว่ายังมี feature เหลืออยู่อย่างน้อย 1 column (นอกจาก target)
    non_target_columns = [column for column in transformed_dataset.columns if column != target_column]
    if len(non_target_columns) == 0:
        raise ValueError(
            "ไม่สามารถ apply transformation ได้ เพราะ Feature Selection ตัด feature ออกจนหมด "
            " ต้องเหลือ feature อย่างน้อย 1 column (นอกจาก target)"
        )

    # Target Sanitization (แก้ชนิดข้อมูลถ้าเป็นตัวเลขที่เก็บเป็น String)
    if transformed_dataset[target_column].dtype == object:
        converted_target = pd.to_numeric(transformed_dataset[target_column], errors="coerce")
        if converted_target.notna().all():
            transformed_dataset[target_column] = converted_target

    # Task Detection (ใช้กลางสำหรับระบุประเภทงาน)
    task_type = detect_task(transformed_dataset, target_column)

    # ❌ ไม่ encode ที่นี่  preprocess.py จะทำหลัง train/test split (ป้องกัน Data Leakage)

    # Target Encoding (เฉพาะ Classification ถ้ายังเป็น categorical)
    if task_type == "classification" and (
        transformed_dataset[target_column].dtype == object or transformed_dataset[target_column].dtype.name == "category"
    ):
        label_encoder = LabelEncoder()
        transformed_dataset[target_column] = label_encoder.fit_transform(transformed_dataset[target_column].astype(str))

    # ❌ ไม่ scale ที่นี่  preprocess.py จะทำให้หลัง split

    # ── ดึง Rule จาก analysis เพื่อส่งต่อให้ trace_log ──
    enc_rule_refs = {}
    if encoding_analysis:
        for info in encoding_analysis:
            col = info.get("col")
            if col and col in encoding_decisions:
                enc_rule_refs[col] = {
                    "rule_id":    info.get("rule_id", ""),
                }

    transformation_summary = {
        "original_rows":      dataset.shape[0],
        "original_cols":      dataset.shape[1],
        "dropped_cols":       len(all_drops),
        "encoded_cols":       len(encoding_decisions),
        "final_cols":         transformed_dataset.shape[1],
        "scaling_method":     scaling_method,
        "task_type":          task_type,
        "encoding_decisions": encoding_decisions,
        # ── rule engine metadata ──
        "scaling_rule_id":    scaling_analysis.get("rule_id", "")    if scaling_analysis else "",
        "enc_rule_refs":      enc_rule_refs,
    }

    return transformed_dataset, transformation_summary
