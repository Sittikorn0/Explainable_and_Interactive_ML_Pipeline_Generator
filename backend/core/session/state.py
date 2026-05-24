# Libraries
import os
import json
import pickle
import pandas as pd
import streamlit as st

# Logic Import
from backend.core.session.session_manager import (
    local_path, metadata_path, cleaned_csv_path, target_path,
    ml_cache_path, ml_meta_path, ensure_cache_dir, transformed_path
)
from backend.function.data_loader.file_reader import sanitize_for_parquet

# Functions
# บันทึก dataset (parquet) และ filename (txt) ลง cache ใช้ใน upload_page
def save_to_local(dataset: pd.DataFrame, filename: str) -> None:
    sanitize_for_parquet(dataset).to_parquet(local_path(), index=False)
    with open(metadata_path(), "w", encoding="utf-8") as file:
        file.write(filename)

# โหลด dataset และ filename จาก cache คืน (df, filename) หรือ (None, None) ใช้ใน restore_session และ rollback_to
def load_from_local() -> tuple[pd.DataFrame | None, str | None]:
    cache_file_path = local_path()
    if not os.path.exists(cache_file_path):
        return None, None
    try:
        dataset = pd.read_parquet(cache_file_path)
        filename = None
        metadata_file = metadata_path()
        if os.path.exists(metadata_file):
            with open(metadata_file, "r", encoding="utf-8") as file:
                filename = file.read()
        return dataset, filename
    except Exception as error:
        print(f"Error reading local cache: {error}")
        return None, None

# บันทึก target column name ลง cache (txt) ใช้ใน upload_page
def save_target_col(target_column: str) -> None:
    ensure_cache_dir()
    with open(target_path(), "w", encoding="utf-8") as file:
        file.write(target_column)

# โหลด target column name จาก cache ใช้ใน restore_session
def load_target_col() -> str | None:
    target_file_path = target_path()
    if not os.path.exists(target_file_path):
        return None
    try:
        with open(target_file_path, "r", encoding="utf-8") as file:
            return file.read().strip()
    except Exception:
        return None

# ลบ cache ไฟล์ทั้งหมดของ session นี้ ใช้ตอน reset หรือ upload ใหม่
def delete_local() -> None:
    from backend.core.session.session_manager import (
        outlier_bounds_path, trans_meta_path, trace_log_path
    )
    files_to_delete = [
        local_path(), metadata_path(), cleaned_csv_path(), target_path(),
        ml_cache_path(), ml_meta_path(),
        outlier_bounds_path(), trans_meta_path(), trace_log_path(),
        transformed_path()
    ]
    for file_path in files_to_delete:
        try:
            if os.path.exists(file_path):
                os.remove(file_path)
        except (FileNotFoundError, PermissionError) as error:
            print(f"Could not delete temp file: {error}")

# บันทึก cleaned dataset ทั้ง csv และ parquet อัปเดต session_state ใช้ใน cleaning_page
def save_cleaned_data(dataset: pd.DataFrame, original_filename: str) -> str:
    base_filename = original_filename.rsplit(".", 1)[0]
    if base_filename.endswith("_cleaned"):
        base_filename = base_filename[: -len("_cleaned")]
    cleaned_filename = f"{base_filename}_cleaned.csv"

    ensure_cache_dir()
    csv_file_path = cleaned_csv_path()
    dataset.to_csv(csv_file_path, index=False)
    sanitize_for_parquet(dataset).to_parquet(local_path(), index=False)

    with open(metadata_path(), "w", encoding="utf-8") as file:
        file.write(cleaned_filename)

    st.session_state["main_df"] = dataset
    st.session_state["last_uploaded_file"] = cleaned_filename
    st.session_state["cleaned_csv_path"] = csv_file_path

    return cleaned_filename

# บันทึก ml_result (pkl) และ metadata (json) ลง cache ใช้ใน model_process_page
def save_ml_cache(ml_result: dict, ml_metrics_data: dict,
                  transformation_summary: dict, target_column: str,
                  scaling_used: str = None, leakage_warnings: list = None) -> None:
    try:
        with open(ml_cache_path(), "wb") as file:
            pickle.dump(ml_result, file)
        metadata = {
            "ml_metrics": ml_metrics_data,
            "trans_summary": transformation_summary,
            "target_col": target_column,
            "scaling_used": scaling_used,
            "leakage_warnings": leakage_warnings
        }
        with open(ml_meta_path(), "w", encoding="utf-8") as file:
            json.dump(metadata, file, ensure_ascii=False)
    except Exception as error:
        print(f"save_ml_cache error: {error}")

# ลบ ml_result และ ml_meta cache files ใช้ใน rollback_to และ clear_downstream
def delete_ml_cache() -> None:
    for file_path in [ml_cache_path(), ml_meta_path()]:
        try:
            if os.path.exists(file_path):
                os.remove(file_path)
        except (FileNotFoundError, PermissionError) as error:
            print(f"Could not delete ML cache: {error}")

# โหลด ml_result และ metadata จาก cache คืน tuple 6 ค่า ใช้ใน restore_session
def load_ml_cache() -> tuple[dict | None, dict, dict, str | None, str | None, list | None]:
    pkl_file_path = ml_cache_path()
    meta_file_path = ml_meta_path()
    if not os.path.exists(pkl_file_path):
        return None, {}, {}, None, None, None
    try:
        with open(pkl_file_path, "rb") as file:
            ml_result = pickle.load(file)
        ml_metrics_data, transformation_summary, target_column = {}, {}, None
        scaling_used, leakage_warnings = None, None
        if os.path.exists(meta_file_path):
            with open(meta_file_path, "r", encoding="utf-8") as file:
                metadata = json.load(file)
            ml_metrics_data = metadata.get("ml_metrics", {})
            transformation_summary = metadata.get("trans_summary", {})
            target_column = metadata.get("target_col")
            scaling_used = metadata.get("scaling_used")
            leakage_warnings = metadata.get("leakage_warnings")
        return ml_result, ml_metrics_data, transformation_summary, target_column, scaling_used, leakage_warnings
    except Exception as error:
        print(f"load_ml_cache error: {error}")
        return None, {}, {}, None, None, None

# บันทึก outlier bounds (IQR/Z-score) ลง cache JSON ใช้ใน cleaning_page
def save_outlier_bounds(bounds: dict) -> None:
    from backend.core.session.session_manager import outlier_bounds_path
    try:
        with open(outlier_bounds_path(), "w", encoding="utf-8") as file:
            json.dump(bounds, file, ensure_ascii=False)
    except Exception as error:
        print(f"save_outlier_bounds error: {error}")

# โหลด outlier bounds จาก cache JSON ใช้ใน cleaning_page และ preprocess pipeline
def load_outlier_bounds() -> dict:
    from backend.core.session.session_manager import outlier_bounds_path
    path = outlier_bounds_path()
    if not os.path.exists(path):
        return {}
    try:
        with open(path, "r", encoding="utf-8") as file:
            return json.load(file)
    except Exception as error:
        print(f"load_outlier_bounds error: {error}")
        return {}

# บันทึก transformation summary และ target_col ลง cache JSON ใช้ใน transformation_page
def save_trans_metadata(summary: dict, target_col: str) -> None:
    from backend.core.session.session_manager import trans_meta_path
    try:
        metadata = {"summary": summary, "target_col": target_col}
        with open(trans_meta_path(), "w", encoding="utf-8") as file:
            json.dump(metadata, file, ensure_ascii=False)
    except Exception as error:
        print(f"save_trans_metadata error: {error}")

# โหลด transformation summary และ target_col จาก cache JSON ใช้ใน restore_session
def load_trans_metadata() -> tuple[dict | None, str | None]:
    from backend.core.session.session_manager import trans_meta_path
    path = trans_meta_path()
    if not os.path.exists(path):
        return None, None
    try:
        with open(path, "r", encoding="utf-8") as file:
            data = json.load(file)
            return data.get("summary"), data.get("target_col")
    except Exception as error:
        print(f"load_trans_metadata error: {error}")
        return None, None

# บันทึก transformed DataFrame (parquet) ลง cache ใช้ใน transformation_page
def save_transformed_df(dataset: pd.DataFrame) -> None:
    ensure_cache_dir()
    sanitize_for_parquet(dataset).to_parquet(transformed_path(), index=False)

# โหลด transformed DataFrame จาก cache parquet ใช้ใน restore_session
def load_transformed_df() -> pd.DataFrame | None:
    path = transformed_path()
    if not os.path.exists(path):
        return None
    try:
        return pd.read_parquet(path)
    except Exception:
        return None
