import os
import json
import pickle
import pandas as pd
import streamlit as st

from data_prepare.loading.session_manager import (
    local_path, metadata_path, cleaned_csv_path, target_path,
    ml_cache_path, ml_meta_path, ensure_cache_dir, transformed_path
)
from data_prepare.loading.file_reader import sanitize_for_parquet

def save_to_local(dataset: pd.DataFrame, filename: str) -> None:
    sanitize_for_parquet(dataset).to_parquet(local_path(), index=False)
    with open(metadata_path(), "w", encoding="utf-8") as file:
        file.write(filename)

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

def save_target_col(target_column: str) -> None:
    ensure_cache_dir()
    with open(target_path(), "w", encoding="utf-8") as file:
        file.write(target_column)

def load_target_col() -> str | None:
    target_file_path = target_path()
    if not os.path.exists(target_file_path):
        return None
    try:
        with open(target_file_path, "r", encoding="utf-8") as file:
            return file.read().strip()
    except Exception:
        return None

def delete_local() -> None:
    from data_prepare.loading.session_manager import (
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

def save_ml_cache(ml_result: dict, ml_metrics_data: dict,
                  transformation_summary: dict, target_column: str,
                  scaling_used: str = None, leakage_warnings: list = None) -> None:
    try:
        with open(ml_cache_path(), "wb") as file:
            pickle.dump(ml_result, file)
        metadata = {
            "ml_metrics":    ml_metrics_data,
            "trans_summary": transformation_summary,
            "target_col":    target_column,
            "scaling_used":  scaling_used,
            "leakage_warnings": leakage_warnings
        }
        with open(ml_meta_path(), "w", encoding="utf-8") as file:
            json.dump(metadata, file, ensure_ascii=False)
    except Exception as error:
        print(f"save_ml_cache error: {error}")

def delete_ml_cache() -> None:
    for file_path in [ml_cache_path(), ml_meta_path()]:
        try:
            if os.path.exists(file_path):
                os.remove(file_path)
        except (FileNotFoundError, PermissionError) as error:
            print(f"Could not delete ML cache: {error}")

def load_ml_cache() -> tuple[dict | None, dict, dict, str | None, str | None, list | None]:
    pkl_file_path  = ml_cache_path()
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
            ml_metrics_data    = metadata.get("ml_metrics", {})
            transformation_summary = metadata.get("trans_summary", {})
            target_column    = metadata.get("target_col")
            scaling_used     = metadata.get("scaling_used")
            leakage_warnings = metadata.get("leakage_warnings")
        return ml_result, ml_metrics_data, transformation_summary, target_column, scaling_used, leakage_warnings
    except Exception as error:
        print(f"load_ml_cache error: {error}")
        return None, {}, {}, None, None, None

def save_outlier_bounds(bounds: dict) -> None:
    from data_prepare.loading.session_manager import outlier_bounds_path
    try:
        with open(outlier_bounds_path(), "w", encoding="utf-8") as file:
            json.dump(bounds, file, ensure_ascii=False)
    except Exception as error:
        print(f"save_outlier_bounds error: {error}")

def load_outlier_bounds() -> dict:
    from data_prepare.loading.session_manager import outlier_bounds_path
    path = outlier_bounds_path()
    if not os.path.exists(path):
        return {}
    try:
        with open(path, "r", encoding="utf-8") as file:
            return json.load(file)
    except Exception as error:
        print(f"load_outlier_bounds error: {error}")
        return {}

def save_trans_metadata(summary: dict, target_col: str) -> None:
    from data_prepare.loading.session_manager import trans_meta_path
    try:
        metadata = {"summary": summary, "target_col": target_col}
        with open(trans_meta_path(), "w", encoding="utf-8") as file:
            json.dump(metadata, file, ensure_ascii=False)
    except Exception as error:
        print(f"save_trans_metadata error: {error}")

def load_trans_metadata() -> tuple[dict | None, str | None]:
    from data_prepare.loading.session_manager import trans_meta_path
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

def save_transformed_df(dataset: pd.DataFrame) -> None:
    ensure_cache_dir()
    sanitize_for_parquet(dataset).to_parquet(transformed_path(), index=False)

def load_transformed_df() -> pd.DataFrame | None:
    path = transformed_path()
    if not os.path.exists(path):
        return None
    try:
        return pd.read_parquet(path)
    except Exception:
        return None
