import os
import json
import pickle
import pandas as pd
import streamlit as st

from data_prepare.loading.session_manager import (
    local_path, metadata_path, cleaned_csv_path, target_path,
    ml_cache_path, ml_meta_path, ensure_cache_dir
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
    for file_path in [local_path(), metadata_path(), cleaned_csv_path(), target_path()]:
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
                  transformation_summary: dict, target_column: str) -> None:
    try:
        with open(ml_cache_path(), "wb") as file:
            pickle.dump(ml_result, file)
        metadata = {
            "ml_metrics":    ml_metrics_data,
            "trans_summary": transformation_summary,
            "target_col":    target_column,
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

def load_ml_cache() -> tuple[dict | None, dict, dict, str | None]:
    pkl_file_path  = ml_cache_path()
    meta_file_path = ml_meta_path()
    if not os.path.exists(pkl_file_path):
        return None, {}, {}, None
    try:
        with open(pkl_file_path, "rb") as file:
            ml_result = pickle.load(file)
        ml_metrics_data, transformation_summary, target_column = {}, {}, None
        if os.path.exists(meta_file_path):
            with open(meta_file_path, "r", encoding="utf-8") as file:
                metadata = json.load(file)
            ml_metrics_data    = metadata.get("ml_metrics", {})
            transformation_summary = metadata.get("trans_summary", {})
            target_column    = metadata.get("target_col")
        return ml_result, ml_metrics_data, transformation_summary, target_column
    except Exception as error:
        print(f"load_ml_cache error: {error}")
        return None, {}, {}, None
