import io
import os
import time
import uuid

import pandas as pd
import streamlit as st

_CACHE_DIR = "temp_cache"
_CACHE_TTL = 24 * 3600  # 24 ชม.


# ── Helpers ───────────────────────────────────────────────────────────────────

def get_session_id() -> str:
    if "user_uuid" not in st.session_state:
        url_uid = st.query_params.get("uid")
        if url_uid:
            st.session_state["user_uuid"] = url_uid
        else:
            new_id = str(uuid.uuid4())[:8]
            st.session_state["user_uuid"] = new_id
            st.query_params["uid"] = new_id
    return st.session_state["user_uuid"]


def _ensure_cache_dir() -> None:
    os.makedirs(_CACHE_DIR, exist_ok=True)


def _cache_path(prefix: str, ext: str) -> str:
    return os.path.join(_CACHE_DIR, f"{prefix}_{get_session_id()}.{ext}")


def _cleanup_old_files() -> None:
    """ลบไฟล์ใน temp_cache ที่เก่าเกิน TTL"""
    now = time.time()
    try:
        for fname in os.listdir(_CACHE_DIR):
            fpath = os.path.join(_CACHE_DIR, fname)
            if os.path.isfile(fpath) and (now - os.path.getmtime(fpath)) > _CACHE_TTL:
                os.remove(fpath)
    except Exception as e:
        print(f"Cleanup error: {e}")


# ── Paths ─────────────────────────────────────────────────────────────────────

def _local_path() -> str:
    _ensure_cache_dir()
    _cleanup_old_files()
    return _cache_path("temp", "parquet")

def _metadata_path() -> str:
    return _cache_path("meta", "txt")

def _cleaned_csv_path() -> str:
    return _cache_path("cleaned", "csv")

def _target_path() -> str:
    return _cache_path("target", "txt")


# ── Load / Save ───────────────────────────────────────────────────────────────

@st.cache_data(show_spinner="Loading data...", ttl=3600, max_entries=5)
def _process_cached(file_name: str, file_size: int, file_bytes: bytes) -> pd.DataFrame | None:
    buf = io.BytesIO(file_bytes)
    try:
        if file_name.endswith(".csv"):
            return pd.read_csv(buf)
        elif file_name.endswith((".xlsx", ".xls")):
            return pd.read_excel(buf)
        elif file_name.endswith(".json"):
            return pd.read_json(buf)
    except Exception as e:
        print(f"Error processing data: {e}")
    return None


def process_data(uploaded_file) -> pd.DataFrame | None:
    """รับ UploadedFile จาก st.file_uploader แล้ว return DataFrame"""
    if uploaded_file is None:
        return None
    file_bytes = uploaded_file.getvalue()
    return _process_cached(uploaded_file.name, len(file_bytes), file_bytes)


def save_to_local(df: pd.DataFrame, filename: str) -> None:
    """บันทึก DataFrame และชื่อไฟล์ลง disk เพื่อป้องกันข้อมูลหายตอน Refresh"""
    df.to_parquet(_local_path(), index=False)
    with open(_metadata_path(), "w", encoding="utf-8") as f:
        f.write(filename)


def load_from_local() -> tuple[pd.DataFrame | None, str | None]:
    """โหลด DataFrame และชื่อไฟล์กลับมาจาก disk"""
    path = _local_path()
    if not os.path.exists(path):
        return None, None
    try:
        df = pd.read_parquet(path)
        filename = None
        meta = _metadata_path()
        if os.path.exists(meta):
            with open(meta, "r", encoding="utf-8") as f:
                filename = f.read()
        return df, filename
    except Exception as e:
        print(f"Error reading local cache: {e}")
        return None, None


def save_target_col(target_col: str) -> None:
    """บันทึก target column ลง disk เพื่อป้องกันหายตอน Refresh"""
    _ensure_cache_dir()
    with open(_target_path(), "w", encoding="utf-8") as f:
        f.write(target_col)


def load_target_col() -> str | None:
    """โหลด target column กลับมาจาก disk"""
    path = _target_path()
    if not os.path.exists(path):
        return None
    try:
        with open(path, "r", encoding="utf-8") as f:
            return f.read().strip()
    except Exception:
        return None


def delete_local() -> None:
    """ลบ temp files ทั้งหมดของ session นี้"""
    for path in [_local_path(), _metadata_path(), _cleaned_csv_path(), _target_path()]:
        try:
            if os.path.exists(path):
                os.remove(path)
        except Exception as e:
            print(f"Could not delete temp file: {e}")


def save_cleaned_data(df: pd.DataFrame, original_filename: str) -> str:
    """
    บันทึก cleaned DataFrame และอัปเดต session_state

    1. save CSV  → temp_cache/cleaned_<sid>.csv  (สำหรับ download)
    2. replace parquet cache เดิม → step ถัดไปโหลดได้ถูกต้อง
    3. อัปเดต metadata filename
    4. อัปเดต session_state

    Returns: cleaned_filename เช่น "hospital_cleaned.csv"
    """
    base = original_filename.rsplit(".", 1)[0]
    cleaned_filename = f"{base}_cleaned.csv"

    _ensure_cache_dir()
    csv_path = _cleaned_csv_path()
    df.to_csv(csv_path, index=False)
    df.to_parquet(_local_path(), index=False)

    with open(_metadata_path(), "w", encoding="utf-8") as f:
        f.write(cleaned_filename)

    st.session_state["main_df"] = df
    st.session_state["last_uploaded_file"] = cleaned_filename
    st.session_state["cleaned_csv_path"] = csv_path

    return cleaned_filename
