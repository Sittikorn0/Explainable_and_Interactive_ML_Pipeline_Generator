# Libraries
import os
import time
import uuid
import streamlit as st

CACHE_DIR = "cache"
CACHE_TTL = 3 * 3600  # 3 ชม.
MAX_SESSIONS = 10      # จำนวน session สูงสุดที่เก็บไว้พร้อมกัน

SUBDIR_DATASET        = "dataset"
SUBDIR_CLEANING       = "cleaning"
SUBDIR_TRANSFORMATION = "transformation"
SUBDIR_MODEL          = "model"
SUBDIR_LOG            = "log"
SUBDIR_CATBOOST       = "catboost_info"

ALL_SUBDIRS = [SUBDIR_DATASET, SUBDIR_CLEANING, SUBDIR_TRANSFORMATION, SUBDIR_MODEL, SUBDIR_LOG, SUBDIR_CATBOOST]

# สร้างหรือดึง session ID จาก URL query param หรือ session_state ใช้สร้าง cache path ทุกไฟล์
def get_session_id() -> str:
    url_uid = st.query_params.get("uid")

    if "user_uuid" in st.session_state:
        current_id = st.session_state["user_uuid"]
        if url_uid != current_id:
            st.query_params["uid"] = current_id
        return current_id

    if url_uid:
        st.session_state["user_uuid"] = url_uid
        return url_uid

    new_id = str(uuid.uuid4())[:8]
    st.session_state["user_uuid"] = new_id
    st.query_params["uid"] = new_id
    return new_id

# สร้าง subdirectory ใน cache ถ้ายังไม่มี ใช้ก่อน cache_path ทุกครั้ง
def ensure_cache_dir() -> None:
    for subdir in ALL_SUBDIRS:
        os.makedirs(os.path.join(CACHE_DIR, subdir), exist_ok=True)

# สร้าง path ไฟล์ cache ตาม session ID, prefix, extension, subdir ใช้โดย path helper functions ทั้งหมด
def cache_path(prefix: str, extension: str, subdir: str = "") -> str:
    if subdir:
        return os.path.join(CACHE_DIR, subdir, f"{prefix}_{get_session_id()}.{extension}")
    return os.path.join(CACHE_DIR, f"{prefix}_{get_session_id()}.{extension}")

# คืน list path ไฟล์ทั้งหมดใน cache (ทุก subdir) ใช้ใน cleanup_old_files
def list_cache_files() -> list[str]:
    if not os.path.exists(CACHE_DIR):
        return []
    all_files = []
    for subdir in ALL_SUBDIRS:
        dirpath = os.path.join(CACHE_DIR, subdir)
        if os.path.exists(dirpath):
            for filename in os.listdir(dirpath):
                full_path = os.path.join(dirpath, filename)
                if os.path.isfile(full_path):
                    all_files.append(full_path)
    return all_files

# แยก session ID ออกจาก filename (format: prefix_sessionid.ext) ใช้ใน cleanup_old_files
def session_id_of(filepath: str) -> str | None:
    name_without_extension = os.path.splitext(os.path.basename(filepath))[0]
    parts = name_without_extension.rsplit("_", 1)
    return parts[1] if len(parts) == 2 else None

# ลบ cache ไฟล์ที่เก่าเกิน TTL และ session ส่วนเกินเกิน MAX_SESSIONS ใช้ใน local_path ทุก rerun
def cleanup_old_files() -> None:
    current_time = time.time()
    try:
        for file_path in list_cache_files():
            if (current_time - os.path.getmtime(file_path)) > CACHE_TTL:
                try:
                    os.remove(file_path)
                except (FileNotFoundError, PermissionError) as error:
                    print(f"Cleanup warning: Could not remove old file {file_path}: {error}")

        session_mtime: dict[str, float] = {}
        for file_path in list_cache_files():
            session_id = session_id_of(file_path)
            if session_id is None:
                continue
            modified_time = os.path.getmtime(file_path)
            if session_id not in session_mtime or modified_time > session_mtime[session_id]:
                session_mtime[session_id] = modified_time

        if len(session_mtime) > MAX_SESSIONS:
            sorted_sessions = sorted(session_mtime, key=lambda sid: session_mtime[sid])
            sessions_to_delete = set(sorted_sessions[:len(session_mtime) - MAX_SESSIONS])
            for file_path in list_cache_files():
                if session_id_of(file_path) in sessions_to_delete:
                    try:
                        os.remove(file_path)
                    except (FileNotFoundError, PermissionError) as error:
                        print(f"Cleanup warning: Could not remove session file {file_path}: {error}")

    except Exception as error:
        print(f"Cleanup error: {error}")

# คืน path ไฟล์ dataset หลัก (parquet) พร้อม ensure_cache_dir และ cleanup ใช้ใน state.py
def local_path() -> str:
    ensure_cache_dir()
    cleanup_old_files()
    return cache_path("temp", "parquet", SUBDIR_DATASET)

# คืน path ไฟล์ metadata (txt) สำหรับเก็บชื่อไฟล์ต้นทาง ใช้ใน state.py
def metadata_path() -> str:
    return cache_path("meta", "txt", SUBDIR_DATASET)

# คืน path ไฟล์ cleaned dataset (csv) สำหรับ download ใช้ใน cleaning_page
def cleaned_csv_path() -> str:
    return cache_path("cleaned", "csv", SUBDIR_CLEANING)

# คืน path ไฟล์ target column name (txt) ใช้ใน state.py
def target_path() -> str:
    return cache_path("target", "txt", SUBDIR_DATASET)

# คืน path ไฟล์ ml_result (pkl) สำหรับ cache model competition ใช้ใน state.py
def ml_cache_path() -> str:
    ensure_cache_dir()
    return cache_path("ml_result", "pkl", SUBDIR_MODEL)

# คืน path ไฟล์ ml_meta (json) สำหรับ cache task_type และ best_model info ใช้ใน state.py
def ml_meta_path() -> str:
    ensure_cache_dir()
    return cache_path("ml_meta", "json", SUBDIR_MODEL)

# คืน path ไฟล์ outlier_bounds (json) สำหรับ cache IQR/Z-score bounds ใช้ใน state.py
def outlier_bounds_path() -> str:
    ensure_cache_dir()
    return cache_path("outlier_bounds", "json", SUBDIR_CLEANING)

# คืน path ไฟล์ trans_meta (json) สำหรับ cache scaling_method/encoding_decisions ใช้ใน state.py
def trans_meta_path() -> str:
    ensure_cache_dir()
    return cache_path("trans_meta", "json", SUBDIR_TRANSFORMATION)

# คืน path ไฟล์ trace_log (json) ใช้ใน trace_log.py
def trace_log_path() -> str:
    ensure_cache_dir()
    return cache_path("trace_log", "json", SUBDIR_LOG)

# คืน path ไฟล์ transformed dataset (parquet) ใช้ใน state.py
def transformed_path() -> str:
    ensure_cache_dir()
    return cache_path("transformed", "parquet", SUBDIR_TRANSFORMATION)
