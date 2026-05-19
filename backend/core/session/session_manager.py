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

ALL_SUBDIRS = [SUBDIR_DATASET, SUBDIR_CLEANING, SUBDIR_TRANSFORMATION, SUBDIR_MODEL, SUBDIR_LOG]

def get_session_id() -> str:
    # 1. Check URL first
    url_uid = st.query_params.get("uid")

    # 2. If already in session state
    if "user_uuid" in st.session_state:
        current_id = st.session_state["user_uuid"]
        if url_uid != current_id:
            # st.query_params updates the browser URL bar immediately — no rerun needed
            st.query_params["uid"] = current_id
        return current_id

    # 3. If missing in session but present in URL (recovery after refresh)
    if url_uid:
        st.session_state["user_uuid"] = url_uid
        return url_uid

    # 4. Truly new session
    new_id = str(uuid.uuid4())[:8]
    st.session_state["user_uuid"] = new_id
    st.query_params["uid"] = new_id
    return new_id

def ensure_cache_dir() -> None:
    for subdir in ALL_SUBDIRS:
        os.makedirs(os.path.join(CACHE_DIR, subdir), exist_ok=True)

def cache_path(prefix: str, extension: str, subdir: str = "") -> str:
    if subdir:
        return os.path.join(CACHE_DIR, subdir, f"{prefix}_{get_session_id()}.{extension}")
    return os.path.join(CACHE_DIR, f"{prefix}_{get_session_id()}.{extension}")

def list_cache_files() -> list[str]:
    """คืน full path ของไฟล์ทุกตัวใน cache และโฟลเดอร์ย่อยทั้งหมด"""
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

def session_id_of(filepath: str) -> str | None:
    """แยก session id จาก path เช่น cache/dataset/temp_abc12345.parquet → 'abc12345'"""
    name_without_extension = os.path.splitext(os.path.basename(filepath))[0]
    parts = name_without_extension.rsplit("_", 1)
    return parts[1] if len(parts) == 2 else None

def cleanup_old_files() -> None:
    """ลบไฟล์ใน cache ที่เก่าเกิน TTL และจำกัดจำนวน session
    ทำงานกับทุกนามสกุลไฟล์ (.parquet, .csv, .txt ฯลฯ) ในทุกโฟลเดอร์ย่อย
    """
    current_time = time.time()
    try:
        # ลบไฟล์ที่เกิน TTL ก่อน (ทุกนามสกุล)
        for file_path in list_cache_files():
            if (current_time - os.path.getmtime(file_path)) > CACHE_TTL:
                try:
                    os.remove(file_path)
                except (FileNotFoundError, PermissionError) as error:
                    print(f"Cleanup warning: Could not remove old file {file_path}: {error}")

        # จัดกลุ่มไฟล์ที่เหลือตาม session id → หา mtime ล่าสุดของแต่ละ session
        session_mtime: dict[str, float] = {}
        for file_path in list_cache_files():
            session_id = session_id_of(file_path)
            if session_id is None:
                continue
            modified_time = os.path.getmtime(file_path)
            if session_id not in session_mtime or modified_time > session_mtime[session_id]:
                session_mtime[session_id] = modified_time

        # ถ้าจำนวน session เกิน limit → ลบ session เก่าที่สุดออก (ทุกนามสกุล)
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

def local_path() -> str:
    ensure_cache_dir()
    cleanup_old_files()
    return cache_path("temp", "parquet", SUBDIR_DATASET)

def metadata_path() -> str:
    return cache_path("meta", "txt", SUBDIR_DATASET)

def cleaned_csv_path() -> str:
    return cache_path("cleaned", "csv", SUBDIR_CLEANING)

def target_path() -> str:
    return cache_path("target", "txt", SUBDIR_DATASET)

def ml_cache_path() -> str:
    ensure_cache_dir()
    return cache_path("ml_result", "pkl", SUBDIR_MODEL)

def ml_meta_path() -> str:
    ensure_cache_dir()
    return cache_path("ml_meta", "json", SUBDIR_MODEL)

def outlier_bounds_path() -> str:
    ensure_cache_dir()
    return cache_path("outlier_bounds", "json", SUBDIR_CLEANING)

def trans_meta_path() -> str:
    ensure_cache_dir()
    return cache_path("trans_meta", "json", SUBDIR_TRANSFORMATION)

def trace_log_path() -> str:
    ensure_cache_dir()
    return cache_path("trace_log", "json", SUBDIR_LOG)

def transformed_path() -> str:
    ensure_cache_dir()
    return cache_path("transformed", "parquet", SUBDIR_TRANSFORMATION)
