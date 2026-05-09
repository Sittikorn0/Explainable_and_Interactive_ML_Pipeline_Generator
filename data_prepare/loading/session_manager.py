import os
import time
import uuid
import streamlit as st

CACHE_DIR = "temp_cache"
CACHE_TTL = 3 * 3600  # 3 ชม.
MAX_SESSIONS = 20      # จำนวน session สูงสุดที่เก็บไว้พร้อมกัน

def get_session_id() -> str:
    import uuid
    # Priority 1: Already in session state
    if "user_uuid" in st.session_state:
        return st.session_state["user_uuid"]
        
    # Priority 2: Try to recover from URL (using .uid property for stability)
    try:
        url_uid = st.query_params.uid if "uid" in st.query_params else None
        if url_uid:
            st.session_state["user_uuid"] = url_uid
            return url_uid
    except Exception:
        pass
        
    # Priority 3: Truly new session
    new_id = str(uuid.uuid4())[:8]
    st.session_state["user_uuid"] = new_id
    st.query_params["uid"] = new_id
    return new_id

def ensure_cache_dir() -> None:
    os.makedirs(CACHE_DIR, exist_ok=True)

def cache_path(prefix: str, extension: str) -> str:
    return os.path.join(CACHE_DIR, f"{prefix}_{get_session_id()}.{extension}")

def list_cache_files() -> list[str]:
    """คืนรายชื่อไฟล์ทุกนามสกุลใน temp_cache"""
    if not os.path.exists(CACHE_DIR):
        return []
    return [
        filename for filename in os.listdir(CACHE_DIR)
        if os.path.isfile(os.path.join(CACHE_DIR, filename))
    ]

def session_id_of(filename: str) -> str | None:
    """แยก session id จากชื่อไฟล์ เช่น temp_abc12345.parquet → 'abc12345'"""
    name_without_extension = os.path.splitext(filename)[0]
    parts = name_without_extension.rsplit("_", 1)
    return parts[1] if len(parts) == 2 else None

def cleanup_old_files() -> None:
    """ลบไฟล์ใน temp_cache ที่เก่าเกิน TTL และจำกัดจำนวน session
    ทำงานกับทุกนามสกุลไฟล์ (.parquet, .csv, .txt ฯลฯ)
    """
    current_time = time.time()
    try:
        # ลบไฟล์ที่เกิน TTL ก่อน (ทุกนามสกุล)
        for filename in list_cache_files():
            file_path = os.path.join(CACHE_DIR, filename)
            if (current_time - os.path.getmtime(file_path)) > CACHE_TTL:
                try:
                    os.remove(file_path)
                except (FileNotFoundError, PermissionError) as error:
                    print(f"Cleanup warning: Could not remove old file {filename}: {error}")

        # จัดกลุ่มไฟล์ที่เหลือตาม session id → หา mtime ล่าสุดของแต่ละ session
        session_mtime: dict[str, float] = {}
        for filename in list_cache_files():
            session_id = session_id_of(filename)
            if session_id is None:
                continue
            file_path = os.path.join(CACHE_DIR, filename)
            modified_time = os.path.getmtime(file_path)
            if session_id not in session_mtime or modified_time > session_mtime[session_id]:
                session_mtime[session_id] = modified_time

        # ถ้าจำนวน session เกิน limit → ลบ session เก่าที่สุดออก (ทุกนามสกุล)
        if len(session_mtime) > MAX_SESSIONS:
            sorted_sessions = sorted(session_mtime, key=lambda sid: session_mtime[sid])
            sessions_to_delete = set(sorted_sessions[:len(session_mtime) - MAX_SESSIONS])
            for filename in list_cache_files():
                if session_id_of(filename) in sessions_to_delete:
                    try:
                        os.remove(os.path.join(CACHE_DIR, filename))
                    except (FileNotFoundError, PermissionError) as error:
                        print(f"Cleanup warning: Could not remove session file {filename}: {error}")

    except Exception as error:
        print(f"Cleanup error: {error}")

def local_path() -> str:
    ensure_cache_dir()
    cleanup_old_files()
    return cache_path("temp", "parquet")

def metadata_path() -> str:
    return cache_path("meta", "txt")

def cleaned_csv_path() -> str:
    return cache_path("cleaned", "csv")

def target_path() -> str:
    return cache_path("target", "txt")

def ml_cache_path() -> str:
    ensure_cache_dir()
    return cache_path("ml_result", "pkl")

def ml_meta_path() -> str:
    ensure_cache_dir()
    return cache_path("ml_meta", "json")

def outlier_bounds_path() -> str:
    ensure_cache_dir()
    return cache_path("outlier_bounds", "json")

def trans_meta_path() -> str:
    ensure_cache_dir()
    return cache_path("trans_meta", "json")

def trace_log_path() -> str:
    ensure_cache_dir()
    return cache_path("trace_log", "json")

def transformed_path() -> str:
    ensure_cache_dir()
    return cache_path("transformed", "parquet")
