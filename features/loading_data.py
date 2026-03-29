import pandas as pd
import streamlit as st
import os
import uuid

def load_data(file_path):
    if file_path is None:
        raise ValueError("No file uploaded. Please upload a dataset.")

    file_name = file_path.name
    print(f"Loading data from: {file_name}")

    try:
        if file_name.endswith(".csv"):
            return pd.read_csv(file_path)
        elif file_name.endswith((".xlsx", ".xls")):
            return pd.read_excel(file_path)
        elif file_name.endswith(".json"):
            return pd.read_json(file_path)
        else:
            raise ValueError("Unsupported file format. Please provide a CSV, Excel, or JSON file.")
    except Exception as e:
        print(f"Error loading data: {e}")
        return None

@st.cache_data(show_spinner="Loading data...", ttl=3600, max_entries=5)
def _process_cached(file_name: str, file_size: int, file_bytes: bytes) -> pd.DataFrame | None:
    import io
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
    file_bytes = uploaded_file.getvalue()          # อ่านครั้งเดียว
    return _process_cached(uploaded_file.name, len(file_bytes), file_bytes)


def get_session_id() -> str:
    if "user_uuid" not in st.session_state:
        # 2. ถ้าไม่มี (เช่น เพิ่ง Refresh) ให้เช็คใน URL
        url_uid = st.query_params.get("uid")
        if url_uid:
            st.session_state["user_uuid"] = url_uid
        else:
            # 3. ถ้าไม่มีจริงๆ (เช่น เข้าเว็บครั้งแรก) ให้สร้างใหม่
            new_id = str(uuid.uuid4())[:8]
            st.session_state["user_uuid"] = new_id
            st.query_params["uid"] = new_id
            
    return st.session_state["user_uuid"]

# DEBUG
# def get_session_id() -> str:
#     # ทดสอบด้วยค่าคงที่ ถ้าใช้ค่านี้แล้ว Refresh ไม่เด้ง 
#     # แสดงว่าเป็นที่ session_id เปลี่ยนตอน Refresh
#     return "test_user_001"

def _local_path() -> str:
    folder = "temp_cache"
    if not os.path.exists(folder):
        os.makedirs(folder)
        
    return os.path.join(folder, f"temp_{get_session_id()}.parquet")

def _metadata_path() -> str:
    import os
    folder = "temp_cache"
    return os.path.join(folder, f"meta_{get_session_id()}.txt")

def save_to_local(df: pd.DataFrame, filename: str) -> str:
    """เซฟ DataFrame ลง disk เพื่อป้องกันข้อมูลหายตอน Refresh"""
    path = _local_path()
    df.to_parquet(path, index=False)
    
    with open(_metadata_path(), "w", encoding="utf-8") as f:
        f.write(filename)
    return path

def load_from_local() -> pd.DataFrame | None:
    """ดึง DataFrame กลับมาจาก disk เมื่อ session state ว่างเปล่า"""
    path = _local_path()
    meta_path = _metadata_path()
    df = None
    filename = None
    
    if os.path.exists(path):
        try:
            df = pd.read_parquet(path)
            if os.path.exists(meta_path):
                with open(meta_path, "r", encoding="utf-8") as f:
                    filename = f.read()
                    
        except Exception as e:
            print(f"Error reading local cache: {e}")
            return None
    return df, filename

def delete_local() -> None:
    path = _local_path()
    meta = _metadata_path()
    if os.path.exists(path):
        try:
            for f in [path, meta]:
                if os.path.exists(f):
                    os.remove(f)
        except Exception as e:
            print(f"Could not delete temp file: {e}")
            
def _cleaned_name(original_filename: str) -> str:
    """แปลงชื่อไฟล์ต้นฉบับเป็นชื่อ cleaned"""
    base = original_filename.rsplit(".", 1)[0]
    return f"{base}_cleaned.csv"
 
 
def save_cleaned_data(df: pd.DataFrame, original_filename: str) -> str:
    """
    บันทึก cleaned DataFrame ลง temp_cache และอัปเดต session_state
 
    ทำ 4 อย่างพร้อมกัน:
      1. save CSV  → temp_cache/cleaned_<sid>.csv  (สำหรับ download)
      2. replace parquet cache เดิม → step ถัดไปโหลดได้ถูกต้อง
      3. อัปเดต metadata filename → แสดงชื่อใหม่ทุกหน้า
      4. อัปเดต session_state → ใช้ได้ทันทีโดยไม่ต้อง refresh
 
    Returns:
        cleaned_filename: ชื่อไฟล์ใหม่ เช่น "hospital_cleaned.csv"
    """
    cleaned_filename = _cleaned_name(original_filename)
 
    folder = "temp_cache"
    if not os.path.exists(folder):
        os.makedirs(folder)
 
    # 1. save CSV ไว้ให้ download
    csv_path = os.path.join(folder, f"cleaned_{get_session_id()}.csv")
    df.to_csv(csv_path, index=False)
 
    # 2. replace parquet cache เดิม (ทับ path เดิมได้เลย)
    df.to_parquet(_local_path(), index=False)
 
    # 3. อัปเดต metadata ให้ชื่อใหม่
    with open(_metadata_path(), "w", encoding="utf-8") as f:
        f.write(cleaned_filename)
 
    # 4. อัปเดต session_state ทันที
    st.session_state["main_df"]            = df
    st.session_state["last_uploaded_file"] = cleaned_filename
    st.session_state["cleaned_csv_path"]   = csv_path  # path สำหรับ download button
 
    return cleaned_filename