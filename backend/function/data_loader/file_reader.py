# Libraries
import streamlit as st
import pandas as pd
import io

# Logic
SUPPORTED_ENCODINGS = ["utf-8", "utf-8-sig", "cp874", "cp1252", "latin1"]
from backend.function.data_loader.json_parser import json_normalized

# Functions
# แปลง object Series ให้เป็น numeric หรือ string ที่ถูกต้อง ใช้ภายใน normalize_dtypes
def fix_column_dtype(series: pd.Series) -> pd.Series:
    non_null = series.dropna()
    if non_null.size == 0:
        return series

    # ลองแปลงเป็นตัวเลข
    as_numeric = pd.to_numeric(non_null, errors="coerce")
    if as_numeric.notna().all():
        return pd.to_numeric(series, errors="coerce")

    # ถ้ามีค่าที่ไม่ใช่ string จะบังคับเป็น string
    has_non_string = not non_null.map(lambda v: isinstance(v, (str, bytes))).all()
    if has_non_string:
        return series.apply(lambda v: str(v) if not pd.isna(v) else v)

    return series

# แก้ dtype ทุก object column ในทั้ง DataFrame ใช้หลัง read_csv/read_excel/json_normalized
def normalize_dtypes(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    for col in df.select_dtypes(include="object").columns:
        df[col] = fix_column_dtype(df[col])
    return df

# แปลง mixed-type object columns เป็น string ป้องกัน error ตอน save parquet ใช้ใน state.py
def sanitize_for_parquet(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    for col in df.select_dtypes(include="object").columns:
        non_null = df[col].dropna()
        if non_null.size > 0 and not all(isinstance(v, (str, bytes)) for v in non_null):
            df[col] = df[col].where(df[col].isna(), df[col].astype(str))
    return df

# Read Files
# อ่าน CSV ลอง encoding หลายแบบจนสำเร็จ รองรับไฟล์ภาษาไทย ใช้ใน process_cached
def read_csv_with_fallback(file_bytes: bytes) -> pd.DataFrame:
    last_error = None
    for encoding in SUPPORTED_ENCODINGS:
        try:
            return pd.read_csv(io.BytesIO(file_bytes), encoding=encoding, sep=None, engine="python")
        except UnicodeDecodeError as e:
            last_error = e
    raise last_error

# อ่าน Excel sheet แรก (xlrd สำหรับ .xls, openpyxl สำหรับ .xlsx) พร้อม warning ถ้ามีหลาย sheet ใช้ใน process_cached
def read_excel(file_bytes: bytes, filename: str = "") -> tuple[pd.DataFrame, list[str]]:
    engine = "xlrd" if filename.lower().endswith(".xls") else "openpyxl"
    excel = pd.ExcelFile(io.BytesIO(file_bytes), engine=engine)
    df = excel.parse(excel.sheet_names[0])
    warnings = (
        [f"__excel_sheets__:{','.join(str(s) for s in excel.sheet_names)}"]
        if len(excel.sheet_names) > 1 else []
    )
    return df, warnings

# Cache & Process Data
# อ่านไฟล์ตาม format (csv/xlsx/json) และ normalize dtypes cached ด้วย st.cache_data ใช้โดย process_data
@st.cache_data(show_spinner="Loading data...", ttl=3600, max_entries=5)
def process_cached(
    filename: str,
    filesize: int,  # ใช้เป็น cache key เท่านั้น
    file_bytes: bytes,
) -> tuple[pd.DataFrame | None, list[str], dict]:
    try:
        if filename.endswith(".csv"):
            df = normalize_dtypes(read_csv_with_fallback(file_bytes))
            return df, [], {}

        elif filename.endswith((".xlsx", ".xls")):
            df, warnings = read_excel(file_bytes, filename)
            return normalize_dtypes(df), warnings, {}

        elif filename.endswith(".json"):
            df, joined_cols, meta = json_normalized(file_bytes)
            df = normalize_dtypes(df)
            if meta.get("raw_df") is not None:
                meta = {**meta, "raw_df": normalize_dtypes(meta["raw_df"])}
            return df, joined_cols, meta

    except ValueError:
        raise
    except Exception as e:
        print(f"Error reading file: {e}")

    return None, [], {}

# entry point หลัก รับ Streamlit UploadedFile แล้วส่งผ่าน process_cached ใช้ใน upload_page
def process_data(uploaded_file) -> tuple[pd.DataFrame | None, list[str], dict]:
    if uploaded_file is None:
        return None, [], {}
    file_bytes = uploaded_file.getvalue()
    return process_cached(uploaded_file.name, len(file_bytes), file_bytes)