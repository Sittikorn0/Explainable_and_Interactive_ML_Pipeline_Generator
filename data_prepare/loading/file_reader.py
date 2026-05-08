import io
import csv
import pandas as pd
import streamlit as st

from data_prepare.loading.json_parser import read_json_normalized

CSV_ENCODINGS = ["utf-8", "utf-8-sig", "cp874", "cp1252", "latin1"]

def coerce_object_columns(dataset: pd.DataFrame) -> pd.DataFrame:
    """แปลง object columns ให้ Arrow-compatible:
    - ตัวเลขล้วน → numeric dtype (int64/float64)
    - มี non-string objects (dict, list, int ปน) → string
    - pure string → คงไว้
    """
    dataset = dataset.copy()
    for column_name in dataset.select_dtypes(include="object").columns:
        non_null_values = dataset[column_name].dropna()
        if non_null_values.size == 0:
            continue
        numeric_values = pd.to_numeric(non_null_values, errors="coerce")
        if numeric_values.notna().all():
            dataset[column_name] = pd.to_numeric(dataset[column_name], errors="coerce")
        elif not non_null_values.map(lambda val: isinstance(val, (str, bytes))).all():
            def force_to_string(value):
                try:
                    return value if pd.isna(value) else str(value)
                except (ValueError, TypeError):
                    return str(value)
            dataset[column_name] = dataset[column_name].apply(force_to_string)
    return dataset

def sanitize_for_parquet(dataset: pd.DataFrame) -> pd.DataFrame:
    """แปลง object columns ที่มี mixed types (เช่น int ปนกับ str) เป็น string
    เพื่อป้องกัน pyarrow ArrowTypeError ตอน write parquet"""
    dataset = dataset.copy()
    for column_name in dataset.select_dtypes(include="object").columns:
        non_null_values = dataset[column_name].dropna()
        if non_null_values.size > 0 and not all(isinstance(val, (str, bytes)) for val in non_null_values):
            dataset[column_name] = dataset[column_name].where(dataset[column_name].isna(), dataset[column_name].astype(str))
    return dataset

def read_csv_with_fallback(file_bytes: bytes) -> pd.DataFrame:
    """ลองตรวจสอบรูปแบบ 10KB แรกด้วย csv.Sniffer แล้วโหลดด้วย C engine เพื่อความเร็วสูงสุด
    หากไม่สำเร็จจะลอง fallback ถอด encoding ที่เป็นไปได้
    """
    sample_text = file_bytes[:10240].decode('utf-8', errors='ignore')
    
    try:
        csv_dialect = csv.Sniffer().sniff(sample_text)
        csv_delimiter = csv_dialect.delimiter
    except csv.Error:
        csv_delimiter = ','
        
    last_error = None
    for encoding in CSV_ENCODINGS:
        try:
            return pd.read_csv(io.BytesIO(file_bytes), encoding=encoding, sep=csv_delimiter, engine="c")
        except UnicodeDecodeError as error:
            last_error = error
    raise last_error

@st.cache_data(show_spinner="Loading data...", ttl=3600, max_entries=5)
def process_cached(filename: str, filesize: int, file_bytes: bytes) -> tuple[pd.DataFrame | None, list[str], dict]:
    try:
        if filename.endswith(".csv"):
            dataframe = coerce_object_columns(read_csv_with_fallback(file_bytes))
            return dataframe, [], {}
        elif filename.endswith((".xlsx", ".xls")):
            excel_file = pd.ExcelFile(io.BytesIO(file_bytes))
            sheet_names = excel_file.sheet_names
            excel_dataframe = coerce_object_columns(excel_file.parse(sheet_names[0]))
            excel_warnings = (
                [f"__excel_sheets__:{','.join(str(sheet) for sheet in sheet_names)}"]
                if len(sheet_names) > 1
                else []
            )
            return excel_dataframe, excel_warnings, {}
        elif filename.endswith(".json"):
            dataframe, joined_columns, json_metadata = read_json_normalized(file_bytes)
            dataframe = coerce_object_columns(dataframe)
            if json_metadata.get("raw_df") is not None:
                json_metadata = {**json_metadata, "raw_df": coerce_object_columns(json_metadata["raw_df"])}
            return dataframe, joined_columns, json_metadata
    except ValueError as error:
        raise
    except Exception as error:
        print(f"Error processing data: {error}")
    return None, [], {}

def process_data(uploaded_file) -> tuple[pd.DataFrame | None, list[str], dict]:
    if uploaded_file is None:
        return None, [], {}
    file_bytes = uploaded_file.getvalue()
    dataframe, warnings, json_metadata = process_cached(uploaded_file.name, len(file_bytes), file_bytes)
    if dataframe is not None:
        dataframe = coerce_object_columns(dataframe)
    if json_metadata.get("raw_df") is not None:
        json_metadata = {**json_metadata, "raw_df": coerce_object_columns(json_metadata["raw_df"])}
    return dataframe, warnings, json_metadata
