import io
import json
import os
import time
import uuid

import pandas as pd
import streamlit as st

_CACHE_DIR = "temp_cache"
_CACHE_TTL = 3 * 3600  # 3 ชม.
_MAX_SESSIONS = 5       # จำนวน session สูงสุดที่เก็บไว้พร้อมกัน


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


def _list_cache_files() -> list[str]:
    """คืนรายชื่อไฟล์ทุกนามสกุลใน temp_cache"""
    return [
        f for f in os.listdir(_CACHE_DIR)
        if os.path.isfile(os.path.join(_CACHE_DIR, f))
    ]


def _session_id_of(fname: str) -> str | None:
    """แยก session id จากชื่อไฟล์ เช่น temp_abc12345.parquet → 'abc12345'"""
    name = os.path.splitext(fname)[0]   # ตัด extension ออก
    parts = name.rsplit("_", 1)
    return parts[1] if len(parts) == 2 else None


def _cleanup_old_files() -> None:
    """ลบไฟล์ใน temp_cache ที่เก่าเกิน TTL และ enforce _MAX_SESSIONS
    ทำงานกับทุกนามสกุลไฟล์ (.parquet, .csv, .txt ฯลฯ)
    """
    now = time.time()
    try:
        # 1. ลบไฟล์ที่เกิน TTL ก่อน (ทุกนามสกุล)
        for fname in _list_cache_files():
            fpath = os.path.join(_CACHE_DIR, fname)
            if (now - os.path.getmtime(fpath)) > _CACHE_TTL:
                try:
                    os.remove(fpath)
                except Exception:
                    pass

        # 2. จัดกลุ่มไฟล์ที่เหลือตาม session id → หา mtime ล่าสุดของแต่ละ session
        session_mtime: dict[str, float] = {}
        for fname in _list_cache_files():
            sid = _session_id_of(fname)
            if sid is None:
                continue
            fpath = os.path.join(_CACHE_DIR, fname)
            mtime = os.path.getmtime(fpath)
            if sid not in session_mtime or mtime > session_mtime[sid]:
                session_mtime[sid] = mtime

        # 3. ถ้าจำนวน session เกิน limit → ลบ session เก่าที่สุดออก (ทุกนามสกุล)
        if len(session_mtime) > _MAX_SESSIONS:
            sorted_sessions = sorted(session_mtime, key=lambda s: session_mtime[s])
            to_delete = set(sorted_sessions[:len(session_mtime) - _MAX_SESSIONS])
            for fname in _list_cache_files():
                if _session_id_of(fname) in to_delete:
                    try:
                        os.remove(os.path.join(_CACHE_DIR, fname))
                    except Exception:
                        pass

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

_CSV_ENCODINGS = ["utf-8", "utf-8-sig", "cp874", "cp1252", "latin1"]
_JSON_ENCODINGS = ["utf-8", "utf-8-sig", "cp874", "cp1252", "latin1"]
_JSON_MAX_LEVEL = 5


def _read_csv_with_fallback(file_bytes: bytes) -> pd.DataFrame:
    """ลอง encoding หลายตัวตามลำดับ กรอง UnicodeDecodeError เท่านั้น
    error ประเภทอื่น (parse error, format error) raise ทันทีโดยไม่ลอง encoding ถัดไป
    """
    last_err = None
    for enc in _CSV_ENCODINGS:
        try:
            return pd.read_csv(io.BytesIO(file_bytes), encoding=enc)
        except UnicodeDecodeError as e:
            last_err = e
    raise last_err


def _decode_json_bytes(file_bytes: bytes) -> str:
    """ลอง decode JSON bytes ด้วย encoding หลายแบบ คืน string แรกที่สำเร็จ"""
    last_err = None
    for enc in _JSON_ENCODINGS:
        try:
            return file_bytes.decode(enc)
        except UnicodeDecodeError as e:
            last_err = e
    raise ValueError(
        f"ไม่สามารถอ่านไฟล์ JSON ได้ — ลอง encoding {_JSON_ENCODINGS} แล้วไม่สำเร็จ"
    ) from last_err


def _parse_json_raw(text: str) -> list[dict]:
    """Parse JSON text → list of dicts รองรับ 3 รูปแบบ:
    1. Array of objects:  [{...}, {...}]
    2. Single object:     {...}  (wrap เป็น list)
    3. JSONL:             {...}\\n{...}\\n  (parse ทีละบรรทัด)

    Raises:
        ValueError: ถ้าโครงสร้างไม่รองรับ
    """
    # พยายาม parse เป็น standard JSON ก่อน
    try:
        raw = json.loads(text)
    except json.JSONDecodeError:
        # fallback: ลอง JSONL (newline-delimited JSON)
        lines = [ln.strip() for ln in text.splitlines() if ln.strip()]
        if not lines:
            raise ValueError("ไฟล์ JSON ว่างเปล่า ไม่มีข้อมูล")
        try:
            raw = [json.loads(ln) for ln in lines]
        except json.JSONDecodeError as e:
            raise ValueError(
                f"ไม่สามารถ parse ไฟล์ JSON ได้: {e}\n\n"
                "ระบบรองรับ:\n"
                "• Array of objects: [{...}, {...}]\n"
                "• Single object: {...}\n"
                "• JSONL (1 object ต่อบรรทัด): {...}\\n{...}"
            ) from e

    # top-level dict → ค้นหา key ที่เป็น array of objects (records)
    if isinstance(raw, dict):
        record_keys = [
            k for k, v in raw.items()
            if isinstance(v, list) and len(v) > 0 and isinstance(v[0], dict)
        ]
        if len(record_keys) == 1:
            # มี key เดียวที่เป็น records array → ใช้นั้น
            raw = raw[record_keys[0]]
        elif len(record_keys) > 1:
            raise ValueError(
                f"ไฟล์ JSON มีหลาย key ที่เป็น array of objects: {record_keys}\n\n"
                "กรุณาระบุ key ที่ต้องการ หรือแปลงไฟล์ให้เป็น array of objects โดยตรง เช่น:\n"
                f'  python -c "import json; d=json.load(open(\'file.json\')); '
                f"print(json.dumps(d['{record_keys[0]}']))\""
            )
        else:
            # ไม่มี key ที่เป็น records array → single object → wrap เป็น list
            raw = [raw]

    if not isinstance(raw, list):
        raise ValueError(
            "รูปแบบ JSON ไม่รองรับ — ต้องเป็น array of objects เช่น [{...}, {...}] "
            "หรือ JSONL (1 object ต่อบรรทัด)"
        )
    if len(raw) == 0:
        raise ValueError("ไฟล์ JSON ว่างเปล่า ไม่มีข้อมูล")
    if not isinstance(raw[0], dict):
        raise ValueError(
            "รูปแบบ JSON ไม่รองรับ — แต่ละ element ต้องเป็น object เช่น [{...}, {...}]"
        )

    return raw


def _read_json_normalized(file_bytes: bytes) -> tuple[pd.DataFrame, list[str]]:
    """โหลด JSON แล้ว flatten nested objects และ join nested arrays เป็น string

    Returns:
        df: DataFrame ที่ flatten แล้ว
        warnings: รายชื่อ column ที่ถูก join from array

    Raises:
        ValueError: ถ้า JSON ไม่ใช่ array of objects หรือโครงสร้างไม่รองรับ
    """
    text = _decode_json_bytes(file_bytes)
    raw = _parse_json_raw(text)

    # flatten nested objects ด้วย json_normalize (max_level ป้องกัน column explosion)
    df = pd.json_normalize(raw, max_level=_JSON_MAX_LEVEL)

    # ตรวจสอบว่ามี column ที่เป็น list of dicts (nested arrays of objects)
    # ซึ่ง json_normalize ไม่สามารถ flatten ได้อัตโนมัติ → แจ้ง error ทันที
    for col in df.columns:
        sample = df[col].dropna()
        if len(sample) > 0:
            first_val = sample.iloc[0]
            if (
                isinstance(first_val, list)
                and len(first_val) > 0
                and isinstance(first_val[0], dict)
            ):
                raise ValueError(
                    f"ไฟล์ JSON มีโครงสร้าง Nested Arrays ที่ซับซ้อนเกินไป\n\n"
                    f"Column '{col}' เป็น array of objects ซึ่งระบบไม่สามารถ flatten ได้อัตโนมัติ\n\n"
                    "ระบบรองรับเฉพาะ JSON ที่มีโครงสร้างแบบนี้:\n"
                    "• Flat objects: [{\"a\": 1, \"b\": 2}, ...]\n"
                    "• Nested objects: [{\"a\": {\"x\": 1}}, ...] (ถูก flatten เป็น a.x)\n\n"
                    "กรุณา flatten ข้อมูลก่อน upload โดยใช้ pandas.json_normalize() "
                    "พร้อม record_path ที่เหมาะสมกับโครงสร้างไฟล์ของคุณ"
                )

    # detect column ที่ยังเป็น list หรือ dict อยู่หลัง flatten → แปลงเป็น string
    # cache mask ก่อนเพื่อไม่ต้อง apply isinstance ซ้ำในรอบเดียวกัน
    joined_cols = []
    for col in df.columns:
        is_list = df[col].apply(lambda x: isinstance(x, list))
        if is_list.any():
            df[col] = df[col].apply(
                lambda x: ", ".join(str(v) for v in x) if isinstance(x, list) else x
            )
            joined_cols.append(col)
            continue

        is_dict = df[col].apply(lambda x: isinstance(x, dict))
        if is_dict.any():
            df[col] = df[col].apply(lambda x: str(x) if isinstance(x, dict) else x)
            joined_cols.append(f"{col} (object → string)")

    # แปลง empty string → NaN เพื่อให้ isnull() ตรวจจับได้ (JSON อาจมี "" แทน null)
    df = df.replace("", pd.NA)

    return df, joined_cols


@st.cache_data(show_spinner="Loading data...", ttl=3600, max_entries=5)
def _process_cached(file_name: str, file_size: int, file_bytes: bytes) -> tuple[pd.DataFrame | None, list[str]]:
    """Returns (df, json_warnings) — json_warnings ว่างเสมอยกเว้นไฟล์ JSON ที่มี nested columns"""
    try:
        if file_name.endswith(".csv"):
            return _read_csv_with_fallback(file_bytes), []
        elif file_name.endswith((".xlsx", ".xls")):
            return pd.read_excel(io.BytesIO(file_bytes)), []
        elif file_name.endswith(".json"):
            return _read_json_normalized(file_bytes)
    except ValueError as e:
        raise
    except Exception as e:
        print(f"Error processing data: {e}")
    return None, []


def process_data(uploaded_file) -> tuple[pd.DataFrame | None, list[str]]:
    """รับ UploadedFile จาก st.file_uploader แล้ว return (DataFrame, json_warnings)"""
    if uploaded_file is None:
        return None, []
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
    if base.endswith("_cleaned"):
        base = base[: -len("_cleaned")]
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
