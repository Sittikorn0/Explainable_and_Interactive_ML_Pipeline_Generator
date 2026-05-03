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
    ใช้ sep=None + engine='python' เพื่อให้ pandas sniff delimiter อัตโนมัติ
    รองรับ comma (,), semicolon (;), tab (\\t) และ delimiter อื่นๆ
    """
    last_err = None
    for enc in _CSV_ENCODINGS:
        try:
            return pd.read_csv(io.BytesIO(file_bytes), encoding=enc, sep=None, engine="python")
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


# ── JSON Column Analysis Helpers ─────────────────────────────────────────────

def _get_nested_fields(sample: pd.Series) -> list[str]:
    """รวบรวม field names จาก nested array of dicts เรียงตาม frequency"""
    freq: dict[str, int] = {}
    for lst in sample.head(20).tolist():
        if isinstance(lst, list):
            for item in lst:
                if isinstance(item, dict):
                    for k in item.keys():
                        freq[k] = freq.get(k, 0) + 1
    return sorted(freq, key=lambda k: -freq[k])


def _recommend_for_col(
    sample: pd.Series, col_type: str, available_fields: list[str]
) -> tuple[str, str]:
    """คำนวณ recommended action และเหตุผล — คืน (action, reason)"""
    if col_type == "array":
        non_empty = sample.apply(lambda x: isinstance(x, list) and len(x) > 0)
        if not non_empty.any():
            return "join", "ไม่มีข้อมูลใน list"
        avg_len   = sample[non_empty].apply(len).mean()
        all_items = [v for lst in sample[non_empty] for v in lst if not isinstance(v, (list, dict))]
        n_unique  = len(set(str(x) for x in all_items))
        if avg_len <= 1.2:
            return "first", f"array มี 1 ค่าโดยเฉลี่ย (avg {avg_len:.1f}) — First เหมาะกว่า"
        if n_unique > 30:
            return "count", f"มี {n_unique} unique values — Count ดีกว่าการ Join"
        return "join", f"มี {n_unique} unique values — Join เป็น string ใช้ได้"

    if col_type == "dict":
        all_keys: set[str] = set()
        for x in sample.head(5):
            if isinstance(x, dict):
                all_keys.update(x.keys())
        if len(all_keys) <= 10:
            return "flatten_more", f"มี {len(all_keys)} keys — Flatten ให้ข้อมูลที่ใช้ได้มากกว่า"
        return "to_string", f"มี {len(all_keys)} keys — ซับซ้อนเกินไป แนะนำแปลงเป็น string"

    if col_type == "nested_array_of_dicts":
        if not available_fields:
            return "count", "ไม่สามารถตรวจ field ได้ — Count ปลอดภัยที่สุด"
        if len(available_fields) == 1:
            return (
                f"extract_field:{available_fields[0]}",
                f"มีแค่ 1 field ('{available_fields[0]}') — ดึงตรงๆ ได้เลย",
            )
        name_like = [
            f for f in available_fields
            if f.lower() in {"name", "label", "title", "value", "text", "key"}
        ]
        if name_like:
            return (
                f"extract_field:{name_like[0]}",
                f"พบ field '{name_like[0]}' — น่าจะเป็น label ที่ต้องการ",
            )
        list_mask = sample.apply(lambda x: isinstance(x, list))
        avg = sample[list_mask].apply(len).mean() if list_mask.any() else 0
        return "count", f"avg {avg:.1f} items/แถว — Count บอกปริมาณ relationship ได้"

    return "drop", "ไม่แน่ใจ — Drop ปลอดภัยที่สุด"


def _generate_previews(sample: pd.Series, available_actions: list[str]) -> dict[str, str]:
    """สร้าง preview ผลลัพธ์ของแต่ละ action จาก sample data"""
    _MAX  = 70
    head2 = sample.head(2).tolist()

    def _fmt(vals: list) -> str:
        s = "  |  ".join(str(v) for v in vals)
        return s[:_MAX] + "…" if len(s) > _MAX else s

    previews: dict[str, str] = {}
    for action in available_actions:
        if action == "drop":
            previews[action] = "(column จะถูกลบ)"
        elif action == "join":
            previews[action] = _fmt(
                [", ".join(str(v) for v in x) if isinstance(x, list) else str(x) for x in head2]
            )
        elif action == "first":
            previews[action] = _fmt(
                [x[0] if isinstance(x, list) and x else None for x in head2]
            )
        elif action == "count":
            previews[action] = _fmt([len(x) if isinstance(x, (list, dict)) else x for x in head2])
        elif action == "to_string":
            previews[action] = _fmt([str(x)[:50] for x in head2])
        elif action == "count_keys":
            previews[action] = _fmt([len(x) if isinstance(x, dict) else x for x in head2])
        elif action == "flatten_more":
            keys: set[str] = set()
            for x in head2:
                if isinstance(x, dict):
                    keys.update(x.keys())
            key_list = ", ".join(f"`{k}`" for k in sorted(keys)[:5])
            extra    = f" …+{len(keys) - 5}" if len(keys) > 5 else ""
            previews[action] = f"เพิ่ม {len(keys)} columns ใหม่: {key_list}{extra}"
        elif action.startswith("extract_field:"):
            field = action.removeprefix("extract_field:")
            previews[action] = _fmt([
                ", ".join(str(item.get(field, "")) for item in x if isinstance(item, dict))
                if isinstance(x, list) else str(x)
                for x in head2
            ])
    return previews


# ── Core JSON Processing ──────────────────────────────────────────────────────

def _read_json_normalized(file_bytes: bytes) -> tuple[pd.DataFrame, list[str], dict]:
    """โหลด JSON แล้ว flatten nested objects และจัดการ nested columns

    ตรวจจับ 3 ประเภท column ที่ต้องการจัดการ:
    - array              : list ของค่าธรรมดา — join / count / first / drop
    - dict               : object ที่ลึกเกิน max_level — flatten_more / to_string / count_keys / drop
    - nested_array_of_dicts: list ของ objects — count / extract_field / to_string / drop

    Returns:
        df         : DataFrame ที่ใช้ default actions แล้ว
        joined_cols: รายชื่อ columns ที่ถูกแปลง (backward-compat format)
        json_meta  : {"raw_df": DataFrame ก่อน apply actions, "col_decisions": list[dict]}
    """
    text   = _decode_json_bytes(file_bytes)
    raw    = _parse_json_raw(text)
    df_raw = pd.json_normalize(raw, max_level=_JSON_MAX_LEVEL)

    col_decisions: list[dict] = []
    for col in df_raw.columns:
        sample = df_raw[col].dropna()
        if len(sample) == 0:
            continue

        is_list_mask = sample.apply(lambda x: isinstance(x, list))
        if is_list_mask.any():
            has_dict_items = sample[is_list_mask].apply(
                lambda x: len(x) > 0 and isinstance(x[0], dict)
            ).any()
            if has_dict_items:
                col_type      = "nested_array_of_dicts"
                avail_fields  = _get_nested_fields(sample)
                avail_actions = (
                    ["count"]
                    + [f"extract_field:{f}" for f in avail_fields[:5]]
                    + ["to_string", "drop"]
                )
            else:
                col_type      = "array"
                avail_fields  = []
                avail_actions = ["join", "count", "first", "drop"]

            rec_action, rec_reason = _recommend_for_col(sample, col_type, avail_fields)
            # nested_array_of_dicts: ใช้ recommendation เป็น default (ดีกว่า "drop" เสมอ)
            # array: คง "join" เป็น default (safe, backward-compat)
            default_action = rec_action if col_type == "nested_array_of_dicts" else "join"

            col_decisions.append({
                "col": col, "type": col_type,
                "default_action":        default_action,
                "recommended_action":    rec_action,
                "recommendation_reason": rec_reason,
                "available_actions":     avail_actions,
                "available_fields":      avail_fields,
                "sample_before":         "  |  ".join(str(x)[:60] for x in sample.head(2).tolist()),
                "previews":              _generate_previews(sample, avail_actions),
            })
            continue

        is_dict_mask = sample.apply(lambda x: isinstance(x, dict))
        if is_dict_mask.any():
            avail_actions          = ["flatten_more", "to_string", "count_keys", "drop"]
            rec_action, rec_reason = _recommend_for_col(sample, "dict", [])
            col_decisions.append({
                "col": col, "type": "dict",
                "default_action":        "to_string",   # conservative auto-apply
                "recommended_action":    rec_action,
                "recommendation_reason": rec_reason,
                "available_actions":     avail_actions,
                "available_fields":      [],
                "sample_before":         "  |  ".join(str(x)[:60] for x in sample.head(2).tolist()),
                "previews":              _generate_previews(sample, avail_actions),
            })

    df_auto = apply_json_overrides(df_raw, col_decisions, {})
    df_auto = df_auto.replace("", pd.NA)

    joined_cols = [d["col"] for d in col_decisions if d["default_action"] != "drop"]
    return df_auto, joined_cols, {"raw_df": df_raw, "col_decisions": col_decisions}


def apply_json_overrides(
    df_raw: pd.DataFrame,
    col_decisions: list[dict],
    user_choices: dict[str, str],
) -> pd.DataFrame:
    """Apply user choices (หรือ defaults) ต่อ raw normalized DataFrame

    Actions รองรับ:
        join, first, count, to_string, count_keys, flatten_more,
        extract_field:{field_name}, drop
    """
    df         = df_raw.copy()
    drop_cols: list[str]              = []
    expand_ops: dict[str, pd.DataFrame] = {}   # col → sub-DataFrame จาก flatten_more

    for d in col_decisions:
        col    = d["col"]
        action = user_choices.get(col, d["default_action"])
        if col not in df.columns:
            continue

        if action == "drop":
            drop_cols.append(col)
        elif action == "join":
            df[col] = df[col].apply(
                lambda x: ", ".join(str(v) for v in x) if isinstance(x, list) else x
            )
        elif action == "first":
            df[col] = df[col].apply(
                lambda x: x[0] if isinstance(x, list) and x else (None if isinstance(x, list) else x)
            )
        elif action == "count":
            df[col] = df[col].apply(
                lambda x: len(x) if isinstance(x, (list, dict)) else x
            )
        elif action == "to_string":
            df[col] = df[col].apply(
                lambda x: str(x) if isinstance(x, (list, dict)) else x
            )
        elif action == "count_keys":
            df[col] = df[col].apply(
                lambda x: len(x) if isinstance(x, dict) else x
            )
        elif action == "flatten_more":
            try:
                vals     = df[col].apply(lambda x: x if isinstance(x, dict) else {})
                expanded = pd.json_normalize(vals.tolist())
                expanded.index = df.index
                if not expanded.empty and len(expanded.columns) > 0:
                    expanded.columns = [f"{col}.{c}" for c in expanded.columns]
                    expand_ops[col]  = expanded
                else:
                    df[col] = df[col].apply(lambda x: str(x) if isinstance(x, dict) else x)
            except Exception:
                df[col] = df[col].apply(lambda x: str(x) if isinstance(x, dict) else x)
        elif action.startswith("extract_field:"):
            field   = action.removeprefix("extract_field:")
            df[col] = df[col].apply(
                lambda x: ", ".join(
                    str(item.get(field, "")) for item in x if isinstance(item, dict)
                ) if isinstance(x, list) else x
            )

    if drop_cols:
        df = df.drop(columns=drop_cols)

    # แทรก expanded columns ตำแหน่งเดิมของ original column
    for orig_col, expanded in expand_ops.items():
        if orig_col in df.columns:
            pos = df.columns.get_loc(orig_col)
            df  = df.drop(columns=[orig_col])
            for i, new_col in enumerate(expanded.columns):
                df.insert(min(pos + i, len(df.columns)), new_col, expanded[new_col])

    return df


@st.cache_data(show_spinner="Loading data...", ttl=3600, max_entries=5)
def _process_cached(file_name: str, file_size: int, file_bytes: bytes) -> tuple[pd.DataFrame | None, list[str], dict]:
    """Returns (df, file_warnings, json_meta)
    - file_warnings: columns ที่ถูกแปลง + Excel sheet warnings
    - json_meta    : {"raw_df": df, "col_decisions": [...]} สำหรับ JSON; {} สำหรับ format อื่น
    """
    try:
        if file_name.endswith(".csv"):
            df = _coerce_object_columns(_read_csv_with_fallback(file_bytes))
            return df, [], {}
        elif file_name.endswith((".xlsx", ".xls")):
            xf = pd.ExcelFile(io.BytesIO(file_bytes))
            sheet_names = xf.sheet_names
            df_excel = _coerce_object_columns(xf.parse(sheet_names[0]))
            warnings_excel = (
                [f"__excel_sheets__:{','.join(str(s) for s in sheet_names)}"]
                if len(sheet_names) > 1
                else []
            )
            return df_excel, warnings_excel, {}
        elif file_name.endswith(".json"):
            df, joined_cols, json_meta = _read_json_normalized(file_bytes)
            df = _coerce_object_columns(df)
            if json_meta.get("raw_df") is not None:
                json_meta = {**json_meta, "raw_df": _coerce_object_columns(json_meta["raw_df"])}
            return df, joined_cols, json_meta
    except ValueError as e:
        raise
    except Exception as e:
        print(f"Error processing data: {e}")
    return None, [], {}


def process_data(uploaded_file) -> tuple[pd.DataFrame | None, list[str], dict]:
    """รับ UploadedFile จาก st.file_uploader แล้ว return (DataFrame, file_warnings, json_meta)"""
    if uploaded_file is None:
        return None, [], {}
    file_bytes = uploaded_file.getvalue()
    df, warnings, json_meta = _process_cached(uploaded_file.name, len(file_bytes), file_bytes)
    # coerce อยู่นอก @st.cache_data เพื่อให้รันเสมอ แม้ cache จะ hit
    if df is not None:
        df = _coerce_object_columns(df)
    if json_meta.get("raw_df") is not None:
        json_meta = {**json_meta, "raw_df": _coerce_object_columns(json_meta["raw_df"])}
    return df, warnings, json_meta


def _coerce_object_columns(df: pd.DataFrame) -> pd.DataFrame:
    """แปลง object columns ให้ Arrow-compatible:
    - ตัวเลขล้วน → numeric dtype (int64/float64)
    - มี non-string objects (dict, list, int ปน) → string
    - pure string → คงไว้
    """
    df = df.copy()
    for col in df.select_dtypes(include="object").columns:
        non_null = df[col].dropna()
        if non_null.size == 0:
            continue
        numeric = pd.to_numeric(non_null, errors="coerce")
        if numeric.notna().all():
            # ทุกค่าแปลงเป็น numeric ได้ → ใช้ int64/float64
            df[col] = pd.to_numeric(df[col], errors="coerce")
        elif not non_null.map(lambda v: isinstance(v, (str, bytes))).all():
            # มี non-string objects (เช่น int, dict, list ปน) → แปลงเป็น string
            def _to_str(v):
                try:
                    return v if pd.isna(v) else str(v)
                except (ValueError, TypeError):
                    # pd.isna ล้มเหลวเมื่อ v เป็น list/array → ไม่ใช่ null
                    return str(v)
            df[col] = df[col].apply(_to_str)
    return df


def _sanitize_for_parquet(df: pd.DataFrame) -> pd.DataFrame:
    """แปลง object columns ที่มี mixed types (เช่น int ปนกับ str) เป็น string
    เพื่อป้องกัน pyarrow ArrowTypeError ตอน write parquet"""
    df = df.copy()
    for col in df.select_dtypes(include="object").columns:
        non_null = df[col].dropna()
        if non_null.size > 0 and not all(isinstance(v, (str, bytes)) for v in non_null):
            df[col] = df[col].where(df[col].isna(), df[col].astype(str))
    return df


def save_to_local(df: pd.DataFrame, filename: str) -> None:
    """บันทึก DataFrame และชื่อไฟล์ลง disk เพื่อป้องกันข้อมูลหายตอน Refresh"""
    _sanitize_for_parquet(df).to_parquet(_local_path(), index=False)
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
    _sanitize_for_parquet(df).to_parquet(_local_path(), index=False)

    with open(_metadata_path(), "w", encoding="utf-8") as f:
        f.write(cleaned_filename)

    st.session_state["main_df"] = df
    st.session_state["last_uploaded_file"] = cleaned_filename
    st.session_state["cleaned_csv_path"] = csv_path

    return cleaned_filename


# ── ML Result Cache ───────────────────────────────────────────────────────────

import pickle

def _ml_cache_path() -> str:
    _ensure_cache_dir()
    return _cache_path("ml_result", "pkl")

def _ml_meta_path() -> str:
    _ensure_cache_dir()
    return _cache_path("ml_meta", "json")


def save_ml_cache(ml_result: dict, ml_metrics: dict,
                  trans_summary: dict, target_col: str) -> None:
    """บันทึกผล ML ลง disk เพื่อกู้คืนหลัง Refresh"""
    try:
        with open(_ml_cache_path(), "wb") as f:
            pickle.dump(ml_result, f)
        meta = {
            "ml_metrics":    ml_metrics,
            "trans_summary": trans_summary,
            "target_col":    target_col,
        }
        with open(_ml_meta_path(), "w", encoding="utf-8") as f:
            json.dump(meta, f, ensure_ascii=False)
    except Exception as e:
        print(f"save_ml_cache error: {e}")


def delete_ml_cache() -> None:
    """ลบ ML result cache files ของ session นี้ — เรียกเมื่อ upload dataset ใหม่"""
    for path in [_ml_cache_path(), _ml_meta_path()]:
        try:
            if os.path.exists(path):
                os.remove(path)
        except Exception as e:
            print(f"Could not delete ML cache: {e}")


def load_ml_cache() -> tuple[dict | None, dict, dict, str | None]:
    """โหลดผล ML จาก disk คืน (ml_result, ml_metrics, trans_summary, target_col)"""
    pkl_path  = _ml_cache_path()
    meta_path = _ml_meta_path()
    if not os.path.exists(pkl_path):
        return None, {}, {}, None
    try:
        with open(pkl_path, "rb") as f:
            ml_result = pickle.load(f)
        ml_metrics, trans_summary, target_col = {}, {}, None
        if os.path.exists(meta_path):
            with open(meta_path, "r", encoding="utf-8") as f:
                meta = json.load(f)
            ml_metrics    = meta.get("ml_metrics", {})
            trans_summary = meta.get("trans_summary", {})
            target_col    = meta.get("target_col")
        return ml_result, ml_metrics, trans_summary, target_col
    except Exception as e:
        print(f"load_ml_cache error: {e}")
        return None, {}, {}, None
