import json
import pandas as pd
from backend.function.data_loader.file_reader import SUPPORTED_ENCODINGS

JSON_MAX_LEVEL = 5 # Max depth when flattening nested JSON

# Read File
def decode_json_bytes(file_bytes: bytes) -> str:
    """Decode bytes เป็น string ด้วย encoding fallback"""
    last_error = None
    for encoding in SUPPORTED_ENCODINGS:
        try:
            return file_bytes.decode(encoding)
        except UnicodeDecodeError as e:
            last_error = e
    raise ValueError(
        f"ไม่สามารถอ่านไฟล์ JSON ได้: ลอง encoding {SUPPORTED_ENCODINGS} แล้วไม่สำเร็จ"
    ) from last_error
    
def parse_json_raw(json_text: str) -> list[dict]:
    """Parse JSON text เป็น list of dicts รองรับ 3 รูปแบบ:
      • Array of objects : [{...}, {...}]
      • Single object    : {...}
      • JSONL            : {...}\\n{...}
    """
    # ลอง parse ปกติก่อน ถ้าไม่ได้ ลอง JSONL (1 object ต่อบรรทัด)
    try:
        data = json.loads(json_text)
    except json.JSONDecodeError:
        lines = [line.strip() for line in json_text.splitlines() if line.strip()]
        if not lines:
            raise ValueError("ไฟล์ JSON ว่างเปล่า ไม่มีข้อมูล")
        try:
            data = [json.loads(line) for line in lines]
        except json.JSONDecodeError as e:
            raise ValueError(
                f"ไม่สามารถ parse ไฟล์ JSON ได้: {e}\n\n"
                "ระบบรองรับ:\n"
                "• Array of objects: [{...}, {...}]\n"
                "• Single object: {...}\n"
                "• JSONL (1 object ต่อบรรทัด): {...}\\n{...}"
            ) from e

    # Single object ถ้ามี key ที่เป็น array of dicts ให้ใช้ key นั้น
    if isinstance(data, dict):
        record_keys = [
            k for k, v in data.items()
            if isinstance(v, list) and len(v) > 0 and isinstance(v[0], dict)
        ]
        if len(record_keys) == 1:
            data = data[record_keys[0]]
        elif len(record_keys) > 1:
            raise ValueError(
                f"ไฟล์ JSON มีหลาย key ที่เป็น array of objects: {record_keys}\n\n"
                "กรุณาระบุ key ที่ต้องการ"
            )
        else:
            data = [data]

    # Validate ผลลัพธ์
    if not isinstance(data, list):
        raise ValueError("รูปแบบ JSON ไม่รองรับ ต้องเป็น array of objects หรือ JSONL")
    if len(data) == 0:
        raise ValueError("ไฟล์ JSON ว่างเปล่า ไม่มีข้อมูล")
    if not isinstance(data[0], dict):
        raise ValueError("รูปแบบ JSON ไม่รองรับ แต่ละ element ต้องเป็น object")

    return data

# Analyze Nested Structure
def get_nested_fields(data_sample: pd.Series) -> list[str]:
    """นับ field ที่ปรากฏใน array-of-dicts แล้วเรียงจากที่พบบ่อยที่สุด"""
    freq: dict[str, int] = {}
    for item in data_sample.head(20).tolist():
        if isinstance(item, list):
            for d in item:
                if isinstance(d, dict):
                    for key in d.keys():
                        freq[key] = freq.get(key, 0) + 1
    return sorted(freq, key=lambda k: -freq[k])

def recommend_for_column(
    data_sample: pd.Series, column_type: str, available_fields: list[str]) -> tuple[str, str]:
    """แนะนำ action ที่เหมาะสมที่สุดสำหรับ column ประเภทนั้น ๆ"""

    if column_type == "array":
        non_empty = data_sample.apply(lambda v: isinstance(v, list) and len(v) > 0)
        if not non_empty.any():
            return "join", "ไม่มีข้อมูลใน list"
        avg_len = data_sample[non_empty].apply(len).mean()
        flat_vals = [v for row in data_sample[non_empty] for v in row if not isinstance(v, (list, dict))]
        unique_count = len(set(str(v) for v in flat_vals))
        # 1.2 = รองรับ array ที่บาง row อาจมี 2 ค่า แต่ส่วนใหญ่มีแค่ 1 → first ยังเหมาะ
        if avg_len <= 1.2:
            return "first", f"array มี 1 ค่าโดยเฉลี่ย (avg {avg_len:.1f}) ใช้ First เหมาะกว่า"
        # 30 = ถ้า unique values มากกว่านี้ การ join จะได้ string ยาวไม่มีประโยชน์
        if unique_count > 30:
            return "count", f"มี {unique_count} unique values ใช้ Count ดีกว่าการ Join"
        return "join", f"มี {unique_count} unique values มี Join เป็น string ใช้ได้"

    if column_type == "dict":
        all_keys: set[str] = set()
        for v in data_sample.head(5):
            if isinstance(v, dict):
                all_keys.update(v.keys())
        if len(all_keys) <= 10:
            return "flatten_more", f"มี {len(all_keys)} keys ซึ่ง Flatten ให้ข้อมูลที่ใช้ได้มากกว่า"
        return "to_string", f"มี {len(all_keys)} keys ซับซ้อนเกินไป แนะนำแปลงเป็น string"

    if column_type == "nested_array_of_dicts":
        if not available_fields:
            return "count", "ไม่สามารถตรวจ field ได้ ใช้ Count ปลอดภัยที่สุด"
        if len(available_fields) == 1:
            return f"extract_field:{available_fields[0]}", f"มีแค่ 1 field ('{available_fields[0]}') สามารถดึงตรงๆ ได้เลย"
        name_like = [f for f in available_fields if f.lower() in {"name", "label", "title", "value", "text", "key"}]
        if name_like:
            return f"extract_field:{name_like[0]}", f"พบ field '{name_like[0]}' น่าจะเป็น label ที่ต้องการ"
        is_list = data_sample.apply(lambda v: isinstance(v, list))
        avg_len = data_sample[is_list].apply(len).mean() if is_list.any() else 0
        return "count", f"avg {avg_len:.1f} items/แถว ใช้ Count บอกปริมาณ relationship ได้"

    return "drop", "ไม่แน่ใจ แต่ใช้ Drop ปลอดภัยที่สุด"

# Transform
def apply_json_overrides(raw_dataframe: pd.DataFrame,column_decisions: list[dict],user_choices: dict[str, str],) -> pd.DataFrame:
    """Apply action ที่ผู้ใช้เลือก (หรือ default) กับแต่ละ nested column"""
    df = raw_dataframe.copy()
    to_drop: list[str] = []
    expansions: dict[str, pd.DataFrame] = {}  # flatten_more  รอแทรก column ทีหลัง

    for decision in column_decisions:
        col = decision["col"]
        action = user_choices.get(col, decision["default_action"])
        if col not in df.columns:
            continue

        if action == "drop":
            to_drop.append(col)
        elif action == "join":
            df[col] = df[col].apply(lambda v: ", ".join(str(x) for x in v) if isinstance(v, list) else v)
        elif action == "first":
            df[col] = df[col].apply(lambda v: v[0] if isinstance(v, list) and v else (None if isinstance(v, list) else v))
        elif action == "count":
            df[col] = df[col].apply(lambda v: len(v) if isinstance(v, (list, dict)) else v)
        elif action == "to_string":
            df[col] = df[col].apply(lambda v: str(v) if isinstance(v, (list, dict)) else v)
        elif action == "flatten_more":
            try:
                expanded = pd.json_normalize(df[col].apply(lambda v: v if isinstance(v, dict) else {}).tolist())
                expanded.index = df.index
                if not expanded.empty:
                    expanded.columns = [f"{col}.{c}" for c in expanded.columns]
                    expansions[col] = expanded
                else:
                    df[col] = df[col].apply(lambda v: str(v) if isinstance(v, dict) else v)
            except Exception:
                df[col] = df[col].apply(lambda v: str(v) if isinstance(v, dict) else v)
        elif action.startswith("extract_field:"):
            field = action.removeprefix("extract_field:")
            df[col] = df[col].apply(
                lambda v: ", ".join(str(d.get(field, "")) for d in v if isinstance(d, dict))
                if isinstance(v, list) else v
            )

    # ลบ column ที่ drop ออกก่อน
    if to_drop:
        df = df.drop(columns=to_drop)

    # แทรก expanded columns ในตำแหน่งเดิมของ column นั้น
    for original_col, expanded in expansions.items():
        if original_col in df.columns:
            pos = df.columns.get_loc(original_col)
            df = df.drop(columns=[original_col])
            for i, new_col in enumerate(expanded.columns):
                df.insert(min(pos + i, len(df.columns)), new_col, expanded[new_col])

    return df

# Apply Functions
def json_normalized(file_bytes: bytes) -> tuple[pd.DataFrame, list[str], dict]:
    """อ่านไฟล์ JSON ครบ pipeline:
      Step 1  decode bytes → parse → flatten ด้วย json_normalize
      Step 2  วิเคราะห์ column ที่ยังเป็น nested (list / dict)
      Step 3  apply default actions อัตโนมัติ
      Return : (processed_df, joined_col_names, metadata)
    """
    # Step 1: Read & flatten
    json_text  = decode_json_bytes(file_bytes)
    raw_data   = parse_json_raw(json_text)
    raw_df     = pd.json_normalize(raw_data, max_level=JSON_MAX_LEVEL)

    # Step 2: Analyze remaining nested columns
    col_decisions: list[dict] = []
    for col in raw_df.columns:
        sample = raw_df[col].dropna()
        if len(sample) == 0:
            continue

        is_list = sample.apply(lambda v: isinstance(v, list))
        if is_list.any():
            has_dicts = sample[is_list].apply(lambda v: len(v) > 0 and isinstance(v[0], dict)).any()
            if has_dicts:
                col_type         = "nested_array_of_dicts"
                fields           = get_nested_fields(sample)
                actions          = ["count"] + [f"extract_field:{f}" for f in fields[:5]] + ["to_string", "drop"]
            else:
                col_type         = "array"
                fields           = []
                actions          = ["join", "count", "first", "drop"]

            rec_action, rec_reason = recommend_for_column(sample, col_type, fields)
            default = rec_action if col_type == "nested_array_of_dicts" else "join"
            col_decisions.append({
                "col":                  col,
                "type":                 col_type,
                "default_action":       default,
                "recommended_action":   rec_action,
                "recommendation_reason":rec_reason,
                "available_actions":    actions,
                "available_fields":     fields,
                "sample_before":        "  |  ".join(str(v)[:60] for v in sample.head(2).tolist()),
            })
            continue

        is_dict = sample.apply(lambda v: isinstance(v, dict))
        if is_dict.any():
            actions              = ["flatten_more", "to_string", "count", "drop"]
            rec_action, rec_reason = recommend_for_column(sample, "dict", [])
            col_decisions.append({
                "col":                  col,
                "type":                 "dict",
                "default_action":       "to_string",
                "recommended_action":   rec_action,
                "recommendation_reason":rec_reason,
                "available_actions":    actions,
                "available_fields":     [],
                "sample_before":        "  |  ".join(str(v)[:60] for v in sample.head(2).tolist()),
            })

    # Step 3: Apply default actions
    processed_df = apply_json_overrides(raw_df, col_decisions, {})
    processed_df = processed_df.replace("", pd.NA)

    joined_cols = [d["col"] for d in col_decisions if d["default_action"] != "drop"]
    return processed_df, joined_cols, {"raw_df": raw_df, "col_decisions": col_decisions}