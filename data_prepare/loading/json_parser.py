import json
import pandas as pd

JSON_ENCODINGS = ["utf-8", "utf-8-sig", "cp874", "cp1252", "latin1"]
JSON_MAX_LEVEL = 5

def decode_json_bytes(file_bytes: bytes) -> str:
    """ลอง decode JSON bytes ด้วย encoding หลายแบบ คืน string แรกที่สำเร็จ"""
    last_error = None
    for encoding in JSON_ENCODINGS:
        try:
            return file_bytes.decode(encoding)
        except UnicodeDecodeError as error:
            last_error = error
    raise ValueError(
        f"ไม่สามารถอ่านไฟล์ JSON ได้ — ลอง encoding {JSON_ENCODINGS} แล้วไม่สำเร็จ"
    ) from last_error

def parse_json_raw(json_text: str) -> list[dict]:
    """Parse JSON text → list of dicts รองรับ 3 รูปแบบ"""
    try:
        raw_data = json.loads(json_text)
    except json.JSONDecodeError:
        json_lines = [line.strip() for line in json_text.splitlines() if line.strip()]
        if not json_lines:
            raise ValueError("ไฟล์ JSON ว่างเปล่า ไม่มีข้อมูล")
        try:
            raw_data = [json.loads(line) for line in json_lines]
        except json.JSONDecodeError as error:
            raise ValueError(
                f"ไม่สามารถ parse ไฟล์ JSON ได้: {error}\n\n"
                "ระบบรองรับ:\n"
                "• Array of objects: [{...}, {...}]\n"
                "• Single object: {...}\n"
                "• JSONL (1 object ต่อบรรทัด): {...}\\n{...}"
            ) from error

    if isinstance(raw_data, dict):
        record_keys = [
            key for key, value in raw_data.items()
            if isinstance(value, list) and len(value) > 0 and isinstance(value[0], dict)
        ]
        if len(record_keys) == 1:
            raw_data = raw_data[record_keys[0]]
        elif len(record_keys) > 1:
            raise ValueError(
                f"ไฟล์ JSON มีหลาย key ที่เป็น array of objects: {record_keys}\n\n"
                "กรุณาระบุ key ที่ต้องการ"
            )
        else:
            raw_data = [raw_data]

    if not isinstance(raw_data, list):
        raise ValueError("รูปแบบ JSON ไม่รองรับ — ต้องเป็น array of objects หรือ JSONL")
    if len(raw_data) == 0:
        raise ValueError("ไฟล์ JSON ว่างเปล่า ไม่มีข้อมูล")
    if not isinstance(raw_data[0], dict):
        raise ValueError("รูปแบบ JSON ไม่รองรับ — แต่ละ element ต้องเป็น object")

    return raw_data

def get_nested_fields(data_sample: pd.Series) -> list[str]:
    field_frequency: dict[str, int] = {}
    for item_list in data_sample.head(20).tolist():
        if isinstance(item_list, list):
            for dictionary_item in item_list:
                if isinstance(dictionary_item, dict):
                    for key in dictionary_item.keys():
                        field_frequency[key] = field_frequency.get(key, 0) + 1
    return sorted(field_frequency, key=lambda key: -field_frequency[key])

def recommend_for_column(
    data_sample: pd.Series, column_type: str, available_fields: list[str]
) -> tuple[str, str]:
    if column_type == "array":
        non_empty_mask = data_sample.apply(lambda value: isinstance(value, list) and len(value) > 0)
        if not non_empty_mask.any():
            return "join", "ไม่มีข้อมูลใน list"
        
        average_length = data_sample[non_empty_mask].apply(len).mean()
        all_items_flat = [val for item_list in data_sample[non_empty_mask] for val in item_list if not isinstance(val, (list, dict))]
        unique_count = len(set(str(val) for val in all_items_flat))
        
        if average_length <= 1.2:
            return "first", f"array มี 1 ค่าโดยเฉลี่ย (avg {average_length:.1f}) — First เหมาะกว่า"
        if unique_count > 30:
            return "count", f"มี {unique_count} unique values — Count ดีกว่าการ Join"
        return "join", f"มี {unique_count} unique values — Join เป็น string ใช้ได้"

    if column_type == "dict":
        all_keys: set[str] = set()
        for val in data_sample.head(5):
            if isinstance(val, dict):
                all_keys.update(val.keys())
        if len(all_keys) <= 10:
            return "flatten_more", f"มี {len(all_keys)} keys — Flatten ให้ข้อมูลที่ใช้ได้มากกว่า"
        return "to_string", f"มี {len(all_keys)} keys — ซับซ้อนเกินไป แนะนำแปลงเป็น string"

    if column_type == "nested_array_of_dicts":
        if not available_fields:
            return "count", "ไม่สามารถตรวจ field ได้ — Count ปลอดภัยที่สุด"
        if len(available_fields) == 1:
            return (
                f"extract_field:{available_fields[0]}",
                f"มีแค่ 1 field ('{available_fields[0]}') — ดึงตรงๆ ได้เลย",
            )
        name_like_fields = [
            field for field in available_fields
            if field.lower() in {"name", "label", "title", "value", "text", "key"}
        ]
        if name_like_fields:
            return (
                f"extract_field:{name_like_fields[0]}",
                f"พบ field '{name_like_fields[0]}' — น่าจะเป็น label ที่ต้องการ",
            )
        list_mask = data_sample.apply(lambda val: isinstance(val, list))
        average_length = data_sample[list_mask].apply(len).mean() if list_mask.any() else 0
        return "count", f"avg {average_length:.1f} items/แถว — Count บอกปริมาณ relationship ได้"

    return "drop", "ไม่แน่ใจ — Drop ปลอดภัยที่สุด"

def generate_previews(data_sample: pd.Series, available_actions: list[str]) -> dict[str, str]:
    max_length = 70
    head_values = data_sample.head(2).tolist()

    def format_preview(values: list) -> str:
        string_value = "  |  ".join(str(val) for val in values)
        return string_value[:max_length] + "…" if len(string_value) > max_length else string_value

    action_previews: dict[str, str] = {}
    for action in available_actions:
        if action == "drop":
            action_previews[action] = "(column จะถูกลบ)"
        elif action == "join":
            action_previews[action] = format_preview(
                [", ".join(str(val) for val in item) if isinstance(item, list) else str(item) for item in head_values]
            )
        elif action == "first":
            action_previews[action] = format_preview(
                [item[0] if isinstance(item, list) and item else None for item in head_values]
            )
        elif action == "count":
            action_previews[action] = format_preview([len(item) if isinstance(item, (list, dict)) else item for item in head_values])
        elif action == "to_string":
            action_previews[action] = format_preview([str(item)[:50] for item in head_values])
        elif action == "count_keys":
            action_previews[action] = format_preview([len(item) if isinstance(item, dict) else item for item in head_values])
        elif action == "flatten_more":
            extracted_keys: set[str] = set()
            for item in head_values:
                if isinstance(item, dict):
                    extracted_keys.update(item.keys())
            key_list_str = ", ".join(f"`{key}`" for key in sorted(extracted_keys)[:5])
            extra_count_str = f" …+{len(extracted_keys) - 5}" if len(extracted_keys) > 5 else ""
            action_previews[action] = f"เพิ่ม {len(extracted_keys)} columns ใหม่: {key_list_str}{extra_count_str}"
        elif action.startswith("extract_field:"):
            field_name = action.removeprefix("extract_field:")
            action_previews[action] = format_preview([
                ", ".join(str(dict_item.get(field_name, "")) for dict_item in item if isinstance(dict_item, dict))
                if isinstance(item, list) else str(item)
                for item in head_values
            ])
    return action_previews

def apply_json_overrides(
    raw_dataframe: pd.DataFrame,
    column_decisions: list[dict],
    user_choices: dict[str, str],
) -> pd.DataFrame:
    processed_dataframe = raw_dataframe.copy()
    columns_to_drop: list[str] = []
    expansion_operations: dict[str, pd.DataFrame] = {}

    for decision in column_decisions:
        column_name = decision["col"]
        action = user_choices.get(column_name, decision["default_action"])
        if column_name not in processed_dataframe.columns:
            continue

        if action == "drop":
            columns_to_drop.append(column_name)
        elif action == "join":
            processed_dataframe[column_name] = processed_dataframe[column_name].apply(
                lambda val: ", ".join(str(item) for item in val) if isinstance(val, list) else val
            )
        elif action == "first":
            processed_dataframe[column_name] = processed_dataframe[column_name].apply(
                lambda val: val[0] if isinstance(val, list) and val else (None if isinstance(val, list) else val)
            )
        elif action == "count":
            processed_dataframe[column_name] = processed_dataframe[column_name].apply(
                lambda val: len(val) if isinstance(val, (list, dict)) else val
            )
        elif action == "to_string":
            processed_dataframe[column_name] = processed_dataframe[column_name].apply(
                lambda val: str(val) if isinstance(val, (list, dict)) else val
            )
        elif action == "count_keys":
            processed_dataframe[column_name] = processed_dataframe[column_name].apply(
                lambda val: len(val) if isinstance(val, dict) else val
            )
        elif action == "flatten_more":
            try:
                values = processed_dataframe[column_name].apply(lambda val: val if isinstance(val, dict) else {})
                expanded_df = pd.json_normalize(values.tolist())
                expanded_df.index = processed_dataframe.index
                if not expanded_df.empty and len(expanded_df.columns) > 0:
                    expanded_df.columns = [f"{column_name}.{sub_col}" for sub_col in expanded_df.columns]
                    expansion_operations[column_name] = expanded_df
                else:
                    processed_dataframe[column_name] = processed_dataframe[column_name].apply(lambda val: str(val) if isinstance(val, dict) else val)
            except Exception:
                processed_dataframe[column_name] = processed_dataframe[column_name].apply(lambda val: str(val) if isinstance(val, dict) else val)
        elif action.startswith("extract_field:"):
            field_name = action.removeprefix("extract_field:")
            processed_dataframe[column_name] = processed_dataframe[column_name].apply(
                lambda val: ", ".join(
                    str(dict_item.get(field_name, "")) for dict_item in val if isinstance(dict_item, dict)
                ) if isinstance(val, list) else val
            )

    if columns_to_drop:
        processed_dataframe = processed_dataframe.drop(columns=columns_to_drop)

    for original_column, expanded_df in expansion_operations.items():
        if original_column in processed_dataframe.columns:
            insert_position = processed_dataframe.columns.get_loc(original_column)
            processed_dataframe = processed_dataframe.drop(columns=[original_column])
            for index, new_column in enumerate(expanded_df.columns):
                processed_dataframe.insert(min(insert_position + index, len(processed_dataframe.columns)), new_column, expanded_df[new_column])

    return processed_dataframe

def read_json_normalized(file_bytes: bytes) -> tuple[pd.DataFrame, list[str], dict]:
    json_text = decode_json_bytes(file_bytes)
    raw_json_data = parse_json_raw(json_text)
    raw_dataframe = pd.json_normalize(raw_json_data, max_level=JSON_MAX_LEVEL)

    column_decisions: list[dict] = []
    for column_name in raw_dataframe.columns:
        data_sample = raw_dataframe[column_name].dropna()
        if len(data_sample) == 0:
            continue

        is_list_mask = data_sample.apply(lambda val: isinstance(val, list))
        if is_list_mask.any():
            has_dict_items = data_sample[is_list_mask].apply(
                lambda val_list: len(val_list) > 0 and isinstance(val_list[0], dict)
            ).any()
            if has_dict_items:
                column_type = "nested_array_of_dicts"
                available_fields = get_nested_fields(data_sample)
                available_actions = (
                    ["count"]
                    + [f"extract_field:{field}" for field in available_fields[:5]]
                    + ["to_string", "drop"]
                )
            else:
                column_type = "array"
                available_fields = []
                available_actions = ["join", "count", "first", "drop"]

            recommended_action, recommendation_reason = recommend_for_column(data_sample, column_type, available_fields)
            default_action = recommended_action if column_type == "nested_array_of_dicts" else "join"

            column_decisions.append({
                "col": column_name, "type": column_type,
                "default_action":        default_action,
                "recommended_action":    recommended_action,
                "recommendation_reason": recommendation_reason,
                "available_actions":     available_actions,
                "available_fields":      available_fields,
                "sample_before":         "  |  ".join(str(val)[:60] for val in data_sample.head(2).tolist()),
                "previews":              generate_previews(data_sample, available_actions),
            })
            continue

        is_dict_mask = data_sample.apply(lambda val: isinstance(val, dict))
        if is_dict_mask.any():
            available_actions = ["flatten_more", "to_string", "count_keys", "drop"]
            recommended_action, recommendation_reason = recommend_for_column(data_sample, "dict", [])
            column_decisions.append({
                "col": column_name, "type": "dict",
                "default_action":        "to_string",
                "recommended_action":    recommended_action,
                "recommendation_reason": recommendation_reason,
                "available_actions":     available_actions,
                "available_fields":      [],
                "sample_before":         "  |  ".join(str(val)[:60] for val in data_sample.head(2).tolist()),
                "previews":              generate_previews(data_sample, available_actions),
            })

    auto_processed_dataframe = apply_json_overrides(raw_dataframe, column_decisions, {})
    auto_processed_dataframe = auto_processed_dataframe.replace("", pd.NA)

    joined_columns = [decision["col"] for decision in column_decisions if decision["default_action"] != "drop"]
    return auto_processed_dataframe, joined_columns, {"raw_df": raw_dataframe, "col_decisions": column_decisions}
