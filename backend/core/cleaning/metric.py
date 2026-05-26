# Libraries
import streamlit as st
import pandas as pd
from scipy.stats import skew as _scipy_skew

# Logic Import
from backend.function.data_type.dtype_detection import actual_type
from backend.core.insight.trace_log import *
from backend.core.cleaning.data_distribution import *
from backend.core.cleaning.statistic import *
from backend.core.cleaning.main_logic import *
from backend.core.session.state import load_outlier_bounds, save_outlier_bounds, save_cleaned_data
from backend.core.insight.reasoning_engine.engine import suggest as _rule_suggest

# ── Rule-based default suggestion helpers ─────────────────────────────────────

def _suggest_missing_strategy(df: pd.DataFrame, col: str, missing_count: int) -> tuple[str | None, str, str]:
    """คืน (strategy, rule_id, explanation) ที่ Rule Engine แนะนำสำหรับ missing value imputation"""
    col_dtype = actual_type(df[col])
    is_numeric = col_dtype in ("float", "int")
    missing_ratio = missing_count / max(len(df), 1)

    if is_numeric:
        vals = df[col].dropna()
        skewness_abs = abs(float(_scipy_skew(vals))) if len(vals) > 1 else 0.0
        is_skewed = skewness_abs > 1.0
        has_outliers = False
        dist = st.session_state.get("dist_result")
        if dist:
            _, outlier_details = dist
            outlier_map = {item["Column"]: item["Outliers"] for item in outlier_details}
            has_outliers = outlier_map.get(col, 0) > 0
        facts = {
            "missing_ratio": missing_ratio,
            "dtype": "numeric",
            "has_outliers": has_outliers,
            "is_skewed": is_skewed,
        }
    else:
        facts = {
            "missing_ratio": missing_ratio,
            "dtype": "categorical",
            "has_outliers": False,
            "is_skewed": False,
        }

    result = _rule_suggest("missing_value", facts)
    if not result:
        return None, "", ""

    action_map = {
        "no_action":     None,
        "drop_column":   "drop rows",
        "median_impute": "median (rounded)" if col_dtype == "int" else "median",
        "mean_impute":   "mean",
        "mode_impute":   "most frequent",
    }
    return action_map.get(result["action"]), result.get("rule_id", ""), result.get("explanation", "")


def _suggest_outlier_strategy(outlier_count: int, df_len: int) -> tuple[str, str, str]:
    """คืน (strategy, rule_id, explanation) ที่ Rule Engine แนะนำสำหรับ outlier treatment"""
    outlier_ratio = outlier_count / max(df_len, 1)
    result = _rule_suggest("outlier", {"outlier_ratio": outlier_ratio})
    if not result:
        return "clip", "", ""
    action_map = {
        "no_action": "clip",
        "clip":      "clip",
        "drop rows": "drop rows",
    }
    return action_map.get(result["action"], "clip"), result.get("rule_id", ""), result.get("explanation", "")


def render_drop_columns(working_dataframe: pd.DataFrame, target_column: str):
    st.subheader("Drop Columns")

    with st.expander("Drop Columns คืออะไร?", expanded=False):
        st.markdown(
            "**ทำไมต้องลบคอลัมน์ (Drop Columns)?**\n\n"
            "คอลัมน์บางประเภทไม่ได้ช่วยให้โมเดลเรียนรู้ได้ดีขึ้น ควรพิจารณาลบทิ้ง เช่น:\n"
            "- **ID / รหัสระบุตัวตน:** ไม่มีผลต่อการทำนาย และอาจทำให้โมเดลจำแพทเทิร์นผิด\n"
            "- **ข้อความยาวๆ (Free-text):** ไม่สามารถนำมาคำนวณได้โดยตรงหากไม่ผ่านกระบวนการ NLP\n"
            "- **ค่าว่างมากเกินไป:** หากข้อมูลแหว่งเกิน 80% การเติมค่าใหม่ (Imputation) อาจทำให้ได้ข้อมูลที่ไม่สมจริง"
        )

    if not target_column or target_column not in working_dataframe.columns:
        target_column = None
        st.warning("ไม่พบ Target Column  กรุณากลับไปเลือก Target ที่หน้า Upload ก่อน Drop Columns")

    droppable_columns = [col for col in working_dataframe.columns if col != target_column]

    columns_to_drop = st.multiselect(
        "เลือกคอลัมน์ที่ต้องการลบ",
        options=droppable_columns,
        placeholder="เลือกคอลัมน์...",
        label_visibility="collapsed",
        key="drop_cols_select",
        disabled=target_column is None,
    )

    column_drop1, column_drop2, _ = st.columns([1.5, 1.5, 5])
    with column_drop1:
        if st.button("Drop Selected", key="drop_cols_apply", disabled=not columns_to_drop or target_column is None):
            remaining_columns = [col for col in working_dataframe.columns if col not in columns_to_drop]
            if len(remaining_columns) < 2:
                st.error("ต้องเหลือคอลัมน์อย่างน้อย 2 คอลัมน์")
            else:
                st.session_state["working_df"] = (
                    working_dataframe.drop(columns=columns_to_drop).reset_index(drop=True)
                )
                st.session_state["cleaning_confirmed"] = False
                track_cleaning_bulk("drop_col", columns_to_drop, "dropped manually")
                st.rerun()
    with column_drop2:
        suggested_drops_list = []
        suggested_drop_reasons = {}
        for col in droppable_columns:
            missing_ratio = float(working_dataframe[col].isnull().mean())
            is_constant = working_dataframe[col].nunique() <= 1
            result = _rule_suggest("column_drop", {"missing_ratio": missing_ratio, "is_constant": is_constant})
            if result:
                suggested_drops_list.append(col)
                suggested_drop_reasons[col] = result["rule_id"]

        if suggested_drops_list and st.button("Drop แนะนำ", key="drop_cols_suggested", disabled=target_column is None):
            remaining_columns = [col for col in working_dataframe.columns if col not in suggested_drops_list]
            if len(remaining_columns) < 2:
                st.error("ต้องเหลือคอลัมน์อย่างน้อย 2 คอลัมน์")
            else:
                st.session_state["working_df"] = (
                    working_dataframe.drop(columns=suggested_drops_list).reset_index(drop=True)
                )
                st.session_state["cleaning_confirmed"] = False
                track_cleaning_bulk("drop_col", suggested_drops_list, "dropped by rule engine (CDR_001/CDR_002)")
                st.rerun()

    if suggested_drops_list:
        rule_tags = ", ".join(f"`{suggested_drop_reasons.get(c, '')}`" for c in suggested_drops_list)
        st.caption(f"แนะนำให้ลบ: {', '.join(suggested_drops_list)} — {rule_tags}")

def render_duplicates(working_dataframe: pd.DataFrame, duplicate_count: int):
    st.subheader("Duplicates")

    if duplicate_count == 0:
        st.success("ไม่พบแถวซ้ำ")
    else:
        with st.expander("Duplicates คืออะไร?", expanded=False):
            st.markdown(
                "**ข้อมูลซ้ำ (Duplicated Entries) คืออะไร?**\n\n"
                "แถวข้อมูลที่มีค่าเหมือนกันทุกคอลัมน์ มักเกิดจากข้อผิดพลาดในการบันทึกหรือการดึงข้อมูลซ้อนทับกัน\n\n"
                "**ทำไมต้องลบ?**\n\n"
                "หากปล่อยไว้ โมเดลจะให้น้ำหนักกับข้อมูลตัวอย่างนั้นมากเกินความเป็นจริง (Bias) จึงควร **ลบทิ้ง (Drop)** เสมอ"
            )
        st.write(f"พบแถวซ้ำ **{duplicate_count:,}** แถว")
        if st.button("Drop duplicate rows", key="drop_dup"):
            st.session_state["working_df"] = (
                working_dataframe.drop_duplicates().reset_index(drop=True)
            )
            st.session_state["cleaning_confirmed"] = False
            track_cleaning("drop_dup", "duplicates_dropped", f"dropped {duplicate_count:,} duplicate rows")
            st.rerun()

def determine_missing_compatible(column_type: str) -> list:
    if column_type == "float":
        return ["mean", "median", "forward fill", "backward fill", "drop rows"]
    elif column_type == "int":
        return ["median (rounded)", "most frequent", "forward fill", "backward fill", "drop rows"]
    elif column_type == "datetime":
        return ["forward fill", "backward fill", "drop rows"]
    else:
        return ["most frequent", "forward fill", "backward fill", "drop rows"]

def render_missing_values(working_dataframe: pd.DataFrame, missing_columns_dict: dict, null_counts_series: pd.Series):
    st.subheader("Missing Values")

    if not missing_columns_dict:
        st.success("ไม่มี Missing Values")
        return

    with st.expander("Missing Data คืออะไร?", expanded=False):
        st.markdown(
            "**Missing Data (ข้อมูลสูญหาย) คืออะไร?**\n\n"
            "ค่าว่างที่ขาดหายไปในชุดข้อมูล ซึ่งอาจเกิดจากการกรอกข้อมูลไม่ครบ หรือระบบบันทึกผิดพลาด\n\n"
            "**รูปแบบการหายไป 3 ประเภท:**\n"
            "1. **MCAR (หายแบบสุ่มแท้จริง):** หายไปโดยบังเอิญ ไม่เกี่ยวกับข้อมูลช่องอื่นเลย\n"
            "2. **MAR (หายแบบมีเงื่อนไข):** การหายไปมีความสัมพันธ์กับข้อมูลคอลัมน์อื่น (เช่น ผู้หญิงมักไม่ระบุน้ำหนัก)\n"
            "3. **MNAR (จงใจไม่ตอบ):** การหายไปสัมพันธ์กับค่าความลับของตัวมันเอง (เช่น คนรายได้สูงมักไม่ระบุรายได้)\n\n"
            "**วิธีรับมือที่แนะนำ:**\n"
            "- **Mean / Median:** เติมด้วยค่าเฉลี่ยหรือค่ามัธยฐาน (เหมาะกับข้อมูลตัวเลข)\n"
            "- **Most Frequent:** เติมด้วยข้อมูลที่พบซ้ำบ่อยที่สุด (เหมาะกับข้อมูลหมวดหมู่)\n"
            "- **Drop Rows:** ลบแถวนั้นทิ้ง (ใช้เมื่อข้อมูลหายเพียงเล็กน้อยมากๆ เพื่อไม่ให้กระทบภาพรวม)"
        )

    select_all_col, deselect_all_col, _, global_strategy_col, apply_selected_col, apply_all_col = st.columns(ACTION_BAR_COLUMNS)
    with select_all_col:
        if st.button("Select All", key="miss_sel_all", width="stretch"):
            for col_name in missing_columns_dict:
                st.session_state[f"miss_check_{col_name}"] = True
            st.rerun()
    with deselect_all_col:
        if st.button("Deselect All", key="miss_desel_all", width="stretch"):
            for col_name in missing_columns_dict:
                st.session_state[f"miss_check_{col_name}"] = False
            st.rerun()
    with global_strategy_col:
        global_missing_strategy = st.selectbox(
            "Global Strategy",
            ["mean", "median", "median (rounded)", "most frequent", "forward fill", "backward fill", "drop rows"],
            key="miss_global_strategy",
            label_visibility="collapsed",
        )
    checked_missing_columns = [
        col_name for col_name in missing_columns_dict
        if st.session_state.get(f"miss_check_{col_name}", False)
    ]

    with apply_selected_col:
        if st.button("Apply Selected", key="miss_apply_selected", disabled=not checked_missing_columns, width="stretch"):
            dataframe_work = st.session_state["working_df"]
            strategies_to_apply_dict = {}
            for col_name in checked_missing_columns:
                compatible_strategies = determine_missing_compatible(actual_type(dataframe_work[col_name]))
                selected_strategy = global_missing_strategy if global_missing_strategy in compatible_strategies else compatible_strategies[0]
                strategies_to_apply_dict[col_name] = selected_strategy
            
            dataframe_work = use_missing_strategy_bulk(dataframe_work, strategies_to_apply_dict)
            for col_name, applied_strategy in strategies_to_apply_dict.items():
                track_cleaning("missing", col_name, applied_strategy)
                st.session_state.setdefault("missing_rules", {})[col_name] = applied_strategy
                
            st.session_state["working_df"] = dataframe_work
            st.session_state["cleaning_confirmed"] = False
            st.rerun()
    with apply_all_col:
        if st.button("Apply All", key="miss_apply_all", width="stretch"):
            dataframe_work = st.session_state["working_df"]
            strategies_to_apply_dict = {}
            for col_name in missing_columns_dict:
                compatible_strategies = determine_missing_compatible(actual_type(dataframe_work[col_name]))
                selected_strategy = global_missing_strategy if global_missing_strategy in compatible_strategies else compatible_strategies[0]
                strategies_to_apply_dict[col_name] = selected_strategy

            dataframe_work = use_missing_strategy_bulk(dataframe_work, strategies_to_apply_dict)
            for col_name, applied_strategy in strategies_to_apply_dict.items():
                track_cleaning("missing", col_name, applied_strategy)
                st.session_state.setdefault("missing_rules", {})[col_name] = applied_strategy

            st.session_state["working_df"] = dataframe_work
            st.session_state["cleaning_confirmed"] = False
            st.rerun()

    _, rec_btn_col = st.columns([4, 2])
    with rec_btn_col:
        if st.button("Apply All (Recommend)", key="miss_apply_recommend", type="primary", width="stretch"):
            dataframe_work = st.session_state["working_df"]
            strategies_to_apply_dict = {}
            explanations_dict = {}
            for col_name, missing_count in missing_columns_dict.items():
                rec_strategy, _, rec_explanation = _suggest_missing_strategy(dataframe_work, col_name, missing_count)
                if rec_strategy is None:
                    compatible_strategies = determine_missing_compatible(actual_type(dataframe_work[col_name]))
                    rec_strategy = compatible_strategies[0]
                strategies_to_apply_dict[col_name] = rec_strategy
                explanations_dict[col_name] = rec_explanation

            dataframe_work = use_missing_strategy_bulk(dataframe_work, strategies_to_apply_dict)
            for col_name, applied_strategy in strategies_to_apply_dict.items():
                track_cleaning("missing", col_name, applied_strategy, explanations_dict.get(col_name, ""))
                st.session_state.setdefault("missing_rules", {})[col_name] = applied_strategy

            st.session_state["working_df"] = dataframe_work
            st.session_state["cleaning_confirmed"] = False
            st.rerun()

    st.caption(MISSING_STRATEGY_INFO.get(global_missing_strategy, ""))
    st.markdown(HORIZONTAL_RULE_HTML, unsafe_allow_html=True)

    last_missing_column = list(missing_columns_dict)[-1]
    for col_name, missing_count in missing_columns_dict.items():
        missing_percentage = missing_count / len(working_dataframe) * 100
        column_data_type = actual_type(working_dataframe[col_name])

        suggested_strategy, suggested_rule_id, suggested_explanation = _suggest_missing_strategy(working_dataframe, col_name, missing_count)

        checkbox_col, name_col, strategy_col, apply_col, _ = st.columns([0.4, 2.8, 2, 0.8, 0.5], vertical_alignment="center")
        with checkbox_col:
            st.checkbox("Select", key=f"miss_check_{col_name}", label_visibility="hidden")
        with name_col:
            rule_badge = f" `{suggested_rule_id}`" if suggested_rule_id else ""
            st.markdown(f"**{col_name}**  {missing_count:,} ค่า ({missing_percentage:.1f}%){rule_badge}")
        with strategy_col:
            available_options = determine_missing_compatible(column_data_type)
            default_idx = available_options.index(suggested_strategy) if suggested_strategy in available_options else 0
            chosen_strategy = st.selectbox("Strategy", available_options, index=default_idx, key=f"miss_strategy_{col_name}", label_visibility="collapsed")
        with apply_col:
            if st.button("Apply", key=f"miss_apply_{col_name}"):
                dataframe_work = use_missing_strategy(st.session_state["working_df"], col_name, chosen_strategy)
                st.session_state["working_df"] = dataframe_work
                st.session_state["cleaning_confirmed"] = False
                track_cleaning("missing", col_name, chosen_strategy, suggested_explanation)
                st.session_state.setdefault("missing_rules", {})[col_name] = chosen_strategy
                st.rerun()

        st.caption(MISSING_STRATEGY_INFO.get(chosen_strategy, ""))

        if col_name != last_missing_column:
            st.markdown(HORIZONTAL_RULE_HTML, unsafe_allow_html=True)


def render_outliers(working_dataframe: pd.DataFrame, outlier_columns_dict: dict, outlier_details: list):
    st.subheader("Outliers")

    if not outlier_columns_dict:
        st.success("ไม่พบ Outliers")
        return

    with st.expander("ระบบตรวจจับ Outlier อย่างไร?", expanded=False):
        st.markdown(
            "**Outlier (ข้อมูลผิดปกติ) คืออะไร?**\n\n"
            "ข้อมูลที่มีค่าผิดปกติหรือห่างจากข้อมูลส่วนใหญ่อย่างมีนัยสำคัญ ซึ่งอาจทำให้โมเดลเรียนรู้รูปแบบที่ผิดเพี้ยนไป\n\n"
            "**ระบบเลือกวิธีตรวจจับอัตโนมัติตามรูปทรงข้อมูล (Skewness):**\n"
            "- **ใช้ Z-Score** (เมื่อ |Skew| < 0.5) \n"
            "  เหมาะกับข้อมูลที่มีการกระจายตัวแบบสมมาตร (ระฆังคว่ำ) โดยจะตัดค่าที่ห่างจากค่าเฉลี่ยเกิน 3 SD\n"
            "- **ใช้ IQR** (เมื่อ |Skew| ≥ 0.5) \n"
            "  เหมาะกับข้อมูลที่เบ้ไปทางใดทางหนึ่ง โดยจะใช้ระยะห่างระหว่างควอไทล์ที่ 1 และ 3 ในการกำหนดขอบเขต\n\n"
            "**วิธีรับมือ:**\n"
            "มักใช้การ **จำกัดขอบเขต (Clipping)** โดยบีบค่าที่เกินเกณฑ์ให้กลับมาอยู่ในขอบเขตสูงสุด/ต่ำสุด เพื่อรักษาจำนวนข้อมูลไว้"
        )

    select_all_col, deselect_all_col, _, global_strategy_col, apply_selected_col, apply_all_col = st.columns(ACTION_BAR_COLUMNS)
    with select_all_col:
        if st.button("Select All", key="out_sel_all", width="stretch"):
            for col_name in outlier_columns_dict:
                st.session_state[f"out_check_{col_name}"] = True
            st.rerun()
    with deselect_all_col:
        if st.button("Deselect All", key="out_desel_all", width="stretch"):
            for col_name in outlier_columns_dict:
                st.session_state[f"out_check_{col_name}"] = False
            st.rerun()
    with global_strategy_col:
        global_outlier_strategy = st.selectbox(
            "Global Strategy",
            ["clip", "drop rows"],
            key="out_global_strategy",
            label_visibility="collapsed",
        )
    checked_outlier_columns = [
        col_name for col_name in outlier_columns_dict
        if st.session_state.get(f"out_check_{col_name}", False)
    ]
    with apply_selected_col:
        if st.button("Apply Selected", key="out_apply_selected", disabled=not checked_outlier_columns, width="stretch"):
            dataframe_work = st.session_state["working_df"]
            strategies_to_apply_dict = {}
            for col_name in checked_outlier_columns:
                outlier_bounds = outlier_columns_dict[col_name]
                strategies_to_apply_dict[col_name] = {"strategy": global_outlier_strategy, "lower": outlier_bounds["Lower"], "upper": outlier_bounds["Upper"]}
            
            dataframe_work = use_outlier_strategy_bulk(dataframe_work, strategies_to_apply_dict)
            
            treated_outliers = st.session_state.setdefault("treated_outlier_cols", {})
            for col_name in checked_outlier_columns:
                track_cleaning("outlier", col_name, global_outlier_strategy)
                treated_outliers[col_name] = global_outlier_strategy
                outlier_bounds = outlier_columns_dict[col_name]
                st.session_state.setdefault("outlier_rules", {})[col_name] = {"strategy": global_outlier_strategy, "lower": outlier_bounds["Lower"], "upper": outlier_bounds["Upper"]}
                
            st.session_state["working_df"] = dataframe_work
            st.session_state["cleaning_confirmed"] = False
            st.rerun()
    with apply_all_col:
        if st.button("Apply All", key="out_apply_all", width="stretch"):
            dataframe_work = st.session_state["working_df"]
            strategies_to_apply_dict = {}
            for col_name, outlier_bounds in outlier_columns_dict.items():
                strategies_to_apply_dict[col_name] = {"strategy": global_outlier_strategy, "lower": outlier_bounds["Lower"], "upper": outlier_bounds["Upper"]}

            dataframe_work = use_outlier_strategy_bulk(dataframe_work, strategies_to_apply_dict)

            treated_outliers = st.session_state.setdefault("treated_outlier_cols", {})
            for col_name in outlier_columns_dict:
                track_cleaning("outlier", col_name, global_outlier_strategy)
                treated_outliers[col_name] = global_outlier_strategy
                outlier_bounds = outlier_columns_dict[col_name]
                st.session_state.setdefault("outlier_rules", {})[col_name] = {"strategy": global_outlier_strategy, "lower": outlier_bounds["Lower"], "upper": outlier_bounds["Upper"]}

            st.session_state["working_df"] = dataframe_work
            st.session_state["cleaning_confirmed"] = False
            st.rerun()

    _, rec_btn_col = st.columns([4, 2])
    with rec_btn_col:
        if st.button("Apply All (Recommend)", key="out_apply_recommend", type="primary", width="stretch"):
            dataframe_work = st.session_state["working_df"]
            strategies_to_apply_dict = {}
            explanations_dict = {}
            for col_name, outlier_bounds in outlier_columns_dict.items():
                rec_strategy, _, rec_explanation = _suggest_outlier_strategy(outlier_bounds["Outliers"], len(dataframe_work))
                strategies_to_apply_dict[col_name] = {"strategy": rec_strategy, "lower": outlier_bounds["Lower"], "upper": outlier_bounds["Upper"]}
                explanations_dict[col_name] = rec_explanation

            dataframe_work = use_outlier_strategy_bulk(dataframe_work, strategies_to_apply_dict)

            treated_outliers = st.session_state.setdefault("treated_outlier_cols", {})
            for col_name, outlier_bounds in outlier_columns_dict.items():
                rec_strategy = strategies_to_apply_dict[col_name]["strategy"]
                track_cleaning("outlier", col_name, rec_strategy, explanations_dict.get(col_name, ""))
                treated_outliers[col_name] = rec_strategy
                st.session_state.setdefault("outlier_rules", {})[col_name] = {"strategy": rec_strategy, "lower": outlier_bounds["Lower"], "upper": outlier_bounds["Upper"]}

            st.session_state["working_df"] = dataframe_work
            st.session_state["cleaning_confirmed"] = False
            st.rerun()

    st.caption(OUTLIER_STRATEGY_INFO.get(global_outlier_strategy, ""))
    st.markdown(HORIZONTAL_RULE_HTML, unsafe_allow_html=True)

    last_outlier_column = list(outlier_columns_dict)[-1]
    for col_name, outlier_bounds in outlier_columns_dict.items():
        outlier_count = outlier_bounds["Outliers"]
        outlier_reason = outlier_bounds["Reason"]
        lower_bound = outlier_bounds["Lower"]
        upper_bound = outlier_bounds["Upper"]

        suggested_out_strategy, suggested_out_rule_id, suggested_out_explanation = _suggest_outlier_strategy(outlier_count, len(working_dataframe))
        out_options = ["clip", "drop rows"]
        out_default_idx = out_options.index(suggested_out_strategy) if suggested_out_strategy in out_options else 0

        checkbox_col, name_col, strategy_col, apply_col, _ = st.columns([0.4, 2.8, 2, 0.8, 0.5], vertical_alignment="center")
        with checkbox_col:
            st.checkbox("Select", key=f"out_check_{col_name}", label_visibility="hidden")
        with name_col:
            rule_badge = f" `{suggested_out_rule_id}`" if suggested_out_rule_id else ""
            st.markdown(f"**{col_name}**  {outlier_count:,} ค่า{rule_badge}")
            st.caption(outlier_reason)
            st.caption(f"ขอบเขต: [{lower_bound:,.2f}, {upper_bound:,.2f}]")
        with strategy_col:
            chosen_strategy = st.selectbox("Strategy", out_options, index=out_default_idx, key=f"out_strategy_{col_name}", label_visibility="collapsed")
        with apply_col:
            if st.button("Apply", key=f"out_apply_{col_name}"):
                dataframe_work = use_outlier_strategy(st.session_state["working_df"], col_name, chosen_strategy, lower_bound, upper_bound)
                st.session_state["working_df"] = dataframe_work
                st.session_state["cleaning_confirmed"] = False
                st.session_state.setdefault("treated_outlier_cols", {})[col_name] = chosen_strategy
                track_cleaning("outlier", col_name, chosen_strategy, suggested_out_explanation)
                st.session_state.setdefault("outlier_rules", {})[col_name] = {"strategy": chosen_strategy, "lower": lower_bound, "upper": upper_bound}
                st.rerun()

        st.caption(OUTLIER_STRATEGY_INFO.get(chosen_strategy, ""))

        treated_columns = st.session_state.get("treated_outlier_cols", {})
        if col_name in treated_columns:
            st.success(f"**{col_name}** ถูกจัดการ Outliers เรียบร้อยแล้ว")

        if col_name != last_outlier_column:
            st.markdown(HORIZONTAL_RULE_HTML, unsafe_allow_html=True)