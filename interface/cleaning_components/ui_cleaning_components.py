import streamlit as st
import pandas as pd
from data_prepare.logic.data_type_detection import actual_type
from data_prepare.logic.cleaning_logic import (
    use_missing_strategy, 
    use_outlier_strategy,
    use_missing_strategy_bulk,
    use_outlier_strategy_bulk
)
from explainable.state_manager.trace_log import track_cleaning, track_cleaning_bulk

MISSING_STRATEGY_INFO = {
    "mean": "แทนค่าว่างด้วยค่าเฉลี่ย — เหมาะกับข้อมูล Continuous ที่กระจายแบบ Normal",
    "median": "แทนค่าว่างด้วยค่ากลาง — เหมาะกับข้อมูลที่มี Skew หรือมี Outlier",
    "median (rounded)": "แทนค่าว่างด้วยค่ากลาง แล้วปัดเป็นจำนวนเต็ม — เหมาะกับข้อมูล Discrete เช่น อายุ จำนวนสินค้า",
    "most frequent": "แทนค่าว่างด้วยค่าที่พบบ่อยสุด (Mode) — เหมาะกับข้อมูล Categorical หรือ Discrete",
    "forward fill": "ใช้ค่าแถวก่อนหน้ามาแทน (Forward Fill) — เหมาะกับข้อมูลที่เรียงตามเวลา เช่น Time Series (อ้างอิง Topic 7)",
    "backward fill": "ใช้ค่าแถวถัดไปมาแทน (Backward Fill) — เหมาะกับข้อมูลที่เรียงตามเวลา เช่น Time Series (อ้างอิง Topic 7)",
    "drop rows": "ลบทั้งแถวที่มีค่าว่าง (Listwise Deletion) — ข้อมูลสะอาด แต่อาจสูญเสียข้อมูล",
}

OUTLIER_STRATEGY_INFO = {
    "clip": "ตัดค่าให้อยู่ในขอบเขต — เก็บทุกแถวไว้ แต่ลดอิทธิพลของค่าสุดโต่ง",
    "drop rows": "ลบแถวที่มีค่าเกินขอบเขต — เหมาะเมื่อค่าสุดโต่งเป็นข้อผิดพลาด",
}

HORIZONTAL_RULE_HTML = (
    "<hr style='margin:0.75rem 0;border:none;"
    "border-top:1px solid rgba(255,255,255,0.06)'>"
)

ACTION_BAR_COLUMNS = [0.9, 1.1, 0.2, 2, 1.1, 0.9]

def format_percentage(count: int, percentage: float) -> str:
    if count == 0:
        return "0 (0.0%)"
    percentage_string = f"{percentage:.1f}%" if percentage >= 0.1 else "< 0.1%"
    return f"{count:,} ({percentage_string})"

def color_changed(column):
    style_list = []
    for index, value in enumerate(column):
        if value == 0:
            style_list.append("color: rgba(255,255,255,0.35)")
        elif index == 0:
            style_list.append("color: #f87171" if value < 0 else "color: rgba(255,255,255,0.35)")
        else:
            style_list.append("color: #4ade80" if value < 0 else "color: #f87171")
    return style_list

def render_profile_tab(working_dataframe: pd.DataFrame, outlier_details: list):
    st.subheader("Data Profile")

    outlier_dictionary = {item["Column"]: item for item in outlier_details}
    profile_data_list = []
    for column_name in working_dataframe.columns:
        data_series = working_dataframe[column_name]
        profile_data_list.append({
            "Column": column_name,
            "Missing": int(data_series.isnull().sum()),
            "Outliers": outlier_dictionary.get(column_name, {}).get("Outliers", 0),
            "Unique": int(data_series.nunique()),
        })

    st.dataframe(
        pd.DataFrame(profile_data_list),
        width="stretch",
        hide_index=True,
        column_config={
            "Column": st.column_config.TextColumn("Column"),
            "Missing": st.column_config.NumberColumn("Missing", format="%d"),
            "Unique": st.column_config.ProgressColumn(
                "Unique", min_value=0, max_value=working_dataframe.shape[0], format="%d"
            ),
            "Outliers": st.column_config.NumberColumn("Outliers", format="%d"),
        },
    )

def render_drop_columns(working_dataframe: pd.DataFrame, target_column: str):
    st.subheader("Drop Columns")

    with st.expander("Drop Columns คืออะไร?", expanded=False):
        st.markdown(
            "ลบคอลัมน์ที่ไม่ต้องการออกจาก Dataset เช่น ID column, free-text column, "
            "หรือคอลัมน์ที่มีค่าว่างมากเกินไป ซึ่งไม่มีประโยชน์ต่อการสร้างโมเดล"
        )

    if not target_column or target_column not in working_dataframe.columns:
        target_column = None
        st.warning("ไม่พบ Target Column — กรุณากลับไปเลือก Target ที่หน้า Upload ก่อน Drop Columns")

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
        suggested_drops_list = [
            col for col in droppable_columns
            if (working_dataframe[col].isnull().mean() > 0.8) or (working_dataframe[col].nunique() <= 1)
        ]
        if suggested_drops_list and st.button("Drop แนะนำ", key="drop_cols_suggested", disabled=target_column is None):
            remaining_columns = [col for col in working_dataframe.columns if col not in suggested_drops_list]
            if len(remaining_columns) < 2:
                st.error("ต้องเหลือคอลัมน์อย่างน้อย 2 คอลัมน์")
            else:
                st.session_state["working_df"] = (
                    working_dataframe.drop(columns=suggested_drops_list).reset_index(drop=True)
                )
                st.session_state["cleaning_confirmed"] = False
                track_cleaning_bulk("drop_col", suggested_drops_list, "dropped (missing>80% or single unique)")
                st.rerun()

    if suggested_drops_list:
        st.caption(f"แนะนำให้ลบ: {', '.join(suggested_drops_list)} (missing > 80% หรือมีค่าเดียว)")

def render_duplicates(working_dataframe: pd.DataFrame, duplicate_count: int):
    st.subheader("Duplicates")

    if duplicate_count == 0:
        st.success("ไม่พบแถวซ้ำ")
    else:
        with st.expander("Duplicates คืออะไร?", expanded=False):
            st.markdown(
                "**Duplicated Entries** คือแถวข้อมูลที่ซ้ำกันทุกคอลัมน์ "
                "อาจเกิดจากการบันทึกซ้ำ, import ข้อมูลซ้อน, หรือ system error "
                "ส่งผลให้โมเดลให้น้ำหนักกับข้อมูลนั้นมากเกินจริง"
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

    with st.expander("Missing Data คืออะไร? (อ้างอิง Topic 7)", expanded=False):
        st.markdown(
            "**Missing Data** คือค่าว่างที่ขาดหายไปในชุดข้อมูล "
            "อาจเกิดจากการป้อนข้อมูลไม่สมบูรณ์ อุปกรณ์ทำงานผิดพลาด หรือไฟล์สูญหาย\n\n"
            "**ประเภทของ Missing Data** (Little and Rubin, 1987):\n"
            "- **MCAR** (Missing Completely at Random): หายไปโดยบังเอิญ ไม่เกี่ยวกับตัวแปรใดเลย\n"
            "- **MAR** (Missing at Random): การหายไปเกี่ยวข้องกับตัวแปรอื่น เช่น ผู้หญิงมักไม่เปิดเผยน้ำหนัก\n"
            "- **MNAR** (Missing Not at Random): การหายไปเกี่ยวข้องกับค่าของตัวเอง เช่น คนน้ำหนักมากมักไม่ตอบ\n\n"
            "**วิธีจัดการ** (อ้างอิง Topic 7 — Handling Missing Data):\n"
            "- **Mean/Median Imputation**: แทนด้วยค่ากลาง เหมาะกับข้อมูลตัวเลข\n"
            "- **Most Frequent**: แทนด้วย Mode เหมาะกับข้อมูล Categorical\n"
            "- **Forward/Backward Fill**: ใช้ค่าแถวข้างเคียง เหมาะกับ Time Series\n"
            "- **Listwise Deletion (Drop Rows)**: ลบแถวที่มีค่าว่าง"
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
                
            st.session_state["working_df"] = dataframe_work
            st.session_state["cleaning_confirmed"] = False
            st.rerun()
    st.caption(MISSING_STRATEGY_INFO.get(global_missing_strategy, ""))
    st.markdown(HORIZONTAL_RULE_HTML, unsafe_allow_html=True)

    last_missing_column = list(missing_columns_dict)[-1]
    for col_name, missing_count in missing_columns_dict.items():
        missing_percentage = missing_count / len(working_dataframe) * 100
        column_data_type = actual_type(working_dataframe[col_name])

        checkbox_col, name_col, strategy_col, apply_col, _ = st.columns([0.4, 2.8, 2, 0.8, 0.5], vertical_alignment="center")
        with checkbox_col:
            st.checkbox("Select", key=f"miss_check_{col_name}", label_visibility="hidden")
        with name_col:
            st.markdown(f"**{col_name}** — {missing_count:,} ค่า ({missing_percentage:.1f}%)")
        with strategy_col:
            available_options = determine_missing_compatible(column_data_type)
            chosen_strategy = st.selectbox("Strategy", available_options, key=f"miss_strategy_{col_name}", label_visibility="collapsed")
        with apply_col:
            if st.button("Apply", key=f"miss_apply_{col_name}"):
                dataframe_work = use_missing_strategy(st.session_state["working_df"], col_name, chosen_strategy)
                st.session_state["working_df"] = dataframe_work
                st.session_state["cleaning_confirmed"] = False
                track_cleaning("missing", col_name, chosen_strategy)
                st.rerun()

        st.caption(MISSING_STRATEGY_INFO.get(chosen_strategy, ""))

        if col_name != last_missing_column:
            st.markdown(HORIZONTAL_RULE_HTML, unsafe_allow_html=True)


def render_outliers(working_dataframe: pd.DataFrame, outlier_columns_dict: dict, outlier_details: list):
    st.subheader("Outliers")

    if not outlier_columns_dict:
        st.success("ไม่พบ Outliers")
        return

    with st.expander("ระบบตรวจจับ Outlier อย่างไร? (อ้างอิง Topic 8)", expanded=False):
        st.markdown(
            "**Outlier** คือค่าที่ผิดปกติ ห่างจากข้อมูลส่วนใหญ่อย่างมีนัยสำคัญ "
            "(อ้างอิง Topic 8 — Outlier Detection)\n\n"
            "ระบบเลือกวิธีตรวจจับ **อัตโนมัติตาม Skewness** ของข้อมูล:\n"
            "- **|Skew| < 0.5** → ข้อมูลใกล้ Normal → ใช้ **Z-Score** "
            "(ค่าที่ห่างจาก Mean เกิน **3 SD** — 3-sigma rule ครอบคลุม 99.7% ของข้อมูล Normal)\n"
            "- **|Skew| ≥ 0.5** → ข้อมูลเบ้ → ใช้ **IQR** "
            "(ค่านอกช่วง Q1 − 1.5×IQR ถึง Q3 + 1.5×IQR)\n\n"
            "**เหตุผลที่ใช้ Skewness = 0.5 เป็น threshold**: "
            "|Skew| < 0.5 ถือว่าข้อมูลสมมาตรเพียงพอที่จะสมมติการกระจายแบบ Normal ได้ "
            "จึงใช้ Z-Score ได้อย่างน่าเชื่อถือ (อ้างอิง Topic 2 — Basic Statistical Description of Data)"
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

        checkbox_col, name_col, strategy_col, apply_col, _ = st.columns([0.4, 2.8, 2, 0.8, 0.5], vertical_alignment="center")
        with checkbox_col:
            st.checkbox("Select", key=f"out_check_{col_name}", label_visibility="hidden")
        with name_col:
            st.markdown(f"**{col_name}** — {outlier_count:,} ค่า")
            st.caption(outlier_reason)
            st.caption(f"ขอบเขต: [{lower_bound:,.2f}, {upper_bound:,.2f}]")
        with strategy_col:
            chosen_strategy = st.selectbox("Strategy", ["clip", "drop rows"], key=f"out_strategy_{col_name}", label_visibility="collapsed")
        with apply_col:
            if st.button("Apply", key=f"out_apply_{col_name}"):
                dataframe_work = use_outlier_strategy(st.session_state["working_df"], col_name, chosen_strategy, lower_bound, upper_bound)
                st.session_state["working_df"] = dataframe_work
                st.session_state["cleaning_confirmed"] = False
                st.session_state.setdefault("treated_outlier_cols", {})[col_name] = chosen_strategy
                track_cleaning("outlier", col_name, chosen_strategy)
                st.rerun()

        st.caption(OUTLIER_STRATEGY_INFO.get(chosen_strategy, ""))

        treated_columns = st.session_state.get("treated_outlier_cols", {})
        if col_name in treated_columns:
            st.success(f"**{col_name}** ถูกจัดการ Outliers เรียบร้อยแล้ว")

        if col_name != last_outlier_column:
            st.markdown(HORIZONTAL_RULE_HTML, unsafe_allow_html=True)


def render_summary(working_dataframe: pd.DataFrame, original_dataframe: pd.DataFrame, duplicate_before: int, outlier_before: int, total_missing: int, duplicate_count: int, total_outlier: int):
    st.subheader("Summary")

    if st.session_state.get("cleaning_confirmed") and "cleaning_summary_snapshot" in st.session_state:
        snapshot = st.session_state["cleaning_summary_snapshot"]
        before_state = snapshot["before"]
        after_state = snapshot["after"]
        changed_values = [
            after_state["rows"] - before_state["rows"],
            after_state["cols"] - before_state["cols"],
            after_state["missing"] - before_state["missing"],
            after_state["dups"] - before_state["dups"],
            after_state["outliers"] - before_state["outliers"],
        ]
        summary_dataframe = pd.DataFrame({
            "Metric": ["Rows", "Columns", "Missing Values", "Duplicates", "Outliers"],
            "Before": [f"{before_state['rows']:,}", f"{before_state['cols']}", f"{before_state['missing']:,}", f"{before_state['dups']:,}", f"{before_state['outliers']:,}"],
            "After":  [f"{after_state['rows']:,}", f"{after_state['cols']}", f"{after_state['missing']:,}", f"{after_state['dups']:,}", f"{after_state['outliers']:,}"],
            "Changed": changed_values,
        })
    else:
        changed_values = [
            working_dataframe.shape[0] - original_dataframe.shape[0],
            working_dataframe.shape[1] - original_dataframe.shape[1],
            total_missing - int(original_dataframe.isnull().sum().sum()),
            duplicate_count - duplicate_before,
            total_outlier - outlier_before,
        ]
        summary_dataframe = pd.DataFrame({
            "Metric": ["Rows", "Columns", "Missing Values", "Duplicates", "Outliers"],
            "Before": [f"{original_dataframe.shape[0]:,}", f"{original_dataframe.shape[1]}", f"{int(original_dataframe.isnull().sum().sum()):,}", f"{duplicate_before:,}", f"{outlier_before:,}"],
            "After": [f"{working_dataframe.shape[0]:,}", f"{working_dataframe.shape[1]}", f"{int(working_dataframe.isnull().sum().sum()):,}", f"{duplicate_count:,}", f"{total_outlier:,}"],
            "Changed": changed_values,
        })

    styled_summary = (
        summary_dataframe.style
        .apply(color_changed, subset=["Changed"])
        .format({"Changed": lambda value: "—" if value == 0 else f"+{value:,}" if value > 0 else f"{value:,}"})
    )
    st.dataframe(styled_summary, width="stretch", hide_index=True)
