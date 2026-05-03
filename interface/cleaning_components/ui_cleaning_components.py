import streamlit as st
import pandas as pd
from data_prepare.features.data_type_detection import actual_type
from data_prepare.features.cleaning_logic import (
    use_missing_strategy, 
    use_outlier_strategy,
    use_missing_strategy_bulk,
    use_outlier_strategy_bulk
)
from explainable.features.trace_log import track_cleaning, track_cleaning_bulk

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

_HR = (
    "<hr style='margin:0.75rem 0;border:none;"
    "border-top:1px solid rgba(255,255,255,0.06)'>"
)

_ACTION_BAR_COLS = [0.9, 1.1, 0.2, 2, 1.1, 0.9]

def _fmt_pct(count: int, pct: float) -> str:
    if count == 0:
        return "0 (0.0%)"
    pct_str = f"{pct:.1f}%" if pct >= 0.1 else "< 0.1%"
    return f"{count:,} ({pct_str})"

def _color_changed(col):
    styles = []
    for i, val in enumerate(col):
        if val == 0:
            styles.append("color: rgba(255,255,255,0.35)")
        elif i == 0:
            styles.append("color: #f87171" if val < 0 else "color: rgba(255,255,255,0.35)")
        else:
            styles.append("color: #4ade80" if val < 0 else "color: #f87171")
    return styles

def render_profile_tab(working_df: pd.DataFrame, outls_details: list):
    st.subheader("Data Profile")

    outlier_dict = {item["Column"]: item for item in outls_details}
    profile_list = []
    for col in working_df.columns:
        series = working_df[col]
        profile_list.append({
            "Column": col,
            "Missing": int(series.isnull().sum()),
            "Outliers": outlier_dict.get(col, {}).get("Outliers", 0),
            "Unique": int(series.nunique()),
        })

    st.dataframe(
        pd.DataFrame(profile_list),
        width="stretch",
        hide_index=True,
        column_config={
            "Column": st.column_config.TextColumn("Column"),
            "Missing": st.column_config.NumberColumn("Missing", format="%d"),
            "Unique": st.column_config.ProgressColumn(
                "Unique", min_value=0, max_value=working_df.shape[0], format="%d"
            ),
            "Outliers": st.column_config.NumberColumn("Outliers", format="%d"),
        },
    )

def render_drop_columns(working_df: pd.DataFrame, target_col: str):
    st.subheader("Drop Columns")

    with st.expander("Drop Columns คืออะไร?", expanded=False):
        st.markdown(
            "ลบคอลัมน์ที่ไม่ต้องการออกจาก Dataset เช่น ID column, free-text column, "
            "หรือคอลัมน์ที่มีค่าว่างมากเกินไป ซึ่งไม่มีประโยชน์ต่อการสร้างโมเดล"
        )

    if not target_col or target_col not in working_df.columns:
        target_col = None
        st.warning("ไม่พบ Target Column — กรุณากลับไปเลือก Target ที่หน้า Upload ก่อน Drop Columns")

    droppable_cols = [c for c in working_df.columns if c != target_col]

    cols_to_drop = st.multiselect(
        "เลือกคอลัมน์ที่ต้องการลบ",
        options=droppable_cols,
        placeholder="เลือกคอลัมน์...",
        label_visibility="collapsed",
        key="drop_cols_select",
        disabled=target_col is None,
    )

    c_drop1, c_drop2, _ = st.columns([1.5, 1.5, 5])
    with c_drop1:
        if st.button("Drop Selected", key="drop_cols_apply", disabled=not cols_to_drop or target_col is None):
            remaining = [c for c in working_df.columns if c not in cols_to_drop]
            if len(remaining) < 2:
                st.error("ต้องเหลือคอลัมน์อย่างน้อย 2 คอลัมน์")
            else:
                st.session_state["working_df"] = (
                    working_df.drop(columns=cols_to_drop).reset_index(drop=True)
                )
                st.session_state["cleaning_confirmed"] = False
                track_cleaning_bulk("drop_col", cols_to_drop, "dropped manually")
                st.rerun()
    with c_drop2:
        suggested_drops = [
            c for c in droppable_cols
            if (working_df[c].isnull().mean() > 0.8) or (working_df[c].nunique() <= 1)
        ]
        if suggested_drops and st.button("Drop แนะนำ", key="drop_cols_suggested", disabled=target_col is None):
            remaining = [c for c in working_df.columns if c not in suggested_drops]
            if len(remaining) < 2:
                st.error("ต้องเหลือคอลัมน์อย่างน้อย 2 คอลัมน์")
            else:
                st.session_state["working_df"] = (
                    working_df.drop(columns=suggested_drops).reset_index(drop=True)
                )
                st.session_state["cleaning_confirmed"] = False
                track_cleaning_bulk("drop_col", suggested_drops, "dropped (missing>80% or single unique)")
                st.rerun()

    if suggested_drops:
        st.caption(f"แนะนำให้ลบ: {', '.join(suggested_drops)} (missing > 80% หรือมีค่าเดียว)")

def render_duplicates(working_df: pd.DataFrame, duplicate_count: int):
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
                working_df.drop_duplicates().reset_index(drop=True)
            )
            st.session_state["cleaning_confirmed"] = False
            track_cleaning("drop_dup", "_duplicates_", f"dropped {duplicate_count:,} duplicate rows")
            st.rerun()

def _miss_compatible(col_type: str) -> list:
    if col_type == "float":
        return ["mean", "median", "forward fill", "backward fill", "drop rows"]
    elif col_type == "int":
        return ["median (rounded)", "most frequent", "forward fill", "backward fill", "drop rows"]
    elif col_type == "datetime":
        return ["forward fill", "backward fill", "drop rows"]
    else:
        return ["most frequent", "forward fill", "backward fill", "drop rows"]

def render_missing_values(working_df: pd.DataFrame, missing_cols: dict, null_counts: pd.Series):
    st.subheader("Missing Values")

    if not missing_cols:
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

    col_sel_all, col_desel_all, _, col_global_strat, col_apply_sel, col_apply_all = st.columns(_ACTION_BAR_COLS)
    with col_sel_all:
        if st.button("Select All", key="miss_sel_all", width="stretch"):
            for col in missing_cols:
                st.session_state[f"miss_check_{col}"] = True
            st.rerun()
    with col_desel_all:
        if st.button("Deselect All", key="miss_desel_all", width="stretch"):
            for col in missing_cols:
                st.session_state[f"miss_check_{col}"] = False
            st.rerun()
    with col_global_strat:
        global_miss_strategy = st.selectbox(
            "Global Strategy",
            ["mean", "median", "median (rounded)", "most frequent", "forward fill", "backward fill", "drop rows"],
            key="miss_global_strategy",
            label_visibility="collapsed",
        )
    checked_miss_cols = [
        col for col in missing_cols
        if st.session_state.get(f"miss_check_{col}", False)
    ]

    with col_apply_sel:
        if st.button("Apply Selected", key="miss_apply_selected", disabled=not checked_miss_cols, width="stretch"):
            df_work = st.session_state["working_df"]
            strategies_to_apply = {}
            for col in checked_miss_cols:
                compatible = _miss_compatible(actual_type(df_work[col]))
                strategy = global_miss_strategy if global_miss_strategy in compatible else compatible[0]
                strategies_to_apply[col] = strategy
            
            df_work = use_missing_strategy_bulk(df_work, strategies_to_apply)
            for col, strat in strategies_to_apply.items():
                track_cleaning("missing", col, strat)
                
            st.session_state["working_df"] = df_work
            st.session_state["cleaning_confirmed"] = False
            st.rerun()
    with col_apply_all:
        if st.button("Apply All", key="miss_apply_all", width="stretch"):
            df_work = st.session_state["working_df"]
            strategies_to_apply = {}
            for col in missing_cols:
                compatible = _miss_compatible(actual_type(df_work[col]))
                strategy = global_miss_strategy if global_miss_strategy in compatible else compatible[0]
                strategies_to_apply[col] = strategy
                
            df_work = use_missing_strategy_bulk(df_work, strategies_to_apply)
            for col, strat in strategies_to_apply.items():
                track_cleaning("missing", col, strat)
                
            st.session_state["working_df"] = df_work
            st.session_state["cleaning_confirmed"] = False
            st.rerun()
    st.caption(MISSING_STRATEGY_INFO.get(global_miss_strategy, ""))
    st.markdown(_HR, unsafe_allow_html=True)

    last_missing_col = list(missing_cols)[-1]
    for col, count in missing_cols.items():
        pct = count / len(working_df) * 100
        col_type = actual_type(working_df[col])

        col_check, col_name, col_strategy, col_apply, _ = st.columns([0.4, 2.8, 2, 0.8, 0.5], vertical_alignment="center")
        with col_check:
            st.checkbox("Select", key=f"miss_check_{col}", label_visibility="hidden")
        with col_name:
            st.markdown(f"**{col}** — {count:,} ค่า ({pct:.1f}%)")
        with col_strategy:
            options = _miss_compatible(col_type)
            strategy = st.selectbox("Strategy", options, key=f"miss_strategy_{col}", label_visibility="collapsed")
        with col_apply:
            if st.button("Apply", key=f"miss_apply_{col}"):
                df_work = use_missing_strategy(st.session_state["working_df"], col, strategy)
                st.session_state["working_df"] = df_work
                st.session_state["cleaning_confirmed"] = False
                track_cleaning("missing", col, strategy)
                st.rerun()

        st.caption(MISSING_STRATEGY_INFO.get(strategy, ""))

        if col != last_missing_col:
            st.markdown(_HR, unsafe_allow_html=True)


def render_outliers(working_df: pd.DataFrame, outlier_cols: dict, outls_details: list):
    st.subheader("Outliers")

    if not outlier_cols:
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

    col_sel_all, col_desel_all, _, col_global_strat, col_apply_sel, col_apply_all = st.columns(_ACTION_BAR_COLS)
    with col_sel_all:
        if st.button("Select All", key="out_sel_all", width="stretch"):
            for col in outlier_cols:
                st.session_state[f"out_check_{col}"] = True
            st.rerun()
    with col_desel_all:
        if st.button("Deselect All", key="out_desel_all", width="stretch"):
            for col in outlier_cols:
                st.session_state[f"out_check_{col}"] = False
            st.rerun()
    with col_global_strat:
        global_out_strategy = st.selectbox(
            "Global Strategy",
            ["clip", "drop rows"],
            key="out_global_strategy",
            label_visibility="collapsed",
        )
    checked_out_cols = [
        col for col in outlier_cols
        if st.session_state.get(f"out_check_{col}", False)
    ]
    with col_apply_sel:
        if st.button("Apply Selected", key="out_apply_selected", disabled=not checked_out_cols, width="stretch"):
            df_work = st.session_state["working_df"]
            strategies_to_apply = {}
            for col in checked_out_cols:
                bounds = outlier_cols[col]
                strategies_to_apply[col] = {"strategy": global_out_strategy, "lower": bounds["Lower"], "upper": bounds["Upper"]}
            
            df_work = use_outlier_strategy_bulk(df_work, strategies_to_apply)
            
            treated = st.session_state.setdefault("_treated_outlier_cols", {})
            for col in checked_out_cols:
                track_cleaning("outlier", col, global_out_strategy)
                treated[col] = global_out_strategy
                
            st.session_state["working_df"] = df_work
            st.session_state["cleaning_confirmed"] = False
            st.rerun()
    with col_apply_all:
        if st.button("Apply All", key="out_apply_all", width="stretch"):
            df_work = st.session_state["working_df"]
            strategies_to_apply = {}
            for col, bounds in outlier_cols.items():
                strategies_to_apply[col] = {"strategy": global_out_strategy, "lower": bounds["Lower"], "upper": bounds["Upper"]}
                
            df_work = use_outlier_strategy_bulk(df_work, strategies_to_apply)
            
            treated = st.session_state.setdefault("_treated_outlier_cols", {})
            for col in outlier_cols:
                track_cleaning("outlier", col, global_out_strategy)
                treated[col] = global_out_strategy
                
            st.session_state["working_df"] = df_work
            st.session_state["cleaning_confirmed"] = False
            st.rerun()
    st.caption(OUTLIER_STRATEGY_INFO.get(global_out_strategy, ""))
    st.markdown(_HR, unsafe_allow_html=True)

    last_outlier_col = list(outlier_cols)[-1]
    for col, bounds in outlier_cols.items():
        count = bounds["Outliers"]
        reason = bounds["Reason"]
        lower = bounds["Lower"]
        upper = bounds["Upper"]

        col_check, col_name, col_strategy, col_apply, _ = st.columns([0.4, 2.8, 2, 0.8, 0.5], vertical_alignment="center")
        with col_check:
            st.checkbox("Select", key=f"out_check_{col}", label_visibility="hidden")
        with col_name:
            st.markdown(f"**{col}** — {count:,} ค่า")
            st.caption(reason)
            st.caption(f"ขอบเขต: [{lower:,.2f}, {upper:,.2f}]")
        with col_strategy:
            strategy = st.selectbox("Strategy", ["clip", "drop rows"], key=f"out_strategy_{col}", label_visibility="collapsed")
        with col_apply:
            if st.button("Apply", key=f"out_apply_{col}"):
                df_work = use_outlier_strategy(st.session_state["working_df"], col, strategy, lower, upper)
                st.session_state["working_df"] = df_work
                st.session_state["cleaning_confirmed"] = False
                st.session_state.setdefault("_treated_outlier_cols", {})[col] = strategy
                track_cleaning("outlier", col, strategy)
                st.rerun()

        st.caption(OUTLIER_STRATEGY_INFO.get(strategy, ""))

        treated_cols = st.session_state.get("_treated_outlier_cols", {})
        if col in treated_cols:
            prev_strategy = treated_cols[col]
            if prev_strategy == "clip":
                st.warning(
                    f"**{col}** ถูก Clip ไปแล้ว แต่ระบบยังตรวจพบ {count:,} Outlier อยู่ "
                    "เนื่องจาก Clip เปลี่ยน distribution → bounds ถูกคำนวณใหม่แล้วแคบลง "
                    "ค่าที่อยู่ขอบจึงยังถูกจับซ้ำ — "
                    "**หากต้องการกำจัดให้หมด ให้เปลี่ยนเป็น Drop Rows แทน**"
                )
            else:
                st.warning(
                    f"**{col}** ถูก Drop Rows ไปแล้ว แต่ระบบยังตรวจพบ {count:,} Outlier อยู่ "
                    "— ลองกด Reset แล้ว Apply ใหม่อีกครั้ง "
                    "หรือเปลี่ยนเป็น **Clip** เพื่อจำกัดค่าแทนการลบแถว"
                )

        if col != last_outlier_col:
            st.markdown(_HR, unsafe_allow_html=True)


def render_summary(working_df: pd.DataFrame, df: pd.DataFrame, dup_before: int, outl_before: int, total_missing: int, duplicate_count: int, total_outl: int):
    st.subheader("Summary")

    if st.session_state.get("cleaning_confirmed") and "cleaning_summary_snapshot" in st.session_state:
        snap = st.session_state["cleaning_summary_snapshot"]
        b = snap["before"]
        a = snap["after"]
        changed_values = [
            a["rows"] - b["rows"],
            a["cols"] - b["cols"],
            a["missing"] - b["missing"],
            a["dups"] - b["dups"],
            a["outliers"] - b["outliers"],
        ]
        summary_df = pd.DataFrame({
            "Metric": ["Rows", "Columns", "Missing Values", "Duplicates", "Outliers"],
            "Before": [f"{b['rows']:,}", f"{b['cols']}", f"{b['missing']:,}", f"{b['dups']:,}", f"{b['outliers']:,}"],
            "After":  [f"{a['rows']:,}", f"{a['cols']}", f"{a['missing']:,}", f"{a['dups']:,}", f"{a['outliers']:,}"],
            "Changed": changed_values,
        })
    else:
        changed_values = [
            working_df.shape[0] - df.shape[0],
            working_df.shape[1] - df.shape[1],
            total_missing - int(df.isnull().sum().sum()),
            duplicate_count - dup_before,
            total_outl - outl_before,
        ]
        summary_df = pd.DataFrame({
            "Metric": ["Rows", "Columns", "Missing Values", "Duplicates", "Outliers"],
            "Before": [f"{df.shape[0]:,}", f"{df.shape[1]}", f"{int(df.isnull().sum().sum()):,}", f"{dup_before:,}", f"{outl_before:,}"],
            "After": [f"{working_df.shape[0]:,}", f"{working_df.shape[1]}", f"{int(working_df.isnull().sum().sum()):,}", f"{duplicate_count:,}", f"{total_outl:,}"],
            "Changed": changed_values,
        })

    styled_summary = (
        summary_df.style
        .apply(_color_changed, subset=["Changed"])
        .format({"Changed": lambda x: "—" if x == 0 else f"+{x:,}" if x > 0 else f"{x:,}"})
    )
    st.dataframe(styled_summary, width="stretch", hide_index=True)
