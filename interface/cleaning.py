import streamlit as st
import pandas as pd
import os
from data_prepare.features.data_distribute import data_distribution
from data_prepare.features.loading_data import save_cleaned_data
from data_prepare.features.cleaning_logic import use_missing_strategy, use_outlier_strategy
from data_prepare.features.data_type_detection import actual_type


MISSING_STRATEGY_INFO = {
    "mean": "แทนค่าว่างด้วยค่าเฉลี่ย — เหมาะกับข้อมูล Continuous ที่กระจายแบบ Normal",
    "median": "แทนค่าว่างด้วยค่ากลาง — เหมาะกับข้อมูลที่มี Skew หรือมี Outlier",
    "median (rounded)": "แทนค่าว่างด้วยค่ากลาง แล้วปัดเป็นจำนวนเต็ม — เหมาะกับข้อมูล Discrete เช่น อายุ จำนวนสินค้า",
    "most frequent": "แทนค่าว่างด้วยค่าที่พบบ่อยสุด (Mode) — เหมาะกับข้อมูล Categorical หรือ Discrete",
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


def _color_changed(col):
    styles = []
    for i, val in enumerate(col):
        if val == 0:
            styles.append("color: rgba(255,255,255,0.35)")
        elif i == 0:  # Rows: ลดลง = เสียข้อมูล
            styles.append("color: #f87171" if val < 0 else "color: rgba(255,255,255,0.35)")
        else:  # Missing / Duplicates / Outliers: ลดลง = ดี
            styles.append("color: #4ade80" if val < 0 else "color: #f87171")
    return styles


def render_cleaning():
    from app import page_header

    page_header(
        "Data Cleaning",
        "กระบวนการลบหรือแก้ไขข้อมูลที่ผิดพลาด ไม่สมบูรณ์ หรือไม่มีความสอดคล้องกันจากชุดข้อมูล",
    )

    if st.session_state.get("main_df") is None:
        from app import navigate
        navigate("upload")
        return

    df = st.session_state["main_df"]
    file_name = st.session_state.get("last_uploaded_file", "Unknown File")

    st.info(f"**Current Dataset:** {file_name}")
    with st.expander("Raw Data"):
        st.dataframe(df, width="stretch")

    # ── working_df init ───────────────────────────────────────
    current_shape = df.shape
    if (
        "working_df" not in st.session_state
        or st.session_state.get("working_df_source_shape") != current_shape
    ):
        st.session_state["working_df"] = df.copy()
        st.session_state["working_df_source_shape"] = current_shape
        st.session_state["original_dup_count"] = int(df.duplicated().sum())
        st.session_state.pop("original_outlier_count", None)

    working_df = st.session_state["working_df"]

    # ── Dataset Overview ──────────────────────────────────────
    st.subheader("Dataset Overview")

    # cache ผลลัพธ์ตาม shape+hash เพื่อไม่ต้องคำนวณใหม่ทุก rerun
    _dist_key = ("_dist_cache", working_df.shape, int(pd.util.hash_pandas_object(working_df).sum()))
    if st.session_state.get("_dist_key") != _dist_key:
        with st.spinner("Calculating Data..."):
            total_outl, outls_details = data_distribution(working_df)
        st.session_state["_dist_key"] = _dist_key
        st.session_state["_dist_result"] = (total_outl, outls_details)
    else:
        total_outl, outls_details = st.session_state["_dist_result"]


    if "original_outlier_count" not in st.session_state:
        st.session_state["original_outlier_count"] = total_outl

    total_cells = working_df.size
    total_missing = working_df.isnull().sum().sum()
    missing_pct = (total_missing / total_cells * 100) if total_cells > 0 else 0
    duplicate_count = int(working_df.duplicated().sum())
    dup_pct = (duplicate_count / working_df.shape[0] * 100) if working_df.shape[0] > 0 else 0
    outlier_pct = (total_outl / total_cells * 100) if total_cells > 0 else 0

    m1, m2, m3, m4, m5 = st.columns(5)
    m1.metric("Rows", f"{working_df.shape[0]:,}")
    m2.metric("Columns", working_df.shape[1])
    def _fmt_pct(count, pct):
        if count == 0:
            return "0 (0.0%)"
        pct_str = f"{pct:.1f}%" if pct >= 0.1 else "< 0.1%"
        return f"{count:,} ({pct_str})"

    m3.metric("Missing Values", _fmt_pct(total_missing, missing_pct))
    m4.metric("Duplicate Rows", _fmt_pct(duplicate_count, dup_pct))
    m5.metric("Outliers", _fmt_pct(total_outl, outlier_pct))

    tab1, tab2 = st.tabs(["Profile", "Cleaning"], width="stretch")

    # ── Tab 1: Profile ────────────────────────────────────────
    with tab1:
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

    # ── Tab 2: Cleaning ───────────────────────────────────────
    with tab2:
        working_df = st.session_state["working_df"]

        # ── Duplicates ────────────────────────────────────────
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
                st.rerun()

        st.divider()

        # ── Missing Values ────────────────────────────────────
        st.subheader("Missing Values")
        missing_cols = {
            col: int(working_df[col].isnull().sum())
            for col in working_df.columns
            if working_df[col].isnull().sum() > 0
        }

        if not missing_cols:
            st.success("ไม่มี Missing Values")
        else:
            with st.expander("Missing Data คืออะไร?", expanded=False):
                st.markdown(
                    "**Missing Data** คือค่าว่างที่ขาดหายไปในชุดข้อมูล "
                    "อาจเกิดจากการป้อนข้อมูลไม่สมบูรณ์ อุปกรณ์ทำงานผิดพลาด หรือไฟล์สูญหาย\n\n"
                    "**ประเภทของ Missing Data**:\n"
                    "- **MCAR** (Missing Completely at Random): หายไปโดยบังเอิญ ไม่เกี่ยวกับตัวแปรใดเลย\n"
                    "- **MAR** (Missing at Random): การหายไปเกี่ยวข้องกับตัวแปรอื่น เช่น ผู้หญิงมักไม่เปิดเผยน้ำหนัก\n"
                    "- **MNAR** (Missing Not at Random): การหายไปเกี่ยวข้องกับค่าของตัวเอง เช่น คนน้ำหนักมากมักไม่ตอบ"
                )

            last_missing_col = list(missing_cols.keys())[-1]
            for col, count in missing_cols.items():
                pct = count / len(working_df) * 100
                col_type = actual_type(working_df[col])

                st.markdown(f"**{col}** — {count:,} ค่า ({pct:.1f}%)")

                c1, c2, _ = st.columns([2, 0.8, 3.2])
                with c1:
                    if col_type == "float":
                        options = ["mean", "median", "drop rows"]
                    elif col_type == "int":
                        options = ["median (rounded)", "most frequent", "drop rows"]
                    else:
                        options = ["most frequent", "drop rows"]
                    strategy = st.selectbox(
                        "Strategy",
                        options,
                        key=f"miss_strategy_{col}",
                        label_visibility="collapsed",
                    )
                with c2:
                    if st.button("Apply", key=f"miss_apply_{col}"):
                        wdf = use_missing_strategy(st.session_state["working_df"], col, strategy)
                        st.session_state["working_df"] = wdf
                        st.rerun()

                st.caption(MISSING_STRATEGY_INFO.get(strategy, ""))

                if col != last_missing_col:
                    st.markdown(_HR, unsafe_allow_html=True)

        st.divider()

        # ── Outliers ──────────────────────────────────────────
        st.subheader("Outliers")
        outlier_cols = {
            item["Column"]: item
            for item in outls_details
            if item["Outliers"] > 0
            and pd.api.types.is_numeric_dtype(
                working_df.get(item["Column"], pd.Series())
            )
        }

        if not outlier_cols:
            st.success("ไม่พบ Outliers")
        else:
            with st.expander("ระบบตรวจจับ Outlier อย่างไร?", expanded=False):
                st.markdown(
                    "**Outlier** คือค่าที่ผิดปกติ ห่างจากข้อมูลส่วนใหญ่อย่างมีนัยสำคัญ\n\n"
                    "ระบบเลือกวิธีตรวจจับอัตโนมัติตาม Skewness ของข้อมูล:\n"
                    "- |Skew| < 0.5 ข้อมูลใกล้ Normal ใช้ **Z-Score** "
                    "(ค่าที่ห่างจาก Mean เกิน 3 เท่าของ SD)\n"
                    "- |Skew| >= 0.5 ข้อมูลเบ้ ใช้ **IQR** "
                    "(ค่านอกช่วง Q1 - 1.5 x IQR ถึง Q3 + 1.5 x IQR)"
                )

            last_outlier_col = list(outlier_cols.keys())[-1]
            for col, info in outlier_cols.items():
                count = info["Outliers"]
                reason = info["Reason"]
                lower = info["Lower"]
                upper = info["Upper"]

                st.markdown(f"**{col}** — {count:,} ค่า")
                st.caption(reason)
                st.caption(f"ขอบเขต: [{lower:,.2f}, {upper:,.2f}]")

                c1, c2, _ = st.columns([2, 0.8, 3.2])
                with c1:
                    strategy = st.selectbox(
                        "Strategy",
                        ["clip", "drop rows"],
                        key=f"out_strategy_{col}",
                        label_visibility="collapsed",
                    )
                with c2:
                    if st.button("Apply", key=f"out_apply_{col}"):
                        wdf = use_outlier_strategy(st.session_state["working_df"], col, strategy, lower, upper)
                        st.session_state["working_df"] = wdf
                        st.rerun()

                st.caption(OUTLIER_STRATEGY_INFO.get(strategy, ""))

                if col != last_outlier_col:
                    st.markdown(_HR, unsafe_allow_html=True)

        st.divider()

        # ── Summary ───────────────────────────────────────────
        st.subheader("Summary")

        dup_before = st.session_state.get("original_dup_count", int(df.duplicated().sum()))
        outl_before = st.session_state["original_outlier_count"]

        changed_values = [
            working_df.shape[0] - df.shape[0],
            int(working_df.isnull().sum().sum()) - int(df.isnull().sum().sum()),
            duplicate_count - dup_before,
            total_outl - outl_before,
        ]
        summary_df = pd.DataFrame({
            "Metric": ["Rows", "Missing Values", "Duplicates", "Outliers"],
            "Before": [f"{df.shape[0]:,}", f"{int(df.isnull().sum().sum()):,}", f"{dup_before:,}", f"{outl_before:,}"],
            "After": [f"{working_df.shape[0]:,}", f"{int(working_df.isnull().sum().sum()):,}", f"{duplicate_count:,}", f"{total_outl:,}"],
            "Changed": changed_values,
        })

        styled_summary = (
            summary_df.style
            .apply(_color_changed, subset=["Changed"])
            .format({"Changed": lambda x: "—" if x == 0 else f"+{x:,}" if x > 0 else f"{x:,}"})
        )
        st.dataframe(styled_summary, width="stretch", hide_index=True)

        # ── Confirm / Reset ───────────────────────────────────
        _, cf1, cf2, _ = st.columns([2, 1.5, 1.5, 2])
        with cf1:
            if st.button("Confirm & Save", type="primary", width="stretch"):
                original_filename = st.session_state.get(
                    "last_uploaded_file", "dataset.csv"
                )
                save_cleaned_data(
                    st.session_state["working_df"].copy(),
                    original_filename,
                )
                st.session_state["cleaning_confirmed"] = True
                st.success("บันทึกข้อมูลที่ Cleaned แล้ว")
                st.rerun()
        with cf2:
            if st.button("Reset", type="secondary", width="stretch"):
                st.session_state["working_df"] = df.copy()
                st.session_state["cleaning_confirmed"] = False
                st.info("Reset กลับ original data แล้ว")
                st.rerun()

        # ── Download (หลัง Confirm เท่านั้น) ──────────────────
        if st.session_state.get("cleaning_confirmed"):
            csv_path = st.session_state.get("cleaned_csv_path")
            if csv_path and os.path.exists(csv_path):
                with open(csv_path, "rb") as f:
                    st.download_button(
                        label=f"Download {st.session_state['last_uploaded_file']}",
                        data=f,
                        file_name=st.session_state["last_uploaded_file"],
                        mime="text/csv",
                        width="stretch",
                    )

    # ── Navigation ────────────────────────────────────────────
    st.divider()
    confirmed = st.session_state.get("cleaning_confirmed", False)
    if not confirmed:
        st.info("กรุณากด **Confirm & Save** ก่อนไปขั้นตอนถัดไป")
    col1, _, col2 = st.columns([0.8, 8, 0.8])
    with col1:
        if st.button("Back", type="secondary", width="stretch"):
            from app import navigate
            st.session_state.pop("working_df", None)
            st.session_state.pop("working_df_source_shape", None)
            st.session_state.pop("cleaning_confirmed", None)
            navigate("upload")
    with col2:
        if st.button(
            "Next Step",
            type="primary",
            width="stretch",
            disabled=not confirmed,
        ):
            from app import navigate
            navigate("eda")
