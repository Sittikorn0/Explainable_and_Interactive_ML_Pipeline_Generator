import streamlit as st
import pandas as pd
import os
from features.data_distribute import data_distribution
from features.loading_data import save_cleaned_data


# ── คำอธิบาย Strategy (อ้างอิง Topic 7 & Topic 2) ──────────────────
MISSING_STRATEGY_INFO = {
    "mean": "**Mean (ค่าเฉลี่ย):** แทนค่าว่างด้วยค่าเฉลี่ยของคอลัมน์ "
            "เหมาะกับข้อมูลที่กระจายแบบ Normal และไม่มี Outlier มาก "
            "เพราะ Outlier จะดึงค่าเฉลี่ยให้เบี่ยงเบน",
    "median": "**Median (ค่ากลาง):** แทนค่าว่างด้วยค่ากลางของคอลัมน์ "
              "เหมาะกับข้อมูลที่มี Skew หรือมี Outlier เพราะ Median ไม่ถูกดึงโดยค่าสุดโต่ง",
    "most frequent": "**Most Frequent (Mode/ฐานนิยม):** แทนค่าว่างด้วยค่าที่พบบ่อยที่สุด "
                     "เหมาะกับข้อมูลประเภท Categorical/Nominal เช่น เพศ, สีผม, จังหวัด",
    "drop rows": "**Drop Rows (Listwise Deletion):** ลบทั้งแถวที่มีค่าว่าง "
                 "ข้อดีคือข้อมูลที่เหลือสะอาดครบถ้วน ข้อเสียคือสูญเสียข้อมูลที่อาจมีค่าในคอลัมน์อื่น",
}

OUTLIER_STRATEGY_INFO = {
    "clip": "**Clip (ตัดค่าให้อยู่ในขอบเขต):** ค่าที่เกินขอบเขตจะถูกปรับให้เท่ากับขอบเขต "
            "เหมาะเมื่อต้องการเก็บจำนวนแถวไว้ทั้งหมด แต่ลดอิทธิพลของค่าสุดโต่ง",
    "drop rows": "**Drop Rows (ลบแถว):** ลบแถวที่มีค่าเกินขอบเขตออก "
                 "เหมาะเมื่อค่าสุดโต่งนั้นเป็นข้อมูลผิดพลาด (error) ไม่ใช่ข้อมูลจริง",
}


def render_cleaning():
    from app import page_header

    page_header(
        "Data Cleaning",
        "กระบวนการลบหรือแก้ไขข้อมูลที่ผิดพลาด ไม่สมบูรณ์ หรือไม่มีความสอดคล้องกันจากชุดข้อมูล",
    )

    if st.session_state.get("main_df") is None:
        st.query_params["step"] = "upload"
        st.rerun()
        return

    df = st.session_state["main_df"]
    file_name = st.session_state.get("last_uploaded_file", "Unknown File")

    st.info(f"**Current Dataset:** {file_name}")
    with st.expander("Raw Data"):
        st.dataframe(df, width="stretch")

    # Dataset Overview
    st.subheader("Dataset Overview")
    with st.spinner("Calculating Data..."):
        total_outl, outls_details = data_distribution(df)

    total_cells = df.size
    total_missing = df.isnull().sum().sum()
    missing_pct = (total_missing / total_cells * 100) if total_cells > 0 else 0
    duplicate_count = int(df.duplicated().sum())
    dup_pct = (duplicate_count / df.shape[0] * 100) if df.shape[0] > 0 else 0
    outlier_pct = (total_outl / total_cells * 100) if total_cells > 0 else 0

    m1, m2, m3, m4, m5 = st.columns(5)
    m1.metric("Rows", f"{df.shape[0]:,}")
    m2.metric("Columns", df.shape[1])
    m3.metric("Missing Values", f"{total_missing:,} ({missing_pct:.1f}%)")
    m4.metric("Duplicate Rows", f"{duplicate_count:,} ({dup_pct:.1f}%)")
    m5.metric("Outliers", f"{total_outl:,} ({outlier_pct:.1f}%)")

    tab1, tab2 = st.tabs(["Profile", "Cleaning"], width="stretch")

    # Profile
    with tab1:
        st.subheader("Data Profile")

        outlier_dict = {item["Column"]: item for item in outls_details}
        profile_list = []
        for col in df.columns:
            series = df[col]
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
                    "Unique", min_value=0, max_value=df.shape[0], format="%d"
                ),
                "Outliers": st.column_config.NumberColumn("Outliers", format="%d"),
            },
        )

    # Cleaning
    with tab2:
        # working_df = สำเนาที่ใช้แก้ไข ยังไม่ overwrite main_df
        current_shape = df.shape
        if (
            "working_df" not in st.session_state
            or st.session_state.get("working_df_source_shape") != current_shape
        ):
            st.session_state["working_df"] = df.copy()
            st.session_state["working_df_source_shape"] = current_shape

        working_df = st.session_state["working_df"]

        # Duplicates
        st.subheader("Duplicates")

        with st.expander("Duplicates คืออะไร?"):
            st.markdown(
                "**Duplicated Entries** คือแถวข้อมูลที่ซ้ำกันทุกคอลัมน์ "
                "อาจเกิดจากการบันทึกข้อมูลซ้ำ, import ข้อมูลซ้อน, หรือ system error\n\n"
                "**ทำไมต้องลบ?** แถวซ้ำทำให้โมเดลให้น้ำหนักกับข้อมูลนั้นมากเกินจริง "
                "ส่งผลให้ผลการวิเคราะห์และ prediction เบี่ยงเบน"
            )

        dup_count = int(working_df.duplicated().sum())
        st.write(f"พบแถวซ้ำ **{dup_count:,}** แถว")

        if dup_count > 0:
            if st.button("Drop duplicate rows", key="drop_dup"):
                st.session_state["working_df"] = (
                    working_df.drop_duplicates().reset_index(drop=True)
                )
                st.success(f"ลบแถวซ้ำ {dup_count:,} แถวแล้ว")
                st.rerun()

        st.divider()

        # Missing Values
        st.subheader("Missing Values")

        with st.expander("Missing Data คืออะไร?"):
            st.markdown(
                "**Missing Data** คือค่าว่างที่ขาดหายไปในชุดข้อมูล "
                "อาจเกิดจากการป้อนข้อมูลไม่สมบูรณ์ อุปกรณ์ทำงานผิดพลาด หรือไฟล์สูญหาย\n\n"
                "**ประเภทของ Missing Data**:\n"
                "- **MCAR** (Missing Completely at Random): หายไปโดยบังเอิญ ไม่เกี่ยวกับตัวแปรใดเลย\n"
                "- **MAR** (Missing at Random): การหายไปเกี่ยวข้องกับตัวแปรอื่น เช่น ผู้หญิงมักไม่เปิดเผยน้ำหนัก\n"
                "- **MNAR** (Missing Not at Random): การหายไปเกี่ยวข้องกับค่าของตัวเอง เช่น คนน้ำหนักมากมักไม่ตอบ"
            )

        missing_cols = {
            col: int(working_df[col].isnull().sum())
            for col in working_df.columns
            if working_df[col].isnull().sum() > 0
        }

        if not missing_cols:
            st.success("ไม่มี Missing values")
        else:
            for col, count in missing_cols.items():
                pct = count / len(working_df) * 100
                is_num = pd.api.types.is_numeric_dtype(working_df[col])

                c1, c2, c3 = st.columns([3, 2, 2])
                with c1:
                    st.write(f"**{col}** — {count:,} ค่า ({pct:.1f}%)")
                with c2:
                    options = (
                        ["mean", "median", "drop rows"]
                        if is_num
                        else ["most frequent", "drop rows"]
                    )
                    strategy = st.selectbox(
                        "Strategy",
                        options,
                        key=f"miss_strategy_{col}",
                        label_visibility="collapsed",
                    )
                with c3:
                    if st.button("Apply", key=f"miss_apply_{col}"):
                        wdf = st.session_state["working_df"]
                        if strategy == "mean":
                            wdf[col] = wdf[col].fillna(wdf[col].mean())
                        elif strategy == "median":
                            wdf[col] = wdf[col].fillna(wdf[col].median())
                        elif strategy == "most frequent":
                            wdf[col] = wdf[col].fillna(wdf[col].mode()[0])
                        elif strategy == "drop rows":
                            wdf = wdf.dropna(subset=[col]).reset_index(drop=True)
                        st.session_state["working_df"] = wdf
                        st.success(f"{col}: applied '{strategy}'")
                        st.rerun()

                # แสดงคำอธิบายวิธีการจัดการ Missing Values ที่เลือก
                st.caption(MISSING_STRATEGY_INFO.get(strategy, ""))

        st.divider()

        # Outliers
        st.subheader("Outliers")

        with st.expander("Outlier และ วิธีตรวจจับ"):
            st.markdown(
                "**Outlier** คือค่าที่ผิดปกติ ห่างจากข้อมูลส่วนใหญ่อย่างมีนัยสำคัญ "
                "อาจเกิดจากข้อผิดพลาดในการบันทึก หรือเป็นค่าจริงที่หายาก\n\n"
                "**วิธีที่ระบบใช้ตรวจจับ** (เลือกอัตโนมัติตาม Skewness หรือความเบี่ยงเบนของข้อมูล):\n\n"
                "| วิธี | เหมาะกับ | เกณฑ์ |\n"
                "|------|---------|-------|\n"
                "| **Z-Score** | ข้อมูลกระจายแบบ Normal (Skewness ใกล้ 0) | ค่าที่ห่างจาก Mean เกิน 3 เท่าของ SD |\n"
                "| **IQR** | ข้อมูลเบ้ (Skewed) | ค่าที่ต่ำกว่า Q1−1.5×IQR หรือสูงกว่า Q3+1.5×IQR |\n"
            )

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
            for col, info in outlier_cols.items():
                count = info["Outliers"]
                method = info["Method"]
                reason = info["Reason"]
                lower = info["Lower"]
                upper = info["Upper"]

                c1, c2, c3 = st.columns([3, 2, 2])
                with c1:
                    st.write(f"**{col}** — {count:,} ค่า")
                    st.caption(f"{reason} → ขอบเขต: [{lower:,.2f}, {upper:,.2f}]")
                with c2:
                    strategy = st.selectbox(
                        "Strategy",
                        ["clip", "drop rows"],
                        key=f"out_strategy_{col}",
                        label_visibility="collapsed",
                    )
                with c3:
                    if st.button("Apply", key=f"out_apply_{col}"):
                        wdf = st.session_state["working_df"]
                        series = wdf[col]

                        if strategy == "clip":
                            wdf[col] = series.clip(lower=lower, upper=upper)
                        elif strategy == "drop rows":
                            mask = (series >= lower) & (series <= upper)
                            wdf = wdf[mask].reset_index(drop=True)

                        st.session_state["working_df"] = wdf
                        st.success(f"{col}: applied '{strategy}'")
                        st.rerun()

                # แสดงคำอธิบายวิธีการจัดการ Outliers ที่เลือก
                st.caption(OUTLIER_STRATEGY_INFO.get(strategy, ""))

        st.divider()

        # Before & After Summary
        st.subheader("Summary")
        rows_before = df.shape[0]
        rows_after = working_df.shape[0]
        missing_before = df.isnull().sum().sum()
        missing_after = working_df.isnull().sum().sum()

        s1, s2, s3 = st.columns(3)
        s1.metric("Rows", f"{rows_after:,}", delta=f"{rows_after - rows_before:,}")
        s2.metric(
            "Missing Values",
            f"{int(missing_after):,}",
            delta=f"{int(missing_after - missing_before):,}",
        )
        s3.metric(
            "Duplicates",
            f"{int(working_df.duplicated().sum()):,}",
            delta=f"{int(working_df.duplicated().sum()) - duplicate_count:,}",
        )

        # Confirm / Reset
        cf1, cf2 = st.columns(2)
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

        # Download button จะแสดงหลัง Confirm แล้วเท่านั้น
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

    # Page Navigation
    st.divider()
    col1, space, col2 = st.columns([0.8, 8, 0.8])
    with col1:
        if st.button("Back", type="secondary", width="stretch"):
            st.session_state.pop("working_df", None)
            st.session_state.pop("working_df_source_shape", None)
            st.session_state.pop("cleaning_confirmed", None)
            st.query_params["step"] = "upload"
            st.rerun()
    with col2:
        confirmed = st.session_state.get("cleaning_confirmed", False)
        if st.button(
            "Next Step",
            type="primary",
            width="stretch",
            disabled=not confirmed,
        ):
            st.query_params["step"] = "eda"
            st.rerun()
        if not confirmed:
            st.caption("กด Confirm & Save ก่อนไปขั้นตอนถัดไป")