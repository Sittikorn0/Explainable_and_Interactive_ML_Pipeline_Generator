import streamlit as st
import pandas as pd
import numpy as np
import os
from features.data_distribute import data_distribution
from features.loading_data import save_cleaned_data


def render_cleaning():
    from app import page_header

    page_header(
        "Data Cleaning",
        "กระบวนการลบหรือแก้ไขข้อมูลที่ผิดพลาด ไม่สมบูรณ์ หรือไม่มีความสอดคล้องกันจากชุดข้อมูล",
    )

    if st.session_state.get("main_df") is not None:
        df        = st.session_state["main_df"]
        file_name = st.session_state.get("last_uploaded_file", "Unknown File")

        st.info(f"**Current Dataset:** {file_name}")
        with st.expander("Raw Data"):
            st.dataframe(df, width="stretch")

        # ── Dataset Overview ──────────────────────────────────────
        st.subheader("Dataset Overview")
        with st.spinner("Calculating Data..."):
            total_outl, outls_details = data_distribution(df)

        total_cells     = df.size
        total_missing   = df.isnull().sum().sum()
        missing_pct     = (total_missing / total_cells * 100) if total_cells > 0 else 0
        duplicate_count = int(df.duplicated().sum())
        dup_pct         = (duplicate_count / df.shape[0] * 100) if df.shape[0] > 0 else 0
        outlier_pct     = (total_outl / total_cells * 100) if total_cells > 0 else 0

        m1, m2, m3, m4, m5 = st.columns(5)
        m1.metric("Rows",           f"{df.shape[0]:,}")
        m2.metric("Columns",        df.shape[1])
        m3.metric("Missing Values", f"{total_missing:,} ({missing_pct:.1f}%)")
        m4.metric("Duplicate Rows", f"{duplicate_count:,} ({dup_pct:.1f}%)")
        m5.metric("Outliers",       f"{total_outl:,} ({outlier_pct:.1f}%)")

        tab1, tab2 = st.tabs(["Profile", "Cleaning"], width="stretch")

        # ── Tab 1: Profile ────────────────────────────────────────
        with tab1:
            st.subheader("Data Profile")

            outlier_dict = {item["Column"]: item for item in outls_details}
            profile_list = []

            for col in df.columns:
                series = df[col]
                profile_list.append({
                    "Column"  : col,
                    "Missing" : int(series.isnull().sum()),
                    "Outliers": outlier_dict.get(col, {}).get("Outliers", 0),
                    "Unique"  : int(series.nunique())
                })

            st.dataframe(
                pd.DataFrame(profile_list),
                width="stretch",
                hide_index=True,
                column_config={
                    "Column"  : st.column_config.TextColumn("Column"),
                    "Missing" : st.column_config.NumberColumn("Missing", format="%d"),
                    "Unique"  : st.column_config.ProgressColumn(
                                    "Unique", min_value=0,
                                    max_value=df.shape[0], format="%d"),
                    "Outliers": st.column_config.NumberColumn("Outliers", format="%d"),
                },
            )

        # ── Tab 2: Cleaning ───────────────────────────────────────
        with tab2:
            # working_df = สำเนาที่ใช้แก้ไข ยังไม่ overwrite main_df
            # reset เมื่อ dataset เปลี่ยน (upload ไฟล์ใหม่) โดยเช็คจาก shape
            # ไม่ reset เมื่อกด Back แล้วกลับมา เพราะ key ยังอยู่ใน session_state
            current_shape = df.shape
            if (
                "working_df" not in st.session_state
                or st.session_state.get("working_df_source_shape") != current_shape
            ):
                st.session_state["working_df"]              = df.copy()
                st.session_state["working_df_source_shape"] = current_shape

            working_df = st.session_state["working_df"]

            # ── Section 1: Duplicates ─────────────────────────────
            st.subheader("Duplicates")
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

            # ── Section 2: Missing Values ─────────────────────────
            st.subheader("Missing Values")

            missing_cols = {
                col: int(working_df[col].isnull().sum())
                for col in working_df.columns
                if working_df[col].isnull().sum() > 0
            }

            if not missing_cols:
                st.success("ไม่มี Missing values")
            else:
                for col, count in missing_cols.items():
                    pct    = count / len(working_df) * 100
                    is_num = pd.api.types.is_numeric_dtype(working_df[col])

                    c1, c2, c3 = st.columns([3, 2, 2])
                    with c1:
                        st.write(f"**{col}** — {count:,} ค่า ({pct:.1f}%)")
                    with c2:
                        options  = ["mean", "median", "drop rows"] if is_num \
                                   else ["most frequent", "drop rows"]
                        strategy = st.selectbox(
                            "Strategy", options,
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

            st.divider()

            # ── Section 3: Outliers ───────────────────────────────
            st.subheader("Outliers")

            outlier_cols = {
                item["Column"]: item
                for item in outls_details
                if item["Outliers"] > 0
                and pd.api.types.is_numeric_dtype(working_df.get(item["Column"], pd.Series()))
            }

            if not outlier_cols:
                st.success("ไม่พบ Outliers")
            else:
                for col, info in outlier_cols.items():
                    count  = info["Outliers"]
                    method = info["Method"]   # "IQR" หรือ "Z-Score"

                    c1, c2, c3 = st.columns([3, 2, 2])
                    with c1:
                        st.write(f"**{col}** — {count:,} ค่า (detected by {method})")
                    with c2:
                        strategy = st.selectbox(
                            "Strategy", ["clip", "drop rows"],
                            key=f"out_strategy_{col}",
                            label_visibility="collapsed",
                        )
                    with c3:
                        if st.button("Apply", key=f"out_apply_{col}"):
                            wdf    = st.session_state["working_df"]
                            series = wdf[col]

                            # คำนวณ boundary ตาม method ที่ detect ไว้ใน data_distribution()
                            if method == "IQR":
                                q1, q3  = series.quantile(0.25), series.quantile(0.75)
                                iqr     = q3 - q1
                                lower, upper = q1 - 1.5 * iqr, q3 + 1.5 * iqr
                            else:  # Z-Score
                                lower = series.mean() - 3 * series.std()
                                upper = series.mean() + 3 * series.std()

                            if strategy == "clip":
                                wdf[col] = series.clip(lower=lower, upper=upper)
                            elif strategy == "drop rows":
                                mask = (series >= lower) & (series <= upper)
                                wdf  = wdf[mask].reset_index(drop=True)

                            st.session_state["working_df"] = wdf
                            st.success(f"{col}: applied '{strategy}'")
                            st.rerun()

            st.divider()

            # ── Confirm / Reset ───────────────────────────────────
            rows_before = df.shape[0]
            rows_after  = working_df.shape[0]
            st.write(
                f"Rows: **{rows_before:,}** → **{rows_after:,}** "
                f"({rows_before - rows_after:,} ถูกลบ)"
            )

            cf1, cf2 = st.columns(2)
            with cf1:
                if st.button("Confirm & Save", type="primary", width="stretch"):
                    original_filename = st.session_state.get("last_uploaded_file", "dataset.csv")
                    cleaned_filename  = save_cleaned_data(
                        st.session_state["working_df"].copy(),
                        original_filename,
                    )
                    st.session_state["cleaning_confirmed"] = True
                    st.success(f"บันทึกเป็น **{cleaned_filename}** แล้ว")
                    st.rerun()
            with cf2:
                if st.button("Reset", type="secondary", width="stretch"):
                    st.session_state["working_df"]        = df.copy()
                    st.session_state["cleaning_confirmed"] = False
                    st.info("Reset กลับ original data แล้ว")
                    st.rerun()

            # ── Download button (แสดงหลัง Confirm แล้วเท่านั้น) ──
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
            if st.button("Next Step", type="primary", width="stretch"):
                # ถ้ายังไม่ได้ Confirm → auto-save ด้วย working_df ปัจจุบัน
                if not confirmed:
                    original_filename = st.session_state.get("last_uploaded_file", "dataset.csv")
                    save_cleaned_data(
                        st.session_state["working_df"].copy(),
                        original_filename,
                    )
                    st.session_state["cleaning_confirmed"] = True
                st.query_params["step"] = "eda"
                st.rerun()

    else:
        st.query_params["step"] = "upload"
        st.rerun()