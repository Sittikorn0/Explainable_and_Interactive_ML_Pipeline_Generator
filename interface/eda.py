import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
from scipy.stats import skew
from data_prepare.features.data_distribute import data_distribution
from data_prepare.features.data_type_detection import actual_type, ml_category
from data_prepare.features.target_col import describe_target


def _skew_insight(col_skew: float) -> str:
    """สร้างข้อความอธิบาย skewness"""
    abs_skew = abs(col_skew)
    if abs_skew < 0.5:
        shape = "ใกล้ Normal (สมมาตร)"
    elif abs_skew < 1:
        shape = "เบ้เล็กน้อย (Moderately Skewed)"
    else:
        shape = "เบ้มาก (Highly Skewed)"

    direction = ""
    if col_skew > 0.5:
        direction = " → หางยาวไปทางขวา (Right-skewed)"
    elif col_skew < -0.5:
        direction = " → หางยาวไปทางซ้าย (Left-skewed)"

    return f"Skewness = {col_skew:.2f} → {shape}{direction}"

def render_eda():
    from app import page_header

    page_header("Exploratory Data Analysis", "สำรวจและทำความเข้าใจข้อมูลก่อนสร้างโมเดล")

    if st.session_state.get("main_df") is None:
        st.query_params["step"] = "upload"
        st.rerun()
        return

    has_working_df = "working_df" in st.session_state
    df = st.session_state["working_df"] if has_working_df else st.session_state["main_df"]
    file_name = st.session_state.get("last_uploaded_file", "Unknown File")
    is_cleaned = has_working_df and st.session_state.get("cleaning_confirmed")

    st.info(f"**Current Dataset:** {file_name}")
    with st.expander("Cleaned Data" if is_cleaned else "Raw Data"):
        st.dataframe(df, width="stretch")

    # Dataset Overview
    st.subheader("Dataset Overview")
    _dist_key = ("_eda_dist_cache", df.shape, int(pd.util.hash_pandas_object(df).sum()))
    if st.session_state.get("_eda_dist_key") != _dist_key:
        with st.spinner("Calculating Data..."):
            total_outl, outls_details = data_distribution(df)
        st.session_state["_eda_dist_key"] = _dist_key
        st.session_state["_eda_dist_result"] = (total_outl, outls_details)
    else:
        total_outl, outls_details = st.session_state["_eda_dist_result"]

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

    # ดึง target column จากที่เลือกไว้ในหน้า Upload พร้อม fallback ถ้าไม่มีหรือไม่ valid
    target_col = st.session_state.get("target_col", df.columns[-1])
    if target_col not in df.columns:
        target_col = df.columns[-1]
    st.info(f"**Target Column:** {target_col}  \n{describe_target(df, target_col)}")

    tab1, tab2, tab3 = st.tabs(
        ["Profile", "Distributions", "Relationships"], width="stretch"
    )

    # Profile
    with tab1:
        st.subheader("Data Profile")

        outlier_dict = {item["Column"]: item for item in outls_details}
        profile_list = []

        for col in df.columns:
            series = df[col]
            actual = actual_type(series)
            is_target = col == target_col
            profile_list.append({
                "Column": col,
                "Data Types": actual,
                "ML Category": ml_category(actual, is_target),
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
                "Data Types": st.column_config.TextColumn("Data Types", width="small"),
                "ML Category": st.column_config.TextColumn(
                    "ML Category", width="medium"
                ),
                "Missing": st.column_config.NumberColumn("Missing", format="%d"),
                "Unique": st.column_config.ProgressColumn(
                    "Unique", min_value=0, max_value=df.shape[0], format="%d"
                ),
                "Outliers": st.column_config.NumberColumn("Outliers", format="%d"),
            },
        )

        # คำอธิบาย Attribute Types
        with st.expander("Attribute Types & ML Category คืออะไร?"):
            st.markdown(
                "**ประเภทข้อมูลตาม Machine Learning** (อ้างอิง Topic 2 - Getting to Know Your Data):\n\n"
                "| ประเภท | ความหมาย | ตัวอย่าง | ค่ากลางที่ใช้ได้ |\n"
                "|--------|---------|---------|----------------|\n"
                "| **Categorical/Nominal** | ข้อมูลเชิงหมวดหมู่ ไม่มีลำดับ | สีผม, เพศ, จังหวัด | Mode เท่านั้น |\n"
                "| **Numeric/Discrete** | ตัวเลขจำนวนเต็ม นับได้ | จำนวนสินค้า, อายุ | Mean, Median, Mode |\n"
                "| **Numeric/Continuous** | ตัวเลขทศนิยม วัดได้ | น้ำหนัก, อุณหภูมิ | Mean, Median |\n"
                "| **Datetime** | วันที่/เวลา | 2024-01-01 | — |\n\n"
                "**Target** คือคอลัมน์ที่ต้องการทำนาย ซึ่งเลือกไว้ในขั้นตอน Upload"
            )

    # Distributions
    with tab2:
        st.subheader("Data Distributions")

        num_cols = df.select_dtypes(include=["number"]).columns.tolist()
        cat_cols = df.select_dtypes(include=["object", "category"]).columns.tolist()

        selected_col = st.selectbox("Select column to visualize", num_cols + cat_cols)

        if selected_col in num_cols:
            fig = px.histogram(
                df,
                x=selected_col,
                marginal="box",
                nbins=30,
                color_discrete_sequence=["#0082CE"],
            )
            fig.update_layout(template="plotly_dark", height=450, showlegend=False)
            st.plotly_chart(fig, width="stretch")

            # Insight สำหรับ Numeric
            col_data = df[selected_col].dropna()
            col_skew = float(skew(col_data))
            insight = _skew_insight(col_skew)

            # คำแนะนำ Transformation ถ้าเบ้มาก
            if abs(col_skew) >= 1:
                insight += (
                    "\n\n**คำแนะนำ:** ข้อมูลเบ้มาก อาจต้อง Transform ก่อนสร้างโมเดล "
                    "เช่น Log, Box-Cox, หรือ Yeo-Johnson Transformation"
                )

            st.info(f"**{selected_col}:** {insight}")

        else:
            counts = df[selected_col].value_counts().reset_index()
            counts.columns = [selected_col, "count"]
            fig = px.bar(
                counts.head(20),
                x=selected_col,
                y="count",
                color_discrete_sequence=["#0082CE"],
            )
            fig.update_layout(template="plotly_dark", height=450, showlegend=False)
            st.plotly_chart(fig, width="stretch")

            # Insight สำหรับ Categorical
            n_unique = df[selected_col].nunique()
            top_value = counts.iloc[0][selected_col] if len(counts) > 0 else "N/A"
            top_count = int(counts.iloc[0]["count"]) if len(counts) > 0 else 0
            top_pct = (top_count / len(df) * 100) if len(df) > 0 else 0

            insight = f"มี **{n_unique}** ค่า Unique — ค่าที่พบมากสุดคือ **{top_value}** ({top_pct:.1f}%)"

            # ตรวจ Class Imbalance (ถ้าเป็น Target)
            if selected_col == target_col and n_unique <= 20:
                min_count = int(counts["count"].min())
                max_count = int(counts["count"].max())
                if max_count > 3 * min_count:
                    insight += (
                        "\n\n**Class Imbalance:** ค่าที่พบมากสุดมากกว่าค่าที่พบน้อยสุด "
                        f"ถึง {max_count / min_count:.1f} เท่า — อาจต้องจัดการก่อนสร้างโมเดล"
                    )

            st.info(f"**{selected_col}:** {insight}")

    # Relationships
    with tab3:
        st.subheader("Relationships & Redundancy")

        p_col1, p_col2 = st.columns(2)

        with p_col1:
            st.write("**Missing Value Scan**")
            missing_data = df.isnull().astype(int)
            fig_miss = px.imshow(
                missing_data,
                labels=dict(x="Columns", y="Rows", color="Is Missing?"),
                color_continuous_scale="Viridis",
            )
            fig_miss.update_layout(template="plotly_dark", height=400)
            st.plotly_chart(fig_miss, width="stretch")

        with p_col2:
            st.write("**Correlation Heatmap**")
            numeric_df = df.select_dtypes(include=[np.number])
            if numeric_df.shape[1] > 1:
                corr = numeric_df.corr()
                fig_corr = px.imshow(
                    corr,
                    text_auto=".2f",
                    aspect="auto",
                    color_continuous_scale="RdBu_r",
                    range_color=[-1, 1],
                )
                fig_corr.update_layout(template="plotly_dark", height=400)
                st.plotly_chart(fig_corr, width="stretch")

                # Detect High Correlation
                high_corr_pairs = []
                for i in range(len(corr.columns)):
                    for j in range(i + 1, len(corr.columns)):
                        r = corr.iloc[i, j]
                        if abs(r) > 0.8:
                            high_corr_pairs.append(
                                (corr.columns[i], corr.columns[j], round(r, 2))
                            )

                if high_corr_pairs:
                    pairs_text = ", ".join(
                        [f"**{a}** ↔ **{b}** (r={r})" for a, b, r in high_corr_pairs]
                    )
                    st.warning(
                        f"**Multicollinearity ที่อาจเป็นปัญหา:** {pairs_text}\n\n"
                        "คอลัมน์ที่มี Correlation สูง (|r| > 0.8) อาจให้ข้อมูลซ้ำซ้อนกัน "
                        "ซึ่งอาจทำให้โมเดลบางชนิด (เช่น Linear Regression) ทำงานได้ไม่ดี "
                        "อาจพิจารณาลบคอลัมน์ใดคอลัมน์หนึ่งออก"
                    )
            else:
                st.info("Need more numeric columns for correlation.")

    # Page Navigation
    col1, _, col2 = st.columns([0.8, 8, 0.8])
    with col1:
        if st.button("Back", type="secondary", width="stretch", key="back"):
            st.query_params["step"] = "cleaning"
            st.rerun()
    with col2:
        if st.button("Next Step", type="primary", width="stretch"):
            st.query_params["step"] = "model"
            st.rerun()